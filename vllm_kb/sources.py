"""数据源抽象层：所有来源（github / markdown / pdf / image / excel / ...）实现统一接口。

接口约定（新来源只需实现这两个方法 + 注册 type）：
    pull() -> int                     把原始数据拉取/导入到 raw_dir（幂等、可续传），返回新增条数
    canonicalize() -> list[KbDocument] 从原始数据再生 canonical 文档（确定性、可重放）

布局约定：
    - 原始数据按来源分目录存储：data/raw/{source_id}/...（不同来源互不干扰）；
    - 二进制/文本资产（PDF/Markdown/图片原件）统一存 data/assets/{sub}/（不可变，sha256 记录）；
    - 解析产物（Markdown 正文、结构化表格 JSON、OCR 结果）存 data/parsed/{sub}/（可重跑）；
    - canonical 统一单文件（storage.canonical_file），不按来源拆分；
    - doc 的唯一标识（source_id）由来源自己保证跨来源不冲突（github 源用 repo 命名空间）。

新增来源类型：继承 BaseSource + 注册到 _REGISTRY，即可接入全链路
（canonical 合并 -> 分块 -> 嵌入 -> 图/向量 -> 置信度 -> 检索），无需改动其他模块。
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .config import PROJECT_ROOT, SourceCfg
from .models import KbDocument
from .tagging import (
    TagEntry,
    TagRegistry,
    extract_tags,
    headings_from_markdown,
    headings_from_pdf,
)

if TYPE_CHECKING:  # 仅类型标注用，避免循环导入
    from .config import AppConfig

# Markdown 图片引用解析（形态支持 + 代码感知 + 兜底占位）见 md_images.py；
# 这里只保留资产落盘需要的常量
_BASE64_IMG_RE = re.compile(r"data:image/(png|jpe?g|webp|gif);base64,([A-Za-z0-9+/=]+)", re.I)
_IMG_EXT = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "webp": "webp", "gif": "gif"}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_tag(rel: str) -> str:
    """相对路径指纹（8 hex），用于同名 md 消歧。

    用**路径**而非内容的 sha：md 每次编辑内容都变，内容寻址会让 source_id 漂移
    （旧文档变孤儿、审核状态/遥测丢失、检索重复）；路径指纹在内容改动下稳定，
    且**不可读**——source_id 会出现在 /search 的 doc_id、/doc/{id} 与遥测库里，
    不能暴露目录名（可能含客户/项目名）。
    """
    return hashlib.sha256(rel.replace("\\", "/").encode("utf-8")).hexdigest()[:8]


def _asset_entry(rel: str, sha: str, stype: str, path: Path) -> tuple:
    """asset_registry 注册条目 `(rel_path, sha256, source_type, size)`；size 取不到则 0。"""
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return (rel, sha, stype, size)


# 资产层"同名异内容"版本后缀（_copy_asset 写的 `stem.<sha12>.suffix`）
_VER_SUFFIX_RE = re.compile(r"^(?P<base>.+)\.(?P<sha>[0-9a-f]{12})$")


def _latest_versions(paths: list[Path]) -> list[tuple[Path, str]]:
    """资产层扁平副本按**版本族**收敛，返回 `[(最新版本的路径, 族名 stem)]`（按路径排序）。

    `case.md` 与 `case.<sha12>.md` 是同一篇的不同版本，每族只保留最新（mtime 最大；
    并列时取文件名字典序较大者，保证确定性）的一个。

    `_copy_asset` 在"同名异内容"时写成 `stem.<sha12>.suffix`，所以资产层会累积历史版本
    （原始 `case.md` 永久保留 + 每个不同内容一份）。回退模式若把它们都当文档，会把旧版本
    **复活成独立文档**（源文件删掉后尤其明显，一篇变多篇）。

    返回的**族名**（而非带 `.<sha12>` 的文件名）才是文档身份：否则收敛到 sha 副本时
    `source_id` 会从 `md:case` 漂移成 `md:case.<sha12>`，等于换了篇文档（审核状态/标签丢失）。

    后缀无关（按 `(族名, 后缀)` 分组），md/word 等扁平副本层共用。
    """
    fams: dict[tuple[str, str], tuple[float, str, Path, str]] = {}
    for p in paths:
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        m = _VER_SUFFIX_RE.match(p.stem)
        base = m.group("base") if m else p.stem
        key = (base, p.suffix.lower())
        cand = (mtime, p.name, p, base)
        cur = fams.get(key)
        if cur is None or (cand[0], cand[1]) > (cur[0], cur[1]):
            fams[key] = cand
    return sorted(((v[2], v[3]) for v in fams.values()), key=lambda t: t[0])


def _discover_source_files(src: "BaseSource", patterns: tuple[str, ...],
                           import_dir: Path, assets_dir: Path,
                           ) -> tuple[list[tuple[Path, bool, str]], bool]:
    """业务文本来源的文件发现（md / word 共用），返回 `([(路径, 是否来自导入目录, 逻辑 stem)], 是否回退)`。

    - **优先导入目录**：`rglob` 保留操作者的目录树 → 同名不同目录的文件能按相对路径指纹
      消歧，且编辑源文件不会因资产层累积副本而变成多篇；
    - 导入目录不存在/扫不到文件 → **回退资产层扁平副本**（`pull()` 的产物），并按版本族
      收敛（见 `_latest_versions`），逻辑 stem 取族名以防 `source_id` 漂移。

    回退的代价由调用方负责提示（图片相对路径失锚等），本函数只做发现与收敛。
    """
    files: list[tuple[Path, bool, str]] = []
    if import_dir.exists():
        for pat in patterns:
            files.extend((p, True, p.stem) for p in sorted(import_dir.rglob(pat)))
    if files:
        return files, False
    if not assets_dir.exists():
        return [], False
    cand: list[Path] = []
    for pat in patterns:
        cand.extend(sorted(assets_dir.glob(pat)))
    return [(p, False, base) for p, base in _latest_versions(cand)], True


def _copy_asset(src: Path, assets_dir: Path, sub: str) -> tuple[str, str, bool]:
    """复制资产到 assets/{sub}/（不可变层）。同名同 sha 幂等跳过；同名异 sha 加 sha 前缀。
    返回 (assets 相对路径, sha256, 是否新增复制)。"""
    target_dir = assets_dir / sub
    target_dir.mkdir(parents=True, exist_ok=True)
    sha = _sha256(src)
    target = target_dir / src.name
    if target.exists() and _sha256(target) == sha:
        return f"assets/{sub}/{target.name}", sha, False
    if target.exists():
        # 同名但内容不同：加 sha 前缀避免覆盖
        target = target_dir / f"{src.stem}.{sha[:12]}{src.suffix}"
    shutil.copy2(src, target)
    return f"assets/{sub}/{target.name}", sha, True


class BaseSource(ABC):
    type: str = "base"

    def __init__(self, cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                 app_cfg: Optional["AppConfig"] = None):
        self.cfg = cfg
        self.id = cfg.id
        self.project_root = project_root
        self.app_cfg = app_cfg  # 提供后路径经 AppConfig.resolve（支持 VLLM_KB_DATA_ROOT 重定向）

    def resolve(self, p: str | Path) -> Path:
        """路径解析：优先 AppConfig.resolve（VLLM_KB_DATA_ROOT 重定向 + data_root），
        否则按 project_root 相对解析。绝对路径原样返回。"""
        path = Path(p)
        if path.is_absolute():
            return path
        if self.app_cfg is not None:
            return self.app_cfg.resolve(str(path))
        return self.project_root / path

    @property
    def raw_dir(self) -> Path:
        """该来源原始数据的独立目录（默认 data/raw/{source_id}）。"""
        return self.resolve(self.cfg.get("raw_dir", f"data/raw/{self.id}"))

    def _ocr_settings(self):
        """OCR 配置（取自 image source 的 ocr_* 字段，与审核工作台/请求期端点同源）。

        无 app_cfg（纯解析测试）或无 image source → None。
        """
        if self.app_cfg is None:
            return None
        from .ocr import ocr_config_from_cfg

        return ocr_config_from_cfg(self.app_cfg)

    def _ocr_artifact_for(self, asset_path: Path, sha: str,
                          image_ref: str) -> tuple[Optional[dict], str]:
        """对已资产化的图片做 OCR（ocr.json 为幂等缓存），返回 (evidence.ocr 摘要, 正文注入后缀)。

        - **高置信**（无自报异常且 ≥ `ocr_min_confidence`）→ 返回注入后缀，调用方拼到占位符后
          （文本随正文进 FTS + 向量）；
        - 低置信 / 自报异常 → 返回摘要但**注入后缀为空**（只留签名线索 + 审核队列）；
        - OCR 不可用 / 未配置 / `ask` 且无 API → (None, "")，导入不受阻。

        provider=ask 且无 ocr_api_base 时**不在此交互询问**（询问由 ImageSource 统一做一次），
        此时本次构建不注入正文——本地 OCR 产出缓存后，下次构建即生效。
        """
        from .ocr import (OcrApiError, OcrUnavailable, build_ocr_artifact,
                          engine_fingerprint, load_ocr_artifact, ocr_image_detail,
                          save_ocr_artifact)

        oc = self._ocr_settings()
        if oc is None or oc.provider == "none":
            return None, ""
        provider = oc.provider
        if provider == "ask":
            if not oc.api_base:
                return None, ""
            provider = "api"
        cache = self.resolve("data/parsed/images") / f"{asset_path.stem}.ocr.json"
        fp = engine_fingerprint(provider, oc.mode, oc.model)
        art = load_ocr_artifact(cache, sha, fp)
        if art is None:
            try:
                res = ocr_image_detail(asset_path, provider, api_base=oc.api_base,
                                       api_key=oc.api_key, model=oc.model, mode=oc.mode)
            except (OcrApiError, OcrUnavailable) as e:
                print(f"[sources:{self.id}] 图片 OCR 失败（{asset_path.name}）：{e}", flush=True)
                return None, ""
            art = build_ocr_artifact(asset_path, sha, res, provider, oc.mode, oc.model,
                                     oc.min_confidence)
            save_ocr_artifact(cache, art, image_ref=image_ref)
        else:
            # 阈值即时生效（不进引擎指纹：调阈值只重判定，不重跑 OCR）
            art.min_confidence = oc.min_confidence
        if art.high_confidence and art.text.strip():
            return art.evidence(), (f"\n图片文字（OCR 置信度 {art.confidence:.2f}）:\n"
                                    f"{art.text.strip()}")
        return art.evidence(), ""

    def _register_asset_mappings(self, items: list[tuple]) -> None:
        """注册资产到审核侧 asset_registry（管理员路径映射；不进 canonical/检索库）。

        items: [(assets相对路径, sha256, source_type[, size])]。app_cfg 缺失（纯解析测试）时跳过。
        幂等 upsert（批量单连接）；失败仅提示，不影响入库。
        """
        if self.app_cfg is None or not items:
            return
        try:
            from .review import register_assets

            db = self.app_cfg.resolve(self.app_cfg.storage.review_path)
            register_assets(db, items)
        except Exception as e:
            print(f"[sources:{self.id}] asset_registry 注册失败（不影响入库）: {e}")

    # ---------------- 内部数据脱敏（config.sanitize 控制启用范围） ----------------

    # 默认启用脱敏的业务来源（github 公开数据不脱敏；PDF 手册默认不启用，config 加 "pdf" 即开）
    DEFAULT_SANITIZE_SOURCES = ("excel", "markdown", "word")

    def sanitize_enabled(self) -> bool:
        """该来源是否启用脱敏（config.sanitize.sources 控制；None=默认业务来源，[]=全关）。"""
        if self.app_cfg is None:
            return self.type in self.DEFAULT_SANITIZE_SOURCES
        cfg_sources = self.app_cfg.sanitize.sources
        if cfg_sources is None:
            return self.type in self.DEFAULT_SANITIZE_SOURCES
        return self.type in cfg_sources

    def sanitize_params(self) -> tuple[bool, Optional[list], Optional[list]]:
        """返回 (enabled, keep_paths, keep_ips)。

        - enabled=False：该来源未启用脱敏，调用方应**跳过** sanitize（原文入库）；
        - enabled=True：keep_paths/keep_ips 为 None=默认白名单、[]=全脱敏
          （app_cfg 缺失时用默认白名单脱敏，但不写日志）。
        """
        if not self.sanitize_enabled():
            return False, None, None
        if self.app_cfg is None:
            return True, None, None
        return True, self.app_cfg.sanitize.keep_paths, self.app_cfg.sanitize.keep_ips

    def save_sanitize_log(self, collector: Optional[dict]) -> None:
        """把被脱敏命中的 IP/路径落盘维护文件（data/sanitize_log.json，幂等合并，不进库）。"""
        if not collector or self.app_cfg is None:
            return
        try:
            from .sanitize import save_sanitize_log

            save_sanitize_log(self.app_cfg, collector)
        except Exception as e:
            print(f"[sources:{self.id}] sanitize_log 写入失败（不影响入库）: {e}")

    @abstractmethod
    def pull(self) -> int:
        """把原始数据拉取到 raw_dir（幂等、断点续传），返回本次新增条数。"""

    @abstractmethod
    def canonicalize(self) -> list[KbDocument]:
        """从 raw_dir 再生 canonical 文档（确定性、可重放）。"""


class GithubSource(BaseSource):
    """GitHub REST 来源：issue + PR + 评论（见 github_pull.GithubPuller）。"""

    type = "github"

    def __init__(self, cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                 app_cfg: Optional["AppConfig"] = None):
        super().__init__(cfg, project_root, app_cfg=app_cfg)
        from .github_pull import GithubPuller, recanonicalize

        self.puller = GithubPuller(cfg, project_root)
        self._recanonicalize = recanonicalize

    def pull(self, max_issues: int | None = None, incremental: bool = False,
             missing: bool = False, numbers: list[int] | None = None,
             force_numbers: bool = False) -> int:
        """拉取 GitHub 数据（incremental=True 时 done 后仍时间窗增量；missing=True 补差
        只拉缺失；numbers 走 REST 单条补拉，force_numbers=True 强制重拉已有编号，
        见 GithubPuller.pull）。"""
        if max_issues is not None:
            self.puller.max_issues = max_issues
        return self.puller.pull(incremental=incremental, missing=missing, numbers=numbers,
                                force_numbers=force_numbers)

    def canonicalize(self) -> list[KbDocument]:
        return self._recanonicalize(self.cfg, self.project_root)


class MarkdownSource(BaseSource):
    """Markdown 文档来源（案例 / 架构说明 / 经验总结，内容不固定）。

    配置示例：
        {"id": "wiki", "type": "markdown", "path": "data/imports/md",
         "title_pattern": "^#\\s+(.+)"}

    - pull(): 扫描配置 path 下 *.md / *.markdown，复制到 data/assets/md/（不可变层）；
    - canonicalize(): 每个文件一个 KbDocument（title=首个 # 标题或文件名，body=全文）；
    - verification=unverified（质量参差，后续经审核工作台补标为 tested/expert）。
    """

    type = "markdown"

    def __init__(self, cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                 app_cfg: Optional["AppConfig"] = None):
        super().__init__(cfg, project_root, app_cfg=app_cfg)
        self.import_dir = self.resolve(self.cfg.get("path", f"data/imports/{self.id}"))
        self.title_pattern = str(self.cfg.get("title_pattern", r"^#\s+(.+)"))
        self._title_re = re.compile(self.title_pattern)

    def _assets_dir(self) -> Path:
        return self.resolve("data/assets/md")

    def pull(self, max_issues: Optional[int] = None) -> int:
        """扫描导入目录，把 md 复制到资产层（幂等）。返回新增条数。"""
        if not self.import_dir.exists():
            print(f"[sources:{self.id}] 导入目录不存在: {self.import_dir}")
            return 0
        added = 0
        registered: list[tuple[str, str, str]] = []
        for p in sorted(self.import_dir.rglob("*.md")) + sorted(self.import_dir.rglob("*.markdown")):
            rel, sha, copied = _copy_asset(p, self.resolve("data/assets"), "md")
            registered.append(_asset_entry(rel, sha, "doc_markdown", self.resolve(rel)))
            if copied:
                added += 1
        self._register_asset_mappings(registered)
        print(f"[sources:{self.id}] 资产层扫描完成（新增 {added} 个 md）")
        return added

    def canonicalize(self) -> list[KbDocument]:
        """从原始 md 再生 KbDocument，并收集正文图片到资产层。

        - **优先读 imports 源文件**：图片相对路径以 md 所在目录为基准解析
          （md 复制到 assets 后相对路径会失锚）；imports 被清空时回退 assets 副本；
        - 正文图片引用改为**不透明占位**（`[图片]`），原引用只进 evidence（含资产 asset_id），
          **正文与 canonical 不暴露任何服务器路径**；
        - **后置脱敏**：body/title 以**原文入库**（原文检索）；仅按 config.sanitize 扫描
          会被脱敏的 IP/路径落盘 sanitize_log.json（出口脱敏由 serve_api 返回时统一做）；
        - evidence 记录图片清单（供 ImageSource OCR 与图文互证消费）。
        """
        from .sanitize import collect_sanitize_hits

        docs: list[KbDocument] = []
        registry = TagRegistry.load(self.app_cfg) if self.app_cfg else TagRegistry()
        sanitize_on, keep_paths, keep_ips = self.sanitize_params()
        collector: dict = {}  # 会被脱敏的原始 IP/路径（落盘维护，不进库）
        img_assets: list[tuple] = []  # 图片资产（末尾统一注册到 asset_registry）
        md_files, fallback = _discover_source_files(
            self, ("*.md", "*.markdown"), self.import_dir, self._assets_dir())
        if fallback:
            print(f"[sources:{self.id}] ⚠ 导入目录无 md（{self.import_dir}），已回退到资产层副本 "
                  f"{self._assets_dir()}：图片相对路径**已失锚**（assets/md 是扁平副本），"
                  f"只能按文件名到 assets/images 尽力反查，未命中的一律占位"
                  f"（正文仍不含路径）；如需完整图片/OCR 请恢复 imports 目录", flush=True)
        # 同名 stem 冲突检测：md:<stem> 会互相覆盖（ingest 用 INSERT OR REPLACE，后者胜）
        stem_counts: dict[str, int] = {}
        for _p, _fi, stem in md_files:
            stem_counts[stem] = stem_counts.get(stem, 0) + 1
        dup_stems = {s for s, n in stem_counts.items() if n > 1}
        if dup_stems:
            shown = ", ".join(sorted(dup_stems)[:5]) + (" …" if len(dup_stems) > 5 else "")
            print(f"[sources:{self.id}] ⚠ 检测到 {len(dup_stems)} 组同名 md（{shown}）："
                  f"已用相对路径指纹消歧为 md:<stem>--<sha8>，避免互相覆盖", flush=True)
        total = len(md_files)
        start_ts = time.time()
        if total:
            print(f"[sources:{self.id}] 解析 {total} 个 Markdown …", flush=True)
        for p, from_imports, stem in md_files:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                print(f"[sources:{self.id}] 跳过 {p.name}: {e}")
                continue
            body, evidence, registered = self._resolve_images(
                p, text, assets_only=not from_imports)
            img_assets.extend(registered)
            if sanitize_on:
                ips, paths = collect_sanitize_hits(body, keep_paths, keep_ips)
                if ips:
                    collector.setdefault("ips", set()).update(ips)
                if paths:
                    collector.setdefault("paths", set()).update(paths)
            title_raw = self._title_re.search(text)
            title = title_raw.group(1).strip() if title_raw else stem
            sha = _sha256(p)
            asset_id = sha[:16]
            # 同名 stem：加相对路径指纹（默认不动 → 存量 source_id 不漂移）
            if stem in dup_stems:
                base_dir = self.import_dir if from_imports else self._assets_dir()
                try:
                    rel_key = p.relative_to(base_dir).as_posix()
                except ValueError:
                    rel_key = p.name
                sid = f"md:{stem}--{_path_tag(rel_key)}"
            else:
                sid = f"md:{stem}"
            # 文档级自动标签：文件名 + Markdown 标题（两级分类，见 tagging.py）
            tags, cands = extract_tags(stem, headings_from_markdown(text), registry=registry)
            extra: dict[str, Any] = {
                "asset": {"asset_id": asset_id, "sha256": sha, "format": "markdown"},
                "quality": {
                    "text_source": "text_layer",
                    "parsed_with": "raw",
                    # 回退模式（imports 缺失）→ 图片只能按文件名尽力反查
                    "source_mode": "imports" if from_imports else "assets_fallback",
                    "images_unresolved": sum(1 for e in evidence
                                             if e.get("kind") == "unresolved"),
                },
                "verification": "unverified",  # 质量参差：先入库，审核工作台补标
                "structure": {},
                # 未收录强候选（进审核队列 tag_candidate 人工采纳后入词典）
                "tag_candidates": [{"name": c.name, "tier": c.tier} for c in cands],
            }
            if evidence:
                extra["evidence"] = evidence
            docs.append(KbDocument(
                source_type="doc_markdown",
                source_id=sid,
                url="",
                title=title,
                body=body,
                created_at=None,
                component="",
                tags=[t.name for t in tags],
                extra=extra,
            ))
        if total:
            print(f"[sources:{self.id}] 解析完成：{len(docs)}/{total} 篇（耗时 "
                  f"{time.time() - start_ts:.0f}s）", flush=True)
        # 图片资产注册（审核侧经 asset_id 反查路径/预览）；与 md 文件同批注册
        self._register_asset_mappings(img_assets)
        self.save_sanitize_log(collector)
        return docs

    # ---------- Markdown 图片收集（确保图片与 md 一起入库） ----------

    def _resolve_images(self, md_path: Path, text: str, *,
                        assets_only: bool = False) -> tuple[str, list[dict], list[tuple]]:
        """扫描正文图片引用：本地/base64 资产化（**不透明占位**替换引用，原引用只进 evidence）；
        URL 引用标记 remote（V1 不下载）；解析失败标记 unresolved。
        返回 (占位化后的 body, evidence 列表, 新增资产 [(rel, sha, "image", size)])。

        形态解析与代码感知在 `md_images.py`：行内（含空格/尖括号/括号/跨行）、引用式
        （含定义行去路径）、HTML `<img>` 均支持；**任何未识别或未闭合的图片语法也一律占位**，
        保证"正文与 canonical 不含服务器路径"对任意输入都成立。

        assets_only=True（imports 缺失的回退模式）：相对路径基准 assets/md 是**扁平副本**，
        原相对结构已丢失——先按 md 同目录找，未命中再按**文件名**到 assets/images 尽力反查；
        仍未命中的照常占位（不记 source_ref），所以回退模式同样不泄漏路径。

        安全约束：正文与 canonical **不含任何服务器路径**——evidence 只记 asset_id/sha256
        （管理员侧经 asset_registry 映射回文件），unresolved 不保留原文引用（可能是路径形态）。

        **图片 OCR 文本注入**：本地/base64 图片资产化后立即走 OCR（ocr.json 幂等缓存）；
        高置信文本追加在占位符之后随正文进 FTS + 向量，低置信/自报异常不注入（只留签名线索）。
        """
        from .md_images import rewrite_images

        evidence: list[dict] = []
        registered: list[tuple] = []
        images_dir = self.resolve("data/assets/images")

        def _reg(rel: str, sha: str, path: Path) -> None:
            """登记图片资产（审核侧反查路径/预览；不进 canonical）。"""
            registered.append(_asset_entry(rel, sha, "image", path))

        def resolve(ref) -> str:
            alt = ref.alt or ""
            placeholder = f"[图片:{alt}]" if alt.strip() else "[图片]"
            dest = (ref.dest or "").strip()
            ev: dict = {"kind": "unresolved", "ocr": None}
            if not dest:
                evidence.append(ev)
                return placeholder
            if dest.startswith(("http://", "https://")):
                ev = {"kind": "remote", "source_ref": dest, "ocr": None}
                evidence.append(ev)
                return placeholder
            if dest.startswith("data:"):
                bm = _BASE64_IMG_RE.match(dest)
                if not bm:
                    evidence.append(ev)
                    return placeholder
                ext = _IMG_EXT.get(bm.group(1).lower(), "png")
                try:
                    data = base64.b64decode(bm.group(2))
                except Exception:
                    evidence.append(ev)
                    return placeholder
                b64_sha = hashlib.sha256(data).hexdigest()
                # 内容寻址命名：同图只落一份（跨文档自动去重），且不含 md 文件名 → 同名 md 也不撞
                name = f"img_{b64_sha[:16]}.{ext}"
                target = images_dir / name
                if not target.is_file() or _sha256(target) != b64_sha:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                sha = b64_sha
                ev.update({"kind": "base64", "asset_id": sha[:16], "sha256": sha})
                _reg(f"assets/images/{name}", sha, target)
                ocr_ev, inject = self._ocr_artifact_for(target, sha, f"assets/images/{name}")
                if ocr_ev is not None:
                    ev["ocr"] = ocr_ev
                evidence.append(ev)
                return placeholder + inject
            # 本地路径（file:// 剥前缀；相对路径以 md 目录为基准）
            local = dest[len("file://"):] if dest.startswith("file://") else dest
            p = Path(local)
            if not p.is_absolute():
                p = md_path.parent / p
            try:
                p = p.resolve()
            except OSError:
                evidence.append(ev)
                return placeholder
            if not p.is_file() and assets_only:
                # 回退模式：相对结构已丢失，按文件名到资产层尽力反查（命中即视为该图）
                cand = images_dir / Path(local).name
                if cand.is_file():
                    p = cand.resolve()
            if not p.is_file():   # 目录/不存在一律 unresolved（不记 source_ref，避免路径形态进库）
                evidence.append(ev)
                return placeholder
            rel, sha, _ = _copy_asset(p, self.resolve("data/assets"), "images")
            ev.update({"kind": "local", "asset_id": sha[:16], "sha256": sha})
            _reg(rel, sha, self.resolve(rel))
            ocr_ev, inject = self._ocr_artifact_for(self.resolve(rel), sha, rel)
            if ocr_ev is not None:
                ev["ocr"] = ocr_ev
            evidence.append(ev)
            return placeholder + inject

        body, _refs = rewrite_images(text, resolve)
        return body, evidence, registered


class PdfSource(BaseSource):
    """PDF 手册来源（操作手册 / 接口指南：硬件排查命令、错误码参考）。

    配置示例：
        {"id": "manuals", "type": "pdf", "path": "data/imports/pdf"}

    - pull(): 扫描配置 path 下 *.pdf，复制到 data/assets/pdf/（不可变层，sha256）；
    - canonicalize(): PyMuPDF 文字层 → Markdown 全文；页面表格（错误码表/命令表）转
      Markdown 表格附于正文（保证 FTS 可检索），另存结构化 JSON 到 data/parsed/pdf/；
    - verification=expert（官方操作手册默认专家验证，无需审核补标）。
    """

    type = "pdf"

    def __init__(self, cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                 app_cfg: Optional["AppConfig"] = None):
        super().__init__(cfg, project_root, app_cfg=app_cfg)
        self.import_dir = self.resolve(self.cfg.get("path", f"data/imports/{self.id}"))

    # ---------- 布局 ----------

    def _assets_dir(self) -> Path:
        return self.resolve("data/assets/pdf")

    def _parsed_dir(self) -> Path:
        return self.resolve("data/parsed/pdf")

    # ---------- 采集 ----------

    def pull(self, max_issues: Optional[int] = None) -> int:
        """扫描导入目录，把 PDF 复制到资产层（幂等）。返回新增条数。"""
        if not self.import_dir.exists():
            print(f"[sources:{self.id}] 导入目录不存在: {self.import_dir}")
            return 0
        added = 0
        registered: list[tuple[str, str, str]] = []
        for p in sorted(self.import_dir.rglob("*.pdf")):
            rel, sha, copied = _copy_asset(p, self.resolve("data/assets"), "pdf")
            registered.append(_asset_entry(rel, sha, "doc_pdf", self.resolve(rel)))
            if copied:
                added += 1
        self._register_asset_mappings(registered)
        print(f"[sources:{self.id}] 资产层扫描完成（新增 {added} 个 pdf）")
        return added

    # ---------- 解析（可重跑：只读资产层） ----------

    def canonicalize(self) -> list[KbDocument]:
        """从资产层 PDF 解析出 KbDocument（每篇一个，body=Markdown 全文）。

        表格策略：页面表格转 Markdown 表格拼入正文（错误码/命令可被 FTS 检索），
        同时写入 parsed/pdf/{asset_id}.tables.json 供结构化消费（图/查询）。

        PyMuPDF 逐页提取文字与表格较耗时（大手册如 200+ 页需数秒~数十秒）——
        逐篇打印进度（序号/页数/耗时），recanonicalize / 重新入库时可见进展。
        """
        try:
            import pymupdf  # PyMuPDF 1.28+（旧名 fitz 已弃用）
        except ImportError as e:
            print(f"[sources:{self.id}] 未安装 pymupdf：pip install pymupdf（{e}）")
            return []
        docs: list[KbDocument] = []
        assets = self._assets_dir()
        if not assets.exists():
            return docs
        parsed_dir = self._parsed_dir()
        parsed_dir.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(assets.glob("*.pdf"))
        if not pdfs:
            return docs
        total = len(pdfs)
        start_ts = time.time()
        print(f"[sources:{self.id}] 解析 {total} 个 PDF（PyMuPDF 逐页提取，大手册耗时较长）…",
              flush=True)
        for i, p in enumerate(pdfs, 1):
            t0 = time.time()
            try:
                result = self._parse_pdf(p, parsed_dir)
            except Exception as e:
                print(f"[sources:{self.id}] [{i}/{total}] 解析失败 {p.name}: {e}（跳过）",
                      flush=True)
                continue
            if result is None:
                continue
            doc, cached = result
            docs.append(doc)
            pages = (doc.extra.get("asset") or {}).get("pages", "?")
            cache_tag = "，缓存命中" if cached else ""
            print(f"[sources:{self.id}] [{i}/{total}] 解析完成 {p.name}（{pages} 页，"
                  f"{time.time() - t0:.1f}s{cache_tag}）", flush=True)
        print(f"[sources:{self.id}] 解析完成：成功 {len(docs)}/{total}（耗时 "
              f"{time.time() - start_ts:.0f}s）", flush=True)
        return docs

    def _parse_pdf(self, p: Path, parsed_dir: Path):
        """解析单篇 PDF，返回 (KbDocument | None, 是否缓存命中)。

        **缓存优先**：耗时大头是 PyMuPDF 逐页提取（大手册 200+ 页约数秒~数十秒）；
        解析中间产物（文字层 body + 表格 + 首行/页数）按 asset_id（sha256 前缀，内容寻址）
        缓存到 `parsed/pdf/<asset_id>.extract.json`——PDF 未变（sha256 一致）时直接复用缓存，
        仅重跑确定性提取（标签/元数据，毫秒级），提取规则升级后**无需清缓存**即可生效；
        删除 `parsed/pdf/` 目录即强制全量重解析。
        """
        sha = _sha256(p)
        asset_id = sha[:16]
        registry = TagRegistry.load(self.app_cfg) if self.app_cfg else TagRegistry()
        cache = parsed_dir / f"{asset_id}.extract.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if data.get("sha256") == sha:
                    return self._doc_from_extract(p, sha, asset_id, data, registry), True
            except (OSError, ValueError):
                pass  # 缓存损坏 → 重新解析
        parsed = self._extract_pdf(p)
        if parsed is None:
            return None, False
        cache.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        self._write_tables(parsed_dir, asset_id, parsed.get("tables") or [])
        return self._doc_from_extract(p, sha, asset_id, parsed, registry), False

    def _extract_pdf(self, p: Path):
        """PyMuPDF 逐页提取（慢，结果可缓存）：文字层 → Markdown 正文 + 结构化表格。

        返回 {"sha256", "asset_id", "pages", "first_text", "body", "tables"}；
        加密/无文字层返回 None（调用方跳过）。
        """
        import pymupdf

        pdf = pymupdf.open(str(p))
        try:
            if pdf.needs_pass:
                print(f"[sources:{self.id}] 跳过加密 PDF: {p.name}")
                return None
            md_parts: list[str] = []
            tables: list[dict] = []
            first_text = ""
            for page_no, page in enumerate(pdf, 1):
                text = page.get_text("text")
                if text.strip() and not first_text:
                    first_text = text.strip().splitlines()[0][:120]
                # 页面表格 → Markdown 表格 + 结构化 JSON
                try:
                    page_tables = page.find_tables()
                except Exception:
                    page_tables = None
                for i, tab in enumerate((page_tables or {}).tables or []):
                    data = tab.extract()
                    if not data:
                        continue
                    md_parts.append(_table_to_markdown(data))
                    tables.append({"page": page_no, "index": i, "rows": data})
                if text.strip():
                    md_parts.append(text.strip())
            body = "\n\n".join(md_parts).strip()
            if not body:
                print(f"[sources:{self.id}] 跳过无文字层 PDF（可能为扫描件，待 OCR）: {p.name}")
                return None
            return {
                "sha256": _sha256(p),
                "asset_id": _sha256(p)[:16],
                "pages": pdf.page_count,
                "first_text": first_text,
                "body": body,
                "tables": tables,
            }
        finally:
            pdf.close()

    def _write_tables(self, parsed_dir: Path, asset_id: str, tables: list[dict]) -> list[str]:
        """结构化表格落盘（可重跑产物，以 asset_id 命名——不暴露文件名/路径）。"""
        if not tables:
            return []
        tpath = parsed_dir / f"{asset_id}.tables.json"
        tpath.write_text(
            json.dumps({"source": f"pdf:{asset_id}", "tables": tables},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        return [f"parsed/pdf/{tpath.name}"]

    def _doc_from_extract(self, p: Path, sha: str, asset_id: str, parsed: dict,
                          registry: TagRegistry) -> KbDocument:
        """用解析中间产物构造 KbDocument（确定性提取，毫秒级，每次运行重算）。

        缓存命中与首次解析共用本函数——标签/元数据提取始终以最新规则执行，
        解析器升级（pymupdf 版本/提取逻辑）不影响提取结果的一致性。
        """
        body = str(parsed.get("body") or "")
        title = str(parsed.get("first_text") or "").strip() or p.stem
        tags, cands = extract_tags(p.stem, headings_from_pdf(body), registry=registry)
        # tables.json 由 _parse_pdf 首次解析时写入；缓存命中时文件已存在，rel 引用直接构造
        tables_rel = [f"parsed/pdf/{asset_id}.tables.json"] if parsed.get("tables") else []
        return KbDocument(
            source_type="doc_pdf",
            source_id=f"pdf:{p.stem}",
            url="",
            title=title,
            body=body,
            created_at=None,
            component="",
            tags=[t.name for t in tags],
            extra={
                "asset": {"asset_id": asset_id, "sha256": sha,
                          "format": "pdf", "pages": int(parsed.get("pages") or 0)},
                "quality": {"text_source": "text_layer", "parsed_with": "pymupdf"},
                "verification": "expert",  # 官方操作手册默认专家验证
                "structure": {"tables": tables_rel},
                # 未收录强候选（进审核队列 tag_candidate 人工采纳后入词典）
                "tag_candidates": [{"name": c.name, "tier": c.tier} for c in cands],
            },
        )


def _table_to_markdown(rows: list[list]) -> str:
    """二维表 → Markdown 表格（第一行作表头；列数取最长行并补齐）。"""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in norm[0]) + " |"]
    lines.append("|" + "---|" * width)
    for r in norm[1:]:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
    return "\n".join(lines)


# ---------------- Word（.docx）解析辅助 ----------------

# 标题样式名：英文 "Heading 1" / 中文 "标题 1" / 繁体 "標題 1"；style_id 通常恒为 "Heading1"
_WORD_HEADING_RE = re.compile(r"^(?:heading|标题|標題)\s*([1-6])$", re.IGNORECASE)
_WORD_TITLE_STYLES = {"title", "标题", "標題"}


def _word_heading_level(par) -> int:
    """段落标题层级：Heading 1-6 / "标题 1" → 1-6；Title / "标题" → 1；非标题 → 0。

    中英界面下样式名不同（"Heading 1" vs "标题 1"），但 `style_id` 一般不受界面语言影响，
    两者都匹配、任一命中即可。
    """
    st = getattr(par, "style", None)
    name = (getattr(st, "name", "") or "").strip()
    sid = (getattr(st, "style_id", "") or "").replace(" ", "").strip()
    for cand in (name, sid):
        m = _WORD_HEADING_RE.match(cand)
        if m:
            return int(m.group(1))
    return 1 if name.lower() in _WORD_TITLE_STYLES else 0


def _word_list_prefix(par) -> str:
    """列表项前缀：无序 → `- `；有序 → `1. `（Markdown 自动编号）；非列表 → ``。

    列表可能来自直接编号格式（`pPr/numPr`）或样式名（List Bullet / List Number），两者都看
    （Word 里用样式而非直接格式建的列表很常见，只看 numPr 会漏）。
    """
    name = (getattr(getattr(par, "style", None), "name", "") or "").strip().lower()
    pPr = getattr(getattr(par, "_p", None), "pPr", None)
    if not name.startswith("list") and (pPr is None or pPr.numPr is None):
        return ""
    if "number" in name or "编号" in name or "数字" in name:
        return "1. "
    return "- "


def _iter_docx_blocks(document):
    """按**文档顺序**产出 Paragraph / Table。

    python-docx 的 `.paragraphs` 与 `.tables` 是两个互不相干的列表，各自都不保留混排顺序——
    直接用会把表格全挪到正文末尾，章节归属也就错了（表格落在错误的 section 下）。
    """
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


# ---------------- Word 内嵌图片（OOXML 包内 media part） ----------------

# VML 命名空间：旧版/兼容模式粘贴的图片走 `v:imagedata`。python-docx 的 nsmap **不含 v**，
# 不能经 qn() 取，这里显式写 URI。
_VML_IMAGEDATA = "{urn:schemas-microsoft-com:vml}imagedata"
# 标记兼容分支：Word 给部分图形/文本框写 `mc:AlternateContent`（Choice=DrawingML +
# Fallback=VML），**两支引用同一个 rId**。全遍历会把一张图数成两张（重复占位 + 重复资产），
# 按 OOXML 惯例只认 Choice、丢掉 Fallback 子树。
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
# 送 OCR 的栅格格式：emf/wmf/svg 是矢量、tiff 多页支持参差，OCR 引擎不收（白花一次调用）
_OCR_IMAGE_EXTS = frozenset({"png", "jpg", "jpeg", "gif", "webp", "bmp"})
# content_type → 后缀（partname 后缀缺失/异常时的兜底）
_CT_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/gif": "gif",
           "image/webp": "webp", "image/bmp": "bmp", "image/x-ms-bmp": "bmp",
           "image/tiff": "tiff", "image/x-emf": "emf", "image/emf": "emf",
           "image/x-wmf": "wmf", "image/wmf": "wmf", "image/svg+xml": "svg"}
# 解析中间产物 schema 位：1=纯文本/表格；2=+内嵌图片清单（占位 + 资产）。
# **必须校验**——`_extract_docx` 升级后旧缓存仍带旧 sha256，不校验的话图片永远不会生效，
# 而且现象是"静默无图"（比报错难查得多）。
_WORD_EXTRACT_SCHEMA = 2


def _word_walk(el):
    """按文档顺序深度优先遍历子元素，跳过 `mc:Fallback` 分支与注释/PI。"""
    for child in el:
        if not isinstance(child.tag, str):
            continue
        if child.tag == _MC_FALLBACK:
            continue
        yield child
        yield from _word_walk(child)


def _word_image_ext(part) -> str:
    """图片 part 的后缀（小写，jpeg→jpg）：优先 partname 后缀，其次 content_type。

    注意 python-docx 的 `ImagePart` **没有** `.ext` 属性（实测），只能从 partname/content_type 推。
    """
    ext = Path(str(getattr(part, "partname", "") or "")).suffix.lstrip(".").lower()
    if ext:
        return "jpg" if ext == "jpeg" else ext
    return _CT_EXT.get(str(getattr(part, "content_type", "") or "").lower(), "png")


def _word_images_in(el) -> list[tuple[str, str]]:
    """元素内的图片引用（按 XML 文档顺序）→ `[(rId, alt)]`；段落与表格共用。

    两种承载方式都扫：DrawingML（`a:blip/@r:embed`，现代 Word）与 VML（`v:imagedata/@r:id`，
    兼容模式/旧版粘贴）。一个块里多张图时先后关系是准的。
    `r:link` 是**外链**图片（内容不在包里）——照样产占位（用户看得见图），但取不到字节、
    也不该把目标路径记进库，所以只取 rId 不记链接。
    alt 取 `wp:docPr/@descr`（`wp:docPr` 位于 `a:blip` 之前，先记后用）。
    """
    from docx.oxml.ns import qn

    out: list[tuple[str, str]] = []
    alt = ""
    for node in _word_walk(el):
        tag = node.tag
        if tag == qn("wp:docPr"):
            alt = str(node.get("descr") or "").strip()
        elif tag == qn("a:blip"):
            rid = node.get(qn("r:embed")) or node.get(qn("r:link"))
            if rid:
                out.append((rid, alt))
        elif tag == _VML_IMAGEDATA:
            rid = node.get(qn("r:id"))
            if rid:
                out.append((rid, alt))
    return out


def _word_alt_text(alt: str) -> str:
    """alt 规范化：压掉换行/连续空白并限长——alt 是文档内容（随正文进 FTS），畸形值能到几 KB。"""
    return " ".join(str(alt or "").split())[:120]


def _word_img_placeholder(alt: str) -> str:
    """图片**不透明占位符**（与 markdown 同构）：有 alt → `[图片:alt]`，否则 `[图片]`。"""
    a = _word_alt_text(alt)
    return f"[图片:{a}]" if a else "[图片]"


def _word_image_entry(document, rid: str, alt: str, *, in_body: bool) -> dict:
    """图片清单条目：读包内字节算 sha256（**内容寻址**），取不到内容则 `sha256=""`。

    sha256 为空 = 外链图（`r:link`）或损坏引用——内容不在包里，只留占位、不登记资产。
    条目按**出现次数**记（同图在正文出现两次就是两条），顺序即正文占位符顺序。
    """
    part = document.part.related_parts.get(rid)
    blob = getattr(part, "blob", None) if part is not None else None
    if not isinstance(blob, (bytes, bytearray)):
        return {"rid": rid, "sha256": "", "ext": "", "alt": _word_alt_text(alt),
                "in_body": in_body}
    return {"rid": rid, "sha256": hashlib.sha256(bytes(blob)).hexdigest(),
            "ext": _word_image_ext(part), "alt": _word_alt_text(alt), "in_body": in_body}


class ImageSource(BaseSource):
    """图片证据 OCR 来源：对 data/assets/images/ 未 OCR 的图片做**签名导向 OCR**。

    配置示例：
        {"id": "images", "type": "image",
         "ocr_provider": "ask",               # ask(默认) | api | paddle | none
         "ocr_api_base": "http://<ocr-svc>:8000",   # api 模式：HTTP OCR 服务
         "ocr_api_key": ""}                   # 可选，或环境变量 OCR_API_KEY

    - 图片随 markdown/pdf 导入进资产层（assets/images/），本来源只做 OCR；
    - canonicalize(): 扫描 assets/images/*，对每张图（幂等：ocr.json 记录 sha256 一致则跳过）
      OCR → 提取错误签名 → 写 data/parsed/images/<stem>.ocr.json；
      **返回 []**（图片不单独成文档；OCR 产物由所属文档 extra.evidence 引用）；
    - OCR 引擎决策（无 API 时的交互）：
      * provider=ask（默认）：有 ocr_api_base → 走 API；无 → **询问**"是否本地运行（paddle）"，
        否定（或非交互终端）→ **跳过 OCR**（不写产物，导入不受阻）；
      * provider=api：调 API；调用失败 → 询问本地/跳过（每次运行最多问一次）；
      * provider=paddle：本地运行（明确选择，不询问；未安装 → 提示并跳过）；
      * provider=none：明确跳过。
    """

    type = "image"
    _IMG_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif")

    def _images_dir(self) -> Path:
        return self.resolve("data/assets/images")

    def _parsed_dir(self) -> Path:
        return self.resolve("data/parsed/images")

    def pull(self, max_issues: Optional[int] = None) -> int:
        """图片随 md/pdf 导入资产层，本来源无独立采集。"""
        return 0

    def _ask_local_ocr(self) -> bool:
        """无可用 OCR API 时询问是否本地运行（paddle）。非交互终端默认跳过。"""
        try:
            interactive = __import__("sys").stdin.isatty()
        except Exception:
            interactive = False
        if not interactive:
            return False
        try:
            ans = input("[ocr] OCR API 不可用，是否本地运行（paddleocr）？[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in ("y", "yes")

    def canonicalize(self) -> list[KbDocument]:
        import os

        from .ocr import (OcrApiError, OcrUnavailable, as_confidence_threshold,
                          build_ocr_artifact, engine_fingerprint, load_ocr_artifact,
                          ocr_image_detail, save_ocr_artifact)

        images = self._images_dir()
        if not images.exists():
            return []
        all_imgs = sorted(p for g in self._IMG_GLOBS for p in images.glob(g))
        shas = {p: _sha256(p) for p in all_imgs}
        # 图片资产注册（审核侧经 asset_id 反查路径/预览）：**与 OCR 是否启用无关**——
        # 手工投放到 assets/images/ 的图也要可反查，故在 provider 决策之前注册。
        # （md 引用到的图片由 MarkdownSource 注册；此处覆盖"无 md 引用"的图片，幂等 upsert）
        self._register_asset_mappings([
            _asset_entry(f"assets/images/{p.name}", shas[p], "image", p) for p in all_imgs
        ])
        parsed_dir = self._parsed_dir()
        parsed_dir.mkdir(parents=True, exist_ok=True)

        # ---- OCR 引擎决策 ----
        # 阈值与其他 ocr_* 字段一样直接读本来源配置（本来源即 image source）
        min_conf = as_confidence_threshold(self.cfg.get("ocr_min_confidence"))
        provider = str(self.cfg.get("ocr_provider", "ask") or "ask").lower()
        api_base = str(self.cfg.get("ocr_api_base", "") or "")
        api_key = str(self.cfg.get("ocr_api_key", "") or os.environ.get("OCR_API_KEY", ""))
        api_model = str(self.cfg.get("ocr_api_model", "") or "")
        api_mode = str(self.cfg.get("ocr_api_mode", "custom") or "custom")
        if provider == "ask":
            provider = "api" if api_base else ("paddle" if self._ask_local_ocr() else "none")
        if provider == "none":
            print(f"[sources:{self.id}] 跳过 OCR（可配置 ocr_provider: paddle 本地 / "
                  f"api + ocr_api_base 服务，或安装 paddleocr）")
            return []

        processed, skipped, failed, review = 0, 0, 0, 0
        asked = False  # API 失败后的本地询问只问一次
        for img in all_imgs:
            if provider == "none":
                break
            sha = shas[img]
            ocr_path = parsed_dir / f"{img.stem}.ocr.json"
            fp = engine_fingerprint(provider, api_mode, api_model)
            cached = load_ocr_artifact(ocr_path, sha, fp)
            if cached is not None:
                skipped += 1
                # 阈值改了但图片/引擎未变：不重跑 OCR，只按新阈值刷新判定字段
                if cached.min_confidence != min_conf:
                    cached.min_confidence = min_conf
                    save_ocr_artifact(ocr_path, cached, image_ref=f"assets/images/{img.name}")
                continue
            result = None
            while True:
                try:
                    result = ocr_image_detail(img, provider, api_base=api_base,
                                              api_key=api_key, model=api_model, mode=api_mode)
                    break
                except OcrApiError as e:
                    print(f"[sources:{self.id}] OCR API 失败: {e}", flush=True)
                    if not asked:
                        asked = True
                        if self._ask_local_ocr():
                            provider = "paddle"
                            fp = engine_fingerprint(provider, api_mode, api_model)
                            continue  # 换本地重试当前图
                    provider = "none"
                    break
                except OcrUnavailable as e:
                    print(f"[sources:{self.id}] OCR 不可用（{e}）——跳过 OCR", flush=True)
                    provider = "none"
                    break
            if provider == "none":
                print(f"[sources:{self.id}] 跳过 OCR（图片 {len(list(images.iterdir()))} 张，"
                      f"已处理 {processed}）。可配置 ocr_provider: paddle 或 api + ocr_api_base")
                break
            if result is None:
                failed += 1
                continue
            art = build_ocr_artifact(img, sha, result, provider, api_mode, api_model, min_conf)
            save_ocr_artifact(ocr_path, art, image_ref=f"assets/images/{img.name}")
            if art.review_reason:
                review += 1
                print(f"[sources:{self.id}] 图片 {img.name} OCR 需人工复核"
                      f"（reason={art.review_reason} conf={art.confidence}）——不进正文", flush=True)
            processed += 1
        total = len(list(images.iterdir()))
        print(f"[sources:{self.id}] OCR 完成：新增 {processed}，跳过（幂等）{skipped}，"
              f"失败 {failed}，待人工复核 {review}（图片 {total} 张；"
              f"高置信文本已随所属文档正文入库）")
        return []


class ExcelSource(BaseSource):
    """Excel 表格来源（工程师问题定位记录/已知问题登记表等，**格式未知**）。

    配置示例：
        {"id": "engineer-troubleshooting", "type": "excel",
         "path": "data/imports/engineer/问题定位记录.xlsx", "enabled": true}

    **schema-free 设计（不写死任何列名/sheet 名/行号）**：
    - pull：把配置 path（文件或目录）下的 .xlsx/.xlsm 复制到资产层；
    - canonicalize：遍历**所有 sheet、所有行**，把每行的非空 cell **按列序拼接成一段
      自由文本**作为 body——不依赖表头/列语义；每行一条 KbDocument（source_id 用
      `excel:{stem}:{sheet序号}:{行号}` 保证唯一，行号仅作标识、非解析依赖）；
    - **实体提取复用现有线路**：body 进入 canonical 后，错误码/算子/模型/版本由
      signature 三层提取自动入图（scheme-free：图构建只依赖 canonical）；
    - **脱敏**：cell 值经 sanitize_text（内部 IP → &lt;IP&gt;、内部路径 → &lt;PATH&gt;，
      默认路径/日志路径白名单保留），防止内部数据经检索外泄；
    - 验证状态：登记表默认 `unverified`、status=open（低优先级，按未解决 issue 处理）。
    """

    type = "excel"
    _EXCEL_SUFFIXES = (".xlsx", ".xlsm")

    def __init__(self, cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                 app_cfg: Optional["AppConfig"] = None):
        super().__init__(cfg, project_root, app_cfg=app_cfg)
        self.import_path = self.resolve(self.cfg.get("path", f"data/imports/{self.id}"))

    def _assets_dir(self) -> Path:
        return self.resolve("data/assets/excel")

    def _excel_files(self) -> list[Path]:
        p = self.import_path
        if p.is_dir():
            out = []
            for suffix in self._EXCEL_SUFFIXES:
                out.extend(sorted(p.rglob(f"*{suffix}")))
            return out
        if p.is_file() and p.suffix in self._EXCEL_SUFFIXES:
            return [p]
        return []

    def pull(self, max_issues: Optional[int] = None) -> int:
        """把 excel 文件复制到资产层（幂等）。返回新增条数。"""
        files = self._excel_files()
        if not files:
            print(f"[sources:{self.id}] 导入路径无 excel 文件: {self.import_path}")
            return 0
        added = 0
        registered: list[tuple[str, str, str]] = []
        for f in files:
            rel, sha, copied = _copy_asset(f, self.resolve("data/assets"), "excel")
            registered.append(_asset_entry(rel, sha, "doc_excel", self.resolve(rel)))
            if copied:
                added += 1
        self._register_asset_mappings(registered)
        print(f"[sources:{self.id}] 资产层扫描完成（新增 {added} 个 excel）")
        return added

    def canonicalize(self) -> list[KbDocument]:
        """遍历所有 sheet/行，行 cell 拼接为 body（schema-free）。

        **后置脱敏**：body 以**原文入库**（原文检索）；仅按 config.sanitize 扫描会被脱敏的
        IP/路径，落盘 sanitize_log.json 供维护白名单（出口脱敏由 serve_api 返回时统一做）。
        """
        from .sanitize import collect_sanitize_hits

        sanitize_on, keep_paths, keep_ips = self.sanitize_params()
        collector: dict = {}  # 会被脱敏的原始 IP/路径（落盘维护，不进库）

        try:
            import openpyxl
        except ImportError as e:
            print(f"[sources:{self.id}] 未安装 openpyxl：pip install openpyxl（{e}）")
            return []
        docs: list[KbDocument] = []
        assets = self._assets_dir()
        if not assets.exists():
            return docs
        n_files = 0
        for suffix in self._EXCEL_SUFFIXES:
            for p in sorted(assets.glob(f"*{suffix}")):
                n_files += 1
                sha = _sha256(p)
                asset_id = sha[:16]
                try:
                    wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
                except Exception as e:
                    print(f"[sources:{self.id}] 读取失败 {p.name}: {e}（跳过）")
                    continue
                try:
                    for sheet_idx, ws in enumerate(wb.worksheets, 1):
                        for row_idx, row in enumerate(ws.iter_rows(values_only=True), 1):
                            cells = [str(c).strip() for c in row
                                     if c is not None and str(c).strip()]
                            if not cells:
                                continue  # 空行跳过（不依赖行号语义）
                            body = " ".join(cells)
                            if sanitize_on:
                                ips, paths = collect_sanitize_hits(body, keep_paths, keep_ips)
                                if ips:
                                    collector.setdefault("ips", set()).update(ips)
                                if paths:
                                    collector.setdefault("paths", set()).update(paths)
                            if not body.strip():
                                continue
                            title = cells[0][:80]
                            docs.append(KbDocument(
                                source_type="doc_excel",
                                source_id=f"excel:{p.stem}:{sheet_idx}:{row_idx}",
                                url="",
                                title=title or f"{p.stem} {sheet_idx} 行 {row_idx}",
                                body=body,
                                created_at=None,
                                component="",
                                tags=[],  # Excel 不做文件名/标题标签（正文候选走 build_tag_candidates）
                                extra={
                                    "asset": {"asset_id": asset_id, "sha256": sha,
                                              "format": "excel"},
                                    "quality": {"text_source": "table", "parsed_with": "openpyxl"},
                                    "verification": "unverified",  # 登记表低优先级
                                },
                            ))
                finally:
                    wb.close()
        # 被脱敏命中的 IP/路径落盘维护文件（data/sanitize_log.json，幂等合并；app_cfg 缺失时跳过）
        self.save_sanitize_log(collector)
        print(f"[sources:{self.id}] canonical {len(docs)} 条（{n_files} 个 excel）")
        return docs


class WordSource(BaseSource):
    """Word 文档来源（案例 / 故障复盘 / 操作记录，`.docx` / `.docm`）。

    配置示例：
        {"id": "cases", "type": "word", "path": "data/imports/word", "enabled": true}

    - pull()：扫描配置 path（文件或目录）下 `*.docx`/`*.docm`，复制到 `data/assets/word/`
      （不可变层，sha256）；
    - canonicalize()：python-docx 按**文档顺序**单遍解析 → Markdown 全文：
      * 标题样式（Heading 1-6 / 中文"标题 1" / Title）→ `#`~`######`。这是相对 PDF 的实质增益：
        分块直接复用 `_split_markdown_sections`，chunk 带 `section`（命中正文能看到所属章节）；
      * 列表样式（List Bullet / List Number，或直接编号格式）→ `- ` / `1. `，避免条目粘成一段
        （粘成一段会让 FTS 命中粒度变差）；
      * 表格 → Markdown 表格拼入正文（表内错误码/命令可被 FTS 检索），另存
        `data/parsed/word/{asset_id}.tables.json` 供结构化消费（图/查询）；
      * 内嵌图片 → **不透明占位符** `[图片]`/`[图片:alt]` 按文档顺序插入正文，图片本身落
        `data/assets/images/img_{sha256[:16]}.{ext}`（内容寻址，跨文档去重）并注册 asset_registry；
        正文出现过的图片走 OCR（`ocr.json` 幂等缓存），高置信文本注入占位符之后参与检索，
        低置信/自报异常只留签名线索并进审核队列（与 markdown 图片同一套通路）；
    - 解析中间产物按 asset_id（内容寻址）缓存到 `parsed/word/{asset_id}.extract.json`，与 PDF 同构：
      耗时提取复用缓存，标签/元数据/OCR 注入每轮重算（升级提取规则或调 OCR 阈值**无需清缓存**；
      缓存带 schema 位，提取逻辑升级会自动失效旧缓存）；
    - verification=unverified（与 markdown/excel **统一路径**：人工文档先入库，审核工作台补标）；
    - 与 markdown 同构：**优先读导入目录**（保留目录树 → 同名不同目录可消歧、编辑源文件不会
      因资产层累积副本而变成多篇），导入目录扫不到时回退资产层扁平副本并按版本族收敛。

    **本版边界**（因此正文不含任何路径）：
    - 页眉/页脚/脚注/尾注是**独立部件**（`/word/header1.xml` 等），不在正文遍历范围内 →
      其文字与图片都不提取；
    - 文本框：其**图片**在正文 XML 内（`w:txbxContent`）会被遍历到并产占位，但其**文字**
      不在 `Paragraph.text` 里（python-docx 只取直接 run）→ 文字不提取（两者不对称，已知）；
    - 表格内图片的占位符统一排在表格之后（塞进单元格会破 Markdown 表格）；
    - 嵌套表格只取外层单元格文本；合并单元格重复文本；
    - 矢量/多页图片（emf/wmf/svg/tiff）注册资产但不送 OCR（引擎不收，白花一次调用）；
    - `.doc`（旧二进制格式）与加密 docx 不支持 → 明确跳过并提示另存为 `.docx`。
    """

    type = "word"
    _SUFFIXES = ("*.docx", "*.docm")

    def __init__(self, cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                 app_cfg: Optional["AppConfig"] = None):
        super().__init__(cfg, project_root, app_cfg=app_cfg)
        self.import_dir = self.resolve(self.cfg.get("path", f"data/imports/{self.id}"))

    # ---------- 布局 ----------

    def _assets_dir(self) -> Path:
        return self.resolve("data/assets/word")

    def _parsed_dir(self) -> Path:
        return self.resolve("data/parsed/word")

    def _word_files(self) -> list[Path]:
        """导入路径下的 Word 文件（path 可为文件或目录）。"""
        p = self.import_dir
        if p.is_dir():
            out: list[Path] = []
            for pat in self._SUFFIXES:
                out.extend(sorted(p.rglob(pat)))
            return out
        if p.is_file() and p.suffix.lower() in (".docx", ".docm"):
            return [p]
        return []

    # ---------- 采集 ----------

    def pull(self, max_issues: Optional[int] = None) -> int:
        """扫描导入路径，把 Word 文件复制到资产层（幂等）。返回新增条数。"""
        files = self._word_files()
        if not files:
            print(f"[sources:{self.id}] 导入路径无 Word 文件: {self.import_dir}")
            return 0
        added = 0
        registered: list[tuple] = []
        for f in files:
            rel, sha, copied = _copy_asset(f, self.resolve("data/assets"), "word")
            registered.append(_asset_entry(rel, sha, "doc_word", self.resolve(rel)))
            if copied:
                added += 1
        self._register_asset_mappings(registered)
        print(f"[sources:{self.id}] 资产层扫描完成（新增 {added} 个 word）")
        return added

    # ---------- 解析（可重跑：优先读导入目录） ----------

    def canonicalize(self) -> list[KbDocument]:
        """解析 Word → KbDocument（每篇一个，body=Markdown 全文，标题层级保留为 `#`）。

        **后置脱敏**：body 以**原文入库**（原文检索）；仅按 config.sanitize 扫描会被脱敏的
        IP/路径落盘 sanitize_log.json 供维护白名单（出口脱敏由 serve_api 返回时统一做）。
        """
        try:
            import docx  # noqa: F401  （仅探测可用性，实际解析在 _extract_docx）
        except ImportError as e:
            print(f"[sources:{self.id}] 未安装 python-docx：pip install python-docx（{e}）")
            return []
        from .sanitize import collect_sanitize_hits

        docs: list[KbDocument] = []
        registry = TagRegistry.load(self.app_cfg) if self.app_cfg else TagRegistry()
        sanitize_on, keep_paths, keep_ips = self.sanitize_params()
        collector: dict = {}  # 会被脱敏的原始 IP/路径（落盘维护，不进库）
        files, fallback = _discover_source_files(
            self, self._SUFFIXES, self.import_dir, self._assets_dir())
        if fallback:
            print(f"[sources:{self.id}] ⚠ 导入目录无 Word 文件（{self.import_dir}），已回退到资产层"
                  f"副本 {self._assets_dir()}（按版本族收敛，每族只取最新一份）；"
                  f"如需按目录树消歧/保留原始组织方式，请恢复 imports 目录", flush=True)
        if not files:
            return docs
        # 同名 stem 冲突检测：word:<stem> 会互相覆盖（ingest 用 INSERT OR REPLACE，后者胜）
        stem_counts: dict[str, int] = {}
        for _p, _fi, stem in files:
            stem_counts[stem] = stem_counts.get(stem, 0) + 1
        dup_stems = {s for s, n in stem_counts.items() if n > 1}
        if dup_stems:
            shown = ", ".join(sorted(dup_stems)[:5]) + (" …" if len(dup_stems) > 5 else "")
            print(f"[sources:{self.id}] ⚠ 检测到 {len(dup_stems)} 组同名 Word（{shown}）："
                  f"已用相对路径指纹消歧为 word:<stem>--<sha8>，避免互相覆盖", flush=True)
        parsed_dir = self._parsed_dir()
        parsed_dir.mkdir(parents=True, exist_ok=True)
        total = len(files)
        start_ts = time.time()
        img_assets: list[tuple] = []   # 内嵌图片资产（末尾与 word 原件同批注册）
        print(f"[sources:{self.id}] 解析 {total} 个 Word …", flush=True)
        for i, (p, from_imports, stem) in enumerate(files, 1):
            t0 = time.time()
            try:
                parsed, cached = self._parse_docx(p, parsed_dir)
            except Exception as e:
                print(f"[sources:{self.id}] [{i}/{total}] 解析失败 {p.name}: {e}（跳过）", flush=True)
                continue
            if parsed is None:
                continue
            sha = _sha256(p)
            evidence, injects, registered = self._materialize_images(p, parsed)
            img_assets.extend(registered)
            doc = self._doc_from_extract(
                p, sha, sha[:16], parsed, registry,
                sid=self._sid_for(p, stem, dup_stems, from_imports),
                evidence=evidence, injects=injects)
            if sanitize_on:
                ips, paths = collect_sanitize_hits(doc.body, keep_paths, keep_ips)
                if ips:
                    collector.setdefault("ips", set()).update(ips)
                if paths:
                    collector.setdefault("paths", set()).update(paths)
            docs.append(doc)
            cache_tag = "，缓存命中" if cached else ""
            n_img = int(doc.extra.get("quality", {}).get("images", 0))
            print(f"[sources:{self.id}] [{i}/{total}] 解析完成 {p.name}"
                  f"（{parsed.get('paragraphs', '?')} 段 / {parsed.get('tables', 0)} 表"
                  f" / {n_img} 图，{time.time() - t0:.1f}s{cache_tag}）", flush=True)
        print(f"[sources:{self.id}] 解析完成：成功 {len(docs)}/{total}（耗时 "
              f"{time.time() - start_ts:.0f}s）", flush=True)
        # 内嵌图片资产注册（审核侧经 asset_id 反查路径/预览；低置信 OCR 项由 review 扫描入队）
        self._register_asset_mappings(img_assets)
        self.save_sanitize_log(collector)
        return docs

    def _sid_for(self, p: Path, stem: str, dup_stems: set[str], from_imports: bool) -> str:
        """文档身份 `word:<stem>`；同名 stem 加相对路径指纹消歧（默认不动 → 存量 id 不漂移）。"""
        if stem not in dup_stems:
            return f"word:{stem}"
        base_dir = self.import_dir if from_imports else self._assets_dir()
        try:
            rel_key = p.relative_to(base_dir).as_posix()
        except ValueError:
            rel_key = p.name
        return f"word:{stem}--{_path_tag(rel_key)}"

    def _parse_docx(self, p: Path, parsed_dir: Path):
        """解析单篇 Word，返回 (parsed | None, 是否缓存命中)。

        **缓存优先**：`parsed/word/<asset_id>.extract.json`（asset_id = sha256 前缀，内容寻址）——
        文件未变时直接复用提取结果，仅重跑确定性提取（标签/元数据，毫秒级）；
        删除 `parsed/word/` 目录即强制全量重解析。
        """
        sha = _sha256(p)
        asset_id = sha[:16]
        cache = parsed_dir / f"{asset_id}.extract.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                # schema 位必须校验：`_extract_docx` 升级后旧缓存的 sha256 仍然匹配，
                # 不校验就会**静默无图**（比报错难查得多）
                if (data.get("sha256") == sha
                        and data.get("schema") == _WORD_EXTRACT_SCHEMA):
                    return data, True
            except (OSError, ValueError):
                pass  # 缓存损坏 → 重新解析
        parsed = self._extract_docx(p)
        if parsed is None:
            return None, False
        cache.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        self._write_tables(parsed_dir, asset_id, parsed.get("tables_data") or [])
        return parsed, False

    def _extract_docx(self, p: Path):
        """python-docx 按文档顺序单遍解析（可缓存）：正文 → Markdown + 结构化表格 + 图片清单。

        返回 {"schema", "sha256", "asset_id", "paragraphs", "tables", "images", "img_offsets",
              "first_heading", "first_text", "body", "tables_data"}；
        无法解析（非 OOXML / 加密 / 既无正文又无图片）返回 None。

        `images` 是**逐次出现**的清单（同一张图在正文出现两次就是两条），顺序即正文占位符顺序：
        `in_body=true` 的条目与正文里的 `[图片]` 占位符**一一对应**（含解析不到内容的外链图，
        以 `sha256=""` 标记）；`in_body=false` 的是包里存在但正文未引用的孤儿图，只注册资产。

        `img_offsets` 是各正文占位符在 `body` 中的**结束**偏移——OCR 文本在每轮运行时按它回插
        （见 `_apply_ocr_injections`），所以缓存里只有占位符、**不含 OCR 文本**。
        """
        import docx

        try:
            document = docx.Document(str(p))
        except Exception as e:
            print(f"[sources:{self.id}] 跳过无法解析的 Word 文件 {p.name}: {e}"
                  f"（仅支持 OOXML 的 .docx/.docm；旧版 .doc 请先另存为 .docx）")
            return None
        parts: list[str] = []
        tables_data: list[dict] = []
        images: list[dict] = []
        img_part_idx: list[int] = []   # 正文占位符在 parts 中的下标（→ 换算 body 偏移）
        first_heading = ""
        first_text = ""
        n_par = 0

        def _add_images(refs: list[tuple[str, str]]) -> None:
            """登记本块的图片引用：清单条目 + 一个独立 part 的占位符（正文不含任何路径）。"""
            for rid, alt in refs:
                images.append(_word_image_entry(document, rid, alt, in_body=True))
                img_part_idx.append(len(parts))
                parts.append(_word_img_placeholder(alt))

        for block in _iter_docx_blocks(document):
            if hasattr(block, "rows"):  # Table
                rows = [[c.text.strip() for c in row.cells] for row in block.rows]
                rows = [r for r in rows if any(r)]      # 全空行丢掉（合并单元格常见）
                if rows:
                    tables_data.append({"index": len(tables_data), "rows": rows})
                    md = _table_to_markdown(rows)
                    if md:
                        parts.append(md)
                # 表格里的图片：占位符塞进单元格会破 Markdown 表格，统一跟在表格之后。
                # 扫表格 XML（而非逐 cell.paragraphs）——合并单元格会让同一段落被重复枚举。
                _add_images(_word_images_in(block._tbl))
                continue
            text = (block.text or "").strip()
            refs = _word_images_in(block._p)
            if not text and not refs:
                continue
            if text:
                n_par += 1
                if not first_text:
                    first_text = text[:120]
                level = _word_heading_level(block)
                if level:
                    if not first_heading:
                        first_heading = text[:120]
                    parts.append("#" * level + " " + text)
                else:
                    prefix = _word_list_prefix(block)
                    parts.append(f"{prefix}{text}" if prefix else text)
            _add_images(refs)
        # 孤儿图片：包内已建立关系、正文却未引用的 media（残留关系/未走正文的部件引用）。
        # 注册资产（审核台可预览原图）但**不插占位**——没有正文落点，OCR 文本也无处可去。
        # 注意页眉/页脚/脚注是**独立部件**（/word/header1.xml 等），不在此列（本版不提取）。
        used = {str(e.get("rid") or "") for e in images}
        for rid, part in document.part.related_parts.items():
            if rid in used or not str(getattr(part, "content_type", "")).startswith("image/"):
                continue
            images.append(_word_image_entry(document, rid, "", in_body=False))
        body = "\n\n".join(parts)
        # 占位符的**结束**偏移（回插 OCR 文本用；从后往前插，偏移不失效）
        starts: list[int] = []
        off = 0
        for s in parts:
            starts.append(off)
            off += len(s) + 2
        img_offsets = [starts[i] + len(parts[i]) for i in img_part_idx]
        lead = len(body) - len(body.lstrip())
        body = body.strip()
        if lead:                       # parts 均非空且不以空白开头，理论上 lead=0；防御性对齐
            img_offsets = [o - lead for o in img_offsets]
        if not body and not images:
            print(f"[sources:{self.id}] 跳过无正文 Word 文件（可能只含页眉/页脚/文本框）: {p.name}")
            return None
        sha = _sha256(p)
        return {
            "schema": _WORD_EXTRACT_SCHEMA,
            "sha256": sha,
            "asset_id": sha[:16],
            "paragraphs": n_par,
            "tables": len(tables_data),
            "images": images,
            "img_offsets": img_offsets,
            "first_heading": first_heading,
            "first_text": first_text,
            "body": body,
            "tables_data": tables_data,
        }

    def _write_tables(self, parsed_dir: Path, asset_id: str, tables: list[dict]) -> list[str]:
        """结构化表格落盘（可重跑产物，以 asset_id 命名——不暴露文件名/路径）。"""
        if not tables:
            return []
        tpath = parsed_dir / f"{asset_id}.tables.json"
        tpath.write_text(
            json.dumps({"source": f"word:{asset_id}", "tables": tables},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        return [f"parsed/word/{tpath.name}"]

    # ---------- 内嵌图片：资产化 + OCR ----------

    def _materialize_images(self, p: Path, parsed: dict,
                            ) -> tuple[list[dict], list[str], list[tuple]]:
        """把内嵌图片落到资产层（内容寻址）并按需 OCR。

        返回 `(evidence, 正文注入后缀, 资产注册项)`；`injects` 与正文占位符**逐位对齐**
        （未 OCR / 解析不到的图片对应空串）。

        **每轮运行都执行**（不进 extract 缓存）：OCR 文本与阈值判定因此不被缓存冻结——
        调 `ocr_min_confidence`、OCR 服务从不可用恢复，下次构建即生效（与 markdown 同构）。

        - 资产命名 `img_{sha256[:16]}.{ext}`：**内容寻址**。包内 media 名（image1.png…）
          在文档之间必然重名，且同图跨文档自动去重（与 markdown 的 base64 图片共用命名空间）；
        - 只对**正文出现过的**图片做 OCR：孤儿图没有正文落点，OCR 文本无处可去
          （仍是有效资产，照常注册供审核台预览）；
        - 矢量/多页格式（emf/wmf/svg/tiff）不送 OCR（引擎不收，白花一次调用），只注册资产。
        """
        entries = parsed.get("images") or []
        if not entries:
            return [], [], []
        images_dir = self.resolve("data/assets/images")
        try:
            import docx

            rels = docx.Document(str(p)).part.related_parts
        except Exception as e:      # 已成功解析过，这里只可能是文件被移走/改坏
            print(f"[sources:{self.id}] 内嵌图片读取失败（{p.name}）：{e}（跳过图片）")
            return [], [], []
        evidence: list[dict] = []
        injects: list[str] = []
        registered: list[tuple] = []
        for ent in entries:
            in_body = bool(ent.get("in_body"))
            sha = str(ent.get("sha256") or "")
            ev: dict = {"kind": "embedded", "ocr": None}
            evidence.append(ev)
            if not in_body:
                # 孤儿图：注册资产即可，不插占位、不 OCR（没有正文落点）
                if sha:
                    target = self._write_image(images_dir, rels, ent, sha)
                    if target is not None:
                        ev.update({"asset_id": sha[:16], "sha256": sha})
                        registered.append(
                            _asset_entry(f"assets/images/{target.name}", sha, "image", target))
                continue
            if not sha:      # 外链图/损坏引用：内容不在包里 → 只留占位
                injects.append("")
                continue
            target = self._write_image(images_dir, rels, ent, sha)
            if target is None:
                injects.append("")
                continue
            rel = f"assets/images/{target.name}"
            ev.update({"asset_id": sha[:16], "sha256": sha})
            registered.append(_asset_entry(rel, sha, "image", target))
            if str(ent.get("ext") or "").lower() in _OCR_IMAGE_EXTS:
                ocr_ev, inject = self._ocr_artifact_for(target, sha, rel)
                if ocr_ev is not None:
                    ev["ocr"] = ocr_ev
                injects.append(inject)
            else:
                injects.append("")      # 矢量/多页：资产照常注册，不送 OCR
        return evidence, injects, registered

    @staticmethod
    def _write_image(images_dir: Path, rels, ent: dict, sha: str) -> Optional[Path]:
        """把清单条目对应的图片写进资产层（同 sha 已存在则跳过），返回落盘路径。"""
        target = images_dir / f"img_{sha[:16]}.{str(ent.get('ext') or 'png')}"
        if target.is_file() and _sha256(target) == sha:
            return target            # 内容寻址：同图只落一份，稳态重建不读 docx
        blob = getattr(rels.get(str(ent.get("rid") or "")), "blob", None)
        if not isinstance(blob, (bytes, bytearray)):
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(blob))
        return target

    @staticmethod
    def _apply_ocr_injections(body: str, parsed: dict, injects: list[str]) -> str:
        """把本轮 OCR 文本插回各占位符之后（extract 缓存里只有占位符、不含 OCR 文本）。

        按 `img_offsets`（占位符在 body 中的结束偏移）**从后往前**插，前面的偏移不失效。
        缓存因此不会冻结 OCR 结果——调 `ocr_min_confidence`、OCR 服务恢复，下次构建即生效。
        """
        offs = parsed.get("img_offsets") or []
        if not body or not offs or not injects:
            return body
        out = body
        for off, inj in sorted(zip(offs, injects), key=lambda t: t[0], reverse=True):
            if inj:
                out = out[:off] + inj + out[off:]
        return out

    def _doc_from_extract(self, p: Path, sha: str, asset_id: str, parsed: dict,
                          registry: TagRegistry, *, sid: str,
                          evidence: Optional[list[dict]] = None,
                          injects: Optional[list[str]] = None) -> KbDocument:
        """用解析中间产物构造 KbDocument（确定性提取，毫秒级，每次运行重算）。

        缓存命中与首次解析共用本函数——标签/元数据/OCR 注入始终以最新规则执行，
        解析器与 OCR 阈值升级都不受缓存影响。
        """
        body = self._apply_ocr_injections(str(parsed.get("body") or ""), parsed, injects or [])
        title = (str(parsed.get("first_heading") or "").strip()
                 or str(parsed.get("first_text") or "").strip() or p.stem)
        # 正文标题已渲染为 Markdown `#`，标题结构与标签提取与 markdown 同源
        tags, cands = extract_tags(p.stem, headings_from_markdown(body), registry=registry)
        tables_rel = [f"parsed/word/{asset_id}.tables.json"] if parsed.get("tables_data") else []
        imgs = parsed.get("images") or []
        n_body = sum(1 for e in imgs if e.get("in_body"))
        extra: dict[str, Any] = {
            "asset": {"asset_id": asset_id, "sha256": sha, "format": "word",
                      "paragraphs": int(parsed.get("paragraphs") or 0),
                      "tables": int(parsed.get("tables") or 0),
                      "images": n_body},
            "quality": {
                "text_source": "text_layer",
                "parsed_with": "python-docx",
                "images": n_body,
                # 包里存在但正文未引用的图（已注册资产，无占位、无 OCR）
                "images_unreferenced": len(imgs) - n_body,
                # 正文引用了但内容不在包里（外链/损坏）→ 只有占位
                "images_unresolved": sum(1 for e in imgs
                                         if e.get("in_body") and not e.get("sha256")),
            },
            "verification": "unverified",  # 与 markdown/excel 统一：先入库，审核台补标
            "structure": {"tables": tables_rel},
            # 未收录强候选（进审核队列 tag_candidate 人工采纳后入词典）
            "tag_candidates": [{"name": c.name, "tier": c.tier} for c in cands],
        }
        if evidence:
            extra["evidence"] = evidence
        return KbDocument(
            source_type="doc_word",
            source_id=sid,
            url="",
            title=title,
            body=body,
            created_at=None,
            component="",
            tags=[t.name for t in tags],
            extra=extra,
        )


_REGISTRY: dict[str, type[BaseSource]] = {
    "github": GithubSource,
    "markdown": MarkdownSource,
    "pdf": PdfSource,
    "image": ImageSource,
    "excel": ExcelSource,
    "word": WordSource,
}


def register_source(source_type: str, cls: type[BaseSource]) -> None:
    """注册新的来源类型（第三方 adapter 接入点）。"""
    _REGISTRY[source_type] = cls


def create_source(cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                  app_cfg: Optional["AppConfig"] = None) -> BaseSource:
    cls = _REGISTRY.get(cfg.type)
    if cls is None:
        raise ValueError(f"未知数据源类型: {cfg.type}（已注册: {sorted(_REGISTRY)}）")
    return cls(cfg, project_root, app_cfg=app_cfg)


def build_sources(app_cfg: "AppConfig", project_root: Path = PROJECT_ROOT) -> list[BaseSource]:
    """按配置构建生效的数据源列表（跳过 enabled=false 与未注册类型，并提示）。"""
    sources: list[BaseSource] = []
    for sc in app_cfg.effective_sources():
        if not sc.enabled:
            print(f"[sources] 来源 {sc.id} ({sc.type}) 已禁用（enabled=false），跳过")
            continue
        try:
            sources.append(create_source(sc, project_root, app_cfg=app_cfg))
        except ValueError as e:
            print(f"[warn] 跳过来源 {sc.id}: {e}")
    if not sources:
        print("[sources] 没有生效的数据源（请在 config.json 配置 sources）")
    return sources
