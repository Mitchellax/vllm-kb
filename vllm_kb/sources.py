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

# Markdown 图片引用：![alt](url "title")——url 取到空白/右括号前
_IMG_REF_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)")
_BASE64_IMG_RE = re.compile(r"data:image/(png|jpe?g|webp|gif);base64,([A-Za-z0-9+/=]+)", re.I)
_IMG_EXT = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "webp": "webp", "gif": "gif"}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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

    def _register_asset_mappings(self, items: list[tuple[str, str, str]]) -> None:
        """注册资产到审核侧 asset_registry（管理员路径映射；不进 canonical/检索库）。

        items: [(assets相对路径, sha256, source_type)]。app_cfg 缺失（纯解析测试）时跳过。
        幂等 upsert；失败仅提示，不影响入库。
        """
        if self.app_cfg is None or not items:
            return
        try:
            from .review import register_asset

            db = self.app_cfg.resolve(self.app_cfg.storage.review_path)
            for rel, sha, stype in items:
                register_asset(db, sha[:16], rel, sha256=sha, source_type=stype)
        except Exception as e:
            print(f"[sources:{self.id}] asset_registry 注册失败（不影响入库）: {e}")

    # ---------------- 内部数据脱敏（config.sanitize 控制启用范围） ----------------

    # 默认启用脱敏的业务来源（github 公开数据不脱敏；PDF 手册默认不启用，config 加 "pdf" 即开）
    DEFAULT_SANITIZE_SOURCES = ("excel", "markdown")

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
             missing: bool = False, numbers: list[int] | None = None) -> int:
        """拉取 GitHub 数据（incremental=True 时 done 后仍时间窗增量；missing=True 补差
        只拉缺失；numbers 走 REST 单条补拉，见 GithubPuller.pull）。"""
        if max_issues is not None:
            self.puller.max_issues = max_issues
        return self.puller.pull(incremental=incremental, missing=missing, numbers=numbers)

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
            registered.append((rel, sha, "doc_markdown"))
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
        md_files: list[tuple[Path, bool]] = []
        if self.import_dir.exists():
            md_files = [(p, True) for p in sorted(self.import_dir.rglob("*.md"))
                        + sorted(self.import_dir.rglob("*.markdown"))]
        if not md_files:
            assets = self._assets_dir()
            if assets.exists():
                md_files = [(p, False) for p in sorted(assets.glob("*.md"))
                            + sorted(assets.glob("*.markdown"))]
        total = len(md_files)
        start_ts = time.time()
        if total:
            print(f"[sources:{self.id}] 解析 {total} 个 Markdown …", flush=True)
        for p, from_imports in md_files:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                print(f"[sources:{self.id}] 跳过 {p.name}: {e}")
                continue
            body, evidence = self._resolve_images(p, text) if from_imports else (text, [])
            if sanitize_on:
                ips, paths = collect_sanitize_hits(body, keep_paths, keep_ips)
                if ips:
                    collector.setdefault("ips", set()).update(ips)
                if paths:
                    collector.setdefault("paths", set()).update(paths)
            title_raw = self._title_re.search(text)
            title = title_raw.group(1).strip() if title_raw else p.stem
            sha = _sha256(p)
            asset_id = sha[:16]
            # 文档级自动标签：文件名 + Markdown 标题（两级分类，见 tagging.py）
            tags, cands = extract_tags(p.stem, headings_from_markdown(text), registry=registry)
            extra: dict[str, Any] = {
                "asset": {"asset_id": asset_id, "sha256": sha, "format": "markdown"},
                "quality": {"text_source": "text_layer", "parsed_with": "raw"},
                "verification": "unverified",  # 质量参差：先入库，审核工作台补标
                "structure": {},
                # 未收录强候选（进审核队列 tag_candidate 人工采纳后入词典）
                "tag_candidates": [{"name": c.name, "tier": c.tier} for c in cands],
            }
            if evidence:
                extra["evidence"] = evidence
            docs.append(KbDocument(
                source_type="doc_markdown",
                source_id=f"md:{p.stem}",
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
        self.save_sanitize_log(collector)
        return docs

    # ---------- Markdown 图片收集（确保图片与 md 一起入库） ----------

    def _resolve_images(self, md_path: Path, text: str) -> tuple[str, list[dict]]:
        """扫描正文图片引用：本地/base64 资产化（**不透明占位**替换引用，原引用只进 evidence）；
        URL 引用标记 remote（V1 不下载）；解析失败标记 unresolved。
        返回 (占位化后的 body, evidence 列表)。

        安全约束：正文与 canonical **不含任何服务器路径**——evidence 只记 asset_id/sha256
        （管理员侧经 asset_registry 映射回文件），unresolved 不保留原文引用（可能是路径形态）。
        """
        evidence: list[dict] = []
        counter: dict[str, int] = {}

        def repl(m: re.Match) -> str:
            alt, ref = m.group(1), m.group(2)
            placeholder = f"[图片:{alt}]" if alt.strip() else "[图片]"
            ev: dict = {"kind": "unresolved", "ocr": None}
            if ref.startswith(("http://", "https://")):
                ev = {"kind": "remote", "source_ref": ref, "ocr": None}
                evidence.append(ev)
                return placeholder  # URL 引用不下载，正文占位
            if ref.startswith("data:"):
                bm = _BASE64_IMG_RE.match(ref)
                if not bm:
                    evidence.append(ev)
                    return placeholder
                ext = _IMG_EXT.get(bm.group(1).lower(), "png")
                try:
                    data = base64.b64decode(bm.group(2))
                except Exception:
                    evidence.append(ev)
                    return placeholder
                counter[ext] = counter.get(ext, 0) + 1
                name = f"{md_path.stem}_img{counter[ext]}.{ext}"
                target = self.resolve("data/assets/images") / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                sha = _sha256(target)
                ev.update({"kind": "base64", "asset_id": sha[:16], "sha256": sha})
                evidence.append(ev)
                return placeholder
            # 本地路径（file:// 剥前缀；相对路径以 md 目录为基准）
            local = ref[len("file://"):] if ref.startswith("file://") else ref
            p = Path(local)
            if not p.is_absolute():
                p = md_path.parent / p
            p = p.resolve()
            if not p.exists():
                evidence.append(ev)  # unresolved（不记 source_ref，避免路径形态进库）
                return placeholder
            asset_path, sha, _ = _copy_asset(p, self.resolve("data/assets"), "images")
            ev.update({"kind": "local", "asset_id": sha[:16], "sha256": sha})
            evidence.append(ev)
            return placeholder

        body = _IMG_REF_RE.sub(repl, text)
        return body, evidence


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
            registered.append((rel, sha, "doc_pdf"))
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
        import json
        import os

        from .ocr import OcrApiError, OcrUnavailable, ocr_image

        images = self._images_dir()
        if not images.exists():
            return []
        parsed_dir = self._parsed_dir()
        parsed_dir.mkdir(parents=True, exist_ok=True)

        # ---- OCR 引擎决策 ----
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

        processed, skipped, failed = 0, 0, 0
        asked = False  # API 失败后的本地询问只问一次
        for img in sorted(p for g in self._IMG_GLOBS for p in images.glob(g)):
            if provider == "none":
                break
            sha = _sha256(img)
            ocr_path = parsed_dir / f"{img.stem}.ocr.json"
            if ocr_path.exists():
                try:
                    meta = json.loads(ocr_path.read_text(encoding="utf-8"))
                    if meta.get("sha256") == sha:
                        skipped += 1
                        continue
                except Exception:
                    pass
            while True:
                try:
                    text, conf = ocr_image(img, provider, api_base=api_base, api_key=api_key,
                                           model=api_model, mode=api_mode)
                    break
                except OcrApiError as e:
                    print(f"[sources:{self.id}] OCR API 失败: {e}")
                    if not asked:
                        asked = True
                        if self._ask_local_ocr():
                            provider = "paddle"
                            continue  # 换本地重试当前图
                    provider = "none"
                    break
                except OcrUnavailable as e:
                    print(f"[sources:{self.id}] OCR 不可用（{e}）——跳过 OCR")
                    provider = "none"
                    break
            if provider == "none":
                print(f"[sources:{self.id}] 跳过 OCR（图片 {len(list(images.iterdir()))} 张，"
                      f"已处理 {processed}）。可配置 ocr_provider: paddle 或 api + ocr_api_base")
                break
            # 签名导向：只提取可判错的签名（算子/错误码/模型/版本）
            from .signature import extract_signatures

            sigs = extract_signatures(text)
            matched = [
                {"text": s.text, "kind": s.kind}
                for s in sigs if s.kind in ("kernel", "op", "errcode", "model", "version")
            ]
            ocr_path.write_text(
                json.dumps({
                    "image": f"assets/images/{img.name}", "sha256": sha,
                    "provider": provider, "text": text,
                    "confidence": round(conf, 4),
                    "signatures": matched,
                }, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            processed += 1
        print(f"[sources:{self.id}] OCR 完成：新增 {processed}，跳过（幂等）{skipped}（图片 {len(list(images.iterdir()))} 张）")
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
            registered.append((rel, sha, "doc_excel"))
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


_REGISTRY: dict[str, type[BaseSource]] = {
    "github": GithubSource,
    "markdown": MarkdownSource,
    "pdf": PdfSource,
    "image": ImageSource,
    "excel": ExcelSource,
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
