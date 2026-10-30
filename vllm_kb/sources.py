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

**文件型来源**（pdf / markdown / word / excel 这类"一个文件一篇文档"的来源）另有一层接缝：
- 文档种类的知识（`source_type` / 资产子目录 / source_id 前缀 / 后缀 / 分块策略）统一声明在
  `doc_kinds.py`，**不要在各来源类里再散落一份**；
- 文件发现 / 版本族收敛 / 同名消歧 / 身份构造统一走 `source_files.discover`
  （经 `_discover_for` 装配），**不要在来源类里自己 glob**。
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
from .doc_kinds import BY_SOURCE, DocKind
# 表格渲染全库唯一实现在 html_convert（html 来源主用；pdf/word 别名引用，调用点不变）
from . import html_convert
from .html_convert import table_to_markdown as _table_to_markdown
from .models import KbDocument
from .source_files import DiscoverSpec, Discovery, discover
from .tagging import (
    TagEntry,
    TagRegistry,
    extract_tags,
    headings_from_markdown,
    headings_from_pdf,
    tier_for,
)

if TYPE_CHECKING:  # 仅类型标注用，避免循环导入
    from .config import AppConfig

# Markdown 图片引用解析（形态支持 + 代码感知 + 兜底占位）见 md_images.py；
# 这里只保留资产落盘需要的常量
_BASE64_IMG_RE = re.compile(r"data:image/(png|jpe?g|webp|gif);base64,([A-Za-z0-9+/=]+)", re.I)
_IMG_EXT = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "webp": "webp", "gif": "gif"}

_DECORATIVE_MAX_DIM = 2   # 图片最长边 ≤2px 视为装饰（1x1 埋点 gif/logo），不进正文/资产


def _image_dimensions(data: bytes) -> Optional[tuple[int, int]]:
    """从字节流解析图片像素尺寸（PNG/GIF/JPEG，零依赖）；无法识别返回 None（不过滤）。

    html_convert 已按显式 width/height 属性过滤一层；这里按**文件内容**兜底——
    无属性或 CSS 缩放的 1x1 装饰图也拦得住。webp 不解析（浏览器另存页装饰图
    几乎都是 gif/png），宁可漏放不误杀。
    """
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
            return (int.from_bytes(data[16:20], "big"),
                    int.from_bytes(data[20:24], "big"))
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return (int.from_bytes(data[6:8], "little"),
                    int.from_bytes(data[8:10], "little"))
        if data[:2] == b"\xff\xd8":  # JPEG：扫 SOF 段取宽高
            i = 2
            n = len(data)
            while i + 9 < n:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    return (int.from_bytes(data[i + 7:i + 9], "big"),
                            int.from_bytes(data[i + 5:i + 7], "big"))
                i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    except (IndexError, ValueError):
        pass
    return None


def _is_decorative_image(data: bytes) -> bool:
    dims = _image_dimensions(data)
    return bool(dims) and max(dims) <= _DECORATIVE_MAX_DIM


def _sha256(p: Path) -> str:
    """文件 sha256——带 **size+mtime_ns 快速通道缓存**（判重提速）。

    判重链路每次构建每篇都要全文件哈希：pdf/word/html 的 extract 缓存键（asset_id=
    sha 前缀）必须先算 sha 才能查缓存；md 资产化落盘校验、image 源逐图判重同理——
    大 pdf 场景（几十 MB × 几千篇）是构建主瓶颈。size+mtime_ns 未变 → 直接复用
    上次哈希（git racily-clean 同款思路；"同秒修改同大小"的理论窗口在导入即冻结的
    业务文档场景可忽略）。缓存持久化 data/parsed/.file_sha.json（跨构建复用；
    损坏忽略重建；上限 5 万条，超出裁掉最早写入的 10%）。构建结束 save_sha_cache()。
    """
    global _SHA_CACHE, _SHA_CACHE_LOADED
    try:
        st = p.stat()
    except OSError:
        pass  # 文件不存在等 → 让下面 open 抛自然异常
    else:
        if not _SHA_CACHE_LOADED:
            _SHA_CACHE_LOADED = True
            _SHA_CACHE = _load_sha_cache()
        hit = _SHA_CACHE.get(f"{p}|{st.st_size}|{st.st_mtime_ns}")
        if hit is not None:
            return hit
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    if _SHA_CACHE_LOADED:
        if len(_SHA_CACHE) >= _SHA_CACHE_MAX:
            for k in list(_SHA_CACHE)[: _SHA_CACHE_MAX // 10]:
                _SHA_CACHE.pop(k, None)     # dict 保序 = 最早写入先裁
        _SHA_CACHE[f"{p}|{st.st_size}|{st.st_mtime_ns}"] = sha
    return sha


_SHA_CACHE: dict[str, str] = {}
_SHA_CACHE_LOADED = False
_SHA_CACHE_MAX = 50_000


def _sha_cache_file() -> Path:
    from .config import PROJECT_ROOT

    return PROJECT_ROOT / "data" / "parsed" / ".file_sha.json"


def _load_sha_cache() -> dict[str, str]:
    try:
        data = json.loads(_sha_cache_file().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_sha_cache() -> None:
    """构建结束统一回写文件哈希缓存（幂等；写盘失败只告警不影响构建产物）。"""
    if not _SHA_CACHE:
        return
    try:
        fp = _sha_cache_file()
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(_SHA_CACHE, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        print(f"[sources] 文件哈希缓存写盘失败（不影响构建）：{e}")


def _asset_entry(rel: str, sha: str, stype: str, path: Path) -> tuple:
    """asset_registry 注册条目 `(rel_path, sha256, source_type, size)`；size 取不到则 0。"""
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return (rel, sha, stype, size)


# stem 无意义形态：纯数字 / 纯十六进制（hash、uuid 片段）/ 纯符号——爬虫批量下载件常见
_JUNK_STEM_RE = re.compile(r"^(?:[0-9a-fA-F\-_]+|[0-9]+|\W+)$")
# 泛化页名（page.html 统一文件名布局等）：stem 无信息量，标题回退解析器提取值
_GENERIC_STEM_RE = re.compile(r"^(?:page|index|default|article|view|detail|main|home)$",
                              re.IGNORECASE)
# 审计元数据日期门（x-created → docs.created_at）：ISO 日期前缀才透传，脏值不进时序列
_X_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _stem_title(stem: str, extracted: str) -> str:
    """文档标题策略（2026 定版）：**原始文件名 stem 优先**。

    文件型来源的文件名由工程师/官方下载命名（含产品/版本线/手册名/期号），
    信息量高于解析提取的标题——如华为手册 PDF 封面/元数据只有产品名
    （"Atlas A3 中心推理和训练硬件"），多篇手册互相重名；而文件名
    "… 26.1.x npu-smi 命令参考 02" 天然消歧。source_id 本就用 stem 构造，
    标题与 sid 一致后 UI/图/skill 展示统一。

    stem 为无意义名（纯数字 / 纯十六进制 hash / 纯符号 / 过短）或**泛化页名**
    （page/index/default/…——审计来源统一 page.html 布局，stem 全相同）时回退
    解析器提取的标题（如爬虫件的 html <title>），两者皆空再回退 stem。
    """
    s = (stem or "").strip()
    e = (extracted or "").strip()
    if (s and len(s) >= 4 and not _JUNK_STEM_RE.fullmatch(s)
            and not _GENERIC_STEM_RE.fullmatch(s)):
        return s
    return e or s


def _discover_for(src: "BaseSource", kind: DocKind, import_path: Path) -> Discovery:
    """按**文档种类声明**做文件发现——文件型来源的唯一入口。

    这里只做"声明 → 输入"的装配；发现逻辑本身在 `source_files.discover`
    （与格式/来源无关，不 import 解析库、不知道 source_type 字符串）。
    """
    return discover(DiscoverSpec(
        patterns=kind.patterns,
        import_path=import_path,
        assets_dir=src.resolve(f"data/assets/{kind.asset_sub}"),
        sid_prefix=kind.sid_prefix,
        label=kind.label,
        log_prefix=f"[sources:{src.id}]",
        fallback_note=src.FALLBACK_NOTE,
    ))


def _copy_asset(src: Path, assets_dir: Path, sub: str,
                sha: Optional[str] = None) -> tuple[str, str, bool]:
    """复制资产到 assets/{sub}/（不可变层）。同名同 sha 幂等跳过；同名异 sha 加 sha 前缀。
    返回 (assets 相对路径, sha256, 是否新增复制)。`sha` 已知时传入可省一次整文件读取。"""
    target_dir = assets_dir / sub
    target_dir.mkdir(parents=True, exist_ok=True)
    sha = sha or _sha256(src)
    target = target_dir / src.name
    if target.exists() and _sha256(target) == sha:
        return f"assets/{sub}/{target.name}", sha, False
    if target.exists():
        # 同名但内容不同：加 sha 前缀避免覆盖
        target = target_dir / f"{src.stem}.{sha[:12]}{src.suffix}"
    shutil.copy2(src, target)
    return f"assets/{sub}/{target.name}", sha, True


# 解析中间产物（`parsed/*/<asset_id>.extract.json`）的 schema 位：**每次改提取逻辑都要 +1**。
# 只校验 sha256 是不够的——提取逻辑升级后旧缓存的 sha256 仍然匹配，会**静默沿用旧解析结果**
# （Word 加内嵌图片时踩过：图片永远不生效且不报错）。改这个常量即让旧缓存整体失效并重解析。
_PDF_EXTRACT_SCHEMA = 1
_WORD_EXTRACT_SCHEMA = 2      # 1=纯文本/表格；2=+内嵌图片清单（占位 + 资产）


class BaseSource(ABC):
    type: str = "base"
    # 文件型来源的文档种类声明（见 doc_kinds.py）；非文件型来源为 None
    kind: Optional[DocKind] = None
    # 回退到资产层时附加的说明（各来源特有，如 md 的"图片相对路径已失锚"）
    FALLBACK_NOTE: str = ""

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

    def _ocr_fingerprint(self) -> str:
        """当前 OCR 引擎指纹（无 OCR 配置 → "none"）——**纳入含图文档的解析缓存幂等键**。

        2026-10 缺口：`_OCR_PROMPT_VERSION` 升级只让 ImageSource / md 图片重算——
        **pdf/word/html 的 extract.json 缓存**（sha256+schema 幂等）命中时直接复用旧
        evidence，`_ocr_artifact_for` 不被调用，内嵌图 OCR 永不重跑。把指纹写进缓存
        键后：提示词/模型/模式升级 → 缓存 miss → 重解析并重跑 OCR（canonical 更新 →
        ingest 增量生效）。**升级后首次构建全量重解析一次**（含无图文档，一次性成本）。
        """
        try:
            oc = self._ocr_settings()
        except Exception:
            oc = None
        if oc is None or oc.provider in ("none", ""):
            return "none"
        from .ocr import engine_fingerprint

        return engine_fingerprint(oc.provider, oc.mode, oc.model)

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
                                       api_key=oc.api_key, model=oc.model, mode=oc.mode,
                                       timeout=oc.timeout, max_tokens=oc.max_tokens,
                                       temperature=oc.temperature, prompt=oc.prompt)
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

    # ---------- 正文图片收集（md/html 通用：资产化 + 不透明占位 + OCR 注入） ----------

    def _resolve_images(self, md_path: Path, text: str, *,
                        assets_only: bool = False) -> tuple[str, list[dict], list[tuple]]:
        """扫描正文图片引用：本地/base64 资产化（**不透明占位**替换引用，原引用只进 evidence）；
        URL 引用标记 remote（V1 不下载）；解析失败标记 unresolved。
        返回 (占位化后的 body, evidence 列表, 新增资产 [(rel, sha, "image", size)])。

        形态解析与代码感知在 `md_images.py`：行内（含空格/尖括号/括号/跨行）、引用式
        （含定义行去路径）、HTML `<img>` 均支持；**任何未识别或未闭合的图片语法也一律占位**，
        保证"正文与 canonical 不含服务器路径"对任意输入都成立。
        md 与 html 来源共用（html 的 src 由 html_convert 原样保留后交到这里）。

        assets_only=True（imports 缺失的回退模式）：相对路径基准 assets/<kind> 是**扁平副本**，
        原相对结构已丢失——先按文档同目录找，未命中再按**文件名**到 assets/images 尽力反查；
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
                if _is_decorative_image(data):  # 1x1 埋点/装饰图：不进正文/资产
                    evidence.append({"kind": "decorative", "ocr": None})
                    return ""
                b64_sha = hashlib.sha256(data).hexdigest()
                # 内容寻址命名：同图只落一份（跨文档自动去重），且不含文档文件名 → 同名文档也不撞
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
            # 本地路径（file:// 剥前缀；相对路径以文档所在目录为基准）
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
            try:
                if _is_decorative_image(p.read_bytes()):  # 内容级：1x1 埋点/装饰图
                    evidence.append({"kind": "decorative", "ocr": None})
                    return ""
            except OSError:
                pass  # 读取失败按正常图处理，后续资产化自会报错
            # 资产化目标 = 写入时的绝对路径（**不能**再 resolve(rel)）——rel 形如
            # assets/images/x.png，未设 VLLM_KB_DATA_ROOT 时 resolve() 按 PROJECT_ROOT
            # 拼接会丢掉 data/ 一层（业务实测：写入 data/assets/…、OCR 读 assets/…，
            # 大量"图片不存在"报错且 asset_registry 登记坏路径）；设了 DATA_ROOT 时
            # 恰好剥前缀自洽，测试环境全设了 DATA_ROOT 故此前未暴露。
            assets_root = self.resolve("data/assets")
            rel, sha, _ = _copy_asset(p, assets_root, "images")
            img_path = assets_root / "images" / Path(rel).name
            ev.update({"kind": "local", "asset_id": sha[:16], "sha256": sha})
            _reg(rel, sha, img_path)
            ocr_ev, inject = self._ocr_artifact_for(img_path, sha, rel)
            if ocr_ev is not None:
                ev["ocr"] = ocr_ev
            evidence.append(ev)
            return placeholder + inject

        body, _refs = rewrite_images(text, resolve)
        return body, evidence, registered

    def _materialize_doc_asset(self, p: Path, sha: Optional[str] = None) -> tuple:
        """把**文档原件**复制进资产层并返回 asset_registry 条目。

        `canonicalize()` 也落资产层：`scripts/build_canonical.py` 与 `--skip-pull` 只跑
        canonicalize，不落盘就会断掉 `asset_id → asset_registry.rel_path` 这条链
        （审核台「文档管理」拿不到原件、无法预览/下载，只能看到一个 sha）。这与
        "canonicalize 已把内嵌图片落资产层"是同一套约定，不引入新的副作用——
        `_copy_asset` 幂等，与 `pull()` 重复执行不会多出副本。
        """
        rel, sha_out, _copied = _copy_asset(p, self.resolve("data/assets"),
                                            self.kind.asset_sub, sha)
        return _asset_entry(rel, sha_out, self.kind.source_type, self.resolve(rel))

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
    DEFAULT_SANITIZE_SOURCES = ("excel", "html", "markdown", "word")

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
    - verification=unverified（质量参差；审核工作台「补标」认证后写 doc_verification 覆盖层，
      ingest 每轮合并进 extra/reliability —— 见 review.review_verification）。
    """

    type = "markdown"
    kind = BY_SOURCE["markdown"]
    FALLBACK_NOTE = ("图片相对路径**已失锚**（assets/md 是扁平副本），"
                     "只能按文件名到 assets/images 尽力反查，未命中的一律占位"
                     "（正文仍不含路径）；如需完整图片/OCR 请恢复 imports 目录")

    def __init__(self, cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                 app_cfg: Optional["AppConfig"] = None):
        super().__init__(cfg, project_root, app_cfg=app_cfg)
        self.import_dir = self.resolve(self.cfg.get("path", f"data/imports/{self.id}"))
        self.title_pattern = str(self.cfg.get("title_pattern", r"^#\s+(.+)"))
        self._title_re = re.compile(self.title_pattern)

    def _assets_dir(self) -> Path:
        return self.resolve(f"data/assets/{self.kind.asset_sub}")

    def pull(self, max_issues: Optional[int] = None) -> int:
        """扫描导入目录，把 md 复制到资产层（幂等）。返回新增条数。"""
        if not self.import_dir.exists():
            print(f"[sources:{self.id}] 导入目录不存在: {self.import_dir}")
            return 0
        added = 0
        registered: list[tuple[str, str, str]] = []
        for pat in self.kind.patterns:
            for p in sorted(self.import_dir.rglob(pat)):
                rel, sha, copied = _copy_asset(p, self.resolve("data/assets"),
                                               self.kind.asset_sub)
                registered.append(_asset_entry(rel, sha, self.kind.source_type,
                                               self.resolve(rel)))
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
        doc_assets: list[tuple] = []   # md 原件资产（§1.6：canonicalize 也落资产层）
        img_assets: list[tuple] = []  # 图片资产（末尾统一注册到 asset_registry）
        # 发现 + 回退告警 + 同名消歧告警统一由 source_files.discover 负责
        found = _discover_for(self, self.kind, self.import_dir)
        md_files = found.files
        total = len(md_files)
        start_ts = time.time()
        if total:
            print(f"[sources:{self.id}] 解析 {total} 个 Markdown …", flush=True)
        for f in md_files:
            p, from_imports, stem = f.path, f.from_imports, f.stem
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
            title = _stem_title(stem, title_raw.group(1) if title_raw else "")
            sha = _sha256(p)
            asset_id = sha[:16]
            sid = f.sid(self.kind.sid_prefix, self.import_dir, self._assets_dir())
            doc_assets.append(self._materialize_doc_asset(p, sha))
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
                "verification": self.kind.default_verification,  # 质量参差：先入库，审核台补标
                "structure": {},
                # 未收录强候选（进审核队列 tag_candidate 人工采纳后入词典）
                "tag_candidates": [{"name": c.name, "tier": c.tier} for c in cands],
            }
            if evidence:
                extra["evidence"] = evidence
            docs.append(KbDocument(
                source_type=self.kind.source_type,
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
        # 图片资产注册（审核侧经 asset_id 反查路径/预览）；与 md 原件、图片同批注册
        self._register_asset_mappings(doc_assets + img_assets)
        self.save_sanitize_log(collector)
        return docs


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
    kind = BY_SOURCE["pdf"]

    def __init__(self, cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                 app_cfg: Optional["AppConfig"] = None):
        super().__init__(cfg, project_root, app_cfg=app_cfg)
        self.import_dir = self.resolve(self.cfg.get("path", f"data/imports/{self.id}"))

    # ---------- 布局 ----------

    def _assets_dir(self) -> Path:
        return self.resolve(f"data/assets/{self.kind.asset_sub}")

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
        for pat in self.kind.patterns:
            for p in sorted(self.import_dir.rglob(pat)):
                rel, sha, copied = _copy_asset(p, self.resolve("data/assets"),
                                               self.kind.asset_sub)
                registered.append(_asset_entry(rel, sha, self.kind.source_type,
                                               self.resolve(rel)))
                if copied:
                    added += 1
        self._register_asset_mappings(registered)
        print(f"[sources:{self.id}] 资产层扫描完成（新增 {added} 个 pdf）")
        return added

    # ---------- 解析（可重跑：优先读导入目录） ----------

    def canonicalize(self) -> list[KbDocument]:
        """从导入目录（缺失时回退资产层）的 PDF 解析出 KbDocument（每篇一个，body=Markdown 全文）。

        **导入路径权威**：与 md/word 同构——`path` 下扫到的 PDF 就是文档集，资产层只作为
        回退与诊断（`stray` 告警会点出"资产层有、导入路径没有"的族，因为那些文档会**静默停更**）。

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
        doc_assets: list[tuple] = []   # PDF 原件资产（§1.6：canonicalize 也落资产层）
        found = _discover_for(self, self.kind, self.import_dir)
        if not found.files:
            return docs
        parsed_dir = self._parsed_dir()
        parsed_dir.mkdir(parents=True, exist_ok=True)
        total = len(found.files)
        start_ts = time.time()
        print(f"[sources:{self.id}] 解析 {total} 个 PDF（PyMuPDF 逐页提取，大手册耗时较长）…",
              flush=True)
        for i, f in enumerate(found.files, 1):
            p = f.path
            t0 = time.time()
            sid = f.sid(self.kind.sid_prefix, self.import_dir, self._assets_dir())
            doc_assets.append(self._materialize_doc_asset(p))
            try:
                result = self._parse_pdf(p, parsed_dir, sid=sid, stem=f.stem,
                                         from_imports=f.from_imports)
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
        # 原件资产注册（§1.6：--skip-pull / build_canonical.py 也留下完整预览链）
        self._register_asset_mappings(doc_assets)
        return docs

    def _parse_pdf(self, p: Path, parsed_dir: Path, *, sid: str, stem: str,
                   from_imports: bool):
        """解析单篇 PDF，返回 (KbDocument | None, 是否缓存命中)。

        `sid` / `stem` / `from_imports` 由发现层给出（`stem` 是**族名**，不是可能带
        `.<sha12>` 的文件名）——文档身份必须与"哪一份文件"解耦，否则收敛到历史版本
        副本时 id 会漂移。

        **缓存优先**：耗时大头是 PyMuPDF 逐页提取（大手册 200+ 页约数秒~数十秒）；
        解析中间产物（文字层 body + 表格 + 首行/页数）按 asset_id（sha256 前缀，内容寻址）
        缓存到 `parsed/pdf/<asset_id>.extract.json`——PDF 未变（sha256 一致）时直接复用缓存，
        仅重跑确定性提取（标签/元数据，毫秒级），提取规则升级后**无需清缓存**即可生效；
        删除 `parsed/pdf/` 目录即强制全量重解析。
        """
        sha = _sha256(p)
        asset_id = sha[:16]
        registry = TagRegistry.load(self.app_cfg) if self.app_cfg else TagRegistry()
        ocr_fp = self._ocr_fingerprint()
        cache = parsed_dir / f"{asset_id}.extract.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                # schema 位必须校验：`_extract_pdf` 升级后旧缓存的 sha256 仍然匹配，
                # 不校验就会**静默沿用旧解析结果**（例如新增的表格提取永远不生效）
                # ocr_fp 同理（2026-10）：OCR 引擎/提示词升级后内嵌图必须重跑
                if (data.get("sha256") == sha
                        and data.get("schema") == _PDF_EXTRACT_SCHEMA
                        and data.get("ocr_fp") == ocr_fp):
                    return self._doc_from_extract(p, sha, asset_id, data, registry,
                                                  sid=sid, stem=stem,
                                                  from_imports=from_imports), True
            except (OSError, ValueError):
                pass  # 缓存损坏 → 重新解析
        parsed = self._extract_pdf(p)
        if parsed is None:
            return None, False
        parsed["ocr_fp"] = ocr_fp
        cache.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        self._write_tables(parsed_dir, asset_id, parsed.get("tables") or [])
        return self._doc_from_extract(p, sha, asset_id, parsed, registry,
                                      sid=sid, stem=stem,
                                      from_imports=from_imports), False

    def _extract_pdf(self, p: Path):
        """PyMuPDF 逐页提取（慢，结果可缓存）：文字层 → Markdown 正文 + 结构化表格。

        返回 {"schema", "sha256", "asset_id", "pages", "first_text", "body", "tables"}；
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
                "schema": _PDF_EXTRACT_SCHEMA,
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
                          registry: TagRegistry, *, sid: str, stem: str,
                          from_imports: bool) -> KbDocument:
        """用解析中间产物构造 KbDocument（确定性提取，毫秒级，每次运行重算）。

        缓存命中与首次解析共用本函数——标签/元数据提取始终以最新规则执行，
        解析器升级（pymupdf 版本/提取逻辑）不影响提取结果的一致性。

        `stem` 是发现层给出的**族名**（文档身份来源）；`p.stem` 只在回退模式收敛到
        `.<sha12>` 历史副本时会不同，直接用它会造出"新文档"。
        """
        body = str(parsed.get("body") or "")
        title = _stem_title(stem, str(parsed.get("first_text") or ""))
        tags, cands = extract_tags(stem, headings_from_pdf(body), registry=registry)
        # tables.json 由 _parse_pdf 首次解析时写入；缓存命中时文件已存在，rel 引用直接构造
        tables_rel = [f"parsed/pdf/{asset_id}.tables.json"] if parsed.get("tables") else []
        return KbDocument(
            source_type=self.kind.source_type,
            source_id=sid,
            url="",
            title=title,
            body=body,
            created_at=None,
            component="",
            tags=[t.name for t in tags],
            extra={
                "asset": {"asset_id": asset_id, "sha256": sha,
                          "format": "pdf", "pages": int(parsed.get("pages") or 0)},
                "quality": {"text_source": "text_layer", "parsed_with": "pymupdf",
                            # 回退模式（imports 缺失）→ 资产层扁平副本，无目录树
                            "source_mode": "imports" if from_imports else "assets_fallback"},
                "verification": self.kind.default_verification,  # 官方手册默认专家验证
                "structure": {"tables": tables_rel},
                # 未收录强候选（进审核队列 tag_candidate 人工采纳后入词典）
                "tag_candidates": [{"name": c.name, "tier": c.tier} for c in cands],
            },
        )


# ---------------- Word（.docx）解析辅助 ----------------

# WordprocessingML 主命名空间：直接比 URI 而不是每次调 qn()（这些比较在逐段落热路径上）
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
# "透明"容器：本身不产出文字，但里面的 run 是正文（内容控件/修订插入/智能标记）
_WORD_TRANSPARENT = frozenset({_W + "ins", _W + "smartTag", _W + "sdt", _W + "sdtContent"})
# OLE2/CFB 容器魔数：**旧版 .doc 与加密/受 IRM 保护的 docx 是同一个格式**，靠它区分
_CFB_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
# 绝对路径（盘符 + UNC）：仅用于日志脱敏
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)[^\s'\"]+")

# 标题样式名：英文 "Heading 1" / 中文 "标题 1" / 繁体 "標題 1"；style_id 通常恒为 "Heading1"
_WORD_HEADING_RE = re.compile(r"^(?:heading|标题|標題)\s*([1-6])$", re.IGNORECASE)
# 表驱动兜底：这些样式名里没有"数字"可正则提取，且各语言界面写法不同
_WORD_HEADING_NAMES = {
    "title": 1, "标题": 1, "標題": 1,
    "subtitle": 2, "副标题": 2, "副標題": 2,
}


def _word_hyperlink_url(par, rid: str) -> str:
    """超链接 rId → 目标 URL；**只认 http/https**。

    内部书签（`w:anchor`）没有 URL；`file://` / UNC（`\\\\server\\share`）等目标是**内部路径**，
    正是"正文不含服务器路径"要挡的东西，一律返回空（只留链接文字）。
    """
    if not rid:
        return ""
    try:
        rel = par.part.rels.get(rid)
    except Exception:
        return ""
    if rel is None or not getattr(rel, "is_external", False):
        return ""
    target = str(getattr(rel, "target_ref", "") or "")
    return target if target.startswith(("http://", "https://")) else ""


def _word_inline_text(el, par, *, link: bool = True) -> str:
    """元素内联文字：run 取文字，超链接渲染成 `[text](url)`，透明容器递归。

    `Paragraph.text` 会把 `w:hyperlink` 里的文字并进来但**丢掉链接地址**——业务文档里
    「见 XX 手册」这类链接是重要线索，丢了就只剩一个裸标题。
    只下潜到透明容器，**不进 `w:drawing`/`w:pict`**——文本框文字仍不提取（见类文档边界）。
    """
    from docx.text.run import Run

    out: list[str] = []
    for child in el.iterchildren():
        if not isinstance(child.tag, str):     # 注释/PI
            continue
        tag = child.tag
        if tag == _W + "r":
            out.append(Run(child, par).text)
        elif tag == _W + "hyperlink":
            text = _word_inline_text(child, par, link=False)
            url = _word_hyperlink_url(par, child.get(_R + "id")) if link else ""
            out.append(f"[{text}]({url})" if (link and url and text) else text)
        elif tag in _WORD_TRANSPARENT:
            out.append(_word_inline_text(child, par, link=link))
    return "".join(out)


def _word_paragraph_text(par) -> str:
    """段落文字（含超链接 URL）：`[text](url)` / 内部书签只留文字。"""
    return _word_inline_text(par._p, par)


def _word_outline_level(par) -> int:
    """`w:outlineLvl` → 标题层级（0-5 → 1-6；其余 → 0）。

    **自定义/本地化标题样式的兜底**：样式名可以任意（"我的章节"），但大纲级别存在
    `w:outlineLvl`（0 基）里——Word 的导航窗格与自动目录就是靠它。先看段落直接格式，
    再看样式定义（内置 Heading N 的级别写在样式里，段落元素上没有）。
    """
    holders = (getattr(par, "_p", None),
               getattr(getattr(par, "style", None), "element", None))
    for holder in holders:
        pPr = getattr(holder, "pPr", None) if holder is not None else None
        lvl = getattr(getattr(pPr, "outlineLvl", None), "val", None) if pPr is not None else None
        if lvl is None:
            continue
        try:
            n = int(lvl)
        except (TypeError, ValueError):
            continue
        return n + 1 if 0 <= n <= 5 else 0
    return 0


def _word_heading_level(par) -> int:
    """段落标题层级：Heading 1-6 / "标题 1" → 1-6；Title/Subtitle → 1/2；非标题 → 0。

    三重信号（依次退化）：① 样式名/style_id 里的 `heading|标题|標題` + 数字（最精确）；
    ② 名字表（Title/Subtitle 这类没有数字的）；③ `w:outlineLvl`（覆盖用户自定义的
    标题样式名，如"我的章节"）。
    """
    st = getattr(par, "style", None)
    name = (getattr(st, "name", "") or "").strip()
    sid = (getattr(st, "style_id", "") or "").replace(" ", "").strip()
    for cand in (name, sid):
        m = _WORD_HEADING_RE.match(cand)
        if m:
            return int(m.group(1))
    lvl = _WORD_HEADING_NAMES.get(name.lower())
    if lvl:
        return lvl
    return _word_outline_level(par)


def _redact_paths(text: str) -> str:
    """日志里不出现服务器绝对路径。

    异常文本常常自带绝对路径（`Package not found at 'C:\\...\\加密案例.docx'`），而日志会被
    贴进工单/群里——与"正文不含服务器路径"同一条约束。文件名本身保留（其余日志也这么打）。
    """
    return _PATH_RE.sub("<路径>", text)


def _word_bad_file_reason(p: Path) -> str:
    """打不开的 Word 文件的**可操作**原因（只说"解析失败"用户不知道下一步做什么）。"""
    try:
        with p.open("rb") as f:
            head = f.read(8)
    except OSError:
        return "文件无法读取（可能被占用/权限不足）"
    if head.startswith(_CFB_MAGIC):
        return ("这是 OLE2/CFB 容器——**旧版 .doc 或加密/受保护的 docx**（同一格式，无法区分）；"
                "请在 Word 里另存为未加密的 .docx")
    if not head.startswith(b"PK"):
        return "不是 OOXML 包（.docx 本质是 zip）；请确认文件类型"
    return "OOXML 结构损坏或缺少必要部件（zip 可打开但 Word 部件不全）"


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
    块级内容控件（`w:sdt`，模板/目录常用）会包住整段甚至整表，也要下潜，否则内容整块丢失。
    """
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    def _walk(parent):
        for child in parent.iterchildren():
            if child.tag == qn("w:p"):
                yield Paragraph(child, document)
            elif child.tag == qn("w:tbl"):
                yield Table(child, document)
            elif child.tag == qn("w:sdt"):
                content = child.find(qn("w:sdtContent"))
                if content is not None:
                    yield from _walk(content)

    yield from _walk(document.element.body)


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
         "ocr_api_key": "",                   # 可选，或环境变量 OCR_API_KEY
         "ocr_timeout": 300}                  # HTTP 超时秒（可选，默认 VLLM_KB_OCR_TIMEOUT=60；
                                              # VLM 类 OCR 大图/密集图建议 300）

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
                          as_timeout_seconds,
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
        ocr_timeout = as_timeout_seconds(self.cfg.get("ocr_timeout"))
        from .ocr import as_max_tokens, as_prompt, as_temperature

        max_tokens = as_max_tokens(self.cfg.get("ocr_max_tokens"))
        temperature = as_temperature(self.cfg.get("ocr_temperature"))
        prompt = as_prompt(self.cfg.get("ocr_api_prompt"))
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
        from collections import Counter

        conf_sources: Counter = Counter()   # 置信度来源分布（部署版本判据：p3 产物=logprobs）
        for img in all_imgs:
            if provider == "none":
                break
            sha = shas[img]
            ocr_path = parsed_dir / f"{img.stem}.ocr.json"
            fp = engine_fingerprint(provider, api_mode, api_model)
            cached = load_ocr_artifact(ocr_path, sha, fp)
            if cached is not None:
                skipped += 1
                conf_sources[cached.confidence_source or "?"] += 1
                # 阈值改了但图片/引擎未变：不重跑 OCR，只按新阈值刷新判定字段
                if cached.min_confidence != min_conf:
                    cached.min_confidence = min_conf
                    save_ocr_artifact(ocr_path, cached, image_ref=f"assets/images/{img.name}")
                continue
            result = None
            while True:
                try:
                    result = ocr_image_detail(img, provider, api_base=api_base,
                                              api_key=api_key, model=api_model, mode=api_mode,
                                              timeout=ocr_timeout, max_tokens=max_tokens,
                                              temperature=temperature, prompt=prompt)
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
            conf_sources[art.confidence_source or "?"] += 1
            if art.review_reason:
                review += 1
                print(f"[sources:{self.id}] 图片 {img.name} OCR 需人工复核"
                      f"（reason={art.review_reason} conf={art.confidence}）——不进正文", flush=True)
            processed += 1
        total = len(list(images.iterdir()))
        dist = " ".join(f"{k}={v}" for k, v in sorted(conf_sources.items())) or "无"
        print(f"[sources:{self.id}] OCR 完成：新增 {processed}，跳过（幂等）{skipped}，"
              f"失败 {failed}，待人工复核 {review}（图片 {total} 张；指纹 fp={fp}；"
              f"置信度来源分布 {dist}）——旧产物 model=自报/p2 前实现，logprobs=p3 起的"
              f"新实现；全部是 model/无 logprobs 却应重算时，先确认部署代码含"
              f" _OCR_PROMPT_VERSION=3")
        return []


# Excel 结构页 sheet 名（封面/前言/目录/版权类）——不是知识内容：
# 进正文会让 body 开头变成"版权所有 ©…"（审核页标题观感受污染）、sheet 名混进标签提取。
# 精确词 + 可选尾随数字；业务特有名单用 config `skip_sheets`（正则数组）追加。
_BOILERPLATE_SHEET_RE = re.compile(
    r"^(?:封面|扉页|前言|序言|目录|版权|版权页|声明|声明页|修订记录|修订历史|文档信息|"
    r"cover(?:\s*page)?|front\s*matter|preface|foreword|table\s+of\s+contents|toc|"
    r"contents|copyright|revision(?:\s*history)?|document\s+info(?:rmation)?)\s*\d*\s*$",
    re.IGNORECASE)


class ExcelSource(BaseSource):
    """Excel 表格来源（工程师问题定位记录/已知问题登记表等，**格式未知**）。

    配置示例：
        {"id": "engineer-troubleshooting", "type": "excel",
         "path": "data/imports/engineer/问题定位记录.xlsx", "enabled": true}

    **schema-free 设计（不写死任何列名/sheet 名/行号）**：
    - pull：把配置 path（文件或目录）下的 .xlsx/.xlsm 复制到资产层；
    - canonicalize：**一个文件一篇文档**（`source_id = excel:<文件名>`，与 md/word/pdf 同构）——
      body 为 `## <sheet 名>` + 各非空行；**结构页过滤**（封面/前言/目录/版权类 sheet 名
      不进正文/标签，config `skip_sheets` 可追加业务名单）；
      **表头键值对**（auto 判定）：sheet 非空行 ≥3 且首个非空行 ≥2 列、全为短标签（≤30 字符）
      → 视为表头，数据行变 `列名=值 | 列名=值`（chunk 自包含列语义），否则按列序裸拼；
      分块以"行"为单位（每个非空行是独立段落 → 独立 chunk，带 sheet 作 section），
      所以按文件聚合**不损失检索粒度**；
    - **为什么不再逐行一篇**：逐行的 `source_id` 只能靠位置（`文件:sheet:行`），中间插一行会让
      其后所有行身份平移，人工治理（标签/补标）**串到别的行**上；按文件聚合后身份稳定，
      且标签/验证状态是**整体评估**（登记表整体入库、整体补标，不做逐行判断）；
    - **实体提取复用现有线路**：body 进入 canonical 后，错误码/算子/模型/版本由
      signature 三层提取自动入图（scheme-free：图构建只依赖 canonical）；
    - **脱敏**：cell 值经 sanitize_text（内部 IP → &lt;IP&gt;、内部路径 → &lt;PATH&gt;，
      默认路径/日志路径白名单保留），防止内部数据经检索外泄；
    - 验证状态：登记表默认 `unverified`（低优先级），审核台整体补标。
    """

    type = "excel"
    kind = BY_SOURCE["excel"]

    def __init__(self, cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                 app_cfg: Optional["AppConfig"] = None):
        super().__init__(cfg, project_root, app_cfg=app_cfg)
        self.import_path = self.resolve(self.cfg.get("path", f"data/imports/{self.id}"))

    def _assets_dir(self) -> Path:
        return self.resolve(f"data/assets/{self.kind.asset_sub}")

    def _excel_files(self) -> list[Path]:
        p = self.import_path
        if p.is_dir():
            out = []
            for pat in self.kind.patterns:
                out.extend(sorted(p.rglob(pat)))
            return out
        if p.is_file() and p.suffix in self.kind.suffixes:
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
            rel, sha, copied = _copy_asset(f, self.resolve("data/assets"),
                                           self.kind.asset_sub)
            registered.append(_asset_entry(rel, sha, self.kind.source_type, self.resolve(rel)))
            if copied:
                added += 1
        self._register_asset_mappings(registered)
        print(f"[sources:{self.id}] 资产层扫描完成（新增 {added} 个 excel）")
        return added

    def canonicalize(self) -> list[KbDocument]:
        """**一个文件一篇文档**（与 md/word/pdf 同构）：body = 各 sheet 的 Markdown 段落。

        体例（schema-free，不假设列语义）：

            ## <sheet 名>

            行1单元格1 行1单元格2 …
            行2单元格1 …

        - **为什么按文件聚合**：逐行一篇时 `source_id` 只能靠**位置**（`excel:文件:sheet:行`），
          在中间插一行就会让其后所有行的身份整体平移——人工治理（标签/补标）会**串到别的行**上。
          按文件聚合后身份 = `excel:<文件名>`，与 md/word/pdf 完全一致，插行/改行都不影响；
        - **检索粒度不损失**：分块仍以"行"为单位（每个非空行是独立段落 → 独立 chunk），
          且 `## sheet 名` 让每个 chunk 带 `section`，命中时能看出属于哪张表；
        - **标签/置信度是整体评估**：`tags` 由文件名 + sheet 名走同一套 `extract_tags`
          （与 md/word 同构），`verification` 按文件给一个默认值——登记表整体入库、整体补标，
          不做逐行判断；
        - 每个非空行仍做**后置脱敏**扫描（命中落 sanitize_log.json，不阻断入库）。

        **后置脱敏**：body 以**原文入库**（原文检索）；出口脱敏由 serve_api 返回时统一做。
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
        registry = TagRegistry.load(self.app_cfg) if self.app_cfg else TagRegistry()
        doc_assets: list[tuple] = []   # excel 原件资产（§1.6：canonicalize 也落资产层）
        found = _discover_for(self, self.kind, self.import_path)
        if not found.files:
            return docs
        total = len(found.files)
        start_ts = time.time()
        print(f"[sources:{self.id}] 解析 {total} 个 Excel …", flush=True)
        for f in found.files:
            p, stem = f.path, f.stem
            sha = _sha256(p)
            asset_id = sha[:16]
            doc_assets.append(self._materialize_doc_asset(p, sha))
            try:
                wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
            except Exception as e:
                print(f"[sources:{self.id}] 读取失败 {p.name}: {e}（跳过）", flush=True)
                continue
            parts: list[str] = []
            sheet_names: list[str] = []
            sheets_skipped: list[str] = []
            n_rows = 0
            skip_pats = [re.compile(str(p), re.IGNORECASE)
                         for p in (self.cfg.get("skip_sheets") or [])]
            try:
                for ws in wb.worksheets:
                    sheet = str(ws.title or "").strip() or "Sheet"
                    # 结构页过滤：封面/前言/目录/版权类 sheet 不是知识内容——
                    # 不进正文（否则 body 开头是"版权所有 ©…"）、不进标签提取
                    if _BOILERPLATE_SHEET_RE.match(sheet) or any(
                            pat.match(sheet) for pat in skip_pats):
                        sheets_skipped.append(sheet)
                        continue
                    rows_raw: list[list[str]] = []
                    for row in ws.iter_rows(values_only=True):
                        cells = [str(c).strip() for c in row
                                 if c is not None and str(c).strip()]
                        if cells:
                            rows_raw.append(cells)  # 空行跳过（不依赖行号语义）
                    if not rows_raw:
                        continue
                    sheet_names.append(sheet)
                    n_rows += len(rows_raw)
                    # 表头键值对（auto 判定）：sheet 是"表头+数据行"结构（≥3 行、表头
                    # ≥2 列且全为短标签）→ 数据行变 `列名=值 | 列名=值`，行 chunk 自包含
                    # 列语义（"故障名称=xxx" 可直接搜）；说明页/无表头结构原样裸拼。
                    header = rows_raw[0]
                    use_kv = (len(rows_raw) >= 3 and len(header) >= 2
                              and all(len(c) <= 30 for c in header))
                    rows: list[str] = []
                    if use_kv:
                        rows.append("[表头] " + " ".join(header))
                        for cells in rows_raw[1:]:
                            pairs = [f"{header[i]}={v}" for i, v in enumerate(cells)
                                     if i < len(header)]
                            rows.append(" | ".join(pairs + cells[len(header):]))
                    else:
                        rows.extend(" ".join(cells) for cells in rows_raw)
                    for line in rows:
                        if sanitize_on:
                            ips, paths = collect_sanitize_hits(line, keep_paths, keep_ips)
                            if ips:
                                collector.setdefault("ips", set()).update(ips)
                            if paths:
                                collector.setdefault("paths", set()).update(paths)
                    parts.append(f"## {sheet}\n\n" + "\n".join(rows))
            finally:
                wb.close()
            body = "\n\n".join(parts).strip()
            if not body:
                print(f"[sources:{self.id}] 跳过 {p.name}：所有 sheet 均为空或为结构页",
                      flush=True)
                continue
            # 标签与 md/word 同构：文件名 + sheet 名（两级分类，见 tagging.py）
            tags, cands = extract_tags(stem, sheet_names, registry=registry)
            docs.append(KbDocument(
                source_type=self.kind.source_type,
                source_id=f.sid(self.kind.sid_prefix, self.import_path, self._assets_dir()),
                url="",
                title=stem,
                body=body,
                created_at=None,
                component="",
                tags=[t.name for t in tags],
                extra={
                    "asset": {"asset_id": asset_id, "sha256": sha, "format": "excel"},
                    "quality": {"text_source": "table", "parsed_with": "openpyxl",
                                "sheets": len(sheet_names), "rows": n_rows,
                                "sheets_skipped": sheets_skipped,
                                # 回退模式（imports 缺失）→ 资产层扁平副本，无目录树
                                "source_mode": "imports" if f.from_imports
                                               else "assets_fallback"},
                    "verification": self.kind.default_verification,
                    # 未收录强候选（进审核队列 tag_candidate 人工采纳后入词典）
                    "tag_candidates": [{"name": c.name, "tier": c.tier} for c in cands],
                },
            ))
            print(f"[sources:{self.id}] 解析完成 {p.name}"
                  f"（{len(sheet_names)} sheet / {n_rows} 行）", flush=True)
        print(f"[sources:{self.id}] 解析完成：{len(docs)}/{total} 篇（耗时 "
              f"{time.time() - start_ts:.0f}s）", flush=True)
        # 原件资产注册（§1.6：--skip-pull / build_canonical.py 也留下完整预览链）
        self._register_asset_mappings(doc_assets)
        # 被脱敏命中的 IP/路径落盘维护文件（data/sanitize_log.json，幂等合并；app_cfg 缺失时跳过）
        self.save_sanitize_log(collector)
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
    kind = BY_SOURCE["word"]
    # 本来源**不支持**、但用户很可能放进来的格式：只用于给出指向性提示
    _LEGACY_SUFFIXES = ("*.doc", "*.rtf", "*.wps", "*.odt", "*.pages", "*.pdf")

    def __init__(self, cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                 app_cfg: Optional["AppConfig"] = None):
        super().__init__(cfg, project_root, app_cfg=app_cfg)
        self.import_dir = self.resolve(self.cfg.get("path", f"data/imports/{self.id}"))

    # ---------- 布局 ----------

    def _assets_dir(self) -> Path:
        return self.resolve(f"data/assets/{self.kind.asset_sub}")

    def _parsed_dir(self) -> Path:
        return self.resolve("data/parsed/word")

    def _word_files(self) -> list[Path]:
        """导入路径下的 Word 文件（path 可为文件或目录）。"""
        p = self.import_dir
        if p.is_dir():
            out: list[Path] = []
            for pat in self.kind.patterns:
                out.extend(sorted(p.rglob(pat)))
            return out
        if p.is_file() and p.suffix.lower() in self.kind.suffixes:
            return [p]
        return []

    def _legacy_hint(self) -> str:
        """目录里只有本来源不支持的格式时给出**指向性**提示。

        否则只报"无 Word 文件"，用户不知道该改什么（最常见就是丢了一堆 .doc 进来）。
        """
        if not self.import_dir.is_dir():
            return ""
        found: list[str] = []
        for pat in self._LEGACY_SUFFIXES:
            found.extend(sorted(p.name for p in self.import_dir.rglob(pat)))
        if not found:
            return ""
        shown = ", ".join(found[:5]) + (" …" if len(found) > 5 else "")
        return (f"；另检测到 {len(found)} 个**非 OOXML** 文件（{shown}）——本来源只认 "
                f".docx/.docm，旧版 .doc / RTF / WPS / ODT 请先在 Word 里另存为 .docx"
                f"（.pdf 请改用 pdf 来源）")

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
            rel, sha, copied = _copy_asset(f, self.resolve("data/assets"),
                                           self.kind.asset_sub)
            registered.append(_asset_entry(rel, sha, self.kind.source_type, self.resolve(rel)))
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
        # 发现 + 回退告警 + 同名消歧告警统一由 source_files.discover 负责
        found = _discover_for(self, self.kind, self.import_dir)
        files = found.files
        if not files:
            # 提示只在 canonicalize 打一次（pull 也会打"无 Word 文件"，两处都带上就重复了）
            print(f"[sources:{self.id}] 导入路径无 Word 文件: {self.import_dir}"
                  f"{self._legacy_hint()}", flush=True)
            return docs
        parsed_dir = self._parsed_dir()
        parsed_dir.mkdir(parents=True, exist_ok=True)
        total = len(files)
        start_ts = time.time()
        img_assets: list[tuple] = []   # 内嵌图片资产（末尾与 word 原件同批注册）
        doc_assets: list[tuple] = []   # word 原件资产（§1.6：canonicalize 也落资产层）
        print(f"[sources:{self.id}] 解析 {total} 个 Word …", flush=True)
        for i, f in enumerate(files, 1):
            p, from_imports, stem = f.path, f.from_imports, f.stem
            t0 = time.time()
            try:
                parsed, cached = self._parse_docx(p, parsed_dir)
            except Exception as e:
                print(f"[sources:{self.id}] [{i}/{total}] 解析失败 {p.name}: {e}（跳过）", flush=True)
                continue
            if parsed is None:
                continue
            sha = _sha256(p)
            doc_assets.append(self._materialize_doc_asset(p, sha))
            evidence, injects, registered = self._materialize_images(p, parsed)
            img_assets.extend(registered)
            doc = self._doc_from_extract(
                p, sha, sha[:16], parsed, registry,
                sid=f.sid(self.kind.sid_prefix, self.import_dir, self._assets_dir()),
                stem=stem, evidence=evidence, injects=injects)
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
        # 内嵌图片与 word 原件资产注册（审核侧经 asset_id 反查路径/预览）
        self._register_asset_mappings(doc_assets + img_assets)
        self.save_sanitize_log(collector)
        return docs

    def _parse_docx(self, p: Path, parsed_dir: Path):
        """解析单篇 Word，返回 (parsed | None, 是否缓存命中)。

        **缓存优先**：`parsed/word/<asset_id>.extract.json`（asset_id = sha256 前缀，内容寻址）——
        文件未变时直接复用提取结果，仅重跑确定性提取（标签/元数据，毫秒级）；
        删除 `parsed/word/` 目录即强制全量重解析。
        """
        sha = _sha256(p)
        asset_id = sha[:16]
        ocr_fp = self._ocr_fingerprint()
        cache = parsed_dir / f"{asset_id}.extract.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                # schema 位必须校验：`_extract_docx` 升级后旧缓存的 sha256 仍然匹配，
                # 不校验就会**静默无图**（比报错难查得多）
                # ocr_fp 同理（2026-10）：OCR 引擎/提示词升级后内嵌图必须重跑
                if (data.get("sha256") == sha
                        and data.get("schema") == _WORD_EXTRACT_SCHEMA
                        and data.get("ocr_fp") == ocr_fp):
                    return data, True
            except (OSError, ValueError):
                pass  # 缓存损坏 → 重新解析
        parsed = self._extract_docx(p)
        if parsed is None:
            return None, False
        parsed["ocr_fp"] = ocr_fp
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
            print(f"[sources:{self.id}] 跳过无法解析的 Word 文件 {p.name}："
                  f"{_word_bad_file_reason(p)}"
                  f"（{type(e).__name__}: {_redact_paths(str(e))}）")
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
                rows = [["\n".join(_word_paragraph_text(par) for par in c.paragraphs).strip()
                         for c in row.cells] for row in block.rows]
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
            text = _word_paragraph_text(block).strip()
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
                          registry: TagRegistry, *, sid: str, stem: str,
                          evidence: Optional[list[dict]] = None,
                          injects: Optional[list[str]] = None) -> KbDocument:
        """用解析中间产物构造 KbDocument（确定性提取，毫秒级，每次运行重算）。

        缓存命中与首次解析共用本函数——标签/元数据/OCR 注入始终以最新规则执行，
        解析器与 OCR 阈值升级都不受缓存影响。

        `stem` 是发现层给出的**族名**（文档身份来源）；回退模式收敛副本的 p.stem
        会带 .sha12 后缀，标题/标签一律用族名（与 PdfSource 一致）。
        """
        body = self._apply_ocr_injections(str(parsed.get("body") or ""), parsed, injects or [])
        title = _stem_title(stem, str(parsed.get("first_heading") or "").strip()
                            or str(parsed.get("first_text") or "").strip())
        # 正文标题已渲染为 Markdown `#`，标题结构与标签提取与 markdown 同源
        tags, cands = extract_tags(stem, headings_from_markdown(body), registry=registry)
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
            "verification": self.kind.default_verification,  # 与 markdown/excel 统一：先入库，审核台补标
            "structure": {"tables": tables_rel},
            # 未收录强候选（进审核队列 tag_candidate 人工采纳后入词典）
            "tag_candidates": [{"name": c.name, "tier": c.tier} for c in cands],
        }
        if evidence:
            extra["evidence"] = evidence
        return KbDocument(
            source_type=self.kind.source_type,
            source_id=sid,
            url="",
            title=title,
            body=body,
            created_at=None,
            component="",
            tags=[t.name for t in tags],
            extra=extra,
        )


class HtmlSource(BaseSource):
    """网页案例来源（浏览器"另存为"的 .html/.htm 单文件；V1 不联网抓取）。

    配置示例：
        {"id": "webpages", "type": "html", "path": "data/imports/html", "enabled": true}

    - pull()：扫描配置 path 下 *.html/*.htm，复制到 data/assets/html/（不可变层）；
    - canonicalize()：html_convert 混合引擎（表格收割 → trafilatura 正文）→ Markdown：
      * 表格**抽取前收割**：内容表 → 结构化 rows 写 data/parsed/html/{asset_id}.tables.json
        （建图抽错误码消费；graph 按 doc_kinds.asset_sub 寻径），正文以 pipe table 补回
        （渲染单一实现 html_convert.table_to_markdown，pdf/word 同源）；
      * 图片 src 原样保留 → `_resolve_images` 统一资产化/占位/OCR（本地相对路径以 html
        所在目录为基准、data:URI 落资产、远程只留占位——与 markdown 来源同一通路）；
      * 浏览器 saved-from-url 注释 → KbDocument.url（phase2 设计"保留 URL"）；
      * 页面骨架（nav/aside/footer）先于引擎修剪；trafilatura 失败 → baseline 兜底。
    - verification=unverified（与 md/word 统一路径：人工文档先入库，审核台补标）；
    - 脱敏默认启用（与 md/word/excel 同组：内网 wiki 保存页含内部信息）；
    - 解析中间产物缓存 parsed/html/{asset_id}.extract.json（schema 位，与 pdf/word 同构：
      缓存只有收割补表后的正文/表格元数据，图片占位解析与标签每轮重算）。

    **V1 边界**：只认本地保存的 .html/.htm（远程图片不下载只留占位）；.mht/.mhtml/.url
    明确跳过并提示另存；trafilatura 正文不含 markdown 标题 → chunk 无 section（已知代价，
    chunking 有段落级兜底切块，见 html_convert 模块说明）。
    """

    type = "html"
    kind = BY_SOURCE["html"]
    # 本来源**不支持**、但用户很可能放进来的格式：只用于给出指向性提示
    _LEGACY_SUFFIXES = ("*.mht", "*.mhtml", "*.url", "*.webarchive")
    # schema=2：html_convert 升级为"原位占位补回"（图文/表文位置=原文位置）+ x-* meta 收割
    # schema=3：修复文末兜底条件写反（缺席 token 的图/表文末补回）+ x-categories 多分隔符
    _EXTRACT_SCHEMA = 3

    def __init__(self, cfg: SourceCfg, project_root: Path = PROJECT_ROOT,
                 app_cfg: Optional["AppConfig"] = None):
        super().__init__(cfg, project_root, app_cfg=app_cfg)
        self.import_dir = self.resolve(self.cfg.get("path", f"data/imports/{self.id}"))

    def _assets_dir(self) -> Path:
        return self.resolve(f"data/assets/{self.kind.asset_sub}")

    def _parsed_dir(self) -> Path:
        return self.resolve("data/parsed/html")

    def _html_files(self) -> list[Path]:
        """导入路径下的网页文件（path 可为文件或目录）。"""
        p = self.import_dir
        if p.is_dir():
            out: list[Path] = []
            for pat in self.kind.patterns:
                out.extend(sorted(p.rglob(pat)))
            return out
        if p.is_file() and p.suffix.lower() in self.kind.suffixes:
            return [p]
        return []

    def _legacy_hint(self) -> str:
        """目录里只有本来源不支持的格式时给出**指向性**提示（与 WordSource 同构）。"""
        if not self.import_dir.is_dir():
            return ""
        found: list[str] = []
        for pat in self._LEGACY_SUFFIXES:
            found.extend(sorted(p.name for p in self.import_dir.rglob(pat)))
        if not found:
            return ""
        shown = ", ".join(found[:5]) + (" …" if len(found) > 5 else "")
        return (f"；另检测到 {len(found)} 个**单文件归档/快捷方式**（{shown}）——本来源只认"
                f"另存为的 .html/.htm 网页文件（.mht 请在浏览器里另存为「网页，仅 HTML」）")

    def pull(self, max_issues: Optional[int] = None) -> int:
        """扫描导入路径，把网页复制到资产层（幂等）。返回新增条数。"""
        files = self._html_files()
        if not files:
            print(f"[sources:{self.id}] 导入路径无网页文件: {self.import_dir}"
                  f"{self._legacy_hint()}")
            return 0
        added = 0
        registered: list[tuple] = []
        for f in files:
            rel, sha, copied = _copy_asset(f, self.resolve("data/assets"),
                                           self.kind.asset_sub)
            registered.append(_asset_entry(rel, sha, self.kind.source_type,
                                           self.resolve(rel)))
            if copied:
                added += 1
        self._register_asset_mappings(registered)
        print(f"[sources:{self.id}] 资产层扫描完成（新增 {added} 个 html）")
        return added

    def canonicalize(self) -> list[KbDocument]:
        """从原始网页再生 KbDocument（每篇一个，body=Markdown，正文不含任何服务器路径）。

        - **优先读 imports 源文件**：图片相对路径以 html 所在目录为基准解析
          （复制到 assets 后相对路径失锚）；imports 被清空时回退 assets 副本；
        - 缺依赖预检：lxml 未安装 → 整来源跳过（与 pdf/word 缺依赖同一姿态）；
          trafilatura 未安装 → 醒目告警后用 lxml 全文兜底（表格收割不受影响）；
        - **后置脱敏**：body/title 以**原文入库**（原文检索）；仅按 config.sanitize 扫描
          会被脱敏的 IP/路径落盘 sanitize_log.json（出口脱敏由 serve_api 统一做）；
        - evidence 记录图片清单（供审核台与图文互证消费）。
        """
        from .sanitize import collect_sanitize_hits

        try:
            import lxml  # noqa: F401
        except ImportError as e:
            print(f"[sources:{self.id}] 未安装 lxml：pip install trafilatura（{e}）"
                  f"——html 来源需要它解析页面，本轮跳过全部网页")
            return []
        try:
            import trafilatura  # noqa: F401
        except ImportError as e:
            print(f"[sources:{self.id}] 未安装 trafilatura：pip install trafilatura（{e}）"
                  f"——将用 lxml 全文兜底（无去噪；表格收割不受影响）")

        docs: list[KbDocument] = []
        registry = TagRegistry.load(self.app_cfg) if self.app_cfg else TagRegistry()
        sanitize_on, keep_paths, keep_ips = self.sanitize_params()
        collector: dict = {}           # 会被脱敏的原始 IP/路径（落盘维护，不进库）
        doc_assets: list[tuple] = []   # html 原件资产（canonicalize 也落资产层）
        img_assets: list[tuple] = []   # 图片资产（末尾统一注册到 asset_registry）
        # 发现 + 回退告警 + 同名消歧告警统一由 source_files.discover 负责
        found = _discover_for(self, self.kind, self.import_dir)
        pages = found.files
        total = len(pages)
        start_ts = time.time()
        if total:
            print(f"[sources:{self.id}] 解析 {total} 个网页 …"
                  "（图片多时含逐图 OCR，进度每 10 页打印一次）", flush=True)
        for idx, f in enumerate(pages, 1):
            p, from_imports, stem = f.path, f.from_imports, f.stem
            try:
                raw = p.read_bytes()
            except OSError as e:
                print(f"[sources:{self.id}] 跳过 {p.name}: {e}")
                continue
            parsed, cached = self._parse_html(p, raw)
            if parsed is None:
                continue
            sha, asset_id = parsed["sha256"], parsed["asset_id"]
            body, evidence, registered = self._resolve_images(
                p, parsed["body"], assets_only=not from_imports)
            img_assets.extend(registered)
            if sanitize_on:
                ips, paths = collect_sanitize_hits(body, keep_paths, keep_ips)
                if ips:
                    collector.setdefault("ips", set()).update(ips)
                if paths:
                    collector.setdefault("paths", set()).update(paths)
            sid = f.sid(self.kind.sid_prefix, self.import_dir, self._assets_dir())
            doc_assets.append(self._materialize_doc_asset(p, sha))
            quality: dict[str, Any] = {
                "text_source": "text_layer",
                "parsed_with": f"html_convert({parsed['extractor']})",
                "charset": parsed.get("charset", ""),
                "source_mode": "imports" if from_imports else "assets_fallback",
                "images_unresolved": sum(1 for e in evidence
                                         if e.get("kind") == "unresolved"),
                "images_decorative_skipped": sum(1 for e in evidence
                                                 if e.get("kind") == "decorative"),
                "layout_tables_dropped": int(parsed.get("layout_tables_dropped") or 0),
                "tables_unwrapped": int(parsed.get("tables_unwrapped") or 0),
                "cached": cached,
            }
            if parsed.get("warnings"):
                quality["convert_warnings"] = parsed["warnings"]
            # 进度日志（每 10 页 + 末页）：html 图片多时逐图 OCR 串行耗时，静默窗口
            # 不可接受（2026-10 业务实测：159 网页卡在"解析…"无日志，实为逐图 OCR 在跑）
            if idx % 10 == 0 or idx == total:
                n_assets = sum(1 for e in evidence
                               if e.get("kind") in ("local", "base64"))
                print(f"[sources:{self.id}] 进度 {idx}/{total} 网页"
                      f"（本页 {len(evidence)} 图 / 资产化 {n_assets} / "
                      f"未解析 {quality['images_unresolved']}，"
                      f"累计 {time.time() - start_ts:.0f}s）", flush=True)
            # 文档级自动标签：文件名 + Markdown 标题（两级分类，见 tagging.py）
            tags, cands = extract_tags(stem, headings_from_markdown(parsed["body"]),
                                       registry=registry)
            # x-categories（审计侧人工分类）消费：词典命中 → 自动标签；未收录 → 强候选
            # （tag_candidate 队列，采纳后全局进词典）——与 extract_tags 同一产出契约，
            # 复用候选→采纳→词典全链。page.html 布局下文件名/标题信号薄，
            # 审计分类是打标的主要信号源。
            meta = parsed.get("meta") or {}
            _seen = {t.name for t in tags} | {c.name for c in cands}
            # 分隔符：实测页面用 | 分隔，兼容中英文逗号/顿号/分号
            for cat in re.split(r"[,，|;；、]", meta.get("x-categories") or ""):
                cat = cat.strip()
                if not cat or cat in _seen:
                    continue
                if registry.contains(cat):
                    tags.append(TagEntry(name=cat, tier=registry.tier(cat)))
                else:
                    cands.append(TagEntry(name=cat, tier=tier_for(cat)))
                _seen.add(cat)
            extra: dict[str, Any] = {
                "asset": {"asset_id": asset_id, "sha256": sha, "format": "html"},
                "quality": quality,
                "verification": self.kind.default_verification,  # 先入库，审核台补标
                "structure": {"tables": parsed.get("tables_rel") or []},
                # 未收录强候选（进审核队列 tag_candidate 人工采纳后入词典）
                "tag_candidates": [{"name": c.name, "tier": c.tier} for c in cands],
            }
            if evidence:
                extra["evidence"] = evidence
            if parsed.get("meta"):
                extra["meta"] = parsed["meta"]  # x-* 审计元数据（url/created/status/author…）
            docs.append(KbDocument(
                source_type=self.kind.source_type,
                source_id=sid,
                url=str(parsed.get("source_url") or ""),
                title=_stem_title(stem, str(parsed.get("title") or "")),
                body=body,
                created_at=(meta["x-created"]
                            if _X_DATE_RE.match(meta.get("x-created") or "") else None),
                component="",
                tags=[t.name for t in tags],
                extra=extra,
            ))
        if total:
            print(f"[sources:{self.id}] 解析完成：{len(docs)}/{total} 篇（耗时 "
                  f"{time.time() - start_ts:.0f}s）", flush=True)
        # 图片资产注册（审核侧经 asset_id 反查路径/预览）；与 html 原件、图片同批注册
        self._register_asset_mappings(doc_assets + img_assets)
        self.save_sanitize_log(collector)
        return docs

    def _parse_html(self, p: Path, raw: bytes) -> tuple[Optional[dict], bool]:
        """解析单页（可缓存）：html_convert 混合引擎 + tables.json 落盘。

        返回 (parsed dict, 是否缓存命中)；无正文返回 (None, False)。
        缓存里是**收割补表后的正文**与表格元数据（图片占位解析、标签每轮重算，不进缓存）；
        schema 位必须校验：转换器升级后旧缓存的 sha256 仍然匹配，不校验就会静默沿用旧解析。
        """
        sha = _sha256(p)
        asset_id = sha[:16]
        ocr_fp = self._ocr_fingerprint()
        cache = self._parsed_dir() / f"{asset_id}.extract.json"
        if cache.is_file():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if (data.get("sha256") == sha
                        and data.get("schema") == self._EXTRACT_SCHEMA
                        and data.get("ocr_fp") == ocr_fp):
                    data["tables_rel"] = ([f"parsed/html/{asset_id}.tables.json"]
                                          if data.get("tables") else [])
                    return data, True
            except (OSError, ValueError):
                pass  # 缓存损坏 → 重新解析
        res = html_convert.html_to_markdown(raw, fallback_title=p.stem)
        if not res.body.strip():
            print(f"[sources:{self.id}] 跳过无正文网页: {p.name}")
            return None, False
        parsed: dict[str, Any] = {
            "schema": self._EXTRACT_SCHEMA,
            "sha256": sha,
            "asset_id": asset_id,
            "ocr_fp": ocr_fp,          # OCR 引擎指纹进缓存键（升级后内嵌图重跑）
            "body": res.body,
            "title": res.title,
            "tables": res.tables,
            "charset": res.charset,
            "source_url": res.source_url,
            "extractor": res.extractor,
            "warnings": res.warnings,
            "meta": dict(res.meta or {}),
            "layout_tables_dropped": res.layout_tables_dropped,
            "tables_unwrapped": res.tables_unwrapped,
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        parsed["tables_rel"] = self._write_tables(asset_id, res.tables)
        return parsed, False

    def _write_tables(self, asset_id: str, tables: list[dict]) -> list[str]:
        """结构化表格落盘（可重跑产物，以 asset_id 命名——不暴露文件名/路径）。"""
        if not tables:
            return []
        tpath = self._parsed_dir() / f"{asset_id}.tables.json"
        tpath.write_text(
            json.dumps({"source": f"html:{asset_id}", "tables": tables},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        return [f"parsed/html/{tpath.name}"]


_REGISTRY: dict[str, type[BaseSource]] = {
    "github": GithubSource,
    "markdown": MarkdownSource,
    "pdf": PdfSource,
    "image": ImageSource,
    "excel": ExcelSource,
    "word": WordSource,
    "html": HtmlSource,
}


def register_source(source_type: str, cls: type[BaseSource]) -> None:
    """注册新的来源类型（第三方 adapter 接入点）。

    **文件型**来源还应：① 在 `doc_kinds.KINDS` 里声明种类；② 设 `cls.kind` 指向它。
    两处不一致会被 `tests/test_doc_kinds.py` 的护栏测试拦住。
    """
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
