"""OCR 引擎抽象：签名导向 OCR（图片 → 文本 → 错误签名提取）。

provider 可插拔（config source 的 ocr_provider 字段）：
- "api":    HTTP OCR 服务（内网外接，业务环境默认形态），两种调用模式（ocr_api_mode）：
  * "custom"（默认）: 自研协议 POST {ocr_api_base}/ocr
        body: {"image": "<base64>", "filename": "x.png", "model": "<可选>"}
        resp: {"text": "...", "confidence": 0.93}
        置信度来源 = 服务端（confidence_source="service"）。
  * "openai": OpenAI 兼容接口（vLLM 部署的 DeepSeek-OCR 等）：
        POST {ocr_api_base}/chat/completions，model=ocr_api_model（必填），
        messages 内联 data URI 图片；响应取 choices[0].message.content。
        置信度来源 = **模型自报**（confidence_source="model"）。
- "paddle": PaddleOCR（本地、中文强）。**首次运行会下载模型**，离线业务环境需预置
          模型目录（有网环境先跑一次再拷贝）。paddleocr 2.x/3.x API 有差异，已做容错。
          置信度来源 = 引擎逐行平均（confidence_source="engine"）。
- "none":   占位——仅框架，返回空文本（跳过 OCR）。

**置信度单一来源原则**（重要）：同一份 OCR 结果只采用一个来源的置信度，不做多源
融合或启发式二次打分——混合来源会让回溯时无法判断分数出处。OpenAI 兼容协议本身
不返回置信度，故由提示词要求模型在末行自报 `CONFIDENCE: <0~1 小数>`；自报异常
（缺失 / 不可解析 / 越界 / 格式污染 / 自相矛盾）统一置 confidence=None 并打
anomaly 标记，交人工审核，而不是回退到启发式评分（中文 OCR 主要错误是形近字，
产出仍是合法可打印字符，纯文本启发式无法识别）。

无 API 且未显式选本地时的交互询问见 sources.ImageSource（tty 询问本地/跳过）。
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

# 扩展名 → data URI mime
_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
         "webp": "image/webp", "gif": "image/gif"}

# OpenAI 兼容模式：要求模型末行自报置信度（单一来源）
_CONFIDENCE_PROMPT = (
    "请识别图片中的全部文字，原样返回，不要添加解释。\n"
    "最后另起一行，只输出你对本次识别整体准确度的置信度，"
    "格式严格为 CONFIDENCE: <0 到 1 之间的小数>（例如 CONFIDENCE: 0.92）。"
)
_CONF_MARK_RE = re.compile(r"(?i)^\s*confidence\b")          # 行首出现 CONFIDENCE 标记
_CONF_LINE_RE = re.compile(r"(?i)^\s*confidence\s*[:：]\s*(.*)$")
_CONF_INLINE_RE = re.compile(r"(?i)\bconfidence\s*[:：]\s*([0-9]*\.?[0-9]+)\s*$")

# 异常码（confidence=None 时的原因，用于审核分流与日志）
ANOMALY_MISSING = "missing"              # 未找到 CONFIDENCE 行
ANOMALY_UNPARSABLE = "unparsable"        # 找到但数值不可解析
ANOMALY_OUT_OF_RANGE = "out_of_range"    # 数值不在 [0,1]
ANOMALY_MISPLACED = "misplaced"          # 不在末尾 / 出现多次（格式污染正文）
ANOMALY_CONTRADICTORY = "contradictory"  # 正文为空却自报高置信

# 提示词版本：改动 _CONFIDENCE_PROMPT / 解析规则时必须 +1——引擎指纹随之变化，
# 已有 ocr.json 缓存全部失效并自动重算（避免"图片没变、提示词变了，结果还是旧的"）。
_OCR_PROMPT_VERSION = 1

# 高置信阈值默认值（source 级 ocr_min_confidence 可覆盖）：只有 ≥ 阈值才注入正文进检索库
DEFAULT_MIN_CONFIDENCE = 0.6

# 签名导向：只保留"可判错"的签名类型（能定位问题的算子/错误码/模型/版本）
JUDGABLE_KINDS = ("kernel", "op", "errcode", "model", "version")


class OcrUnavailable(RuntimeError):
    """OCR 引擎不可用（未安装 / 初始化失败）。调用方应降级处理，不中断导入。"""


class OcrApiError(RuntimeError):
    """OCR API 调用失败（不可达 / 非 200 / 响应格式错误）。消息带定位上下文。"""


@dataclass
class OcrResult:
    """单张图片的 OCR 结果（含置信度来源与自报异常标记，便于回溯与审核分流）。"""
    text: str = ""
    confidence: Optional[float] = None
    confidence_source: str = "none"   # model（模型自报）| service（服务端）| engine（本地引擎）| none
    anomaly: str = ""                 # "" 表示正常；否则见 ANOMALY_*
    raw_confidence: str = ""          # 自报原文（回溯用）
    provider: str = ""
    mode: str = ""
    model: str = ""
    elapsed_s: float = 0.0

    @property
    def needs_review(self) -> bool:
        """自报异常 → 交人工审核（不静默当作低置信处理）。"""
        return bool(self.anomaly)


@dataclass
class OcrConfig:
    """从 config 解析出的 OCR 配置（请求期端点与审核工作台连通性测试共用同一来源）。"""
    source_id: str = ""
    provider: str = "ask"
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    mode: str = "custom"
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    extra: dict = field(default_factory=dict)

    @property
    def usable_for_request(self) -> bool:
        """请求期（服务端）可用：仅 api 模式（本地 OCR 不在服务端执行）。"""
        return self.provider == "api" and bool(self.api_base)


def as_confidence_threshold(v) -> float:
    """解析 ocr_min_confidence（非法/缺失回默认值，钳制到 [0,1]）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return DEFAULT_MIN_CONFIDENCE
    return min(max(f, 0.0), 1.0)


def ocr_config_from_cfg(cfg) -> Optional[OcrConfig]:
    """从 config 的 image source 解析 OCR 配置。

    与 review.probe_ocr_connectivity（审核工作台"测试连通"）同源，避免两处解析漂移。
    无 image source 返回 None。
    """
    sc = next((s for s in cfg.effective_sources() if s.type == "image"), None)
    if sc is None:
        return None
    return OcrConfig(
        source_id=str(getattr(sc, "id", "") or ""),
        provider=str(sc.get("ocr_provider", "ask") or "ask").lower(),
        api_base=str(sc.get("ocr_api_base", "") or ""),
        api_key=str(sc.get("ocr_api_key", "") or os.environ.get("OCR_API_KEY", "")),
        model=str(sc.get("ocr_api_model", "") or ""),
        mode=str(sc.get("ocr_api_mode", "custom") or "custom").lower(),
        min_confidence=as_confidence_threshold(sc.get("ocr_min_confidence")),
    )


def sniff_image_ext(raw: bytes, fallback: str = "png") -> str:
    """按魔数判断图片类型（不信任扩展名），返回 png/jpg/webp/gif。"""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return fallback


def parse_self_confidence(content: str) -> tuple[str, Optional[float], str, str]:
    """解析 OpenAI 兼容 OCR 返回文本中模型自报的置信度（末行 `CONFIDENCE: 0.xx`）。

    返回 (纯文本, 置信度或 None, 异常码, 自报原文)。异常码见 ANOMALY_*。
    严格只在**末行**接受该标记：出现在中间视为格式污染（misplaced），避免把正文里
    偶然出现的 "confidence:" 当成自报值而截断正文。
    """
    raw = content or ""
    lines = raw.splitlines()
    idxs = [i for i, ln in enumerate(lines) if _CONF_MARK_RE.match(ln)]
    if len(idxs) > 1:
        return raw.strip(), None, ANOMALY_MISPLACED, lines[idxs[0]].strip()
    if idxs:
        i = idxs[0]
        if any(ln.strip() for ln in lines[i + 1:]):
            return raw.strip(), None, ANOMALY_MISPLACED, lines[i].strip()
        line = lines[i]
        body = "\n".join(lines[:i]).strip()
        m = _CONF_LINE_RE.match(line)
        value_raw = (m.group(1) if m else "").strip()
        try:
            value = float(value_raw)
        except ValueError:
            return body, None, ANOMALY_UNPARSABLE, line.strip()
        if not 0.0 <= value <= 1.0:
            return body, None, ANOMALY_OUT_OF_RANGE, line.strip()
        if not body and value >= 0.5:
            return body, None, ANOMALY_CONTRADICTORY, line.strip()
        return body, value, "", line.strip()
    # 兜底：标记内联在末行文字尾部（"…报错 CONFIDENCE: 0.9"）
    m = _CONF_INLINE_RE.search(raw)
    if m:
        body = raw[:m.start()].strip()
        try:
            value = float(m.group(1))
        except ValueError:
            return body, None, ANOMALY_UNPARSABLE, m.group(0)
        if not 0.0 <= value <= 1.0:
            return body, None, ANOMALY_OUT_OF_RANGE, m.group(0)
        if not body and value >= 0.5:
            return body, None, ANOMALY_CONTRADICTORY, m.group(0)
        return body, value, "", m.group(0)
    return raw.strip(), None, ANOMALY_MISSING, ""


def engine_fingerprint(provider: str, mode: str, model: str) -> str:
    """OCR 引擎指纹 = provider | mode | model | 提示词版本。

    ocr.json 的幂等键 = **sha256 + 指纹**：换模型 / 换调用模式 / 改提示词后自动重算，
    避免"图片没变、引擎变了，结果还是旧的"。阈值（ocr_min_confidence）**不进指纹**——
    调阈值只需重新判定，无需重跑 OCR。
    """
    import hashlib

    raw = f"{provider}|{mode}|{model}|p{_OCR_PROMPT_VERSION}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def extract_judgable_signatures(text: str) -> list[dict]:
    """签名导向：只提取"可判错"的签名（算子/错误码/模型/版本），返回 [{"text","kind"}]。"""
    from .signature import extract_signatures

    return [{"text": s.text, "kind": s.kind}
            for s in extract_signatures(text or "") if s.kind in JUDGABLE_KINDS]


@dataclass
class OcrArtifact:
    """单张图片的 OCR 产物（`data/parsed/images/<stem>.ocr.json` 的内容）。

    既是幂等缓存，也是"能否进正文"的判定依据：只有 `high_confidence` 才注入正文
    （进 FTS + 向量）；低置信 / 自报异常只留签名线索并进审核队列。
    """
    sha256: str
    provider: str = ""
    mode: str = ""
    model: str = ""
    fingerprint: str = ""
    text: str = ""
    confidence: Optional[float] = None
    confidence_source: str = "none"
    anomaly: str = ""
    signatures: list = field(default_factory=list)
    min_confidence: float = DEFAULT_MIN_CONFIDENCE

    @property
    def high_confidence(self) -> bool:
        """高置信 = 有置信度 **且** 无自报异常 **且** ≥ 阈值。"""
        return (self.confidence is not None and not self.anomaly
                and self.confidence >= self.min_confidence)

    @property
    def review_reason(self) -> str:
        """需人工复核的原因（空串表示无需复核）：anomaly 优先于 low。"""
        if self.anomaly:
            return "anomaly"
        if self.confidence is None:
            return "no_confidence"
        if self.confidence < self.min_confidence:
            return "low"
        return ""

    def to_json(self, image_ref: str = "") -> dict:
        return {
            "image": image_ref,
            "sha256": self.sha256,
            "provider": self.provider,
            "mode": self.mode,
            "model": self.model,
            "engine_fingerprint": self.fingerprint,
            "text": self.text,
            "confidence": (round(self.confidence, 4)
                           if self.confidence is not None else None),
            "confidence_source": self.confidence_source,
            "anomaly": self.anomaly,
            "min_confidence": self.min_confidence,
            "text_included": self.high_confidence,
            "signatures": self.signatures,
        }

    @classmethod
    def from_json(cls, d: dict) -> "OcrArtifact":
        conf = d.get("confidence")
        return cls(
            sha256=str(d.get("sha256", "") or ""),
            provider=str(d.get("provider", "") or ""),
            mode=str(d.get("mode", "") or ""),
            model=str(d.get("model", "") or ""),
            fingerprint=str(d.get("engine_fingerprint", "") or ""),
            text=str(d.get("text", "") or ""),
            confidence=(float(conf) if isinstance(conf, (int, float)) else None),
            confidence_source=str(d.get("confidence_source", "none") or "none"),
            anomaly=str(d.get("anomaly", "") or ""),
            signatures=list(d.get("signatures") or []),
            min_confidence=as_confidence_threshold(d.get("min_confidence")),
        )

    def evidence(self) -> dict:
        """进 `extra.evidence[].ocr` 的摘要（**不含任何路径**；出口白名单会再过一遍）。"""
        return {
            "confidence": (round(self.confidence, 4)
                           if self.confidence is not None else None),
            "confidence_source": self.confidence_source,
            "anomaly": self.anomaly,
            "text_included": self.high_confidence,
            "signatures": self.signatures,
        }


def build_ocr_artifact(src: "str | Path | bytes", sha256: str, result: OcrResult,
                       provider: str, mode: str, model: str,
                       min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> OcrArtifact:
    """由一次 OCR 调用结果构造产物（含签名提取与引擎指纹）。"""
    return OcrArtifact(
        sha256=sha256, provider=provider, mode=mode, model=model,
        fingerprint=engine_fingerprint(provider, mode, model),
        text=result.text or "", confidence=result.confidence,
        confidence_source=result.confidence_source, anomaly=result.anomaly,
        signatures=extract_judgable_signatures(result.text or ""),
        min_confidence=min_confidence,
    )


def load_ocr_artifact(cache_path, sha256: str, fingerprint: str) -> Optional[OcrArtifact]:
    """读 ocr.json 缓存：**sha256 与引擎指纹都一致**才算命中，否则返回 None（需重算）。

    旧版产物没有 engine_fingerprint → 视为未命中（升级后首次全量重算一次，
    因为提示词/置信度语义已变）。
    """
    try:
        d = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("sha256") != sha256:
        return None
    if fingerprint and d.get("engine_fingerprint") != fingerprint:
        return None
    return OcrArtifact.from_json(d)


def save_ocr_artifact(cache_path, artifact: OcrArtifact, image_ref: str = "") -> None:
    """写 ocr.json（幂等缓存 + 入库判定依据）。"""
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(artifact.to_json(image_ref), ensure_ascii=False, indent=1),
                 encoding="utf-8")


_paddle_ocr = None  # 全局单例（模型加载一次）

# OCR HTTP 超时（秒）：VLM 类 OCR 对大图可能需要数十秒，可用 VLLM_KB_OCR_TIMEOUT 调整
_HTTP_TIMEOUT = int(os.environ.get("VLLM_KB_OCR_TIMEOUT", "60") or 60)


def _get_paddle():
    global _paddle_ocr
    if _paddle_ocr is not None:
        return _paddle_ocr
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise OcrUnavailable(f"未安装 paddleocr（pip install paddleocr paddlepaddle）：{e}") from e
    try:
        # 2.x 接口；lang=ch 覆盖中英文
        _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    except TypeError:
        # 部分版本不支持 show_log / use_angle_cls
        _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    return _paddle_ocr


def _read_image_bytes(src: "str | Path | bytes") -> tuple[bytes, str]:
    """读入图片字节，返回 (bytes, 扩展名提示)。bytes 直通（请求期不落盘）。"""
    if isinstance(src, (bytes, bytearray)):
        return bytes(src), ""
    p = Path(src)
    if not p.exists():
        raise OcrApiError(f"图片不存在：{p}")
    return p.read_bytes(), p.suffix.lstrip(".").lower()


def _post_json(url: str, payload: dict, api_key: str, timeout: int = 0) -> dict:
    """POST JSON 并解析响应；所有失败路径都带定位上下文（URL / 状态码 / body 前缀 / 耗时）。

    非 JSON 响应经 net.parse_json 统一防护（200 空体、网关/限流页等），
    便于服务不可用时给出可读提示与日志。timeout=0 用 _HTTP_TIMEOUT。
    """
    from .net import parse_json

    timeout = timeout or _HTTP_TIMEOUT
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib_request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except HTTPError as e:  # 须先于 URLError 捕获（HTTPError 是其子类）
        snippet = ""
        try:
            snippet = e.read().decode("utf-8", errors="replace")[:200].replace("\n", " ")
        except Exception:
            pass
        raise OcrApiError(
            f"OCR API 返回 HTTP {e.code}（{url}，耗时 {time.time() - t0:.1f}s）："
            f"{snippet or e.reason}"
        ) from e
    except (URLError, OSError) as e:
        raise OcrApiError(
            f"OCR API 不可达（{url}，耗时 {time.time() - t0:.1f}s）：{e}"
            f"（请确认 OCR 服务已启动、ocr_api_base 可达）"
        ) from e
    try:
        return parse_json(body, f"OCR API {url}")
    except RuntimeError as e:
        raise OcrApiError(f"{e}（耗时 {time.time() - t0:.1f}s）") from e


def _ocr_via_custom(src: "str | Path | bytes", api_base: str, api_key: str,
                    model: str, t0: float) -> OcrResult:
    """自研协议：POST {api_base}/ocr。model 可选透传（服务端默认）。"""
    if not api_base:
        raise OcrApiError("ocr_api_base 未配置（OCR API provider 需要服务地址）")
    raw, ext = _read_image_bytes(src)
    b64 = base64.b64encode(raw).decode()
    name = str(src).rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if not isinstance(src, (bytes, bytearray)) else ""
    body: dict = {"image": b64, "filename": name or f"image.{sniff_image_ext(raw, ext or 'png')}"}
    if model:
        body["model"] = model
    data = _post_json(api_base.rstrip("/") + "/ocr", body, api_key)
    text = data.get("text") or ""
    try:
        conf: Optional[float] = float(data["confidence"])
    except (KeyError, TypeError, ValueError):
        conf = None
    return OcrResult(text=str(text), confidence=conf, confidence_source="service",
                     anomaly="" if conf is not None else ANOMALY_MISSING,
                     raw_confidence=str(data.get("confidence", "")),
                     provider="api", mode="custom", model=model,
                     elapsed_s=time.time() - t0)


def _ocr_via_openai(src: "str | Path | bytes", api_base: str, api_key: str,
                    model: str, t0: float) -> OcrResult:
    """OpenAI 兼容接口（vLLM 部署的 DeepSeek-OCR 等）：/chat/completions + 内联图片。

    置信度 = 模型自报（末行 CONFIDENCE: 0.xx）；自报异常置 None 并打 anomaly。
    """
    if not api_base:
        raise OcrApiError("ocr_api_base 未配置（OCR API provider 需要服务地址）")
    if not model:
        raise OcrApiError("openai 模式的 OCR 需要 ocr_api_model（如 deepseek-ai/DeepSeek-OCR）")
    raw, ext = _read_image_bytes(src)
    mime = _MIME.get(sniff_image_ext(raw, ext or "png"), "image/png")
    b64 = base64.b64encode(raw).decode()
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": _CONFIDENCE_PROMPT},
            ],
        }],
    }
    data = _post_json(api_base.rstrip("/") + "/chat/completions", payload, api_key)
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise OcrApiError(f"OpenAI 兼容 OCR 响应格式异常: {e}（{str(data)[:200]}）") from e
    text, conf, anomaly, raw_conf = parse_self_confidence(str(content))
    return OcrResult(text=text, confidence=conf, confidence_source="model", anomaly=anomaly,
                     raw_confidence=raw_conf, provider="api", mode="openai", model=model,
                     elapsed_s=time.time() - t0)


def _ocr_via_paddle(src: "str | Path | bytes", t0: float) -> OcrResult:
    """本地 PaddleOCR（仅导入期使用；服务端请求期不执行本地 OCR）。"""
    ocr = _get_paddle()
    tmp: Optional[str] = None
    if isinstance(src, (bytes, bytearray)):
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        Path(tmp).write_bytes(bytes(src))
        target = tmp
    else:
        target = str(src)
    try:
        result = ocr.ocr(target, cls=True)
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    lines: list[str] = []
    confs: list[float] = []
    # 2.x 返回 [[ [box, (text, conf)], ... ], ...]；3.x 可能不同——兼容解析
    for page in result or []:
        for item in page or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                payload = item[1]
                if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                    text, conf = payload[0], float(payload[1])
                    if text:
                        lines.append(str(text))
                        confs.append(conf)
    conf_avg = (sum(confs) / len(confs)) if confs else None
    return OcrResult(text="\n".join(lines), confidence=conf_avg, confidence_source="engine",
                     anomaly="" if conf_avg is not None else ANOMALY_MISSING,
                     provider="paddle", mode="", model="", elapsed_s=time.time() - t0)


def ocr_image_detail(src: "str | Path | bytes", provider: str = "none",
                     api_base: str = "", api_key: str = "", model: str = "",
                     mode: str = "custom") -> OcrResult:
    """对单张图片做 OCR，返回完整结果（含置信度来源与自报异常标记）。

    src 可以是图片路径，也可以是**原始字节**（请求期端点用，避免落盘）。
    provider: api（HTTP 服务，需 api_base；mode=custom 自研协议 / openai OpenAI 兼容，
    model 在 openai 模式必填）| paddle（本地）| none（占位）。
    引擎不可用抛 OcrUnavailable / OcrApiError，调用方决定询问或跳过。
    """
    provider = (provider or "none").lower()
    t0 = time.time()
    if provider == "none":
        return OcrResult(provider="none", mode=mode, model=model, elapsed_s=0.0)
    if provider == "api":
        mode = (mode or "custom").lower()
        if mode == "openai":
            return _ocr_via_openai(src, api_base, api_key, model, t0)
        return _ocr_via_custom(src, api_base, api_key, model, t0)
    if provider == "paddle":
        return _ocr_via_paddle(src, t0)
    raise OcrUnavailable(f"未知 OCR provider: {provider}（支持 api | paddle | none）")


def ocr_image(src: "str | Path | bytes", provider: str = "none",
              api_base: str = "", api_key: str = "", model: str = "",
              mode: str = "custom") -> tuple[str, float]:
    """对单张图片做 OCR，返回 (识别文本, 平均置信度)。

    兼容旧签名：置信度缺失（自报异常 / 引擎无输出）统一回退为 0.0（低置信语义，
    调用方据此不把结果当作可信文本入库）。需要区分"异常"与"低置信"时用
    ocr_image_detail。
    """
    r = ocr_image_detail(src, provider, api_base=api_base, api_key=api_key,
                         model=model, mode=mode)
    return r.text, (r.confidence if r.confidence is not None else 0.0)
