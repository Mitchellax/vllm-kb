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
    extra: dict = field(default_factory=dict)

    @property
    def usable_for_request(self) -> bool:
        """请求期（服务端）可用：仅 api 模式（本地 OCR 不在服务端执行）。"""
        return self.provider == "api" and bool(self.api_base)


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
