"""OCR 引擎抽象：签名导向 OCR（图片 → 文本 → 错误签名提取）。

provider 可插拔（config source 的 ocr_provider 字段）：
- "api":    HTTP OCR 服务（预留 API），支持两种调用模式（ocr_api_mode）：
  * "custom"（默认）: 自研协议 POST {ocr_api_base}/ocr
        body: {"image": "<base64>", "filename": "x.png", "model": "<可选>"}
        resp: {"text": "...", "confidence": 0.93}
  * "openai": OpenAI 兼容接口（如 siliconflow 的 DeepSeek-OCR）：
        POST {ocr_api_base}/chat/completions，model=ocr_api_model（必填），
        messages 内联 data URI 图片；响应取 choices[0].message.content。
      **model 为必填**（openai 模式）；custom 模式 model 可选透传（服务端默认）。
- "paddle": PaddleOCR（本地、中文强）。**首次运行会下载模型**，离线业务环境需预置
          模型目录（有网环境先跑一次再拷贝）。paddleocr 2.x/3.x API 有差异，已做容错。
- "none":   占位——仅框架，返回空文本（跳过 OCR）。

无 API 且未显式选本地时的交互询问见 sources.ImageSource（tty 询问本地/跳过）。
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

# 扩展名 → data URI mime
_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
         "webp": "image/webp", "gif": "image/gif"}


class OcrUnavailable(RuntimeError):
    """OCR 引擎不可用（未安装 / 初始化失败）。调用方应降级处理，不中断导入。"""


class OcrApiError(RuntimeError):
    """OCR API 调用失败（不可达 / 非 200 / 响应格式错误）。"""


_paddle_ocr = None  # 全局单例（模型加载一次）


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


def _post_json(url: str, payload: dict, api_key: str, timeout: int = 60) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib_request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, OSError, json.JSONDecodeError) as e:
        raise OcrApiError(f"OCR API 调用失败 {url}: {e}") from e


def _ocr_via_custom(path: str | Path, api_base: str, api_key: str, model: str) -> tuple[str, float]:
    """自研协议：POST {api_base}/ocr。model 可选透传（服务端默认）。"""
    if not api_base:
        raise OcrApiError("ocr_api_base 未配置（OCR API provider 需要服务地址）")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body: dict = {"image": b64, "filename": str(path).rsplit("/", 1)[-1]}
    if model:
        body["model"] = model
    data = _post_json(api_base.rstrip("/") + "/ocr", body, api_key)
    text = data.get("text") or ""
    conf = float(data.get("confidence") or 0.0)
    return text, conf


def _ocr_via_openai(path: str | Path, api_base: str, api_key: str, model: str) -> tuple[str, float]:
    """OpenAI 兼容接口（如 siliconflow DeepSeek-OCR）：/chat/completions + 内联图片。"""
    if not api_base:
        raise OcrApiError("ocr_api_base 未配置（OCR API provider 需要服务地址）")
    if not model:
        raise OcrApiError("openai 模式的 OCR 需要 ocr_api_model（如 deepseek-ai/DeepSeek-OCR）")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = str(path).rsplit(".", 1)[-1].lower() if "." in str(path) else "png"
    mime = _MIME.get(ext, "image/png")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": "请识别图片中的全部文字，原样返回，不要添加解释。"},
            ],
        }],
    }
    data = _post_json(api_base.rstrip("/") + "/chat/completions", payload, api_key)
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise OcrApiError(f"OpenAI 兼容 OCR 响应格式异常: {e}（{str(data)[:200]}）") from e
    return text.strip(), 0.0  # OpenAI 兼容无置信度


def ocr_image(path: str | Path, provider: str = "none",
              api_base: str = "", api_key: str = "", model: str = "",
              mode: str = "custom") -> tuple[str, float]:
    """对单张图片做 OCR，返回 (识别文本, 平均置信度)。

    provider: api（HTTP 服务，需 api_base；mode=custom 自研协议 / openai OpenAI 兼容，
    model 在 openai 模式必填）| paddle（本地）| none（占位）。
    引擎不可用抛 OcrUnavailable / OcrApiError，调用方决定询问或跳过。
    """
    provider = (provider or "none").lower()
    if provider == "none":
        return "", 0.0
    if provider == "api":
        mode = (mode or "custom").lower()
        if mode == "openai":
            return _ocr_via_openai(path, api_base, api_key, model)
        return _ocr_via_custom(path, api_base, api_key, model)
    if provider == "paddle":
        ocr = _get_paddle()
        result = ocr.ocr(str(path), cls=True)
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
        text = "\n".join(lines)
        conf = (sum(confs) / len(confs)) if confs else 0.0
        return text, conf
    raise OcrUnavailable(f"未知 OCR provider: {provider}（支持 api | paddle | none）")
