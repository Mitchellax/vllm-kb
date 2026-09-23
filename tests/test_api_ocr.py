"""请求期图片 OCR 端点测试（需 fastapi + httpx；未安装时跳过）。

覆盖：端点注册开关（有/无 image source）、provider 校验（400 且说明原因）、
入参校验（空/base64 非法/非图片魔数/超大）、成功路径（文本+签名+置信度来源）、
置信度自报异常（needs_review）、OCR 服务不可用（503 且含 base_url）、/health 的 ocr 状态。

不依赖真实 OCR 服务——mock vllm_kb.api_image.ocr_image_detail。
"""
import base64
import json
import os
import shutil
import tempfile
import unittest

from vllm_kb.ocr import ANOMALY_MISSING, OcrApiError, OcrResult


def _png_bytes() -> bytes:
    """最小合法 PNG（1x1）——端点按魔数校验，不依赖 PIL。"""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _has_fastapi() -> bool:
    import importlib.util

    return importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(_has_fastapi(), "fastapi 未安装（pip install fastapi uvicorn）")
class _OcrApiBase(unittest.TestCase):
    """合成配置 + TestClient；子类用 SOURCES 决定 image source 形态。"""

    SOURCES: list = []

    def setUp(self):
        from fastapi.testclient import TestClient

        from vllm_kb.api import create_app
        from vllm_kb.config import AppConfig

        self._dir = tempfile.mkdtemp()
        path = os.path.join(self._dir, "config.json")
        cfg = AppConfig.model_validate({
            "embedding": {"provider": "echo", "dimensions": 64},
            "storage": {
                "vector_backend": "python",
                "lancedb_path": os.path.join(self._dir, "lancedb"),
                "sqlite_path": os.path.join(self._dir, "kb.sqlite3"),
                "canonical_file": os.path.join(self._dir, "canonical.jsonl"),
            },
            "sources": [dict(s, enabled=True) for s in self.SOURCES],
        })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg.model_dump(by_alias=True), f, ensure_ascii=False)
        self._old = {k: os.environ.get(k) for k in ("EMBEDDING_API_KEY", "GITHUB_TOKEN")}
        os.environ["EMBEDDING_API_KEY"] = "dummy"
        os.environ["GITHUB_TOKEN"] = "dummy"
        self.client = TestClient(create_app(path))

    def tearDown(self):
        for k, old in self._old.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        shutil.rmtree(self._dir, ignore_errors=True)


class TestOcrEndpointUnconfigured(_OcrApiBase):
    """无 image source：端点不存在（404）——"没配 OCR"。"""

    SOURCES: list = []

    def test_endpoint_absent(self):
        r = self.client.post("/ocr", json={"image": _b64(_png_bytes())})
        self.assertEqual(r.status_code, 404)

    def test_health_reports_unconfigured(self):
        h = self.client.get("/health").json()
        self.assertEqual(h["ocr"]["state"], "unconfigured")


class TestOcrEndpointProviderGuard(_OcrApiBase):
    """有 image source 但 provider 不是 api：端点存在但 400 说明原因（比 404 好诊断）。"""

    SOURCES = [{"id": "images", "type": "image", "ocr_provider": "paddle"}]

    def test_paddle_rejected_with_reason(self):
        r = self.client.post("/ocr", json={"image": _b64(_png_bytes())})
        self.assertEqual(r.status_code, 400)
        detail = r.json()["detail"]
        self.assertIn("ocr_provider=api", detail)
        self.assertIn("paddle", detail)
        self.assertIn("本地 OCR 不在服务端执行", detail)

    def test_health_reports_provider(self):
        h = self.client.get("/health").json()
        self.assertEqual(h["ocr"]["state"], "configured")
        self.assertIsNone(h["ocr"]["endpoint"])  # 非 api：无请求期端点
        self.assertIn("provider=paddle", h["ocr"]["note"])


class TestOcrEndpointValidation(_OcrApiBase):
    SOURCES = [{"id": "images", "type": "image", "ocr_provider": "api",
                "ocr_api_base": "http://ocr:8000", "ocr_api_mode": "openai",
                "ocr_api_model": "deepseek-ai/DeepSeek-OCR"}]

    def test_empty_image_400(self):
        r = self.client.post("/ocr", json={"image": "   "})
        self.assertEqual(r.status_code, 400)
        self.assertIn("不能为空", r.json()["detail"])

    def test_bad_base64_400(self):
        r = self.client.post("/ocr", json={"image": "!!!not-base64!!!"})
        self.assertEqual(r.status_code, 400)

    def test_non_image_magic_400(self):
        r = self.client.post("/ocr", json={"image": _b64(b"this is not an image at all")})
        self.assertEqual(r.status_code, 400)
        self.assertIn("不支持的图片格式", r.json()["detail"])

    def test_oversize_413(self):
        from vllm_kb.api_image import MAX_B64_CHARS

        r = self.client.post("/ocr", json={"image": "A" * (MAX_B64_CHARS + 4)})
        self.assertEqual(r.status_code, 413)
        self.assertIn("图片过大", r.json()["detail"])

    def test_missing_image_field_422(self):
        r = self.client.post("/ocr", json={})
        self.assertEqual(r.status_code, 422)


class TestOcrEndpointSuccess(_OcrApiBase):
    SOURCES = [{"id": "images", "type": "image", "ocr_provider": "api",
                "ocr_api_base": "http://ocr:8000", "ocr_api_mode": "openai",
                "ocr_api_model": "deepseek-ai/DeepSeek-OCR"}]

    def _post(self, result):
        from unittest import mock

        with mock.patch("vllm_kb.api_image.ocr_image_detail", return_value=result) as m:
            r = self.client.post("/ocr", json={"image": _b64(_png_bytes())})
            return r, m

    def test_success_returns_text_signatures_and_confidence(self):
        res = OcrResult(
            text="error code 107020, dispatch_ffn_combine failed, GLM-5.1",
            confidence=0.92, confidence_source="model", anomaly="",
            raw_confidence="CONFIDENCE: 0.92", provider="api", mode="openai",
            model="deepseek-ai/DeepSeek-OCR", elapsed_s=1.5)
        r, m = self._post(res)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("dispatch_ffn_combine", body["text"])
        self.assertEqual(body["confidence"], 0.92)
        self.assertEqual(body["confidence_source"], "model")
        self.assertEqual(body["anomaly"], "")
        self.assertFalse(body["needs_review"])
        self.assertEqual(body["model"], "deepseek-ai/DeepSeek-OCR")
        kinds = {s["kind"] for s in body["signatures"]}
        self.assertTrue(kinds & {"errcode", "op", "model"}, kinds)
        self.assertIn("不可信输入", body["note"])
        # 图片以字节直传 OCR 引擎（不落盘、不读客户端路径）
        passed = m.call_args.args[0]
        self.assertIsInstance(passed, bytes)

    def test_anomaly_marks_needs_review(self):
        res = OcrResult(text="halMemCreate failed", confidence=None,
                        confidence_source="model", anomaly=ANOMALY_MISSING,
                        provider="api", mode="openai", model="m", elapsed_s=0.4)
        r, _ = self._post(res)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIsNone(body["confidence"])
        self.assertEqual(body["anomaly"], "missing")
        self.assertTrue(body["needs_review"])

    def test_text_truncated_but_signatures_use_full_text(self):
        from vllm_kb.api_image import MAX_TEXT_CHARS

        res = OcrResult(text="x" * (MAX_TEXT_CHARS + 100) + " dispatch_ffn_combine failed",
                        confidence=0.9, confidence_source="model", provider="api",
                        mode="openai", model="m")
        r, _ = self._post(res)
        body = r.json()
        self.assertTrue(body["truncated"])
        self.assertEqual(len(body["text"]), MAX_TEXT_CHARS)

    def test_engine_unavailable_503_with_base_url(self):
        from unittest import mock

        with mock.patch("vllm_kb.api_image.ocr_image_detail",
                        side_effect=OcrApiError("OCR API 不可达（http://ocr:8000/chat/completions）："
                                                "connection refused")):
            r = self.client.post("/ocr", json={"image": _b64(_png_bytes())})
        self.assertEqual(r.status_code, 503)
        detail = r.json()["detail"]
        self.assertIn("OCR 服务不可用", detail)
        self.assertIn("http://ocr:8000", detail)
        self.assertIn("粘贴图片中的文字", detail)

    def test_openai_mode_requires_model(self):
        """openai 模式缺 ocr_api_model → 400（配置问题，非服务不可用）。"""
        import shutil

        from fastapi.testclient import TestClient

        from vllm_kb.api import create_app
        from vllm_kb.config import AppConfig

        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "config.json")
            cfg = AppConfig.model_validate({
                "embedding": {"provider": "echo", "dimensions": 64},
                "storage": {"vector_backend": "python",
                            "lancedb_path": os.path.join(d, "lancedb"),
                            "sqlite_path": os.path.join(d, "kb.sqlite3"),
                            "canonical_file": os.path.join(d, "canonical.jsonl")},
                "sources": [{"id": "images", "type": "image", "enabled": True,
                             "ocr_provider": "api", "ocr_api_base": "http://ocr:8000",
                             "ocr_api_mode": "openai"}],
            })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg.model_dump(by_alias=True), f, ensure_ascii=False)
            client = TestClient(create_app(path))
            r = client.post("/ocr", json={"image": _b64(_png_bytes())})
            self.assertEqual(r.status_code, 400)
            self.assertIn("ocr_api_model", r.json()["detail"])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_health_reports_configured_endpoint(self):
        h = self.client.get("/health").json()
        self.assertEqual(h["ocr"]["state"], "configured")
        self.assertEqual(h["ocr"]["endpoint"], "/ocr")
        self.assertIn("mode=openai", h["ocr"]["note"])


class TestOcrEndpointNoPathLeak(_OcrApiBase):
    """响应不得含服务端配置/存储路径（与 /search 等端点同一安全约束）。"""

    SOURCES = [{"id": "images", "type": "image", "ocr_provider": "api",
                "ocr_api_base": "http://ocr:8000", "ocr_api_mode": "openai",
                "ocr_api_model": "m"}]

    def test_response_has_no_server_paths(self):
        from unittest import mock

        res = OcrResult(text="halMemCreate failed drvRetCode=6", confidence=0.9,
                        confidence_source="model", provider="api", mode="openai", model="m")
        with mock.patch("vllm_kb.api_image.ocr_image_detail", return_value=res):
            raw = self.client.post("/ocr", json={"image": _b64(_png_bytes())}).text
        self.assertNotIn(self._dir, raw)
        self.assertNotIn("lancedb", raw)
        self.assertNotIn("kb.sqlite3", raw)


if __name__ == "__main__":
    unittest.main()
