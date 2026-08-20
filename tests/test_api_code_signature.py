"""API 端点测试（需 fastapi + httpx；未安装时跳过）。

覆盖：/signature-search、/code/versions、/code/search、/code/file。
结构只读断言复用 test_api_readonly 的思路。
"""
import unittest

from vllm_kb.config import AppConfig
from vllm_kb.signature import extract_signatures, format_hits, signature_search


@unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi"),
    "fastapi 未安装（pip install fastapi uvicorn）",
)
class TestApiCodeAndSignature(unittest.TestCase):
    def setUp(self):
        import os

        from fastapi.testclient import TestClient

        from vllm_kb.api import create_app

        # config.json 密钥走环境变量（脱敏后无明文 api_key）
        self._old_key = os.environ.get("EMBEDDING_API_KEY")
        self._old_gh = os.environ.get("GITHUB_TOKEN")
        os.environ["EMBEDDING_API_KEY"] = "dummy-for-test"
        os.environ["GITHUB_TOKEN"] = "dummy-for-test"
        self.client = TestClient(create_app("config.json"))

    def tearDown(self):
        import os

        for name, old in (("EMBEDDING_API_KEY", self._old_key), ("GITHUB_TOKEN", self._old_gh)):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old

    def test_signature_search_endpoint(self):
        r = self.client.post(
            "/signature-search",
            json={
                "text": "kernel_name=DispatchFFNCombine errorStr: timeout or trap error GLM-5.1",
                "component": "vllm-ascend",
                "top_k": 5,
            },
        )
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertTrue(d["signatures"])
        self.assertIsInstance(d["results"], list)

    def test_code_versions_endpoint(self):
        r = self.client.get("/code/versions")
        self.assertEqual(r.status_code, 200)
        self.assertIn("versions", r.json())

    def test_code_search_endpoint(self):
        r = self.client.post(
            "/code/search", json={"keyword": "dispatch_ffn_combine", "version": "v0.23.0rc1", "limit": 5}
        )
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertIn("mode", d)

    def test_code_file_endpoint(self):
        r = self.client.get(
            "/code/file",
            params={"version": "v0.23.0rc1",
                    "path": "csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp",
                    "max_chars": 1000},
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("content", r.json())


class TestSignaturePipeline(unittest.TestCase):
    """签名提取 → 检索的 Python 层集成（不依赖 API/网络）。"""

    def test_extract_and_format(self):
        text = "halMemCreate failed drvRetCode=6 kernel_name=DispatchFFNCombine GLM-5.1"
        sigs = extract_signatures(text)
        self.assertTrue(any(s.kind == "kernel" and "dispatchffncombine" in s.text for s in sigs))
        self.assertTrue(any(s.kind == "errcode" for s in sigs))
        self.assertTrue(any("glm-5.1" in s.text for s in sigs))


if __name__ == "__main__":
    unittest.main()
