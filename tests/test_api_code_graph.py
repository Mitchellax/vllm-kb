"""代码图谱检索 API 端点测试（需 fastapi + httpx；未安装时跳过）。

覆盖：端点注册开关、gh-puller 不可达时 503 + 引导、健康探测、参数校验。
不依赖真实 gh-puller——mock CodeGraphClient 方法。
"""
import os
import tempfile
import unittest
from unittest import mock


@unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi"),
    "fastapi 未安装（pip install fastapi uvicorn）",
)
class TestApiCodeGraph(unittest.TestCase):
    def setUp(self):
        import json

        from fastapi.testclient import TestClient

        from vllm_kb.api import create_app

        # 造一个 enabled=True 的临时 config，复用 config.json 的其余字段
        base = "config.json"
        from vllm_kb.config import AppConfig

        cfg = AppConfig.load(base, require_keys=False)
        cfg.code_graph.enabled = True
        cfg.code_graph.base_url = "http://localhost:8001"

        self._cfg_dir = tempfile.mkdtemp()
        path = os.path.join(self._cfg_dir, "config.json")
        # round-trip 保证 extra 字段不丢
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg.model_dump(by_alias=True), f, ensure_ascii=False)

        self._old_e = os.environ.get("EMBEDDING_API_KEY")
        self._old_g = os.environ.get("GITHUB_TOKEN")
        os.environ["EMBEDDING_API_KEY"] = "dummy"
        os.environ["GITHUB_TOKEN"] = "dummy"
        self.client = TestClient(create_app(path))

    def tearDown(self):
        import shutil

        for name, old in (("EMBEDDING_API_KEY", self._old_e), ("GITHUB_TOKEN", self._old_g)):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        shutil.rmtree(self._cfg_dir, ignore_errors=True)

    def test_endpoints_registered_when_enabled(self):
        # 端点存在（不再 404）；gh-puller 未启动 → 503
        r = self.client.get("/code-graph/health")
        self.assertIn(r.status_code, (200, 503))  # health 探测可达/不可达都合理

    def test_search_requires_query_or_pattern(self):
        r = self.client.post("/code-graph/search", json={})
        self.assertEqual(r.status_code, 400)

    def test_unreachable_returns_503_with_guide(self):
        r = self.client.post("/code-graph/search", json={"query": "auth"})
        self.assertEqual(r.status_code, 503)
        detail = r.json().get("detail", "")
        self.assertIn("code", detail)  # 引导用 code 命令

    def test_mock_success_passthrough(self):
        with mock.patch("vllm_kb.code_graph.CodeGraphClient.search_graph",
                        return_value={"results": [{"name": "foo"}]}):
            r = self.client.post("/code-graph/search",
                                 json={"query": "auth", "repo": "vllm"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["results"])

    def test_repo_mapping_in_request(self):
        """repo=vllm 映射到 project=vllm-project/vllm，传给上游。"""
        with mock.patch("vllm_kb.code_graph.CodeGraphClient.search_graph",
                        return_value={"ok": True}) as m:
            self.client.post("/code-graph/search", json={"query": "x", "repo": "vllm"})
        m.assert_called_once()
        self.assertEqual(m.call_args.kwargs["project"], "vllm-project/vllm")

    def test_changes_endpoint(self):
        with mock.patch("vllm_kb.code_graph.CodeGraphClient.detect_changes",
                        return_value={"impacted": []}):
            r = self.client.post("/code-graph/changes",
                                 json={"diff": "diff --git", "repo": "vllm-ascend"})
        self.assertEqual(r.status_code, 200)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi"),
    "fastapi 未安装",
)
class TestApiCodeGraphDisabled(unittest.TestCase):
    """code_graph.enabled=False（默认）时端点不注册（404 比 503 更干净）。"""

    def setUp(self):
        import os

        from fastapi.testclient import TestClient

        from vllm_kb.api import create_app

        self._old_e = os.environ.get("EMBEDDING_API_KEY")
        self._old_g = os.environ.get("GITHUB_TOKEN")
        os.environ["EMBEDDING_API_KEY"] = "dummy"
        os.environ["GITHUB_TOKEN"] = "dummy"
        self.client = TestClient(create_app("config.json"))

    def tearDown(self):
        for name, old in (("EMBEDDING_API_KEY", self._old_e), ("GITHUB_TOKEN", self._old_g)):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old

    def test_endpoints_not_registered(self):
        self.assertEqual(self.client.get("/code-graph/health").status_code, 404)
        self.assertEqual(self.client.post("/code-graph/search", json={"query": "x"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
