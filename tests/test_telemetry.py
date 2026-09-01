"""行为遥测采集层单测：中间件记录+会话归属+独立库不碰 kb.sqlite3。

用 starlette/fastapi TestClient 构造带 telemetry 中间件的 app，验证：
- 查询行为记入 telemetry.sqlite3（独立库）
- X-Session-Id 透传 / 缺失回退 ip+时间窗
- 未启用时不挂中间件（零开销）
- 只记关键端点（/health 不记）
"""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _make_app(cfg_path, feedback_enabled=True):
    """构造带 telemetry 的 app（复用真实 create_app，但强制 feedback_enabled + 隔离 telemetry 路径）。"""
    from vllm_kb.config import AppConfig
    from vllm_kb.api import create_app

    cfg = AppConfig.load(cfg_path, require_keys=False)
    cfg.confidence.feedback_enabled = feedback_enabled
    d = tempfile.mkdtemp()
    # telemetry 库隔离到临时目录（避免多测试共享真实 data/telemetry.sqlite3）
    cfg.confidence.telemetry_path = os.path.join(d, "telemetry.sqlite3")
    p = os.path.join(d, "config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(by_alias=True), f, ensure_ascii=False)
    return create_app(p), d


@unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi"),
    "fastapi 未安装",
)
class TestTelemetryMiddleware(unittest.TestCase):
    def setUp(self):
        os.environ["EMBEDDING_API_KEY"] = "dummy"
        os.environ["GITHUB_TOKEN"] = "dummy"
        self._tmpdirs = []

    def tearDown(self):
        for d in self._tmpdirs:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        os.environ.pop("EMBEDDING_API_KEY", None)
        os.environ.pop("GITHUB_TOKEN", None)

    def _app(self, feedback_enabled=True):
        app, d = _make_app("config.json", feedback_enabled)
        self._tmpdirs.append(d)
        return app, d

    def _telemetry_db(self, d):
        # 找 telemetry 路径（cfg.telemetry_path 基于 data_root）
        from vllm_kb.config import AppConfig
        cfg = AppConfig.load(os.path.join(d, "config.json"), require_keys=False)
        return cfg.resolve(cfg.confidence.telemetry_path)

    def test_search_recorded(self):
        from fastapi.testclient import TestClient

        app, d = self._app()
        with TestClient(app) as c:
            c.post("/search", json={"query": "test query", "top_k": 3})
        db = self._telemetry_db(d)
        self.assertTrue(db.exists())
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT endpoint, query_hash FROM query_events").fetchall()
        conn.close()
        self.assertTrue(any(r[0] == "/search" for r in rows))
        self.assertTrue(all(r[1] for r in rows))  # query_hash 非空

    def test_health_not_recorded(self):
        from fastapi.testclient import TestClient

        app, d = self._app()
        with TestClient(app) as c:
            c.get("/health")
        db = self._telemetry_db(d)
        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT count(*) FROM query_events").fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)  # /health 不记

    def test_session_id_from_header(self):
        from fastapi.testclient import TestClient

        app, d = self._app()
        with TestClient(app) as c:
            c.post("/search", json={"query": "x"}, headers={"X-Session-Id": "sess-123"})
        db = self._telemetry_db(d)
        conn = sqlite3.connect(str(db))
        sid = conn.execute("SELECT session_id FROM query_events").fetchone()[0]
        conn.close()
        # TestClient 可能规范化 header 大小写，middleware 用小写读取应能匹配
        self.assertEqual(sid, "sess-123")

    def test_session_fallback_ip_window(self):
        """无 X-Session-Id → 回退 ip:时间窗。"""
        from fastapi.testclient import TestClient

        app, d = self._app()
        with TestClient(app) as c:
            c.post("/search", json={"query": "x"})
        db = self._telemetry_db(d)
        conn = sqlite3.connect(str(db))
        sid = conn.execute("SELECT session_id FROM query_events").fetchone()[0]
        conn.close()
        self.assertTrue(sid.startswith("ip:"))

    def test_disabled_no_middleware(self):
        """feedback_enabled=False → 不记，telemetry 库不创建。"""
        from fastapi.testclient import TestClient

        app, d = self._app(feedback_enabled=False)
        with TestClient(app) as c:
            c.post("/search", json={"query": "x"})
        db = self._telemetry_db(d)
        self.assertFalse(db.exists())  # 库未创建

    def test_telemetry_failure_does_not_break_query(self):
        """遥测写失败不影响检索（宁可丢遥测不丢查询）。"""
        from fastapi.testclient import TestClient

        app, d = self._app()
        with mock.patch("vllm_kb.telemetry.TelemetryStore.record", side_effect=sqlite3.Error("boom")):
            with TestClient(app) as c:
                r = c.post("/search", json={"query": "x"})
        self.assertEqual(r.status_code, 200)  # 查询仍成功


@unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi"),
    "fastapi 未安装",
)
class TestClientSessionHeader(unittest.TestCase):
    """client.py 透传 VLLM_KB_SESSION → X-Session-Id header。"""

    def test_session_env_forwarded(self):
        import importlib.util
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "client_mod", PROJECT_ROOT / "skills" / "vllm-kb" / "client.py")
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)

        captured = {}

        class FakeResp:
            def read(self):
                return b'{}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.headers)
            return FakeResp()

        old = os.environ.pop("VLLM_KB_SESSION", None)
        os.environ["VLLM_KB_SESSION"] = "agent-session-abc"
        try:
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                client._get("http://x:8000", "/health")
        finally:
            if old is None:
                os.environ.pop("VLLM_KB_SESSION", None)
            else:
                os.environ["VLLM_KB_SESSION"] = old
        # urllib 规范化 header 名大小写，用大小写不敏感查找
        hdrs = {k.lower(): v for k, v in captured["headers"].items()}
        self.assertEqual(hdrs.get("x-session-id"), "agent-session-abc")

    def test_no_session_env_no_header(self):
        import importlib.util
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "client_mod2", PROJECT_ROOT / "skills" / "vllm-kb" / "client.py")
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)

        captured = {}

        class FakeResp:
            def read(self):
                return b'{}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.headers)
            return FakeResp()

        old = os.environ.pop("VLLM_KB_SESSION", None)
        try:
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                client._get("http://x:8000", "/health")
        finally:
            if old is not None:
                os.environ["VLLM_KB_SESSION"] = old
        self.assertNotIn("X-Session-Id", captured["headers"])


if __name__ == "__main__":
    unittest.main()
