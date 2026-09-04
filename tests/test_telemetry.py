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
class TestProbeTagging(unittest.TestCase):
    """探索/测试行为打标：显式 header=1 / 启发式范例词·占位词·范例 doc_id=2 / 真实=0。"""

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

    def _app(self):
        app, d = _make_app("config.json", feedback_enabled=True)
        self._tmpdirs.append(d)
        return app, d

    def _probes(self, d):
        from vllm_kb.config import AppConfig
        cfg = AppConfig.load(os.path.join(d, "config.json"), require_keys=False)
        db = cfg.resolve(cfg.confidence.telemetry_path)
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT endpoint, query_normalized, probe FROM query_events").fetchall()
        conn.close()
        return rows

    def test_probe_levels(self):
        from fastapi.testclient import TestClient

        app, d = self._app()
        with TestClient(app) as c:
            # 1. 显式声明（header）→ probe=1
            c.post("/search", json={"query": "真实故障排查问题"},
                   headers={"X-VLLM-KB-Probe": "1"})
            # 2. 范例词复现（SKILL.md 示例）→ probe=2
            c.post("/search", json={"query": "CUDA illegal memory access"})
            # 3. 占位探测词 → probe=2
            c.post("/search", json={"query": "test"})
            # 4. 范例 doc_id → probe=2
            c.get("/doc/github:vllm-project-vllm-ascend:issue:13042")
            # 5. 真实查询（含中文领域问题）→ probe=0
            c.post("/search", json={"query": "Atlas 300I Duo 卡片温度过高告警如何处理"})
        rows = self._probes(d)
        by_norm = {r[1]: r[2] for r in rows if r[0] == "/search"}
        self.assertEqual(by_norm.get("真实故障排查问题"), 1)
        self.assertEqual(by_norm.get("cudaillegalmemoryaccess"), 2)
        self.assertEqual(by_norm.get("test"), 2)
        self.assertEqual(by_norm.get("atlas300iduo卡片温度过高告警如何处理"), 0)
        doc_rows = [r for r in rows if r[0].startswith("/doc/")]
        self.assertTrue(doc_rows)
        self.assertEqual(doc_rows[0][2], 2)

    def test_exemplar_words_match_skill_md_examples(self):
        """启发式范例词集合与 SKILL.md 示例对齐（防文档改示例后词表失同步）。

        只对齐"用法"各节的示例命令；"探索/验证请求约定"节中的示例是对比演示
        （真实查询示例故意非范例词），跳过。
        """
        import re as _re

        from pathlib import Path
        skill = (Path(__file__).resolve().parent.parent
                 / "skills" / "vllm-kb" / "SKILL.md").read_text(encoding="utf-8")
        # 跳过"探索/验证请求约定"节（该节示例是 --probe 用法对比，非用法示例）
        probe_section = skill.find("**探索/验证请求约定")
        next_heading = skill.find("\n## ", probe_section) if probe_section >= 0 else -1
        usable = skill if probe_section < 0 else skill[:probe_section] + skill[next_heading:]
        examples = set()
        for m in _re.finditer(
                r'client\.py\s+(?:--probe\s+)?(?:search|signature|title)\s+"([^"]+)"', usable):
            examples.add(m.group(1))
        for m in _re.finditer(r'client\.py\s+(?:code|diff)\s+"([^"]+)"', usable):
            examples.add(m.group(1))
        self.assertTrue(examples, "未提取到 SKILL.md 示例——正则与文档格式失配")
        from vllm_kb.telemetry import _query_normalized
        from vllm_kb import telemetry as tm
        for ex in examples:
            norm = _query_normalized(ex)
            self.assertTrue(
                norm in tm._PROBE_EXEMPLARS or norm in tm._PROBE_PLACEHOLDERS,
                f"SKILL.md 示例 {ex!r}（正规化 {norm!r}）未收录范例词表")

    def test_legacy_db_migration_adds_probe_column(self):
        """旧库（无 probe 列）→ TelemetryStore 初始化自动补列，旧行 probe=0。"""
        import tempfile

        from vllm_kb.telemetry import TelemetryStore
        d = tempfile.mkdtemp()
        self._tmpdirs.append(d)
        db = os.path.join(d, "legacy.sqlite3")
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE query_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            client_ip TEXT, ts TEXT NOT NULL, endpoint TEXT NOT NULL, method TEXT,
            query_hash TEXT, query_normalized TEXT, signature_hash TEXT,
            signature_entities TEXT, signature_text TEXT, result_doc_ids TEXT,
            result_count INTEGER, component TEXT, target_version TEXT, repo TEXT)""")
        conn.execute("INSERT INTO query_events (session_id, ts, endpoint, query_normalized) "
                     "VALUES ('s1', '2026-01-01', '/search', 'oldquery')")
        conn.commit()
        conn.close()

        store = TelemetryStore(Path(db))  # 初始化触发迁移
        store.record(session_id="s2", ts="2026-01-02", endpoint="/search",
                     query_normalized="x", probe=2)
        conn = sqlite3.connect(db)
        probes = dict(conn.execute("SELECT query_normalized, probe FROM query_events").fetchall())
        conn.close()
        self.assertEqual(probes["oldquery"], 0)  # 旧行默认 0（视为真实）
        self.assertEqual(probes["x"], 2)


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


@unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi"),
    "fastapi 未安装",
)
class TestClientProbeHeader(unittest.TestCase):
    """client.py 透传 VLLM_KB_PROBE=1 → X-VLLM-KB-Probe header（--probe flag 在 main 设该 env）。"""

    def _client_mod(self, name):
        import importlib.util
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            name, PROJECT_ROOT / "skills" / "vllm-kb" / "client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_probe_env_forwarded(self):
        client = self._client_mod("client_probe1")
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

        old = os.environ.pop("VLLM_KB_PROBE", None)
        os.environ["VLLM_KB_PROBE"] = "1"
        try:
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                client._get("http://x:8000", "/health")
        finally:
            if old is None:
                os.environ.pop("VLLM_KB_PROBE", None)
            else:
                os.environ["VLLM_KB_PROBE"] = old
        hdrs = {k.lower(): v for k, v in captured["headers"].items()}
        self.assertEqual(hdrs.get("x-vllm-kb-probe"), "1")

    def test_no_probe_env_no_header(self):
        client = self._client_mod("client_probe2")
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

        old = os.environ.pop("VLLM_KB_PROBE", None)
        try:
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                client._get("http://x:8000", "/health")
        finally:
            if old is not None:
                os.environ["VLLM_KB_PROBE"] = old
        hdrs = {k.lower(): v for k, v in captured["headers"].items()}
        self.assertNotIn("x-vllm-kb-probe", hdrs)


if __name__ == "__main__":
    unittest.main()
