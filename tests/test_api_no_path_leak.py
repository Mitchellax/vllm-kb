"""端到端安全审计：全部只读端点响应**无服务器路径、无枚举形态**。

- 造含"脏"数据的库（extra.asset.path / evidence.path 残留 + 干净正文），
  遍历 /health /stats /components /doc /search /signature-search /title /version
  /companion /matrix /tags /tags/{tag}/docs /tags/match /graph/* /code/*，
  断言响应 JSON 不含路径特征（assets/、data/{imports,parsed,...}、盘符、反斜杠形态）；
- 断言能力目录为"标签名+计数"的非枚举形态（/tags 无文档 id 列表、/tags/{tag}/docs 只含该标签文档）；
- 纵深防御：即使存量库残留历史路径字段（asset.path），API 出口白名单也会剥离。
"""
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig
from vllm_kb.embed import EmbeddingClient
from vllm_kb.ingest import ingest_docs
from vllm_kb.models import KbDocument
from vllm_kb.vectorstore import PythonVectorStore

# 服务器路径形态（GitHub URL 等社区链接是允许的，不属于服务器路径）
_PATH_PATTERNS = [
    re.compile(r"assets[/\\]"),
    re.compile(r"data[/\\](?:imports|parsed|raw|code|compatibility|assets)"),
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"\\assets\\"),
    re.compile(r"\\parsed\\"),
]
# 内部数据形态（后置脱敏：出口不得出现内部 IP/路径原文）
_INTERNAL_PATTERNS = [
    re.compile(r"10\.0\.0\.5"),
    re.compile(r"/home/user/"),
    re.compile(r"192\.168\."),
]


def _contains_path(text: str) -> str | None:
    for pat in _PATH_PATTERNS:
        m = pat.search(text or "")
        if m:
            return m.group(0)
    return None


def _contains_internal(text: str) -> str | None:
    for pat in _INTERNAL_PATTERNS:
        m = pat.search(text or "")
        if m:
            return m.group(0)
    return None


def build_env(tmp: Path) -> AppConfig:
    data_root = tmp / "data_root"
    data_root.mkdir()
    os.environ["VLLM_KB_DATA_ROOT"] = str(data_root)
    cfg_path = tmp / "config.json"
    cfg_path.write_text(json.dumps({
        "embedding": {"provider": "echo", "dimensions": 64},
        "tags": {"registry": [
            {"name": "HCCL", "tier": "domain"},
            {"name": "超时排查", "tier": "purpose"},
        ]},
        "storage": {
            "vector_backend": "python",
            "lancedb_path": "data/lancedb",
            "sqlite_path": "data/kb.sqlite3",
            "canonical_file": "data/raw/canonical.jsonl",
            "code_root": "data/code",
            "graph_path": "data/graph",
            "review_path": "data/review.sqlite3",
        },
        "code": {"repo": "vllm-project/vllm-ascend", "versions": []},
    }), encoding="utf-8")
    return AppConfig.load(str(cfg_path), require_keys=False)


# 脏数据：extra 含历史路径字段（模拟旧库残留），正文含内部 IP/路径（后置脱敏验证出口）
DIRTY_DOCS = [
    KbDocument(
        source_type="doc_pdf",
        source_id="pdf:guide",
        url="",
        title="HCCL 超时排查指南",
        body=("命令格式\nhccn_tool [-i %d] -bandwidth -g\n"
              "错误码 107020 memory allocation failed\n"
              "节点 10.0.0.5 超时，日志 /home/user/logs/oom.log，默认 /var/log/npu/"),
        tags=["HCCL", "超时排查"],
        extra={
            "verification": "expert",
            "kind": "manual",
            # 历史残留：asset.path / evidence[].path——出口必须剥离
            "asset": {"asset_id": "abc123", "path": "assets/pdf/guide.pdf",
                      "sha256": "abc123"},
            "evidence": [{"kind": "local", "path": "assets/images/topo.png",
                          "asset_id": "def456", "sha256": "def456"}],
            "tag_candidates": [{"name": "命令参考", "tier": "purpose"}],
        },
    ),
    KbDocument(
        source_type="doc_markdown",
        source_id="md:wiki",
        url="",
        title="网络超时案例",
        body="节点间通信超时，检查 192.168.1.100 配置。[图片:拓扑]",
        tags=["超时排查"],
        extra={"verification": "unverified"},
    ),
]


class TestNoPathLeak(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = build_env(self.root)
        store = PythonVectorStore(self.cfg.resolve("data/vec.json"))
        ingest_docs(self.cfg, DIRTY_DOCS, EmbeddingClient(self.cfg.embedding), store)
        from fastapi.testclient import TestClient

        from vllm_kb.api import create_app

        self.client = TestClient(create_app(str(self.root / "config.json")))

    def tearDown(self):
        self.client.close()
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def _assert_no_path(self, payload, where: str):
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        hit = _contains_path(text)
        if hit is not None:
            pos = max(0, text.find(hit) - 60)
            self.fail(f"{where} 泄漏服务器路径 {hit!r}: ...{text[pos:pos + 120]}...")

    def _assert_no_internal(self, payload, where: str):
        """后置脱敏：出口响应不得出现内部 IP/路径原文。"""
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        hit = _contains_internal(text)
        if hit is not None:
            pos = max(0, text.find(hit) - 60)
            self.fail(f"{where} 泄漏内部数据 {hit!r}: ...{text[pos:pos + 120]}...")

    def test_all_readonly_endpoints_no_path(self):
        """遍历全部只读端点，响应无服务器路径（图/code 未建时 503 提示也不泄漏）。"""
        endpoints = [
            ("GET", "/health", {}),
            ("GET", "/stats", {}),
            ("GET", "/components", {}),
            ("GET", "/doc/pdf:guide", {}),
            ("GET", "/doc/md:wiki", {}),
            ("POST", "/search", {"json": {"query": "hccn_tool", "top_k": 5}}),
            ("POST", "/search", {"json": {"query": "超时", "filters": {"tags": ["HCCL"]}}}),
            ("POST", "/signature-search", {"json": {"text": "hccn_tool 错误码 107020"}}),
            ("GET", "/title", {"params": {"keyword": "超时"}}),
            ("GET", "/version", {"params": {"version": "0.26.0"}}),
            ("GET", "/companion", {"params": {"component": "vllm-ascend", "version": "0.26.0"}}),
            ("GET", "/matrix", {}),
            ("GET", "/tags", {}),
            ("GET", "/tags/HCCL/docs", {}),
            ("GET", "/tags/超时排查/docs", {}),
            ("POST", "/tags/match", {"json": {"text": "vllm-ascend HCCL 超时"}}),
            ("GET", "/graph/stats", {}),
            ("GET", "/graph/chain", {"params": {"doc": "github:vllm-project-vllm:issue:1"}}),
            ("GET", "/graph/fixes", {"params": {"doc": "github:vllm-project-vllm:pr:1"}}),
            ("GET", "/graph/sig", {"params": {"sig": "dispatch"}}),
            ("GET", "/graph/doc", {"params": {"doc": "pdf:guide"}}),
            ("GET", "/graph/tags", {"params": {"tag": "HCCL"}}),
            ("GET", "/code/versions", {}),
            ("POST", "/code/search", {"json": {"keyword": "dispatch"}}),
            ("GET", "/code/file", {"params": {"version": "v0.23.0rc1", "path": "x.py"}}),
            ("GET", "/code/diff", {"params": {"version1": "v0.22.1rc1", "version2": "v0.23.0rc1",
                                              "path": "x.py"}}),
        ]
        for method, path, kwargs in endpoints:
            with self.subTest(endpoint=f"{method} {path}"):
                r = self.client.request(method, path, **kwargs)
                self.assertIn(r.status_code, (200, 404, 503), f"{path} 异常状态 {r.status_code}")
                try:
                    payload = r.json()
                except Exception:
                    self.fail(f"{path} 返回非 JSON")
                self._assert_no_path(payload, f"{method} {path}")
                self._assert_no_internal(payload, f"{method} {path}")
                if r.status_code == 200:
                    # 200 的 JSON 里不允许出现任何路径特征（404/503 提示也不应泄漏，已统一检查）
                    pass

    def test_doc_sanitizes_dirty_extra(self):
        """脏 extra（asset.path / evidence.path 残留）经 /doc 出口白名单剥离。"""
        r = self.client.get("/doc/pdf:guide")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertNotIn("asset", d["extra"])          # asset 整体剥离
        ev = d["extra"].get("evidence", [])
        for e in ev:
            self.assertNotIn("path", e)                # evidence 无 path
        self._assert_no_path(d, "/doc/pdf:guide")
        # 检索字段保留（tags/verification 正常）
        self.assertEqual(d["tags"], ["HCCL", "超时排查"])
        self.assertEqual(d["extra"]["verification"], "expert")
        self.assertEqual(d["extra"]["kind"], "manual")

    def test_search_signature_results_no_path(self):
        """检索类端点（search/signature/title）响应无路径。"""
        for method, path, kwargs in [
            ("POST", "/search", {"json": {"query": "hccn_tool", "top_k": 5}}),
            ("POST", "/signature-search", {"json": {"text": "hccn_tool 错误码"}}),
            ("GET", "/title", {"params": {"keyword": "超时"}}),
        ]:
            r = self.client.request(method, path, **kwargs)
            self._assert_no_path(r.json(), f"{method} {path}")

    def test_tags_endpoints_non_enumeration(self):
        """能力目录非枚举形态：/tags 只返回标签名+计数，/tags/{tag}/docs 只含该标签文档。"""
        r = self.client.get("/tags").json()
        # 目录项只有 name/docs，无 doc_id 列表（不暴露文档枚举）
        for group in ("domain", "purpose"):
            for item in r["groups"].get(group, []):
                self.assertEqual(set(item.keys()), {"name", "docs"})
        self.assertNotIn("doc_id", json.dumps(r))
        # 标签检索只返回该标签的文档
        r2 = self.client.get("/tags/HCCL/docs").json()
        self.assertEqual({d["doc_id"] for d in r2["docs"]}, {"pdf:guide"})
        r3 = self.client.get("/tags/超时排查/docs").json()
        self.assertEqual({d["doc_id"] for d in r3["docs"]}, {"pdf:guide", "md:wiki"})
        for d in r2["docs"] + r3["docs"]:
            self.assertNotIn("path", json.dumps(d))

    def test_context_no_path(self):
        """问题→标签匹配响应无路径（文档线索只含标题/文档id/验证状态）。"""
        r = self.client.post("/tags/match", json={"text": "vllm-ascend HCCL 超时"})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertTrue(d["matched"])
        self._assert_no_path(d, "/tags/match")
        for m in d["matched"]:
            self.assertEqual(set(m.keys()) & {"path", "rel_path"}, set())

    def test_outbound_sanitization_but_raw_stored(self):
        """后置脱敏：库中存原文（可原文检索），出口（/doc、/search snippet）统一脱敏。"""
        # 库中原文（内部检索用）
        import sqlite3

        conn = sqlite3.connect(f"file:{self.cfg.resolve('data/kb.sqlite3').as_posix()}?mode=ro",
                               uri=True)
        try:
            row = conn.execute(
                "SELECT text FROM chunks_fts WHERE doc_id='pdf:guide' LIMIT 1").fetchone()
        finally:
            conn.close()
        self.assertIn("10.0.0.5", row[0])  # 原文入库
        # 出口：/doc body 脱敏（<IP>/<PATH>，保留默认路径）
        r = self.client.get("/doc/pdf:guide")
        d = r.json()
        self.assertNotIn("10.0.0.5", d["body"])
        self.assertIn("<IP>", d["body"])
        self.assertNotIn("/home/user/logs/oom.log", d["body"])
        self.assertIn("<PATH>", d["body"])
        self.assertIn("/var/log/npu/", d["body"])  # 默认路径保留（诊断价值）
        # 出口：/search snippet 脱敏
        r2 = self.client.post("/search", json={"query": "hccn_tool", "top_k": 5})
        blob = json.dumps(r2.json(), ensure_ascii=False)
        self.assertNotIn("10.0.0.5", blob)
        self.assertNotIn("/home/user/logs/", blob)


if __name__ == "__main__":
    unittest.main()
