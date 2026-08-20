"""serve_api 的 /graph/* 端点自动化测试（TestClient）+ /code/search 新参数 + /search verification 透传。

此前 /graph/* 只做过手工 client 验证。用临时数据根 + 小图 + 最小 kb 覆盖：
stats / chain / fixes / sig / doc / 未建图 503；code 的 path/per_version；search 的 verification。
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from vllm_kb.config import AppConfig
from vllm_kb.embed import EmbeddingClient
from vllm_kb.graph import GraphBuilder
from vllm_kb.ingest import ingest_docs
from vllm_kb.models import KbDocument, VersionSpan
from vllm_kb.vectorstore import PythonVectorStore

# 与 test_graph_build 同构的小样本：issue 10700 被 PR 12885 修复 → v0.23.0
CANONICAL = [
    {
        "source_type": "github_issue",
        "source_id": "github:vllm-project-vllm-ascend:issue:10700",
        "url": "https://github.com/vllm-project/vllm-ascend/issues/10700",
        "title": "GLM5.1 崩溃",
        "body": "halMemCreate failed drvRetCode=6, kernel_name=DispatchFFNCombine. fixed by #12885",
        "status": "open",
        "created_at": "2026-07-01T00:00:00Z",
        "extra": {"repo": "vllm-project/vllm-ascend", "github_number": 10700, "merged_at": None},
        "version_span": {"min": "0.23.0rc1"},
    },
    {
        "source_type": "github_pr",
        "source_id": "github:vllm-project-vllm-ascend:pr:12885",
        "url": "https://github.com/vllm-project/vllm-ascend/pull/12885",
        "title": "Fix GLM5.1 P2P hang",
        "body": "Fixes #10700. Also aclnnMoeDistributeDispatchV4 failed 561000.",
        "status": "closed",
        "created_at": "2026-07-20T00:00:00Z",
        "resolved_at": "2026-07-31T00:00:00Z",
        "extra": {"repo": "vllm-project/vllm-ascend", "github_number": 12885,
                  "merged": True, "merged_at": "2026-07-31T00:00:00Z"},
        "version_span": {"max": "0.23.0"},
    },
]

CAL = [
    {"tag": "v0.23.0rc1", "date": "2026-07-19T13:55:17Z", "prerelease": True, "kind": "rc"},
    {"tag": "v0.23.0", "date": "2026-08-16T22:18:14Z", "prerelease": False, "kind": "release"},
]


def build_env(tmp: Path) -> AppConfig:
    data_root = tmp / "data_root"
    data_root.mkdir()
    os.environ["VLLM_KB_DATA_ROOT"] = str(data_root)
    # 版本日历（供 MERGED_IN 映射）
    cal_dir = data_root / "compatibility"
    cal_dir.mkdir(parents=True)
    (cal_dir / "release_calendar.vllm-project-vllm-ascend.json").write_text(
        json.dumps({"repo": "vllm-project/vllm-ascend", "releases": CAL}), encoding="utf-8")
    (cal_dir / "release_calendar.json").write_text(
        json.dumps({"repo": "vllm-project/vllm", "releases": []}), encoding="utf-8")
    cfg_path = tmp / "config.json"
    cfg_path.write_text(json.dumps({
        "embedding": {"provider": "echo", "dimensions": 64},
        "sources": [{"id": "vllm-ascend", "type": "github", "repo": "vllm-project/vllm-ascend",
                     "token_env": "GITHUB_TOKEN", "enabled": True}],
        "storage": {
            "vector_backend": "python",
            "lancedb_path": "data/lancedb",
            "sqlite_path": "data/kb.sqlite3",
            "canonical_file": "data/raw/canonical.jsonl",
            "code_root": "data/code",
            "graph_path": "data/graph",
        },
        "code": {"repo": "vllm-project/vllm-ascend", "versions": []},
    }), encoding="utf-8")
    return AppConfig.load(str(cfg_path), require_keys=False)


def ingest_kb(cfg: AppConfig) -> None:
    docs = []
    for c in CANONICAL:
        extra = dict(c["extra"])
        docs.append(KbDocument(
            source_type=c["source_type"], source_id=c["source_id"], url=c["url"],
            title=c["title"], body=c["body"], status=c["status"],
            created_at=c["created_at"], resolved_at=c.get("resolved_at"),
            version_span=VersionSpan(min=(c.get("version_span") or {}).get("min"),
                                     max=(c.get("version_span") or {}).get("max")),
            extra=extra | {"verification": "expert"},
        ))
    store = PythonVectorStore(cfg.resolve("data/vec_py.json"))
    ingest_docs(cfg, docs, EmbeddingClient(cfg.embedding), store)


def write_canonical(cfg: AppConfig) -> None:
    p = cfg.resolve(cfg.storage.canonical_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for c in CANONICAL:
            f.write(json.dumps(c, ensure_ascii=True) + "\n")


class GraphEndpointsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = build_env(self.root)
        ingest_kb(self.cfg)
        write_canonical(self.cfg)
        # 建小图（复用 GraphBuilder + 版本日历）
        builder = GraphBuilder(self.cfg.resolve("data/graph"))
        builder.create_schema()
        builder.build_from_canonical(self.cfg.resolve("data/raw/canonical.jsonl"))
        builder.close()
        from vllm_kb.api import create_app

        self.client = TestClient(create_app(str(self.root / "config.json")))

    def tearDown(self):
        self.client.close()
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def test_graph_stats(self):
        r = self.client.get("/graph/stats")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertTrue(d["built"])
        self.assertEqual(d["nodes"]["Issue"], 1)
        self.assertEqual(d["nodes"]["PR"], 1)
        self.assertEqual(d["rels"]["FIXES"], 1)

    def test_graph_chain(self):
        r = self.client.get("/graph/chain",
                            params={"doc": "github:vllm-project-vllm-ascend:issue:10700"})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertTrue(d["found"])
        self.assertEqual(d["fixes"][0]["pr_id"], "github:vllm-project-vllm-ascend:pr:12885")
        self.assertEqual(d["fixes"][0]["release_tag"], "v0.23.0")
        self.assertTrue(d["released"])

    def test_graph_chain_not_found(self):
        r = self.client.get("/graph/chain", params={"doc": "github:vllm-project-vllm:issue:99999"})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["found"])

    def test_graph_fixes(self):
        r = self.client.get("/graph/fixes",
                            params={"doc": "github:vllm-project-vllm-ascend:pr:12885"})
        d = r.json()
        self.assertTrue(d["found"])
        self.assertEqual(d["fixes"][0]["issue_id"], "github:vllm-project-vllm-ascend:issue:10700")
        self.assertEqual([x["tag"] for x in d["releases"]], ["v0.23.0"])

    def test_graph_sig(self):
        r = self.client.get("/graph/sig", params={"sig": "dispatchffncombine"})
        d = r.json()
        self.assertEqual(d["entity_type"], "operator")
        self.assertGreater(d["count"], 0)
        r2 = self.client.get("/graph/sig", params={"sig": "561000"})
        self.assertEqual(r2.json()["entity_type"], "error_code")

    def test_graph_doc_neighbors(self):
        r = self.client.get("/graph/doc",
                            params={"doc": "github:vllm-project-vllm-ascend:issue:10700"})
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.json()["mentions"]), 0)

    def test_code_search_not_built_503(self):
        # 未构建代码快照时 /code/search 应 503 提示（修复：不再 500 崩溃）
        r = self.client.post("/code/search", json={
            "keyword": "dispatch", "path": "worker/model_runner_v1.py", "per_version": True,
        })
        self.assertEqual(r.status_code, 503)
        self.assertIn("code", r.json()["detail"].lower())

    def test_search_verification_field(self):
        r = self.client.post("/search", json={"query": "GLM5.1 崩溃"})
        self.assertEqual(r.status_code, 200)
        results = r.json().get("results") or []
        self.assertTrue(results)
        self.assertIn("verification", results[0])
        # 检索结果透传 verification（库中 doc 标了 expert）
        self.assertEqual(results[0]["verification"], "expert")


class GraphNotBuiltTest(unittest.TestCase):
    """图未构建时：stats 返回 built=False，链路端点 503（不崩溃）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = build_env(self.root)  # 不建图
        from vllm_kb.api import create_app

        self.client = TestClient(create_app(str(self.root / "config.json")))

    def tearDown(self):
        self.client.close()
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def test_stats_not_built(self):
        d = self.client.get("/graph/stats").json()
        self.assertFalse(d["built"])

    def test_chain_503_when_not_built(self):
        r = self.client.get("/graph/chain", params={"doc": "x:1"})
        self.assertEqual(r.status_code, 503)


if __name__ == "__main__":
    unittest.main()
