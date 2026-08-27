"""标签 API 测试：能力目录 / 标签检索 / 问题→标签匹配 / /doc 出口白名单 / search tags 透传与过滤。"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig
from vllm_kb.embed import EmbeddingClient
from vllm_kb.ingest import ingest_docs
from vllm_kb.models import KbDocument
from vllm_kb.vectorstore import PythonVectorStore


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
            {"name": "命令参考", "tier": "purpose"},
        ]},
        "storage": {
            "vector_backend": "python",
            "lancedb_path": "data/lancedb",
            "sqlite_path": "data/kb.sqlite3",
            "canonical_file": "data/raw/canonical.jsonl",
            "review_path": "data/review.sqlite3",
        },
    }), encoding="utf-8")
    return AppConfig.load(str(cfg_path), require_keys=False)


DOCS = [
    KbDocument(
        source_type="doc_pdf",
        source_id="pdf:guide",
        url="",
        title="HCCL 超时排查指南",
        body="hccn_tool 排查命令。错误码 107020 memory allocation failed。",
        tags=["HCCL", "超时排查"],
        # 旧数据形态：asset 含 path——验证 /doc 出口白名单剥离（纵深防御）
        extra={"verification": "expert", "kind": "manual",
               "asset": {"asset_id": "abc123", "path": "assets/pdf/guide.pdf"}},
    ),
    KbDocument(
        source_type="doc_markdown",
        source_id="md:case",
        url="",
        title="网络超时案例",
        body="节点间通信超时，检查 HCCL 配置。",
        tags=["超时排查"],
        extra={"verification": "unverified"},
    ),
]


class TestTagsApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = build_env(self.root)
        store = PythonVectorStore(self.cfg.resolve("data/vec.json"))
        ingest_docs(self.cfg, DOCS, EmbeddingClient(self.cfg.embedding), store)
        from fastapi.testclient import TestClient

        from vllm_kb.api import create_app

        self.client = TestClient(create_app(str(self.root / "config.json")))

    def tearDown(self):
        self.client.close()
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def test_tags_catalog(self):
        r = self.client.get("/tags")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        domain = {x["name"]: x["docs"] for x in d["groups"]["domain"]}
        purpose = {x["name"]: x["docs"] for x in d["groups"]["purpose"]}
        self.assertEqual(domain["HCCL"], 1)
        self.assertEqual(purpose["超时排查"], 2)
        # registry-only 标签（docs=0）不在能力目录出现（避免噪音）
        self.assertNotIn("命令参考", purpose)
        self.assertEqual(d["total_tags"], 2)

    def test_tags_docs(self):
        r = self.client.get("/tags/超时排查/docs")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        ids = {x["doc_id"] for x in d["docs"]}
        self.assertEqual(ids, {"pdf:guide", "md:case"})
        r2 = self.client.get("/tags/HCCL/docs")
        self.assertEqual({x["doc_id"] for x in r2.json()["docs"]}, {"pdf:guide"})

    def test_tags_match(self):
        """问题→标签匹配：domain 过滤 + purpose 能力提示 + 文档线索。"""
        r = self.client.post("/tags/match", json={"text": "vllm-ascend HCCL 超时排查"})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        names = {m["name"]: m["tier"] for m in d["matched"]}
        self.assertEqual(names["HCCL"], "domain")
        self.assertEqual(names["超时排查"], "purpose")
        hccl = next(m for m in d["matched"] if m["name"] == "HCCL")
        self.assertEqual(hccl["top"][0]["doc_id"], "pdf:guide")
        # 无命中
        r2 = self.client.post("/tags/match", json={"text": "totally unrelated keyword"})
        self.assertEqual(r2.json()["count"], 0)

    def test_doc_extra_sanitized(self):
        """/doc 返回的 extra 经白名单清理：不暴露 asset.path（纵深防御，历史数据也剥离）。"""
        r = self.client.get("/doc/pdf:guide")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["tags"], ["HCCL", "超时排查"])
        self.assertEqual(d["extra"]["verification"], "expert")
        self.assertNotIn("path", json.dumps(d["extra"]))
        self.assertNotIn("asset", d["extra"])
        self.assertNotIn("assets/pdf", d["body"])

    def test_search_has_tags_and_filter(self):
        # FTS 命中（body 含 "memory"）
        r = self.client.post("/search", json={"query": "memory", "top_k": 5})
        self.assertEqual(r.status_code, 200)
        results = r.json()["results"]
        self.assertTrue(results)
        self.assertIn("tags", results[0])
        self.assertEqual(results[0]["doc_id"], "pdf:guide")
        # filters.tags：含 HCCL → 保留；不存在的标签 → 空
        r2 = self.client.post("/search", json={"query": "memory", "filters": {"tags": ["HCCL"]}})
        docs = {x["doc_id"] for x in r2.json()["results"]}
        self.assertEqual(docs, {"pdf:guide"})
        r3 = self.client.post("/search", json={"query": "memory", "filters": {"tags": ["不存在"]}})
        self.assertEqual(r3.json()["results"], [])


if __name__ == "__main__":
    unittest.main()
