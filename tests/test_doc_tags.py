"""文档级标签覆盖层（doc_tags）与资产注册表（asset_registry）测试。

覆盖：入库合并公式（(auto − excluded) ∪ manual）、auto_snapshot 回写、
审核页修改覆盖层后 docs.tags 立即同步、asset_registry 注册与查询。
"""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig
from vllm_kb.embed import EmbeddingClient
from vllm_kb.ingest import ingest_docs
from vllm_kb.models import KbDocument
from vllm_kb.review import (
    get_doc_tags_conn,
    list_assets,
    register_asset,
    set_doc_tags_conn,
    upsert_auto_snapshot_conn,
)
from vllm_kb.vectorstore import PythonVectorStore


class DocTagsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data_root")
        cfg_path = self.root / "config.json"
        cfg_path.write_text(json.dumps({
            "embedding": {"provider": "echo", "dimensions": 64},
            "storage": {
                "vector_backend": "python",
                "lancedb_path": "data/lancedb",
                "sqlite_path": "data/kb.sqlite3",
                "canonical_file": "data/raw/canonical.jsonl",
                "review_path": "data/review.sqlite3",
            },
        }), encoding="utf-8")
        self.cfg = AppConfig.load(str(cfg_path), require_keys=False)
        self.store = PythonVectorStore(self.cfg.resolve("data/vec.json"))
        self.embed = EmbeddingClient(self.cfg.embedding)
        self.kb = self.cfg.resolve("data/kb.sqlite3")

    def tearDown(self):
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def doc(self, source_id: str, tags: list[str], body: str = "正文内容") -> KbDocument:
        return KbDocument(source_type="doc_pdf", source_id=source_id, url="",
                          title=source_id, body=body, tags=tags)

    def _tags_of(self, source_id: str) -> list[str]:
        conn = sqlite3.connect(self.kb)
        try:
            row = conn.execute("SELECT tags FROM docs WHERE source_id=?", (source_id,)).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row and row[0] else []

    def test_ingest_merges_overlay(self):
        """入库时最终标签 = (自动 − 排除) ∪ 人工。"""
        ingest_docs(self.cfg, [self.doc("pdf:a", ["HCCL", "超时排查", "网络"])], self.embed, self.store)
        conn = sqlite3.connect(self.kb)
        try:
            set_doc_tags_conn(conn, "pdf:a", excluded=["超时排查"], manual=["命令参考"],
                              reviewer="tester")
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(self._tags_of("pdf:a"), ["HCCL", "网络", "命令参考"])

    def test_manual_priority_over_exclude(self):
        """同一标签既被排除又人工添加 → 人工优先（合并公式语义）。"""
        ingest_docs(self.cfg, [self.doc("pdf:a", ["HCCL", "超时排查"])], self.embed, self.store)
        conn = sqlite3.connect(self.kb)
        try:
            set_doc_tags_conn(conn, "pdf:a", excluded=["超时排查"], manual=["超时排查"],
                              reviewer="tester")
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(self._tags_of("pdf:a"), ["HCCL", "超时排查"])

    def test_auto_snapshot_refreshed_and_restore(self):
        """auto_snapshot 随入库刷新；恢复排除项后 docs.tags 立即更新（检索侧生效）。"""
        ingest_docs(self.cfg, [self.doc("pdf:a", ["HCCL", "超时排查"])], self.embed, self.store)
        conn = sqlite3.connect(self.kb)
        try:
            set_doc_tags_conn(conn, "pdf:a", excluded=["超时排查"], reviewer="tester")
            conn.commit()
            self.assertEqual(self._tags_of("pdf:a"), ["HCCL"])
            # 恢复：从 excluded 移除 → docs.tags 重新含超时排查
            set_doc_tags_conn(conn, "pdf:a", excluded=[], reviewer="tester")
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(self._tags_of("pdf:a"), ["HCCL", "超时排查"])
        g = get_doc_tags_conn(sqlite3.connect(self.kb), "pdf:a")
        self.assertEqual(g["excluded"], [])
        self.assertEqual(g["reviewer"], "tester")

    def test_ingest_refreshes_auto_snapshot(self):
        """重新入库后 auto_snapshot 为最新自动标签，覆盖层（排除/人工）不受影响。"""
        ingest_docs(self.cfg, [self.doc("pdf:a", ["HCCL", "超时排查"])], self.embed, self.store)
        conn = sqlite3.connect(self.kb)
        try:
            set_doc_tags_conn(conn, "pdf:a", excluded=["超时排查"], reviewer="t")
            conn.commit()
        finally:
            conn.close()
        # 自动标签变化（如提取规则更新）→ 重入库
        ingest_docs(self.cfg, [self.doc("pdf:a", ["HCCL", "超时排查", "网络"])], self.embed, self.store)
        g = get_doc_tags_conn(sqlite3.connect(self.kb), "pdf:a")
        self.assertEqual(g["auto_snapshot"], ["HCCL", "超时排查", "网络"])
        self.assertEqual(g["excluded"], ["超时排查"])  # 覆盖层保留
        # 最终标签 = (新自动 − 排除) ∪ 人工
        self.assertEqual(self._tags_of("pdf:a"), ["HCCL", "网络"])

    def test_upsert_auto_snapshot_conn(self):
        self.kb.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.kb)
        try:
            upsert_auto_snapshot_conn(conn, "x:1", ["A", "B"])
            conn.commit()
            g = get_doc_tags_conn(conn, "x:1")
            self.assertEqual(g["auto_snapshot"], ["A", "B"])
            upsert_auto_snapshot_conn(conn, "x:1", ["A", "C"])
            conn.commit()
            g = get_doc_tags_conn(conn, "x:1")
            self.assertEqual(g["auto_snapshot"], ["A", "C"])
        finally:
            conn.close()


class AssetRegistryTest(unittest.TestCase):
    def test_register_and_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "review.sqlite3"
            register_asset(db, "abc123", "assets/pdf/guide.pdf", sha256="abc123...", size=1024,
                           source_type="doc_pdf")
            register_asset(db, "abc123", "assets/pdf/guide.pdf", source_type="doc_pdf")  # 幂等 upsert
            register_asset(db, "def456", "assets/md/wiki.md", source_type="doc_markdown")
            assets = list_assets(db)
            self.assertEqual(assets["abc123"]["rel_path"], "assets/pdf/guide.pdf")
            self.assertEqual(assets["abc123"]["size"], 1024)
            self.assertEqual(assets["def456"]["source_type"], "doc_markdown")
            self.assertEqual(len(assets), 2)
        # 空库 → 空
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_assets(Path(tmp) / "none.sqlite3"), {})


if __name__ == "__main__":
    unittest.main()
