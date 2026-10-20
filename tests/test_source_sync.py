"""mirror 来源对账（deletion_policy=mirror，本地原始数据优先）测试。

设计：docs/DESIGN-audited-git-source.md。审计驱动 git 来源：PR 合入落盘（增/改走
现有 upsert），回撤 PR 删文件 → 快照对账 + 宽限 → 自动 mark_deleted（备份+四层删+
墓碑+待实际删除审核项）；文件重新出现 → 解除墓碑自动恢复入库资格。
"""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")


class SourceSyncTest(unittest.TestCase):
    """mirror 对账内核：宽限期、自动回撤、复活解除墓碑、快照损坏首轮、配置校验。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data_root")
        self.cfg_path = self.root / "config.json"
        self.cfg_path.write_text(json.dumps({
            "embedding": {"provider": "echo", "dimensions": 64},
            "storage": {
                "vector_backend": "python",
                "lancedb_path": "data/lancedb",
                "sqlite_path": "data/kb.sqlite3",
                "canonical_file": "data/raw/canonical.jsonl",
                "review_path": "data/review.sqlite3",
            },
            "sources": [{"id": "audited", "type": "markdown",
                         "path": "data/imports/audited",
                         "deletion_policy": "mirror"}],
        }, ensure_ascii=False), encoding="utf-8")
        self.cfg = __import__("vllm_kb.config", fromlist=["AppConfig"]).AppConfig.load(
            str(self.cfg_path), require_keys=False)
        from vllm_kb.review import ReviewStore, default_review_path

        self.store = ReviewStore(default_review_path(self.cfg))
        self.rdb = default_review_path(self.cfg)
        self.kb = self.cfg.resolve(self.cfg.storage.sqlite_path)

    def tearDown(self):
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def _ingest(self, source_id):
        from vllm_kb.embed import EmbeddingClient
        from vllm_kb.ingest import ingest_docs
        from vllm_kb.models import KbDocument
        from vllm_kb.vectorstore import PythonVectorStore

        doc = KbDocument(source_type="doc_pdf", source_id=source_id, url="",
                         title=source_id, body="正文内容", tags=[], extra={})
        ingest_docs(self.cfg, [doc], EmbeddingClient(self.cfg.embedding),
                    PythonVectorStore(self.cfg.resolve("data/vec.json")))

    def _row(self, sid):
        conn = sqlite3.connect(self.kb)
        try:
            return conn.execute(
                "SELECT source_id FROM docs WHERE source_id=?", (sid,)).fetchone()
        finally:
            conn.close()

    def _reconcile(self, discovered, sid_prefix="md:"):
        from vllm_kb.source_sync import reconcile_mirror_source

        return reconcile_mirror_source(self.cfg, "audited", discovered, self.kb,
                                       self.rdb, sid_prefix=sid_prefix)

    def test_grace_then_withdraw_then_reappear(self):
        """首轮建快照不删除；缺席 1 次宽限；连续 2 次缺席 → 自动回撤（四层删+墓碑+
        source_withdraw 审核项 deleted）；文件回来 → 解除墓碑可重新入库。"""
        from vllm_kb.embed import EmbeddingClient
        from vllm_kb.ingest import ingest_docs
        from vllm_kb.models import KbDocument
        from vllm_kb.review import load_tombstones
        from vllm_kb.vectorstore import PythonVectorStore

        self._ingest("md:doc1")
        self._ingest("md:doc2")
        r0 = self._reconcile(["md:doc1", "md:doc2"])
        self.assertTrue(r0["first_round"])
        self.assertEqual(r0["withdrawn"], [])
        # 第 2 轮：doc1 缺席一次 → 宽限，kb 行仍在
        r1 = self._reconcile(["md:doc2"])
        self.assertEqual(r1["grace"], ["md:doc1"])
        self.assertIsNotNone(self._row("md:doc1"))
        # 第 3 轮：连续第 2 次缺席 → 自动回撤
        r2 = self._reconcile(["md:doc2"])
        self.assertEqual(r2["withdrawn"], ["md:doc1"])
        self.assertIsNone(self._row("md:doc1"), "回撤后 kb 行应删除")
        self.assertIsNotNone(self._row("md:doc2"), "在场文档不受影响")
        self.assertIn("md:doc1", load_tombstones(self.rdb))
        item = [i for i in self.store.list_items(category="source_withdraw")
                if i["item_ref"] == "md:doc1"][0]
        self.assertEqual(item["status"], "deleted")
        # 第 4 轮：doc1 重新出现（PR 重新合入）→ 解除墓碑 → ingest 正常入库
        r3 = self._reconcile(["md:doc1", "md:doc2"])
        self.assertEqual(r3["untombstoned"], ["md:doc1"])
        self.assertNotIn("md:doc1", load_tombstones(self.rdb))
        doc = KbDocument(source_type="doc_pdf", source_id="md:doc1", url="",
                         title="md:doc1", body="重新合入的正文", tags=[], extra={})
        ingest_docs(self.cfg, [doc], EmbeddingClient(self.cfg.embedding),
                    PythonVectorStore(self.cfg.resolve("data/vec.json")))
        self.assertIsNotNone(self._row("md:doc1"), "解除墓碑后应可重新入库")

    def test_corrupt_snapshot_first_round_no_deletion(self):
        """快照损坏 → 视为首轮：不删除、重建有效快照（宁漏删不误删）。"""
        self._ingest("md:keepme")
        self._reconcile(["md:keepme"])  # 建立快照
        from vllm_kb.source_sync import _state_path

        sp = _state_path(self.cfg, "audited")
        sp.write_text("{不是json", encoding="utf-8")
        r = self._reconcile([])  # 全部缺席，但快照损坏 → 不删
        self.assertTrue(r["first_round"])
        self.assertEqual(r["withdrawn"], [])
        self.assertIsNotNone(self._row("md:keepme"))
        # 首轮全量对账：keepme 在库但缺席 → 计宽限 1（次轮再缺席即回撤，本轮不删）
        self.assertEqual(json.loads(sp.read_text(encoding="utf-8")), {"md:keepme": 1})

    def test_first_round_seeds_historically_missing(self):
        """开启 mirror 前源文件就已缺失的存量文档：首轮全量对账计入宽限，
        次轮再缺席即自动回撤（四层删+墓碑）——历史滞留自愈。"""
        from vllm_kb.review import load_tombstones

        self._ingest("md:live")
        self._ingest("md:ghost")
        r0 = self._reconcile(["md:live"])   # 首轮：ghost 在库但缺席 → 宽限
        self.assertTrue(r0["first_round"])
        self.assertEqual(r0["grace"], ["md:ghost"])
        self.assertIsNotNone(self._row("md:ghost"))
        r1 = self._reconcile(["md:live"])   # 次轮：缺席第 2 次 → 回撤
        self.assertEqual(r1["withdrawn"], ["md:ghost"])
        self.assertIsNone(self._row("md:ghost"))
        self.assertIn("md:ghost", load_tombstones(self.rdb))
        self.assertIsNotNone(self._row("md:live"))

    def test_first_round_scoped_to_kind_prefix(self):
        """首轮存量对账按 sid 前缀圈定——其他 kind 的文档不进对账范围。"""
        self._ingest("md:live")
        self._ingest("pdf:other")
        r0 = self._reconcile(["md:live"])
        self.assertEqual(r0["grace"], [])        # pdf:other 不在 md: 范围
        r1 = self._reconcile(["md:live"])
        self.assertEqual(r1["withdrawn"], [])
        self.assertIsNotNone(self._row("pdf:other"))

    def test_repeated_withdraw_idempotent(self):
        """同一 sid 反复回撤不炸（审核项幂等 + 墓碑 REPLACE）。"""
        self._ingest("md:once")
        self._reconcile(["md:once"])
        self._reconcile([])
        self._reconcile([])
        r3 = self._reconcile([])
        self.assertEqual(r3["withdrawn"], [])
        self.assertIsNone(self._row("md:once"))


class DeletionPolicyConfigTest(unittest.TestCase):
    """deletion_policy 配置校验：非法值告警并回落 keep；合法值保留。"""

    def test_invalid_policy_falls_back_to_keep(self):
        from vllm_kb.config import SourceCfg

        s = SourceCfg(id="x", type="markdown", deletion_policy="bogus")
        self.assertEqual(s.deletion_policy, "keep")
        self.assertEqual(SourceCfg(id="y", type="markdown").deletion_policy, "keep")
        self.assertEqual(SourceCfg(id="z", type="markdown",
                                   deletion_policy="mirror").deletion_policy, "mirror")
