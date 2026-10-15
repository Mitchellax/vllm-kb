"""文档级验证状态覆盖层（doc_verification）：人工补标回写闭环。

这是"审核台点了认证但 w_rel 一直停在 0.5"的修复。核心不变量：
1. 覆盖层**优先于 canonical**——重新入库不能被 canonical 的 unverified 冲掉；
2. 认证后**立即**刷新 docs.extra.verification（查询期就是从这里取该值的），不必等下次构建；
3. 撤销（存疑）回到**覆盖前**的 canonical 值，而不是上次写入的覆盖值；
4. `--rebuild` 删库前先备份覆盖层（人工成果，canonical 里没有）；
5. 认证过的不再进补标队列；认证不触发重嵌（embed_hash 不受 extra 影响）。
"""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig
from vllm_kb.embed import EmbeddingClient
from vllm_kb.ingest import ingest_docs, rebuild
from vllm_kb.models import KbDocument, doc_to_json
from vllm_kb.review import (
    ReviewStore,
    apply_verification,
    clear_doc_verification_conn,
    default_review_path,
    get_doc_verification_conn,
    load_doc_verification_conn,
    review_verification,
    seed_verification_pending,
    set_doc_verification_conn,
)
from vllm_kb.vectorstore import PythonVectorStore


class DocVerificationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data_root")
        self.cfg_path = self.root / "config.json"
        self.cfg_path.write_text(json.dumps({
            "embedding": {"provider": "echo", "dimensions": 64},
            # min_similarity=0：echo 嵌入的相似度是噪声，别让阈值把结果滤掉
            "retrieval": {"min_similarity": 0.0},
            "storage": {
                "vector_backend": "python",
                "lancedb_path": "data/lancedb",
                "sqlite_path": "data/kb.sqlite3",
                "canonical_file": "data/raw/canonical.jsonl",
                "review_path": "data/review.sqlite3",
            },
        }), encoding="utf-8")
        self.cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        self.store = PythonVectorStore(self.cfg.resolve("data/vec.json"))
        self.embed = EmbeddingClient(self.cfg.embedding)
        self.kb = self.cfg.resolve("data/kb.sqlite3")
        self.review_store = ReviewStore(default_review_path(self.cfg))

    def tearDown(self):
        self.review_store.close()
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    # ---------- 夹具 ----------

    def doc(self, source_id: str = "md:case1", verification: str = "unverified",
            body: str = "HCCL 超时排查步骤") -> KbDocument:
        return KbDocument(source_type="doc_markdown", source_id=source_id, url="",
                          title=source_id, body=body,
                          extra={"verification": verification})

    def ingest(self, *docs) -> dict:
        return ingest_docs(self.cfg, list(docs or (self.doc(),)), self.embed, self.store)

    def verif(self, source_id: str = "md:case1") -> tuple[str, float]:
        conn = sqlite3.connect(self.kb)
        try:
            row = conn.execute("SELECT extra, reliability FROM docs WHERE source_id=?",
                               (source_id,)).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row, f"{source_id} 不在库中")
        return json.loads(row[0]).get("verification", ""), row[1]

    def item_id(self, category: str = "verification_pending") -> int:
        q = self.review_store.list_items(category=category)
        self.assertTrue(q, f"队列里没有 {category} 项")
        return q[0]["id"]

    # ---------- 覆盖层 CRUD ----------

    def test_set_get_load_clear(self):
        conn = sqlite3.connect(self.kb)
        try:
            self.assertEqual(load_doc_verification_conn(conn), {})
            set_doc_verification_conn(conn, "md:a", "tested", "alice", canonical="unverified")
            conn.commit()
            self.assertEqual(load_doc_verification_conn(conn), {"md:a": "tested"})
            g = get_doc_verification_conn(conn, "md:a")
            self.assertEqual((g["verification"], g["canonical_verification"], g["reviewer"]),
                             ("tested", "unverified", "alice"))
            # 二次覆盖**不改** canonical 记录（否则撤销会退到上次的覆盖值）
            set_doc_verification_conn(conn, "md:a", "expert", "bob")
            conn.commit()
            self.assertEqual(get_doc_verification_conn(conn, "md:a")["canonical_verification"],
                             "unverified")
            # 撤销返回覆盖前的 canonical 值，且删掉行
            self.assertEqual(clear_doc_verification_conn(conn, "md:a"), "unverified")
            conn.commit()
            self.assertEqual(load_doc_verification_conn(conn), {})
            # 无覆盖行时撤销 → None（调用方保持原值）
            self.assertIsNone(clear_doc_verification_conn(conn, "md:a"))
        finally:
            conn.close()

    def test_invalid_level_rejected(self):
        conn = sqlite3.connect(self.kb)
        try:
            for bad in ("bogus", "verified", "", None):
                with self.assertRaises(ValueError):
                    set_doc_verification_conn(conn, "md:a", bad)
        finally:
            conn.close()

    def test_load_ignores_corrupt_rows(self):
        """库里被手工塞了非法值 → 不参与合并（当没有覆盖层）。"""
        conn = sqlite3.connect(self.kb)
        try:
            set_doc_verification_conn(conn, "md:a", "tested")
            conn.execute("UPDATE doc_verification SET verification='乱写' WHERE source_id='md:a'")
            conn.commit()
            self.assertEqual(load_doc_verification_conn(conn), {})
        finally:
            conn.close()

    # ---------- 回写闭环 ----------

    def test_approve_writes_overlay_and_refreshes_docs(self):
        """认证 → 覆盖层 + docs 行**立即**更新（不等下次构建）。"""
        self.ingest()
        self.assertEqual(self.verif(), ("unverified", 0.5))
        seed_verification_pending(self.cfg, self.review_store)
        r = review_verification(self.cfg, self.review_store, self.item_id(), "approved", "tester")
        self.assertEqual(r["verification"], "tested")
        self.assertEqual(self.verif(), ("tested", 0.85))
        it = self.review_store.get_item(self.item_id())
        self.assertEqual(it["status"], "approved")
        self.assertEqual(it["result"]["verification"], "tested")

    def test_expert_level_supported(self):
        """result.verification=expert → 0.95（专家认证，比 tested 更高）。"""
        self.ingest()
        seed_verification_pending(self.cfg, self.review_store)
        review_verification(self.cfg, self.review_store, self.item_id(), "approved", "tester",
                            result={"verification": "expert"})
        self.assertEqual(self.verif(), ("expert", 0.95))

    def test_approve_rejects_bogus_level(self):
        self.ingest()
        seed_verification_pending(self.cfg, self.review_store)
        with self.assertRaises(ValueError):
            review_verification(self.cfg, self.review_store, self.item_id(), "approved", "t",
                                result={"verification": "unverified"})

    def test_overlay_survives_reingest(self):
        """**核心**：canonical 仍写 unverified，重新入库后仍是 tested（不被冲掉）。"""
        self.ingest()
        seed_verification_pending(self.cfg, self.review_store)
        review_verification(self.cfg, self.review_store, self.item_id(), "approved", "tester")
        # 再次入库同一份 canonical（verification=unverified）
        stats = self.ingest()
        self.assertEqual(self.verif(), ("tested", 0.85), "覆盖层被 canonical 冲掉了")
        # 且不重嵌（embed_hash 只覆盖 source_id+title+body，不受 extra 影响）
        self.assertEqual(stats["embedded"], 0, "认证不该触发重新嵌入")

    def test_approved_doc_not_requeued(self):
        """认证后生效值为 tested → seed 不再补单（审核台不再反复打扰）。"""
        self.ingest()
        self.assertEqual(seed_verification_pending(self.cfg, self.review_store), 1)
        review_verification(self.cfg, self.review_store, self.item_id(), "approved", "tester")
        self.assertEqual(seed_verification_pending(self.cfg, self.review_store), 0)

    def test_suspected_clears_and_reverts(self):
        """存疑 → 撤销覆盖层，回到**覆盖前**的 canonical 值（不是上次写入的覆盖值）。"""
        self.ingest()
        seed_verification_pending(self.cfg, self.review_store)
        iid = self.item_id()
        review_verification(self.cfg, self.review_store, iid, "approved", "tester")
        self.assertEqual(self.verif(), ("tested", 0.85))
        r = review_verification(self.cfg, self.review_store, iid, "suspected", "tester")
        self.assertEqual(r["verification"], "unverified")
        self.assertEqual(self.verif(), ("unverified", 0.5))
        conn = sqlite3.connect(self.kb)
        try:
            self.assertEqual(load_doc_verification_conn(conn), {}, "覆盖层没清干净")
        finally:
            conn.close()

    def test_clear_reverts_to_canonical_expert(self):
        """canonical 本身是 expert 的文档（如 PDF）被误补标后，撤销要退回 expert 而非 unverified。"""
        self.ingest(self.doc("pdf:guide", verification="expert"))
        self.assertEqual(self.verif("pdf:guide"), ("expert", 0.95))
        # 人工改判为 tested
        apply_verification(self.cfg, "pdf:guide", "tested", "tester")
        self.assertEqual(self.verif("pdf:guide"), ("tested", 0.85))
        # 撤销 → 回 expert（覆盖层记着覆盖前的值）
        r = apply_verification(self.cfg, "pdf:guide", None, "tester", clear=True)
        self.assertEqual(r["verification"], "expert")
        self.assertEqual(self.verif("pdf:guide"), ("expert", 0.95))

    def test_review_verification_rejects_other_category(self):
        self.ingest()
        self.review_store.add_item("case_title_flag", "md:case1", {})
        with self.assertRaises(ValueError):
            review_verification(self.cfg, self.review_store, self.item_id("case_title_flag"),
                                "approved", "t")

    def test_apply_verification_missing_doc(self):
        """文档不在库中（已删除/队列陈旧）→ 明确报错，而不是静默写个孤儿覆盖层。"""
        self.ingest()
        with self.assertRaises(ValueError):
            apply_verification(self.cfg, "md:不存在", "tested", "t")

    def test_apply_verification_missing_kb(self):
        from vllm_kb.config import AppConfig as _Cfg

        cfg = _Cfg.model_validate({"storage": {"sqlite_path": str(self.root / "nope.sqlite3")}})
        with self.assertRaises(ValueError):
            apply_verification(cfg, "md:case1", "tested", "t")

    # ---------- 查询期效果 ----------

    def test_query_time_reliability_reflects_approval(self):
        """**目的验证**：认证后检索算出的 w_rel 从 0.5 提到 0.85（不只是库里的数字）。"""
        from vllm_kb.search import SearchEngine

        self.ingest()

        def rel() -> float:
            engine = SearchEngine(self.cfg, read_only=True)
            try:
                res = engine.search("HCCL 超时排查", top_k=5)
            finally:
                engine.close()
            self.assertTrue(res, "检索无结果")
            return res[0].confidence.reliability

        self.assertEqual(rel(), 0.5)
        seed_verification_pending(self.cfg, self.review_store)
        review_verification(self.cfg, self.review_store, self.item_id(), "approved", "tester")
        self.assertEqual(rel(), 0.85, "认证后检索期 w_rel 未提升")

    # ---------- --rebuild 保住人工成果 ----------

    def test_rebuild_preserves_overlays(self):
        """`--rebuild` 删 kb.sqlite3 前先备份覆盖层——人工补标/标签治理不能被清掉。"""
        canonical = self.cfg.resolve(self.cfg.storage.canonical_file)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(doc_to_json(self.doc()) + "\n", encoding="utf-8")
        self.ingest()
        seed_verification_pending(self.cfg, self.review_store)
        review_verification(self.cfg, self.review_store, self.item_id(), "approved", "tester")
        # 同时放一条人工标签（doc_tags 也靠同一机制保住）
        from vllm_kb.review import set_doc_tags_conn

        conn = sqlite3.connect(self.kb)
        try:
            set_doc_tags_conn(conn, "md:case1", manual=["人工标签"], reviewer="tester")
            conn.commit()
        finally:
            conn.close()

        stats = rebuild(self.cfg)

        self.assertGreaterEqual(stats["overlays_kept"], 2, "覆盖层没被备份/恢复")
        self.assertEqual(self.verif(), ("tested", 0.85), "rebuild 后人工补标丢了")
        conn = sqlite3.connect(self.kb)
        try:
            tags = json.loads(conn.execute(
                "SELECT tags FROM docs WHERE source_id='md:case1'").fetchone()[0])
        finally:
            conn.close()
        self.assertIn("人工标签", tags, "rebuild 后人工标签丢了")


class TestContentShaOverlay(unittest.TestCase):
    """内容指纹覆盖层：认证时记 sha；内容变更后旧认证不适用（表格"改了要重审"的底层）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kb = Path(self.tmp.name) / "kb.sqlite3"

    def tearDown(self):
        self.tmp.cleanup()

    def test_content_sha_roundtrip(self):
        from vllm_kb.review import load_doc_verification_map_conn

        conn = sqlite3.connect(self.kb)
        try:
            set_doc_verification_conn(conn, "excel:t", "tested", "alice", content_sha="s1")
            conn.commit()
            g = get_doc_verification_conn(conn, "excel:t")
            self.assertEqual(g["content_sha"], "s1")
            m = load_doc_verification_map_conn(conn)
            self.assertEqual(m, {"excel:t": {"verification": "tested", "content_sha": "s1"}})
            # 重新认证（新内容）→ 指纹刷新
            set_doc_verification_conn(conn, "excel:t", "tested", "bob", content_sha="s2")
            conn.commit()
            self.assertEqual(get_doc_verification_conn(conn, "excel:t")["content_sha"], "s2")
        finally:
            conn.close()

    def test_effective_overlay(self):
        from vllm_kb.review import effective_overlay

        # 指纹一致 → 适用；不一致 → 内容已变，旧认证不适用
        self.assertEqual(effective_overlay({"verification": "tested", "content_sha": "s1"}, "s1"),
                         "tested")
        self.assertIsNone(effective_overlay({"verification": "tested", "content_sha": "s1"}, "s2"))
        # 旧数据无指纹（升级前的覆盖层）→ 仍然适用（向后兼容）
        self.assertEqual(effective_overlay({"verification": "tested", "content_sha": ""}, None),
                         "tested")
        self.assertIsNone(effective_overlay(None, "s1"))
        self.assertIsNone(effective_overlay({}, "s1"))


if __name__ == "__main__":
    unittest.main()
