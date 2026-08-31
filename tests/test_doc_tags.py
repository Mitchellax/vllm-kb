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


class TagManagementTest(unittest.TestCase):
    """词典管理 + tag_candidate seed/采纳 + 审核页标签编辑（数据层）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data_root")
        self.cfg_path = self.root / "config.json"
        self.cfg_path.write_text(json.dumps({
            "embedding": {"provider": "echo", "dimensions": 64},
            "tags": {"registry": [{"name": "HCCL", "tier": "domain"}]},
            "storage": {
                "vector_backend": "python",
                "lancedb_path": "data/lancedb",
                "sqlite_path": "data/kb.sqlite3",
                "canonical_file": "data/raw/canonical.jsonl",
                "review_path": "data/review.sqlite3",
            },
        }), encoding="utf-8")
        self.cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        from vllm_kb.review import ReviewStore, default_review_path

        self.store = ReviewStore(default_review_path(self.cfg))
        self.kb = self.cfg.resolve(self.cfg.storage.sqlite_path)

    def tearDown(self):
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def _ingest(self, source_id="pdf:guide", tags=None, cands=None):
        from vllm_kb.ingest import ingest_docs

        extra = {}
        if cands:
            extra["tag_candidates"] = cands
        doc = KbDocument(source_type="doc_pdf", source_id=source_id, url="", title=source_id,
                         body="正文内容", tags=tags or [], extra=extra)
        ingest_docs(self.cfg, [doc], EmbeddingClient(self.cfg.embedding),
                    PythonVectorStore(self.cfg.resolve("data/vec.json")))

    def test_seed_and_adopt_tag_candidate(self):
        from vllm_kb.review import adopt_tag_candidate, seed_tag_candidates

        self._ingest(cands=[{"name": "超时排查", "tier": "purpose"},
                            {"name": "HCCL", "tier": "domain"}])
        # seed：HCCL 已收录不生成，只生成超时排查
        added = seed_tag_candidates(self.cfg, self.store)
        self.assertEqual(added, 1)
        q = self.store.list_items(category="tag_candidate")
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["payload"]["candidate"], "超时排查")
        # 幂等：重复 seed 不新增
        self.assertEqual(seed_tag_candidates(self.cfg, self.store), 0)
        # 采纳 → 入词典（config.json）+ 写 manual（立即生效）+ approved
        r = adopt_tag_candidate(self.cfg, self.store, q[0]["id"], "tester",
                                config_path=self.cfg_path)
        self.assertEqual(r["tag"], "超时排查")
        data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        names = [x["name"] for x in data["tags"]["registry"]]
        self.assertIn("超时排查", names)
        conn = sqlite3.connect(self.kb)
        try:
            row = conn.execute("SELECT tags FROM docs WHERE source_id='pdf:guide'").fetchone()
        finally:
            conn.close()
        self.assertIn("超时排查", json.loads(row[0]))  # final 立即包含
        it = self.store.get_item(q[0]["id"])
        self.assertEqual(it["status"], "approved")

    def test_seed_tag_candidates_aggregates_by_word(self):
        """同词跨文档聚合：一条审核项带 doc_count，采纳时全部提及文档打标。"""
        from vllm_kb.review import adopt_tag_candidate, seed_tag_candidates

        self._ingest(source_id="pdf:a", cands=[{"name": "拓扑", "tier": "domain"}])
        self._ingest(source_id="pdf:b", cands=[{"name": "拓扑", "tier": "domain"}])
        self._ingest(source_id="pdf:c", cands=[{"name": "单独词", "tier": "purpose"}])
        added = seed_tag_candidates(self.cfg, self.store)
        self.assertEqual(added, 2)  # 拓扑（2 篇）+ 单独词（1 篇），而非 3 条
        q = self.store.list_items(category="tag_candidate")
        by_name = {i["payload"]["candidate"]: i for i in q}
        self.assertEqual(by_name["拓扑"]["payload"]["doc_count"], 2)
        self.assertEqual(len(by_name["拓扑"]["payload"]["docs"]), 2)
        self.assertEqual(by_name["单独词"]["payload"]["doc_count"], 1)
        # 采纳"拓扑" → 两篇文档 manual 都写入（立即生效）
        r = adopt_tag_candidate(self.cfg, self.store, by_name["拓扑"]["id"], "tester",
                                config_path=self.cfg_path)
        self.assertEqual(r["docs_count"], 2)
        conn = sqlite3.connect(self.kb)
        try:
            for sid in ("pdf:a", "pdf:b"):
                row = conn.execute("SELECT tags FROM docs WHERE source_id=?", (sid,)).fetchone()
                self.assertIn("拓扑", json.loads(row[0]))
            row = conn.execute("SELECT tags FROM docs WHERE source_id='pdf:c'").fetchone()
            self.assertNotIn("拓扑", json.loads(row[0]))
        finally:
            conn.close()
        # 忽略过的候选（approved 任意状态）不再重复生成
        self.assertEqual(seed_tag_candidates(self.cfg, self.store), 0)

    def test_seed_cleans_old_format_tag_candidates(self):
        """旧格式（source_id::name）审核项在 seed 时清理，新聚合格式重建。"""
        from vllm_kb.review import seed_tag_candidates

        self._ingest(source_id="pdf:x", cands=[{"name": "带宽", "tier": "domain"}])
        # 手工制造旧格式项（模拟升级前遗留）
        self.store.add_item("tag_candidate", "pdf:x::带宽", {
            "source_id": "pdf:x", "title": "pdf:x",
            "candidate": "带宽", "suggested_tier": "domain",
        })
        added = seed_tag_candidates(self.cfg, self.store)
        q = self.store.list_items(category="tag_candidate")
        self.assertEqual(added, 1)  # 旧格式被清理，新聚合格式生成 1 条
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["item_ref"], "tag:带宽")
        self.assertEqual(q[0]["payload"]["doc_count"], 1)

    def test_tag_dict_management(self):
        from vllm_kb.review import (
            add_tag_to_registry,
            delete_tag,
            rename_tag,
            set_tag_tier,
            tag_dict,
        )

        self._ingest(tags=["HCCL", "故障排查"])
        # add
        add_tag_to_registry(self.cfg, "网络", "domain", config_path=self.cfg_path)
        data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        self.assertIn({"name": "网络", "tier": "domain"}, data["tags"]["registry"])
        # rename：先入词典再改名（registry + docs.tags 全库替换）
        add_tag_to_registry(self.cfg, "故障排查", "purpose", config_path=self.cfg_path)
        rename_tag(self.cfg, "故障排查", "故障定位", self.kb, config_path=self.cfg_path)
        data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        names = [x["name"] for x in data["tags"]["registry"]]
        self.assertIn("故障定位", names)
        self.assertNotIn("故障排查", names)
        conn = sqlite3.connect(self.kb)
        try:
            row = conn.execute("SELECT tags FROM docs WHERE source_id='pdf:guide'").fetchone()
        finally:
            conn.close()
        self.assertIn("故障定位", json.loads(row[0]))
        self.assertNotIn("故障排查", json.loads(row[0]))
        # tier 修改（全局生效）
        set_tag_tier(self.cfg, "HCCL", "purpose", config_path=self.cfg_path)
        data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        hccl = next(x for x in data["tags"]["registry"] if x["name"] == "HCCL")
        self.assertEqual(hccl["tier"], "purpose")
        # delete：仅移出词典（已打标文档保留）
        delete_tag(self.cfg, "HCCL", config_path=self.cfg_path)
        data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        self.assertNotIn("HCCL", [x["name"] for x in data["tags"]["registry"]])
        conn = sqlite3.connect(self.kb)
        try:
            row = conn.execute("SELECT tags FROM docs WHERE source_id='pdf:guide'").fetchone()
        finally:
            conn.close()
        self.assertIn("HCCL", json.loads(row[0]))
        # tag_dict 分组 + 计数（词典以临时 config 文件为准）
        td = tag_dict(self.cfg, self.kb, config_path=self.cfg_path)
        # 故障定位（原名故障排查）：add 时 tier 指定 purpose → purpose 组
        # 网络 → domain；故 domain=1(网络), purpose=1(故障定位)
        self.assertEqual(td["stats"]["domain"], 1)
        self.assertEqual(td["stats"]["purpose"], 1)
        self.assertEqual(td["stats"]["tagged_docs"], 2)  # HCCL + 故障定位 打标 1 篇


if __name__ == "__main__":
    unittest.main()
