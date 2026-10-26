"""审核工作台测试：ReviewStore CRUD、seed 补单（幂等）、API 配置中心 key 脱敏、OCR 连通性。"""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vllm_kb.config import AppConfig
from vllm_kb.ocr import OcrApiError
from vllm_kb.review import (
    ReviewStore,
    api_configs,
    get_doc_verification_conn,
    review_verification,
    seed_case_title_flags,
    seed_verification_pending,
    probe_ocr_connectivity,
    update_config_json,
)
from vllm_kb.secrets import load_secrets, save_secret, secrets_path


def make_cfg(tmp: Path, extra_sources=None) -> AppConfig:
    return AppConfig.model_validate(
        {
            "sources": [
                {"id": "a", "type": "github", "token_env": "GITHUB_TOKEN", "enabled": True},
            ] + (extra_sources or []),
            "embedding": {"provider": "openai_compatible", "base_url": "https://api.example.com/v1",
                          "api_key": "sk-secret-abc", "model": "bge-m3"},
            "storage": {
                "vector_backend": "python",
                "lancedb_path": str(tmp / "vec.json"),
                "sqlite_path": str(tmp / "kb.sqlite3"),
                "canonical_file": str(tmp / "canonical.jsonl"),
                "review_path": str(tmp / "review.sqlite3"),
            },
        }
    )


def make_kb(tmp: Path) -> None:
    """造最小 kb.sqlite3（docs 表，含 unverified 与待审核标题样例）。"""
    conn = sqlite3.connect(tmp / "kb.sqlite3")
    conn.execute("CREATE TABLE docs (source_id TEXT, title TEXT, url TEXT, extra TEXT)")
    rows = [
        ("md:glm三板斧", "GLM5.1 崩溃三板斧", "", json.dumps({"verification": "unverified"})),
        ("md:caseA", "【待审核】HCCN 排查案例", "", json.dumps({"verification": "unverified"})),
        ("pdf:guide", "NPU 接口指南", "", json.dumps({"verification": "expert"})),
    ]
    conn.executemany("INSERT INTO docs VALUES(?,?,?,?)", rows)
    conn.commit()
    conn.close()


class TestReviewStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ReviewStore(Path(self.tmp.name) / "review.sqlite3")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_add_dedupe_and_list(self):
        self.assertTrue(self.store.add_item("verification_pending", "md:doc1", {"title": "t"}))
        # 同 (category, item_ref) 任意状态去重（认证/存疑后也不再自动补单）
        self.assertFalse(self.store.add_item("verification_pending", "md:doc1", {"title": "t"}))
        self.store.review(1, "approved", "alice")
        self.assertFalse(self.store.add_item("verification_pending", "md:doc1", {"title": "t"}))

    def test_invalid_category(self):
        with self.assertRaises(ValueError):
            self.store.add_item("nope", "x")

    def test_review_approved(self):
        self.store.add_item("ocr_mismatch", "img:a", {"note": "x"})
        self.assertTrue(self.store.review(1, "approved", "bob", {"note": "ok"}))
        it = self.store.get_item(1)
        self.assertEqual(it["status"], "approved")
        self.assertEqual(it["reviewer"], "bob")
        self.assertEqual(it["result"]["note"], "ok")
        self.assertIsNotNone(it["reviewed_at"])

    def test_review_suspected(self):
        self.store.add_item("verification_pending", "md:a")
        self.store.add_item("verification_pending", "md:b")
        self.store.review(1, "suspected", "c")
        items = self.store.list_items()
        # 未审核（pending）排在存疑（suspected）前面
        self.assertEqual([i["item_ref"] for i in items], ["md:b", "md:a"])
        self.assertEqual(items[0]["status"], "pending")
        self.assertEqual(items[1]["status"], "suspected")

    def test_review_invalid_action(self):
        self.store.add_item("case_title_flag", "x")
        for bad in ("confirmed", "rejected", "modified", "deleted", "bogus"):
            with self.assertRaises(ValueError):
                self.store.review(1, bad, "r")

    def test_stats(self):
        self.store.add_item("verification_pending", "a")
        self.store.add_item("verification_pending", "b")
        self.store.review(1, "suspected", "c")
        s = self.store.stats()
        self.assertEqual(s["verification_pending"], {"pending": 1, "suspected": 1, "total": 2})


class TestDeleteAndUndo(unittest.TestCase):
    """标记删除（只删数据库记录，原始资产保留）+ 撤回恢复，模拟真实 kb.sqlite3。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.kb = self.root / "kb.sqlite3"
        conn = sqlite3.connect(self.kb)
        conn.execute("CREATE TABLE docs (source_id TEXT PRIMARY KEY, source_type TEXT, title TEXT, "
                     "extra TEXT, content_hash TEXT)")
        conn.execute(
            "INSERT INTO docs VALUES(?,?,?,?,?)",
            ("md:case1", "doc_markdown", "测试案例",
             json.dumps({"verification": "unverified",
                         "asset": {"path": "assets/md/case1.md"}}), "h1"),
        )
        conn.commit()
        conn.close()
        self.store = ReviewStore(self.root / "review.sqlite3")
        self.store.add_item("verification_pending", "md:case1", {"title": "测试案例"})

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_mark_deleted_removes_kb_row_keeps_asset(self):
        self.assertTrue(self.store.mark_deleted(1, "alice", self.kb, note="重复文档"))
        conn = sqlite3.connect(self.kb)
        self.assertIsNone(conn.execute("SELECT source_id FROM docs WHERE source_id='md:case1'").fetchone())
        conn.close()
        it = self.store.get_item(1)
        self.assertEqual(it["status"], "deleted")
        # 资产路径记录在 payload（供人工本地删除原始文件）
        self.assertEqual(it["payload"]["asset"]["path"], "assets/md/case1.md")
        # 原始文件保留：资产文件不受影响（本测试无真实文件，验证 kb 行已删即可）

    def test_mark_deleted_missing_kb_row(self):
        conn = sqlite3.connect(self.kb)
        conn.execute("DELETE FROM docs WHERE source_id='md:case1'")
        conn.commit()
        conn.close()
        with self.assertRaises(ValueError):
            self.store.mark_deleted(1, "alice", self.kb)

    def test_undo_delete_restores_and_requeues(self):
        self.store.mark_deleted(1, "alice", self.kb)
        self.assertTrue(self.store.undo_delete(1, self.kb))
        conn = sqlite3.connect(self.kb)
        row = conn.execute("SELECT title, extra FROM docs WHERE source_id='md:case1'").fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "测试案例")
        self.assertEqual(json.loads(row[1])["verification"], "unverified")  # 恢复完整
        it = self.store.get_item(1)
        self.assertEqual(it["status"], "pending")  # 重新入队

    def test_undo_delete_non_deleted(self):
        with self.assertRaises(ValueError):
            self.store.undo_delete(1, self.kb)  # 仍是 pending


class TestExternalDocManagement(unittest.TestCase):
    """文档管理页签：外源文档列表 + 彻底删除（docs + chunks + 向量，本地文件不动）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.kb = self.root / "kb.sqlite3"
        conn = sqlite3.connect(self.kb)
        conn.executescript(
            """
            CREATE TABLE docs (source_id TEXT PRIMARY KEY, source_type TEXT, title TEXT,
                               component TEXT, url TEXT, extra TEXT);
            CREATE TABLE chunks_fts (chunk_id TEXT, doc_id TEXT, text TEXT);
            CREATE TABLE chunks_meta (chunk_id TEXT PRIMARY KEY, doc_id TEXT, seq INTEGER, section TEXT);
            """
        )
        docs = [
            ("pdf:guide", "doc_pdf", "接口指南", "ascend", "",
             json.dumps({"verification": "expert", "asset": {"path": "assets/pdf/guide.pdf"}})),
            ("md:case", "doc_markdown", "案例", "", "",
             json.dumps({"verification": "unverified", "asset": {"path": "assets/md/case.md"}})),
            ("github:vllm-project-vllm:issue:1", "github_issue", "gh issue", "vllm", "", "{}"),
        ]
        conn.executemany("INSERT INTO docs VALUES(?,?,?,?,?,?)", docs)
        conn.execute("INSERT INTO chunks_fts VALUES('pdf:guide#0','pdf:guide','【1.1 简介】正文')")
        conn.execute("INSERT INTO chunks_meta VALUES('pdf:guide#0','pdf:guide',0,'1.1 简介')")
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_only_external_docs(self):
        from vllm_kb.review import list_external_docs

        docs = list_external_docs(self.kb)
        ids = [d["source_id"] for d in docs]
        self.assertEqual(ids, ["md:case", "pdf:guide"])  # 排除 github 来源
        g = [d for d in docs if d["source_id"] == "pdf:guide"][0]
        self.assertEqual(g["source_type"], "doc_pdf")
        self.assertEqual(g["verification"], "expert")
        # 路径脱敏：返回 asset_id/asset_path（asset_path 为空——旧数据无 asset_registry）
        self.assertNotIn("asset", g)
        self.assertEqual(g["tags"]["final"], [])
        self.assertIn("tags", g)
        self.assertFalse(g["duplicate"])

    def test_delete_external_doc_removes_all_layers(self):
        from vllm_kb.review import delete_external_doc

        r = delete_external_doc(self.kb, "pdf:guide")
        self.assertEqual(r["chunks_deleted"], 1)
        conn = sqlite3.connect(self.kb)
        self.assertIsNone(conn.execute(
            "SELECT source_id FROM docs WHERE source_id='pdf:guide'").fetchone())
        self.assertEqual(conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE doc_id='pdf:guide'").fetchone()[0], 0)
        self.assertEqual(conn.execute(
            "SELECT COUNT(*) FROM chunks_meta WHERE doc_id='pdf:guide'").fetchone()[0], 0)
        # 其他文档不受影响
        self.assertIsNotNone(conn.execute(
            "SELECT source_id FROM docs WHERE source_id='md:case'").fetchone())
        conn.close()

    def test_delete_missing_doc_raises(self):
        from vllm_kb.review import delete_external_doc

        with self.assertRaises(ValueError):
            delete_external_doc(self.kb, "pdf:nope")


class TestSeeds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_kb(self.root)
        self.cfg = make_cfg(self.root)
        self.store = ReviewStore(self.root / "review.sqlite3")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_seed_verification_pending(self):
        n = seed_verification_pending(self.cfg, self.store)
        self.assertEqual(n, 2)  # unverified 的 md 两条；expert 不补单
        # 幂等
        self.assertEqual(seed_verification_pending(self.cfg, self.store), 0)
        refs = {i["item_ref"] for i in self.store.list_items()}
        self.assertEqual(refs, {"md:glm三板斧", "md:caseA"})

    def test_seed_case_title_flags(self):
        n = seed_case_title_flags(self.cfg, self.store)
        self.assertEqual(n, 1)  # 只有"待审核"标题
        items = self.store.list_items(category="case_title_flag")
        self.assertEqual(items[0]["item_ref"], "md:caseA")


class TestTagCandidateVariants(unittest.TestCase):
    """tag_candidate 同义变体（2026-10 业务积压）：聚合键 tag_key 收敛 +
    存量 pending 变体合并迁移 + 采纳映射到词典正式名。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.store = ReviewStore(self.root / "review.sqlite3")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_seed_aggregates_by_tag_key(self):
        """同一词的排版变体（Atlas A3 / atlas_a3）聚合为一条候选，display 保留首见原形。"""
        from vllm_kb.review import seed_tag_candidates

        conn = sqlite3.connect(self.root / "kb.sqlite3")
        conn.execute("CREATE TABLE docs (source_id TEXT PRIMARY KEY, title TEXT, extra TEXT)")
        conn.executemany("INSERT INTO docs VALUES(?,?,?)", [
            ("md:a", "A", json.dumps({"tag_candidates": [{"name": "Atlas A3"}]})),
            ("md:b", "B", json.dumps({"tag_candidates": [{"name": "atlas_a3"}]})),
            ("md:c", "C", json.dumps({"tag_candidates": [{"name": "ATLAS-A3"}]})),
        ])
        conn.commit()
        conn.close()
        seed_tag_candidates(self.cfg, self.store)
        items = [i for i in self.store.list_items(category="tag_candidate", status="pending")]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["item_ref"], "tag:atlas-a3")
        self.assertEqual(items[0]["payload"]["candidate"], "Atlas A3")   # 首见原形
        self.assertEqual(items[0]["payload"]["doc_count"], 3)

    def test_merge_existing_variants(self):
        """存量 pending 同 key 变体：docs 并入规范 key 条目（item_ref=tag:{tag_key}），
        其余 approved(merged_variant)；无规范条目时保留最早 id。"""
        from vllm_kb.review import _merge_tag_candidate_variants

        self.store.add_item("tag_candidate", "tag:Atlas A3",
                            {"candidate": "Atlas A3", "doc_count": 2,
                             "docs": [{"source_id": "md:a", "title": "A"},
                                      {"source_id": "md:b", "title": "B"}]})
        self.store.add_item("tag_candidate", "tag:atlas_a3",
                            {"candidate": "atlas_a3", "doc_count": 1,
                             "docs": [{"source_id": "md:c", "title": "C"}]})
        self.store.add_item("tag_candidate", "tag:atlas-a3",       # 规范 key 条目（新 seed 产物）
                            {"candidate": "Atlas A3", "doc_count": 0, "docs": []})
        n = _merge_tag_candidate_variants(self.store)
        self.assertEqual(n, 2)
        pending = [i for i in self.store.list_items(category="tag_candidate", status="pending")]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["item_ref"], "tag:atlas-a3")     # 保留规范 key 条目
        self.assertEqual(pending[0]["payload"]["doc_count"], 3)      # docs 全部并入
        gone = {i["id"]: i for i in self.store.list_items(category="tag_candidate")}
        self.assertEqual(gone[1]["status"], "approved")
        self.assertEqual(gone[1]["result"]["action"], "merged_variant")
        # 幂等：再跑无操作
        self.assertEqual(_merge_tag_candidate_variants(self.store), 0)

    def test_case_variant_not_requeued_after_ignore(self):
        """业务实测（GLM5/glm5、DeepSeek-V4/Deepseek-V4 仍在队列）："GLM5" 被认证忽略后，
        变体 "glm5" 换 item_ref 不得重新入队（忽略记忆按 key）；未处理词照常入队。"""
        from vllm_kb.review import seed_tag_candidates

        conn = sqlite3.connect(self.root / "kb.sqlite3")
        conn.execute("CREATE TABLE docs (source_id TEXT PRIMARY KEY, title TEXT, extra TEXT)")
        conn.executemany("INSERT INTO docs VALUES(?,?,?)", [
            ("md:a", "A", json.dumps({"tag_candidates": [{"name": "GLM5"}, {"name": "DeepSeek-V4"}]})),
            ("md:b", "B", json.dumps({"tag_candidates": [{"name": "glm5"}, {"name": "Deepseek-V4"}]})),
        ])
        conn.commit()
        conn.close()
        self.store.add_item("tag_candidate", "tag:GLM5",
                            {"candidate": "GLM5", "doc_count": 1,
                             "docs": [{"source_id": "md:a", "title": "A"}]})
        self.store.review(1, "approved", "bob", {"action": "ignore"})   # 用户忽略 GLM5
        seed_tag_candidates(self.cfg, self.store)
        pending = [i for i in self.store.list_items(category="tag_candidate", status="pending")]
        # glm5 不重入队（同 key 已忽略）；DeepSeek-V4/Deepseek-V4 从未处理 → 一条规范条目
        self.assertEqual([i["item_ref"] for i in pending], ["tag:deepseek-v4"])
        self.assertEqual(pending[0]["payload"]["candidate"], "DeepSeek-V4")
        self.assertEqual(pending[0]["payload"]["doc_count"], 2)

    def test_seed_aggregates_case_variants(self):
        """双 pending 原形条目 + extras 双变体 → seed 后 pending 仅剩规范 key 条目。"""
        from vllm_kb.review import seed_tag_candidates

        conn = sqlite3.connect(self.root / "kb.sqlite3")
        conn.execute("CREATE TABLE docs (source_id TEXT PRIMARY KEY, title TEXT, extra TEXT)")
        conn.executemany("INSERT INTO docs VALUES(?,?,?)", [
            ("md:a", "A", json.dumps({"tag_candidates": [{"name": "GLM5"}, {"name": "DeepSeek-V4"}]})),
            ("md:b", "B", json.dumps({"tag_candidates": [{"name": "glm5"}, {"name": "Deepseek-V4"}]})),
        ])
        conn.commit()
        conn.close()
        self.store.add_item("tag_candidate", "tag:GLM5",
                            {"candidate": "GLM5", "doc_count": 1,
                             "docs": [{"source_id": "md:a", "title": "A"}]})
        self.store.add_item("tag_candidate", "tag:DeepSeek-V4",
                            {"candidate": "DeepSeek-V4", "doc_count": 1,
                             "docs": [{"source_id": "md:a", "title": "A"}]})
        seed_tag_candidates(self.cfg, self.store)
        pending = [i for i in self.store.list_items(category="tag_candidate", status="pending")]
        self.assertEqual({i["item_ref"] for i in pending},
                         {"tag:glm5", "tag:deepseek-v4"})           # 规范 key，原形条目已合并
        docs_by_ref = {i["item_ref"]: i["payload"]["doc_count"] for i in pending}
        self.assertEqual(docs_by_ref, {"tag:glm5": 2, "tag:deepseek-v4": 2})


class TestAdoptedTagVariants(unittest.TestCase):
    """已认证 tag 的归一迁移（2026-10）：词典同 key 变体词条收敛 + 全库标签替换——
    已认证的 tag 直接适用 tag_key 规则，无需删掉重加。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg_path = self.root / "config.json"
        self.cfg_path.write_text(json.dumps({
            "project": {"name": "x", "data_root": str(self.root)},
            "tags": {"registry": [
                {"name": "CANN版本说明", "tier": "purpose"},
                {"name": "CANN 版本说明", "tier": "domain"},
                {"name": "HCCL", "tier": "domain"},
            ]},
            "embedding": {"provider": "echo", "dimensions": 8},
            "storage": {"vector_backend": "python",
                        "sqlite_path": str(self.root / "kb.sqlite3")},
        }, ensure_ascii=False), encoding="utf-8")
        conn = sqlite3.connect(self.root / "kb.sqlite3")
        conn.executescript(
            "CREATE TABLE docs (source_id TEXT PRIMARY KEY, source_type TEXT, title TEXT,"
            " component TEXT, url TEXT, status TEXT, resolved_at TEXT, extra TEXT, reliability REAL,"
            " tags TEXT);"
            "CREATE TABLE doc_tags (source_id TEXT PRIMARY KEY, manual TEXT, excluded TEXT, reviewer TEXT)")
        conn.executemany("INSERT INTO docs VALUES(?,?,?,?,?,?,?,?,?,?)", [
            ("md:a", "doc_markdown", "A", "", "", None, None, "{}", 0.5,
             json.dumps(["CANN版本说明", "HCCL"], ensure_ascii=False)),
            ("md:b", "doc_markdown", "B", "", "", None, None, "{}", 0.5,
             json.dumps(["CANN 版本说明"], ensure_ascii=False)),
        ])
        conn.execute("INSERT INTO doc_tags VALUES(?,?,?,?)",
                     ("md:b", json.dumps(["CANN 版本说明"], ensure_ascii=False),
                      json.dumps([], ensure_ascii=False), "old"))
        conn.commit()
        conn.close()
        self.cfg = AppConfig.model_validate({
            "embedding": {"provider": "echo", "dimensions": 8},
            "storage": {"vector_backend": "python",
                        "sqlite_path": str(self.root / "kb.sqlite3")},
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_normalize_variants_merges_dict_and_rewrites_docs(self):
        from vllm_kb.review import normalize_tag_variants

        r = normalize_tag_variants(self.cfg, config_path=self.cfg_path)
        self.assertEqual(r["merged_variants"], 1)
        # 词典收敛：config 只剩正式名 + HCCL（HCCL 无变体不动）
        data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        names = [e["name"] for e in data["tags"]["registry"]]
        self.assertEqual(names, ["CANN版本说明", "HCCL"])
        # 全库替换：docs.tags 与 doc_tags.manual 里的变体 → 正式名
        conn = sqlite3.connect(self.root / "kb.sqlite3")
        tags_a = json.loads(conn.execute("SELECT tags FROM docs WHERE source_id='md:a'").fetchone()[0])
        tags_b = json.loads(conn.execute("SELECT tags FROM docs WHERE source_id='md:b'").fetchone()[0])
        manual_b = json.loads(conn.execute("SELECT manual FROM doc_tags WHERE source_id='md:b'").fetchone()[0])
        conn.close()
        self.assertEqual(tags_a, ["CANN版本说明", "HCCL"])
        self.assertEqual(tags_b, ["CANN版本说明"])
        self.assertEqual(manual_b, ["CANN版本说明"])
        # 幂等：再跑零开销
        self.assertEqual(normalize_tag_variants(self.cfg, config_path=self.cfg_path),
                         {"merged_variants": 0, "docs_updated": 0})


class TestVerificationReset(unittest.TestCase):
    """认证级别三档全覆盖：approved+unverified = 清人工覆盖层（重置），生效值回 canonical。

    2026-10 业务复核：UI 只有 tested/expert 两档，无法撤销认证——补 unverified 重置语义
    （不写 unverified 覆盖层：canonical 可能自带更高值如 PDF expert，清层后按 canonical 生效）。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # docs 表用完整列结构（apply_verification 会 UPDATE source_type/title/extra）
        conn = sqlite3.connect(self.root / "kb.sqlite3")
        conn.executescript(
            "CREATE TABLE docs (source_id TEXT PRIMARY KEY, source_type TEXT, title TEXT,"
            " component TEXT, url TEXT, status TEXT, resolved_at TEXT, extra TEXT, reliability REAL)"
        )
        conn.executemany("INSERT INTO docs VALUES(?,?,?,?,?,?,?,?,?)", [
            ("md:glm三板斧", "doc_markdown", "GLM5.1 崩溃三板斧", "", "", None, None,
             json.dumps({"verification": "unverified"}), 0.5),
            ("md:caseA", "doc_markdown", "【待审核】HCCN 排查案例", "", "", None, None,
             json.dumps({"verification": "unverified"}), 0.5),
            ("pdf:guide", "doc_pdf", "NPU 接口指南", "", "", None, None,
             json.dumps({"verification": "expert"}), 0.85),
        ])
        conn.commit()
        conn.close()
        self.cfg = make_cfg(self.root)
        self.store = ReviewStore(self.root / "review.sqlite3")
        self.assertEqual(seed_verification_pending(self.cfg, self.store), 2)
        # 先认证一档
        item = next(i for i in self.store.list_items(category="verification_pending")
                    if i["item_ref"] == "md:glm三板斧")
        review_verification(self.cfg, self.store, item["id"], "approved",
                            "alice", result={"verification": "expert"})
        conn = sqlite3.connect(self.root / "kb.sqlite3")
        self.assertEqual(get_doc_verification_conn(conn, "md:glm三板斧")["verification"],
                         "expert")
        conn.close()
        # 队列里该文档重新入队（认证后 seed 收敛）——直接用原 item_id 走重置
        self.item_id = item["id"]

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_reset_unverified_clears_overlay_and_reopens(self):
        """重置 = 清覆盖层（生效值回 canonical）+ 审核项回 pending（保留重审入口）。"""
        review_verification(self.cfg, self.store, self.item_id, "approved",
                            "bob", result={"verification": "unverified"})
        conn = sqlite3.connect(self.root / "kb.sqlite3")
        try:
            self.assertIsNone(
                get_doc_verification_conn(conn, "md:glm三板斧")["verification"])
        finally:
            conn.close()
        it = self.store.get_item(self.item_id)
        self.assertEqual(it["status"], "pending")        # 队列项重开（可重新审核）
        self.assertEqual(it["reviewer"], "bob")

    def test_reset_reopens_queue_item(self):
        """重置后队列项回 pending：文档回到待补标状态可被再次认证/重置（add_item 去重
        铁律下 seed 不会补新项——重开原项是唯一正确入口）。"""
        review_verification(self.cfg, self.store, self.item_id, "approved",
                            "bob", result={"verification": "unverified"})
        review_verification(self.cfg, self.store, self.item_id, "approved",
                            "carol", result={"verification": "tested"})
        conn = sqlite3.connect(self.root / "kb.sqlite3")
        try:
            self.assertEqual(
                get_doc_verification_conn(conn, "md:glm三板斧")["verification"], "tested")
        finally:
            conn.close()
        self.assertEqual(self.store.get_item(self.item_id)["status"], "approved")


class TestFileLevelVerification(unittest.TestCase):
    """表格文件粒度审核：seed 按文件聚合、认证批量应用到全部行、内容变更重新审核。"""

    FILE_SHA = "a" * 64

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.kb = self.root / "kb.sqlite3"
        conn = sqlite3.connect(self.kb)
        conn.executescript(
            "CREATE TABLE docs (source_id TEXT PRIMARY KEY, source_type TEXT, title TEXT,"
            " component TEXT, url TEXT, status TEXT, resolved_at TEXT, extra TEXT, reliability REAL)"
        )
        rows = [
            # 旧逐行格式（业务存量库形态）：同文件两行 + 一篇 md
            ("excel:登记表:Sheet1:1", "doc_excel", "问题登记表 行1", "", "", None, None,
             json.dumps({"verification": "unverified",
                         "asset": {"sha256": self.FILE_SHA}}), 0.5),
            ("excel:登记表:Sheet1:2", "doc_excel", "问题登记表 行2", "", "", None, None,
             json.dumps({"verification": "unverified",
                         "asset": {"sha256": self.FILE_SHA}}), 0.5),
            ("md:case", "doc_markdown", "案例", "", "", None, None,
             json.dumps({"verification": "unverified",
                         "asset": {"sha256": "b" * 64}}), 0.5),
        ]
        conn.executemany("INSERT INTO docs VALUES(?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
        self.cfg = make_cfg(self.root)
        self.store = ReviewStore(self.root / "review.sqlite3")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _file_item(self):
        items = [i for i in self.store.list_items(category="verification_pending")
                 if i["item_ref"].startswith("excelfile:")]
        self.assertEqual(len(items), 1)
        return items[0]

    def test_seed_groups_excel_rows_by_file(self):
        n = seed_verification_pending(self.cfg, self.store)
        self.assertEqual(n, 2)  # 1 个文件级项 + 1 篇 md
        it = self._file_item()
        self.assertEqual(it["payload"]["row_count"], 2)
        self.assertEqual([r["source_id"] for r in it["payload"]["rows"]],
                         ["excel:登记表:Sheet1:1", "excel:登记表:Sheet1:2"])
        self.assertEqual(it["payload"]["file_sha"], self.FILE_SHA)
        self.assertEqual(it["payload"]["aggregate"], "excel_file")
        # 幂等
        self.assertEqual(seed_verification_pending(self.cfg, self.store), 0)

    def test_legacy_row_items_replaced(self):
        """旧版逐行 pending 项被文件级项取代（自动清理）。"""
        self.store.add_item("verification_pending", "excel:登记表:Sheet1:1", {"title": "旧行项"})
        seed_verification_pending(self.cfg, self.store)
        refs = {i["item_ref"] for i in self.store.list_items(category="verification_pending")}
        self.assertNotIn("excel:登记表:Sheet1:1", refs)  # 旧逐行项已清
        self.assertTrue(any(r.startswith("excelfile:") for r in refs))

    def test_approve_applies_to_all_rows(self):
        seed_verification_pending(self.cfg, self.store)
        it = self._file_item()
        r = review_verification(self.cfg, self.store, it["id"], "approved", "alice",
                                result={"verification": "expert"})
        self.assertEqual(r["rows_count"], 2)
        self.assertEqual(r["rows_verified"], 2)
        conn = sqlite3.connect(self.kb)
        for sid in ("excel:登记表:Sheet1:1", "excel:登记表:Sheet1:2"):
            extra = json.loads(
                conn.execute("SELECT extra FROM docs WHERE source_id=?", (sid,)).fetchone()[0])
            self.assertEqual(extra["verification"], "expert")  # 整表生效
            ov = get_doc_verification_conn(conn, sid)
            self.assertEqual(ov["verification"], "expert")
            self.assertEqual(ov["content_sha"], self.FILE_SHA)  # 内容指纹入覆盖层
        conn.close()
        # 认证后再 seed 不再入队（同内容）
        self.assertEqual(seed_verification_pending(self.cfg, self.store), 0)

    def test_suspected_clears_all_rows(self):
        seed_verification_pending(self.cfg, self.store)
        it = self._file_item()
        review_verification(self.cfg, self.store, it["id"], "suspected", "alice")
        conn = sqlite3.connect(self.kb)
        for sid in ("excel:登记表:Sheet1:1", "excel:登记表:Sheet1:2"):
            self.assertIsNone(get_doc_verification_conn(conn, sid)["verification"])
        conn.close()

    def test_content_change_requeues(self):
        """表格内容变更 → 旧认证失效（覆盖层清、docs 回 canonical）→ 新指纹重新入队。"""
        seed_verification_pending(self.cfg, self.store)
        it = self._file_item()
        review_verification(self.cfg, self.store, it["id"], "approved", "alice")
        # 文件被编辑（sha 变）重新入库；未升级的库生效值仍是旧覆盖层的 tested
        conn = sqlite3.connect(self.kb)
        new_sha = "c" * 64
        for sid in ("excel:登记表:Sheet1:1", "excel:登记表:Sheet1:2"):
            extra = json.loads(
                conn.execute("SELECT extra FROM docs WHERE source_id=?", (sid,)).fetchone()[0])
            extra["asset"]["sha256"] = new_sha
            extra["verification"] = "tested"
            conn.execute("UPDATE docs SET extra=? WHERE source_id=?",
                         (json.dumps(extra, ensure_ascii=False), sid))
        conn.commit()
        conn.close()
        n = seed_verification_pending(self.cfg, self.store)
        self.assertEqual(n, 1)  # 新 sha → 新文件级项
        refs = {i["item_ref"] for i in self.store.list_items(category="verification_pending")}
        self.assertIn(f"excelfile:登记表:{new_sha[:8]}", refs)
        conn = sqlite3.connect(self.kb)
        extra = json.loads(conn.execute(
            "SELECT extra FROM docs WHERE source_id='excel:登记表:Sheet1:1'").fetchone()[0])
        conn.close()
        self.assertEqual(extra["verification"], "unverified")  # 旧认证已失效
        ov = get_doc_verification_conn(sqlite3.connect(self.kb), "excel:登记表:Sheet1:1")
        self.assertIsNone(ov["verification"])

    def test_mark_deleted_and_undo_whole_file(self):
        seed_verification_pending(self.cfg, self.store)
        it = self._file_item()
        self.assertTrue(self.store.mark_deleted(it["id"], "alice", self.kb, note="整表废弃"))
        conn = sqlite3.connect(self.kb)
        for sid in ("excel:登记表:Sheet1:1", "excel:登记表:Sheet1:2"):
            self.assertIsNone(conn.execute("SELECT source_id FROM docs WHERE source_id=?",
                                           (sid,)).fetchone())
        conn.close()
        self.assertTrue(self.store.undo_delete(it["id"], self.kb))
        conn = sqlite3.connect(self.kb)
        for sid in ("excel:登记表:Sheet1:1", "excel:登记表:Sheet1:2"):
            self.assertIsNotNone(conn.execute("SELECT source_id FROM docs WHERE source_id=?",
                                              (sid,)).fetchone())
        conn.close()
        self.assertEqual(self.store.get_item(it["id"])["status"], "pending")


class TestApiConfigs(unittest.TestCase):
    def test_key_redacted(self):
        tmp = tempfile.TemporaryDirectory()
        cfg = make_cfg(Path(tmp.name), extra_sources=[
            {"id": "images", "type": "image", "ocr_provider": "api",
             "ocr_api_base": "http://ocr:8000", "ocr_api_key": "ocr-secret-key"},
        ])
        cs = {c["name"]: c for c in api_configs(cfg)}
        # key 脱敏：响应中不出现真实密钥
        blob = json.dumps(cs, ensure_ascii=False)
        self.assertNotIn("sk-secret-abc", blob)
        self.assertNotIn("ocr-secret-key", blob)
        # 状态
        self.assertTrue(cs["embedding"]["key_configured"])
        self.assertEqual(cs["embedding"]["status"], "configured")
        self.assertTrue(cs["ocr"]["key_configured"])
        self.assertEqual(cs["ocr"]["provider"], "api")
        self.assertEqual(cs["ocr"]["base_url"], "http://ocr:8000")
        tmp.cleanup()

    def test_ocr_model_field(self):
        tmp = tempfile.TemporaryDirectory()
        cfg = make_cfg(Path(tmp.name), extra_sources=[
            {"id": "images", "type": "image", "ocr_provider": "api",
             "ocr_api_base": "http://ocr:8000", "ocr_api_model": "table-ocr"},
        ])
        cs = {c["name"]: c for c in api_configs(cfg)}
        self.assertEqual(cs["ocr"]["model"], "table-ocr")
        tmp.cleanup()

    def test_code_graph_config_item(self):
        """api_configs 包含 code_graph 项：disabled/enabled/missing 三态。"""
        tmp = tempfile.TemporaryDirectory()
        # 默认 enabled=False, base_url 空 → missing
        cfg = make_cfg(Path(tmp.name))
        cs = {c["name"]: c for c in api_configs(cfg)}
        self.assertIn("code_graph", cs)
        self.assertEqual(cs["code_graph"]["status"], "missing")
        self.assertEqual(cs["code_graph"]["provider"], "gh-puller")
        # enabled=True + base_url 有 → configured
        cfg.code_graph.enabled = True
        cfg.code_graph.base_url = "http://localhost:8787"
        cs = {c["name"]: c for c in api_configs(cfg)}
        self.assertEqual(cs["code_graph"]["status"], "configured")
        self.assertEqual(cs["code_graph"]["path"], "/gh-puller/graph")
        # enabled=False + base_url 有 → disabled
        cfg.code_graph.enabled = False
        cs = {c["name"]: c for c in api_configs(cfg)}
        self.assertEqual(cs["code_graph"]["status"], "disabled")
        tmp.cleanup()


class TestSecrets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data_root")
        (self.root / "data_root").mkdir(exist_ok=True)
        self.cfg = make_cfg(self.root)
        os.environ.pop("EMBEDDING_API_KEY", None)
        os.environ.pop("OCR_API_KEY", None)

    def tearDown(self):
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def test_save_and_load_secret(self):
        save_secret(self.cfg, "EMBEDDING_API_KEY", "sk-local-xyz")
        p = secrets_path(self.cfg)
        self.assertTrue(p.exists())
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(data["EMBEDDING_API_KEY"], "sk-local-xyz")
        # load 注入环境变量（未设置时）
        loaded = load_secrets(self.cfg)
        self.assertEqual(loaded.get("EMBEDDING_API_KEY"), "sk-local-xyz")
        self.assertEqual(os.environ.get("EMBEDDING_API_KEY"), "sk-local-xyz")
        # 环境变量已设置时不覆盖
        os.environ["EMBEDDING_API_KEY"] = "env-keep"
        self.assertEqual(load_secrets(self.cfg)["EMBEDDING_API_KEY"], "sk-local-xyz")
        self.assertEqual(os.environ["EMBEDDING_API_KEY"], "env-keep")
        # 空值删除
        save_secret(self.cfg, "EMBEDDING_API_KEY", "")
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertNotIn("EMBEDDING_API_KEY", data)

    def test_unsupported_key_rejected(self):
        with self.assertRaises(ValueError):
            save_secret(self.cfg, "FOO_KEY", "x")

    def test_appconfig_load_auto_injects_secrets(self):
        """AppConfig.load 自动加载 secrets 文件（build_kb/serve_api/review_ui 全入口覆盖）。"""
        cfg_path = self.root / "config.json"
        cfg_path.write_text(json.dumps({"embedding": {"provider": "echo"}}), encoding="utf-8")
        save_secret(self.cfg, "EMBEDDING_API_KEY", "sk-autoload")
        os.environ.pop("EMBEDDING_API_KEY", None)
        cfg2 = AppConfig.load(str(cfg_path), require_keys=False)
        # 环境变量被注入，effective_api_key 自动生效
        self.assertEqual(os.environ.get("EMBEDDING_API_KEY"), "sk-autoload")
        self.assertEqual(cfg2.embedding.effective_api_key, "sk-autoload")

    def test_appconfig_load_secrets_before_require_keys_validation(self):
        """secrets 必须在 require_keys 校验**之前**注入——否则 build_kb 写入路径看不到 key。"""
        # openai_compatible + 无 config key/env，但 secrets 文件里有 key：
        # require_keys=True 也应通过（secrets 先注入 env，校验可见）
        cfg_path = self.root / "config.json"
        cfg_path.write_text(json.dumps({
            "embedding": {"provider": "openai_compatible",
                          "base_url": "https://api.example.com/v1",
                          "api_key_env": "EMBEDDING_API_KEY"},
        }), encoding="utf-8")
        save_secret(self.cfg, "EMBEDDING_API_KEY", "sk-from-secrets")
        os.environ.pop("EMBEDDING_API_KEY", None)
        cfg2 = AppConfig.load(str(cfg_path), require_keys=True)  # 不应抛错
        self.assertEqual(cfg2.embedding.effective_api_key, "sk-from-secrets")

    def test_appconfig_load_secret_absent_no_error(self):
        """无 secrets 文件时 load 不报错（静默）。"""
        cfg_path = self.root / "config.json"
        cfg_path.write_text(json.dumps({"embedding": {"provider": "echo"}}), encoding="utf-8")
        cfg2 = AppConfig.load(str(cfg_path), require_keys=False)
        self.assertEqual(cfg2.embedding.provider, "echo")

    def test_update_config_json_embedding(self):
        cfg_path = self.root / "cfg.json"
        cfg_path.write_text(json.dumps({"embedding": {"provider": "openai_compatible",
                                                      "base_url": "old", "model": "m",
                                                      "api_key_env": "EMBEDDING_API_KEY"},
                                        "sources": []}), encoding="utf-8")
        update_config_json(self.cfg, "embedding", {"base_url": "https://new/v1", "model": "bge-m3"},
                           config_path=cfg_path)
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(data["embedding"]["base_url"], "https://new/v1")
        self.assertEqual(data["embedding"]["model"], "bge-m3")

    def test_update_config_json_ocr_missing_source(self):
        cfg_path = self.root / "cfg.json"
        cfg_path.write_text(json.dumps({"sources": []}), encoding="utf-8")
        with self.assertRaises(ValueError):
            update_config_json(self.cfg, "ocr", {"ocr_provider": "paddle"}, config_path=cfg_path)

    def test_update_config_json_code_graph(self):
        """code_graph 段可写（回归：review UI 编辑保存曾因不支持该段抛 ValueError
        → 500 纯文本 → 前端 r.json() 报 unexpected token）。"""
        cfg_path = self.root / "cfg.json"
        cfg_path.write_text(json.dumps({"code_graph": {"enabled": False,
                                                       "base_url": "", "path": "/old"},
                                        "sources": []}), encoding="utf-8")
        update_config_json(self.cfg, "code_graph",
                           {"enabled": True, "base_url": "http://gh-puller:8080",
                            "path": "/api/graph"},
                           config_path=cfg_path)
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(data["code_graph"]["enabled"], True)
        self.assertEqual(data["code_graph"]["base_url"], "http://gh-puller:8080")
        self.assertEqual(data["code_graph"]["path"], "/api/graph")

    def test_update_config_json_code_graph_creates_section(self):
        # config.json 原本无 code_graph 段：setdefault 创建
        cfg_path = self.root / "cfg.json"
        cfg_path.write_text(json.dumps({"sources": []}), encoding="utf-8")
        update_config_json(self.cfg, "code_graph", {"enabled": True}, config_path=cfg_path)
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(data["code_graph"], {"enabled": True})

    def test_update_config_json_unknown_section(self):
        cfg_path = self.root / "cfg.json"
        cfg_path.write_text(json.dumps({"sources": []}), encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            update_config_json(self.cfg, "nope", {}, config_path=cfg_path)
        self.assertIn("code_graph", str(cm.exception))  # 错误信息列出全部支持段


class TestOcrConnectivity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data_root")
        (self.root / "data_root").mkdir(exist_ok=True)
        self.cfg = make_cfg(self.root)

    def tearDown(self):
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def _cfg_with_image(self, **fields):
        return AppConfig.model_validate({
            "sources": [{"id": "images", "type": "image", "enabled": True, **fields}],
            "embedding": {"provider": "echo"},
            "storage": {"sqlite_path": str(self.root / "kb.sqlite3"),
                        "canonical_file": str(self.root / "c.jsonl")},
        })

    @mock.patch("vllm_kb.ocr.ocr_image", return_value=("vllm-kb OCR 123", 0.95))
    def test_api_ok(self, _m):
        r = probe_ocr_connectivity(self._cfg_with_image(
            ocr_provider="api", ocr_api_base="http://ocr:8000", ocr_api_model="table"))
        self.assertTrue(r["ok"])
        self.assertIn("vllm-kb OCR 123", r["detail"])
        _m.assert_called_once()
        self.assertEqual(_m.call_args.kwargs.get("model"), "table")

    @mock.patch("vllm_kb.ocr.ocr_image", side_effect=OcrApiError("conn refused"))
    def test_api_failure(self, _m):
        r = probe_ocr_connectivity(self._cfg_with_image(ocr_provider="api", ocr_api_base="http://x:1"))
        self.assertFalse(r["ok"])
        self.assertIn("conn refused", r["detail"])

    def test_no_image_source(self):
        with self.assertRaises(ValueError):
            probe_ocr_connectivity(make_cfg(self.root))

    def test_provider_none(self):
        with self.assertRaises(ValueError):
            probe_ocr_connectivity(self._cfg_with_image(ocr_provider="none"))

    def test_ask_without_api(self):
        with self.assertRaises(ValueError):
            probe_ocr_connectivity(self._cfg_with_image(ocr_provider="ask"))

    @mock.patch("vllm_kb.ocr.ocr_image", return_value=("", 0.0))
    def test_paddle_local(self, _m):
        r = probe_ocr_connectivity(self._cfg_with_image(ocr_provider="paddle"))
        self.assertTrue(r["ok"])


if __name__ == "__main__":
    unittest.main()
