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
    seed_case_title_flags,
    seed_verification_pending,
    test_ocr_connectivity,
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
        r = test_ocr_connectivity(self._cfg_with_image(
            ocr_provider="api", ocr_api_base="http://ocr:8000", ocr_api_model="table"))
        self.assertTrue(r["ok"])
        self.assertIn("vllm-kb OCR 123", r["detail"])
        _m.assert_called_once()
        self.assertEqual(_m.call_args.kwargs.get("model"), "table")

    @mock.patch("vllm_kb.ocr.ocr_image", side_effect=OcrApiError("conn refused"))
    def test_api_failure(self, _m):
        r = test_ocr_connectivity(self._cfg_with_image(ocr_provider="api", ocr_api_base="http://x:1"))
        self.assertFalse(r["ok"])
        self.assertIn("conn refused", r["detail"])

    def test_no_image_source(self):
        with self.assertRaises(ValueError):
            test_ocr_connectivity(make_cfg(self.root))

    def test_provider_none(self):
        with self.assertRaises(ValueError):
            test_ocr_connectivity(self._cfg_with_image(ocr_provider="none"))

    def test_ask_without_api(self):
        with self.assertRaises(ValueError):
            test_ocr_connectivity(self._cfg_with_image(ocr_provider="ask"))

    @mock.patch("vllm_kb.ocr.ocr_image", return_value=("", 0.0))
    def test_paddle_local(self, _m):
        r = test_ocr_connectivity(self._cfg_with_image(ocr_provider="paddle"))
        self.assertTrue(r["ok"])


if __name__ == "__main__":
    unittest.main()
