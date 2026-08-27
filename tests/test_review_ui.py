"""审核工作台 API 测试（FastAPI TestClient）+ UI 防重复表单审计。

覆盖此前只有手工验证、无自动化测试的 API 层：
stats / queue / item / review / configs / configs-save / configs-test / 页面 HTML；
以及"多次点击编辑配置不重复建表单"的源码级审计。
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# scripts/ 非包：按路径加载 review_ui 模块
_spec = importlib.util.spec_from_file_location("review_ui", PROJECT_ROOT / "scripts" / "review_ui.py")
review_ui = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(review_ui)

from fastapi.testclient import TestClient  # noqa: E402


def make_env(tmp: Path) -> tuple[Path, Path]:
    """构造临时数据根（VLLM_KB_DATA_ROOT）与临时 config.json。"""
    data_root = tmp / "data_root"
    data_root.mkdir()
    cfg_path = tmp / "config.json"
    cfg_path.write_text(json.dumps({
        "embedding": {"provider": "openai_compatible", "base_url": "https://api.example.com/v1",
                      "model": "bge-m3", "api_key_env": "EMBEDDING_API_KEY"},
        "sources": [
            {"id": "github", "type": "github", "token_env": "GITHUB_TOKEN", "enabled": False},
            {"id": "images", "type": "image", "ocr_provider": "ask",
             "ocr_api_base": "", "enabled": True},
        ],
        "storage": {"sqlite_path": "data/kb.sqlite3", "canonical_file": "data/raw/canonical.jsonl"},
    }), encoding="utf-8")
    return data_root, cfg_path


class ReviewUiApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_root, self.cfg_path = make_env(self.root)
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.data_root)
        for k in ("EMBEDDING_API_KEY", "OCR_API_KEY", "GITHUB_TOKEN"):
            os.environ.pop(k, None)
        self.client = TestClient(review_ui.create_app(str(self.cfg_path), auto_seed=False))

    def tearDown(self):
        self.client.close()
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        for k in ("EMBEDDING_API_KEY", "OCR_API_KEY", "GITHUB_TOKEN"):
            os.environ.pop(k, None)
        self.tmp.cleanup()

    def test_page_html(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        html = r.text
        for token in ("vllm-kb 审核工作台", "编辑配置", "saveConfig", "loadQueue", "review("):
            self.assertIn(token, html, f"页面缺少 {token}")
        # OCR 表单必须含模型名字段与测试连通入口（此前缺失）
        self.assertIn("ocr_api_model", html)
        self.assertIn("ocr_provider", html)
        self.assertIn("testApi('ocr'", html)
        # 标签管理 tab / 文档标签编辑 / 候选采纳
        for token in ("loadTagDict", "tagEdit(", "adoptCandidate", "tagDictAdd", "标签管理"):
            self.assertIn(token, html, f"页面缺少 {token}")

    def test_docs_tags_edit_flow(self):
        """标签编辑端点：exclude/restore/add/remove + final 同步（检索侧立即生效）+ 词典同步。"""
        from vllm_kb.config import AppConfig
        import sqlite3

        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        kb = cfg.resolve(cfg.storage.sqlite_path)
        kb.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(kb)
        conn.execute("CREATE TABLE docs (source_id TEXT PRIMARY KEY, source_type TEXT, title TEXT, extra TEXT)")
        conn.execute("INSERT INTO docs VALUES(?,?,?,?)",
                     ("md:case1", "doc_markdown", "案例",
                      json.dumps({"verification": "unverified"})))
        conn.commit()
        conn.close()
        # 缺 reviewer → 400
        r = self.client.post("/api/docs/tags/edit",
                             json={"source_id": "md:case1", "action": "exclude", "tag": "HCCL"})
        self.assertEqual(r.status_code, 400)
        # 排除自动标签（最终 = (auto − excluded) ∪ manual）
        r = self.client.post("/api/docs/tags/edit",
                             json={"source_id": "md:case1", "action": "exclude",
                                   "tag": "HCCL", "reviewer": "t"})
        self.assertEqual(r.status_code, 200)
        # 人工添加（不在词典 → 自动同步词典到临时 config）
        r = self.client.post("/api/docs/tags/edit",
                             json={"source_id": "md:case1", "action": "add",
                                   "tag": "命令参考", "reviewer": "t"})
        self.assertEqual(r.status_code, 200)
        v = self.client.get("/api/docs/tags", params={"source_id": "md:case1"}).json()
        self.assertEqual(v["excluded"], ["HCCL"])
        self.assertEqual(v["manual"], ["命令参考"])
        self.assertEqual(v["final"], ["命令参考"])  # auto 空 → 只含人工
        cfg_json = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        names = [x["name"] for x in cfg_json["tags"]["registry"]]
        self.assertIn("命令参考", names)  # 词典同步
        # 恢复 + 删除人工
        self.client.post("/api/docs/tags/edit",
                         json={"source_id": "md:case1", "action": "restore",
                               "tag": "HCCL", "reviewer": "t"})
        v = self.client.get("/api/docs/tags", params={"source_id": "md:case1"}).json()
        self.assertEqual(v["excluded"], [])
        self.client.post("/api/docs/tags/edit",
                         json={"source_id": "md:case1", "action": "remove",
                               "tag": "命令参考", "reviewer": "t"})
        v = self.client.get("/api/docs/tags", params={"source_id": "md:case1"}).json()
        self.assertEqual(v["manual"], [])

    def test_tag_dict_api(self):
        """词典端点：add/rename/tier/delete（写临时 config）+ 视图分组。"""
        import sqlite3

        from vllm_kb.config import AppConfig

        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        kb = cfg.resolve(cfg.storage.sqlite_path)
        kb.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(kb)
        conn.execute("CREATE TABLE docs (source_id TEXT PRIMARY KEY, source_type TEXT, title TEXT, extra TEXT)")
        conn.execute("INSERT INTO docs VALUES(?,?,?,?)",
                     ("md:case1", "doc_markdown", "案例", json.dumps({})))
        conn.commit()
        conn.close()
        # add
        r = self.client.post("/api/tag-dict/add", json={"name": "网络", "tier": "domain"})
        self.assertEqual(r.status_code, 200)
        cfg_json = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        self.assertIn({"name": "网络", "tier": "domain"}, cfg_json["tags"]["registry"])
        # 视图（以临时 config 为准）
        td = self.client.get("/api/tag-dict").json()
        self.assertIn("groups", td)
        self.assertEqual(td["groups"]["domain"][0]["name"], "网络")
        # rename
        r = self.client.post("/api/tag-dict/rename", json={"old": "网络", "new": "组网"})
        self.assertEqual(r.status_code, 200)
        cfg_json = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        names = [x["name"] for x in cfg_json["tags"]["registry"]]
        self.assertIn("组网", names)
        self.assertNotIn("网络", names)
        # tier 修改
        r = self.client.post("/api/tag-dict/tier", json={"name": "组网", "tier": "purpose"})
        self.assertEqual(r.status_code, 200)
        td = self.client.get("/api/tag-dict").json()
        self.assertEqual(td["groups"]["purpose"][0]["name"], "组网")
        # 删除
        r = self.client.post("/api/tag-dict/delete", json={"name": "组网"})
        self.assertEqual(r.status_code, 200)
        td = self.client.get("/api/tag-dict").json()
        self.assertEqual(td["groups"]["domain"] + td["groups"]["purpose"], [])

    def test_stats_queue_item_review_flow(self):
        # stats 初始为空
        self.assertEqual(self.client.get("/api/stats").json(), {})
        # 直接向审核库写一条（模拟 seed/导入自动补单）——须经 cfg 解析（VLLM_KB_DATA_ROOT 重定向）
        from vllm_kb.config import AppConfig
        from vllm_kb.review import ReviewStore, default_review_path

        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        store = ReviewStore(default_review_path(cfg))
        store.add_item("verification_pending", "md:doc1", {"title": "待审核文档"})
        store.close()
        # stats
        st = self.client.get("/api/stats").json()
        self.assertEqual(st["verification_pending"]["pending"], 1)
        # queue
        q = self.client.get("/api/queue").json()
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["item_ref"], "md:doc1")
        # item 详情
        it = self.client.get("/api/item/1").json()
        self.assertEqual(it["payload"]["title"], "待审核文档")
        # review 提交（缺 reviewer → 400）
        r = self.client.post("/api/item/1/review", json={"action": "approved"})
        self.assertEqual(r.status_code, 400)
        # 认证
        r = self.client.post("/api/item/1/review",
                             json={"action": "approved", "reviewer": "tester",
                                   "result": {"note": "ok"}})
        self.assertEqual(r.status_code, 200)
        it = self.client.get("/api/item/1").json()
        self.assertEqual(it["status"], "approved")
        self.assertEqual(it["reviewer"], "tester")
        # 非法动作（旧 confirmed/删除等）→ 400
        for bad in ("confirmed", "rejected", "deleted"):
            r = self.client.post("/api/item/1/review", json={"action": bad, "reviewer": "x"})
            self.assertEqual(r.status_code, 400)
        # 不存在的 item
        self.assertEqual(self.client.get("/api/item/999").status_code, 404)

    def test_delete_undo_flow(self):
        """标记删除（只删 kb 记录）+ 撤回（恢复+重新入队）API 闭环。"""
        from vllm_kb.config import AppConfig
        from vllm_kb.review import ReviewStore, default_review_path
        import sqlite3

        # 造最小 kb.sqlite3 + 审核项
        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        kb = cfg.resolve(cfg.storage.sqlite_path)
        kb.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(kb)
        conn.execute("CREATE TABLE docs (source_id TEXT PRIMARY KEY, source_type TEXT, title TEXT, extra TEXT)")
        conn.execute("INSERT INTO docs VALUES(?,?,?,?)",
                     ("md:case1", "doc_markdown", "案例",
                      json.dumps({"verification": "unverified", "asset": {"path": "assets/md/case1.md"}})))
        conn.commit()
        conn.close()
        store = ReviewStore(default_review_path(cfg))
        store.add_item("verification_pending", "md:case1", {"title": "案例"})
        store.close()
        # 标记删除 → kb 行消失
        r = self.client.post("/api/item/1/delete", json={"reviewer": "tester"})
        self.assertEqual(r.status_code, 200)
        conn = sqlite3.connect(kb)
        self.assertIsNone(conn.execute("SELECT source_id FROM docs WHERE source_id='md:case1'").fetchone())
        conn.close()
        it = self.client.get("/api/item/1").json()
        self.assertEqual(it["status"], "deleted")
        # 待实际删除列表（status=deleted）
        del_q = self.client.get("/api/queue?status=deleted").json()
        self.assertEqual(len(del_q), 1)
        self.assertEqual(del_q[0]["payload"]["asset"]["path"], "assets/md/case1.md")
        # 撤回 → kb 行恢复 + 重新入队
        r = self.client.post("/api/item/1/undo")
        self.assertEqual(r.status_code, 200)
        conn = sqlite3.connect(kb)
        row = conn.execute("SELECT title FROM docs WHERE source_id='md:case1'").fetchone()
        conn.close()
        self.assertEqual(row[0], "案例")
        it = self.client.get("/api/item/1").json()
        self.assertEqual(it["status"], "pending")

    def test_configs_list_redacted(self):
        cs = self.client.get("/api/configs").json()
        by_name = {c["name"]: c for c in cs}
        self.assertIn("embedding", by_name)
        self.assertIn("ocr", by_name)
        self.assertIn("github", by_name)
        self.assertEqual(by_name["ocr"]["provider"], "ask")
        # base_url/provider 是非密钥明文（应显示）；**密钥不出现**
        self.assertEqual(by_name["embedding"]["base_url"], "https://api.example.com/v1")
        blob = json.dumps(cs)
        self.assertNotIn("sk-", blob)
        self.assertNotIn("api_key", blob)

    def test_configs_save_non_secret(self):
        r = self.client.post("/api/configs/save",
                             json={"name": "embedding",
                                   "fields": {"model": "bge-m3-large", "base_url": "https://new/v1"}})
        self.assertEqual(r.status_code, 200)
        data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(data["embedding"]["model"], "bge-m3-large")
        self.assertEqual(data["embedding"]["base_url"], "https://new/v1")

    def test_configs_save_secret_goes_to_secrets_file(self):
        r = self.client.post("/api/configs/save",
                             json={"name": "embedding", "fields": {"api_key": "sk-test-123"}})
        self.assertEqual(r.status_code, 200)
        secrets_file = self.data_root / "secrets.local.json"
        self.assertTrue(secrets_file.exists())
        data = json.loads(secrets_file.read_text(encoding="utf-8"))
        self.assertEqual(data["EMBEDDING_API_KEY"], "sk-test-123")
        # config.json 不含密钥
        cfg = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        self.assertNotIn("sk-test-123", json.dumps(cfg))

    def test_configs_save_unknown_name(self):
        r = self.client.post("/api/configs/save", json={"name": "nope", "fields": {}})
        self.assertEqual(r.status_code, 400)

    def test_configs_save_refreshes_status(self):
        """保存后 /api/configs 立即反映新状态（内存 cfg 重载，无需重启页面/服务）。"""
        cs = {c["name"]: c for c in self.client.get("/api/configs").json()}
        self.assertEqual(cs["embedding"]["status"], "missing")
        # 保存 api_key → secrets 文件 + 重载注入 env → 状态变 configured
        r = self.client.post("/api/configs/save",
                             json={"name": "embedding", "fields": {"api_key": "sk-live-1"}})
        self.assertEqual(r.status_code, 200)
        cs2 = {c["name"]: c for c in self.client.get("/api/configs").json()}
        self.assertEqual(cs2["embedding"]["status"], "configured")
        self.assertTrue(cs2["embedding"]["key_configured"])
        # 保存非密钥字段 → base_url/model 立即反映
        self.client.post("/api/configs/save",
                         json={"name": "embedding",
                               "fields": {"model": "bge-m3-live", "base_url": "https://live/v1"}})
        cs3 = {c["name"]: c for c in self.client.get("/api/configs").json()}
        self.assertEqual(cs3["embedding"]["model"], "bge-m3-live")
        self.assertEqual(cs3["embedding"]["base_url"], "https://live/v1")

    def test_configs_save_ocr_model(self):
        """保存 OCR 模型名（ocr_api_model）→ config.json 与 /api/configs 反映。"""
        r = self.client.post("/api/configs/save",
                             json={"name": "ocr",
                                   "fields": {"ocr_api_base": "http://ocr:8000",
                                              "ocr_api_model": "table-ocr"}})
        self.assertEqual(r.status_code, 200)
        cfg = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        img = next(s for s in cfg["sources"] if s["type"] == "image")
        self.assertEqual(img["ocr_api_model"], "table-ocr")
        self.assertEqual(img["ocr_api_base"], "http://ocr:8000")
        cs = {c["name"]: c for c in self.client.get("/api/configs").json()}
        self.assertEqual(cs["ocr"]["model"], "table-ocr")

    def test_configs_test_ocr_connectivity(self):
        """OCR 连通性测试端点：mock 引擎成功 → 200；未配置 → 400；失败 → 502。"""
        from unittest import mock as _mock

        # 未配置 api_base（ask）→ 400
        r = self.client.post("/api/configs/test", json={"name": "ocr"})
        self.assertEqual(r.status_code, 400)
        # 配置 api 后 mock 引擎 → 200
        self.client.post("/api/configs/save", json={"name": "ocr",
                                                    "fields": {"ocr_api_base": "http://ocr:8000"}})
        with _mock.patch("vllm_kb.ocr.ocr_image", return_value=("vllm-kb OCR 123", 0.9)):
            r = self.client.post("/api/configs/test", json={"name": "ocr"})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertTrue(d["ok"])
        self.assertIn("vllm-kb OCR 123", d["detail"])
        # 引擎失败 → 502
        from vllm_kb.ocr import OcrApiError
        with _mock.patch("vllm_kb.ocr.ocr_image", side_effect=OcrApiError("down")):
            r = self.client.post("/api/configs/test", json={"name": "ocr"})
        self.assertEqual(r.status_code, 502)

    def test_configs_test_unknown_name(self):
        r = self.client.post("/api/configs/test", json={"name": "nope"})
        self.assertEqual(r.status_code, 400)


class ReviewUiHtmlAudit(unittest.TestCase):
    """源码级审计：多次点击"编辑配置"不会堆积多个表单（此前 UI 缺陷）。"""

    def test_edit_config_removes_existing_form(self):
        src = (PROJECT_ROOT / "scripts" / "review_ui.py").read_text(encoding="utf-8")
        fn = src[src.index("function editConfig"):src.index("async function saveConfig")]
        # 必须：进入时先移除已存在的同名表单
        self.assertIn("document.getElementById('form-'+name)", fn)
        self.assertIn(".remove()", fn)

    def test_save_config_uses_config_path(self):
        """save 端点应把 config 写入 create_app 传入的 config 路径（可测试、不写死项目根）。"""
        src = (PROJECT_ROOT / "scripts" / "review_ui.py").read_text(encoding="utf-8")
        self.assertIn("config_path=_config_path", src)


if __name__ == "__main__":
    unittest.main()
