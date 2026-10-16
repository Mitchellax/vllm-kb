"""审核工作台 API 测试（FastAPI TestClient）+ UI 防重复表单审计。

覆盖此前只有手工验证、无自动化测试的 API 层：
stats / queue / item / review / configs / configs-save / configs-test / 页面 HTML；
以及"多次点击编辑配置不重复建表单"的源码级审计。
"""
import importlib.util
import json
import os
import sqlite3
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


def seed_kb_doc(cfg_path: Path, source_id: str = "md:doc1", *,
                source_type: str = "doc_markdown", verification: str = "unverified") -> None:
    """在 kb.sqlite3 里造一条外源文档。

    补标回写需要真实的 docs 行（要写覆盖层并刷新 extra/reliability），
    只往审核库塞队列项是不够的。
    """
    from vllm_kb.config import AppConfig
    from vllm_kb.ingest import _connect

    cfg = AppConfig.load(str(cfg_path), require_keys=False)
    conn = _connect(cfg.resolve(cfg.storage.sqlite_path))
    conn.execute(
        "INSERT OR REPLACE INTO docs (source_id, source_type, title, status, reliability, extra) "
        "VALUES (?,?,?,?,?,?)",
        (source_id, source_type, "待审核文档", "open", 0.5,
         json.dumps({"verification": verification}, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


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
        # 审核人记忆（localStorage 回填，免每条重复填写）
        for token in ("loadReviewer", "saveReviewer", "oninput=\"saveReviewer(this.value)\""):
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
        seed_kb_doc(self.cfg_path)      # 补标要写回 docs，得有真实文档
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
        # 认证：**补标项要真的改可靠度**（覆盖层 + docs.extra 立即刷新）
        r = self.client.post("/api/item/1/review",
                             json={"action": "approved", "reviewer": "tester",
                                   "result": {"note": "ok"}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["verification"], "tested")
        self.assertEqual(r.json()["reliability"], 0.85)
        conn = sqlite3.connect(cfg.resolve(cfg.storage.sqlite_path))
        try:
            extra, rel = conn.execute(
                "SELECT extra, reliability FROM docs WHERE source_id='md:doc1'").fetchone()
        finally:
            conn.close()
        self.assertEqual(json.loads(extra)["verification"], "tested")
        self.assertEqual(rel, 0.85)
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

    def test_doc_preview(self):
        """补标详情的正文预览：返回正文；未知文档 404（审核人先看到内容再判定）。"""
        seed_kb_doc(self.cfg_path, "md:doc1")
        r = self.client.get("/api/doc-preview", params={"source_id": "md:doc1"})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["source_id"], "md:doc1")
        self.assertIn("body", d)
        self.assertEqual(r.status_code, 200)
        r2 = self.client.get("/api/doc-preview", params={"source_id": "md:ghost"})
        self.assertEqual(r2.status_code, 404)

    def test_file_level_review_flow_http(self):
        """表格文件级审核（HTTP 层）：seed 聚合 → 认证一次 → 整表生效。"""
        seed_kb_doc(self.cfg_path, "excel:t:Sheet1:1", source_type="doc_excel")
        seed_kb_doc(self.cfg_path, "excel:t:Sheet1:2", source_type="doc_excel")
        from vllm_kb.config import AppConfig
        from vllm_kb.review import ReviewStore, default_review_path, seed_verification_pending

        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        store = ReviewStore(default_review_path(cfg))
        seed_verification_pending(cfg, store)
        fitems = [i for i in store.list_items(category="verification_pending")
                  if i["item_ref"].startswith("excelfile:")]
        self.assertEqual(len(fitems), 1)
        self.assertEqual(fitems[0]["payload"]["row_count"], 2)
        r = self.client.post(f"/api/item/{fitems[0]['id']}/review", json={
            "action": "approved", "reviewer": "alice", "result": {"verification": "expert"}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["rows_verified"], 2)
        conn = sqlite3.connect(cfg.resolve(cfg.storage.sqlite_path))
        try:
            for sid in ("excel:t:Sheet1:1", "excel:t:Sheet1:2"):
                extra = json.loads(conn.execute(
                    "SELECT extra FROM docs WHERE source_id=?", (sid,)).fetchone()[0])
                self.assertEqual(extra["verification"], "expert")
        finally:
            conn.close()
        store.close()

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

    def test_configs_save_code_graph(self):
        """code_graph 编辑保存（回归：update_config_json 曾不支持该段 → 未接的
        ValueError 变纯文本 500 → 前端 r.json() 报 unexpected token 无法保存）。"""
        r = self.client.post("/api/configs/save",
                             json={"name": "code_graph",
                                   "fields": {"enabled": "true",
                                              "base_url": "http://gh-puller:8080",
                                              "path": "/api/graph"}})
        self.assertEqual(r.status_code, 200)
        data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        # config.json 段被创建/更新（enabled 字符串转 bool）
        self.assertEqual(data["code_graph"]["enabled"], True)
        self.assertEqual(data["code_graph"]["base_url"], "http://gh-puller:8080")
        self.assertEqual(data["code_graph"]["path"], "/api/graph")
        # /api/configs 立即反映（enabled+base_url → configured）
        cs = {c["name"]: c for c in self.client.get("/api/configs").json()}
        self.assertEqual(cs["code_graph"]["status"], "configured")
        self.assertEqual(cs["code_graph"]["base_url"], "http://gh-puller:8080")

    def test_configs_save_backend_valueerror_returns_json_400(self):
        """后端 ValueError（如 ocr 无 image 来源）必须返回 JSON 400 而非纯文本 500
        ——前端 j() 对非 JSON 响应报 unexpected token。"""
        # 临时把 config.json 的 sources 换成无 image 条目
        data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        data["sources"] = [{"id": "github", "type": "github", "enabled": False}]
        self.cfg_path.write_text(json.dumps(data), encoding="utf-8")
        r = self.client.post("/api/configs/save",
                             json={"name": "ocr", "fields": {"ocr_provider": "paddle"}})
        self.assertEqual(r.status_code, 400)
        # 响应是 JSON（前端可读 detail），不是纯文本
        body = r.json()
        self.assertIn("image", body["detail"])

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

    def test_configs_ocr_connectivity(self):
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

    def test_tag_dict_edit_uses_modal(self):
        """标签管理的编辑框必须是模态对话框（此前 appendChild 到页面末尾、.card subtle
        样式与列表卡片相同——唯一正在编辑项看上去只是又一个列表项）。"""
        src = (PROJECT_ROOT / "scripts" / "review_ui.py").read_text(encoding="utf-8")
        self.assertIn('id="modal"', src)      # 模态容器存在
        self.assertIn("function openModal(", src)
        self.assertIn("function closeModal(", src)
        fn = src[src.index("function tagDictForm"):src.index("async function tagDictRename")]
        self.assertIn("openModal(", fn)
        self.assertNotIn("appendChild(", fn)  # 不再插进页面流
        self.assertIn("classList.add('editing')", fn)  # 被编辑芯片高亮
        # 三个动作成功后都必须关闭对话框
        import re as _re

        for fname in ("tagDictRename", "tagDictTier", "tagDictDelete"):
            m = _re.search(rf"async function {fname}\([\s\S]*?\nasync function", src)
            self.assertIsNotNone(m, fname)
            self.assertIn("closeModal()", m.group(0), fname)

    def test_queue_list_shows_candidate_for_tag_items(self):
        """队列列表：tag_candidate 条目必须显示候选词本身。

        此前列表裸用 payload.title——那是"兼容旧单文档"的首文档标题，多手册同产品名时
        （4 篇 Atlas A3 手册）四个不同候选在列表里全显示成同一个产品名。
        """
        src = (PROJECT_ROOT / "scripts" / "review_ui.py").read_text(encoding="utf-8")
        self.assertIn("const itemTitle=", src)
        fn = src[src.index("const itemTitle="):src.index("async function showItem")]
        self.assertIn("category==='tag_candidate'", fn)
        self.assertIn("p.candidate||i.item_ref", fn)
        # 队列 pending 列表必须走 itemTitle，不允许裸拼 payload.title（待删除列表不受限——
        # 只含文档类条目）
        qmap = src[src.index("+q.map(i=>"):src.index("待实际删除")]
        self.assertIn("itemTitle(i)", qmap)
        self.assertNotIn("(i.payload||{}).title||i.item_ref", qmap)

    def test_seed_tag_candidate_payload_has_no_legacy_title(self):
        """新 seed 的 tag_candidate payload 不再携带首文档 title/source_id（展示陷阱字段）；
        候选名聚合前过 normalize_tag——存量 extras 的空格变体收敛为同一聚合键。"""
        import json as _json
        import tempfile as _tempfile
        from pathlib import Path as _Path

        from vllm_kb.config import AppConfig
        from vllm_kb.ingest import _connect
        from vllm_kb.review import ReviewStore, default_review_path, seed_tag_candidates

        with _tempfile.TemporaryDirectory() as td:
            root = _Path(td)
            cfg_path = root / "config.json"
            cfg_path.write_text(_json.dumps({
                "embedding": {"provider": "echo", "model": "echo"},
                "sources": [{"id": "images", "type": "image", "ocr_provider": "ask",
                             "ocr_api_base": "", "enabled": True}],
                "storage": {"sqlite_path": "data/kb.sqlite3",
                            "canonical_file": "data/raw/canonical.jsonl"},
            }, ensure_ascii=False), encoding="utf-8")
            os.environ["VLLM_KB_DATA_ROOT"] = str(root / "data_root")
            try:
                (root / "data_root").mkdir()
                cfg = AppConfig.load(str(cfg_path), require_keys=False)
                conn = _connect(cfg.resolve(cfg.storage.sqlite_path))
                # 两篇文档携带同一候选的空格变体（上次构建的 extras 快照常态）
                for i, (sid, name) in enumerate([
                    ("pdf:某手册A 26.1.x 命令参考 02", "配置pip 源"),
                    ("pdf:某手册B 26.1.x 命令参考 01", "配置pip源"),
                ]):
                    conn.execute(
                        "INSERT OR REPLACE INTO docs (source_id, source_type, title, status, extra) "
                        "VALUES (?,?,?,?,?)",
                        (sid, "doc_pdf", f"某产品名{i}", "open",
                         _json.dumps({"tag_candidates": [{"name": name, "tier": "purpose"}]},
                                     ensure_ascii=False)),
                    )
                conn.commit()
                conn.close()
                store = ReviewStore(default_review_path(cfg))
                seed_tag_candidates(cfg, store)
                items = store.list_items(category="tag_candidate")
                self.assertEqual(len(items), 1)  # 变体收敛 → 单条
                it = items[0]
                self.assertEqual(it["item_ref"], "tag:配置pip源")
                p = it["payload"]
                self.assertEqual(p["candidate"], "配置pip源")
                self.assertEqual(p["doc_count"], 2)
                self.assertNotIn("title", p)
                self.assertNotIn("source_id", p)
                # clear_category：提取规则升级后重播用（全状态删除 + 类别校验）
                self.assertEqual(store.clear_category("tag_candidate"), 1)
                self.assertEqual(store.list_items(category="tag_candidate"), [])
                with self.assertRaises(ValueError):
                    store.clear_category("bogus")
                store.close()
            finally:
                os.environ.pop("VLLM_KB_DATA_ROOT", None)


class ReviewUiBatchApiTest(ReviewUiApiTest):
    """批量操作端点：逐项独立生效、部分失败返回明细（不跨文档整体回滚）。"""

    def _seed_two_docs(self, **kw):
        seed_kb_doc(self.cfg_path, "md:a1", **kw)
        seed_kb_doc(self.cfg_path, "md:a2", **kw)

    def test_docs_tags_batch_partial_success(self):
        from vllm_kb.config import AppConfig
        import sqlite3

        self._seed_two_docs()
        r = self.client.post("/api/docs/tags/batch", json={
            "source_ids": ["md:a1", "md:a2", "md:ghost"],
            "action": "add", "tag": "排故手册", "reviewer": "张三"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["updated"], 2)
        self.assertEqual([f["source_id"] for f in body["failed"]], ["md:ghost"])
        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        conn = sqlite3.connect(cfg.resolve(cfg.storage.sqlite_path))
        manual = dict(conn.execute("SELECT source_id, manual FROM doc_tags").fetchall())
        conn.close()
        self.assertEqual(json.loads(manual["md:a1"]), ["排故手册"])
        self.assertEqual(json.loads(manual["md:a2"]), ["排故手册"])

    def test_docs_tags_batch_requires_reviewer_and_ids(self):
        self.assertEqual(self.client.post("/api/docs/tags/batch", json={
            "source_ids": ["md:a1"], "action": "add", "tag": "x",
            "reviewer": ""}).status_code, 400)
        self.assertEqual(self.client.post("/api/docs/tags/batch", json={
            "source_ids": [], "action": "add", "tag": "x",
            "reviewer": "r"}).status_code, 400)

    def test_external_docs_delete_batch(self):
        from vllm_kb.config import AppConfig
        import sqlite3

        self._seed_two_docs()
        r = self.client.post("/api/external-docs/delete-batch",
                             json={"source_ids": ["md:a1", "md:a2", "md:ghost"]})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["deleted"], 2)
        self.assertEqual([f["source_id"] for f in body["failed"]], ["md:ghost"])
        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        conn = sqlite3.connect(cfg.resolve(cfg.storage.sqlite_path))
        n = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        conn.close()
        self.assertEqual(n, 0)

    def test_items_review_batch_verification(self):
        """补标项批量认证：写覆盖层并立即刷新 docs.extra.verification（w_rel 生效前提）。"""
        from vllm_kb.config import AppConfig
        from vllm_kb.review import ReviewStore, default_review_path
        import sqlite3

        seed_kb_doc(self.cfg_path, "md:doc1", verification="unverified")
        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        store = ReviewStore(default_review_path(cfg))
        store.add_item("verification_pending", "md:doc1", {"title": "待审核文档"})
        item = store.list_items(category="verification_pending")[0]
        store.close()
        r = self.client.post("/api/items/review-batch", json={
            "item_ids": [item["id"], 999999], "action": "approved",
            "reviewer": "李四", "result": {"verification": "tested"}})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["reviewed"], 1)
        self.assertEqual([f["item_id"] for f in body["failed"]], [999999])
        conn = sqlite3.connect(cfg.resolve(cfg.storage.sqlite_path))
        ver = json.loads(conn.execute(
            "SELECT extra FROM docs WHERE source_id='md:doc1'").fetchone()[0])["verification"]
        conn.close()
        self.assertEqual(ver, "tested")

    def test_external_docs_rel_dir_passthrough(self):
        """目录分组元数据：extra.rel_src/rel_dir 透出到管理员侧列表（检索出口白名单不透出）。"""
        from vllm_kb.config import AppConfig
        from vllm_kb.ingest import _connect

        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        conn = _connect(cfg.resolve(cfg.storage.sqlite_path))
        conn.executemany(
            "INSERT OR REPLACE INTO docs (source_id, source_type, title, status, reliability, extra) "
            "VALUES (?,?,?,?,?,?)",
            [("md:组内", "doc_markdown", "组内文档", "open", 0.5,
              json.dumps({"verification": "unverified", "rel_src": "md1", "rel_dir": "故障"},
                         ensure_ascii=False)),
             ("md:回退", "doc_markdown", "回退文档", "open", 0.5,
              json.dumps({"verification": "unverified", "rel_src": "md1"},
                         ensure_ascii=False)),
             ("md:裸", "doc_markdown", "无分组文档", "open", 0.5,
              json.dumps({"verification": "unverified"}, ensure_ascii=False))])
        conn.commit()
        conn.close()
        r = self.client.get("/api/external-docs")
        self.assertEqual(r.status_code, 200)
        rows = {d["source_id"]: d for d in r.json()}
        self.assertEqual(rows["md:组内"]["rel_src"], "md1")
        self.assertEqual(rows["md:组内"]["rel_dir"], "故障")
        self.assertEqual(rows["md:回退"]["rel_src"], "md1")
        self.assertIsNone(rows["md:回退"]["rel_dir"])   # 资产回退件：无目录归属
        self.assertIsNone(rows["md:裸"]["rel_src"])     # 网络来源/旧数据：两者皆缺

    def test_tag_candidates_batch_adopt_and_ignore(self):
        """候选批量分拣：采纳=入词典+提及文档打标；忽略=改状态不改数据；非 pending 进 failed。"""
        from vllm_kb.config import AppConfig
        from vllm_kb.review import ReviewStore, default_review_path
        import sqlite3

        seed_kb_doc(self.cfg_path, "md:cand1")
        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        store = ReviewStore(default_review_path(cfg))
        store.add_item("tag_candidate", "tag:配置pip源",
                       {"candidate": "配置pip源", "suggested_tier": "purpose", "doc_count": 1,
                        "docs": [{"source_id": "md:cand1"}]})
        store.add_item("tag_candidate", "tag:RoCE组网",
                       {"candidate": "RoCE组网", "suggested_tier": "domain", "doc_count": 2})
        items = store.list_items(category="tag_candidate")
        store.close()
        ids = [i["id"] for i in items]
        # 忽略一条 + 采纳一条
        r = self.client.post("/api/tag-candidates/batch", json={
            "item_ids": ids, "action": "adopt", "reviewer": "王五"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["done"], 2)
        # 词典已入 + 文档已打标
        conn = sqlite3.connect(cfg.resolve(cfg.storage.sqlite_path))
        manual = dict(conn.execute("SELECT source_id, manual FROM doc_tags").fetchall())
        conn.close()
        self.assertIn("配置pip源", json.loads(manual.get("md:cand1", "[]")))
        cfg_data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        reg_names = [x["name"] if isinstance(x, dict) else x
                     for x in cfg_data["tags"]["registry"]]
        self.assertIn("配置pip源", reg_names)
        # 再分拣 → 全部非 pending → failed
        r2 = self.client.post("/api/tag-candidates/batch", json={
            "item_ids": ids, "action": "ignore", "reviewer": "王五"})
        self.assertEqual(r2.json()["done"], 0)
        self.assertEqual(len(r2.json()["failed"]), 2)

    def test_tag_dict_batch(self):
        """词典批量操作：批量移出（不存在进 failed）+ 批量改 tier（全局生效）。"""
        from vllm_kb.config import AppConfig
        from vllm_kb.tagging import TagRegistry

        cfg_data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        cfg_data.setdefault("tags", {})["registry"] = [
            {"name": "HCCL", "tier": "domain"}, {"name": "组网", "tier": "domain"},
            {"name": "超时排查", "tier": "purpose"}]
        self.cfg_path.write_text(json.dumps(cfg_data, ensure_ascii=False), encoding="utf-8")
        # 批量改 tier：domain → purpose
        r = self.client.post("/api/tag-dict/batch", json={
            "names": ["HCCL", "组网", "ghost"], "action": "tier", "tier": "purpose"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["done"], 2)
        self.assertEqual([f["name"] for f in r.json()["failed"]], ["ghost"])
        reg = {e.name: e.tier for e in TagRegistry.load_from_config_file(self.cfg_path).entries}
        self.assertEqual(reg["HCCL"], "purpose")
        self.assertEqual(reg["组网"], "purpose")
        # 批量移出
        r2 = self.client.post("/api/tag-dict/batch", json={
            "names": ["HCCL", "ghost"], "action": "delete"})
        self.assertEqual(r2.json()["done"], 1)
        self.assertEqual([f["name"] for f in r2.json()["failed"]], ["ghost"])
        reg2 = TagRegistry.load_from_config_file(self.cfg_path)
        self.assertFalse(reg2.contains("HCCL"))
        self.assertTrue(reg2.contains("组网"))
        # 非法 action
        self.assertEqual(self.client.post("/api/tag-dict/batch", json={
            "names": ["x"], "action": "bogus"}).status_code, 400)

    def test_delete_prunes_candidate_refs_and_auto_ignores_when_empty(self):
        """用户确认的缺口：删除文档后，未采纳候选的悬空引用要修剪；全部提及文档
        删光的候选自动忽略（此前永久滞留 pending，采纳会给不存在的 sid 写孤儿覆盖层）。"""
        from vllm_kb.config import AppConfig
        from vllm_kb.review import ReviewStore, default_review_path
        import sqlite3

        seed_kb_doc(self.cfg_path, "md:del1")
        seed_kb_doc(self.cfg_path, "md:keep")
        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        store = ReviewStore(default_review_path(cfg))
        store.add_item("tag_candidate", "tag:RoCE组网", {
            "candidate": "RoCE组网", "suggested_tier": "domain", "doc_count": 2,
            "docs": [{"source_id": "md:del1"}, {"source_id": "md:keep"}]})
        item = store.list_items(category="tag_candidate")[0]
        store.close()
        # 删其一 → 引用修剪、项仍 pending
        self.client.post("/api/external-docs/delete", json={"source_id": "md:del1"})
        store = ReviewStore(default_review_path(cfg))
        it = store.get_item(item["id"])
        store.close()
        self.assertEqual(it["status"], "pending")
        self.assertEqual([d["source_id"] for d in it["payload"]["docs"]], ["md:keep"])
        self.assertEqual(it["payload"]["doc_count"], 1)
        # 删最后一个 → 自动忽略（approved + 原因）
        self.client.post("/api/external-docs/delete", json={"source_id": "md:keep"})
        store = ReviewStore(default_review_path(cfg))
        it2 = store.get_item(item["id"])
        store.close()
        self.assertEqual(it2["status"], "approved")
        res = it2["result"]
        res = json.loads(res) if isinstance(res, str) else res
        self.assertIn("无剩余提及文档", res["reason"])

    def test_adopt_stale_candidate_no_orphan_overlay(self):
        """采纳 docs 全部悬空的候选：词典照入（词有价值），但不给不存在的 sid 写覆盖层。"""
        from vllm_kb.config import AppConfig
        from vllm_kb.review import ReviewStore, default_review_path
        import sqlite3

        seed_kb_doc(self.cfg_path, "md:other")  # 建 doc_tags 表 + 提供一个真实文档
        cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        store = ReviewStore(default_review_path(cfg))
        store.add_item("tag_candidate", "tag:悬空词", {
            "candidate": "悬空词", "suggested_tier": "domain", "doc_count": 1,
            "docs": [{"source_id": "md:ghost"}]})
        item = store.list_items(category="tag_candidate")[0]
        store.close()
        r = self.client.post("/api/tag-candidates/batch", json={
            "item_ids": [item["id"]], "action": "adopt", "reviewer": "赵六"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["done"], 1)
        conn = sqlite3.connect(cfg.resolve(cfg.storage.sqlite_path))
        rows = conn.execute("SELECT source_id, manual FROM doc_tags").fetchall()
        conn.close()
        self.assertEqual(rows, [], "悬空 sid 不得产生孤儿 doc_tags 覆盖层")
        cfg_data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        reg_names = [x["name"] if isinstance(x, dict) else x
                     for x in cfg_data["tags"]["registry"]]
        self.assertIn("悬空词", reg_names)

    def test_page_html_batch_tokens(self):
        """批量操作 UI 入口与分组结构必须在页面里。"""
        html = self.client.get("/").text
        for token in ("docBatchTagModal", "docBatchTagApply", "docBatchDelete",
                      "docGroupToggle", "queueBatchReview", "queueBatchCandidates",
                      "docbatch", "queuebatch", "queuecand",
                      "tdGroupToggle", "tagDictToggleTier", "tagDictBatchDelete",
                      "tagDictBatchTier", "tagdictbatch", "requireReviewer",
                      "docs/tags/batch", "external-docs/delete-batch", "items/review-batch",
                      "tag-candidates/batch", "tag-dict/batch"):
            self.assertIn(token, html, f"页面缺少 {token}")


if __name__ == "__main__":
    unittest.main()
