"""Excel 来源（schema-free）测试：多 sheet/变列解析、脱敏、实体复用入图。"""
import json
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig, SourceCfg
from vllm_kb.sources import ExcelSource


def make_xlsx(path: Path) -> None:
    """生成含 2 个 sheet、列数不同、含空行的测试 xlsx。

    - Sheet1：3 列（问题/现象/结论），含 IP 与内部路径 cell；
    - Sheet2：2 列（错误码/处理），验证"不知道格式"也能解析。
    """
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["HCCL 超时", "节点 10.0.0.5 通信超时", "检查 /var/log/npu/ 日志"])
    ws.append(["内存不足", "OOM", "路径 /home/user/logs/oom.log 有记录"])
    ws.append([None, None, None])  # 空行应跳过
    ws.append(["回环测试", "127.0.0.1 正常", ""])
    ws2 = wb.create_sheet("任意名")
    ws2.append(["错误码 561000", "aclnn 失败"])  # 2 列——列数不同也应解析
    wb.save(str(path))


class TestExcelSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        imports = self.root / "data" / "imports" / "engineer"
        imports.mkdir(parents=True)
        make_xlsx(imports / "问题定位记录.xlsx")
        self.cfg = SourceCfg(id="engineer", type="excel",
                             path="data/imports/engineer/问题定位记录.xlsx", enabled=True)
        self.src = ExcelSource(self.cfg, project_root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pull_copies_to_assets(self):
        n = self.src.pull()
        self.assertEqual(n, 1)
        self.assertTrue((self.root / "data" / "assets" / "excel" / "问题定位记录.xlsx").exists())
        # 幂等
        self.assertEqual(self.src.pull(), 0)

    def test_canonicalize_one_doc_per_file(self):
        """**一个文件一篇**：多 sheet/变列/空行都进同一篇 body，sheet 名作 Markdown 标题。

        身份 = `excel:<文件名>`（不再是位置性的 `文件:sheet:行`）——插行/改行不会让身份平移。
        """
        self.src.pull()
        docs = self.src.canonicalize()
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.source_id, "excel:问题定位记录")
        self.assertEqual(doc.title, "问题定位记录")
        self.assertEqual(doc.source_type, "doc_excel")
        self.assertEqual(doc.extra["verification"], "unverified")
        # 每个 sheet 一个小节标题 + 行内容按列序拼接（自由文本）
        self.assertIn("## Sheet1", doc.body)
        self.assertIn("## 任意名", doc.body)
        self.assertIn("HCCL 超时", doc.body)
        self.assertIn("节点 10.0.0.5 通信超时", doc.body)
        self.assertIn("561000", doc.body)      # 变列（Sheet2 2 列）也解析
        self.assertIn("aclnn", doc.body)
        # 空行不进正文（Sheet1 第 3 行）
        self.assertEqual(doc.body.count("\n\n## "), 1)   # 两个 sheet 之间只有一个分隔
        # 计数元数据（整体评估用；quality 出口整包透传）
        self.assertEqual(doc.extra["quality"]["sheets"], 2)
        self.assertEqual(doc.extra["quality"]["rows"], 4)   # 3 + 1 个有效行
        self.assertEqual(doc.extra["quality"]["source_mode"], "imports")

    def test_chunks_are_still_per_row_with_sheet_section(self):
        """按文件聚合**不损失检索粒度**：每个非空行仍是独立 chunk，且带 sheet 作 section。"""
        from vllm_kb.chunking import chunk_doc

        self.src.pull()
        doc = self.src.canonicalize()[0]
        chunks = chunk_doc(doc)
        texts = [c.text for c in chunks]
        self.assertEqual(len(chunks), 4, texts)
        self.assertEqual({c.section for c in chunks}, {"Sheet1", "任意名"})
        hit = next(c for c in chunks if "HCCL 超时" in c.text)
        self.assertEqual(hit.section, "Sheet1")
        self.assertTrue(hit.text.startswith("【Sheet1】"), hit.text)   # 标题注入正文
        # 每行一个 chunk（不跨行合并成一大块）
        self.assertTrue(all("561000" not in c.text or c.section == "任意名" for c in chunks))

    def test_file_identity_survives_row_insert(self):
        """**核心动机**：中间插一行不会改变文档身份（逐行方案下会整体平移）。"""
        import openpyxl

        self.src.pull()
        before = self.src.canonicalize()[0].source_id
        p = self.root / "data" / "imports" / "engineer" / "问题定位记录.xlsx"
        wb = openpyxl.load_workbook(str(p))
        wb["Sheet1"].insert_rows(2)          # 在表头与第二行之间插一行
        wb["Sheet1"].cell(row=2, column=1, value="新增问题")
        wb.save(str(p))
        after = self.src.canonicalize()[0].source_id
        self.assertEqual(before, after)
        self.assertEqual(after, "excel:问题定位记录")

    def test_tags_from_filename_and_sheets(self):
        """标签是**整体评估**：文件名 + sheet 名走与 md/word 同一套 extract_tags。"""
        from vllm_kb.tagging import TagRegistry

        self.src.pull()
        doc = self.src.canonicalize()[0]
        self.assertIsInstance(doc.tags, list)          # 无词典命中时为空列表
        self.assertIn("tag_candidates", doc.extra)     # 候选走审核队列
        self.assertEqual(TagRegistry().entries, [])

    def test_doc_asset_registered(self):
        """§1.6：canonicalize 也把 excel 原件落资产层并注册（--skip-pull 也要有预览链）。"""
        import os

        from vllm_kb.config import AppConfig
        from vllm_kb.review import list_assets

        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data")
        try:
            cfg = AppConfig.model_validate({})
            src = ExcelSource(self.cfg, project_root=self.root, app_cfg=cfg)
            docs = src.canonicalize()          # 注意：**不调 pull()**
            self.assertEqual(len(docs), 1)
            assets = list_assets(self.root / "data" / "review.sqlite3")
            reg = assets.get(docs[0].extra["asset"]["asset_id"])
            self.assertIsNotNone(reg, assets)
            self.assertEqual(reg["source_type"], "doc_excel")
            self.assertEqual(reg["rel_path"], "assets/excel/问题定位记录.xlsx")
            self.assertTrue((self.root / "data" / reg["rel_path"]).is_file())
        finally:
            os.environ.pop("VLLM_KB_DATA_ROOT", None)

    def test_empty_file_skipped(self):
        """所有 sheet 都为空 → 不产文档（但原件照常落资产层）。"""
        import openpyxl

        p = self.root / "data" / "imports" / "engineer" / "空表.xlsx"
        wb = openpyxl.Workbook()
        wb.save(str(p))
        cfg = SourceCfg(id="e2", type="excel", path=str(p), enabled=True)
        self.assertEqual(ExcelSource(cfg, project_root=self.root).canonicalize(), [])

    def test_fallback_to_assets_when_imports_missing(self):
        """imports 缺失 → 回退资产层副本，身份取族名，quality 标记来源模式。"""
        self.src.pull()
        (self.root / "data" / "imports" / "engineer" / "问题定位记录.xlsx").unlink()
        docs = self.src.canonicalize()
        self.assertEqual([d.source_id for d in docs], ["excel:问题定位记录"])
        self.assertEqual(docs[0].extra["quality"]["source_mode"], "assets_fallback")

    def test_raw_text_stored_sanitize_log_collected(self):
        """后置脱敏：body 原文入库（原文检索）；会被脱敏的 IP/路径落盘维护日志。"""
        self.src.pull()
        docs = self.src.canonicalize()
        body = "\n".join(d.body for d in docs)
        # 原文入库（不脱敏——出口脱敏由 serve_api 返回时做）
        self.assertIn("10.0.0.5", body)
        self.assertIn("/home/user/logs/oom.log", body)
        self.assertIn("/var/log/npu/", body)
        self.assertIn("127.0.0.1", body)

    def test_entities_reuse_existing_pipeline(self):
        """scheme-free：body 实体（错误码）复用现有提取——canonical 后入图走 MENTIONS。"""
        from vllm_kb.graph_rels import extract_doc_relations

        self.src.pull()
        docs = self.src.canonicalize()
        doc = docs[0]
        ex = extract_doc_relations(doc.source_id, "", 0, doc.source_type, doc.body)
        self.assertIn("561000", ex.mentions.get("error_code", set()))
        # 有 body 的文档都能被 canonical/ingest 接受
        self.assertTrue(doc.body)

    def test_sanitize_log_written(self):
        """被脱敏命中的 IP/路径落盘 data/sanitize_log.json（维护用，含被脱敏值）。"""
        import json
        import os

        from vllm_kb.config import AppConfig

        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data")
        try:
            cfg = AppConfig.model_validate({})
            src = ExcelSource(self.cfg, project_root=self.root, app_cfg=cfg)
            src.pull()
            src.canonicalize()
            log = self.root / "data" / "sanitize_log.json"
            self.assertTrue(log.exists())
            data = json.loads(log.read_text(encoding="utf-8"))
            self.assertIn("10.0.0.5", data["ips"])          # 被脱敏的 IP
            self.assertIn("/home/user/logs/oom.log", data["paths"])  # 被脱敏的路径
            # 保留项不进入日志
            self.assertNotIn("/var/log/npu/", data["paths"])
        finally:
            os.environ.pop("VLLM_KB_DATA_ROOT", None)

    def test_sanitize_config_from_app_cfg(self):
        """config.sanitize 控制**收集范围**（后置脱敏：body 始终原文入库）。"""
        import json
        import os

        from vllm_kb.config import AppConfig

        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data")
        try:
            cfg = AppConfig.model_validate({"sanitize": {
                "keep_paths": ["/home/user/logs"],  # 覆盖默认：业务日志目录保留（不收集）
                "keep_ips": ["10.0.0.5"],           # 覆盖默认：该私有 IP 保留（不收集）
            }})
            src = ExcelSource(self.cfg, project_root=self.root, app_cfg=cfg)
            src.pull()
            docs = src.canonicalize()
            body = "\n".join(d.body for d in docs)
            # 原文始终入库（出口脱敏由 serve_api 做）
            self.assertIn("10.0.0.5", body)
            self.assertIn("/home/user/logs/oom.log", body)
            # 维护日志按配置收集：keep 的不收集；覆盖后默认路径不再保留 → 被收集
            log = self.root / "data" / "sanitize_log.json"
            data = json.loads(log.read_text(encoding="utf-8"))
            self.assertNotIn("10.0.0.5", data["ips"])
            self.assertNotIn("/home/user/logs/oom.log", data["paths"])
            self.assertIn("/var/log/npu/", data["paths"])
        finally:
            os.environ.pop("VLLM_KB_DATA_ROOT", None)


if __name__ == "__main__":
    unittest.main()
