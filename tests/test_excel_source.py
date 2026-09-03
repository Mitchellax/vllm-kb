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

    def test_canonicalize_schema_free(self):
        """多 sheet/变列/空行：每行一条，body 按列序拼接（不依赖列名/sheet名/行号）。"""
        self.src.pull()
        docs = self.src.canonicalize()
        # 4 个有效行（Sheet1 3 行 + Sheet2 1 行；空行跳过）
        self.assertEqual(len(docs), 4)
        by_id = {d.source_id: d for d in docs}
        # source_id: excel:{stem}:{sheet序号}:{行号}
        self.assertTrue(any("问题定位记录:1:1" in k for k in by_id))
        self.assertTrue(any("问题定位记录:2:1" in k for k in by_id))
        # body 按列序拼接（自由文本）
        row1 = next(d for d in docs if ":1:1" in d.source_id)
        self.assertIn("HCCL 超时", row1.body)
        self.assertIn("节点", row1.body)
        # 变列（Sheet2 2 列）也解析
        row_sheet2 = next(d for d in docs if ":2:1" in d.source_id)
        self.assertIn("561000", row_sheet2.body)
        self.assertIn("aclnn", row_sheet2.body)
        # 验证状态/类型
        self.assertTrue(all(d.source_type == "doc_excel" for d in docs))
        self.assertTrue(all(d.extra["verification"] == "unverified" for d in docs))

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
        doc = next(d for d in docs if ":2:1" in d.source_id)
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
