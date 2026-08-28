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

    def test_sanitize_ip_and_path(self):
        """内部 IP/路径脱敏；默认路径与回环 IP 保留。"""
        self.src.pull()
        docs = self.src.canonicalize()
        body = "\n".join(d.body for d in docs)
        # 内部 IP 脱敏
        self.assertNotIn("10.0.0.5", body)
        self.assertIn("<IP>", body)
        # 内部路径脱敏
        self.assertNotIn("/home/user/logs/oom.log", body)
        self.assertIn("<PATH>", body)
        # 默认路径（/var/log/ 白名单）保留（诊断价值）
        self.assertIn("/var/log/npu/", body)
        # 回环 IP 保留
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


if __name__ == "__main__":
    unittest.main()
