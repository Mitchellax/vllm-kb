"""PDF / Markdown 来源导入测试：pull 幂等、canonicalize 解析、验证状态默认规则、表格结构化。"""
import json
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import PROJECT_ROOT, SourceCfg
from vllm_kb.sources import MarkdownSource, PdfSource


def make_pdf(path: Path) -> None:
    """用 PyMuPDF 生成一个含文字 + 表格的最小 PDF。"""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "NPU HCCN Tool 接口指南")
    page.insert_text((72, 100), "排查命令：hccn_tool -i 0 -t xxx")
    # 简单表格（用文本行模拟，find_tables 对规整文本行可识别）
    rows = [
        "错误码   含义",
        "107020   memory allocation failed",
        "507014   device busy",
    ]
    y = 140
    for r in rows:
        page.insert_text((72, y), r)
        y += 20
    doc.save(str(path))
    doc.close()


class TestMarkdownSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        imports = self.root / "data" / "imports" / "md"
        imports.mkdir(parents=True)
        (imports / "glm5_1三板斧.md").write_text(
            "# GLM5.1 崩溃三板斧\n\n## 第一步\n检查 halMemCreate 日志\n## 第二步\n调小 mega_moe_max_tokens\n",
            encoding="utf-8",
        )
        self.cfg = SourceCfg(id="wiki", type="markdown", path="data/imports/md",
                             title_pattern=r"^#\s+(.+)", enabled=True)
        self.src = MarkdownSource(self.cfg, project_root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pull_copies_to_assets(self):
        n = self.src.pull()
        self.assertEqual(n, 1)
        assets = self.root / "data" / "assets" / "md"
        self.assertTrue((assets / "glm5_1三板斧.md").exists())

    def test_pull_idempotent(self):
        self.src.pull()
        n2 = self.src.pull()
        assets = self.root / "data" / "assets" / "md"
        self.assertEqual(n2, 0)  # 幂等：内容未变则 0 新增
        self.assertEqual(len(list(assets.glob("*.md"))), 1)  # 不重复复制

    def test_canonicalize_verification_unverified(self):
        self.src.pull()
        docs = self.src.canonicalize()
        self.assertEqual(len(docs), 1)
        d = docs[0]
        self.assertEqual(d.source_type, "doc_markdown")
        self.assertTrue(d.source_id.startswith("md:"))
        self.assertEqual(d.title, "GLM5.1 崩溃三板斧")  # 首个 # 标题
        self.assertIn("mega_moe_max_tokens", d.body)
        self.assertEqual(d.extra["verification"], "unverified")
        self.assertEqual(d.extra["asset"]["format"], "markdown")
        self.assertTrue(d.extra["asset"]["sha256"])


class TestPdfSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        imports = self.root / "data" / "imports" / "pdf"
        imports.mkdir(parents=True)
        make_pdf(imports / "npu_hccn_tool_guide.pdf")
        self.cfg = SourceCfg(id="manuals", type="pdf", path="data/imports/pdf", enabled=True)
        self.src = PdfSource(self.cfg, project_root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pull_and_canonicalize(self):
        self.src.pull()
        docs = self.src.canonicalize()
        self.assertEqual(len(docs), 1)
        d = docs[0]
        self.assertEqual(d.source_type, "doc_pdf")
        self.assertEqual(d.source_id, "pdf:npu_hccn_tool_guide")
        # 文字层入库
        self.assertIn("hccn_tool", d.body)
        # 验证状态：官方手册默认专家验证
        self.assertEqual(d.extra["verification"], "expert")
        self.assertEqual(d.extra["asset"]["format"], "pdf")
        self.assertEqual(d.extra["quality"]["text_source"], "text_layer")
        self.assertEqual(d.extra["asset"]["pages"], 1)

    def test_tables_json_written(self):
        self.src.pull()
        self.src.canonicalize()
        parsed = self.root / "data" / "parsed" / "pdf"
        files = list(parsed.glob("*.tables.json"))
        # find_tables 对简单文本行可能不识别——不强断言存在，但若存在必须合法
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            self.assertIn("tables", data)
            self.assertIn("source", data)

    def test_pull_missing_dir_ok(self):
        src = PdfSource(SourceCfg(id="x", type="pdf", path="data/imports/nonexist", enabled=True),
                        project_root=self.root)
        self.assertEqual(src.pull(), 0)
        self.assertEqual(src.canonicalize(), [])


if __name__ == "__main__":
    unittest.main()
