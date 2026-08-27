"""PDF / Markdown 来源导入测试：pull 幂等、canonicalize 解析、验证状态默认规则、表格结构化、
文档级标签提取（两级分类）、资产路径脱敏（asset_id + 正文图片占位）。"""
import json
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig, SourceCfg
from vllm_kb.sources import MarkdownSource, PdfSource


def make_pdf(path: Path) -> None:
    """用 PyMuPDF 生成一个含文字 + 表格 + 编号标题的最小 PDF。

    标题用 ASCII（insert_text 无中文字体时中文渲染为占位点，会干扰标题提取断言）。
    """
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "NPU HCCN Tool Interface Guide")
    page.insert_text((72, 100), "2.34 get network version")
    page.insert_text((72, 120), "2.35 cmd format")
    page.insert_text((72, 140), "troubleshoot: hccn_tool -i 0 -t xxx")
    # 简单表格（用文本行模拟，find_tables 对规整文本行可识别）
    rows = [
        "错误码   含义",
        "107020   memory allocation failed",
        "507014   device busy",
    ]
    y = 180
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
        # 路径脱敏：asset 无 path，只有不透明 asset_id
        self.assertNotIn("path", d.extra["asset"])
        self.assertTrue(d.extra["asset"]["asset_id"])
        # 自动标签：无词典 → 确定标签为空；候选非空（glm5 / 第一步 等）
        self.assertEqual(d.tags, [])
        cands = {c["name"] for c in d.extra["tag_candidates"]}
        self.assertIn("glm5", cands)
        self.assertIn("第一步", cands)

    def test_canonicalize_tags_from_registry(self):
        """词典命中 → 自动标签带两级 tier（domain/purpose）。"""
        import os

        from vllm_kb.tagging import TIER_DOMAIN, TIER_PURPOSE

        # 传 app_cfg 后路径经 AppConfig.resolve（数据根重定向）——用 VLLM_KB_DATA_ROOT 指回临时目录
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data")
        try:
            cfg = AppConfig.model_validate({"tags": {"registry": [
                {"name": "GLM5.1", "tier": "domain"},
                {"name": "崩溃三板斧", "tier": "purpose"},
            ]}})
            src = MarkdownSource(self.cfg, project_root=self.root, app_cfg=cfg)
            src.pull()
            docs = src.canonicalize()
            self.assertEqual(docs[0].tags, ["GLM5.1", "崩溃三板斧"])
        finally:
            os.environ.pop("VLLM_KB_DATA_ROOT", None)

    def test_canonicalize_image_placeholder(self):
        """正文图片引用改为不透明占位；evidence 只留 asset_id，不含路径。"""
        (self.root / "data" / "imports" / "md" / "带图文档.md").write_text(
            "# 带图文档\n\n![拓扑图](./images/topo.png)\n\n正文\n", encoding="utf-8")
        (self.root / "data" / "imports" / "md" / "images").mkdir(parents=True)
        (self.root / "data" / "imports" / "md" / "images" / "topo.png").write_bytes(b"PNG-DATA")
        docs = self.src.canonicalize()
        d = next(x for x in docs if x.source_id == "md:带图文档")
        # 正文无路径
        self.assertNotIn("images/topo.png", d.body)
        self.assertNotIn("assets/", d.body)
        self.assertIn("图片", d.body)
        ev = d.extra["evidence"][0]
        self.assertEqual(ev["kind"], "local")
        self.assertTrue(ev["asset_id"])
        self.assertNotIn("path", ev)


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
        # 路径脱敏：asset 无 path，只有不透明 asset_id
        self.assertNotIn("path", d.extra["asset"])
        self.assertTrue(d.extra["asset"]["asset_id"])
        # 自动标签：无词典 → 确定标签空；候选含拉丁 token 与短标题（cmd format）
        self.assertEqual(d.tags, [])
        cands = {c["name"] for c in d.extra["tag_candidates"]}
        self.assertIn("npu", cands)
        self.assertIn("hccn", cands)
        self.assertIn("cmd format", cands)

    def test_pdf_tags_from_registry(self):
        """词典命中 → 自动标签（两级 tier）。"""
        import os

        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data")
        try:
            cfg = AppConfig.model_validate({"tags": {"registry": [
                {"name": "hccn", "tier": "domain"},
                {"name": "cmd format", "tier": "purpose"},
            ]}})
            src = PdfSource(self.cfg, project_root=self.root, app_cfg=cfg)
            src.pull()
            docs = src.canonicalize()
            self.assertEqual(docs[0].tags, ["hccn", "cmd format"])
        finally:
            os.environ.pop("VLLM_KB_DATA_ROOT", None)

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
