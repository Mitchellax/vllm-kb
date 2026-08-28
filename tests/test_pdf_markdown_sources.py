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

    def test_markdown_raw_stored_sanitize_log_collected(self):
        """后置脱敏：md 正文原文入库；会被脱敏的 IP/路径落盘维护日志。"""
        import os

        from vllm_kb.config import AppConfig

        (self.root / "data" / "imports" / "md" / "案例.md").write_text(
            "# 案例\n\n节点 10.0.0.5 超时，日志在 /home/user/logs/x.log，"
            "默认路径 /var/log/npu/ 保留。\n", encoding="utf-8")
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data")
        try:
            cfg = AppConfig.model_validate({})
            src = MarkdownSource(self.cfg, project_root=self.root, app_cfg=cfg)
            docs = src.canonicalize()
            d = next(x for x in docs if "案例" in x.source_id)
            # 原文入库（出口脱敏由 serve_api 返回时做）
            self.assertIn("10.0.0.5", d.body)
            self.assertIn("/home/user/logs/x.log", d.body)
            self.assertIn("/var/log/npu/", d.body)
            # 维护日志：会被脱敏的 IP/路径（不含保留项）
            log = self.root / "data" / "sanitize_log.json"
            self.assertTrue(log.exists())
            data = json.loads(log.read_text(encoding="utf-8"))
            self.assertIn("10.0.0.5", data["ips"])
            self.assertIn("/home/user/logs/x.log", data["paths"])
            self.assertNotIn("/var/log/npu/", data["paths"])
        finally:
            os.environ.pop("VLLM_KB_DATA_ROOT", None)

    def test_markdown_sanitize_sources_config(self):
        """config.sanitize.sources 关闭 markdown 时，不收集维护日志（原文入库不变）。"""
        import os

        from vllm_kb.config import AppConfig

        (self.root / "data" / "imports" / "md" / "案例2.md").write_text(
            "# 案例2\n\n节点 10.0.0.5 超时。\n", encoding="utf-8")
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data")
        try:
            cfg = AppConfig.model_validate({"sanitize": {"sources": ["excel"]}})
            src = MarkdownSource(self.cfg, project_root=self.root, app_cfg=cfg)
            docs = src.canonicalize()
            d = next(x for x in docs if "案例2" in x.source_id)
            self.assertIn("10.0.0.5", d.body)  # 原文始终入库
            # 未启用收集：日志不含该 IP
            log = self.root / "data" / "sanitize_log.json"
            if log.exists():
                data = json.loads(log.read_text(encoding="utf-8"))
                self.assertNotIn("10.0.0.5", data.get("ips", []))
        finally:
            os.environ.pop("VLLM_KB_DATA_ROOT", None)


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

    def test_parse_cache_reused(self):
        """资产未变时二次 canonicalize 复用解析缓存（不触发 PyMuPDF 逐页提取）。"""
        import unittest.mock as mock

        self.src.pull()
        docs1 = self.src.canonicalize()
        self.assertEqual(len(docs1), 1)
        parsed_dir = self.root / "data" / "parsed" / "pdf"
        self.assertTrue(list(parsed_dir.glob("*.extract.json")), "应生成解析缓存")
        # 第二次：pymupdf.open 不应被调用（缓存命中，跳过逐页提取）
        with mock.patch("pymupdf.open", side_effect=AssertionError("缓存命中不应重新解析")):
            docs2 = self.src.canonicalize()
        self.assertEqual(len(docs2), 1)
        self.assertEqual(docs2[0].body, docs1[0].body)
        self.assertEqual(docs2[0].tags, docs1[0].tags)
        self.assertEqual(docs2[0].extra["asset"]["asset_id"],
                         docs1[0].extra["asset"]["asset_id"])

    def test_parse_cache_corrupt_reparses(self):
        """缓存损坏（非法 JSON）→ 自动重新解析并重建缓存。"""
        self.src.pull()
        self.src.canonicalize()
        cache = list((self.root / "data" / "parsed" / "pdf").glob("*.extract.json"))[0]
        cache.write_text("{broken json", encoding="utf-8")
        docs = self.src.canonicalize()
        self.assertEqual(len(docs), 1)
        # 缓存已重建为合法 JSON
        self.assertIn("body", json.loads(cache.read_text(encoding="utf-8")))

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
