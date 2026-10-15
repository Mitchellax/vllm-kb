"""HtmlSource 测试：pull 幂等、canonicalize（标题/URL/表格收割落盘/图片占位与
data:URI 资产/远程占位）、缓存命中、缺依赖降级（lxml 跳过 / trafilatura 兜底）、
.mht 指向性提示、空正文跳过、路径不泄漏。"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vllm_kb.config import SourceCfg
from vllm_kb.sources import HtmlSource

_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def page_html(img_dir: str = "./img") -> str:
    return f"""<!DOCTYPE html>
<!-- saved from url=(0025)https://wiki.example/case/1 -->
<html><head>
<meta charset="utf-8">
<title>GLM5.1 PD分离案例</title>
</head><body>
<nav>站点导航菜单页脚版权</nav>
<h1>GLM5.1 PD分离 P节点挂死</h1>
<p>现象：halMemCreate failed 报错后 P 节点挂死。</p>
<table>
  <caption>错误码表</caption>
  <tr><th>错误码</th><th>含义</th></tr>
  <tr><td>561001</td><td>dispatch 失败</td></tr>
</table>
<p><img src="{img_dir}/a.png"><img src="https://example.com/x.png"></p>
</body></html>
"""


class HtmlSourceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.imports = self.root / "data" / "imports" / "html"
        self.imports.mkdir(parents=True)
        self.cfg = SourceCfg(id="webpages", type="html", path="data/imports/html",
                             enabled=True)
        self.src = HtmlSource(self.cfg, project_root=self.root)

    def tearDown(self):
        self.tmp.cleanup()


class TestPull(HtmlSourceCase):
    def test_pull_copies_to_assets(self):
        (self.imports / "案例.html").write_text(page_html(), encoding="utf-8")
        self.assertEqual(self.src.pull(), 1)
        self.assertTrue((self.root / "data" / "assets" / "html" / "案例.html").exists())

    def test_pull_idempotent(self):
        (self.imports / "案例.html").write_text(page_html(), encoding="utf-8")
        self.src.pull()
        self.assertEqual(self.src.pull(), 0)  # 内容未变 → 0 新增

    def test_mht_gets_legacy_hint(self):
        (self.imports / "旧归档.mht").write_text("MHTML archive", encoding="utf-8")
        self.assertEqual(self.src.pull(), 0)
        self.assertIn("旧归档.mht", self.src._legacy_hint())
        self.assertIn("网页，仅 HTML", self.src._legacy_hint())


class TestCanonicalize(HtmlSourceCase):
    def setUp(self):
        super().setUp()
        (self.imports / "案例.html").write_text(page_html(), encoding="utf-8")
        img_dir = self.imports / "img"
        img_dir.mkdir()
        import base64

        (img_dir / "a.png").write_bytes(base64.b64decode(_PNG_B64))
        self.src.pull()
        self.docs = self.src.canonicalize()

    def test_basic_doc_fields(self):
        self.assertEqual(len(self.docs), 1)
        d = self.docs[0]
        self.assertEqual(d.source_type, "doc_html")
        self.assertTrue(d.source_id.startswith("html:"))
        self.assertEqual(d.title, "GLM5.1 PD分离案例")          # <title>
        self.assertEqual(d.url, "https://wiki.example/case/1")  # saved-from-url 注释
        self.assertEqual(d.extra["verification"], "unverified")
        self.assertEqual(d.extra["asset"]["format"], "html")
        self.assertTrue(d.extra["asset"]["asset_id"])
        self.assertNotIn("path", d.extra["asset"])              # 资产路径不进库
        self.assertTrue(d.extra["quality"]["parsed_with"].startswith("html_convert(trafilatura"))
        self.assertFalse(d.extra["quality"]["cached"])

    def test_body_and_table_rendered(self):
        d = self.docs[0]
        self.assertIn("halMemCreate failed", d.body)
        self.assertNotIn("站点导航菜单页脚版权", d.body)          # 骨架修剪
        self.assertIn("| 错误码 | 含义 |", d.body)               # pipe table 统一渲染补回
        self.assertIn("| 561001 | dispatch 失败 |", d.body)

    def test_tables_json_schema(self):
        d = self.docs[0]
        rels = d.extra["structure"]["tables"]
        self.assertEqual(rels, [f"parsed/html/{d.extra['asset']['asset_id']}.tables.json"])
        import json

        data = json.loads((self.root / "data" / rels[0]).read_text(encoding="utf-8"))
        self.assertEqual(data["source"], f"html:{d.extra['asset']['asset_id']}")
        self.assertEqual(data["tables"][0]["rows"],
                         [["错误码", "含义"], ["561001", "dispatch 失败"]])

    def test_local_image_resolved_to_placeholder(self):
        d = self.docs[0]
        self.assertIn("[图片]", d.body)                          # 不透明占位
        self.assertNotIn("./img", d.body)                        # 路径不进正文
        kinds = {e["kind"] for e in d.extra.get("evidence", [])}
        self.assertIn("local", kinds)
        # 资产真实落盘（assets/images 层；本地复制保留原文件名，base64 才用 img_<sha> 命名）
        assets = list((self.root / "data" / "assets" / "images").glob("*.png"))
        self.assertEqual(len(assets), 1)

    def test_remote_image_placeholder_only(self):
        d = self.docs[0]
        kinds = {e["kind"] for e in d.extra.get("evidence", [])}
        self.assertIn("remote", kinds)                           # V1 不下载，只留占位
        self.assertNotIn("https://example.com", d.body)          # 远程引用不进正文

    def test_cache_hit(self):
        docs2 = self.src.canonicalize()
        self.assertEqual(len(docs2), 1)
        self.assertTrue(docs2[0].extra["quality"]["cached"])     # 二次解析走缓存
        self.assertEqual(docs2[0].body, self.docs[0].body)
        cache = self.root / "data" / "parsed" / "html" / (
            f"{self.docs[0].extra['asset']['asset_id']}.extract.json")
        self.assertTrue(cache.exists())

    def test_recanonicalize_after_cache_deleted(self):
        """缓存损坏（删掉）→ 重解析结果一致（幂等再生）。"""
        cache = self.root / "data" / "parsed" / "html" / (
            f"{self.docs[0].extra['asset']['asset_id']}.extract.json")
        cache.unlink()
        docs2 = self.src.canonicalize()
        self.assertEqual(docs2[0].body, self.docs[0].body)
        self.assertFalse(docs2[0].extra["quality"]["cached"])


class TestDataUriImage(HtmlSourceCase):
    def test_base64_image_lands_as_asset(self):
        (self.imports / "内嵌图.html").write_text(
            page_html().replace('./img/a.png', f'data:image/png;base64,{_PNG_B64}'),
            encoding="utf-8")
        self.src.pull()
        docs = self.src.canonicalize()
        self.assertEqual(len(docs), 1)
        d = docs[0]
        self.assertIn("[图片]", d.body)
        kinds = {e["kind"] for e in d.extra.get("evidence", [])}
        self.assertIn("base64", kinds)
        assets = list((self.root / "data" / "assets" / "images").glob("img_*.png"))
        self.assertEqual(len(assets), 1)


class TestDegradePaths(HtmlSourceCase):
    def test_lxml_missing_skips_whole_source(self):
        """lxml 未安装 → 整来源跳过（与 pdf/word 缺依赖同一姿态），不产半残文档。"""
        (self.imports / "案例.html").write_text(page_html(), encoding="utf-8")
        with mock.patch.dict(sys.modules, {"lxml": None}):
            self.assertEqual(self.src.canonicalize(), [])

    def test_trafilatura_missing_lxml_fallback(self):
        """trafilatura 未安装 → lxml 全文兜底（表格收割不受影响），醒目告警进 quality。"""
        (self.imports / "案例.html").write_text(page_html(), encoding="utf-8")
        with mock.patch.dict(sys.modules, {"trafilatura": None}):
            docs = self.src.canonicalize()
        self.assertEqual(len(docs), 1)
        d = docs[0]
        self.assertEqual(d.extra["quality"]["parsed_with"], "html_convert(lxml_fallback)")
        self.assertIn("halMemCreate failed", d.body)             # 文字保住
        self.assertIn("| 错误码 | 含义 |", d.body)                # 表格收割不受影响

    def test_empty_body_skipped(self):
        (self.imports / "空页.html").write_text(
            "<html><head><title>空</title></head><body><script>var a=1;</script></body></html>",
            encoding="utf-8")
        self.src.pull()
        self.assertEqual(self.src.canonicalize(), [])


if __name__ == "__main__":
    unittest.main()
