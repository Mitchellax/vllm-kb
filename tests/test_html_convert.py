"""html_convert 单测：编码链（BOM/meta/GB 兜底）、表格收割（内容/嵌套/布局/展开）、
正文抽取引擎（trafilatura / baseline 兜底 / 降级链）、页面骨架修剪、
链接安全化（图片不动）、saved-from-url 提取。"""
import codecs
import unittest
from unittest import mock

from vllm_kb.html_convert import (
    _sanitize_links,
    _strip_html_fallback,
    decode_html,
    html_to_markdown,
    table_to_markdown,
)


def _page(charset: str = "utf-8") -> bytes:
    """最小案例页：saved-from-url 注释 + 标题 + 正文 + 错误码表 + 嵌套/布局表 + 三种图片。"""
    doc = f"""<!DOCTYPE html>
<!-- saved from url=(0025)https://wiki.example/case/1 -->
<html><head>
<meta charset="{charset}">
<title>GLM5.1 PD分离案例</title>
</head><body>
<nav>站点导航菜单页脚版权</nav>
<h1>GLM5.1 PD分离 P节点挂死</h1>
<p>现象：halMemCreate failed 报错后 P 节点挂死，重跑不复现。</p>
<table>
  <caption>错误码表</caption>
  <tr><th>错误码</th><th>含义</th></tr>
  <tr><td>561001</td><td>dispatch 失败</td></tr>
</table>
<table><tr><td><table><tr><td>inner</td></tr></table></td></tr></table>
<table><tr><td>layout-only-cell</td></tr></table>
<p><img src="./img/a.png"><img src="data:image/png;base64,iVBORw0KGgo="><img src="https://example.com/x.png"></p>
</body></html>
"""
    return doc.encode("gb18030" if charset.startswith("gb") else "utf-8")


class TestDecodeHtml(unittest.TestCase):
    def test_meta_gb2312(self):
        text, charset = decode_html(
            '<html><head><meta charset="gb2312"></head><body><p>华为故障案例</p></body></html>'.encode("gb18030"))
        self.assertIn("华为故障案例", text)
        self.assertEqual(charset, "gb2312")

    def test_no_meta_falls_back_to_gb18030(self):
        text, charset = decode_html("<html><body><p>故障排查记录</p></body></html>".encode("gb18030"))
        self.assertIn("故障排查记录", text)
        self.assertEqual(charset, "gb18030")

    def test_utf8_preferred_over_gb18030(self):
        """utf-8 必须排在 gb18030 前面——否则中文 utf-8 字节被 gb18030 硬解成乱码。"""
        text, charset = decode_html("<html><body><p>调度超时排查</p></body></html>".encode("utf-8"))
        self.assertIn("调度超时排查", text)
        self.assertEqual(charset, "utf-8")

    def test_bom_utf8_sig(self):
        text, charset = decode_html(codecs.BOM_UTF8 + "<p>x</p>".encode("utf-8"))
        self.assertEqual(charset, "utf-8-sig")

    def test_http_equiv_charset(self):
        raw = ('<html><head><meta http-equiv="Content-Type" '
               'content="text/html; charset=gbk"></head><body><p>端口绑定失败</p></body></html>').encode("gb18030")
        text, charset = decode_html(raw)
        self.assertIn("端口绑定失败", text)
        self.assertEqual(charset, "gbk")


class TestHarvestTables(unittest.TestCase):
    def test_content_harvested_nested_and_layout_dropped(self):
        res = html_to_markdown(_page("utf-8"))
        # 收割结果：错误码表 1 张；嵌套表外壳（1×1 无 th/caption）与 layout-only 表丢弃
        self.assertEqual(len(res.tables), 1)
        self.assertEqual(res.tables[0]["caption"], "错误码表")
        self.assertEqual(res.tables[0]["rows"], [["错误码", "含义"], ["561001", "dispatch 失败"]])
        self.assertEqual(res.layout_tables_dropped, 2)
        self.assertEqual(res.tables_unwrapped, 0)

    def test_long_single_cell_unwrapped(self):
        """1×1 但文字长 → 套壳正文，展开成 <p> 保留（常见单格代码框）。"""
        long_text = ("hccn_tool -i 0 -t link status 现场命令输出足够长，必须明显超过展开阈值："
                     "eighty characters definitely yes plus more text to be safe and long")
        doc = f"<html><body><table><tr><td>{long_text}</td></tr></table></body></html>"
        res = html_to_markdown(doc.encode("utf-8"))
        self.assertEqual(res.tables_unwrapped, 1)
        self.assertEqual(len(res.tables), 0)
        self.assertIn("hccn_tool", res.body)
        self.assertEqual(res.layout_tables_dropped, 0)

    def test_content_table_rendered_back_into_body(self):
        res = html_to_markdown(_page("utf-8"))
        # 表格抽取前移除 → 正文里的表格块是统一渲染的 pipe table
        self.assertIn("| 错误码 | 含义 |", res.body)
        self.assertIn("| 561001 | dispatch 失败 |", res.body)
        self.assertIn("**错误码表**", res.body)


class TestTableToMarkdown(unittest.TestCase):
    def test_basic_and_escape(self):
        md = table_to_markdown([["a|b", "c"], ["d", "e\nf"]])
        self.assertEqual(md, "| a\\|b | c |\n|---|---|\n| d | e f |")


class TestSanitizeLinks(unittest.TestCase):
    def test_keep_http_drop_relative(self):
        md = "见 [详情](./next.html) 与 [官网](https://a.com/x) 与 [联系](mailto:a@b.c)。"
        out = _sanitize_links(md)
        self.assertIn("[官网](https://a.com/x)", out)
        self.assertIn("[联系](mailto:a@b.c)", out)
        self.assertNotIn("(./next.html)", out)
        self.assertIn("详情", out)

    def test_file_dest_stripped(self):
        out = _sanitize_links("打开 [本机手册](file:///C:/manuals/a.pdf) 查看安装介质")
        self.assertNotIn("file://", out)
        self.assertIn("本机手册", out)

    def test_images_untouched(self):
        md = "![本地](file://C:/tmp/a.png) ![内嵌](data:image/png;base64,iVBORw0KGgo=)"
        self.assertEqual(_sanitize_links(md), md)


class TestHtmlToMarkdown(unittest.TestCase):
    def test_full_page_trafilatura(self):
        res = html_to_markdown(_page("utf-8"))
        self.assertEqual(res.extractor, "trafilatura")
        self.assertEqual(res.title, "GLM5.1 PD分离案例")
        self.assertEqual(res.source_url, "https://wiki.example/case/1")
        self.assertEqual(res.charset, "utf-8")
        self.assertIn("halMemCreate failed", res.body)   # 正文在
        self.assertNotIn("站点导航菜单页脚版权", res.body)  # 骨架修剪（先于引擎）
        self.assertNotIn("saved from url", res.body)      # 注释不进正文
        self.assertNotIn("layout-only-cell", res.body)    # 布局表碎片不进正文
        self.assertNotIn("inner", res.body.split("错误码表")[0])  # 嵌套表碎片不进正文
        # 图片收割补回（trafilatura 会丢纯图段落）：src 原样保留交 _resolve_images
        self.assertIn("![](./img/a.png)", res.body)
        self.assertIn("![](https://example.com/x.png)", res.body)

    def test_extract_none_falls_back_to_baseline(self):
        with mock.patch("trafilatura.extract", return_value=None), \
             mock.patch("trafilatura.baseline",
                        return_value=(None, "baseline 全文兜底正文内容 " * 10, 250)):
            res = html_to_markdown(b"<html><body><p>x</p></body></html>")
        self.assertEqual(res.extractor, "trafilatura_baseline")
        self.assertIn("baseline 全文兜底正文内容", res.body)
        self.assertTrue(any("baseline" in w for w in res.warnings))

    def test_link_and_image_passthrough(self):
        canned = ("段落一\n\n![本地](file://C:/tmp/a.png)\n\n![内嵌](data:image/png;base64,iVBORw0KGgo=)"
                  "\n\n[外链](https://a.com/x) [内链](./next.html)")
        with mock.patch("trafilatura.extract", return_value=canned):
            res = html_to_markdown(b"<html><body><p>x</p></body></html>")
        # 图片 dest 原样保留（资产化/占位由 _resolve_images 统一做）
        self.assertIn("![本地](file://C:/tmp/a.png)", res.body)
        self.assertIn("![内嵌](data:image/png;base64,iVBORw0KGgo=)", res.body)
        self.assertIn("[外链](https://a.com/x)", res.body)
        self.assertNotIn("(./next.html)", res.body)      # 相对链接剥掉只留文字
        self.assertIn("内链", res.body)

    def test_empty_bytes_degrades_gracefully(self):
        res = html_to_markdown(b"")
        self.assertEqual(res.body, "")
        self.assertTrue(res.warnings or res.extractor == "stdlib_fallback")


class TestStripFallback(unittest.TestCase):
    def test_tags_stripped_script_removed(self):
        html = ("<div><script>evil()</script><p>第一段</p>"
                "<style>.x{}</style><h2>第二节</h2><p>第二段</p><!-- 注释 --></div>")
        out = _strip_html_fallback(html)
        self.assertIn("第一段", out)
        self.assertIn("第二节", out)
        self.assertNotIn("evil", out)
        self.assertNotIn("<", out)


class TestMetaHarvest(unittest.TestCase):
    def test_x_meta_collected_and_url_preferred(self):
        doc = ("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>案例A</title>
<meta name="x-source-url" content="https://audit.example/articles/9">
<meta name="x-created" content="2026-10-01">
<meta name="x-updated" content="2026-10-20">
<meta name="x-status" content="published">
<meta name="x-author" content="张三">
<meta name="x-categories" content="组网">
<meta name="x-categories" content="升级">
<meta name="x-expiry" content="2027-10-01">
</head><body><p>正文内容足够长——PD 分离场景下 P 节点挂死排查记录，含报错与结论。</p></body></html>""")
        res = html_to_markdown(doc.encode("utf-8"))
        self.assertEqual(res.meta["x-source-url"], "https://audit.example/articles/9")
        self.assertEqual(res.meta["x-author"], "张三")
        self.assertEqual(res.meta["x-categories"], "组网, 升级")   # 同名重复连接
        self.assertIn("x-expiry", res.meta)
        self.assertEqual(res.source_url, "https://audit.example/articles/9")  # 优先于 saved-from

    def test_no_x_meta_keeps_saved_from_url(self):
        res = html_to_markdown(_page("utf-8"))
        self.assertEqual(res.meta, {})
        self.assertEqual(res.source_url, "https://wiki.example/case/1")


class TestInlinePosition(unittest.TestCase):
    def test_images_and_tables_inline_in_document_order(self):
        """图文/表文不再脱节：收割占位在抽取后**原位**还原（开头→图→段→表→结尾）。"""
        doc = """<html><head><title>t</title></head><body>
<p>开头段落讲现象。</p>
<p><img src="./img/mid.png"></p>
<p>中间段落给分析。</p>
<table><tr><th>k</th><th>v</th></tr><tr><td>a</td><td>b</td></tr></table>
<p>结尾段落给结论。</p>
</body></html>"""
        res = html_to_markdown(doc.encode("utf-8"))
        b = res.body
        self.assertIn("![](./img/mid.png)", b)
        self.assertIn("| k | v |", b)
        self.assertLess(b.index("开头段落"), b.index("![](./img/mid.png)"))
        self.assertLess(b.index("![](./img/mid.png)"), b.index("中间段落"))
        self.assertLess(b.index("中间段落"), b.index("| k | v |"))
        self.assertLess(b.index("| k | v |"), b.index("结尾段落"))

    def test_image_inside_table_cell_inlined(self):
        """表格单元格里的截图：占位随行还原为 markdown 图片（交 _resolve_images 资产化）。"""
        doc = ('<html><body><p>说明段落。</p>'
               '<table><tr><th>图</th><th>说明</th></tr>'
               '<tr><td><img src="./cell.png"></td><td>现场截图</td></tr></table>'
               '<p>收尾段落。</p></body></html>').encode("utf-8")
        res = html_to_markdown(doc)
        self.assertIn("![](./cell.png)", res.body)
        self.assertIn("| ![](./cell.png) | 现场截图 |", res.body)

    def test_attr_decorative_image_skipped_at_harvest(self):
        doc = ('<html><body><p>正文段落内容。</p>'
               '<img src="./t.gif" width="1" height="1">'
               '<p><img src="./real.png" width="640" height="480"></p></body></html>')
        res = html_to_markdown(doc.encode("utf-8"))
        self.assertNotIn("t.gif", res.body)
        self.assertIn("./real.png", res.body)


if __name__ == "__main__":
    unittest.main()
