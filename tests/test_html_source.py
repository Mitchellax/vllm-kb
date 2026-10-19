"""HtmlSource 测试：pull 幂等、canonicalize（标题/URL/表格收割落盘/图片占位与
data:URI 资产/远程占位）、缓存命中、缺依赖降级（lxml 跳过 / trafilatura 兜底）、
.mht 指向性提示、空正文跳过、路径不泄漏、x-* 审计元数据、page.html 统一文件名布局、
装饰图（1x1）过滤。"""
import base64
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from vllm_kb.config import SourceCfg
from vllm_kb.sources import HtmlSource, _stem_title


def _make_png(w: int, h: int) -> bytes:
    """最小合法 PNG（8-bit RGB 纯色），尺寸任意——装饰图过滤按最长边 ≤2px 判定。"""
    def chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x40\x80\xc0" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


_GIF_1X1 = (b"GIF89a\x01\x00\x01\x00\x00\xff\x00,\x00\x00\x00\x00"
            b"\x01\x00\x01\x00\x00\x02\x00;")
_PNG_B64 = base64.b64encode(_make_png(3, 3)).decode("ascii")  # 正常内容图（非装饰）


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


class TestAuditedMeta(HtmlSourceCase):
    def test_x_meta_into_extra_and_url_preferred(self):
        """x-* 审计元数据整体进 extra["meta"]；url 字段 x-source-url 优先于 saved-from。"""
        (self.imports / "page.html").write_text(page_html().replace(
            '<meta charset="utf-8">',
            '<meta charset="utf-8">\n'
            '<meta name="x-source-url" content="https://audit.example/articles/9">\n'
            '<meta name="x-created" content="2026-10-01">\n'
            '<meta name="x-updated" content="2026-10-20">\n'
            '<meta name="x-status" content="published">\n'
            '<meta name="x-author" content="张三">\n'
            '<meta name="x-categories" content="组网">\n'
            '<meta name="x-categories" content="升级">\n'
            '<meta name="x-expiry" content="2027-10-01">'), encoding="utf-8")
        img = self.imports / "img"
        img.mkdir()
        (img / "a.png").write_bytes(base64.b64decode(_PNG_B64))
        self.src.pull()
        d = self.src.canonicalize()[0]
        self.assertEqual(d.url, "https://audit.example/articles/9")
        m = d.extra["meta"]
        self.assertEqual(m["x-author"], "张三")
        self.assertEqual(m["x-categories"], "组网, 升级")   # 同名重复 → 逗号连接
        for k in ("x-created", "x-updated", "x-status", "x-expiry"):
            self.assertIn(k, m)


class TestPageHtmlLayout(HtmlSourceCase):
    def test_page_stem_title_and_identity(self):
        """审计布局：每篇文章都是 <目录>/page.html——同名靠路径指纹消歧，
        标题用 <title>（泛名 stem page 不顶替）。"""
        for name, title in (("a", "组网方案A"), ("b", "升级CANN指南")):
            d = self.imports / name
            d.mkdir()
            (d / "page.html").write_text(
                f'<html><head><title>{title}</title></head><body>'
                f'<p>{title} 的正文段落，内容足够长以通过正文抽取阈值。</p></body></html>',
                encoding="utf-8")
        self.src.pull()
        docs = self.src.canonicalize()
        self.assertEqual(len(docs), 2)
        for d in docs:
            self.assertTrue(d.source_id.startswith("html:page--"), d.source_id)
        self.assertEqual({d.title for d in docs}, {"组网方案A", "升级CANN指南"})


class TestDecorativeImages(HtmlSourceCase):
    def test_1x1_gif_dropped_by_content(self):
        """无属性的 1x1 gif（内容级检测）→ 不进正文/资产，evidence 记 decorative。"""
        (self.imports / "案例.html").write_text(
            page_html().replace("./img/a.png", "./img/logo.gif"), encoding="utf-8")
        img = self.imports / "img"
        img.mkdir()
        (img / "logo.gif").write_bytes(_GIF_1X1)
        (img / "a.png").write_bytes(base64.b64decode(_PNG_B64))
        self.src.pull()
        d = self.src.canonicalize()[0]
        self.assertNotIn("./img/logo.gif", d.body)
        kinds = {e["kind"] for e in d.extra["evidence"]}
        self.assertIn("decorative", kinds)
        self.assertNotIn("local", kinds)
        self.assertEqual(d.extra["quality"]["images_decorative_skipped"], 1)
        self.assertEqual(list((self.root / "data" / "assets" / "images").glob("*")), [])

    def test_attr_1x1_dropped_at_harvest(self):
        """显式 width/height=1 的装饰图在收割层丢弃（token 都不产生）；正常图照旧。"""
        (self.imports / "tiny.html").write_text(
            '<html><head><title>t</title></head><body><p>正文段落内容足够长。</p>'
            '<img src="./img/logo.gif" width="1" height="1">'
            '<p><img src="./img/big.png"></p></body></html>', encoding="utf-8")
        img = self.imports / "img"
        img.mkdir()
        (img / "logo.gif").write_bytes(_GIF_1X1)
        (img / "big.png").write_bytes(base64.b64decode(_PNG_B64))
        self.src.pull()
        d = self.src.canonicalize()[0]
        self.assertNotIn("logo.gif", d.body)
        self.assertEqual({e["kind"] for e in d.extra["evidence"]}, {"local"})


class TestStemTitle(unittest.TestCase):
    def test_generic_stems_defer_to_extracted(self):
        """泛化页名（page/index/…）不顶替解析标题；既有 junk 规则与正常文件名优先不变。"""
        self.assertEqual(_stem_title("page", "真标题"), "真标题")
        self.assertEqual(_stem_title("PAGE", "真标题"), "真标题")   # 大小写不敏感
        self.assertEqual(_stem_title("index", ""), "index")         # 提取为空回退 stem
        self.assertEqual(_stem_title("npu-smi 命令参考", "解析标题"), "npu-smi 命令参考")
        self.assertEqual(_stem_title("12345", "提取标题"), "提取标题")


if __name__ == "__main__":
    unittest.main()
