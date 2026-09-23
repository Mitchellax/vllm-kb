"""Markdown 图片引用扫描测试：形态支持、代码感知、路径不泄漏硬约束。

对应 `vllm_kb/md_images.py` 与 `MarkdownSource._resolve_images`。

硬约束（回归防线）：**任意输入下，正文/canonical 都不能留下图片引用里的原始目标**——
否则服务器目录结构会随正文进检索库。`TestNoPathLeak` 就是这条约束的护栏。
"""
import base64
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import SourceCfg
from vllm_kb.md_images import code_spans, find_image_refs, rewrite_images
from vllm_kb.sources import MarkdownSource, _path_tag


def png(path: Path, w=8, h=8, color="white") -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color).save(path)


def png_b64(color="red") -> str:
    buf = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestFindImageRefs(unittest.TestCase):
    """形态矩阵：dest 是否被正确取到。"""

    def _one(self, text):
        refs = find_image_refs(text)
        self.assertEqual(len(refs), 1, f"{text!r} → {refs}")
        return refs[0]

    def test_standard_forms(self):
        for text, dest, form in [
            ("![a](shot.png)", "shot.png", "inline"),
            ("![a](imgs/sub.png)", "imgs/sub.png", "inline"),
            ("![a](../up/x.png)", "../up/x.png", "inline"),
            ("![a](/abs/x.png)", "/abs/x.png", "inline"),
            ("![a](D:/x/shot.png)", "D:/x/shot.png", "inline"),
            ("![a](file:///D:/x/shot.png)", "file:///D:/x/shot.png", "inline"),
            ('![a](shot.png "标题")', "shot.png", "inline"),
            ("![a](shot.png '标题')", "shot.png", "inline"),
            ("![a](shot.png (标题))", "shot.png", "inline"),
            ("![](shot.png)", "shot.png", "inline"),
        ]:
            with self.subTest(text=text):
                r = self._one(text)
                self.assertEqual(r.dest, dest)
                self.assertEqual(r.form, form)

    def test_lenient_forms(self):
        """非标准但常见的写法也要认出来（否则路径会留在正文）。"""
        for text, dest in [
            ("![a](my file.png)", "my file.png"),           # 裸空格
            ("![a](<my file.png>)", "my file.png"),          # 尖括号
            ("![a](img(1).png)", "img(1).png"),              # 目标含括号
            ("![a](my file (1).png)", "my file (1).png"),    # 空格 + 括号
            ('![a](my file.png "标题")', "my file.png"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(self._one(text).dest, dest)

    def test_multiline_dest(self):
        r = self._one("![a](\n  shot.png\n)")
        self.assertEqual(r.dest, "shot.png")
        self.assertEqual(r.form, "inline")

    def test_multiple_on_one_line(self):
        """同一行多个引用：平衡扫描不能互相吞并。"""
        refs = find_image_refs("![a](my file.png) 和 ![b](y.png)")
        self.assertEqual([r.dest for r in refs], ["my file.png", "y.png"])

    def test_reference_forms(self):
        text = "![a][r1]\n\n[r1]: shot.png \"标题\"\n"
        r = self._one(text)
        self.assertEqual((r.dest, r.form), ("shot.png", "reference"))
        self.assertIsNotNone(r.def_dest_span)

    def test_reference_empty_label_uses_alt(self):
        r = self._one("![shot.png][]\n\n[shot.png]: real.png\n")
        self.assertEqual(r.dest, "real.png")

    def test_reference_label_with_space_dest(self):
        """定义行目标含空格（非标准）也必须被识别，否则路径留在正文。"""
        r = self._one("![a][r1]\n\n[r1]: my file.png\n")
        self.assertEqual(r.dest, "my file.png")

    def test_shortcut_reference_needs_definition(self):
        r = self._one("![shot.png]\n\n[shot.png]: real.png\n")
        self.assertEqual(r.dest, "real.png")
        # 无定义 → CommonMark 视作字面文本，且不含目标，无泄漏风险
        self.assertEqual(find_image_refs("![shot.png] 只是方括号"), [])

    def test_html_forms(self):
        for text, dest, alt in [
            ('<img src="shot.png" width="200">', "shot.png", ""),
            ("<img src='shot.png'>", "shot.png", ""),
            ("<img src=shot.png>", "shot.png", ""),
            ('<img src="shot.png" alt="报错截图">', "shot.png", "报错截图"),
            ('<IMG SRC="shot.png">', "shot.png", ""),
        ]:
            with self.subTest(text=text):
                r = self._one(text)
                self.assertEqual((r.dest, r.form, r.alt), (dest, "html", alt))

    def test_unterminated_takes_rest_of_line(self):
        """未闭合 → 取本行剩余部分占位（不吞掉后续引用）。"""
        r = self._one("前文 ![a](broken 没有右括号")
        self.assertEqual(r.form, "unterminated")
        self.assertEqual(r.dest, "broken 没有右括号")

    def test_unterminated_stops_at_next_image(self):
        """未闭合引用不能把同一行后面的正常引用一起吃掉。"""
        refs = find_image_refs("![a](broken and ![b](y.png)")
        self.assertEqual([r.form for r in refs], ["unterminated", "inline"])
        self.assertEqual(refs[1].dest, "y.png")

    def test_plain_text_untouched(self):
        self.assertEqual(find_image_refs("没有图片，只有 [链接](x.md) 和文字"), [])


class TestCodeAwareness(unittest.TestCase):
    """代码区内的图片语法不处理（避免破坏示例 + 虚假 unresolved）。"""

    def test_fenced_backtick_block(self):
        text = "真图 ![a](x.png)\n\n```\n![b](y.png)\n```\n"
        refs = find_image_refs(text)
        self.assertEqual([r.dest for r in refs], ["x.png"])

    def test_fenced_tilde_block(self):
        text = "~~~\n![b](y.png)\n~~~\n真图 ![a](x.png)\n"
        self.assertEqual([r.dest for r in find_image_refs(text)], ["x.png"])

    def test_longer_closing_fence(self):
        text = "````\n![b](y.png)\n````\n![a](x.png)\n"
        self.assertEqual([r.dest for r in find_image_refs(text)], ["x.png"])

    def test_fence_with_info_string(self):
        text = "```markdown\n![b](y.png)\n```\n![a](x.png)\n"
        self.assertEqual([r.dest for r in find_image_refs(text)], ["x.png"])

    def test_inline_code(self):
        text = "行内 `![a](x.png)` 和 ``!`[b](y.png)`` 与真图 ![c](z.png)\n"
        self.assertEqual([r.dest for r in find_image_refs(text)], ["z.png"])

    def test_unclosed_fence_swallows_rest(self):
        text = "![a](x.png)\n\n```\n![b](y.png)\n"
        self.assertEqual([r.dest for r in find_image_refs(text)], ["x.png"])

    def test_definition_inside_code_ignored(self):
        text = "```\n[r1]: evil.png\n```\n![a][r1]\n"
        r = find_image_refs(text)
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].dest, "")          # 定义在代码块里 → 不生效

    def test_code_spans_helper(self):
        spans = code_spans("a `b` c\n```\nd\n```\n")
        self.assertEqual(len(spans), 2)


class TestRewriteImages(unittest.TestCase):
    def test_reference_definition_dest_is_blanked(self):
        """引用式定义里的目标也要去路径（否则整行原文留在正文）。"""
        text = "![a][r1]\n\n[r1]: /srv/secret/shot.png\n"
        out, refs = rewrite_images(text, lambda r: f"[图片:{r.alt}]")
        self.assertNotIn("/srv/secret", out)
        self.assertIn("[图片:a]", out)
        self.assertIn("[r1]: [图片]", out)

    def test_shared_definition_replaced_once(self):
        text = "![a][r1]\n\n![b][r1]\n\n[r1]: /srv/x.png\n"
        out, _ = rewrite_images(text, lambda r: f"[图片:{r.alt}]")
        self.assertNotIn("/srv/x.png", out)
        self.assertEqual(out.count("[图片"), 3)   # 两处引用 + 一处定义目标


class TestMarkdownImageIntegration(unittest.TestCase):
    """真实 canonicalize：图片被收集 + evidence 正确。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.md_dir = self.root / "data" / "imports" / "md"
        self.md_dir.mkdir(parents=True)
        (self.root / "data" / "outside").mkdir(parents=True)
        png(self.md_dir / "shot.png")
        png(self.md_dir / "imgs" / "sub.png")
        png(self.md_dir / "my file.png")
        png(self.md_dir / "img(1).png")
        png(self.root / "data" / "outside" / "out.png")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, body: str) -> tuple[str, list[dict]]:
        (self.md_dir / "doc.md").write_text(body, encoding="utf-8")
        cfg = SourceCfg(id="wiki", type="markdown", path="data/imports/md",
                        title_pattern=r"^#\s+(.+)", enabled=True)
        d = MarkdownSource(cfg, project_root=self.root).canonicalize()[0]
        return d.body, d.extra["evidence"]

    def test_lenient_forms_are_collected(self):
        body, ev = self._run(
            "# T\n\n"
            "1 ![空格](my file.png)\n"
            "2 ![括号](img(1).png)\n"
            "3 ![尖括号](<my file.png>)\n"
            "4 ![子目录](imgs/sub.png)\n"
            "5 ![父目录](../../outside/out.png)\n"
            "6 ![引用式][r1]\n\n[r1]: shot.png\n"
            "7 <img src=\"imgs/sub.png\" alt=\"HTML图\" width=\"200\">\n"
        )
        self.assertEqual([e["kind"] for e in ev], ["local"] * 7)
        for e in ev:
            self.assertTrue(e["sha256"])
            self.assertNotIn("path", e)
        names = {p.name for p in (self.root / "data" / "assets" / "images").glob("*")}
        self.assertIn("my file.png", names)
        self.assertIn("img(1).png", names)
        self.assertIn("sub.png", names)
        self.assertIn("shot.png", names)
        self.assertIn("out.png", names)
        self.assertIn("[图片:HTML图]", body)

    def test_code_block_images_left_alone(self):
        body, ev = self._run(
            "# T\n\n真图 ![a](shot.png)\n\n```\n![b](shot.png)\n```\n\n行内 `![c](shot.png)`\n"
        )
        self.assertEqual(len(ev), 1)                 # 只有真图
        self.assertIn("```\n![b](shot.png)\n```", body)   # 代码块原样保留
        self.assertIn("`![c](shot.png)`", body)           # 行内代码原样保留
        self.assertIn("[图片:a]", body)


class TestNoPathLeak(unittest.TestCase):
    """硬约束护栏：任意形态下正文都不得残留原始目标。"""

    # 覆盖：已存在/不存在、空格、括号、尖括号、引用式、HTML、未闭合、绝对路径、反斜杠
    BODY = (
        "# 泄漏探针\n\n"
        "01 ![a](shot.png)\n"
        "02 ![b](my file.png)\n"
        "03 ![c](img(1).png)\n"
        "04 ![d](<my file.png>)\n"
        "05 ![e][r1]\n"
        "06 ![f][r2]\n"
        "07 ![g][missing]\n"
        "08 <img src=\"my file.png\">\n"
        "09 <img src='imgs/sub.png' width=10>\n"
        "10 ![j](nope_missing.png)\n"
        "11 ![k](broken_unterminated\n"
        "12 ![l](C:\\Users\\secret\\leak.png)\n"
        "13 ![m](/srv/internal/share/leak.png)\n"
        "14 ![n](file:///srv/internal/leak.png)\n"
        "15 ![o](data:image/png;base64,@@@notbase64@@@)\n"
        "16 ![p](https://example.com/remote.png)\n\n"
        "[r1]: shot.png\n"
        "[r2]: my file.png\n"
        "[missing]: nowhere.png\n"
    )

    # 正文里绝不允许出现的片段（原始目标/目录形态）
    FORBIDDEN = [
        "shot.png", "my file.png", "img(1).png", "sub.png",
        "nope_missing.png", "broken_unterminated", "secret", "leak.png",
        "/srv/internal", "share", "@@@notbase64@@@", "nowhere.png",
    ]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.md_dir = self.root / "data" / "imports" / "md"
        self.md_dir.mkdir(parents=True)
        png(self.md_dir / "shot.png")
        png(self.md_dir / "my file.png")
        png(self.md_dir / "img(1).png")
        png(self.md_dir / "imgs" / "sub.png")

    def tearDown(self):
        self.tmp.cleanup()

    def _body(self):
        (self.md_dir / "leak.md").write_text(self.BODY, encoding="utf-8")
        cfg = SourceCfg(id="wiki", type="markdown", path="data/imports/md",
                        title_pattern=r"^#\s+(.+)", enabled=True)
        return MarkdownSource(cfg, project_root=self.root).canonicalize()[0]

    def test_no_raw_target_in_body(self):
        body = self._body().body
        for frag in self.FORBIDDEN:
            with self.subTest(fragment=frag):
                self.assertNotIn(frag, body)

    def test_every_image_ref_became_placeholder(self):
        body = self._body().body
        # 16 处引用 + 3 条引用式定义的目标（[r1]/[r2]/[missing] 也去路径占位）
        self.assertEqual(body.count("[图片"), 19)
        self.assertNotIn("![", body)                   # 没有残留的图片语法
        self.assertNotIn("<img", body.lower())

    def test_evidence_count_matches(self):
        ev = self._body().extra["evidence"]
        self.assertEqual(len(ev), 16)
        kinds = [e["kind"] for e in ev]
        # 01-06 与 08-09 目标真实存在 → local；07/10-15 不存在或不可解析 → unresolved；16 外链
        self.assertEqual(kinds.count("local"), 8, kinds)
        self.assertEqual(kinds.count("unresolved"), 7, kinds)
        self.assertEqual(kinds.count("remote"), 1, kinds)
        self.assertEqual(kinds.count("base64"), 0)     # 15 是坏 base64 → unresolved
        # remote 保留 http(s) URL（文档自身内容，非服务器路径）
        remote = [e for e in ev if e["kind"] == "remote"][0]
        self.assertEqual(remote["source_ref"], "https://example.com/remote.png")

    def test_no_path_field_in_any_evidence(self):
        for e in self._body().extra["evidence"]:
            self.assertNotIn("path", e)

    def test_canonical_extra_serializes_without_paths(self):
        """整条 extra 序列化后也不含被禁片段（白名单之外不得夹带）。"""
        doc = self._body()
        blob = json.dumps(doc.extra, ensure_ascii=False)
        for frag in self.FORBIDDEN:
            with self.subTest(fragment=frag):
                self.assertNotIn(frag, blob)


class _MdCase(unittest.TestCase):
    """共用脚手架：临时项目根 + 一个 markdown source。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.md_dir = self.root / "data" / "imports" / "md"
        self.md_dir.mkdir(parents=True)
        self.cfg = SourceCfg(id="wiki", type="markdown", path="data/imports/md",
                             title_pattern=r"^#\s+(.+)", enabled=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _src(self) -> MarkdownSource:
        return MarkdownSource(self.cfg, project_root=self.root)

    def _write(self, rel: str, body: str) -> Path:
        p = self.md_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
        return p

    def _images(self) -> set[str]:
        d = self.root / "data" / "assets" / "images"
        return {p.name for p in d.glob("*")} if d.exists() else set()


class TestFallbackMode(_MdCase):
    """imports 缺失 → 回退 assets/md 扁平副本：图片按文件名尽力反查，正文仍不含路径。"""

    def setUp(self):
        super().setUp()
        png(self.md_dir / "shot.png")
        png(self.md_dir / "imgs" / "sub.png")
        self._write("doc.md",
                    "# T\n\n"
                    "同目录 ![a](shot.png)\n"
                    "子目录 ![b](imgs/sub.png)\n"
                    "从未收集 ![c](never.png)\n")
        self.src = self._src()
        self.src.pull()                 # 填 assets/md
        self.src.canonicalize()         # 首次（imports 在）→ 图片进 assets/images
        self.assertTrue({"shot.png", "sub.png"} <= self._images())
        shutil.rmtree(self.md_dir)      # 模拟 imports 被清空

    def test_fallback_resolves_by_filename_and_warns(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            docs = self.src.canonicalize()
        out = buf.getvalue()
        self.assertIn("回退到资产层副本", out)          # 显式告警
        self.assertEqual(len(docs), 1)
        d = docs[0]
        self.assertEqual(d.source_id, "md:doc")        # 唯一 stem → id 不漂移
        self.assertEqual(d.extra["quality"]["source_mode"], "assets_fallback")
        # 同目录图 + 扁平化后的子目录图（按文件名）都反查命中；从未收集的仍未命中
        self.assertEqual([e["kind"] for e in d.extra["evidence"]],
                         ["local", "local", "unresolved"])
        self.assertEqual(d.extra["quality"]["images_unresolved"], 1)
        self.assertIn("[图片:a]", d.body)
        self.assertIn("[图片:b]", d.body)

    def test_fallback_body_has_no_path(self):
        """硬约束在回退模式下同样成立（否则相对路径会原样进库）。"""
        d = self.src.canonicalize()[0]
        for frag in ("shot.png", "sub.png", "never.png", "imgs/"):
            with self.subTest(fragment=frag):
                self.assertNotIn(frag, d.body)
        self.assertNotIn("![", d.body)

    def test_imports_mode_marked(self):
        """imports 在时标记为 imports（对照组）。"""
        png(self.md_dir / "shot.png")
        self._write("doc.md", "# T\n\n![a](shot.png)\n")
        d = self._src().canonicalize()[0]
        self.assertEqual(d.extra["quality"]["source_mode"], "imports")
        self.assertEqual(d.extra["quality"]["images_unresolved"], 0)


class TestSameStemDisambiguation(_MdCase):
    """同名 stem 不再互相覆盖（ingest 用 INSERT OR REPLACE，后者胜）。"""

    def setUp(self):
        super().setUp()
        self._write("sub_a/same.md", "# A\n\n甲\n")
        self._write("sub_b/same.md", "# B\n\n乙\n")
        self._write("uniq.md", "# U\n\n丙\n")

    def test_distinct_ids_and_no_churn_for_unique(self):
        docs = self._src().canonicalize()
        ids = sorted(d.source_id for d in docs)
        self.assertEqual(len(docs), 3)
        self.assertEqual(len(set(ids)), 3, ids)         # 不再互相覆盖
        self.assertIn("md:uniq", ids)                   # 唯一 stem 保持原 id（存量不漂移）
        dup = [i for i in ids if i.startswith("md:same--")]
        self.assertEqual(len(dup), 2, ids)
        for i in dup:
            self.assertRegex(i, r"^md:same--[0-9a-f]{8}$")

    def test_fingerprint_hides_directory_names(self):
        """source_id 会出现在 /search 的 doc_id 与遥测库里 → 不能含目录名。"""
        docs = self._src().canonicalize()
        for d in docs:
            for frag in ("sub_a", "sub_b", "same.md", "/"):
                self.assertNotIn(frag, d.source_id)

    def test_warns_on_collision(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._src().canonicalize()
        self.assertIn("同名 md", buf.getvalue())

    def test_bodies_are_not_crossed(self):
        """消歧后两篇内容各自独立（修复前 INSERT OR REPLACE 后者覆盖前者）。"""
        bodies = {d.body for d in self._src().canonicalize()}
        self.assertEqual(len(bodies), 3)
        joined = "".join(bodies)
        self.assertIn("甲", joined)
        self.assertIn("乙", joined)

    def test_path_tag_is_stable_and_normalized(self):
        self.assertEqual(_path_tag("a/b.md"), _path_tag("a\\b.md"))   # 分隔符归一
        self.assertEqual(len(_path_tag("x")), 8)
        self.assertNotEqual(_path_tag("sub_a/same.md"), _path_tag("sub_b/same.md"))


class TestBase64ContentAddressing(_MdCase):
    """内嵌图按内容寻址：同图跨文档只落一份，命名不含 md 文件名。"""

    def test_same_image_shared_across_docs(self):
        b64 = png_b64()
        for name in ("one.md", "two.md"):
            self._write(name, f"# {name}\n\n![内嵌](data:image/png;base64,{b64})\n")
        docs = self._src().canonicalize()
        self.assertEqual(len(docs), 2)
        shas = {d.extra["evidence"][0]["sha256"] for d in docs}
        self.assertEqual(len(shas), 1)                  # 同一张图 → 同一个 sha
        names = [n for n in self._images() if n.startswith("img_")]
        self.assertEqual(len(names), 1, names)          # 只落一份（去重）
        self.assertRegex(names[0], r"^img_[0-9a-f]{16}\.png$")
        for d in docs:
            self.assertNotIn("one", d.extra["evidence"][0].get("asset_id", ""))
            self.assertIn("[图片:内嵌]", d.body)

    def test_different_images_get_different_names(self):
        a, b = png_b64(), png_b64("blue")
        self._write("d.md", f"# T\n\n![a](data:image/png;base64,{a})\n"
                            f"![b](data:image/png;base64,{b})\n")
        d = self._src().canonicalize()[0]
        self.assertEqual(len({e["sha256"] for e in d.extra["evidence"]}), 2)
        self.assertEqual(len([n for n in self._images() if n.startswith("img_")]), 2)


if __name__ == "__main__":
    unittest.main()