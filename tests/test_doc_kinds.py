"""文档种类声明表（doc_kinds）护栏测试。

这张表是"同一个知识只声明一次"的落点。**护栏的意义**在于：新增一种文件型来源时，
声明表、来源类、分块策略、审核台收录范围四处必须自动一致——漏一处就是静默 bug
（典型症状：新来源的 chunk 永远没有 section，或它在「文档管理」页里看不见）。
"""
import unittest

from vllm_kb import doc_kinds as dk
from vllm_kb import sources as src


class TestKindTable(unittest.TestCase):
    def test_all_sources_declare_a_kind(self):
        """每个**文件型**来源类都必须挂上 kind（github/image 不是文件型，允许为空）。"""
        for name, cls in src._REGISTRY.items():
            with self.subTest(source=name):
                if name in ("github", "image"):
                    self.assertIsNone(cls.kind)
                    continue
                self.assertIsNotNone(cls.kind, f"{name} 未声明 DocKind")
                # 声明表里的 source 必须与注册表 key 一致
                self.assertEqual(cls.kind.source, name)
                self.assertEqual(cls.type, name)

    def test_kind_source_type_matches_registry_keys(self):
        self.assertEqual(set(dk.BY_SOURCE), set(src._REGISTRY) - {"github", "image"})

    def test_no_duplicate_source_type_or_source(self):
        self.assertEqual(len(dk.BY_TYPE), len(dk.KINDS))
        self.assertEqual(len(dk.BY_SOURCE), len(dk.KINDS))

    def test_external_doc_types_matches_legacy_literal(self):
        """审核台收录范围原先是一份硬编码字面量（且没人引用）——派生结果必须逐字相等。"""
        self.assertEqual(dk.EXTERNAL_DOC_TYPES,
                         ("doc_pdf", "doc_markdown", "doc_word", "doc_excel", "doc_other"))

    def test_patterns_derived_from_suffixes(self):
        self.assertEqual(dk.BY_SOURCE["pdf"].patterns, ("*.pdf",))
        self.assertEqual(dk.BY_SOURCE["markdown"].patterns, ("*.md", "*.markdown"))
        self.assertEqual(dk.BY_SOURCE["word"].patterns, ("*.docx", "*.docm"))
        self.assertEqual(dk.BY_SOURCE["excel"].patterns, ("*.xlsx", "*.xlsm"))

    def test_suffixes_are_bare(self):
        """后缀统一成裸后缀（原先 word 存 `*.docx`、excel 存 `.xlsx` 两种写法混用）。"""
        for k in dk.KINDS:
            with self.subTest(kind=k.source_type):
                for s in k.suffixes:
                    self.assertTrue(s.startswith("."))
                    self.assertNotIn("*", s)

    def test_default_verification_is_a_known_level(self):
        from vllm_kb.review import VERIFICATION_LEVELS

        for k in dk.KINDS:
            with self.subTest(kind=k.source_type):
                self.assertIn(k.default_verification, VERIFICATION_LEVELS)

    def test_pdf_is_expert_others_unverified(self):
        self.assertEqual(dk.BY_TYPE["doc_pdf"].default_verification, "expert")
        for st in ("doc_markdown", "doc_word", "doc_excel"):
            self.assertEqual(dk.BY_TYPE[st].default_verification, "unverified")


class TestSplitterDerivation(unittest.TestCase):
    def test_section_splitter_for(self):
        self.assertEqual(dk.section_splitter_for("doc_pdf"), dk.SPLITTER_PDF)
        self.assertEqual(dk.section_splitter_for("doc_markdown"), dk.SPLITTER_MARKDOWN)
        self.assertEqual(dk.section_splitter_for("doc_word"), dk.SPLITTER_MARKDOWN)
        self.assertEqual(dk.section_splitter_for("doc_excel"), dk.SPLITTER_LINES)
        self.assertEqual(dk.section_splitter_for("github_issue"), "")
        self.assertEqual(dk.section_splitter_for("不存在的类型"), "")

    def test_every_declared_splitter_has_an_implementation(self):
        """声明了策略名却没有实现 → 该来源的 section 会静默为空。"""
        from vllm_kb.chunking import _SPLITTERS

        for k in dk.KINDS:
            with self.subTest(kind=k.source_type):
                if k.section_splitter:
                    self.assertIn(k.section_splitter, _SPLITTERS)

    def test_chunking_uses_declaration(self):
        """doc_pdf 走编号标题切分、doc_markdown 走 `#` 标题切分（各自拿到 section）。"""
        from vllm_kb.chunking import chunk_doc
        from vllm_kb.models import KbDocument

        pdf = KbDocument(source_type="doc_pdf", source_id="pdf:x", url="", title="t",
                         body="2.34 获取network版本号信息\n\n正文内容")
        self.assertEqual(chunk_doc(pdf)[0].section, "2.34 获取network版本号信息")
        md = KbDocument(source_type="doc_markdown", source_id="md:x", url="", title="t",
                        body="# 安装步骤\n\n正文内容")
        self.assertEqual(chunk_doc(md)[0].section, "安装步骤")
        # 未声明种类的来源：不切 section（保持原逻辑）
        gh = KbDocument(source_type="github_issue", source_id="i:1", url="", title="t",
                        body="# 这不是标题结构\n\n正文内容")
        self.assertEqual(chunk_doc(gh)[0].section, "")


class TestExternalDocScope(unittest.TestCase):
    def test_review_uses_the_declaration(self):
        from vllm_kb import review

        self.assertEqual(review._EXTERNAL_SOURCE_TYPES, dk.EXTERNAL_DOC_TYPES)

    def test_list_external_docs_includes_declared_and_excludes_github(self):
        """真的接上了：声明里的种类能查到，github_* 查不到（原先靠 NOT LIKE 的 hack）。"""
        import sqlite3
        import tempfile
        from pathlib import Path

        from vllm_kb.review import list_external_docs

        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp) / "kb.sqlite3"
            conn = sqlite3.connect(str(kb))
            conn.execute("""CREATE TABLE docs (source_id TEXT PRIMARY KEY, source_type TEXT,
                            title TEXT, component TEXT, url TEXT, extra TEXT)""")
            conn.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?)", [
                ("pdf:a", "doc_pdf", "手册", "", "", "{}"),
                ("md:b", "doc_markdown", "案例", "", "", "{}"),
                ("excel:c", "doc_excel", "登记表", "", "", "{}"),
                ("doc_other:d", "doc_other", "其它", "", "", "{}"),
                ("o/r#1", "github_issue", "issue", "", "", "{}"),
                ("o/r#2", "github_pr", "pr", "", "", "{}"),
            ])
            conn.commit()
            conn.close()
            ids = {d["source_id"] for d in list_external_docs(kb)}
            self.assertEqual(ids, {"pdf:a", "md:b", "excel:c", "doc_other:d"})


if __name__ == "__main__":
    unittest.main()
