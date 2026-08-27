"""分块逻辑单元测试。"""
import unittest

from vllm_kb.chunking import chunk_doc
from vllm_kb.models import KbDocument


def make_doc(body: str, source_id="github:issue:1") -> KbDocument:
    return KbDocument(
        source_type="github_issue",
        source_id=source_id,
        url="https://example.com/1",
        title="t",
        body=body,
    )


class TestChunking(unittest.TestCase):
    def test_short_doc_single_chunk(self):
        doc = make_doc("hello world")
        chunks = chunk_doc(doc, max_chunk_chars=4000, overlap_chars=200)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "hello world")

    def test_empty_doc(self):
        self.assertEqual(chunk_doc(make_doc("   ")), [])

    def test_long_doc_splits(self):
        body = "\n\n".join(f"paragraph {i} " + "x" * 1000 for i in range(10))
        chunks = chunk_doc(make_doc(body), max_chunk_chars=3000, overlap_chars=100)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.text), 3000 + 100 + 50)  # 容忍重叠带来的超额
        # 内容不丢失（重叠会导致文本重复，但首块应包含开头）
        self.assertIn("paragraph 0", chunks[0].text)

    def test_chunk_ids_and_seq(self):
        body = "\n\n".join(f"paragraph {i} " + "y" * 500 for i in range(20))
        chunks = chunk_doc(make_doc(body), max_chunk_chars=1000, overlap_chars=0)
        self.assertGreater(len(chunks), 1)
        for i, c in enumerate(chunks):
            self.assertEqual(c.seq, i)
            self.assertEqual(c.chunk_id, f"github:issue:1#{i}")
            self.assertEqual(c.doc_id, "github:issue:1")

    def test_overlap_present(self):
        body = "\n\n".join(f"paragraph {i} " + "z" * 800 for i in range(6))
        chunks = chunk_doc(make_doc(body), max_chunk_chars=1000, overlap_chars=200)
        self.assertGreater(len(chunks), 1)
        # 第二块应包含第一块尾部内容
        self.assertIn(chunks[0].text[-200:].strip(), chunks[1].text)


class TestMarkdownSectionChunking(unittest.TestCase):
    def make_md(self, body: str) -> KbDocument:
        return KbDocument(
            source_type="doc_markdown",
            source_id="md:wiki",
            url="",
            title="t",
            body=body,
        )

    def test_markdown_sections_injected(self):
        """Markdown 按 # 标题切章节，chunk 前缀注入章节名（命中即知所属章节）。"""
        doc = self.make_md(
            "# GLM5.1 崩溃三板斧\n\n引言段落\n\n## 第一步\n检查 halMemCreate 日志\n"
            "## 第二步\n调小 mega_moe_max_tokens\n")
        chunks = chunk_doc(doc, max_chunk_chars=4000, overlap_chars=0)
        self.assertGreater(len(chunks), 1)
        sections = {c.section for c in chunks}
        self.assertIn("GLM5.1 崩溃三板斧", sections)  # 引言前标题
        self.assertIn("第一步", sections)
        self.assertIn("第二步", sections)
        # 章节文本前缀注入标题（参与匹配）
        step2 = next(c for c in chunks if c.section == "第二步")
        self.assertTrue(step2.text.startswith("【第二步】"))
        self.assertIn("mega_moe_max_tokens", step2.text)

    def test_markdown_no_heading_fallback(self):
        """无 # 标题的 md 回退普通分块（section 为空）。"""
        doc = self.make_md("段落一\n\n段落二 " + "x" * 2000)
        chunks = chunk_doc(doc, max_chunk_chars=1000, overlap_chars=0)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.section == "" for c in chunks))


if __name__ == "__main__":
    unittest.main()
