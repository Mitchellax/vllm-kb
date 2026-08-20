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


if __name__ == "__main__":
    unittest.main()
