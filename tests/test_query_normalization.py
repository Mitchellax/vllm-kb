"""查询侧 CJK↔拉丁边界空格归一测试（业务实测："芯片MAC 地址" 查不到 "芯片MAC地址"）。"""
import sqlite3
import unittest

from vllm_kb.fts_tokenizer import query_tokens, strip_cjk_latin_spaces, tokenize_text
from vllm_kb.signature import _fts_phrase_sql
from vllm_kb.search import title_search


class TestStripCjkLatinSpaces(unittest.TestCase):
    def test_variant_groups(self):
        self.assertEqual(strip_cjk_latin_spaces("芯片MAC 地址"), "芯片MAC地址")
        self.assertEqual(strip_cjk_latin_spaces("配置pip 源"), "配置pip源")
        self.assertEqual(strip_cjk_latin_spaces("升级NPU 驱动"), "升级NPU驱动")
        # 纯拉丁/纯 CJK 词边界空格保留
        self.assertEqual(strip_cjk_latin_spaces("kernel NULL pointer"), "kernel NULL pointer")
        self.assertEqual(strip_cjk_latin_spaces("方式二 构建容器镜像"), "方式二 构建容器镜像")
        self.assertEqual(strip_cjk_latin_spaces("HCCL 超时排查"), "HCCL超时排查")
        # 全角空格与连续空白
        self.assertEqual(strip_cjk_latin_spaces("芯片　MAC  地址"), "芯片MAC地址")

    def test_query_tokens_converge(self):
        self.assertEqual(query_tokens("芯片MAC 地址"), query_tokens("芯片MAC地址"))
        self.assertEqual(query_tokens("配置pip 源"), query_tokens("配置pip源"))

    def test_tokenize_text_index_side(self):
        # 索引侧同规则：带空格与不带空格的原文分词结果一致
        self.assertEqual(tokenize_text("芯片MAC 地址的配置"), tokenize_text("芯片MAC地址的配置"))


class TestFtsVariantHit(unittest.TestCase):
    """mini FTS 端到端：索引 '芯片MAC地址' 类文本，带空格查询必须命中。"""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, doc_id UNINDEXED, text)")
        rows = [
            ("c1", "d1", tokenize_text("芯片MAC地址查询命令说明")),
            ("c2", "d2", tokenize_text("查看网卡的 MAC 地址配置")),
        ]
        self.conn.executemany("INSERT INTO chunks_fts VALUES(?,?,?)", rows)

    def test_phrase_variant_hits(self):
        # "芯片MAC 地址"（用户带空格）命中 c1（文档无空格）
        hits = _fts_phrase_sql(self.conn, "芯片mac 地址", limit=10)
        self.assertIn("c1", {h[0] for h in hits})

    def test_plain_still_hits(self):
        hits = _fts_phrase_sql(self.conn, "mac 地址", limit=10)
        self.assertIn("c2", {h[0] for h in hits})

    def test_title_search_variant(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE docs (source_id TEXT, title TEXT, url TEXT,"
                     " component TEXT, resolved_at TEXT)")
        conn.execute("INSERT INTO docs VALUES('md:a','芯片MAC地址查询','','',NULL)")
        hits = title_search(conn, "芯片MAC 地址")
        self.assertEqual([h.doc_id for h in hits], ["md:a"])


if __name__ == "__main__":
    unittest.main()
