"""能力验证测试：错误短语学习、标题精确检索、信号词命中。"""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from vllm_kb.symbol_table import SymbolTable, _learn_phrases_from_text, load_signal_words, match_signal_words


class TestPhraseLearning(unittest.TestCase):
    def test_learn_vector_core_phrase(self):
        """能力1：驱动层报错短语（vector core timeout）从正文学习。"""
        t = SymbolTable()
        text = (
            "rtDeviceSynchronizeWithTimeout execution failed, reason=vector core timeout\n"
            "rtDeviceSynchronizeWithTimeout execution failed, reason=vector core timeout\n"
            "rtDeviceSynchronizeWithTimeout execution failed, reason=vector core timeout\n"
        )
        _learn_phrases_from_text(text, t)
        e = t.get("vector core timeout")
        self.assertIsNotNone(e, "vector core timeout 应被学习为 phrase")
        self.assertEqual(e.kind, "phrase")

    def test_learn_requires_frequency(self):
        """出现 1 次不学（防噪音）。"""
        t = SymbolTable()
        _learn_phrases_from_text("reason=vector core timeout", t)
        self.assertIsNone(t.get("vector core timeout"))

    def test_learn_skips_stopwords(self):
        t = SymbolTable()
        _learn_phrases_from_text("error occurred\n" * 5, t)
        self.assertIsNone(t.get("error occurred"))


class TestTitleSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = sqlite3.connect(str(Path(self.tmp.name) / "t.sqlite3"))
        self.conn.executescript(
            """
            CREATE TABLE docs (source_id TEXT PRIMARY KEY, title TEXT, url TEXT,
                               component TEXT, resolved_at TEXT);
            """
        )
        self.conn.executemany(
            "INSERT INTO docs VALUES (?,?,?,?,?)",
            [
                ("a:1", "[Bug]: NPU Vector Core Exception when using MTP",
                 "http://1", "vllm-ascend", None),
                ("a:2", "[Bug] GLM-5.1 PD分离 D节点 aivector error",
                 "http://2", "vllm-ascend", "2026-01-01"),
                ("a:3", "[Perf] vector core number function",
                 "http://3", "vllm-ascend", "2026-01-02"),
                ("b:1", "CUDA illegal memory access",
                 "http://4", "vllm", "2026-01-03"),
            ],
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_title_contains(self):
        from vllm_kb.search import title_search

        hits = title_search(self.conn, "vector core")
        self.assertEqual(len(hits), 2)  # a:1, a:3
        self.assertTrue(any("NPU Vector Core Exception" in h.title for h in hits))

    def test_title_component_filter(self):
        from vllm_kb.search import title_search

        hits = title_search(self.conn, "vector core", component="vllm-ascend")
        self.assertEqual(len(hits), 2)
        hits2 = title_search(self.conn, "memory", component="vllm-ascend")
        self.assertEqual(hits2, [])

    def test_title_prefix(self):
        from vllm_kb.search import title_search

        hits = title_search(self.conn, "[Bug]", match="prefix")
        self.assertEqual(len(hits), 2)

    def test_title_matches_source_id_without_exposing_filename(self):
        """业务文档主题词在文件名（source_id）里：title 检索按 source_id 命中；
        返回的 title 字段仍是文档标题（不把文件名当标题暴露）。"""
        from vllm_kb.search import title_search

        self.conn.execute(
            "INSERT INTO docs VALUES (?,?,?,?,?)",
            ("pdf:npu-smi 命令参考 02", "Atlas A3 中心推理和训练硬件", "", "", None),
        )
        self.conn.commit()
        # 主题词只在 source_id（文件名）→ 按 source_id 命中
        hits = title_search(self.conn, "npu-smi")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].doc_id, "pdf:npu-smi 命令参考 02")
        # 返回的 title 是文档标题，不含文件名主题词（不新增文件名暴露面）
        self.assertEqual(hits[0].title, "Atlas A3 中心推理和训练硬件")
        self.assertNotIn("npu-smi", hits[0].title)
        # 标题本身含词时照常命中（两字段 OR）
        hits2 = title_search(self.conn, "Atlas")
        self.assertGreaterEqual(len(hits2), 1)


class TestSignalWords(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sig_path = self.root / "signal_words.json"
        self.sig_path.write_text(
            json.dumps({"words": [
                {"word": "MTP", "count": 10, "docs": 10, "idf": 2.0, "score": 20.0},
                {"word": "Vector Core", "count": 3, "docs": 3, "idf": 4.0, "score": 12.0},
                {"word": "RFC", "count": 100, "docs": 100, "idf": 1.0, "score": 100.0},
            ]}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_match_signal_words(self):
        hits = match_signal_words(
            json.loads(self.sig_path.read_text(encoding="utf-8"))["words"],
            "MTP Vector Core crash",
        )
        words = {h["word"] for h in hits}
        # 长词优先：Vector Core（双词短语）命中；MTP（3字符）被 min_word_len=4 过滤
        self.assertIn("Vector Core", words)
        self.assertNotIn("RFC", words)

    def test_match_signal_words_short_allowed(self):
        """显式放低 min_word_len 时短词可命中（agent 需要时）。"""
        hits = match_signal_words(
            json.loads(self.sig_path.read_text(encoding="utf-8"))["words"],
            "MTP crash", min_word_len=2,
        )
        words = {h["word"] for h in hits}
        self.assertIn("MTP", words)

    def test_match_signal_words_with_phrase_signals(self):
        """已学习错误短语作为信号源（罕见但高价值，不经过统计截断）。"""
        sig = json.loads(self.sig_path.read_text(encoding="utf-8"))["words"]
        phrases = [{"word": "vector core timeout", "weight": 1.5}]
        hits = match_signal_words(sig, "rtDeviceSynchronizeWithTimeout vector core timeout", phrase_signals=phrases)
        words = {h["word"] for h in hits}
        self.assertIn("vector core timeout", words)


if __name__ == "__main__":
    unittest.main()
