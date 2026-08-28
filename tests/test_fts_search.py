"""FTS5 中文分词（jieba）测试：分词/降级/词典词防拆分/中文子词 FTS 命中/build_fts 重建。"""
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from vllm_kb.config import AppConfig
from vllm_kb.embed import EmbeddingClient
from vllm_kb.fts_tokenizer import query_tokens, register_words, tokenize_text
from vllm_kb.ingest import ingest_docs
from vllm_kb.models import KbDocument
from vllm_kb.vectorstore import PythonVectorStore


class TestTokenizer(unittest.TestCase):
    def test_tokenize_chinese_subwords(self):
        """中文词被 jieba 独立切出（"超时"可被 FTS 命中；"超时"是 jieba 内置词）。"""
        t = tokenize_text("检查通信超时原因")
        toks = t.split()
        self.assertIn("超时", toks)
        self.assertIn("检查", toks)

    def test_query_tokens(self):
        toks = query_tokens("vllm-ascend HCCL 超时")
        self.assertIn("HCCL", toks)
        self.assertIn("超时", toks)
        self.assertNotIn("", toks)

    def test_register_words_keeps_compound(self):
        """标签词典词注册后复合词不被拆散。"""
        register_words(["超时排查"])
        toks = tokenize_text("HCCL超时排查指南").split()
        self.assertIn("超时排查", toks)

    def test_downgrade_without_jieba(self):
        """jieba 不可用时降级：索引写入原文、查询按中英 token 提取（旧行为）。"""
        with mock.patch("vllm_kb.fts_tokenizer._jieba", False):
            self.assertEqual(tokenize_text("超时排查 指南"), "超时排查 指南")
            toks = query_tokens("超时排查 ab")
            self.assertEqual(toks, ["超时排查", "ab"])  # 中文整段一个 token


def make_cfg(tmp: Path) -> AppConfig:
    data_root = tmp / "data_root"
    data_root.mkdir()
    os.environ["VLLM_KB_DATA_ROOT"] = str(data_root)
    cfg_path = tmp / "config.json"
    cfg_path.write_text(json.dumps({
        "embedding": {"provider": "echo", "dimensions": 64},
        "tags": {"registry": [{"name": "超时排查", "tier": "purpose"}]},
        "storage": {
            "vector_backend": "python",
            "lancedb_path": "data/lancedb",
            "sqlite_path": "data/kb.sqlite3",
            "canonical_file": "data/raw/canonical.jsonl",
            "review_path": "data/review.sqlite3",
        },
    }), encoding="utf-8")
    return AppConfig.load(str(cfg_path), require_keys=False)


CN_DOC = KbDocument(
    source_type="doc_markdown",
    source_id="md:guide",
    url="",
    title="HCCL 超时排查指南",
    body="HCCL 超时排查指南：检查节点间通信超时原因，hccn_tool 查看链路状态。",
    tags=["超时排查"],
    extra={"verification": "unverified"},
)


class TestChineseFtsHit(unittest.TestCase):
    """中文子词 FTS 命中：查询"超时"命中含"超时排查"的 chunk（旧 unicode61 整段无法命中）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.store = PythonVectorStore(self.cfg.resolve("data/vec.json"))
        ingest_docs(self.cfg, [CN_DOC], EmbeddingClient(self.cfg.embedding), self.store)

    def tearDown(self):
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def _fts_hits(self, query: str):
        from vllm_kb.search import SearchEngine

        engine = SearchEngine(self.cfg, read_only=True)
        conn = engine._ro_conn()
        try:
            return engine._fts_search(query, limit=5, conn=conn)
        finally:
            conn.close()
            engine.close()

    def test_chinese_subword_hit(self):
        hits = self._fts_hits("超时")
        self.assertTrue(hits, "查询'超时'应命中含'超时排查'的 chunk")
        # 返回文本为原文（snippet 展示，非分词版）
        text = next(iter(hits.values()))[1]
        self.assertIn("超时排查", text)

    def test_indexed_text_is_tokenized(self):
        conn = sqlite3.connect(f"file:{self.cfg.resolve('data/kb.sqlite3').as_posix()}?mode=ro",
                               uri=True)
        try:
            row = conn.execute(
                "SELECT indexed_text, text FROM chunks_fts WHERE doc_id='md:guide' LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        self.assertIn("超时", row[0].split())   # 索引列分词
        self.assertEqual(row[1], CN_DOC.body)  # 原文列保留


class TestBuildFtsScript(unittest.TestCase):
    """build_fts.py：基于现有 chunk 原文重新分词重建 FTS（不重嵌）。"""

    @classmethod
    def setUpClass(cls):
        _spec = importlib.util.spec_from_file_location(
            "build_fts", PROJECT_ROOT / "scripts" / "build_fts.py")
        cls.build_fts = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(cls.build_fts)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.store = PythonVectorStore(self.cfg.resolve("data/vec.json"))
        ingest_docs(self.cfg, [CN_DOC], EmbeddingClient(self.cfg.embedding), self.store)
        # 破坏索引列（模拟旧/脏数据），验证重建恢复
        conn = sqlite3.connect(str(self.cfg.resolve("data/kb.sqlite3")))
        conn.execute("UPDATE chunks_fts SET indexed_text='污染 索引'")
        conn.commit()
        conn.close()

    def tearDown(self):
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def test_rebuild_restores_tokenized_index(self):
        with mock.patch.object(sys, "argv",
                               ["build_fts.py", "--config", str(self.root / "config.json")]):
            self.build_fts.main()
        conn = sqlite3.connect(f"file:{self.cfg.resolve('data/kb.sqlite3').as_posix()}?mode=ro",
                               uri=True)
        try:
            row = conn.execute(
                "SELECT indexed_text, text FROM chunks_fts WHERE doc_id='md:guide' LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        self.assertIn("超时", row[0].split())   # 重建后分词索引恢复
        self.assertEqual(row[1], CN_DOC.body)  # 原文保留
        # 向量库未动（chunk 数不变，无需重嵌）
        self.assertEqual(self.store.count(), len(CN_DOC.body) > 0 and 1)


if __name__ == "__main__":
    unittest.main()
