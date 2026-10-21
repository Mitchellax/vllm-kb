"""build_fts.py 重建源正确性测试（2026-10 业务实测回归钉）。

业务库事故：chunks_fts 历史 DROP 后未回填（meta 150202 行 / fts 29078 行），旧版
build_fts 从 chunks_fts **自身**读原文重建——缺行的表重建后仍缺行，缺口永远找不回。
修正后重建源 = chunks_meta（权威 chunk 清单）+ 向量库 texts_by_id()（权威原文）。

本测试用 PythonVectorStore（无 lancedb 依赖）复现该事故并验证找回。
"""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig
from vllm_kb.vectorstore import VectorItem

_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED, doc_id UNINDEXED,
  indexed_text,
  text UNINDEXED
);
"""

_DOCS = {
    "pdf:a": [("c-a-1", 0, "通信超时排查指南"), ("c-a-2", 1, "重试后恢复")],
    "pdf:b": [("c-b-1", 0, "检查节点间通信超时原因"), ("c-b-2", 1, "检查节点内存")],
}


class TestFtsRebuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        kb = self.root / "kb.sqlite3"
        conn = sqlite3.connect(str(kb))
        conn.executescript(_FTS_DDL)
        conn.execute("CREATE TABLE chunks_meta (chunk_id TEXT PRIMARY KEY, "
                     "doc_id TEXT, seq INTEGER, section TEXT)")
        # 模拟业务事故：meta 全量 4 行，fts 只有 doc:a 的 2 行（doc:b 缺行）
        vec_items = []
        for doc_id, chunks in _DOCS.items():
            for chunk_id, seq, text in chunks:
                conn.execute("INSERT INTO chunks_meta VALUES (?,?,?,?)",
                             (chunk_id, doc_id, seq, None))
                vec_items.append({"id": chunk_id, "vector": [0.1, 0.2],
                                  "meta": {"doc_id": doc_id}, "text": text})
                if doc_id == "pdf:a":   # 只有 pdf:a 进了 FTS
                    conn.execute("INSERT INTO chunks_fts (chunk_id, doc_id, "
                                 "indexed_text, text) VALUES (?,?,?,?)",
                                 (chunk_id, doc_id, text, text))
        conn.commit()
        conn.close()
        # 向量库（python 后端 JSON：build_vector_store 在 lancedb_path 后拼 "_py.json"）：
        # 4 条原文齐全
        vec_path = self.root / "vec"
        (self.root / "vec_py.json").write_text(
            json.dumps(vec_items, ensure_ascii=False), encoding="utf-8")
        cfg_path = self.root / "config.json"
        cfg_path.write_text(json.dumps({
            # build_fts 不调嵌入端点，validate_runtime 只要求字段在场
            "embedding": {"base_url": "http://localhost:9/v1", "api_key": "test"},
            "storage": {"vector_backend": "python",
                        "lancedb_path": str(vec_path),
                        "sqlite_path": str(kb)},
        }, ensure_ascii=False), encoding="utf-8")
        self.cfg_path = str(cfg_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_build_fts(self, *extra: str) -> None:
        import scripts.build_fts as bf

        argv_backup = sys.argv
        sys.argv = ["build_fts.py", "--config", self.cfg_path, *extra]
        try:
            bf.main()
        finally:
            sys.argv = argv_backup

    def _fts_rows(self) -> list[tuple[str, str, str, str]]:
        conn = sqlite3.connect(str(self.root / "kb.sqlite3"))
        try:
            return conn.execute("SELECT chunk_id, doc_id, indexed_text, text "
                                "FROM chunks_fts ORDER BY chunk_id").fetchall()
        finally:
            conn.close()

    def test_missing_chunks_recovered_from_vectorstore(self):
        """核心回归：FTS 缺行时重建**从向量库取原文**——4 行全部找回，不再以缺行表为源。"""
        self.assertEqual(len(self._fts_rows()), 2)   # 事故现场：只有 2 行
        self._run_build_fts()
        rows = self._fts_rows()
        self.assertEqual(len(rows), 4)               # 全量找回（此前重建后仍是 2）
        by_id = {r[0]: r for r in rows}
        self.assertEqual(by_id["c-b-1"][3], "检查节点间通信超时原因")   # 缺失行的原文正确
        self.assertEqual(by_id["c-b-1"][1], "pdf:b")
        self.assertEqual(by_id["c-a-2"][3], "重试后恢复")
        # 索引列是分词产物（jieba 在场时与原文不同；未装 jieba 降级原文，均可接受）
        try:
            import jieba  # noqa: F401
            has_jieba = True
        except ImportError:
            has_jieba = False
        if has_jieba:
            self.assertNotEqual(by_id["c-b-1"][2], by_id["c-b-1"][3])
        self.assertTrue(by_id["c-b-1"][2].strip())

    def test_rebuild_is_idempotent(self):
        self._run_build_fts()
        self._run_build_fts()
        self.assertEqual(len(self._fts_rows()), 4)   # 幂等：行数不增不减

    def test_limit_picks_prefix(self):
        self._run_build_fts("--limit", "3")
        self.assertEqual(len(self._fts_rows()), 3)

    def test_vectorstore_gap_reported_not_silent(self):
        """向量库也缺的 chunk 如实暴露（跳过并告警），不静默丢行。"""
        import io
        import contextlib

        items = json.loads((self.root / "vec_py.json").read_text(encoding="utf-8"))
        (self.root / "vec_py.json").write_text(
            json.dumps([i for i in items if i["id"] != "c-b-2"], ensure_ascii=False),
            encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._run_build_fts()
        rows = self._fts_rows()
        self.assertEqual(len(rows), 3)
        self.assertIn("c-b-2", buf.getvalue())       # 缺失告警可见
        self.assertIn("⚠️", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
