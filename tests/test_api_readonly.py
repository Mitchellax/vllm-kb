"""只读结构保证测试：SQLite mode=ro 拒绝写入、向量库写操作抛错、API 源码审计、只读检索可用。"""
import ast
import sqlite3
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig, PROJECT_ROOT
from vllm_kb.embed import EmbeddingClient
from vllm_kb.ingest import ingest_docs
from vllm_kb.models import KbDocument, VersionSpan
from vllm_kb.search import SearchEngine
from vllm_kb.vectorstore import PythonVectorStore, ReadOnlyError, ReadOnlyVectorStore

API_PATH = PROJECT_ROOT / "vllm_kb" / "api.py"

DOC = KbDocument(
    source_type="github_issue",
    source_id="github:vllm-project-vllm:issue:1",
    url="https://github.com/vllm-project/vllm/issues/1",
    title="[Bug]: illegal memory access",
    body="CUDA error: an illegal memory access was encountered. workaround: disable chunked prefill.",
    created_at="2026-01-01T00:00:00Z",
    resolved_at="2026-01-10T00:00:00Z",
    status="closed",
    version_span=VersionSpan(min="0.26.0"),
    component="vllm",
    extra={"kind": "bug"},
)


class TestSqliteReadOnly(unittest.TestCase):
    def test_readonly_connection_rejects_write(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "kb.sqlite3"
            # 先建可写库
            conn = sqlite3.connect(str(db))
            conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY, v TEXT)")
            conn.commit()
            conn.close()
            # 只读连接
            ro = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
            self.assertEqual(ro.execute("SELECT count(*) FROM t").fetchone()[0], 0)
            with self.assertRaises(sqlite3.OperationalError):
                ro.execute("INSERT INTO t VALUES ('x','y')")
            ro.close()

    def test_readonly_connection_missing_file_fails(self):
        with self.assertRaises(sqlite3.OperationalError):
            sqlite3.connect("file:C:/no/such/dir/kb.sqlite3?mode=ro", uri=True)


class TestReadOnlyVectorStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PythonVectorStore(Path(self.tmp.name) / "v.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_raise(self):
        ro = ReadOnlyVectorStore(self.store)
        with self.assertRaises(ReadOnlyError):
            ro.add_items([])
        with self.assertRaises(ReadOnlyError):
            ro.delete_doc("x")
        with self.assertRaises(ReadOnlyError):
            ro.update_doc_meta("x", {})
        with self.assertRaises(ReadOnlyError):
            ro.clear()

    def test_reads_delegate(self):
        from vllm_kb.vectorstore import VectorItem

        self.store.add_items([VectorItem(id="a", vector=[1.0, 0.0], meta={"doc_id": "d"}, text="t")])
        ro = ReadOnlyVectorStore(self.store)
        self.assertEqual(ro.count(), 1)
        self.assertEqual(ro.search([1.0, 0.0], top_k=1)[0].id, "a")


class TestReadOnlySearchEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = AppConfig.model_validate(
            {
                "embedding": {"provider": "echo", "dimensions": 1024},
                "storage": {
                    "vector_backend": "python",
                    "lancedb_path": str(self.root / "vec.json"),
                    "sqlite_path": str(self.root / "kb.sqlite3"),
                    "canonical_file": str(self.root / "canonical.jsonl"),
                    "release_calendar": "",
                },
                "retrieval": {"final_top_k": 5, "resolved_min_similarity": 0.1},
            }
        )
        store = PythonVectorStore(self.cfg.resolve(self.cfg.storage.lancedb_path + "_py.json"))
        ingest_docs(self.cfg, [DOC], EmbeddingClient(self.cfg.embedding), store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_readonly_search_works_and_cannot_write(self):
        engine = SearchEngine(self.cfg, read_only=True)
        try:
            results = engine.search("illegal memory access", target_version="0.26.0")
            self.assertTrue(results)
            self.assertEqual(results[0].doc_id, DOC.source_id)
            # 向量库写操作抛错
            with self.assertRaises(ReadOnlyError):
                engine.vector_store.add_items([])
            # SQLite 写操作抛错
            ro_conn = engine._ro_conn()
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    ro_conn.execute(
                        "INSERT INTO docs (source_id, source_type) VALUES ('z','z')"
                    )
            finally:
                ro_conn.close()
        finally:
            engine.close()


class TestApiSourceAudit(unittest.TestCase):
    """结构审计：api.py 无写操作调用、不导入可写模块、SQLite 连接均 mode=ro。"""

    WRITE_CALLS = {
        "open", "write", "write_text", "write_bytes", "unlink", "rename", "mkdir",
        "add_items", "delete_doc", "update_doc_meta", "clear", "pull",
        "subprocess", "Popen", "system", "check_output", "run",
    }
    WRITE_MODULES = {"ingest", "github_pull", "pipeline", "sources"}

    def test_no_write_calls(self):
        source = API_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                self.assertNotIn(name, self.WRITE_CALLS,
                                 f"api.py 第 {node.lineno} 行出现可写调用 {name}")

    def test_no_write_module_imports(self):
        source = API_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    mod = (alias.name or "").split(".")[0]
                    self.assertNotIn(mod, self.WRITE_MODULES,
                                     f"api.py 第 {node.lineno} 行导入了可写模块 {mod}")

    def test_sqlite_connects_use_mode_ro(self):
        source = API_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        found_connect = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "connect":
                found_connect = True
                seg = ast.get_source_segment(source, node) or ""
                self.assertIn("mode=ro", seg, f"api.py 第 {node.lineno} 行 sqlite 连接未用 mode=ro")
        self.assertTrue(found_connect, "api.py 应包含 sqlite 连接（本测试才有意义）")


if __name__ == "__main__":
    unittest.main()
