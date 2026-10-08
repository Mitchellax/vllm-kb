"""向量后端选择与回退告警测试（不触网、不入库）。

背景：Windows 上 `import lancedb` 会因**安全策略拦截原生 DLL** 而抛 ImportError
（`DLL load failed while importing lib: 应用程序控制策略已阻止此文件。`）。
原实现把一切 ImportError 都说成"未安装 lancedb"，于是：
1. 排查方向被引向 `pip install`（无用，包本来就在）；
2. 更危险的是**换了库**——回退后端写 `data/lancedb_py.json`，而正式库是 `data/lancedb`，
   既有向量在本次检索里全部"消失"，却只有一句轻飘飘的 warn。

本文件钉住：如实区分两种原因、说明换库影响、后端标识可见、告警只打一次。
"""
import io
import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vllm_kb.config import AppConfig
from vllm_kb.vectorstore import (
    BaseVectorStore,
    LanceDBVectorStore,
    PythonVectorStore,
    ReadOnlyVectorStore,
    build_vector_store,
)


def _cfg(root: Path, backend: str = "lancedb") -> AppConfig:
    return AppConfig.model_validate({
        "storage": {
            "vector_backend": backend,
            "lancedb_path": str(root / "lancedb"),
            "sqlite_path": str(root / "kb.sqlite3"),
            "canonical_file": str(root / "canonical.jsonl"),
            "release_calendar": "",
        },
    })


class _FakeImportError(ImportError):
    pass


class TestFallbackDiagnostics(unittest.TestCase):
    def setUp(self):
        # 告警"每次进程只打一次"是模块级状态，测试间必须复位
        import vllm_kb.vectorstore as vs

        self._vs = vs
        self._saved = vs._fallback_warned
        vs._fallback_warned = False
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self._vs._fallback_warned = self._saved
        self.tmp.cleanup()

    def _build_capture(self, *, installed: bool, error: ImportError):
        """让 LanceDB 构造抛指定 ImportError，捕获 stdout 与返回的 store。"""
        buf = io.StringIO()
        with mock.patch.object(self._vs, "LanceDBVectorStore", side_effect=error):
            with mock.patch("importlib.util.find_spec",
                            return_value=object() if installed else None):
                with contextlib.redirect_stdout(buf):
                    store = build_vector_store(_cfg(self.root))
        return store, buf.getvalue()

    def test_installed_but_broken_is_not_reported_as_missing(self):
        """装了但加载不了 → 必须说"已安装但无法加载"，并把真实异常打出来。"""
        err = _FakeImportError(
            "DLL load failed while importing lib: 应用程序控制策略已阻止此文件。")
        store, out = self._build_capture(installed=True, error=err)
        self.assertIsInstance(store, PythonVectorStore)
        self.assertIn("已安装但无法加载", out)
        self.assertIn("应用程序控制策略已阻止此文件", out)
        self.assertIn("Smart App Control", out)
        self.assertNotIn("未安装 lancedb（pip install", out, "不能谎报未安装")

    def test_truly_missing_says_not_installed(self):
        store, out = self._build_capture(installed=False,
                                         error=_FakeImportError("No module named 'lancedb'"))
        self.assertIsInstance(store, PythonVectorStore)
        self.assertIn("未安装 lancedb", out)
        self.assertIn("pip install lancedb", out)

    def test_names_the_two_different_stores(self):
        """回退换的是**另一个库**：两个路径都要打出来。"""
        store, out = self._build_capture(installed=True, error=_FakeImportError("boom"))
        self.assertIn("lancedb_py.json", out)
        self.assertIn(str(self.root / "lancedb"), out)
        self.assertIn("不是同一个库", out)

    def test_warns_when_formal_store_has_data(self):
        """正式库非空时，必须警告"既有向量看不到"并给出两条出路。"""
        formal = self.root / "lancedb"
        formal.mkdir(parents=True)
        (formal / "chunks.lance").write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
        _, out = self._build_capture(installed=True, error=_FakeImportError("boom"))
        self.assertIn("已有 2 MB 数据", out)
        self.assertIn("看不到", out)
        self.assertIn("--rebuild", out)

    def test_no_extra_warning_when_formal_store_empty(self):
        _, out = self._build_capture(installed=True, error=_FakeImportError("boom"))
        self.assertNotIn("已有", out)

    def test_warns_only_once_per_process(self):
        """多来源时 build_vector_store 会被调用多次，告警不该刷屏。"""
        err = _FakeImportError("boom")
        with mock.patch.object(self._vs, "LanceDBVectorStore", side_effect=err):
            with mock.patch("importlib.util.find_spec", return_value=object()):
                buf1 = io.StringIO()
                with contextlib.redirect_stdout(buf1):
                    build_vector_store(_cfg(self.root))
                buf2 = io.StringIO()
                with contextlib.redirect_stdout(buf2):
                    build_vector_store(_cfg(self.root))
        self.assertIn("降级", buf1.getvalue())
        self.assertEqual(buf2.getvalue(), "", "第二次不该重复告警")

    def test_explicit_python_backend_is_silent(self):
        """显式选 python 后端是知情选择，不该告警。"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            store = build_vector_store(_cfg(self.root, backend="python"))
        self.assertIsInstance(store, PythonVectorStore)
        self.assertEqual(buf.getvalue(), "")

    def test_explicit_python_backend_does_not_warn_even_if_lancedb_broken(self):
        buf = io.StringIO()
        with mock.patch.object(self._vs, "LanceDBVectorStore",
                               side_effect=_FakeImportError("boom")):
            with contextlib.redirect_stdout(buf):
                build_vector_store(_cfg(self.root, backend="python"))
        self.assertEqual(buf.getvalue(), "")


class TestBackendIdentity(unittest.TestCase):
    """后端标识必须能被上层看到（汇总/日志用），否则回退是"隐形"的。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_python_store_identity(self):
        store = PythonVectorStore(self.root / "v.json")
        self.assertEqual(store.backend_name, "python")
        self.assertEqual(store.location, str(self.root / "v.json"))

    def test_lancedb_store_identity(self):
        self.assertEqual(LanceDBVectorStore.backend_name, "lancedb")

    def test_readonly_wrapper_forwards_identity(self):
        store = PythonVectorStore(self.root / "v.json")
        ro = ReadOnlyVectorStore(store)
        self.assertEqual(ro.backend_name, "python")
        self.assertEqual(ro.location, store.location)
        self.assertIsInstance(ro, BaseVectorStore)

    def test_default_identity_is_unknown_not_crash(self):
        """兜底：没标 backend_name 的 store 也不能让汇总炸掉。"""

        class Bare(BaseVectorStore):
            def add_items(self, items): ...
            def delete_doc(self, doc_id): ...
            def update_doc_meta(self, doc_id, meta): ...
            def search(self, vector, top_k=10): return []
            def count(self): return 0
            def clear(self): ...

        b = Bare()
        self.assertEqual(getattr(b, "backend_name", "?"), "?")
        self.assertEqual(ReadOnlyVectorStore(b).backend_name, "?")


class TestBackendReachesBuildSummary(unittest.TestCase):
    """回退后端必须出现在 build 汇总里（跑完能从日志看出向量写到哪个库）。"""

    def test_process_source_reports_backend(self):
        import tempfile
        from unittest import mock

        from vllm_kb import pipeline

        with tempfile.TemporaryDirectory() as td:
            cfg = _cfg(Path(td))
            src = mock.Mock()
            src.id, src.type, src.description = "s1", "pdf", ""
            src.canonicalize.return_value = [mock.Mock()]
            with mock.patch("vllm_kb.pipeline.upsert_unified_canonical"):
                with mock.patch("vllm_kb.pipeline.EmbeddingClient"):
                    with mock.patch("vllm_kb.pipeline.build_vector_store") as bvs:
                        store = PythonVectorStore(Path(td) / "v.json")
                        bvs.return_value = store
                        with mock.patch("vllm_kb.pipeline.ingest_docs",
                                        return_value={"docs": 1}):
                            with contextlib.redirect_stdout(io.StringIO()):
                                stats = pipeline.process_source(src, cfg, pull=False,
                                                                limit=None, incremental=False)
            self.assertEqual(stats["vector_backend"], "python")

    def test_run_build_collects_backends(self):
        from unittest import mock

        from vllm_kb import pipeline

        srcs = [mock.Mock(id="a", type="pdf"), mock.Mock(id="b", type="pdf")]
        with mock.patch("vllm_kb.pipeline.build_sources", return_value=srcs):
            with mock.patch("vllm_kb.pipeline.process_source",
                            return_value={"docs": 1, "pulled": 0,
                                          "vector_backend": "python"}):
                with contextlib.redirect_stdout(io.StringIO()):
                    grand = pipeline.run_build(mock.Mock())
        self.assertEqual(grand["vector_backends"], ["python"])


if __name__ == "__main__":
    unittest.main()
