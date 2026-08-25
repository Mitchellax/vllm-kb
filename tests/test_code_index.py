"""版本化代码仓索引测试（不触网）：符号提取、符号检索、grep、文件读取。"""
import shutil
import tempfile
import unittest
from pathlib import Path

from vllm_kb.code_index import CodeIndexError, VersionedCode
from vllm_kb.config import AppConfig


def make_cfg(tmp: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"name": "test", "data_root": "data"},
            "embedding": {"provider": "echo", "dimensions": 64},
            "storage": {
                "vector_backend": "python",
                "lancedb_path": str(tmp / "vec.json"),
                "sqlite_path": str(tmp / "kb.sqlite3"),
                "canonical_file": str(tmp / "canonical.jsonl"),
                "code_root": str(tmp / "code"),
            },
            "code": {"repo": "vllm-project/vllm-ascend", "versions": ["v0.23.0rc1"]},
        }
    )


def make_snapshot(code: VersionedCode, version: str) -> Path:
    """造一个最小源码快照目录（模拟预存）。"""
    snap = code.snapshots_dir / version
    csrc = snap / "csrc" / "mc2" / "dispatch_ffn_combine" / "op_host"
    csrc.mkdir(parents=True)
    (csrc / "dispatch_ffn_combine_tiling.cpp").write_text(
        "constexpr uint32_t ATTR_MAX_OUTPUT_SIZE_INDEX = 1;\n"
        "void tiling() { int max_output_size = 131072; }\n",
        encoding="utf-8",
    )
    py = snap / "vllm_ascend" / "ops" / "fused_moe"
    py.mkdir(parents=True)
    (py / "moe_comm_method.py").write_text(
        "def fused_experts(self):\n"
        "    max_output_size = get_ascend_config().mega_moe_max_tokens\n"
        "    torch.ops._C_ascend.dispatch_ffn_combine(x, max_output_size=max_output_size)\n",
        encoding="utf-8",
    )
    return snap


class TestCodeIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.code = VersionedCode(self.cfg)
        make_snapshot(self.code, "v0.23.0rc1")
        self.code.build_index_for_version("v0.23.0rc1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_available_versions(self):
        self.assertIn("v0.23.0rc1", self.code.available_versions)
        self.assertTrue(self.code.has_version("v0.23.0rc1"))

    def test_symbol_search_exact(self):
        hits = self.code.search_symbols("dispatch_ffn_combine", "v0.23.0rc1")
        self.assertTrue(hits)
        self.assertTrue(all(h["version"] == "v0.23.0rc1" for h in hits))
        files = {h["file"] for h in hits}
        self.assertTrue(any("moe_comm_method.py" in f for f in files))

    def test_symbol_search_version_filter(self):
        # 不存在的版本 -> 空结果（不抛错）
        hits = self.code.search_symbols("dispatch_ffn_combine", "v0.99.0")
        self.assertEqual(hits, [])

    def test_grep(self):
        hits = self.code.grep("max_output_size", "v0.23.0rc1")
        self.assertTrue(hits)
        self.assertTrue(any("moe_comm_method.py" in h["file"] for h in hits))
        self.assertTrue(any("dispatch_ffn_combine_tiling.cpp" in h["file"] for h in hits))

    def test_grep_path_filter(self):
        """path_sub 限定文件路径子串：只返回目标文件命中（不被全仓命中淹没）。"""
        hits = self.code.grep("max_output_size", "v0.23.0rc1", path_sub="moe_comm_method.py")
        self.assertTrue(hits)
        self.assertTrue(all("moe_comm_method.py" in h["file"] for h in hits))
        self.assertFalse(any("tiling.cpp" in h["file"] for h in hits))

    def test_grep_per_version(self):
        """per_version=True：每个版本各自收集，对比版本间差异（定位修复引入版本）。"""
        # 第二个版本新增一行 fill_(-1)（模拟修复引入）
        snap2 = make_snapshot(self.code, "v0.24.0rc1")
        mrv1 = snap2 / "vllm_ascend" / "worker"
        mrv1.mkdir(parents=True, exist_ok=True)
        (mrv1 / "model_runner_v1.py").write_text(
            "self.query_start_loc.gpu.fill_(-1)\n"
            "blk_table.slot_mapping.gpu.fill_(-1)\n",  # 修复特征
            encoding="utf-8",
        )
        hits = self.code.grep("fill_(-1)", path_sub="model_runner_v1.py", per_version=True, limit=5)
        versions = {h["version"] for h in hits}
        self.assertIn("v0.24.0rc1", versions)
        # v0.23.0rc1 快照无该文件 -> 该版本不出现；v0.24.0rc1 出现修复行
        v24 = [h for h in hits if h["version"] == "v0.24.0rc1"]
        self.assertTrue(any("slot_mapping.gpu.fill_(-1)" in h["snippet"] for h in v24))

    def test_read_file(self):
        text = self.code.read_file(
            "v0.23.0rc1",
            "csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp",
        )
        self.assertIsNotNone(text)
        self.assertIn("ATTR_MAX_OUTPUT_SIZE_INDEX", text or "")

    def test_missing_version_raises(self):
        with self.assertRaises(CodeIndexError):
            self.code.ensure_snapshot("v0.99.0")

    def test_missing_file_returns_none(self):
        self.assertIsNone(self.code.read_file("v0.23.0rc1", "nonexistent/file.py"))

    def test_env_symbol_indexed(self):
        # env 名（VLLM_ASCEND_*）应被索引
        snap = make_snapshot(self.code, "v0.24.0rc1")
        (snap / "vllm_ascend" / "ascend_config.py").parent.mkdir(parents=True, exist_ok=True)
        (snap / "vllm_ascend" / "ascend_config.py").write_text(
            'self.enable_fused_mc2 = os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")\n',
            encoding="utf-8",
        )
        self.code.build_index_for_version("v0.24.0rc1")
        hits = self.code.search_symbols("vllm_ascend_enable_fused_mc2", "v0.24.0rc1")
        self.assertTrue(hits)


# Python ast 提取 + 报错字面量索引（kind 语义）
PY_AST_SRC = '''\
"""module docstring: def ghost_fn():  # 字符串里的 def，不应被索引
"""
import logging

logger = logging.getLogger(__name__)


class Foo:
    def bar(self, x):
        return x


async def baz(a,
              b):
    return a + b


def boom():
    raise RuntimeError("boom happened here")
    assert False, "assert boom happened"
    logger.error("log boom happened")
    logging.warning("warn boom happened")
    self.logger.exception("exc boom happened")
    raise TimeoutError("time_out error")
    raise ValueError("timexout error")
'''


class TestPythonAstExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.code = VersionedCode(self.cfg)
        snap = self.code.snapshots_dir / "v0.25.0rc1"
        (snap / "csrc").mkdir(parents=True, exist_ok=True)  # 多子目录，避免 _repo_root 下钻后相对路径失配
        worker = snap / "vllm_ascend" / "worker"
        worker.mkdir(parents=True)
        (worker / "ast_sample.py").write_text(PY_AST_SRC, encoding="utf-8")
        self.code.build_index_for_version("v0.25.0rc1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_ast_indexes_methods_and_async(self):
        # 缩进方法（旧行级正则 ^def 匹配不到的）与 async/多行签名都应命中
        self.assertTrue(self.code.search_symbols("bar", "v0.25.0rc1"))
        self.assertTrue(self.code.search_symbols("baz", "v0.25.0rc1"))

    def test_ast_ignores_string_false_positive(self):
        # 字符串/docstring 里的 "def ghost_fn" 不应被索引
        self.assertEqual(self.code.search_symbols("ghost_fn", "v0.25.0rc1"), [])

    def test_kind_filter(self):
        # "boom" 是函数（kind=def）；报错字面量是长消息，精确搜 "boom" 不命中 msg
        self.assertTrue(self.code.search_symbols("boom", "v0.25.0rc1", kind="def"))
        self.assertEqual(self.code.search_symbols("boom", "v0.25.0rc1", kind="msg"), [])
        # 默认（kind=None）返回全部
        self.assertTrue(self.code.search_symbols("boom", "v0.25.0rc1"))

    def test_ast_fallback_on_syntax_error(self):
        # ast 解析失败的文件退回行级正则，def 仍被索引
        snap = self.code.snapshots_dir / "v0.25.0rc1" / "vllm_ascend" / "ops"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "broken.py").write_text(
            "def fallback_fn():\n    pass\n\ndef broken(:\n", encoding="utf-8"
        )
        self.code.build_index_for_version("v0.25.0rc1")
        self.assertTrue(self.code.search_symbols("fallback_fn", "v0.25.0rc1"))


class TestMessageIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.code = VersionedCode(self.cfg)
        snap = self.code.snapshots_dir / "v0.25.0rc1"
        (snap / "csrc").mkdir(parents=True, exist_ok=True)  # 多子目录，避免 _repo_root 下钻后相对路径失配
        worker = snap / "vllm_ascend" / "worker"
        worker.mkdir(parents=True)
        (worker / "ast_sample.py").write_text(PY_AST_SRC, encoding="utf-8")
        self.code.build_index_for_version("v0.25.0rc1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_search_messages_finds_raise_assert_log(self):
        hits = self.code.search_messages("boom happened", "v0.25.0rc1")
        src = "vllm_ascend/worker/ast_sample.py"
        # raise / assert / logger.error / logging.warning / self.logger.exception
        self.assertTrue(any(h["file"] == src and h["snippet"].startswith("raise RuntimeError") for h in hits))
        self.assertTrue(any("assert False" in h["snippet"] for h in hits))
        self.assertTrue(any("logger.error" in h["snippet"] for h in hits))
        self.assertTrue(any("logging.warning" in h["snippet"] for h in hits))
        self.assertTrue(any("self.logger.exception" in h["snippet"] for h in hits))

    def test_search_messages_wildcard_escape(self):
        # 片段里的 _ 按字面匹配：time_out 只命中 time_out error，不命中 timexout error
        hits = self.code.search_messages("time_out", "v0.25.0rc1")
        snippets = [h["snippet"] for h in hits]
        self.assertTrue(any("time_out error" in s for s in snippets))
        self.assertFalse(any("timexout" in s for s in snippets))

    def test_search_messages_version_filter(self):
        self.assertEqual(self.code.search_messages("boom happened", "v0.99.0"), [])

    def test_msg_not_found_by_partial_symbol_search(self):
        # 符号检索是精确匹配：部分字符串（无通配）不命中长消息；
        # 子串命中由 search_messages（--kind msg）负责
        self.assertEqual(
            self.code.search_symbols("boom happened", "v0.25.0rc1"), []
        )


if __name__ == "__main__":
    unittest.main()
