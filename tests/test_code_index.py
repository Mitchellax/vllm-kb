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


if __name__ == "__main__":
    unittest.main()
