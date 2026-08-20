"""vllm 主仓代码快照测试（不触网）：repo 隔离、符号检索、grep。"""
import tempfile
import unittest
import zipfile
from pathlib import Path

from vllm_kb.code_index import VersionedCode
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
        }
    )


def make_vllm_zip(code: VersionedCode, version: str) -> None:
    """造一个最小 vllm 主仓 zip（vllm-0.22.1/vllm/utils/network_utils.py）。"""
    zpath = code.zips_dir / f"{version}.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "import socket\n"
        "def make_zmq_socket(ctx, addr, socket_type, bind):\n"
        "    sock = ctx.socket(socket_type)\n"
        "    if bind:\n"
        "        sock.bind(addr)\n"
        "    return sock\n"
        "def get_ip():\n"
        "    return '127.0.0.1'\n"
    )
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(f"vllm-{version}/vllm/utils/network_utils.py", content)
        zf.writestr(f"vllm-{version}/vllm/__init__.py", "__version__='0.22.1'\n")


class TestVllmCodeRepo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.code = VersionedCode(self.cfg, repo="vllm")
        make_vllm_zip(self.code, "0.22.1")
        self.code.ensure_snapshot("0.22.1")
        self.code.build_index_for_version("0.22.1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_repo_isolated_root(self):
        # vllm repo 用独立子目录
        self.assertEqual(self.code.root.name, "vllm")
        self.assertIn("0.22.1", self.code.available_versions)

    def test_symbol_search_vllm(self):
        hits = self.code.search_symbols("make_zmq_socket", "0.22.1")
        self.assertTrue(hits)
        self.assertTrue(any("network_utils.py" in h["file"] for h in hits))
        self.assertTrue(all(h["version"] == "0.22.1" for h in hits))

    def test_read_file_vllm(self):
        text = self.code.read_file("0.22.1", "vllm/utils/network_utils.py")
        self.assertIsNotNone(text)
        self.assertIn("make_zmq_socket", text or "")

    def test_grep_vllm(self):
        hits = self.code.grep("bind", "0.22.1")
        self.assertTrue(hits)
        self.assertTrue(any("network_utils.py" in h["file"] for h in hits))

    def test_vllm_ascend_isolated(self):
        # vllm-ascend 默认 repo 不应看到 vllm 的版本
        va = VersionedCode(self.cfg, repo="vllm-ascend")
        self.assertNotIn("0.22.1", va.available_versions)


if __name__ == "__main__":
    unittest.main()
