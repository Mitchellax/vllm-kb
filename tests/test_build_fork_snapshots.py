"""build_fork_snapshots.py 测试（不触网）：fork 行提取、模型名安全、SHA 下载、索引/元信息。"""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_fork_snapshots as bfs  # noqa: E402

from vllm_kb.config import AppConfig  # noqa: E402


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
                "companion_file": str(tmp / "vllm-ascend.json"),
            },
        }
    )


def write_matrix(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")


SHA = "9ab939da68de3acd6acd40365d4e1bc25ae15d79"


class TestForkRows(unittest.TestCase):
    def test_fork_rows_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = make_cfg(tmp)
            write_matrix(cfg.resolve(cfg.storage.companion_file), [
                {"vllm-ascend": "v0.23.0", "vllm": "0.23.0"},                 # 非 fork
                {"vllm-ascend": "hy4", "vllm_repo": "a/vllm", "vllm_sha": SHA},  # fork 完整
                {"vllm-ascend": "glm5.2", "vllm_repo": "b/vllm", "vllm_sha": ""},  # SHA 未回填
            ])
            rows = bfs.fork_rows(cfg)
            self.assertEqual([r["vllm-ascend"] for r in rows], ["hy4"])

    def test_no_matrix(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = make_cfg(Path(td))
            self.assertEqual(bfs.fork_rows(cfg), [])

    def test_model_name_safety(self):
        self.assertEqual(bfs.model_name("hy4"), "hy4")
        self.assertEqual(bfs.model_name("glm5.2"), "glm5.2")
        self.assertEqual(bfs.model_name("DeepSeekV4-flash-0731"), "DeepSeekV4-flash-0731")
        # 路径穿越/非法字符拒绝
        self.assertEqual(bfs.model_name("../evil"), "")
        self.assertEqual(bfs.model_name("a/b"), "")
        self.assertEqual(bfs.model_name("a\\b"), "")
        self.assertEqual(bfs.model_name(""), "")
        self.assertEqual(bfs.model_name("a b"), "")


class TestDownload(unittest.TestCase):
    def test_download_url_by_sha_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "zips" / f"{SHA[:12]}.zip"

            class _Resp:
                def __init__(self):
                    self._chunks = [b"PK-zip-bytes" * 100]

                def read(self, n=-1):
                    return self._chunks.pop(0) if self._chunks else b""

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

            resp = _Resp()
            with mock.patch("build_fork_snapshots._opener") as mo:
                mo.return_value.open.return_value = resp
                ok = bfs.download("voidvelocity/vllm", SHA, dest)
            self.assertTrue(ok)
            req = mo.return_value.open.call_args[0][0]
            # URL 形态：{base}/{repo}/zip/{sha}——按锁定 SHA，非 tag
            self.assertEqual(req.full_url,
                             f"https://codeload.github.com/voidvelocity/vllm/zip/{SHA}")
            self.assertGreater(dest.stat().st_size, 1000)
            # 幂等：已存在且非空 -> 跳过（不再触网）
            with mock.patch("build_fork_snapshots._opener") as mo:
                ok2 = bfs.download("voidvelocity/vllm", SHA, dest)
                mo.return_value.open.assert_not_called()
            self.assertFalse(ok2)

    def test_download_failure_cleans_up(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "zips" / f"{SHA[:12]}.zip"
            with mock.patch("build_fork_snapshots._opener") as mo:
                mo.return_value.open.side_effect = OSError("net down")
                ok = bfs.download("a/vllm", SHA, dest)
            self.assertFalse(ok)
            self.assertFalse(dest.exists())  # 失败清理残片


class TestIndexAndMeta(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "forks" / "hy4"
        # 造最小 fork zip（codeload 形态：{repo}-{sha}/vllm/...）
        zpath = self.root / "zips" / f"{SHA[:12]}.zip"
        zpath.parent.mkdir(parents=True, exist_ok=True)
        content = "def hy4_custom_kernel(x):\n    return x\n"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr(f"vllm-{SHA[:12]}/vllm/hy4_ops.py", content)
            zf.writestr(f"vllm-{SHA[:12]}/README.md", "fork")

    def tearDown(self):
        self.tmp.cleanup()

    def test_ensure_snapshot_and_index(self):
        snap = bfs.ensure_snapshot(self.root, SHA[:12])
        self.assertTrue((snap / f"vllm-{SHA[:12]}" / "vllm" / "hy4_ops.py").exists())
        n = bfs.build_index(self.root, SHA[:12])
        self.assertGreaterEqual(n, 1)
        import sqlite3

        conn = sqlite3.connect(str(self.root / "index.sqlite3"))
        rows = conn.execute(
            "SELECT version, file FROM symbols WHERE symbol = ?",
            ("hy4_custom_kernel",)).fetchall()
        conn.close()
        self.assertEqual(rows, [(SHA[:12], "vllm/hy4_ops.py")])

    def test_write_meta(self):
        row = {"vllm-ascend": "hy4", "vllm_repo": "voidvelocity/vllm",
               "vllm_ref": "dev_hy4", "vllm_base": "0.23.0",
               "vllm_sha": SHA, "image_digest": "sha256:x"}
        bfs.write_meta(self.root, row)
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["repo"], "voidvelocity/vllm")
        self.assertEqual(meta["sha"], SHA)
        self.assertEqual(meta["base"], "0.23.0")


if __name__ == "__main__":
    unittest.main()
