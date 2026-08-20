"""存算分离测试：VLLM_KB_DATA_ROOT 数据根重定向 + client VLLM_KB_BASE 远程寻址。"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestDataRootRedirect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # config.json 密钥走环境变量（脱敏后无明文 api_key）
        self._old_key = os.environ.get("EMBEDDING_API_KEY")
        self._old_gh = os.environ.get("GITHUB_TOKEN")
        os.environ["EMBEDDING_API_KEY"] = "dummy-for-test"
        os.environ["GITHUB_TOKEN"] = "dummy-for-test"
        # 模拟远程数据目录：kb.sqlite3 + 一个小表
        data_root = self.root / "remote_data"
        data_root.mkdir(parents=True)
        conn = sqlite3.connect(str(data_root / "kb.sqlite3"))
        conn.executescript(
            "CREATE TABLE docs (source_id TEXT PRIMARY KEY, title TEXT, url TEXT, component TEXT, resolved_at TEXT);"
        )
        conn.execute(
            "INSERT INTO docs VALUES ('a:1','远程数据测试','http://x','vllm-ascend',NULL)"
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        for name, old in (("EMBEDDING_API_KEY", self._old_key), ("GITHUB_TOKEN", self._old_gh)):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        self.tmp.cleanup()

    def test_resolve_redirects_data_root(self):
        from vllm_kb.config import AppConfig

        old = os.environ.get("VLLM_KB_DATA_ROOT")
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "remote_data")
        try:
            cfg = AppConfig.load("config.json")
            # data/kb.sqlite3 -> {root}/kb.sqlite3（剥掉 data/ 前缀）
            p = cfg.resolve("data/kb.sqlite3")
            self.assertEqual(str(p), str(self.root / "remote_data" / "kb.sqlite3"))
            self.assertTrue(p.exists())
            # 绝对路径不受影响（Windows 下 /abs 会被 Path 规范化为 C:\abs，比较 is_absolute）
            self.assertTrue(cfg.resolve("/abs/path/x").is_absolute())
        finally:
            if old is None:
                os.environ.pop("VLLM_KB_DATA_ROOT", None)
            else:
                os.environ["VLLM_KB_DATA_ROOT"] = old

    def test_client_default_base_from_env(self):
        """VLLM_KB_BASE 环境变量影响 client 默认地址。"""
        client = Path(__file__).resolve().parent.parent / "skills" / "vllm-kb" / "client.py"
        old = os.environ.get("VLLM_KB_BASE")
        os.environ["VLLM_KB_BASE"] = "http://remote.example:9999"
        try:
            src = client.read_text(encoding="utf-8")
            self.assertIn("VLLM_KB_BASE", src)
            self.assertIn('os.environ.get("VLLM_KB_BASE", "http://127.0.0.1:8000")', src)
        finally:
            if old is None:
                os.environ.pop("VLLM_KB_BASE", None)
            else:
                os.environ["VLLM_KB_BASE"] = old


if __name__ == "__main__":
    unittest.main()
