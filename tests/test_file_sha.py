"""文件哈希快速通道缓存（_sha256 size+mtime 缓存）测试——判重提速不破坏正确性。"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vllm_kb.sources as S
from vllm_kb.sources import _sha256, save_sha_cache


def _write(p: Path, data: bytes) -> None:
    p.write_bytes(data)
    p.chmod(0o644)  # 统一模式，避免 mtime 之外的干扰


class TestShaCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_file = Path(self.tmp.name) / ".file_sha.json"
        patcher = mock.patch.object(S, "_sha_cache_file", return_value=self.cache_file)
        patcher.start()
        self.addCleanup(patcher.stop)
        S._SHA_CACHE = {}
        S._SHA_CACHE_LOADED = False
        self.addCleanup(_reset_cache)

    def tearDown(self):
        _reset_cache()


def _reset_cache():
    S._SHA_CACHE = {}
    S._SHA_CACHE_LOADED = False


class TestShaCacheBehavior(TestShaCache):
    def test_unchanged_file_hits_cache(self):
        """size+mtime 未变 → 第二次调用不读文件（内容被外部换掉也返回旧 sha——缓存语义）。"""
        p = Path(self.tmp.name) / "doc.pdf"
        _write(p, b"A" * 4096)
        s1 = _sha256(p)
        # 外部替换内容（同 size），并把 mtime 拨回原值——模拟"不可靠的触碰"
        old_mtime = p.stat().st_mtime_ns
        _write(p, b"B" * 4096)
        os.utime(p, ns=(old_mtime, old_mtime))
        self.assertEqual(_sha256(p), s1, "size+mtime 未变应命中缓存")
        # mtime 变化 → 缓存失效 → 重算出真实新哈希（+1ms：NTFS mtime 分辨率 100ns）
        os.utime(p, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))
        self.assertNotEqual(_sha256(p), s1)

    def test_roundtrip_and_corrupt_cache(self):
        """持久化 roundtrip：save 后重置内存，新"进程"直接命中；损坏文件忽略重建。"""
        p = Path(self.tmp.name) / "doc.pdf"
        _write(p, b"roundtrip")
        s1 = _sha256(p)
        self.assertTrue(S._SHA_CACHE, "读后内存应有缓存")
        save_sha_cache()
        self.assertTrue(self.cache_file.exists())
        _reset_cache()
        self.assertEqual(_sha256(p), s1, "重置后应从磁盘缓存命中")
        self.cache_file.write_text("{broken", encoding="utf-8")
        _reset_cache()
        self.assertEqual(_sha256(p), s1, "缓存损坏应忽略并重算")
        save_sha_cache()
        self.assertTrue(json.loads(self.cache_file.read_text(encoding="utf-8")),
                        "重算后 save 应重新落盘合法缓存")

    def test_missing_file_raises(self):
        p = Path(self.tmp.name) / "ghost.pdf"
        with self.assertRaises(OSError):
            _sha256(p)


if __name__ == "__main__":
    unittest.main()
