"""总日志接口测试：打屏默认、落盘开启（分卷 RotatingFileHandler）、幂等。"""
import io
import logging
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from vllm_kb.config import AppConfig
from vllm_kb.logging_setup import setup_logging


def make_cfg(tmp: Path, **logging_kw) -> AppConfig:
    return AppConfig.model_validate({
        "embedding": {"provider": "echo"},
        "storage": {"sqlite_path": str(tmp / "kb.sqlite3"),
                    "canonical_file": str(tmp / "c.jsonl")},
        "logging": {"file_path": str(tmp / "logs" / "vllm-kb.log"), **logging_kw},
    })


def _reset_loggers():
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()  # 关闭文件句柄，避免 Windows 目录清理失败
        except Exception:
            pass
        root.removeHandler(h)


class TestLoggingSetup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _reset_loggers()

    def tearDown(self):
        _reset_loggers()
        self.tmp.cleanup()

    def test_console_by_default(self):
        cfg = make_cfg(self.root)  # file=False（默认）
        # StreamHandler 构造时绑定当时的 sys.stderr——在捕获上下文内初始化才能抓到打屏输出
        buf = io.StringIO()
        with redirect_stderr(buf):
            setup_logging(cfg)
            root = logging.getLogger()
            handlers = root.handlers
            self.assertTrue(any(isinstance(h, logging.StreamHandler)
                                and not isinstance(h, logging.FileHandler) for h in handlers))
            self.assertFalse(any(isinstance(h, logging.FileHandler) for h in handlers))
            logging.getLogger("vllm-kb").info("hello console")
        self.assertIn("hello console", buf.getvalue())

    def test_file_enabled_rotating(self):
        cfg = make_cfg(self.root, file=True, max_bytes=1024, backup_count=3)
        setup_logging(cfg)
        root = logging.getLogger()
        fhs = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(fhs), 1)
        fh = fhs[0]
        self.assertEqual(fh.maxBytes, 1024)
        self.assertEqual(fh.backupCount, 3)
        # 写日志 → 文件生成
        logging.getLogger("vllm-kb").info("write to file test")
        for h in root.handlers:
            h.flush()
        log_file = Path(cfg.logging.file_path)
        self.assertTrue(log_file.exists())
        self.assertIn("write to file test", log_file.read_text(encoding="utf-8", errors="replace"))

    def test_idempotent(self):
        cfg = make_cfg(self.root, file=True)
        setup_logging(cfg)
        setup_logging(cfg)  # 重复调用不重复添加 handler
        root = logging.getLogger()
        fhs = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(fhs), 1)

    def test_uvicorn_propagates(self):
        cfg = make_cfg(self.root, file=True)
        setup_logging(cfg)
        # uvicorn logger 传播到 root（随总日志落盘），自身无独立 handler
        u = logging.getLogger("uvicorn.access")
        self.assertEqual(u.handlers, [])
        self.assertTrue(u.propagate)


if __name__ == "__main__":
    unittest.main()
