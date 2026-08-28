"""sanitize_text 脱敏测试：IP/路径脱敏、默认路径保留、回环保留。"""
import unittest

from vllm_kb.sanitize import DEFAULT_KEEP_PATHS, sanitize_text


class TestSanitize(unittest.TestCase):
    def test_internal_ip_masked(self):
        self.assertIn("<IP>", sanitize_text("节点 10.0.0.5 超时"))
        self.assertNotIn("10.0.0.5", sanitize_text("节点 10.0.0.5 超时"))
        # 172.16-31 / 192.168 内网段同样脱敏
        self.assertIn("<IP>", sanitize_text("172.16.3.2"))
        self.assertIn("<IP>", sanitize_text("192.168.1.100"))
        # 公网 IP 也脱敏
        self.assertIn("<IP>", sanitize_text("8.8.8.8"))

    def test_ip_port_kept(self):
        """端口保留（诊断价值）。"""
        out = sanitize_text("服务 10.0.0.5:8000 不可达")
        self.assertIn("<IP>:8000", out)

    def test_loopback_kept(self):
        self.assertIn("127.0.0.1", sanitize_text("127.0.0.1 正常"))
        self.assertIn("0.0.0.0", sanitize_text("监听 0.0.0.0"))

    def test_internal_path_masked(self):
        self.assertIn("<PATH>", sanitize_text("/home/user/logs/a.log"))
        self.assertNotIn("/home/user", sanitize_text("/home/user/logs/a.log"))
        self.assertIn("<PATH>", sanitize_text("D:\\workspace\\proj\\data.bin"))
        self.assertIn("<PATH>", sanitize_text("/data/vllm-kb/kb.sqlite3"))

    def test_default_paths_kept(self):
        """默认路径/日志路径白名单保留（诊断价值）。"""
        for p in ("/usr/local/Ascend/ascend-toolkit/set_env.sh",
                  "/var/log/npu/slog/device-0.log",
                  "/etc/ascend/ascend_install.info"):
            self.assertEqual(sanitize_text(p), p, p)
        self.assertTrue(DEFAULT_KEEP_PATHS)

    def test_keep_paths_override(self):
        out = sanitize_text("/opt/internal/x", keep_paths=["/opt/internal"])
        self.assertEqual(out, "/opt/internal/x")

    def test_relative_path_kept(self):
        """相对路径（无盘符、非 / 开头）低风险保留。"""
        self.assertIn("a/b/c", sanitize_text("见 a/b/c 说明"))

    def test_empty(self):
        self.assertEqual(sanitize_text(""), "")
        self.assertEqual(sanitize_text(None), None)


if __name__ == "__main__":
    unittest.main()
