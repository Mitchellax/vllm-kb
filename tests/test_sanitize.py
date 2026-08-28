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

    def test_keep_paths_empty_masks_all(self):
        """显式空列表 = 全部绝对路径脱敏（含默认路径）。"""
        out = sanitize_text("/var/log/npu/x.log 与 /usr/local/Ascend",
                            keep_paths=[], keep_ips=None)
        self.assertNotIn("/var/log", out)
        self.assertNotIn("/usr/local/Ascend", out)
        self.assertIn("<PATH>", out)

    def test_keep_ips_override(self):
        """keep_ips 白名单内保留，其余脱敏。"""
        out = sanitize_text("8.8.8.8 与 10.0.0.5", keep_ips=["8.8.8.8"])
        self.assertIn("8.8.8.8", out)
        self.assertNotIn("10.0.0.5", out)
        self.assertIn("<IP>", out)

    def test_keep_ips_empty_masks_all(self):
        """显式空列表 = 全部 IP 脱敏（含回环）。"""
        out = sanitize_text("127.0.0.1 与 10.0.0.5", keep_ips=[])
        self.assertNotIn("127.0.0.1", out)
        self.assertIn("<IP>", out)

    def test_relative_path_kept(self):
        """相对路径（无盘符、非 / 开头）低风险保留。"""
        self.assertIn("a/b/c", sanitize_text("见 a/b/c 说明"))

    def test_collector_gathers_masked_values(self):
        """collector 收集被脱敏的原始值（白名单内不收集）。"""
        col = {}
        out = sanitize_text("10.0.0.5 与 /home/user/x.log 与 /var/log/npu/ 与 127.0.0.1",
                            collector=col)
        self.assertIn("<IP>", out)
        self.assertIn("<PATH>", out)
        self.assertIn("/var/log/npu/", out)   # 默认路径保留
        self.assertIn("127.0.0.1", out)       # 回环保留
        # 收集：被脱敏的 IP/路径（不含保留项）
        self.assertEqual(col.get("ips"), {"10.0.0.5"})
        self.assertEqual(col.get("paths"), {"/home/user/x.log"})

    def test_save_sanitize_log(self):
        """save_sanitize_log 合并写入维护文件（幂等、累积）。"""
        import json
        import os
        import tempfile
        from pathlib import Path

        from vllm_kb.config import AppConfig
        from vllm_kb.sanitize import save_sanitize_log

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            os.environ["VLLM_KB_DATA_ROOT"] = str(root / "data_root")
            try:
                cfg = AppConfig.model_validate({})
                p = save_sanitize_log(cfg, {"ips": {"10.0.0.5"}, "paths": {"/home/u/a"}})
                data = json.loads(p.read_text(encoding="utf-8"))
                self.assertEqual(data["ips"], ["10.0.0.5"])
                self.assertEqual(data["paths"], ["/home/u/a"])
                # 幂等合并：再次写入新值，旧值保留
                save_sanitize_log(cfg, {"ips": {"192.168.1.1"}, "paths": {"/home/u/a"}})
                data = json.loads(p.read_text(encoding="utf-8"))
                self.assertEqual(set(data["ips"]), {"10.0.0.5", "192.168.1.1"})
                self.assertEqual(data["count"]["paths"], 1)
                self.assertIn("updated_at", data)
            finally:
                os.environ.pop("VLLM_KB_DATA_ROOT", None)

    def test_collect_sanitize_hits(self):
        """collect_sanitize_hits：只收集会被脱敏的值（不替换文本）。"""
        from vllm_kb.sanitize import collect_sanitize_hits

        ips, paths = collect_sanitize_hits(
            "10.0.0.5 与 127.0.0.1 与 /home/user/x.log 与 /var/log/npu/"
        )
        self.assertEqual(ips, {"10.0.0.5"})           # 回环不收集
        self.assertEqual(paths, {"/home/user/x.log"})  # 默认路径不收集
        ips2, paths2 = collect_sanitize_hits("8.8.8.8 /data/a.b", keep_ips=["8.8.8.8"])
        self.assertEqual(ips2, set())                  # keep_ips 白名单不收集
        self.assertEqual(paths2, {"/data/a.b"})

    def test_empty(self):
        self.assertEqual(sanitize_text(""), "")
        self.assertEqual(sanitize_text(None), None)


if __name__ == "__main__":
    unittest.main()
