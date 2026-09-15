"""maintain.py 测试：命令行解析 + 环境变量继承 + 自动降级逻辑（不触网，不调子进程）。"""
import os
import sys
import unittest
from unittest import mock

from scripts.maintain import _insecure_env_from_args, _resolve, _run_step


class FakeArgs:
    insecure = False
    github_base = None
    quay_base = None


class TestMaintainUtils(unittest.TestCase):
    def test_insecure_env_empty(self):
        env = _insecure_env_from_args(FakeArgs())
        self.assertEqual(env, {})

    def test_insecure_env_sets_vars(self):
        args = FakeArgs()
        args.insecure = True
        args.github_base = "http://gh-mirror:8080"
        args.quay_base = "http://quay-mirror:8080"
        env = _insecure_env_from_args(args)
        self.assertEqual(env.get("VLLM_KB_INSECURE"), "1")
        self.assertEqual(env.get("VLLM_KB_GITHUB_BASE"), "http://gh-mirror:8080")
        self.assertEqual(env.get("VLLM_KB_QUAY_BASE"), "http://quay-mirror:8080")

    def test_insecure_env_partial(self):
        args = FakeArgs()
        args.insecure = True
        args.github_base = None
        args.quay_base = "http://q-mirror"
        env = _insecure_env_from_args(args)
        self.assertEqual(env.get("VLLM_KB_INSECURE"), "1")
        self.assertNotIn("VLLM_KB_GITHUB_BASE", env)
        self.assertEqual(env.get("VLLM_KB_QUAY_BASE"), "http://q-mirror")

    def test_resolve_returns_absolute_path(self):
        path = _resolve("scripts/maintain.py")
        self.assertTrue(path.endswith("scripts/maintain.py") or path.endswith("scripts\\maintain.py"))
        self.assertTrue(os.path.isabs(path))


class TestRunStep(unittest.TestCase):
    """自动降级核心逻辑：非致命步骤失败只告警不中断，致命步骤失败才中止。"""

    def _call(self, returncode=0, fatal=False, exc=None):
        with mock.patch("scripts.maintain.subprocess.run") as m:
            if exc:
                m.side_effect = exc
            else:
                m.return_value.returncode = returncode
            return _run_step("scripts/build_kb.py", [], fatal, "测试步骤", {}, None)

    def test_success(self):
        self.assertTrue(self._call(returncode=0))

    def test_nonfatal_failure_degrades(self):
        """非致命步骤失败 → 返回 True（告警继续），不中断。"""
        self.assertTrue(self._call(returncode=2, fatal=False))

    def test_fatal_failure_stops(self):
        """致命步骤失败 → 返回 False（中止流程）。"""
        self.assertFalse(self._call(returncode=1, fatal=True))

    def test_timeout_nonfatal_degrades(self):
        self.assertTrue(self._call(fatal=False, exc=TimeoutError("timeout")))

    def test_timeout_fatal_stops(self):
        self.assertFalse(self._call(fatal=True, exc=TimeoutError("timeout")))

    def test_file_not_found_nonfatal_degrades(self):
        self.assertTrue(self._call(fatal=False, exc=FileNotFoundError("no such file")))

    def test_insecure_env_passed_to_subprocess(self):
        """子进程必须继承 VLLM_KB_INSECURE 等环境变量。"""
        with mock.patch("scripts.maintain.subprocess.run") as m:
            m.return_value.returncode = 0
            _run_step("scripts/build_kb.py", [], True, "测试", {"VLLM_KB_INSECURE": "1"}, None)
            _, kwargs = m.call_args
            self.assertEqual(kwargs["env"]["VLLM_KB_INSECURE"], "1")
            # 基座环境变量保留
            self.assertIn("PATH", kwargs["env"])


if __name__ == "__main__":
    unittest.main()