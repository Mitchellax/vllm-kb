"""maintain.py 测试：命令行解析 + 环境变量继承 + 自动降级逻辑（不触网，不调子进程）。

自动降级两个真实场景的验证：
- jieba 未安装：build_fts.py 依赖的 fts_tokenizer 降级为原文（不崩溃，脚本正常退出 0）；
- kuzu 未安装：build_graph.py 顶层 `import kuzu` 失败 → 脚本非零退出 →
  maintain 的 _run_step 按非致命（fatal=False）告警继续，部署不中断。
"""
import importlib
import os
import sys
import unittest
from unittest import mock

from scripts.maintain import (
    _DEPLOY_STEPS,
    _UPDATE_STEPS,
    _filter_steps,
    _insecure_env_from_args,
    _resolve,
    _run_step,
    cmd_deploy,
    main,
)


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


class TestStepTable(unittest.TestCase):
    """步骤表降级设计：build_kb 致命，其余全部非致命（kuzu/jieba/网络缺失均可降级）。"""

    def test_deploy_build_kb_is_fatal(self):
        fatal_steps = [tag for _s, _x, f, tag in _DEPLOY_STEPS if f]
        self.assertEqual(fatal_steps, ["数据拉取与入库"])

    def test_deploy_graph_fts_calendar_matrix_nonfatal(self):
        """建图（需 kuzu）/建FTS（需 jieba）/日历/矩阵失败均不中断部署。"""
        tags = [tag for _s, _x, f, tag in _DEPLOY_STEPS if not f]
        for expect in ("图构建（Kùzu）", "全文索引重建（FTS5）", "版本日历", "配套矩阵"):
            self.assertIn(expect, tags, f"{expect} 应标记为非致命步骤")

    def test_update_build_kb_is_fatal(self):
        fatal_steps = [tag for _s, _x, f, tag in _UPDATE_STEPS if f]
        self.assertEqual(fatal_steps, ["增量拉取与入库"])

    def test_update_graph_nonfatal(self):
        tags = [tag for _s, _x, f, tag in _UPDATE_STEPS if not f]
        self.assertIn("图重建（Kùzu）", tags)

    def test_filter_steps_removes_matching(self):
        """_filter_steps 按 key 正确过滤步骤。"""
        filtered = _filter_steps(_DEPLOY_STEPS, {"graph", "calendar"})
        remaining = [tag for _s, _x, f, tag in filtered]
        self.assertNotIn("图构建（Kùzu）", remaining)
        self.assertNotIn("版本日历", remaining)
        self.assertIn("数据拉取与入库", remaining)
        self.assertIn("全文索引重建（FTS5）", remaining)
        self.assertIn("配套矩阵", remaining)

    def test_filter_steps_empty_skip_preserves_all(self):
        filtered = _filter_steps(_DEPLOY_STEPS, set())
        self.assertEqual(len(filtered), len(_DEPLOY_STEPS))


class TestJiebaDegradation(unittest.TestCase):
    """jieba 未安装：FTS 分词降级原文，不崩溃（build_fts.py 依赖此路径）。"""

    def test_tokenize_text_returns_original_without_jieba(self):
        from vllm_kb import fts_tokenizer

        orig_cache = fts_tokenizer._jieba
        try:
            with mock.patch.dict(sys.modules, {"jieba": None}):  # import jieba → ImportError
                fts_tokenizer._jieba = None  # 重置懒加载缓存
                self.assertEqual(fts_tokenizer.tokenize_text("超时排查"), "超时排查")
        finally:
            fts_tokenizer._jieba = orig_cache

    def test_register_words_noop_without_jieba(self):
        from vllm_kb import fts_tokenizer

        orig_cache = fts_tokenizer._jieba
        try:
            with mock.patch.dict(sys.modules, {"jieba": None}):
                fts_tokenizer._jieba = None
                fts_tokenizer.register_words(["超时排查", "HCCL"])  # 不抛异常
        finally:
            fts_tokenizer._jieba = orig_cache

    def test_query_tokens_fallback_without_jieba(self):
        from vllm_kb import fts_tokenizer

        orig_cache = fts_tokenizer._jieba
        try:
            with mock.patch.dict(sys.modules, {"jieba": None}):
                fts_tokenizer._jieba = None
                # 降级为中英 token 提取（原文含中文→整段一个 token）
                self.assertEqual(fts_tokenizer.query_tokens("超时排查"), ["超时排查"])
        finally:
            fts_tokenizer._jieba = orig_cache


class TestKuzuDegradation(unittest.TestCase):
    """kuzu 未安装：build_graph.py 顶层 import kuzu 失败 → 脚本非零退出（实测）。"""

    def test_graph_import_fails_without_kuzu(self):
        """模拟 kuzu 缺失：import vllm_kb.graph 必须抛 ImportError（build_graph.py 将非零退出）。"""
        saved = sys.modules.pop("vllm_kb.graph", None)
        try:
            with mock.patch.dict(sys.modules, {"kuzu": None}):
                with self.assertRaises(ImportError):
                    importlib.import_module("vllm_kb.graph")
        finally:
            if saved is not None:
                sys.modules["vllm_kb.graph"] = saved
            elif "vllm_kb.graph" in sys.modules:
                del sys.modules["vllm_kb.graph"]

    def test_graph_module_imports_cleanly_with_kuzu(self):
        """kuzu 可用时 vllm_kb.graph 正常导入（对照基线）。"""
        import vllm_kb.graph  # noqa: F401

        self.assertTrue(hasattr(vllm_kb.graph, "GraphBuilder"))


class TestArgparsePositions(unittest.TestCase):
    """公共参数位置灵活：--insecure/--github-base/--quay-base/--config 可在子命令前后。"""

    @staticmethod
    def _parse(argv):
        """构造与 main() 相同的 parser 并解析（不执行 cmd_*）。"""
        import argparse
        from scripts import maintain as M

        ap = argparse.ArgumentParser()
        M._add_common_args(ap)
        sub = ap.add_subparsers(dest="command", required=True)
        p = sub.add_parser("deploy")
        M._add_common_args(p, suppress_defaults=True)
        p.add_argument("--all-code", action="store_true")
        p.set_defaults(func=lambda a: None)
        return ap.parse_args(argv)

    def test_insecure_after_subcommand(self):
        args = self._parse(["deploy", "--insecure"])
        self.assertTrue(args.insecure)

    def test_insecure_before_subcommand(self):
        args = self._parse(["--insecure", "deploy"])
        self.assertTrue(args.insecure)

    def test_both_positions_equal(self):
        a = self._parse(["deploy", "--insecure", "--all-code"])
        b = self._parse(["--insecure", "deploy", "--all-code"])
        self.assertEqual(a.insecure, b.insecure)
        self.assertEqual(a.all_code, b.all_code)

    def test_github_base_after_subcommand(self):
        args = self._parse(["deploy", "--github-base", "http://gh:8080"])
        self.assertEqual(args.github_base, "http://gh:8080")

    def test_github_base_before_subcommand(self):
        args = self._parse(["--github-base", "http://gh:8080", "deploy"])
        self.assertEqual(args.github_base, "http://gh:8080")

    def test_config_before_subcommand(self):
        args = self._parse(["--config", "cfg.json", "deploy"])
        self.assertEqual(args.config, "cfg.json")

    def test_no_common_args_defaults(self):
        args = self._parse(["deploy"])
        self.assertFalse(args.insecure)
        self.assertIsNone(args.config)
        self.assertIsNone(args.github_base)
        self.assertIsNone(args.quay_base)


class TestAllCodeFlag(unittest.TestCase):
    """--all-code 触发 build_code_snapshots --all 而非默认 config.code.versions。"""

    @staticmethod
    def _make_args(all_code=False, skip_code_snapshots=False):
        import types
        return types.SimpleNamespace(
            skip={"code_snapshots"} if skip_code_snapshots else set(),
            config=None, insecure=False, github_base=None, quay_base=None,
            skip_code_snapshots=skip_code_snapshots, all_code=all_code,
            skip_graph=False, skip_fts=False, skip_calendar=False, skip_matrix=False,
        )

    def _find_code_step_calls(self, args):
        with mock.patch("scripts.maintain.subprocess.run") as m:
            m.return_value.returncode = 0
            with mock.patch("builtins.print"):
                cmd_deploy(args)
            calls = []
            for call_args, kwargs in m.call_args_list:
                cmd = call_args[0] if call_args else kwargs.get("args", [])
                if any("build_code_snapshots" in str(part) for part in cmd):
                    calls.append(cmd)
            return calls

    def test_deploy_all_code_passes_all_flag(self):
        args = self._make_args(all_code=True)
        calls = self._find_code_step_calls(args)
        self.assertEqual(len(calls), 1, "build_code_snapshots 应被调用一次")
        self.assertIn("--all", [str(a) for a in calls[0]])

    def test_deploy_default_no_all_flag(self):
        args = self._make_args(all_code=False)
        calls = self._find_code_step_calls(args)
        self.assertEqual(len(calls), 1, "build_code_snapshots 应被调用一次")
        self.assertNotIn("--all", [str(a) for a in calls[0]],
                         "默认 deploy 不应传递 --all")

    def test_skip_code_snapshots_overrides_all_code(self):
        args = self._make_args(all_code=True, skip_code_snapshots=True)
        calls = self._find_code_step_calls(args)
        self.assertEqual(len(calls), 0, "--skip-code-snapshots 时应不调用 build_code_snapshots")


if __name__ == "__main__":
    unittest.main()