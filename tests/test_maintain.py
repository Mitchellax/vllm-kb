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


if __name__ == "__main__":
    unittest.main()