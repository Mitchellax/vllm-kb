"""来源级隔离与退出码语义测试（不触网、不入库）。

背景：全量部署曾因**第二个**来源（vllm）的 GitHub 传输故障而中止，把已经成功入库的
第一个来源（vllm-ascend 15149 条）与排在最后、根本不依赖网络的本地 PDF 来源一起带走。
本文件钉住修复后的契约：

- `run_build` 逐来源隔离：一个来源失败只告警继续，其余来源照常处理；
- 失败详情进 `failed`，`sources_ok` 记录成功来源数；
- `_report_and_exit`：全成功=0、部分失败=默认 0（`--strict` 时 1）、**全部失败=1**。
"""
import io
import contextlib
import sys
import unittest
from unittest import mock

from vllm_kb.config import SourceCfg
from vllm_kb.github_pull import GithubUnreachable
from vllm_kb.pipeline import _report_and_exit, run_build


def _src(sid: str, stype: str = "github") -> SourceCfg:
    return SourceCfg(id=sid, type=stype, enabled=True)


def _ok_stats(docs: int = 3, pulled: int = 1) -> dict:
    return {"docs": docs, "pulled": pulled, "chunks_in_store": 0}


class TestSourceIsolation(unittest.TestCase):
    def _run(self, sources, side_effect):
        """跑 run_build，返回 (grand, 处理顺序, stdout)。"""
        order: list[str] = []

        def fake_process(src, *a, **k):
            order.append(src.id)
            return side_effect(src)

        buf = io.StringIO()
        with mock.patch("vllm_kb.pipeline.build_sources", return_value=sources):
            with mock.patch("vllm_kb.pipeline.process_source", side_effect=fake_process):
                with contextlib.redirect_stdout(buf):
                    grand = run_build(mock.Mock())
        return grand, order, buf.getvalue()

    def test_one_failure_does_not_stop_the_others(self):
        """中间来源失败 → 后面的来源仍被处理（本次事故的核心）。"""
        sources = [_src("vllm-ascend"), _src("vllm"), _src("manuals", "pdf")]

        def side_effect(src):
            if src.id == "vllm":
                raise GithubUnreachable("GraphQL 网络不可达（已重试 6 次）")
            return _ok_stats(docs=10 if src.id == "vllm-ascend" else 5)

        grand, order, out = self._run(sources, side_effect)
        self.assertEqual(order, ["vllm-ascend", "vllm", "manuals"], "失败来源之后的来源必须继续")
        self.assertEqual(grand["sources_ok"], 2)
        self.assertEqual(grand["ingested_docs"], 15, "成功来源的成果要累加")
        self.assertEqual(len(grand["failed"]), 1)
        self.assertEqual(grand["failed"][0]["id"], "vllm")
        self.assertIn("GithubUnreachable", grand["failed"][0]["error"])
        self.assertIn("已降级继续", out)
        self.assertIn("vllm", out)

    def test_local_source_survives_github_failure(self):
        """GitHub 先失败，本地文件来源（不依赖网络）必须照常入库。"""
        sources = [_src("vllm"), _src("manuals", "pdf")]
        grand, order, _ = self._run(
            sources,
            lambda src: (_ for _ in ()).throw(RuntimeError("boom")) if src.type == "github"
            else _ok_stats(docs=5),
        )
        self.assertEqual(order, ["vllm", "manuals"])
        self.assertEqual(grand["sources_ok"], 1)
        self.assertEqual(grand["ingested_docs"], 5)

    def test_not_implemented_is_skipped_not_failed(self):
        """未实现的来源算跳过（既不算成功也不算失败）。"""
        sources = [_src("weird", "html"), _src("manuals", "pdf")]
        grand, _, out = self._run(
            sources,
            lambda src: (_ for _ in ()).throw(NotImplementedError("未实现"))
            if src.type == "html" else _ok_stats(),
        )
        self.assertEqual(grand["failed"], [])
        self.assertEqual(grand["sources_ok"], 1)
        self.assertIn("未实现，跳过", out)

    def test_all_failed_is_reported(self):
        sources = [_src("a"), _src("b")]
        grand, _, _ = self._run(sources, lambda src: (_ for _ in ()).throw(RuntimeError("x")))
        self.assertEqual(grand["sources_ok"], 0)
        self.assertEqual(len(grand["failed"]), 2)

    def test_success_is_unaffected(self):
        sources = [_src("a"), _src("b")]
        grand, _, out = self._run(sources, lambda src: _ok_stats())
        self.assertEqual((grand["sources_ok"], grand["failed"]), (2, []))
        self.assertNotIn("降级", out)


class TestExitCodeSemantics(unittest.TestCase):
    """部署脚本据退出码判断是否要人工介入，语义必须钉死。"""

    def _report(self, grand, strict=False):
        buf = io.StringIO()
        code = 0
        with contextlib.redirect_stdout(buf):
            try:
                _report_and_exit(grand, strict=strict)
            except SystemExit as e:
                code = e.code
        return code, buf.getvalue()

    def test_all_ok_exits_zero_silently(self):
        code, out = self._report({"sources_ok": 2, "failed": []})
        self.assertEqual(code, 0)
        self.assertEqual(out, "")

    def test_partial_failure_exits_zero_with_loud_warning(self):
        """降级完成：数据可用，不该让部署中止（但必须显眼）。"""
        code, out = self._report({"sources_ok": 1, "failed": [{"id": "vllm", "type": "github",
                                                              "error": "GithubUnreachable: x"}]})
        self.assertEqual(code, 0)
        self.assertIn("降级完成", out)
        self.assertIn("❌ vllm (github)", out)

    def test_partial_failure_exits_one_under_strict(self):
        code, out = self._report({"sources_ok": 1, "failed": [{"id": "vllm", "type": "github",
                                                              "error": "e"}]}, strict=True)
        self.assertEqual(code, 1)
        self.assertIn("--strict", out)

    def test_all_failed_exits_one(self):
        """所有来源都失败 = 真正致命（数据源/密钥问题，本轮没有可用产出）。"""
        code, out = self._report({"sources_ok": 0, "failed": [
            {"id": "vllm", "type": "github", "error": "e1"},
            {"id": "manuals", "type": "pdf", "error": "e2"}]})
        self.assertEqual(code, 1)
        self.assertIn("所有来源均失败", out)

    def test_sources_ok_zero_with_no_failure_exits_zero(self):
        """边界：全是"未实现跳过"→ 没有成功也没有失败，不该判致命。"""
        code, _ = self._report({"sources_ok": 0, "failed": []})
        self.assertEqual(code, 0)


class TestStrictFlagWiring(unittest.TestCase):
    def test_cli_strict_flag_reaches_report(self):
        from vllm_kb import pipeline

        seen = {}

        def fake_report(grand, strict):
            seen["strict"] = strict

        with mock.patch.object(sys, "argv", ["build_kb", "--skip-pull", "--strict"]):
            with mock.patch("vllm_kb.pipeline.AppConfig.load", return_value=mock.Mock()):
                with mock.patch("vllm_kb.pipeline.run_build",
                                return_value={"sources_ok": 1, "failed": [], "pulled": 0,
                                              "ingested_docs": 0}):
                    with mock.patch("vllm_kb.pipeline._report_and_exit", side_effect=fake_report):
                        with contextlib.redirect_stdout(io.StringIO()):
                            pipeline.main()
        self.assertTrue(seen.get("strict"), "--strict 必须传到 _report_and_exit")

    def test_cli_default_is_not_strict(self):
        from vllm_kb import pipeline

        seen = {}
        with mock.patch.object(sys, "argv", ["build_kb", "--skip-pull"]):
            with mock.patch("vllm_kb.pipeline.AppConfig.load", return_value=mock.Mock()):
                with mock.patch("vllm_kb.pipeline.run_build",
                                return_value={"sources_ok": 1, "failed": [], "pulled": 0,
                                              "ingested_docs": 0}):
                    with mock.patch("vllm_kb.pipeline._report_and_exit",
                                    side_effect=lambda grand, strict: seen.update(strict=strict)):
                        with contextlib.redirect_stdout(io.StringIO()):
                            pipeline.main()
        self.assertFalse(seen.get("strict"))


if __name__ == "__main__":
    unittest.main()
