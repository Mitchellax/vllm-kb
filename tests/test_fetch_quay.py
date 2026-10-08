"""fetch_quay_tags.py 的测试（不触网）：看护/排除策略、分类，以及请求异常路径。"""
import contextlib
import inspect
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_quay_tags as fq  # noqa: E402


class TestManagedFilter(unittest.TestCase):
    def test_formal_versions_managed(self):
        for name in ("v0.18.0", "0.18.0", "v0.11.0", "v0.7.3", "v0.9.1"):
            self.assertTrue(fq.is_managed(name), name)

    def test_rc_versions_managed(self):
        for name in ("v0.18.0-rc1", "0.18.0.rc1", "v0.18.0rc2", "v0.23.0rc1"):
            self.assertTrue(fq.is_managed(name), name)

    def test_platform_variants_managed(self):
        """版本型 tag 的平台变体（-a3/-310p/-openeuler）同样看护。"""
        for name in ("v0.18.0-a3", "v0.18.0-310p-openeuler", "v0.13.0rc2-a3-openeuler"):
            self.assertTrue(fq.is_managed(name), name)

    def test_model_specific_images_managed(self):
        """模型专属镜像（0day 适配）需要看护。"""
        for name in (
            "glm5", "glm5.2-a3-openeuler", "deepseekv4", "deepseekv4-a3",
            "kimi-k3", "kimi-k3-a3-openeuler", "bailing-flash-arm-a3-openeuler",
            "DeepSeekV4-flash-0731", "v0.11.0rc0-a3-deepseek-v3.2-exp",
        ):
            self.assertTrue(fq.is_managed(name), name)

    def test_nightly_and_branch_excluded(self):
        """日构建与仓库分支不看护（内容可能随构建变化，版本匹配会误导）。"""
        for name in (
            "nightly-main", "nightly-main-a3-openeuler", "ntightly",
            "nightly-releases-v0.18.0", "nightly-releases-v0.23.0-a5-openeuler",
            "releases-v0.13.0", "releases-v0.13.0-a3-openeuler",
        ):
            self.assertFalse(fq.is_managed(name), name)

    def test_trunk_and_dev_excluded(self):
        for name in ("latest", "main", "main-310p", "main-a3-openeuler", "develop",
                     "v0.11.0-dev", "v0.11.0-dev-a3-openeuler"):
            self.assertFalse(fq.is_managed(name), name)


class TestExcludeCategory(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(fq.categorize_excluded("nightly-main-a3"), "nightly(日构建)")
        self.assertEqual(fq.categorize_excluded("ntightly"), "nightly(日构建)")
        self.assertEqual(fq.categorize_excluded("releases-v0.13.0"), "branch(releases-*)")
        self.assertEqual(fq.categorize_excluded("nightly-releases-v0.18.0"), "nightly(日构建)")
        self.assertEqual(fq.categorize_excluded("latest"), "主干/最新")
        self.assertEqual(fq.categorize_excluded("main-310p"), "主干/最新")
        self.assertEqual(fq.categorize_excluded("v0.11.0-dev"), "dev构建")


class TestRequestFailurePath(unittest.TestCase):
    """请求异常路径（不触网）。

    这里看护的是一个**真实踩到的 bug**：`fetch_tags` 的 `except requests.RequestException`
    用了 `requests` 却没导入 → 一旦真的发生网络异常（quay 在国内链路上会间歇性 TCP 超时），
    第一次异常就抛 `NameError: name 'requests' is not defined`，于是
    ① 重试逻辑完全没执行（`max_retries` 形同虚设）；
    ② 真正的超时原因被 NameError 盖掉，排查方向被带偏（现场一度被当成"TLS 配置问题"）。
    既有测试只覆盖纯过滤逻辑（无网络、无异常），所以这个 bug 长期没被发现。
    """

    def _session(self, responses):
        """假 session：按顺序返回响应，或抛异常。"""
        calls = []

        class FakeSession:
            def get(self, url, params=None, timeout=None):
                calls.append({"url": url, "params": params, "timeout": timeout})
                item = responses[min(len(calls) - 1, len(responses) - 1)]
                if isinstance(item, Exception):
                    raise item
                return item

        return FakeSession(), calls

    def _resp(self, payload, status=200):
        class R:
            status_code = status

            def json(self):
                return payload

            def raise_for_status(self):
                if status >= 400:
                    raise AssertionError("不应走到 raise_for_status")

        return R()

    def _run(self, responses, **kw):
        session, calls = self._session(responses)
        with mock.patch("vllm_kb.net.get_session", return_value=session):
            with mock.patch("time.sleep"):  # 不真等退避
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    tags = fq.fetch_tags(**kw)
        return tags, calls, buf.getvalue()

    def test_requests_is_imported(self):
        """直接钉住导入：缺 `import requests` 时上面那些 except 分支会抛 NameError。"""
        self.assertTrue(hasattr(fq, "requests"), "fetch_quay_tags 必须导入 requests")

    def test_transient_failure_retries_then_succeeds(self):
        """首次连接超时 → 必须重试并成功，而不是 NameError 崩掉。"""
        err = requests.ConnectTimeout("Connection to quay.io timed out. (connect timeout=60)")
        payload = {"tags": [{"name": "v0.18.0"}], "has_additional": False}
        tags, calls, out = self._run([err, self._resp(payload)])
        self.assertEqual([t["name"] for t in tags], ["v0.18.0"])
        self.assertEqual(len(calls), 2, "第一次失败后必须再试一次")
        self.assertIn("ConnectTimeout", out, "重试日志要带异常类型，便于定位")
        self.assertNotIn("NameError", out)

    def test_exhausted_retries_report_real_cause(self):
        """全部重试失败 → RuntimeError 必须带**真实原因**（不是 NameError）。"""
        err = requests.ConnectTimeout("Connection to quay.io timed out. (connect timeout=60)")
        with self.assertRaises(RuntimeError) as ctx:
            self._run([err], max_retries=2)
        msg = str(ctx.exception)
        self.assertIn("ConnectTimeout", msg)
        self.assertIn("timed out", msg, "原始错误信息不能被吞掉")
        self.assertIn("已重试 2 次", msg)

    def test_proxy_hint_shown_when_proxy_in_effect(self):
        """有系统代理时，失败日志要提示可用 NO_PROXY 绕过（本机实测代理会把 quay TLS 卡死）。"""
        err = requests.ConnectTimeout("boom")
        payload = {"tags": [], "has_additional": False}
        with mock.patch("requests.utils.getproxies",
                        return_value={"http": "http://127.0.0.1:7892",
                                      "https": "http://127.0.0.1:7892"}):
            _, calls, out = self._run([err, self._resp(payload)], max_retries=2)
        self.assertEqual(len(calls), 2)
        self.assertIn("127.0.0.1:7892", out)
        self.assertIn("NO_PROXY=quay.io", out)

    def test_no_proxy_hint_when_no_proxy(self):
        with mock.patch("requests.utils.getproxies", return_value={}):
            with self.assertRaises(RuntimeError) as ctx:
                self._run([requests.ConnectTimeout("boom")], max_retries=0)
        self.assertNotIn("NO_PROXY", str(ctx.exception))

    def test_429_retries_then_succeeds(self):
        payload = {"tags": [{"name": "v0.23.0"}], "has_additional": False}
        tags, calls, _ = self._run([self._resp({}, status=429), self._resp(payload)])
        self.assertEqual(len(tags), 1)
        self.assertEqual(len(calls), 2)

    def test_pagination_follows_has_additional(self):
        p1 = {"tags": [{"name": "a"}], "has_additional": True}
        p2 = {"tags": [{"name": "b"}], "has_additional": False}
        tags, calls, _ = self._run([self._resp(p1), self._resp(p2)])
        self.assertEqual([t["name"] for t in tags], ["a", "b"])
        self.assertEqual([c["params"]["page"] for c in calls], [1, 2])
        self.assertTrue(all(c["params"]["onlyActiveTags"] for c in calls))

    def test_default_budget_is_generous_enough_for_quay(self):
        """默认预算针对 quay 实测抖动：首连可 20s+，超时/重试过小会让部署步骤假失败。"""
        sig = inspect.signature(fq.fetch_tags)
        self.assertGreaterEqual(sig.parameters["timeout"].default, 60)
        self.assertGreaterEqual(sig.parameters["max_retries"].default, 4)


if __name__ == "__main__":
    unittest.main()
