"""vllm_kb.net 单元测试：get_session 的 insecure SSL 跳过（不触网）。

回归：requests 的 merge_environment_settings 在请求级 verify 为 None 时会用
环境变量 REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE 覆盖 session.verify=False——
insecure 因此失效（真实业务环境常设这两个变量过 MITM，实测 requests 2.34 复现）。
修复为请求级 verify=False hook，本测试锁定该行为。
"""
import os
import ssl
import sys
import unittest
import urllib.request
from unittest import mock

from vllm_kb.net import get_session, insecure_from_env


def _make_resp():
    import json as _json

    import requests

    r = requests.Response()
    r.status_code = 200
    r.url = "https://quay.io/v2/auth"
    r._content = _json.dumps({}).encode()
    return r


class _FakeAdapter:
    """捕获 Session.send 传入的 verify 设置（不发真请求）。"""

    def __init__(self):
        self.captured = {}

    def send(self, request, **kwargs):
        self.captured.update(kwargs)
        return _make_resp()

    def close(self):
        pass


class TestGetSessionInsecure(unittest.TestCase):
    def setUp(self):
        # 模拟真实业务环境：REQUESTS_CA_BUNDLE 指向陌生 CA（MITM 修复常用配置）
        self._saved = {k: os.environ.get(k) for k in
                       ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")}
        os.environ["REQUESTS_CA_BUNDLE"] = "Z:/fake/ca.pem"
        os.environ.pop("CURL_CA_BUNDLE", None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _capture_verify(self, insecure: bool):
        s = get_session(insecure)
        ad = _FakeAdapter()
        with mock.patch.object(type(s), "get_adapter", return_value=ad):
            s.get("https://quay.io/v2/auth")
        return ad.captured.get("verify")

    def test_insecure_verify_false_survives_ca_bundle_env(self):
        """REQUESTS_CA_BUNDLE 存在时，insecure session 的请求级 verify 仍为 False。

        回归点：session.verify=False（无请求级 hook）会被环境变量覆盖成 CA 路径，
        SSL 校验复活。"""
        self.assertIs(self._capture_verify(True), False)

    def test_secure_verify_uses_ca_bundle_env(self):
        """非 insecure：verify=True 且被 REQUESTS_CA_BUNDLE 覆盖为该路径（requests 原生行为，保留）。"""
        self.assertEqual(self._capture_verify(False), "Z:/fake/ca.pem")

    def test_insecure_keeps_trust_env_for_proxies(self):
        """insecure 不能禁用 trust_env——真实业务环境靠环境/系统代理出网。"""
        s = get_session(True)
        self.assertTrue(s.trust_env)

    def test_explicit_verify_kwarg_not_overridden(self):
        """调用方显式传 verify 时不被 hook 强改（setdefault 语义）。

        注：显式 verify=True 会被 requests 原生逻辑换成 REQUESTS_CA_BUNDLE
        （verify=True 的语义就是"用环境 CA"），故用路径串验证 hook 不覆盖显式值。"""
        s = get_session(True)
        ad = _FakeAdapter()
        with mock.patch.object(type(s), "get_adapter", return_value=ad):
            s.get("https://quay.io/x", verify="Z:/other/ca.pem")
        self.assertEqual(ad.captured.get("verify"), "Z:/other/ca.pem")

    def test_insecure_from_env(self):
        os.environ["VLLM_KB_INSECURE"] = "1"
        self.assertTrue(insecure_from_env())
        os.environ["VLLM_KB_INSECURE"] = "false"
        self.assertFalse(insecure_from_env())
        os.environ.pop("VLLM_KB_INSECURE", None)
        self.assertFalse(insecure_from_env())


class TestUrllibSslContext(unittest.TestCase):
    """urllib 侧证书上下文：默认叠加 certifi（修"同一机器上 requests 能跑、urllib 全废"）。

    实测背景（本机 venv，OpenSSL 1.1.1t）：`ssl.create_default_context()` 只带 26 张 CA，
    缺 GitHub/quay 的根 → 所有 urllib HTTPS 报
    `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate (_ssl.c:1007)`，
    表现为版本日历/代码快照/vllm 快照"全部 SSL 失败"；叠加 certifi（138 张）后正常。
    """

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")}
        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    @staticmethod
    def _certifi_calls(m):
        """只挑"加载 certifi 根证书"的调用——create_default_context() 内部也会调
        load_verify_locations（加载默认 cafile/capath，本机 26 张），不能混为一谈。
        autospec 记录 C 方法时不一定带 self，故按"任一实参等于 certifi 路径"匹配。"""
        import certifi

        path = certifi.where()
        out = []
        for c in m.call_args_list:
            vals = [str(a) for a in c.args] + [str(v) for v in c.kwargs.values()]
            if path in vals:
                out.append(c)
        return out

    def test_secure_adds_certifi_roots(self):
        from vllm_kb.net import ssl_context

        with mock.patch.object(ssl.SSLContext, "load_verify_locations",
                               autospec=True) as m:
            ssl_context(False)
        self.assertEqual(len(self._certifi_calls(m)), 1,
                         "默认校验路径必须叠加一次 certifi 根证书")

    def test_secure_context_really_trusts_more_roots(self):
        """不看 mock 的真实效果：叠加后 CA 数量必须比默认上下文多。"""
        from vllm_kb.net import ssl_context

        before = ssl.create_default_context().cert_store_stats().get("x509_ca", 0)
        after = ssl_context(False).cert_store_stats().get("x509_ca", 0)
        self.assertGreater(after, before,
                           f"叠加 certifi 后 CA 应增多（{before} → {after}）")

    def test_insecure_disables_verification_without_loading_ca(self):
        from vllm_kb.net import ssl_context

        with mock.patch.object(ssl.SSLContext, "load_verify_locations",
                               autospec=True) as m:
            ctx = ssl_context(True)
        self.assertEqual(self._certifi_calls(m), [])
        self.assertEqual(ctx.verify_mode, ssl.CERT_NONE)
        self.assertFalse(ctx.check_hostname)

    def test_explicit_env_ca_is_respected(self):
        """业务环境显式指定 CA（SSL_CERT_FILE / REQUESTS_CA_BUNDLE）时不叠加 certifi，
        避免破坏"只用内部根证书"的既有配置。"""
        from vllm_kb.net import ssl_context

        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            os.environ[key] = "Z:/fake/ca.pem"
            with mock.patch.object(ssl.SSLContext, "load_verify_locations",
                                   autospec=True) as m:
                ssl_context(False)
            self.assertEqual(self._certifi_calls(m), [],
                             f"{key} 存在时不应叠加 certifi")
            os.environ.pop(key, None)

    def test_missing_certifi_does_not_crash(self):
        from vllm_kb.net import ssl_context

        with mock.patch.dict(sys.modules, {"certifi": None}):
            ctx = ssl_context(False)
        self.assertEqual(ctx.verify_mode, ssl.CERT_REQUIRED)

    def test_get_opener_uses_the_shared_context(self):
        from vllm_kb.net import get_opener

        op = get_opener(False)
        https = [h for h in op.handlers if isinstance(h, urllib.request.HTTPSHandler)]
        self.assertEqual(len(https), 1)
        with mock.patch.object(ssl.SSLContext, "load_verify_locations",
                               autospec=True) as m:
            https2 = [h for h in get_opener(True).handlers
                      if isinstance(h, urllib.request.HTTPSHandler)]
        self.assertEqual(self._certifi_calls(m), [], "insecure opener 不加载 CA")
        self.assertEqual(https2[0]._context.verify_mode, ssl.CERT_NONE)


class TestSnapshotsScriptsUseSharedOpener(unittest.TestCase):
    """三个快照脚本的 _opener 必须复用共享上下文（否则各自的 build_opener() 又没 CA）。"""

    def test_scripts_delegate_to_net(self):
        import importlib
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        for name in ("build_vllm_snapshots", "build_code_snapshots",
                     "build_fork_snapshots"):
            mod = importlib.import_module(name)
            with mock.patch("vllm_kb.net.get_opener", return_value="SENTINEL") as m:
                self.assertEqual(mod._opener(True), "SENTINEL", name)
                self.assertEqual(mod._opener(False), "SENTINEL", name)
                self.assertEqual([c.args for c in m.call_args_list], [(True,), (False,)])


if __name__ == "__main__":
    unittest.main()
