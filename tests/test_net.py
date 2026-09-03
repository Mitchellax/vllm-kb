"""vllm_kb.net 单元测试：get_session 的 insecure SSL 跳过（不触网）。

回归：requests 的 merge_environment_settings 在请求级 verify 为 None 时会用
环境变量 REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE 覆盖 session.verify=False——
insecure 因此失效（真实业务环境常设这两个变量过 MITM，实测 requests 2.34 复现）。
修复为请求级 verify=False hook，本测试锁定该行为。
"""
import os
import unittest
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


if __name__ == "__main__":
    unittest.main()
