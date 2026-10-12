"""端到端 TLS 故障注入测试：get_session 的 insecure 在真实 HTTPS 链路上的行为。

固化 fix(0648964) 的故障注入验证（原为手工执行的探索脚本）：
本地自签证书 HTTPS server + REQUESTS_CA_BUNDLE 指向**无关证书**——等价于真实业务环境
MITM 环境的常见配置（环境变量存在，但指向的 CA 不能验证目标站点证书）。

场景矩阵（http://127.0.0.1:{port}，全部离线，不依赖外网）：
  1. insecure=False                        -> SSLError（故障真实存在的负对照）
  2. insecure=False + CA_BUNDLE=无关证书    -> SSLError
  3. insecure=True  + CA_BUNDLE=无关证书    -> 200（**回归点**：修复前
     session.verify=False 被该环境变量覆盖成 CA 路径，校验复活，此场景失败）
  4. insecure=True                         -> 200（基础 insecure 功能）
  5. insecure=False + CA_BUNDLE=server 证书 -> 200（证明环境变量确实在驱动
     校验——注入机制生效，场景 3 的成功才有意义，防测试空洞化）

证书 fixture（tests/fixtures/tls/，预生成提交，零运行时依赖）：
  server.pem/server.key  本地 server 自签证书（SAN=IP:127.0.0.1,DNS=localhost）
  fault_ca.pem           无关自签证书（独立密钥），作假 REQUESTS_CA_BUNDLE
  （均为一次性测试证书，仅 localhost 用途，无私钥泄露风险）
"""
import http.server
import os
import ssl
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from vllm_kb.net import get_session

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tls"
# 显式禁代理（requests 惯用 None 值写法）：测试目标是 TLS 校验行为，
# 排除系统/环境代理对 127.0.0.1 的截获干扰（如本机 clash 7892）
_NO_PROXY = {"http": None, "https": None}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # 静音，避免污染测试输出


class TestInsecureTlsEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_cert = _FIXTURES / "server.pem"
        cls.server_key = _FIXTURES / "server.key"
        cls.fault_ca = _FIXTURES / "fault_ca.pem"
        if not (cls.server_cert.exists() and cls.server_key.exists()
                and cls.fault_ca.exists()):
            raise unittest.SkipTest("缺 TLS fixture（tests/fixtures/tls/）")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cls.server_cert), str(cls.server_key))
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.httpd.socket = ctx.wrap_socket(cls.httpd.socket, server_side=True)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.url = f"https://127.0.0.1:{cls.port}/"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        # REQUESTS_CA_BUNDLE 注入与还原（每用例独立，防串扰）
        self._saved = {k: os.environ.get(k) for k in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")}
        os.environ.pop("CURL_CA_BUNDLE", None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ---- 场景 1：负对照（自签证书不被信任 → SSL 失败，故障真实存在） ----
    def test_secure_fails_on_self_signed(self):
        import requests

        os.environ.pop("REQUESTS_CA_BUNDLE", None)
        with self.assertRaises(requests.exceptions.SSLError):
            get_session(False).get(self.url, timeout=10, proxies=_NO_PROXY)

    # ---- 场景 2：无关 CA 注入 + secure → 仍失败 ----
    def test_secure_fails_with_unrelated_ca_bundle(self):
        import requests

        os.environ["REQUESTS_CA_BUNDLE"] = str(self.fault_ca)
        with self.assertRaises(requests.exceptions.SSLError):
            get_session(False).get(self.url, timeout=10, proxies=_NO_PROXY)

    # ---- 场景 3（回归点）：无关 CA 注入 + insecure → 必须成功 ----
    def test_insecure_survives_unrelated_ca_bundle(self):
        """修复前失败：session.verify=False 被 REQUESTS_CA_BUNDLE 覆盖，
        校验复活（requests merge_environment_settings 请求级 None 时读环境变量）。"""
        os.environ["REQUESTS_CA_BUNDLE"] = str(self.fault_ca)
        r = get_session(True).get(self.url, timeout=10, proxies=_NO_PROXY)
        self.assertEqual(r.status_code, 200)

    # ---- 场景 4：基础 insecure（无注入）→ 成功 ----
    def test_insecure_connects_self_signed(self):
        os.environ.pop("REQUESTS_CA_BUNDLE", None)
        r = get_session(True).get(self.url, timeout=10, proxies=_NO_PROXY)
        self.assertEqual(r.status_code, 200)

    # ---- 场景 5：注入机制生效性证明（防场景 3 空洞化） ----
    def test_secure_with_server_cert_bundle_succeeds(self):
        """REQUESTS_CA_BUNDLE 指向 server 自身证书 → secure 也成功：
        证明环境变量确实驱动校验（注入机制活着），场景 3 的成功因此有区分力。"""
        os.environ["REQUESTS_CA_BUNDLE"] = str(self.server_cert)
        r = get_session(False).get(self.url, timeout=10, proxies=_NO_PROXY)
        self.assertEqual(r.status_code, 200)


class TestUrllibOpenerTlsEndToEnd(unittest.TestCase):
    """urllib 侧（build_*_snapshots / build_release_calendar 用的 get_opener）同样的故障注入。

    与 requests 版同源问题：urllib 走的是 `ssl.create_default_context()`，而本机 venv 的
    OpenSSL 默认 CA 列表不足以验证外网站点（实测 26 张，缺 GitHub 的根）→ 全线
    CERTIFICATE_VERIFY_FAILED。修复是给上下文叠加 certifi；这里用**本地自签服务器**
    确认叠加不是"无脑信任一切"：自签证书依旧被拒（负对照），而 SSL_CERT_FILE 指向
    服务器自身证书时能通过（证明环境变量确实驱动校验，负对照不空洞）。
    """

    @classmethod
    def setUpClass(cls):
        cls.server_cert = _FIXTURES / "server.pem"
        cls.server_key = _FIXTURES / "server.key"
        cls.fault_ca = _FIXTURES / "fault_ca.pem"
        if not (cls.server_cert.exists() and cls.server_key.exists()
                and cls.fault_ca.exists()):
            raise unittest.SkipTest("缺 TLS fixture（tests/fixtures/tls/）")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cls.server_cert), str(cls.server_key))
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.httpd.socket = ctx.wrap_socket(cls.httpd.socket, server_side=True)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.url = f"https://127.0.0.1:{cls.port}/"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")}
        os.environ.pop("REQUESTS_CA_BUNDLE", None)   # 只测 urllib 语境

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _open(self, insecure):
        """绕开系统代理（本机注册表里配了 127.0.0.1:7892），只测 TLS 行为。"""
        from vllm_kb.net import ssl_context

        op = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=ssl_context(insecure)),
        )
        return op.open(self.url, timeout=10)

    def test_secure_still_rejects_self_signed(self):
        """负对照：叠加 certifi ≠ 信任一切，自签证书依旧失败。"""
        os.environ.pop("SSL_CERT_FILE", None)
        with self.assertRaises(urllib.error.URLError):
            self._open(False)

    def test_secure_with_unrelated_ca_fails(self):
        os.environ["SSL_CERT_FILE"] = str(self.fault_ca)
        with self.assertRaises(urllib.error.URLError):
            self._open(False)

    def test_secure_with_server_ca_succeeds(self):
        """环境变量确实驱动校验（防上一条空洞化）。"""
        os.environ["SSL_CERT_FILE"] = str(self.server_cert)
        with self._open(False) as r:
            self.assertEqual(r.status, 200)

    def test_insecure_connects_self_signed(self):
        os.environ["SSL_CERT_FILE"] = str(self.fault_ca)
        with self._open(True) as r:
            self.assertEqual(r.status, 200)

    def test_insecure_ignores_env_ca(self):
        """insecure 时 SSL_CERT_FILE 不得让校验"复活"（与 requests 版同一回归点）。"""
        os.environ["SSL_CERT_FILE"] = str(self.server_cert)
        with self._open(True) as r:
            self.assertEqual(r.status, 200)


if __name__ == "__main__":
    unittest.main()
