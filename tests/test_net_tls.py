"""端到端 TLS 故障注入测试：get_session 的 insecure 在真实 HTTPS 链路上的行为。

固化 fix(0648964) 的故障注入验证（原为手工执行的探索脚本）：
本地自签证书 HTTPS server + REQUESTS_CA_BUNDLE 指向**无关证书**——等价于内网
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


if __name__ == "__main__":
    unittest.main()
