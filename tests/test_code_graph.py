"""代码图谱检索（gh-puller MCP Streamable HTTP 接入）单测。

协议：MCP JSON-RPC 单端点 tools/call，响应 result.structuredContent（优先）/content[0].text。
不依赖真实 gh-puller 服务——mock urllib.request.urlopen 模拟成功/HTTP错误/连接失败/isError。
"""
import json
import unittest
from unittest import mock

from vllm_kb.code_graph import CodeGraphClient, CodeGraphUnavailable
from vllm_kb.config import CodeGraphCfg


def _cfg(**kw) -> CodeGraphCfg:
    base = {"enabled": True, "base_url": "http://localhost:8787",
            "path": "/gh-puller/graph", "timeout_seconds": 5, "max_retries": 1}
    base.update(kw)
    return CodeGraphCfg(**base)


def _rpc_response(result: dict, id_: int = 1) -> mock.MagicMock:
    """造 MCP JSON-RPC 响应（result 是 CallToolResult 信封）。"""
    m = mock.MagicMock()
    payload = {"jsonrpc": "2.0", "id": id_, "result": result}
    m.read.return_value = json.dumps(payload).encode("utf-8")
    m.__enter__ = mock.MagicMock(return_value=m)
    m.__exit__ = mock.MagicMock(return_value=False)
    return m


def _ok_response(data: dict) -> mock.MagicMock:
    """造成功响应：structuredContent=data，isError=False。"""
    return _rpc_response({
        "content": [{"type": "text", "text": json.dumps(data)}],
        "structuredContent": data,
        "isError": False,
    })


class TestProjectMapping(unittest.TestCase):
    def test_default_repo_maps_to_vllm_ascend(self):
        c = CodeGraphClient(_cfg())
        self.assertEqual(c.resolve_project(None), "vllm-project/vllm-ascend")
        self.assertEqual(c.resolve_project("vllm-ascend"), "vllm-project/vllm-ascend")

    def test_vllm_repo_maps(self):
        c = CodeGraphClient(_cfg())
        self.assertEqual(c.resolve_project("vllm"), "vllm-project/vllm")

    def test_custom_map_overrides(self):
        c = CodeGraphClient(_cfg(repo_project_map={"vllm-ascend": "my-org/my-fork"}))
        self.assertEqual(c.resolve_project("vllm-ascend"), "my-org/my-fork")


class TestUnavailableConditions(unittest.TestCase):
    def test_no_base_url_raises(self):
        c = CodeGraphClient(_cfg(base_url=""))
        with self.assertRaises(CodeGraphUnavailable) as cm:
            c.search_graph(project="p", query="q")
        self.assertIn("未配置", str(cm.exception))

    def test_connection_failure_raises(self):
        c = CodeGraphClient(_cfg())
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            with self.assertRaises(CodeGraphUnavailable) as cm:
                c.search_graph(project="p", query="q")
        self.assertIn("不可达", str(cm.exception))

    def test_http_error_raises_with_status(self):
        import urllib.error
        err = urllib.error.HTTPError("http://x", 500, "Server Error", {}, mock.MagicMock())
        err.read = mock.MagicMock(return_value=b'{"error": {"message": "boom"}}')
        c = CodeGraphClient(_cfg())
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(CodeGraphUnavailable) as cm:
                c.search_graph(project="p", query="q")
        self.assertIn("HTTP 500", str(cm.exception))

    def test_jsonrpc_error_raises(self):
        """JSON-RPC 层错误（resp.error）→ CodeGraphUnavailable。"""
        m = mock.MagicMock()
        m.read.return_value = json.dumps({"jsonrpc": "2.0", "id": 1,
                                          "error": {"code": -32601, "message": "Method not found"}}).encode()
        m.__enter__ = mock.MagicMock(return_value=m); m.__exit__ = mock.MagicMock(return_value=False)
        c = CodeGraphClient(_cfg())
        with mock.patch("urllib.request.urlopen", return_value=m):
            with self.assertRaises(CodeGraphUnavailable) as cm:
                c.search_graph(project="p", query="q")
        self.assertIn("JSON-RPC", str(cm.exception))

    def test_tool_iserror_raises_with_detail(self):
        """工具级 isError=true（未知工具/参数错）→ CodeGraphUnavailable，含工具名。"""
        m = _rpc_response({
            "content": [{"type": "text", "text": "unknown tool: bogus"}],
            "structuredContent": {"error": "unknown tool: bogus"},
            "isError": True,
        })
        c = CodeGraphClient(_cfg())
        with mock.patch("urllib.request.urlopen", return_value=m):
            with self.assertRaises(CodeGraphUnavailable) as cm:
                c.search_graph(project="p", query="q")
        self.assertIn("search_graph", str(cm.exception))
        self.assertIn("unknown tool", str(cm.exception))

    def test_non_json_raises(self):
        c = CodeGraphClient(_cfg())
        bad = mock.MagicMock()
        bad.read.return_value = b"not json"
        bad.__enter__ = mock.MagicMock(return_value=bad); bad.__exit__ = mock.MagicMock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=bad):
            with self.assertRaises(CodeGraphUnavailable):
                c.search_graph(project="p", query="q")

    def test_missing_result_envelope_raises(self):
        m = mock.MagicMock()
        m.read.return_value = b'{"jsonrpc":"2.0","id":1}'  # 无 result
        m.__enter__ = mock.MagicMock(return_value=m); m.__exit__ = mock.MagicMock(return_value=False)
        c = CodeGraphClient(_cfg())
        with mock.patch("urllib.request.urlopen", return_value=m):
            with self.assertRaises(CodeGraphUnavailable) as cm:
                c.search_graph(project="p", query="q")
        self.assertIn("result", str(cm.exception))


class TestCircuitBreaker(unittest.TestCase):
    def test_consecutive_failures_open_circuit(self):
        """max_retries=1 → 阈值 2 次连续失败后熔断打开，下次零等待直接抛（不发包）。"""
        c = CodeGraphClient(_cfg(max_retries=1))
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            for _ in range(2):
                with self.assertRaises(CodeGraphUnavailable):
                    c.search_graph(project="p", query="q")
        # 熔断已打开：再调不发包（urlopen 不应再被调）
        with mock.patch("urllib.request.urlopen") as u:
            with self.assertRaises(CodeGraphUnavailable) as cm:
                c.search_graph(project="p", query="q")
            self.assertIn("熔断", str(cm.exception))
            u.assert_not_called()

    def test_success_resets_circuit(self):
        """成功后连续失败计数清零，下次需连续 2 次才熔断。"""
        c = CodeGraphClient(_cfg(max_retries=1))
        with mock.patch("urllib.request.urlopen", return_value=_ok_response({"r": []})):
            c.search_graph(project="p", query="q")  # 成功，清零
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            with self.assertRaises(CodeGraphUnavailable):
                c.search_graph(project="p", query="q")
        self.assertFalse(c._circuit_open())  # 仅 1 次连续，未熔断


class TestRpcEnvelopeAndPassthrough(unittest.TestCase):
    """验证客户端发出的是 MCP JSON-RPC tools/call，且正确解包 structuredContent。"""

    def _capture_request(self, client_method, **call_kwargs):
        c = CodeGraphClient(_cfg())
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["method"] = req.method
            return _ok_response({"ok": True})

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client_method(c, **call_kwargs)
        return captured

    def test_request_is_jsonrpc_tools_call(self):
        cap = self._capture_request(CodeGraphClient.search_graph, project="p", query="auth")
        self.assertEqual(cap["url"], "http://localhost:8787/gh-puller/graph")
        self.assertEqual(cap["body"]["jsonrpc"], "2.0")
        self.assertEqual(cap["body"]["method"], "tools/call")
        self.assertEqual(cap["body"]["params"]["name"], "search_graph")
        self.assertEqual(cap["body"]["params"]["arguments"]["query"], "auth")

    def test_search_graph_args(self):
        cap = self._capture_request(CodeGraphClient.search_graph, project="p", query="auth",
                                    name_pattern=None, label=None, limit=5, offset=10)
        self.assertEqual(cap["body"]["params"]["arguments"]["limit"], 5)
        self.assertEqual(cap["body"]["params"]["arguments"]["offset"], 10)

    def test_search_code_args(self):
        cap = self._capture_request(CodeGraphClient.search_code, project="p", pattern="foo",
                                    mode="full", path_filter="^src/", limit=3)
        a = cap["body"]["params"]["arguments"]
        self.assertEqual(a["pattern"], "foo")
        self.assertEqual(a["mode"], "full")
        self.assertEqual(a["path_filter"], "^src/")

    def test_trace_path_args(self):
        cap = self._capture_request(CodeGraphClient.trace_path, project="p", function_name="bar",
                                    direction="outbound", depth=5, limit=50, cursor=None, mode="data_flow")
        a = cap["body"]["params"]["arguments"]
        self.assertEqual(a["function_name"], "bar")
        self.assertEqual(a["mode"], "data_flow")

    def test_query_graph_args(self):
        cap = self._capture_request(CodeGraphClient.query_graph, project="p",
                                    query="MATCH (n) RETURN n", max_rows=100)
        a = cap["body"]["params"]["arguments"]
        self.assertIn("MATCH", a["query"])
        self.assertEqual(a["max_rows"], 100)

    def test_get_architecture_args(self):
        cap = self._capture_request(CodeGraphClient.get_architecture, project="p",
                                    aspects=["structure", "routes"], path="apps/")
        a = cap["body"]["params"]["arguments"]
        self.assertEqual(a["aspects"], ["structure", "routes"])

    def test_detect_changes_args(self):
        cap = self._capture_request(CodeGraphClient.detect_changes, project="p",
                                    diff="diff --git a/x b/x", scope="impact", direction="both", depth=3, limit=20)
        a = cap["body"]["params"]["arguments"]
        self.assertEqual(a["diff"], "diff --git a/x b/x")
        self.assertEqual(a["direction"], "both")

    def test_structured_content_preferred_over_text(self):
        """structuredContent 存在时返回它，而非 content[0].text。"""
        c = CodeGraphClient(_cfg())
        m = _rpc_response({
            "content": [{"type": "text", "text": "raw text"}],
            "structuredContent": {"key": "value"},
            "isError": False,
        })
        with mock.patch("urllib.request.urlopen", return_value=m):
            result = c.search_graph(project="p", query="q")
        self.assertEqual(result, {"key": "value"})

    def test_text_fallback_when_no_structured_content(self):
        """无 structuredContent 时退回 content[0].text。"""
        c = CodeGraphClient(_cfg())
        m = _rpc_response({
            "content": [{"type": "text", "text": "plain text result"}],
            "isError": False,
        })
        with mock.patch("urllib.request.urlopen", return_value=m):
            result = c.search_graph(project="p", query="q")
        self.assertEqual(result, "plain text result")


class TestHealth(unittest.TestCase):
    def test_not_configured(self):
        c = CodeGraphClient(_cfg(base_url=""))
        self.assertEqual(c.health()["status"], "not_configured")

    def test_unreachable(self):
        c = CodeGraphClient(_cfg())
        with mock.patch("urllib.request.urlopen", side_effect=OSError("x")):
            self.assertEqual(c.health()["status"], "unreachable")

    def test_ok(self):
        c = CodeGraphClient(_cfg())
        m = mock.MagicMock()
        m.read.return_value = json.dumps({"jsonrpc": "2.0", "id": 0,
                                          "result": {"tools": []}}).encode()
        m.__enter__ = mock.MagicMock(return_value=m); m.__exit__ = mock.MagicMock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=m):
            self.assertEqual(c.health()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
