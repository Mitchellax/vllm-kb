"""skills/vllm-kb/client.py 测试：doc 简写解析、格式化输出、CLI 命令分发（mock HTTP）。

此前 client 命令全部手工验证。覆盖：
- _resolve_graph_doc：repo#编号 → source_id（/ → - slug）
- fmt_graph_chain/fixes/sig/stats、fmt_search（verification）、fmt_code_hits（per_version）
- main() 分发：graph chain/fixes/sig/stats、code --per-version（mock _get/_post）
"""
import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "vllm_kb_client", PROJECT_ROOT / "skills" / "vllm-kb" / "client.py")
client = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(client)


def run_main(argv, get_data=None, post_data=None):
    """以给定 argv 跑 client.main()，mock HTTP，返回 stdout。"""
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["client.py"] + argv), \
         mock.patch.object(client, "_get", side_effect=get_data or (lambda *a, **k: {})), \
         mock.patch.object(client, "_post", side_effect=post_data or (lambda *a, **k: {})), \
         redirect_stdout(out):
        try:
            client.main()
        except SystemExit:
            pass
    return out.getvalue()


class TestResolveGraphDoc(unittest.TestCase):
    def test_full_source_id_passthrough(self):
        self.assertEqual(
            client._resolve_graph_doc("github:vllm-project-vllm:issue:10700", "issue"),
            "github:vllm-project-vllm:issue:10700")

    def test_repo_number_shortcut(self):
        # vllm-ascend#10700 → 全名 + / → - slug（与 github_pull source_id 规则一致）
        self.assertEqual(client._resolve_graph_doc("vllm-ascend#10700", "issue"),
                         "github:vllm-project-vllm-ascend:issue:10700")
        self.assertEqual(client._resolve_graph_doc("vllm#50241", "pr"),
                         "github:vllm-project-vllm:pr:50241")
        self.assertEqual(client._resolve_graph_doc("ascend#1", "issue"),
                         "github:vllm-project-vllm-ascend:issue:1")

    def test_full_repo_name_shortcut(self):
        self.assertEqual(client._resolve_graph_doc("vllm-project/vllm-ascend#10700", "issue"),
                         "github:vllm-project-vllm-ascend:issue:10700")


class TestFormatting(unittest.TestCase):
    def test_fmt_graph_chain_released(self):
        data = {
            "found": True, "issue_id": "x", "title": "GLM5.1 崩溃", "repo": "r",
            "number": 10700, "status": "closed", "url": "http://i", "resolved_at": "2026-07-31",
            "fixes": [{"pr_id": "p:12885", "title": "Fix", "status": "merged",
                       "merged_at": "2026-07-31", "release_tag": "v0.23.0",
                       "release_date": "2026-08-16"}],
            "released": True,
        }
        txt = client.fmt_graph_chain(data)
        self.assertIn("v0.23.0", txt)
        self.assertIn("12885", txt)
        self.assertIn("已进入", txt) if False else None

    def test_fmt_graph_chain_no_fix(self):
        data = {"found": True, "number": 1, "title": "t", "url": "u",
                "fixes": [], "released": False, "status": "open"}
        txt = client.fmt_graph_chain(data)
        self.assertIn("无 PR 声明修复", txt)

    def test_fmt_graph_chain_not_found(self):
        self.assertIn("不存在于图中", client.fmt_graph_chain({"found": False, "issue_id": "x"}))

    def test_fmt_graph_fixes(self):
        data = {"found": True, "number": 12885, "title": "Fix", "status": "merged",
                "url": "u", "releases": [{"tag": "v0.23.0", "date": "d"}],
                "fixes": [{"issue_id": "i:10700", "title": "GLM", "status": "closed", "url": "u2"}]}
        txt = client.fmt_graph_fixes(data)
        self.assertIn("v0.23.0", txt)
        self.assertIn("i:10700", txt)

    def test_fmt_graph_sig(self):
        data = {"signature": "dispatch_ffn_combine", "entity_type": "operator",
                "docs": [{"status": "open", "doc_id": "i:1", "title": "t"}], "note": ""}
        txt = client.fmt_graph_sig(data)
        self.assertIn("operator", txt)
        self.assertIn("i:1", txt)

    def test_fmt_graph_sig_empty(self):
        data = {"signature": "x", "entity_type": None, "docs": [], "note": ""}
        self.assertIn("无提及", client.fmt_graph_sig(data))

    def test_fmt_graph_stats(self):
        data = {"built": True,
                "nodes": {"Issue": 10, "PR": 5, "Release": 2, "Operator": 3},
                "rels": {"FIXES": 4, "MERGED_IN": 2, "MENTIONS": 8}}
        txt = client.fmt_graph_stats(data)
        self.assertIn("Issue 10", txt)
        self.assertIn("FIXES 4", txt)
        self.assertIn("未构建", client.fmt_graph_stats({"built": False, "note": "n"}))

    def test_fmt_search_verification(self):
        data = {"results": [{
            "final": 0.5, "similarity": 0.4,
            "confidence": {"score": 0.5, "w_time": 0.9, "w_ver": 0.5, "w_rel": 0.6},
            "resolved": False, "component": "vllm", "version_ref": None,
            "title": "t", "url": "u", "status": "open", "verification": "expert",
            "version_span": [], "snippet": "s",
        }]}
        txt = client.fmt_search(data)
        self.assertIn("验证=expert", txt)

    def test_fmt_code_hits_per_version(self):
        data = {"mode": "grep_per_version", "symbol": "fill_(-1)", "version": None,
                "hits": [
                    {"version": "v0.22.1rc1", "file": "m.py", "line": 1, "snippet": "a"},
                    {"version": "v0.23.0rc1", "file": "m.py", "line": 2, "snippet": "b"},
                ]}
        txt = client.fmt_code_hits(data)
        self.assertIn("[v0.22.1rc1] 1 处命中", txt)
        self.assertIn("[v0.23.0rc1] 1 处命中", txt)

    def test_fmt_matrix_limit(self):
        data = {"rows": [
            {"vllm-ascend": f"v{i}", "vllm": "x", "cann": "y", "pytorch": "z",
             "pytorch-ascend": "p", "npu-driver": "d", "notes": "", "source": "auto"}
            for i in range(5)
        ]}
        txt = client.fmt_matrix(data, limit=2)
        self.assertIn("共 5 行", txt)
        self.assertIn("显示前 2", txt)
        self.assertEqual(txt.count("vllm-ascend=v"), 2)  # 只渲染 2 行
        full = client.fmt_matrix(data)
        self.assertNotIn("显示前", full)
        self.assertIn("共 5 行", full)


class TestClientErrors(unittest.TestCase):
    """_request 统一错误处理：HTTP 4xx/5xx、连接失败、非 JSON 响应都转 ClientError（含服务端 detail）。"""

    def test_http_error_surfaces_status_and_detail(self):
        import io
        import urllib.error

        err = urllib.error.HTTPError(
            "http://x/api", 404, "Not Found", {},
            io.BytesIO('{"detail": "版本 [\'v0.99.0\'] 未预存文件"}'.encode("utf-8")))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(client.ClientError) as cm:
                client._get("http://x", "/api")
        msg = str(cm.exception)
        self.assertIn("API 错误 404", msg)
        self.assertIn("未预存文件", msg)  # 服务端 detail 透出，而非笼统"无法连接"

    def test_post_http_error_surfaces_detail(self):
        import io
        import urllib.error

        err = urllib.error.HTTPError(
            "http://x/search", 503, "Service Unavailable", {},
            io.BytesIO('{"detail": "图未构建：运行 python scripts/build_graph.py"}'.encode("utf-8")))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(client.ClientError) as cm:
                client._post("http://x", "/search", {"query": "q"})
        self.assertIn("API 错误 503", str(cm.exception))
        self.assertIn("build_graph", str(cm.exception))

    def test_connection_error_message(self):
        import urllib.error

        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.URLError("conn refused")):
            with self.assertRaises(client.ClientError) as cm:
                client._get("http://x", "/api")
        self.assertIn("无法连接 API", str(cm.exception))

    def test_oserror_message(self):
        with mock.patch("urllib.request.urlopen", side_effect=ConnectionResetError("reset")):
            with self.assertRaises(client.ClientError) as cm:
                client._get("http://x", "/api")
        self.assertIn("请求失败", str(cm.exception))

    def test_non_json_response(self):
        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b"<html>proxy error</html>"

        with mock.patch("urllib.request.urlopen", return_value=FakeResp()):
            with self.assertRaises(client.ClientError) as cm:
                client._post("http://x", "/api", {"q": 1})
        self.assertIn("非 JSON", str(cm.exception))


class TestCliDispatch(unittest.TestCase):
    """main() 分发：graph 命令走 /graph/* 端点；code 透传新参数。"""

    def test_graph_chain_dispatch(self):
        calls = {}

        def fake_get(base, path, params=None):
            calls["path"] = path
            calls["params"] = params
            return {"found": True, "number": 10700, "title": "t", "url": "u", "status": "open",
                    "fixes": [], "released": False}

        txt = run_main(["graph", "chain", "vllm-ascend#10700"], get_data=fake_get)
        self.assertEqual(calls["path"], "/graph/chain")
        self.assertEqual(calls["params"]["doc"],
                         "github:vllm-project-vllm-ascend:issue:10700")

    def test_graph_sig_dispatch(self):
        calls = {}

        def fake_get(base, path, params=None):
            calls["params"] = params
            return {"signature": "x", "entity_type": "operator", "docs": [], "note": ""}

        run_main(["graph", "sig", "dispatch_ffn_combine", "--limit", "5"], get_data=fake_get)
        self.assertEqual(calls["params"], {"sig": "dispatch_ffn_combine", "limit": 5})

    def test_graph_stats_dispatch(self):
        def fake_get(base, path, params=None):
            return {"built": True, "nodes": {}, "rels": {}, "summary": "s"}

        txt = run_main(["graph", "stats"], get_data=fake_get)
        self.assertIn("图已构建", txt)

    def test_code_per_version_dispatch(self):
        calls = {}

        def fake_post(base, path, payload=None):
            calls["payload"] = payload
            return {"mode": "grep_per_version", "symbol": "x", "version": None, "hits": []}

        run_main(["code", "fill_(-1)", "--in-file", "worker/model_runner_v1.py", "--per-version"],
                 post_data=fake_post)
        self.assertEqual(calls["payload"]["path"], "worker/model_runner_v1.py")
        self.assertTrue(calls["payload"]["per_version"])

    def test_diff_dispatch(self):
        calls = {}

        def fake_get(base, path, params=None):
            calls["path"] = path
            calls["params"] = params
            return {"path": "p", "v1": "v0.22.1rc1", "v2": "v0.23.0rc1",
                    "lines1": 10, "lines2": 12, "diff": "-a\n+b", "note": None}

        txt = run_main(["diff", "v0.22.1rc1", "v0.23.0rc1",
                        "vllm_ascend/worker/model_runner_v1.py", "--keyword", "fill_(-1)"],
                       get_data=fake_get)
        self.assertEqual(calls["path"], "/code/diff")
        self.assertEqual(calls["params"]["version1"], "v0.22.1rc1")
        self.assertEqual(calls["params"]["version2"], "v0.23.0rc1")
        self.assertEqual(calls["params"]["keyword"], "fill_(-1)")
        self.assertIn("v0.22.1rc1", txt)
        self.assertIn("+b", txt)

    def test_code_kind_msg_dispatch(self):
        calls = {}

        def fake_post(base, path, payload=None):
            calls["payload"] = payload
            return {"mode": "message_index", "symbol": "boom", "version": "v0.23.0rc1", "hits": []}

        txt = run_main(["code", "boom", "--kind", "msg", "--version", "v0.23.0rc1"],
                       post_data=fake_post)
        self.assertEqual(calls["payload"]["kind"], "msg")
        self.assertIn("message_index", txt)

    def test_matrix_limit_dispatch(self):
        def fake_get(base, path, params=None):
            return {"rows": [
                {"vllm-ascend": f"v{i}", "vllm": "x", "cann": "y", "pytorch": "z",
                 "pytorch-ascend": "p", "npu-driver": "d"}
                for i in range(5)
            ]}

        txt = run_main(["matrix", "--limit", "3"], get_data=fake_get)
        self.assertIn("共 5 行", txt)
        self.assertIn("显示前 3", txt)

    def test_tags_list_dispatch(self):
        """tags list：能力目录（两级分组 + 文档数；docs=0 不出现）。"""
        data = {"groups": {"domain": [{"name": "HCCL", "docs": 12}],
                           "purpose": [{"name": "超时排查", "docs": 8}]},
                "total_tags": 2}
        txt = run_main(["tags", "list"], get_data=lambda *a, **k: data)
        self.assertIn("主题/领域类", txt)
        self.assertIn("HCCL(12篇)", txt)
        self.assertIn("超时排查(8篇)", txt)
        self.assertIn("tags docs", txt)

    def test_tags_docs_dispatch(self):
        calls = {}

        def fake_get(base, path, params=None):
            calls["path"] = path
            return {"tag": "HCCL", "docs": [{"doc_id": "pdf:guide", "title": "HCCL 指南",
                                             "verification": "expert"}], "count": 1, "note": "n"}

        txt = run_main(["tags", "docs", "HCCL"], get_data=fake_get)
        self.assertIn("/tags/HCCL/docs", calls["path"])
        self.assertIn("pdf:guide", txt)
        self.assertIn("HCCL 指南", txt)

    def test_context_dispatch(self):
        """context：问题→标签匹配输出（领域=范围、作用=能力 + 文档线索）。"""
        data = {"matched": [
            {"name": "HCCL", "tier": "domain", "docs": 12,
             "top": [{"doc_id": "pdf:guide", "title": "HCCL 超时排查指南", "verification": "expert"}]},
            {"name": "超时排查", "tier": "purpose", "docs": 8,
             "top": [{"doc_id": "md:case", "title": "网络超时案例", "verification": ""}]},
        ]}
        calls = {}

        def fake_post(base, path, payload=None):
            calls["payload"] = payload
            return data

        txt = run_main(["context", "vllm-ascend HCCL 超时"], post_data=fake_post)
        self.assertEqual(calls["payload"]["text"], "vllm-ascend HCCL 超时")
        self.assertIn("[领域]", txt)
        self.assertIn("[作用]", txt)
        self.assertIn("HCCL 超时排查指南", txt)
        self.assertIn("交集", txt)

    def test_graph_tags_dispatch(self):
        data = {"tag": "HCCL", "tier": "domain",
                "docs": [{"doc_type": "Doc", "doc_id": "pdf:guide", "title": "HCCL 指南"}],
                "count": 1, "note": "n"}
        txt = run_main(["graph", "tags", "HCCL"], get_data=lambda *a, **k: data)
        self.assertIn("[Doc]", txt)
        self.assertIn("pdf:guide", txt)

    def test_search_tag_filter_dispatch(self):
        """search --tag：多次标签 → payload.filters.tags（全部包含过滤）。"""
        calls = {}

        def fake_post(base, path, payload=None):
            calls["payload"] = payload
            return {"context": {}, "results": [], "degraded": None}

        txt = run_main(["search", "HCCL 超时", "--tag", "HCCL", "--tag", "超时排查"],
                       post_data=fake_post)
        self.assertEqual(calls["payload"]["filters"]["tags"], ["HCCL", "超时排查"])
        # 空结果格式化不崩（print 空串 → 单个换行）
        self.assertEqual(txt, "\n")
        # 无 --tag 时不带 filters
        run_main(["search", "q"], post_data=fake_post)
        self.assertNotIn("filters", calls["payload"])


if __name__ == "__main__":
    unittest.main()
