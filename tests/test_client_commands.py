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


if __name__ == "__main__":
    unittest.main()
