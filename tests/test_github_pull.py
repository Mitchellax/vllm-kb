"""GitHub 采集转换逻辑测试（不触网）：canonical 映射、版本提取、PR 处理、recanonicalize。"""
import json
import unittest
from pathlib import Path

import requests

from vllm_kb.config import PROJECT_ROOT, SourceCfg
from vllm_kb.github_pull import GithubPuller


def make_source(**overrides) -> SourceCfg:
    base = {
        "id": "vllm",
        "type": "github",
        "repo": "vllm-project/vllm",
        "token": "",
        "token_env": "GITHUB_TOKEN",
    }
    base.update(overrides)
    return SourceCfg.model_validate(base)


def make_item(number=1, body="", labels=None, state="closed", closed_at="2026-08-01T00:00:00Z", title="some title"):
    return {
        "number": number,
        "title": title,
        "body": body,
        "labels": [{"name": l} for l in (labels or [])],
        "state": state,
        "created_at": "2026-07-01T00:00:00Z",
        "closed_at": closed_at,
        "html_url": f"https://github.com/vllm-project/vllm/issues/{number}",
    }


def _node(number, created_at="2026-01-01T00:00:00Z"):
    """构造最小 GraphQL issues 节点。"""
    return {
        "number": number, "title": "t", "body": "b", "state": "OPEN",
        "createdAt": created_at, "closedAt": None,
        "url": f"https://github.com/x/issues/{number}",
        "author": None, "labels": {"nodes": []},
    }


class TestGithubConvert(unittest.TestCase):
    def setUp(self):
        self.puller = GithubPuller(make_source(), PROJECT_ROOT)  # 仅构造，不发请求

    def test_basic_mapping(self):
        doc = self.puller._to_doc(make_item(number=42, body="hello"), [])
        self.assertEqual(doc.source_id, "github:vllm-project-vllm:issue:42")
        self.assertEqual(doc.status, "closed")
        self.assertEqual(doc.resolved_at, "2026-08-01T00:00:00Z")
        self.assertEqual(doc.body, "hello")
        self.assertEqual(doc.extra.get("repo"), "vllm-project/vllm")
        self.assertEqual(doc.component, "vllm")  # 仓库推断主组件

    def test_insecure_from_env(self):
        """VLLM_KB_INSECURE=1 时 session 跳过 SSL 校验（真实业务环境增量拉取依赖）。"""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"VLLM_KB_INSECURE": "1"}, clear=False):
            puller = GithubPuller(make_source(), PROJECT_ROOT)
            self.assertTrue(puller.insecure)
            self.assertFalse(puller.session.verify)  # requests verify=False

    def test_insecure_from_source_config(self):
        """来源配置 insecure: true 同样生效（优先级低于环境变量无需区分，任一为真即可）。"""
        puller = GithubPuller(make_source(insecure=True), PROJECT_ROOT)
        self.assertTrue(puller.insecure)
        self.assertFalse(puller.session.verify)

    def test_api_base_and_graphql_follow_config(self):
        """api_base 配置/环境变量：REST base 与 GraphQL 端点均跟随（不再硬编码 api.github.com）。"""
        import os
        from unittest.mock import patch

        # 配置显式指定（优先于环境变量）
        puller = GithubPuller(make_source(api_base="http://gh-mirror.internal:8080"), PROJECT_ROOT)
        self.assertEqual(puller.base, "http://gh-mirror.internal:8080")
        self.assertEqual(puller.graphql_url, "http://gh-mirror.internal:8080/graphql")
        # 仅环境变量（无配置字段）
        with patch.dict(os.environ, {"VLLM_KB_GITHUB_BASE": "http://gh-env.internal"}, clear=False):
            puller2 = GithubPuller(make_source(), PROJECT_ROOT)
            self.assertEqual(puller2.base, "http://gh-env.internal")
            self.assertEqual(puller2.graphql_url, "http://gh-env.internal/graphql")

    def test_default_base_is_github_com(self):
        """无任何配置/环境变量时保持默认 api.github.com。"""
        import os
        from unittest.mock import patch

        env = {k: v for k, v in os.environ.items() if k != "VLLM_KB_GITHUB_BASE"}
        with patch.dict(os.environ, env, clear=True):
            puller = GithubPuller(make_source(), PROJECT_ROOT)
            self.assertEqual(puller.base, "https://api.github.com")
            self.assertEqual(puller.graphql_url, "https://api.github.com/graphql")

    def test_vllm_ascend_component_and_versions(self):
        """vllm-ascend 仓库：主组件 vllm-ascend，正文提取的配套版本进 component_versions。"""
        body = (
            "vllm-ascend version: 0.18.0\n"
            "vllm 0.12.1, CANN 8.1.RC2, torch 2.6.0"
        )
        puller = GithubPuller(make_source(repo="vllm-project/vllm-ascend"), PROJECT_ROOT)
        doc = puller._to_doc(make_item(number=7, body=body), [])
        self.assertEqual(doc.component, "vllm-ascend")
        self.assertEqual(doc.version_span.min, "0.18.0")  # 主组件版本进 version_span
        self.assertEqual(doc.component_versions.get("vllm"), "0.12.1")
        self.assertEqual(doc.component_versions.get("cann"), "8.1.RC2")
        self.assertNotIn("vllm-ascend", doc.component_versions)  # 主组件不进 component_versions

    def test_comments_become_thread(self):
        comments = [
            {"user": {"login": "alice"}, "created_at": "2026-07-02T00:00:00Z", "body": "try X"},
            {"user": {"login": "bob"}, "created_at": "2026-07-03T00:00:00Z", "body": "fixed by Y"},
        ]
        doc = self.puller._to_doc(make_item(body="orig"), comments)
        self.assertIn("orig", doc.body)
        self.assertIn("try X", doc.body)
        self.assertIn("fixed by Y", doc.body)
        self.assertEqual(doc.extra.get("comment_count"), 2)

    def test_version_from_body_template(self):
        body = (
            "### Your current environment\n"
            "- **vLLM version**: 0.26.0\n"
            "- OS: Linux\n"
            "- GPU: H100\n"
        )
        doc = self.puller._to_doc(make_item(body=body), [])
        self.assertEqual(doc.version_span.min, "0.26.0")

    def test_version_from_plain_body(self):
        doc = self.puller._to_doc(make_item(body="using vllm version 0.5.4 here"), [])
        self.assertEqual(doc.version_span.min, "0.5.4")

    def test_version_loose_format(self):
        doc = self.puller._to_doc(make_item(body="### Your current environment\n- vLLM: v0.24.0"), [])
        self.assertEqual(doc.version_span.min, "0.24.0")

    def test_label_version_preferred_over_body(self):
        body = "### Your current environment\n- **vLLM version**: 0.26.0"
        # 标签 "bug: v0.6.x" -> 捕获 "0.6"（.x 非数字，patch 未知）
        doc = self.puller._to_doc(make_item(body=body, labels=["bug: v0.6.x"]), [])
        self.assertEqual(doc.version_span.min, "0.6")  # 标签优先于正文

    def test_no_version_anywhere(self):
        doc = self.puller._to_doc(make_item(body="no version mentioned"), [])
        self.assertIsNone(doc.version_span.min)

    def test_pr_body_version_not_extracted(self):
        """PR 正文提到版本不可靠（兼容性/变更说明常见），版本信号只信标签。"""
        item = make_item(number=99, body="Compatible with vLLM 0.24.0 since the refactor")
        item["pull_request"] = {"merged": True, "merged_at": "2026-08-01T00:00:00Z", "merge_commit_sha": "x"}
        doc = self.puller._to_doc(item, [])
        self.assertIsNone(doc.version_span.min)
        # 标签里的版本仍生效
        item2 = make_item(number=100, body="Compatible with vLLM 0.24.0", labels=["bug: v0.6.x"])
        item2["pull_request"] = {"merged": True, "merged_at": "2026-08-01T00:00:00Z", "merge_commit_sha": "x"}
        doc2 = self.puller._to_doc(item2, [])
        self.assertEqual(doc2.version_span.min, "0.6")

    def test_incomplete_version_not_matched_loosely(self):
        """宽松正则要求完整 semver：'vllm 0.1' / 'vLLM 0.23' 这类残缺版本不匹配。"""
        doc = self.puller._to_doc(make_item(body="known issue with vllm 0.1 and vLLM 0.23"), [])
        self.assertIsNone(doc.version_span.min)

    def test_environment_section_stripped(self):
        """环境转储段（pip list/GPU/网卡等噪音）从语义正文剥离，放入 extra['environment']。"""
        body = (
            "### Bug Description\n"
            "Crash with illegal memory access under concurrency.\n\n"
            "### Your current environment\n"
            "- **vLLM version**: 0.26.0\n"
            "- OS: Linux\n"
            "- GPU: H100\n"
            "- pip list: triton 3.4.0, torch 2.11.0\n\n"
            "### Extra\n"
            "some trailing note"
        )
        doc = self.puller._to_doc(make_item(body=body), [])
        # 语义正文不含环境段
        self.assertNotIn("Your current environment", doc.body)
        self.assertNotIn("triton 3.4.0", doc.body)
        self.assertIn("Bug Description", doc.body)
        self.assertIn("illegal memory access", doc.body)
        # 环境段在 extra 中（版本提取仍在原始正文上进行）
        self.assertIn("triton 3.4.0", doc.extra.get("environment", ""))
        self.assertEqual(doc.version_span.min, "0.26.0")

    def test_environment_strip_keeps_comments(self):
        body = "Description text.\n\n### Your current environment\n- **vLLM version**: 0.26.0"
        comments = [{"user": {"login": "m"}, "created_at": "2026-07-02T00:00:00Z", "body": "try --max-num-seqs 16"}]
        doc = self.puller._to_doc(make_item(body=body), comments)
        self.assertIn("try --max-num-seqs 16", doc.body)
        self.assertIn("Description text.", doc.body)
        self.assertNotIn("Your current environment", doc.body)

    def test_no_environment_unchanged(self):
        doc = self.puller._to_doc(make_item(body="plain description"), [])
        self.assertEqual(doc.body, "plain description")
        self.assertNotIn("environment", doc.extra)

    def test_pr_mapping(self):
        """PR 应映射为 github_pr，merged 状态与合并信息进入 extra。"""
        item = make_item(number=99, body="PR body")
        item["pull_request"] = {
            "url": "https://api.github.com/repos/vllm-project/vllm/pulls/99",
            "merged_at": "2026-08-01T00:00:00Z",
            "merged": True,
            "merge_commit_sha": "abc123def",
        }
        doc = self.puller._to_doc(item, [])
        self.assertEqual(doc.source_type, "github_pr")
        self.assertEqual(doc.source_id, "github:vllm-project-vllm:pr:99")
        self.assertEqual(doc.status, "merged")
        self.assertEqual(doc.resolved_at, "2026-08-01T00:00:00Z")
        self.assertEqual(doc.extra.get("merge_commit_sha"), "abc123def")
        self.assertTrue(doc.extra.get("merged"))

    def test_unmerged_closed_pr(self):
        item = make_item(number=100, state="closed", closed_at="2026-08-02T00:00:00Z")
        item["pull_request"] = {"merged": False, "merged_at": None, "merge_commit_sha": None}
        doc = self.puller._to_doc(item, [])
        self.assertEqual(doc.source_type, "github_pr")
        self.assertEqual(doc.status, "closed")
        self.assertEqual(doc.resolved_at, "2026-08-02T00:00:00Z")

    def test_source_id_namespaced_by_repo(self):
        """不同仓库的 issue 编号不冲突。"""
        p1 = GithubPuller(make_source(repo="vllm-project/vllm"), PROJECT_ROOT)
        p2 = GithubPuller(make_source(repo="vllm-project/vllm-ascend"), PROJECT_ROOT)
        d1 = p1._to_doc(make_item(number=7), [])
        d2 = p2._to_doc(make_item(number=7), [])
        self.assertNotEqual(d1.source_id, d2.source_id)
        self.assertTrue(d1.source_id.startswith("github:vllm-project-vllm:issue:"))
        self.assertTrue(d2.source_id.startswith("github:vllm-project-vllm-ascend:issue:"))

    def test_kind_from_title(self):
        """标题前缀识别 issue 类型（故障库中 bug/fix 权威，doc/feature/rfc 反馈类降权）。"""
        cases = {
            "[Bug]: GLM5.1 PD 分离挂死": "bug",
            "[Fix]: release blocks on OOM": "fix",
            "[Doc]: GLM5.2 文档错误": "doc",
            "[Feature]: support fp8": "feature",
            "[RFC]: tiered offload": "rfc",
            "普通标题": "other",
        }
        for title, expected in cases.items():
            doc = self.puller._to_doc(make_item(title=title, body="x"), [])
            self.assertEqual(doc.extra.get("kind"), expected, title)

    def test_item_from_graphql_node(self):
        """GraphQL 节点 -> REST 形状 item 映射（_to_doc 可直接消费）。"""
        node = {
            "number": 42, "title": "[Bug]: x", "body": "body text",
            "state": "CLOSED", "createdAt": "2026-01-01T00:00:00Z",
            "closedAt": "2026-01-02T00:00:00Z", "url": "https://github.com/x/issues/42",
            "author": {"login": "alice"},
            "labels": {"nodes": [{"name": "bug"}, {"name": "v0.6.x"}]},
        }
        item = GithubPuller._item_from_node(node, is_pr=False)
        self.assertEqual(item["number"], 42)
        self.assertEqual(item["state"], "closed")
        self.assertEqual(item["labels"], [{"name": "bug"}, {"name": "v0.6.x"}])
        self.assertNotIn("pull_request", item)
        doc = GithubPuller(make_source(), PROJECT_ROOT)._to_doc(item, [])
        self.assertEqual(doc.source_id, "github:vllm-project-vllm:issue:42")
        self.assertEqual(doc.status, "closed")

    def test_item_from_graphql_pr_node(self):
        node = {
            "number": 7, "title": "[Fix]: y", "body": "fix",
            "state": "MERGED", "createdAt": "2026-01-01T00:00:00Z",
            "closedAt": "2026-01-03T00:00:00Z", "url": "https://github.com/x/pull/7",
            "author": {"login": "bob"}, "labels": {"nodes": []},
            "merged": True, "mergedAt": "2026-01-03T00:00:00Z",
            "mergeCommit": {"oid": "abc123def"},
        }
        item = GithubPuller._item_from_node(node, is_pr=True)
        self.assertTrue(item["pull_request"]["merged"])
        self.assertEqual(item["pull_request"]["merge_commit_sha"], "abc123def")
        doc = GithubPuller(make_source(), PROJECT_ROOT)._to_doc(item, [])
        self.assertEqual(doc.status, "merged")
        self.assertEqual(doc.extra.get("merge_commit_sha"), "abc123def")
        # 旧 raw（无 head 字段）：extra 不含 head 键
        self.assertNotIn("head_repo", doc.extra)

    def test_item_from_graphql_pr_node_head_metadata(self):
        """PR head 三元组：fork 仓/分支/锁定 commit（commits.last=合并时 head，不随分支漂移）。"""
        node = {
            "number": 9, "title": "[Fix]: fork work", "body": "x",
            "state": "MERGED", "createdAt": "2026-02-01T00:00:00Z",
            "closedAt": "2026-02-03T00:00:00Z", "url": "https://github.com/x/pull/9",
            "author": {"login": "carol"}, "labels": {"nodes": []},
            "merged": True, "mergedAt": "2026-02-03T00:00:00Z",
            "mergeCommit": {"oid": "m" * 40},
            "headRefName": "dev_hy4",
            "headRepository": {"nameWithOwner": "voidvelocity/vllm"},
            "commits": {"nodes": [{"commit": {"oid": "h" * 40}}]},
        }
        item = GithubPuller._item_from_node(node, is_pr=True)
        pr = item["pull_request"]
        self.assertEqual(pr["head_repo"], "voidvelocity/vllm")
        self.assertEqual(pr["head_branch"], "dev_hy4")
        self.assertEqual(pr["head_sha"], "h" * 40)
        doc = GithubPuller(make_source(), PROJECT_ROOT)._to_doc(item, [])
        self.assertEqual(doc.extra.get("head_repo"), "voidvelocity/vllm")
        self.assertEqual(doc.extra.get("head_branch"), "dev_hy4")
        self.assertEqual(doc.extra.get("head_sha"), "h" * 40)

    def test_item_from_graphql_pr_node_no_commits(self):
        # commits 节点缺失（查询裁剪/异常形态）：head_sha 为 None，不抛异常
        node = {
            "number": 11, "title": "t", "body": "b", "state": "OPEN",
            "createdAt": "2026-03-01T00:00:00Z", "closedAt": None,
            "url": "https://github.com/x/pull/11", "author": {"login": "d"},
            "labels": {"nodes": []}, "merged": False, "mergedAt": None,
            "headRefName": "feature",
        }
        item = GithubPuller._item_from_node(node, is_pr=True)
        self.assertIsNone(item["pull_request"]["head_sha"])
        self.assertEqual(item["pull_request"]["head_branch"], "feature")

    def test_item_from_node_records_updated_at(self):
        """GraphQL 节点落盘 updated_at（状态同步锚点），_to_doc 透传到 doc.updated_at。"""
        node = {
            "number": 42, "title": "[Bug]: x", "body": "body text",
            "state": "CLOSED", "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-05T00:00:00Z",
            "closedAt": "2026-01-02T00:00:00Z", "url": "https://github.com/x/issues/42",
            "author": {"login": "alice"},
            "labels": {"nodes": [{"name": "bug"}]},
        }
        item = GithubPuller._item_from_node(node, is_pr=False)
        self.assertEqual(item["updated_at"], "2026-01-05T00:00:00Z")
        doc = GithubPuller(make_source(), PROJECT_ROOT)._to_doc(item, [])
        self.assertEqual(doc.updated_at, "2026-01-05T00:00:00Z")
        # 节点缺 updatedAt（老查询/异常形态）：落 None，不报错
        item2 = GithubPuller._item_from_node(_node(3), is_pr=False)
        self.assertIsNone(item2["updated_at"])

    def test_item_from_rest_records_updated_at(self):
        """REST 响应透传 updated_at（--numbers 单条补拉的 checkpoint 锚点）。"""
        data = {
            "number": 77, "title": "issue 77", "body": "b", "state": "open",
            "created_at": "2026-05-01T00:00:00Z",
            "updated_at": "2026-05-03T00:00:00Z",
            "closed_at": None,
            "html_url": "https://github.com/vllm-project/vllm/issues/77",
            "labels": [], "user": {"login": "bob"},
        }
        item = GithubPuller._item_from_rest(data, is_pr=False)
        self.assertEqual(item["updated_at"], "2026-05-03T00:00:00Z")
        # REST 响应缺 updated_at：落 None
        data2 = dict(data)
        del data2["updated_at"]
        self.assertIsNone(GithubPuller._item_from_rest(data2, is_pr=False)["updated_at"])

    def test_graphql_states_mapping(self):
        from vllm_kb.github_pull import GithubPuller as GP

        self.assertEqual(GP._graphql_states("all", False), ["OPEN", "CLOSED"])
        self.assertEqual(GP._graphql_states("all", True), ["OPEN", "CLOSED", "MERGED"])
        self.assertEqual(GP._graphql_states("open", False), ["OPEN"])
        self.assertEqual(GP._graphql_states("closed", True), ["CLOSED", "MERGED"])

    def test_graphql_cursor_advances(self):
        """游标必须逐页前进（回归：曾因局部变量未更新而无限重复第一页）。"""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = make_source(repo="vllm-project/vllm-ascend", raw_dir=str(Path(td) / "raw"),
                              fetch_comments=False)
            puller = GithubPuller(src, PROJECT_ROOT)
            after_values: list = []

            def fake_request(query, variables):
                after_values.append(variables["after"])
                if variables["after"] is None:
                    return {"issues": {"totalCount": 250,
                                       "pageInfo": {"hasNextPage": True, "endCursor": "C1"},
                                       "nodes": [_node(1), _node(2)]}}
                return {"issues": {"totalCount": 250,
                                   "pageInfo": {"hasNextPage": False, "endCursor": "C2"},
                                   "nodes": [_node(3)]}}

            puller._graphql_request = fake_request
            cp = {"issues": {}}
            g = {}
            new_count, refresh_count, skip_count = puller._collect_graphql(cp, g, kind="issues")
            self.assertEqual(new_count, 3)
            self.assertEqual(refresh_count, 0)
            self.assertEqual(skip_count, 0)
            self.assertEqual(after_values, [None, "C1"])  # 第二次请求必须带上第一页的 endCursor
            self.assertTrue(g["issues_done"])
            self.assertEqual(sorted(cp["issues"]), ["1", "2", "3"])

    def test_graphql_stalled_cursor_retries_then_skips(self):
        """endCursor 重复：先重试 STALL_RETRIES 次；仍重复则跳过该 kind（不中断、不报错、不无限循环）。"""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = make_source(repo="vllm-project/vllm-ascend", raw_dir=str(Path(td) / "raw"),
                              fetch_comments=False)
            puller = GithubPuller(src, PROJECT_ROOT)
            requests = {"n": 0}

            def stuck_request(query, variables):
                requests["n"] += 1
                return {"issues": {"totalCount": 250,
                                   "pageInfo": {"hasNextPage": True, "endCursor": "C1"},
                                   "nodes": [_node(1)]}}

            puller._graphql_request = stuck_request
            cp = {"issues": {}}
            g = {}
            new_count, refresh_count, skip_count = puller._collect_graphql(cp, g, kind="issues")
            # 首页已收 1 条；此后 endCursor 恒为 C1：首页 1 次 + 检测到重复 1 次 +
            # 重试 STALL_RETRIES 次后跳过，不再发请求、不抛异常、kind 不置 done（下次续传）。
            self.assertEqual(new_count, 1)
            self.assertEqual(refresh_count, 0)
            self.assertEqual(requests["n"], 2 + GithubPuller.STALL_RETRIES)
            self.assertFalse(g.get("issues_done"))
            self.assertEqual(g.get("issues_cursor"), "C1")  # 停在本游标，续传不重复

    def test_graphql_stalled_cursor_recovers_on_retry(self):
        """endCursor 重复但重试后前进：正常继续（活跃仓库窗口滑动自愈，不误跳过）。"""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = make_source(repo="vllm-project/vllm-ascend", raw_dir=str(Path(td) / "raw"),
                              fetch_comments=False)
            puller = GithubPuller(src, PROJECT_ROOT)
            requests = {"n": 0}

            def flaky_request(query, variables):
                requests["n"] += 1
                n = requests["n"]
                if n == 1:  # 首页
                    return {"issues": {"totalCount": 250,
                                       "pageInfo": {"hasNextPage": True, "endCursor": "C1"},
                                       "nodes": [_node(1), _node(2)]}}
                if n == 2:  # 窗口滑动：返回与上一页相同的 endCursor
                    return {"issues": {"totalCount": 250,
                                       "pageInfo": {"hasNextPage": True, "endCursor": "C1"},
                                       "nodes": [_node(1)]}}
                # 重试后恢复前进
                return {"issues": {"totalCount": 250,
                                   "pageInfo": {"hasNextPage": False, "endCursor": "C2"},
                                   "nodes": [_node(3)]}}

            puller._graphql_request = flaky_request
            cp = {"issues": {}}
            g = {}
            new_count, refresh_count, skip_count = puller._collect_graphql(cp, g, kind="issues")
            self.assertEqual(new_count, 3)  # 1、2、3 全部入库
            self.assertEqual(requests["n"], 3)  # 首页 + 滑动页 + 重试恢复页
            self.assertTrue(g["issues_done"])
            self.assertEqual(g["issues_cursor"], "C2")
            self.assertEqual(sorted(cp["issues"]), ["1", "2", "3"])

    def test_recanonicalize_from_raw(self):
        """从原始 JSON 重新生成 canonical（含 PR），无需重新拉取。"""
        import json
        import tempfile

        from vllm_kb.github_pull import recanonicalize

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = make_source(raw_dir=str(root / "raw"))
            raw = root / "raw"
            issues_dir = raw / "issues"
            prs_dir = raw / "prs"
            comments_dir = raw / "comments"
            issues_dir.mkdir(parents=True)
            prs_dir.mkdir()
            comments_dir.mkdir()
            item = make_item(number=7, body="### Your current environment\n- **vLLM version**: 0.26.0")
            (issues_dir / "7.json").write_text(json.dumps(item), encoding="utf-8")
            pr_item = make_item(number=8, body="fixes the bug")
            pr_item["pull_request"] = {"merged": True, "merged_at": "2026-08-03T00:00:00Z", "merge_commit_sha": "zzz"}
            (prs_dir / "8.json").write_text(json.dumps(pr_item), encoding="utf-8")
            comments = [{"user": {"login": "a"}, "created_at": "2026-07-02T00:00:00Z", "body": "me too"}]
            (comments_dir / "7.json").write_text(json.dumps(comments), encoding="utf-8")

            docs = recanonicalize(src, project_root=root)
            self.assertEqual(len(docs), 2)
            by_id = {d.source_id: d for d in docs}
            issue = by_id["github:vllm-project-vllm:issue:7"]
            self.assertEqual(issue.version_span.min, "0.26.0")  # 新版提取逻辑生效
            self.assertIn("me too", issue.body)
            pr = by_id["github:vllm-project-vllm:pr:8"]
            self.assertEqual(pr.status, "merged")

    def test_comments_inlined_from_graphql_node(self):
        """fetch_comments 时评论从 GraphQL 节点内联提取，形状与 REST 评论一致。"""
        node = _node(9)
        node["comments"] = {
            "pageInfo": {"hasNextPage": False},
            "nodes": [
                {"author": {"login": "alice"}, "createdAt": "2026-01-02T00:00:00Z", "body": "me too"},
                {"author": None, "createdAt": None, "body": "anon"},
            ],
        }
        comments = GithubPuller._comments_from_node(node)
        self.assertEqual(len(comments), 2)
        self.assertEqual(comments[0]["user"]["login"], "alice")
        self.assertEqual(comments[0]["created_at"], "2026-01-02T00:00:00Z")
        self.assertEqual(comments[0]["body"], "me too")
        self.assertEqual(comments[1]["user"]["login"], "unknown")

    def test_comments_inlined_missing_node(self):
        """无 comments 字段的节点（fetch_comments=False 的查询）返回空列表，不报错。"""
        self.assertEqual(GithubPuller._comments_from_node(_node(10)), [])

    def test_graphql_collect_inlines_comments(self):
        """_collect_graphql 把节点内联评论落盘 comments/{n}.json（不再走 REST）。"""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = make_source(raw_dir=str(Path(td) / "raw"), fetch_comments=True)
            puller = GithubPuller(src, PROJECT_ROOT)
            rest_calls: list = []

            def fake_request(query, variables):
                node = _node(11)
                node["comments"] = {
                    "pageInfo": {"hasNextPage": False},
                    "nodes": [{"author": {"login": "bob"}, "createdAt": "2026-01-03T00:00:00Z", "body": "same here"}],
                }
                return {"issues": {"totalCount": 1,
                                   "pageInfo": {"hasNextPage": False, "endCursor": "E1"},
                                   "nodes": [node]}}

            puller._graphql_request = fake_request
            puller._get_comments = lambda number: rest_calls.append(number) or []
            puller._get_timeline = lambda number: []  # timeline 采集不在本测范围（避免触网）
            cp = {"issues": {}}
            g = {}
            new_count, refresh_count, skip_count = puller._collect_graphql(cp, g, kind="issues")
            self.assertEqual(new_count, 1)
            self.assertEqual(refresh_count, 0)
            self.assertEqual(rest_calls, [])  # 评论全量走 GraphQL，REST 兜底零调用
            comments = json.loads((Path(td) / "raw" / "comments" / "11.json").read_text(encoding="utf-8"))
            self.assertEqual(comments[0]["user"]["login"], "bob")
            self.assertEqual(comments[0]["body"], "same here")

    def test_graphql_rate_limit_error_waits_and_retries(self):
        """GraphQL 主限流：HTTP 200 + body 报错 + remaining=0 时必须等待 reset 重试，而非抛异常中止。"""
        import tempfile
        import time as _time
        from unittest import mock

        with tempfile.TemporaryDirectory() as td:
            src = make_source(raw_dir=str(Path(td) / "raw"), fetch_comments=False)
            puller = GithubPuller(src, PROJECT_ROOT)
            responses = iter([
                # 第一次：主限流错误（200 + errors + remaining=0）
                mock.Mock(
                    status_code=200,
                    headers={"X-RateLimit-Reset": str(int(_time.time()) + 5), "X-RateLimit-Remaining": "0"},
                    json=lambda: {"errors": [{"message": "API rate limit exceeded for user."}]},
                ),
                # 第二次：成功
                mock.Mock(
                    status_code=200,
                    headers={},
                    json=lambda: {"data": {"repository": {"issues": {"totalCount": 0, "pageInfo": {"hasNextPage": False}, "nodes": []}}}},
                ),
            ])
            puller.session = mock.Mock()
            puller.session.request = mock.Mock(side_effect=lambda *a, **k: next(responses))
            sleeps: list = []
            with mock.patch.object(_time, "sleep", side_effect=lambda s: sleeps.append(s)):
                result = puller._graphql_request("q", {})
            self.assertEqual(result["issues"]["totalCount"], 0)
            self.assertTrue(any(s >= 4 for s in sleeps), f"应按 reset 等待，实际 {sleeps}")

    def test_sleep_for_rate_limit_prefers_retry_after(self):
        """限流等待：Retry-After 优先于 X-RateLimit-Reset；无 header 时指数退避。"""
        import time as _time
        from unittest import mock

        src = make_source()
        puller = GithubPuller(src, PROJECT_ROOT)

        # Retry-After 优先
        r1 = mock.Mock(headers={"Retry-After": "7", "X-RateLimit-Reset": str(int(_time.time()) + 500)})
        sleeps: list = []
        with mock.patch.object(_time, "sleep", side_effect=lambda s: sleeps.append(s)):
            puller._sleep_for_rate_limit(r1, tag="REST")
        self.assertEqual(sleeps, [7])

        # 只有 reset：精确等到 reset
        r2 = mock.Mock(headers={"X-RateLimit-Reset": str(int(_time.time()) + 120)})
        sleeps.clear()
        with mock.patch.object(_time, "sleep", side_effect=lambda s: sleeps.append(s)):
            puller._sleep_for_rate_limit(r2, tag="REST")
        self.assertTrue(118 <= sleeps[0] <= 121)

        # 无任何 header：指数退避 attempt 递增
        r3 = mock.Mock(headers={})
        sleeps.clear()
        with mock.patch.object(_time, "sleep", side_effect=lambda s: sleeps.append(s)):
            puller._sleep_for_rate_limit(r3, tag="REST", attempt=2)
        self.assertEqual(sleeps, [puller.retry_backoff_seconds * 3])


class TestTransportResilience(unittest.TestCase):
    """传输类故障（连接/超时/TLS 隧道被掐）的处置——全量部署的实际故障现场。

    背景：外网 + 本地代理环境下 `SSLEOFError: EOF occurred in violation of protocol`
    在 vllm 全量 PR 枚举中途发生，旧实现只有 4 次静默重试、复用同一个连接池，
    耗尽后抛裸 RuntimeError 并让整个部署中止。
    """

    def _puller(self, **overrides):
        from vllm_kb.github_pull import GithubPuller

        overrides.setdefault("retry_backoff_seconds", 0)
        src = make_source(token="fake-token", **overrides)
        return GithubPuller(src, PROJECT_ROOT)

    @staticmethod
    def _sock_err():
        """构造与现场一致的 SSLEOFError（requests 会把它包成 SSLError）。"""
        import ssl as _ssl

        return requests.exceptions.SSLError(
            _ssl.SSLEOFError(8, "EOF occurred in violation of protocol (_ssl.c:1007)"))

    def test_transport_retry_is_independent_and_typed(self):
        """传输异常按 transport_retries 独立预算重试，耗尽抛 GithubUnreachable（可降级）。"""
        import time as _time
        from unittest import mock

        from vllm_kb.github_pull import GithubUnreachable

        puller = self._puller(transport_retries=3)
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise self._sock_err()

        with mock.patch.object(puller.session, "request", side_effect=boom):
            with mock.patch.object(_time, "sleep"):
                with self.assertRaises(GithubUnreachable) as ctx:
                    puller._graphql_request("q", {})
        # 尝试次数 = transport_retries + 1（与 max_retries 无关）
        self.assertEqual(len(calls), 4)
        msg = str(ctx.exception)
        self.assertIn("SSLError", msg)
        self.assertIn("已重试 3 次", msg)
        self.assertNotIn("None", msg)  # 旧实现耗尽时打的是 last_exc=None

    def test_transport_retries_exceed_status_budget(self):
        """传输预算独立于 max_retries：max_retries=1 也要给传输故障足够次数。"""
        import time as _time
        from unittest import mock

        from vllm_kb.github_pull import GithubUnreachable

        puller = self._puller(max_retries=1, transport_retries=5)
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise self._sock_err()

        with mock.patch.object(puller.session, "request", side_effect=boom):
            with mock.patch.object(_time, "sleep"):
                with self.assertRaises(GithubUnreachable):
                    puller._graphql_request("q", {})
        self.assertEqual(len(calls), 6)

    def test_transport_retry_recovers_and_drops_pool(self):
        """中途抖动后恢复：换新连接（session.close 被调用）且最终返回数据。"""
        import time as _time
        from unittest import mock

        puller = self._puller(transport_retries=3)
        ok = mock.Mock(status_code=200, headers={},
                       json=lambda: {"data": {"repository": {"issues": {"totalCount": 7}}}})
        seq = [self._sock_err(), self._sock_err(), ok]
        closes = []

        def flaky(*a, **k):
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with mock.patch.object(puller.session, "request", side_effect=flaky):
            with mock.patch.object(puller.session, "close", side_effect=lambda: closes.append(1)):
                with mock.patch.object(_time, "sleep"):
                    result = puller._graphql_request("q", {})
        self.assertEqual(result["issues"]["totalCount"], 7)
        self.assertEqual(len(closes), 2, "每次重试都应丢弃连接池换新连接")

    def test_retry_is_logged(self):
        """重试必须留痕：旧实现对传输异常完全静默（部署日志看不出重试过）。"""
        import io
        import contextlib
        import time as _time
        from unittest import mock

        from vllm_kb.github_pull import GithubUnreachable

        puller = self._puller(transport_retries=2)
        buf = io.StringIO()

        def boom(*a, **k):
            raise self._sock_err()

        with mock.patch.object(puller.session, "request", side_effect=boom):
            with mock.patch.object(puller.session, "close"):
                with mock.patch.object(_time, "sleep"):
                    with contextlib.redirect_stdout(buf):
                        with self.assertRaises(GithubUnreachable):
                            puller._graphql_request("q", {})
        out = buf.getvalue()
        self.assertIn("网络异常", out)
        self.assertIn("换新连接重试", out)
        self.assertIn("SSLEOFError", out)  # 根因类型要打出来

    def test_permanent_4xx_fails_fast(self):
        """401（token 失效）是永久错误：不重试、信息明确（重试无用）。"""
        from unittest import mock

        puller = self._puller(transport_retries=3, max_retries=3)
        calls = []
        r401 = mock.Mock(status_code=401, headers={}, text='{"message":"Bad credentials"}')
        with mock.patch.object(puller.session, "request",
                               side_effect=lambda *a, **k: calls.append(1) or r401):
            with self.assertRaises(RuntimeError) as ctx:
                puller._graphql_request("q", {})
        self.assertEqual(len(calls), 1, "永久错误不应重试")
        self.assertIn("401", str(ctx.exception))
        self.assertIn("重试无用", str(ctx.exception))

    def test_status_exhaustion_message_not_none(self):
        """5xx/限流耗尽时，报错必须说明原因（旧实现是 `GraphQL 重试失败: None`）。"""
        import time as _time
        from unittest import mock

        from vllm_kb.github_pull import GithubUnreachable

        puller = self._puller(max_retries=2)
        r500 = mock.Mock(status_code=500, headers={}, text="")
        with mock.patch.object(puller.session, "request", return_value=r500):
            with mock.patch.object(_time, "sleep"):
                with self.assertRaises(GithubUnreachable) as ctx:
                    puller._graphql_request("q", {})
        self.assertIn("HTTP 500", str(ctx.exception))
        self.assertIn("已尝试 3 次", str(ctx.exception))

    def test_body_read_failure_is_retried(self):
        """读 body 中途断流（ChunkedEncodingError）也必须重试——否则大页断流无处兜。"""
        import time as _time
        from unittest import mock

        puller = self._puller(transport_retries=2)
        r = mock.Mock(status_code=200, headers={})
        r.json = mock.Mock(side_effect=requests.exceptions.ChunkedEncodingError("断流"))
        ok = mock.Mock(status_code=200, headers={},
                       json=lambda: {"data": {"repository": {"issues": {"totalCount": 1}}}})
        seq = [(r, None), (ok, None)]
        with mock.patch.object(puller.session, "request", side_effect=lambda *a, **k: seq.pop(0)[0]):
            with mock.patch.object(puller.session, "close"):
                with mock.patch.object(_time, "sleep"):
                    result = puller._graphql_request("q", {})
        self.assertEqual(result["issues"]["totalCount"], 1)

    def test_rest_get_keeps_httperror_contract(self):
        """`_get` 的永久 4xx 必须仍抛 requests.HTTPError（带 .response）。

        `_pull_numbers`（`--numbers`）用它按 status_code 分支：404 → 回退
        `/pulls/{n}` → `/issues/{n}`。若改成别的异常类型，回退会**静默失效**；
        而既有测试都 mock 掉了 `_get`，所以必须直接钉住 `_get` 本身。
        """
        from unittest import mock

        puller = self._puller()
        r404 = mock.Mock(status_code=404, headers={}, text="not found")
        r404.raise_for_status = mock.Mock(
            side_effect=requests.HTTPError("404 Client Error", response=r404))
        with mock.patch.object(puller.session, "request", return_value=r404):
            with self.assertRaises(requests.HTTPError) as ctx:
                puller._get("https://api.github.com/repos/o/r/pulls/1")
        self.assertEqual(ctx.exception.response.status_code, 404)

    def test_rest_get_permanent_error_not_retried(self):
        """永久 4xx 不重试（省时间），限流/5xx 才重试。"""
        from unittest import mock

        puller = self._puller(max_retries=3)
        calls = []
        r404 = mock.Mock(status_code=404, headers={}, text="nf")
        r404.raise_for_status = mock.Mock(
            side_effect=requests.HTTPError("404", response=r404))
        with mock.patch.object(puller.session, "request",
                               side_effect=lambda *a, **k: calls.append(1) or r404):
            with self.assertRaises(requests.HTTPError):
                puller._get("https://api.github.com/x")
        self.assertEqual(len(calls), 1)


class TestPageSizeConfig(unittest.TestCase):
    """GraphQL 分页大小可配（弱网调小以缩小单次传输）。"""

    def test_defaults_are_100(self):
        from vllm_kb.github_pull import GithubPuller

        puller = GithubPuller(make_source(token="t"), PROJECT_ROOT)
        self.assertEqual(puller.page_size, 100)
        self.assertEqual(puller.comments_page_size, 100)

    def test_query_uses_variables_not_literals(self):
        """查询里不能再写死 first: 100（否则配置无效）。"""
        from vllm_kb.github_pull import GithubPuller

        for q in (GithubPuller._ISSUES_QUERY, GithubPuller._PRS_QUERY):
            self.assertIn("$first: Int!", q)
            self.assertIn("$commentsFirst: Int!", q)
            self.assertNotIn("first: 100", q)
            self.assertNotIn("comments(first: 100", q)

    def test_configured_page_size_reaches_variables(self):
        import tempfile
        from unittest import mock

        from vllm_kb.github_pull import GithubPuller

        with tempfile.TemporaryDirectory() as td:
            src = make_source(token="t", graphql_page_size=25, comments_page_size=10,
                              fetch_comments=True, raw_dir=f"{td}/raw",
                              checkpoint_file=f"{td}/cp.json")
            puller = GithubPuller(src, PROJECT_ROOT)
            self.assertEqual((puller.page_size, puller.comments_page_size), (25, 10))
            seen = {}

            def fake_request(query, variables):
                seen.update(variables)
                return {"issues": {"totalCount": 0, "pageInfo": {"hasNextPage": False}, "nodes": []}}

            puller._graphql_request = fake_request
            puller._collect_graphql({"issues": {}}, {}, kind="issues", incremental=False)
            self.assertEqual(seen["first"], 25)
            self.assertEqual(seen["commentsFirst"], 10)

    def test_page_size_clamped_to_github_max(self):
        from vllm_kb.github_pull import GithubPuller

        puller = GithubPuller(make_source(token="t", graphql_page_size=500,
                                          comments_page_size=0), PROJECT_ROOT)
        self.assertEqual(puller.page_size, 100)
        self.assertEqual(puller.comments_page_size, 1)


class TestPullIncremental(unittest.TestCase):
    """pull 的 done 跳过 / --incremental 增量拉取（mock GraphQL，不触网）。"""

    def _make_puller(self, td: str, issues_done: bool = True):
        from vllm_kb.github_pull import GithubPuller

        src = make_source(token="fake-token",
                          raw_dir=f"{td}/raw", checkpoint_file=f"{td}/cp.json",
                          fetch_comments=False, fetch_timeline=False)
        puller = GithubPuller(src, PROJECT_ROOT)
        puller._save_checkpoint({
            "issues": {"1": {"fetched_at": "x", "comments": False}},
            "state": "all", "sort": "created", "direction": "desc",
            "graphql": {"issues_done": issues_done, "prs_done": issues_done},
        })
        return puller

    def test_pull_done_skips_with_message(self):
        """done=True 默认跳过拉取（打印说明），不发起 GraphQL 请求。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            calls = []
            puller._graphql_request = lambda *a, **k: calls.append(a) or {}
            out = StringIO()
            with redirect_stdout(out):
                n = puller.pull()
            self.assertEqual(n, 0)
            self.assertEqual(calls, [])  # 未发起任何请求
            self.assertIn("跳过", out.getvalue())
            self.assertIn("--incremental", out.getvalue())

    def test_pull_incremental_fetches_new_and_stops(self):
        """--incremental：无历史窗口时从头拉，新编号落盘、已有跳过、连续 3 页无新增停止。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            issues_pages = iter([
                # issues 第 1 页：新 issue 2、3（created desc 在前）+ 已有 1 跳过
                {"issues": {"totalCount": 100,
                            "pageInfo": {"hasNextPage": True, "endCursor": "C1"},
                            "nodes": [_node(3), _node(2), _node(1)]}},
                # issues 第 2/3/4 页：全是已有 1 → 连续无新增提前停止
                {"issues": {"totalCount": 100,
                            "pageInfo": {"hasNextPage": True, "endCursor": "C2"},
                            "nodes": [_node(1)]}},
                {"issues": {"totalCount": 100,
                            "pageInfo": {"hasNextPage": True, "endCursor": "C3"},
                            "nodes": [_node(1)]}},
                {"issues": {"totalCount": 100,
                            "pageInfo": {"hasNextPage": True, "endCursor": "C4"},
                            "nodes": [_node(1)]}},
            ])
            prs_pages = iter([
                # prs 增量：空页（无新增，立即结束）
                {"pullRequests": {"totalCount": 0,
                                  "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                  "nodes": []}},
            ])

            def fake_request(query, variables):
                # 无历史窗口：issues 不带 filter（since=None）；增量排序 UPDATED_AT DESC
                if "issues" in query:
                    self.assertIsNone(variables["filter"])
                self.assertEqual(variables["order"],
                                 {"field": "UPDATED_AT", "direction": "DESC"})
                return next(issues_pages) if "issues" in query else next(prs_pages)

            puller._graphql_request = fake_request
            with redirect_stdout(StringIO()):
                n = puller.pull(incremental=True)
            self.assertEqual(n, 2)  # 新增 2、3
            self.assertTrue((Path(td) / "raw" / "issues" / "2.json").exists())
            self.assertTrue((Path(td) / "raw" / "issues" / "3.json").exists())
            cp = puller._load_checkpoint()
            self.assertIn("2", cp["issues"])
            self.assertIn("3", cp["issues"])
            self.assertTrue(cp["graphql"]["issues_done"])  # done 状态保持
            # 窗口推进：记录本轮 max createdAt（_node 固定 createdAt=2026-01-01T00:00:00Z）
            self.assertEqual(cp["graphql"].get("issues_since"), "2026-01-01T00:00:00Z")

    def test_pull_incremental_uses_time_window_since(self):
        """--incremental 且已有 since：issues 带 filterBy.since 服务端过滤，PR 不带。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            cp = puller._load_checkpoint()
            cp["graphql"]["issues_since"] = "2026-01-15T00:00:00Z"
            cp["graphql"]["prs_since"] = "2026-01-20T00:00:00Z"
            puller._save_checkpoint(cp)
            seen = {"issues": [], "prs": []}

            def fake_request(query, variables):
                if "issues" in query:
                    seen["issues"].append(variables)
                    return {"issues": {"totalCount": 0,
                                       "pageInfo": {"hasNextPage": False, "endCursor": "I1"},
                                       "nodes": []}}
                seen["prs"].append(variables)
                return {"pullRequests": {"totalCount": 0,
                                         "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                         "nodes": []}}

            puller._graphql_request = fake_request
            with redirect_stdout(StringIO()):
                n = puller.pull(incremental=True)
            self.assertEqual(n, 0)
            # issues：filterBy.since 生效（服务端时间窗口）
            self.assertEqual(seen["issues"][0]["filter"], {"since": "2026-01-15T00:00:00Z"})
            self.assertEqual(seen["issues"][0]["order"],
                             {"field": "UPDATED_AT", "direction": "DESC"})
            # PR：连接无 filterBy 参数（不传 filter），排序 UPDATED_AT DESC
            self.assertNotIn("filter", seen["prs"][0])
            self.assertEqual(seen["prs"][0]["order"],
                             {"field": "UPDATED_AT", "direction": "DESC"})

    def test_pull_incremental_window_advances_on_normal_completion(self):
        """Bug B：已有 since 且正常翻完（hasNextPage=false）时窗口必须推进到本轮 max createdAt。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            cp = puller._load_checkpoint()
            cp["graphql"]["issues_since"] = "2026-01-10T00:00:00Z"
            puller._save_checkpoint(cp)

            def fake_request(query, variables):
                if "issues" in query:
                    return {"issues": {"totalCount": 5,
                                       "pageInfo": {"hasNextPage": False, "endCursor": "I1"},
                                       # 新条目（createdAt 2026-02-01 > since 2026-01-10）
                                       "nodes": [_node(99, created_at="2026-02-01T00:00:00Z")]}}
                return {"pullRequests": {"totalCount": 0,
                                         "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                         "nodes": []}}

            puller._graphql_request = fake_request
            with redirect_stdout(StringIO()):
                n = puller.pull(incremental=True)
            self.assertEqual(n, 1)
            cp2 = puller._load_checkpoint()
            # 窗口推进：max(旧 since, 本轮 max createdAt) = 2026-02-01
            self.assertEqual(cp2["graphql"]["issues_since"], "2026-02-01T00:00:00Z")

    def test_pull_incremental_window_never_regresses(self):
        """窗口只前进不后退：窗口内全是被更新的旧条目（createdAt < since）时 since 不变。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            cp = puller._load_checkpoint()
            cp["graphql"]["issues_since"] = "2026-01-10T00:00:00Z"
            puller._save_checkpoint(cp)

            def fake_request(query, variables):
                if "issues" in query:
                    return {"issues": {"totalCount": 5,
                                       "pageInfo": {"hasNextPage": False, "endCursor": "I1"},
                                       # 只有 createdAt 比 since 更早的条目（被更新过，进来跳过）
                                       "nodes": [_node(5, created_at="2026-01-01T00:00:00Z")]}}
                return {"pullRequests": {"totalCount": 0,
                                         "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                         "nodes": []}}

            puller._graphql_request = fake_request
            with redirect_stdout(StringIO()):
                n = puller.pull(incremental=True)
            self.assertEqual(n, 1)  # 5 号是新的（cp 里没有），仍入库
            cp2 = puller._load_checkpoint()
            # 本轮 max createdAt（2026-01-01）< 旧 since（2026-01-10）→ 窗口不能回退
            self.assertEqual(cp2["graphql"]["issues_since"], "2026-01-10T00:00:00Z")

    def test_pull_incremental_resets_stale_last_cursor(self):
        """Bug A：增量从头拉时清掉残留 last_cursor——否则第一页 endCursor 恰好等于
        上次中断时的游标会被误判"卡住"整轮放弃（第一页不触发停滞检测）。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            cp = puller._load_checkpoint()
            # 模拟上次断点续传中断：last_cursor 恰好等于本次第一页 endCursor
            cp["graphql"]["issues_last_cursor"] = "C1"
            puller._save_checkpoint(cp)
            calls = []

            def fake_request(query, variables):
                if "issues" in query:
                    calls.append(1)
                    return {"issues": {"totalCount": 10,
                                       "pageInfo": {"hasNextPage": False, "endCursor": "C1"},
                                       "nodes": [_node(7)]}}
                return {"pullRequests": {"totalCount": 0,
                                         "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                         "nodes": []}}

            puller._graphql_request = fake_request
            with redirect_stdout(StringIO()):
                n = puller.pull(incremental=True)
            self.assertEqual(n, 1)  # 7 号正常入库，未被误判停滞
            self.assertEqual(len(calls), 1)  # 只发一次请求（无停滞重试）
            self.assertTrue((Path(td) / "raw" / "issues" / "7.json").exists())

    # ---------- 状态同步：已拉条目远端 updatedAt 更新（如 open→closed）时重拉 ----------

    def _closed_node(self, number, updated_at, created_at="2025-06-01T00:00:00Z"):
        """已关闭的 GraphQL issue 节点（updatedAt 晚于 createdAt，模拟 close 事件）。"""
        node = _node(number, created_at=created_at)
        node.update({
            "updatedAt": updated_at,
            "state": "CLOSED",
            "closedAt": updated_at,
            "title": f"[Bug]: issue {number} closed",
        })
        return node

    def test_pull_incremental_refreshes_updated_item(self):
        """已拉条目远端 updatedAt 更新（open→closed）：增量窗口内重拉覆盖 raw，
        checkpoint 更新时间戳；计数计入"重拉"而非"新增"。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            cp = puller._load_checkpoint()
            cp["issues"]["1"] = {"fetched_at": "x", "comments": False,
                                 "updated_at": "2025-07-01T00:00:00Z"}  # 上次拉取时还是 open
            cp["graphql"]["issues_since"] = "2025-06-01T00:00:00Z"
            puller._save_checkpoint(cp)
            # 旧快照：open 状态
            (Path(td) / "raw" / "issues").mkdir(parents=True)
            (Path(td) / "raw" / "issues" / "1.json").write_text(
                json.dumps({"number": 1, "state": "open", "title": "old"}), encoding="utf-8")

            # 远端已关闭：updatedAt 推进
            closed = self._closed_node(1, updated_at="2025-08-10T00:00:00Z")

            def fake_request(query, variables):
                if "issues" in query:
                    self.assertEqual(variables["filter"], {"since": "2025-06-01T00:00:00Z"})
                    return {"issues": {"totalCount": 1,
                                       "pageInfo": {"hasNextPage": False, "endCursor": "I1"},
                                       "nodes": [closed]}}
                return {"pullRequests": {"totalCount": 0,
                                         "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                         "nodes": []}}

            puller._graphql_request = fake_request
            with redirect_stdout(StringIO()):
                n = puller.pull(incremental=True)
            self.assertEqual(n, 1)  # pull 返回值 = 新增 + 重拉
            raw = json.loads((Path(td) / "raw" / "issues" / "1.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["state"], "closed")  # 状态已同步
            self.assertEqual(raw["updated_at"], "2025-08-10T00:00:00Z")
            cp2 = puller._load_checkpoint()
            self.assertEqual(cp2["issues"]["1"]["updated_at"], "2025-08-10T00:00:00Z")
            self.assertEqual(len(cp2["issues"]), 1)  # 编号数不变（重拉不新增）

    def test_pull_incremental_refresh_refetches_comments(self):
        """重拉时评论一并重取（条目有更新，可能有新评论）。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            src = make_source(token="fake-token",
                              raw_dir=f"{td}/raw", checkpoint_file=f"{td}/cp.json",
                              fetch_comments=True)
            puller = GithubPuller(src, PROJECT_ROOT)
            puller._save_checkpoint({
                "issues": {"1": {"fetched_at": "x", "comments": True,
                                 "updated_at": "2025-07-01T00:00:00Z"}},
                "state": "all", "sort": "created", "direction": "desc",
                "graphql": {"issues_done": True, "prs_done": True,
                            "issues_since": "2025-06-01T00:00:00Z"},
            })
            # 旧评论
            (Path(td) / "raw" / "comments").mkdir(parents=True)
            (Path(td) / "raw" / "comments" / "1.json").write_text(
                json.dumps([{"user": {"login": "a"}, "created_at": "2025-07-01T00:00:00Z", "body": "old"}]),
                encoding="utf-8")

            closed = self._closed_node(1, updated_at="2025-08-10T00:00:00Z")
            closed["comments"] = {
                "pageInfo": {"hasNextPage": False},
                "nodes": [{"author": {"login": "bob"}, "createdAt": "2025-08-09T00:00:00Z",
                           "body": "fixed by #999"}],
            }

            def fake_request(query, variables):
                if "issues" in query:
                    return {"issues": {"totalCount": 1,
                                       "pageInfo": {"hasNextPage": False, "endCursor": "I1"},
                                       "nodes": [closed]}}
                return {"pullRequests": {"totalCount": 0,
                                         "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                         "nodes": []}}

            puller._graphql_request = fake_request
            puller._get_timeline = lambda number: []  # timeline 采集不在本测范围（避免触网）
            with redirect_stdout(StringIO()):
                n = puller.pull(incremental=True)
            self.assertEqual(n, 1)
            comments = json.loads((Path(td) / "raw" / "comments" / "1.json").read_text(encoding="utf-8"))
            self.assertEqual(comments[0]["body"], "fixed by #999")  # 评论已覆盖

    def test_pull_incremental_skips_unchanged_item(self):
        """已拉条目远端 updatedAt 未变：增量窗口内跳过，不重写 raw、不更新时间戳。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            cp = puller._load_checkpoint()
            cp["issues"]["1"] = {"fetched_at": "x", "comments": False,
                                 "updated_at": "2025-07-01T00:00:00Z"}
            cp["graphql"]["issues_since"] = "2025-06-01T00:00:00Z"
            puller._save_checkpoint(cp)
            (Path(td) / "raw" / "issues").mkdir(parents=True)
            (Path(td) / "raw" / "issues" / "1.json").write_text(
                json.dumps({"number": 1, "state": "open", "title": "old"}), encoding="utf-8")

            # 远端 updatedAt 与 checkpoint 相同 → 未变化
            node = _node(1, created_at="2025-06-01T00:00:00Z")
            node["updatedAt"] = "2025-07-01T00:00:00Z"

            def fake_request(query, variables):
                if "issues" in query:
                    return {"issues": {"totalCount": 1,
                                       "pageInfo": {"hasNextPage": False, "endCursor": "I1"},
                                       "nodes": [node]}}
                return {"pullRequests": {"totalCount": 0,
                                         "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                         "nodes": []}}

            puller._graphql_request = fake_request
            with redirect_stdout(StringIO()):
                n = puller.pull(incremental=True)
            self.assertEqual(n, 0)
            raw = json.loads((Path(td) / "raw" / "issues" / "1.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["state"], "open")  # raw 未被覆盖
            cp2 = puller._load_checkpoint()
            self.assertEqual(cp2["issues"]["1"]["updated_at"], "2025-07-01T00:00:00Z")

    def test_pull_incremental_legacy_item_refreshes_in_window_only(self):
        """旧 checkpoint（无 updated_at）：since 有界窗口内重拉一次补记时间戳——
        条目能进窗口说明最近有更新（如 open→closed）。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            cp = puller._load_checkpoint()
            # 旧格式：只有 fetched_at/comments，无 updated_at
            cp["graphql"]["issues_since"] = "2025-06-01T00:00:00Z"
            puller._save_checkpoint(cp)
            (Path(td) / "raw" / "issues").mkdir(parents=True)
            (Path(td) / "raw" / "issues" / "1.json").write_text(
                json.dumps({"number": 1, "state": "open", "title": "old"}), encoding="utf-8")

            closed = self._closed_node(1, updated_at="2025-08-10T00:00:00Z")

            def fake_request(query, variables):
                if "issues" in query:
                    return {"issues": {"totalCount": 1,
                                       "pageInfo": {"hasNextPage": False, "endCursor": "I1"},
                                       "nodes": [closed]}}
                return {"pullRequests": {"totalCount": 0,
                                         "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                         "nodes": []}}

            puller._graphql_request = fake_request
            with redirect_stdout(StringIO()):
                n = puller.pull(incremental=True)
            self.assertEqual(n, 1)  # 重拉一次
            raw = json.loads((Path(td) / "raw" / "issues" / "1.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["state"], "closed")
            cp2 = puller._load_checkpoint()
            self.assertEqual(cp2["issues"]["1"]["updated_at"], "2025-08-10T00:00:00Z")  # 已补记

    def test_pull_incremental_legacy_item_skipped_without_since(self):
        """旧 checkpoint 且无 since（升级后首轮增量全扫）：已有条目跳过，
        避免把整库误当"已变化"重拉一遍（防升级后整库重扫）。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            # 无 issues_since：首轮增量从头枚举
            (Path(td) / "raw" / "issues").mkdir(parents=True)
            (Path(td) / "raw" / "issues" / "1.json").write_text(
                json.dumps({"number": 1, "state": "open", "title": "old"}), encoding="utf-8")

            closed = self._closed_node(1, updated_at="2025-08-10T00:00:00Z")

            def fake_request(query, variables):
                if "issues" in query:
                    return {"issues": {"totalCount": 1,
                                       "pageInfo": {"hasNextPage": False, "endCursor": "I1"},
                                       "nodes": [closed]}}
                return {"pullRequests": {"totalCount": 0,
                                         "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                         "nodes": []}}

            puller._graphql_request = fake_request
            with redirect_stdout(StringIO()):
                n = puller.pull(incremental=True)
            self.assertEqual(n, 0)  # 旧条目未重拉
            raw = json.loads((Path(td) / "raw" / "issues" / "1.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["state"], "open")  # raw 保持原样
            cp2 = puller._load_checkpoint()
            self.assertNotIn("updated_at", cp2["issues"]["1"])  # 未补记时间戳


class TestPullMissing(unittest.TestCase):
    """--pull-missing 补差拉取 / --numbers REST 单条补拉（mock，不触网）。"""

    def _make_puller(self, td: str):
        from vllm_kb.github_pull import GithubPuller

        src = make_source(token="fake-token",
                          raw_dir=f"{td}/raw", checkpoint_file=f"{td}/cp.json",
                          fetch_comments=False, fetch_timeline=False)
        puller = GithubPuller(src, PROJECT_ROOT)
        puller._save_checkpoint({
            "issues": {"2": {"fetched_at": "x", "comments": False}},
            "state": "all", "sort": "created", "direction": "desc",
            "graphql": {"issues_done": True, "prs_done": True},
        })
        return puller

    def test_pull_missing_skips_existing_pulls_new(self):
        """补差：raw 有 1、checkpoint 有 2，GraphQL 返回 1/2/3 → 只拉 3（done=True 也强制拉）。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            # raw 已有编号 1（跳过基准以 raw 为准）
            (Path(td) / "raw" / "issues").mkdir(parents=True)
            (Path(td) / "raw" / "issues" / "1.json").write_text('{"number": 1}', encoding="utf-8")

            def fake_request(query, variables):
                if "issues" in query:
                    # 补差：created desc 排序、不带 filterBy.since
                    self.assertIsNone(variables["filter"])
                    self.assertEqual(variables["order"],
                                     {"field": "CREATED_AT", "direction": "DESC"})
                    return {"issues": {"totalCount": 100,
                                       "pageInfo": {"hasNextPage": False, "endCursor": "C1"},
                                       "nodes": [_node(3), _node(2), _node(1)]}}
                return {"pullRequests": {"totalCount": 0,
                                         "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                         "nodes": []}}

            puller._graphql_request = fake_request
            with redirect_stdout(StringIO()):
                n = puller.pull(missing=True)
            self.assertEqual(n, 1)  # 只新增 3（1 raw 已有、2 checkpoint 已有）
            self.assertTrue((Path(td) / "raw" / "issues" / "3.json").exists())
            self.assertFalse((Path(td) / "raw" / "issues" / "2.json").exists())  # 未重复拉
            cp = puller._load_checkpoint()
            self.assertIn("3", cp["issues"])
            # 补差不推进时间窗（与 --incremental 无关）
            self.assertNotIn("issues_since", cp["graphql"])

    def test_pull_missing_prs_also_enumerated(self):
        """补差同时枚举 issues 与 prs（done=True 时两者都强制拉）。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)

            def fake_request(query, variables):
                if "issues" in query:
                    return {"issues": {"totalCount": 1,
                                       "pageInfo": {"hasNextPage": False, "endCursor": "C1"},
                                       "nodes": []}}
                return {"pullRequests": {"totalCount": 1,
                                         "pageInfo": {"hasNextPage": False, "endCursor": "P1"},
                                         "nodes": [_node(9)]}}

            puller._graphql_request = fake_request
            with redirect_stdout(StringIO()):
                n = puller.pull(missing=True)
            self.assertEqual(n, 1)
            self.assertTrue((Path(td) / "raw" / "prs" / "9.json").exists())

    @staticmethod
    def _rest_pull(number=42):
        return {
            "number": number, "title": f"fix {number}", "body": "body",
            "state": "closed", "created_at": "2026-05-01T00:00:00Z",
            "updated_at": "2026-06-02T00:00:00Z",
            "closed_at": "2026-06-01T00:00:00Z",
            "html_url": f"https://github.com/vllm-project/vllm/pull/{number}",
            "labels": [{"name": "bug"}], "user": {"login": "alice"},
            "merged": True, "merged_at": "2026-06-01T00:00:00Z",
            "merge_commit_sha": "abc123",
            # REST /pulls/{n} 原生 head 三元组
            "head": {"ref": "dev_hy4", "sha": "9" * 40,
                     "repo": {"full_name": "voidvelocity/vllm"}},
        }

    @staticmethod
    def _json_resp(data):
        """构造带 json() 方法的假响应（lambda 需接受实例参数 self）。"""
        return type("R", (), {"json": lambda self: data})()

    def test_pull_numbers_rest_single(self):
        """--numbers：REST 单条拉 PR（/pulls/{n} 成功），merged 信息进 item，无 GraphQL 请求。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            calls = []

            def fake_get(url, params=None):
                calls.append(url)
                assert "pulls/42" in url
                return self._json_resp(self._rest_pull(42))

            puller._get = fake_get
            with redirect_stdout(StringIO()):
                n = puller.pull(numbers=[42])
            self.assertEqual(n, 1)
            self.assertTrue((Path(td) / "raw" / "prs" / "42.json").exists())
            item = json.loads((Path(td) / "raw" / "prs" / "42.json").read_text(encoding="utf-8"))
            self.assertTrue(item["pull_request"]["merged"])
            self.assertEqual(item["pull_request"]["merged_at"], "2026-06-01T00:00:00Z")
            self.assertEqual(item["user"]["login"], "alice")
            # REST head 三元组透传到 raw item
            self.assertEqual(item["pull_request"]["head_repo"], "voidvelocity/vllm")
            self.assertEqual(item["pull_request"]["head_branch"], "dev_hy4")
            self.assertEqual(item["pull_request"]["head_sha"], "9" * 40)
            cp = puller._load_checkpoint()
            self.assertIn("42", cp["issues"])
            self.assertEqual(cp["issues"]["42"]["updated_at"], "2026-06-02T00:00:00Z")  # checkpoint 锚点
            self.assertEqual(len(calls), 1)  # 只发 pulls 一次（fetch_comments=False 不拉评论）

    def test_pull_numbers_falls_back_to_issues_on_404(self):
        """--numbers：/pulls/{n} 404 → /issues/{n}（issue 形态落盘）。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest import mock

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            issue_data = {
                "number": 77, "title": "issue 77", "body": "b", "state": "open",
                "created_at": "2026-05-01T00:00:00Z", "closed_at": None,
                "html_url": "https://github.com/vllm-project/vllm/issues/77",
                "labels": [], "user": {"login": "bob"},
            }

            def fake_get(url, params=None):
                if "pulls/77" in url:
                    err = requests.HTTPError("404 Client Error")
                    err.response = mock.Mock(status_code=404)
                    raise err
                assert "issues/77" in url
                return self._json_resp(issue_data)

            puller._get = fake_get
            with redirect_stdout(StringIO()):
                n = puller.pull(numbers=[77])
            self.assertEqual(n, 1)
            self.assertTrue((Path(td) / "raw" / "issues" / "77.json").exists())
            item = json.loads((Path(td) / "raw" / "issues" / "77.json").read_text(encoding="utf-8"))
            self.assertNotIn("pull_request", item)  # issue 形态

    def test_pull_numbers_skips_existing_raw(self):
        """--numbers：raw 已有编号直接跳过。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            (Path(td) / "raw" / "prs").mkdir(parents=True)
            (Path(td) / "raw" / "prs" / "42.json").write_text('{"number": 42}', encoding="utf-8")
            calls = []

            def fake_get(url, params=None):
                calls.append(url)
                return self._json_resp(self._rest_pull(42))

            puller._get = fake_get
            with redirect_stdout(StringIO()):
                n = puller.pull(numbers=[42])
            self.assertEqual(n, 0)
            self.assertEqual(calls, [])  # 未发请求

    def test_pull_numbers_force_refreshes_existing(self):
        """--numbers --force-numbers：raw/checkpoint 已有也强制重拉覆盖（状态同步）。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            puller = self._make_puller(td)
            # 旧快照：open 状态（raw + checkpoint 均已有）
            (Path(td) / "raw" / "issues").mkdir(parents=True)
            (Path(td) / "raw" / "issues" / "77.json").write_text(
                json.dumps({"number": 77, "state": "open", "title": "old",
                            "updated_at": "2026-04-01T00:00:00Z"}), encoding="utf-8")
            cp = puller._load_checkpoint()
            cp["issues"]["77"] = {"fetched_at": "x", "comments": False,
                                  "updated_at": "2026-04-01T00:00:00Z"}
            puller._save_checkpoint(cp)
            issue_data = {
                "number": 77, "title": "issue 77 closed", "body": "b", "state": "closed",
                "created_at": "2026-05-01T00:00:00Z",
                "updated_at": "2026-06-03T00:00:00Z",
                "closed_at": "2026-06-03T00:00:00Z",
                "html_url": "https://github.com/vllm-project/vllm/issues/77",
                "labels": [], "user": {"login": "bob"},
            }

            from unittest import mock

            def fake_get(url, params=None):
                if "pulls/77" in url:
                    err = requests.HTTPError("404 Client Error")
                    err.response = mock.Mock(status_code=404)
                    raise err
                assert "issues/77" in url
                return self._json_resp(issue_data)

            puller._get = fake_get
            with redirect_stdout(StringIO()):
                n = puller.pull(numbers=[77], force_numbers=True)
            self.assertEqual(n, 1)  # 强制重拉计 1
            item = json.loads((Path(td) / "raw" / "issues" / "77.json").read_text(encoding="utf-8"))
            self.assertEqual(item["state"], "closed")  # 状态已同步
            self.assertEqual(item["updated_at"], "2026-06-03T00:00:00Z")
            cp2 = puller._load_checkpoint()
            self.assertEqual(cp2["issues"]["77"]["updated_at"], "2026-06-03T00:00:00Z")


class TestTimelineCrossrefs(unittest.TestCase):
    """timeline 采集：cross-referenced 事件 → extra.crossrefs（graph REFERENCES 边数据源）。

    真实形状锚点（vllm-ascend#8938 实测）：
    {"source": {"issue": {number, title, state, pull_request?, repository.full_name}},
     "event": "cross-referenced"}——GitHub 页面 "Mentioned in" 区的数据源。
    """

    def setUp(self):
        self.puller = GithubPuller(make_source(), PROJECT_ROOT)  # 仅构造，不发请求

    @staticmethod
    def _xref_event(number, repo="vllm-project/vllm-ascend", is_pr=True):
        issue = {"number": number, "title": f"ref {number}", "state": "closed",
                 "repository": {"full_name": repo}}
        if is_pr:
            issue["pull_request"] = {"url": f"https://api.github.com/repos/{repo}/pulls/{number}"}
        return {"event": "cross-referenced", "source": {"type": "issue", "issue": issue}}

    def test_crossrefs_from_timeline_shape(self):
        from vllm_kb.github_pull import _crossrefs_from_timeline

        events = [
            {"event": "labeled", "label": {"name": "bug"}},                # 忽略
            {"event": "mentioned"},                                        # 忽略（@人）
            {"event": "referenced", "commit_id": "a" * 40},                # 忽略（commit 引用）
            self._xref_event(10260),                                       # PR 引用
            self._xref_event(9001, is_pr=False),                           # issue 引用
            self._xref_event(10260),                                       # 重复 → 去重
        ]
        self.assertEqual(_crossrefs_from_timeline(events), [
            {"repo": "vllm-project/vllm-ascend", "number": 10260, "is_pr": True,
             "title": "ref 10260", "state": "closed"},
            {"repo": "vllm-project/vllm-ascend", "number": 9001, "is_pr": False,
             "title": "ref 9001", "state": "closed"},
        ])

    def test_crossrefs_missing_repo_or_number_skipped(self):
        from vllm_kb.github_pull import _crossrefs_from_timeline

        events = [
            {"event": "cross-referenced", "source": {"issue": {"number": 5}}},          # 无 repo
            {"event": "cross-referenced", "source": {"issue": {"repository": {"full_name": "o/r"}}}},  # 无 number
            {"event": "cross-referenced", "source": {}},                                 # 无 issue
        ]
        self.assertEqual(_crossrefs_from_timeline(events), [])
        self.assertEqual(_crossrefs_from_timeline([]), [])

    def test_to_doc_carries_crossrefs(self):
        xrefs = [{"repo": "vllm-project/vllm-ascend", "number": 10260, "is_pr": True,
                  "title": "[EPLB][Test] Change mooncake port", "state": "closed"}]
        doc = self.puller._to_doc(make_item(number=8938, body="b"), [], crossrefs=xrefs)
        self.assertEqual(doc.extra["crossrefs"], xrefs)
        doc2 = self.puller._to_doc(make_item(number=8938, body="b"), [])
        self.assertNotIn("crossrefs", doc2.extra)  # 空不落 extra——canonical 指纹稳定

    def test_get_timeline_paginates(self):
        """timeline 翻页：per_page=100 + Link next 追到尾页。

        默认页只有 30 条——实测 #8938 的 cross-referenced 排在第一页之后，
        不翻页就会漏采（本测试钉住这个回归）。
        """
        page1 = [{"event": "labeled"}] * 100
        page2 = [self._xref_event(10260)]
        seen = []

        def fake_get(url, params=None):
            seen.append((url, dict(params or {})))
            if len(seen) == 1:
                return type("R", (), {"json": lambda self: page1,
                                      "links": {"next": {"url": "https://api.github.com/x?page=2"}}})()
            return type("R", (), {"json": lambda self: page2, "links": {}})()

        self.puller._get = fake_get
        events = self.puller._get_timeline(8938)
        self.assertEqual(len(events), 101)
        self.assertIn("/timeline", seen[0][0])
        self.assertEqual(seen[0][1].get("per_page"), 100)

    def test_pull_single_saves_timeline_raw(self):
        """REST 补拉：fetch_timeline 默认开启 → timeline/{n}.json 落盘。"""
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as td:
            src = make_source(token="t", raw_dir=f"{td}/raw",
                              checkpoint_file=f"{td}/cp.json", fetch_comments=False)
            puller = GithubPuller(src, PROJECT_ROOT)
            issue_data = {
                "number": 8938, "title": "t", "body": "b", "state": "closed",
                "created_at": "2026-05-07T00:00:00Z", "closed_at": "2026-07-21T00:00:00Z",
                "html_url": "https://github.com/vllm-project/vllm/issues/8938",
                "labels": [], "user": {"login": "z"},
            }

            def fake_get(url, params=None):
                if "/timeline" in url:
                    tl = [self._xref_event(10260)]  # 先取值：lambda 的 self 是 R 实例
                    return type("R", (), {"json": lambda s: tl, "links": {}})()
                return type("R", (), {"json": lambda s: issue_data, "links": {}})()

            puller._get = fake_get
            with redirect_stdout(StringIO()):
                puller.pull(numbers=[8938])
            tp = Path(td) / "raw" / "timeline" / "8938.json"
            self.assertTrue(tp.exists())
            events = json.loads(tp.read_text(encoding="utf-8"))
            self.assertEqual(events[0]["event"], "cross-referenced")

    def test_recanonicalize_consumes_timeline(self):
        """canonical 再生：raw timeline 的 cross-referenced → doc.extra.crossrefs。"""
        import tempfile

        from vllm_kb.github_pull import recanonicalize

        with tempfile.TemporaryDirectory() as td:
            src = make_source(token="t", raw_dir=f"{td}/raw",
                              checkpoint_file=f"{td}/cp.json", fetch_comments=False)
            puller = GithubPuller(src, PROJECT_ROOT)
            item = make_item(number=8938, body="b")
            item["user"] = {"login": "z"}
            puller._save_raw("issues", 8938, item)
            puller._save_raw("timeline", 8938, [self._xref_event(10260)])
            docs = recanonicalize(src, Path(td))
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].extra["crossrefs"][0]["number"], 10260)


if __name__ == "__main__":
    unittest.main()
