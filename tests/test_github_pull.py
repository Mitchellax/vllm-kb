"""GitHub 采集转换逻辑测试（不触网）：canonical 映射、版本提取、PR 处理、recanonicalize。"""
import unittest
from pathlib import Path

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


def _node(number):
    """构造最小 GraphQL issues 节点。"""
    return {
        "number": number, "title": "t", "body": "b", "state": "OPEN",
        "createdAt": "2026-01-01T00:00:00Z", "closedAt": None,
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
            new_count, skip_count = puller._collect_graphql(cp, g, kind="issues")
            self.assertEqual(new_count, 3)
            self.assertEqual(skip_count, 0)
            self.assertEqual(after_values, [None, "C1"])  # 第二次请求必须带上第一页的 endCursor
            self.assertTrue(g["issues_done"])
            self.assertEqual(sorted(cp["issues"]), ["1", "2", "3"])

    def test_graphql_dead_loop_guard(self):
        """endCursor 重复（分页卡死）时必须报错而非无限循环。"""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = make_source(repo="vllm-project/vllm-ascend", raw_dir=str(Path(td) / "raw"),
                              fetch_comments=False)
            puller = GithubPuller(src, PROJECT_ROOT)

            def stuck_request(query, variables):
                return {"issues": {"totalCount": 250,
                                   "pageInfo": {"hasNextPage": True, "endCursor": "C1"},
                                   "nodes": [_node(1)]}}

            puller._graphql_request = stuck_request
            with self.assertRaises(RuntimeError) as ctx:
                puller._collect_graphql({"issues": {}}, {}, kind="issues")
            self.assertIn("游标未前进", str(ctx.exception))

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
            cp = {"issues": {}}
            g = {}
            new_count, skip_count = puller._collect_graphql(cp, g, kind="issues")
            self.assertEqual(new_count, 1)
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
            puller.session.post = mock.Mock(side_effect=lambda *a, **k: next(responses))
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


if __name__ == "__main__":
    unittest.main()
