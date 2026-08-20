"""Phase 2 图存储测试：关系抽取规则、Kùzu 建图、链路查询。"""
import json
import tempfile
import unittest
from pathlib import Path

from vllm_kb.graph import GraphBuilder
from vllm_kb.graph_rels import (
    ReleaseInfo,
    extract_doc_relations,
    extract_fix_refs,
    extract_fixed_by_refs,
    map_merged_to_release,
)

CAL = [
    ReleaseInfo(tag="v0.23.0rc1", date="2026-07-19T13:55:17Z", kind="rc", prerelease=True),
    ReleaseInfo(tag="v0.23.0", date="2026-08-16T22:18:14Z", kind="release"),
]

CANONICAL = [
    {
        "source_type": "github_issue",
        "source_id": "github:vllm-project-vllm-ascend:issue:10700",
        "url": "https://github.com/vllm-project/vllm-ascend/issues/10700",
        "title": "GLM5.1 未添加enforce_eager true运行一段时间后崩溃",
        "body": "halMemCreate failed drvRetCode=6, kernel_name=DispatchFFNCombine. fixed by #12885",
        "status": "open",
        "created_at": "2026-07-01T00:00:00Z",
        "extra": {"repo": "vllm-project/vllm-ascend", "github_number": 10700, "merged_at": None},
        "version_span": {"min": "0.23.0rc1"},
    },
    {
        "source_type": "github_pr",
        "source_id": "github:vllm-project-vllm-ascend:pr:12885",
        "url": "https://github.com/vllm-project/vllm-ascend/pull/12885",
        "title": "Fix GLM5.1 P2P hang",
        "body": "Fixes #10700. Also mention aclnnMoeDistributeDispatchV4 failed 561000.",
        "status": "closed",
        "created_at": "2026-07-20T00:00:00Z",
        "resolved_at": "2026-07-31T00:00:00Z",
        "extra": {"repo": "vllm-project/vllm-ascend", "github_number": 12885,
                  "merged": True, "merged_at": "2026-07-31T00:00:00Z"},
        "version_span": {"max": "0.23.0"},
    },
    {
        "source_type": "github_issue",
        "source_id": "github:vllm-project-vllm:issue:1",
        "url": "https://github.com/vllm-project/vllm/issues/1",
        "title": "illegal memory access during inference",
        "body": "CUDA error: an illegal memory access was encountered with GLM-5.1.",
        "status": "closed",
        "extra": {"repo": "vllm-project/vllm", "github_number": 1, "merged_at": None},
        "version_span": {},
    },
    {
        "source_type": "doc_pdf",
        "source_id": "pdf:npu_hccn_tool_guide",
        "url": "",
        "title": "NPU HCCN Tool 接口参考",
        "body": "hccn_tool 排查命令。错误码 107020 memory allocation failed。\n"
                "命令格式\nhccn_tool [-i %d] -bandwidth -g\n"
                "命令格式\nhccn_tool [-i %d] -roce_test %s\n"
                "命令格式\nhccn_tool -h\n",
        "status": "open",
        "extra": {"verification": "expert",
                  "asset": {"path": "assets/pdf/npu_hccn_tool_guide.pdf"},
                  "structure": {"tables": ["parsed/pdf/npu_hccn_tool_guide.tables.json"]}},
        "version_span": {},
    },
]

# 手册表格产物（错误码表）
TABLES_JSON = {
    "source": "assets/pdf/npu_hccn_tool_guide.pdf",
    "tables": [
        {"page": 1, "index": 0, "rows": [
            ["错误码", "含义"],
            ["107020", "memory allocation failed"],
            ["507014", "device busy"],
            ["561000", "dispatch failed"],
        ]},
    ],
}


def write_canonical(tmp: Path) -> Path:
    p = tmp / "canonical.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for doc in CANONICAL:
            f.write(json.dumps(doc, ensure_ascii=True) + "\n")
    return p


def write_parsed(tmp: Path) -> Path:
    """造 parsed/pdf 表格产物（接口指南错误码表）。"""
    parsed = tmp / "parsed" / "pdf"
    parsed.mkdir(parents=True)
    (parsed / "npu_hccn_tool_guide.tables.json").write_text(
        json.dumps(TABLES_JSON, ensure_ascii=False), encoding="utf-8")
    return tmp / "parsed"


class TestExtractRules(unittest.TestCase):
    def test_fix_refs_same_repo(self):
        self.assertEqual(
            extract_fix_refs("Fixes #10700 and closes #1", "vllm-project/vllm-ascend"),
            [("vllm-project/vllm-ascend", 1), ("vllm-project/vllm-ascend", 10700)],
        )

    def test_fix_refs_cross_repo(self):
        self.assertEqual(
            extract_fix_refs("fixes vllm-project/vllm#50241", "vllm-project/vllm-ascend"),
            [("vllm-project/vllm", 50241)],
        )

    def test_fixed_by_refs(self):
        self.assertEqual(
            extract_fixed_by_refs("This is fixed by #12885.", "vllm-project/vllm-ascend"),
            [("vllm-project/vllm-ascend", 12885)],
        )

    def test_map_merged_to_release(self):
        # 合并于 rc 之前 → 首个满足的 release
        self.assertEqual(map_merged_to_release("2026-07-01T00:00:00Z", CAL), "v0.23.0rc1")
        # 合并于 rc 与 release 之间 → 正式版
        self.assertEqual(map_merged_to_release("2026-07-31T00:00:00Z", CAL), "v0.23.0")
        # 晚于所有 release → 尚未发布
        self.assertIsNone(map_merged_to_release("2026-09-01T00:00:00Z", CAL))
        # 空输入
        self.assertIsNone(map_merged_to_release("", CAL))

    def test_doc_relations_mentions(self):
        ex = extract_doc_relations(
            "github:vllm-project-vllm-ascend:issue:10700",
            "vllm-project/vllm-ascend", 10700, "github_issue",
            CANONICAL[0]["body"],
            version_span_min="0.23.0rc1",
        )
        self.assertEqual(ex.fixed_by, [("vllm-project/vllm-ascend", 12885)])
        self.assertTrue(any("dispatchffncombine" in v.lower() for v in ex.mentions.get("operator", set())))
        self.assertIn("0.23.0rc1", ex.mentions.get("version", set()))


class TestGraphBuildAndQuery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.canonical = write_canonical(Path(self.tmp.name))
        self.parsed = write_parsed(Path(self.tmp.name))
        self.builder = GraphBuilder(Path(self.tmp.name) / "graph")
        self.builder.create_schema()
        self.builder.build_from_canonical(
            self.canonical,
            calendars={"vllm-project/vllm-ascend": CAL},
            parsed_root=self.parsed,
        )

    def tearDown(self):
        self.builder.close()
        self.tmp.cleanup()

    def test_stats(self):
        s = self.builder.stats()
        self.assertEqual(s.nodes["Issue"], 2)
        self.assertEqual(s.nodes["PR"], 1)
        # Release 节点只建被 MERGED_IN 引用的（无边的 release 不入图）
        self.assertEqual(s.nodes["Release"], 1)
        self.assertEqual(s.nodes["Doc"], 1)  # doc_pdf
        self.assertEqual(s.rels["FIXES"], 1)
        self.assertEqual(s.rels["MERGED_IN"], 1)
        self.assertGreater(s.rels["MENTIONS"], 0)
        # DOCUMENTS：手册表格提取的 3 个错误码 + 命令格式段的 Interface
        self.assertEqual(s.rels["DOCUMENTS"], 3 + 2)
        self.assertEqual(s.nodes["Interface"], 2)

    def test_chain_issue(self):
        r = self.builder.chain_issue("github:vllm-project-vllm-ascend:issue:10700")
        self.assertTrue(r["found"])
        self.assertEqual(len(r["fixes"]), 1)
        f = r["fixes"][0]
        self.assertEqual(f["pr_id"], "github:vllm-project-vllm-ascend:pr:12885")
        self.assertEqual(f["release_tag"], "v0.23.0")  # merged 07-31 → 08-16 发布
        self.assertTrue(r["released"])

    def test_chain_issue_not_found(self):
        r = self.builder.chain_issue("github:vllm-project-vllm:issue:99999")
        self.assertFalse(r["found"])

    def test_fixes_pr(self):
        r = self.builder.fixes_pr("github:vllm-project-vllm-ascend:pr:12885")
        self.assertTrue(r["found"])
        self.assertEqual(r["fixes"][0]["issue_id"],
                         "github:vllm-project-vllm-ascend:issue:10700")
        self.assertEqual([x["tag"] for x in r["releases"]], ["v0.23.0"])

    def test_sig_lookup(self):
        r = self.builder.sig_lookup("dispatchffncombine")
        self.assertGreater(r["count"], 0)
        self.assertEqual(r["entity_type"], "operator")
        # 错误码签名
        r2 = self.builder.sig_lookup("561000")
        self.assertGreater(r2["count"], 0)
        self.assertEqual(r2["entity_type"], "error_code")

    def test_doc_neighbors(self):
        r = self.builder.doc_neighbors("github:vllm-project-vllm:issue:1")
        kinds = {m["entity_type"] for m in r["mentions"]}
        self.assertIn("model", kinds)

    def test_doc_documents_error_codes(self):
        """手册 Doc 节点：DOCUMENTS 边指向表格提取的错误码（107020/507014/561000）。"""
        r = self.builder.doc_neighbors("pdf:npu_hccn_tool_guide")
        codes = {m["value"] for m in r.get("documents", []) if m["entity_type"] == "error_code"}
        self.assertEqual(codes, {"107020", "507014", "561000"})

    def test_doc_documents_interfaces(self):
        """手册"命令格式"段 → Interface 节点 + DOCUMENTS 边。
        提取 hccn_tool.bandwidth / hccn_tool.roce_test；-h 帮助与方括号内参数不提取。"""
        r = self.builder.doc_neighbors("pdf:npu_hccn_tool_guide")
        ifaces = {m["value"] for m in r.get("documents", []) if m["entity_type"] == "interface"}
        self.assertEqual(ifaces, {"hccn_tool.bandwidth", "hccn_tool.roce_test"})
        # Interface 节点存在
        rows = self.builder.query("MATCH (i:Interface) RETURN i.id ORDER BY i.id")
        self.assertEqual([x[0] for x in rows], ["hccn_tool.bandwidth", "hccn_tool.roce_test"])

    def test_docs_for_error_code(self):
        """错误码 → 定义它的手册（'这个错误码在哪个手册定义'）。"""
        docs = self.builder.docs_for_error_code("107020")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["doc_id"], "pdf:npu_hccn_tool_guide")
        self.assertEqual(docs[0]["source_type"], "doc_pdf")
        # 未被表格定义的错误码 → 空
        self.assertEqual(self.builder.docs_for_error_code("999999"), [])

    def test_doc_mentions_shared_with_issue(self):
        """手册正文提及的错误码与 GitHub issue 共享 ErrorCode 节点（跨来源连接）。"""
        r = self.builder.sig_lookup("107020")
        kinds = {d["entity_type"] for d in r["docs"]}
        # 手册 Doc 通过 DOCUMENTS 边、issue 通过 MENTIONS 边都能触达同一 ErrorCode 节点
        self.assertEqual(r["entity_type"], "error_code")


if __name__ == "__main__":
    unittest.main()
