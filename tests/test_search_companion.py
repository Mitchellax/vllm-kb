"""检索集成测试：组件查询 + 配套反向展开 + 未解决兜底（离线 echo 模式）。"""
import json
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig
from vllm_kb.embed import EmbeddingClient
from vllm_kb.ingest import ingest_docs
from vllm_kb.models import KbDocument, VersionSpan
from vllm_kb.search import SearchEngine
from vllm_kb.vectorstore import PythonVectorStore

ASC_UNRESOLVED = KbDocument(
    source_type="github_issue",
    source_id="github:vllm-project-vllm-ascend:issue:9001",
    url="https://github.com/vllm-project/vllm-ascend/issues/9001",
    title="[Bug]: GLM5.1 PD 分离 P 节点挂死，需手动重启",
    body=(
        "GLM5.1 使用 PD 分离部署，P 节点在长稳压测中挂死，无法自动恢复。\n"
        "### Your current environment\n- **vllm-ascend version**: 0.18.0\n- vllm 0.12.1, CANN 8.1.RC2\n\n"
        "---\n\n### commenter (2026-07-01T00:00:00Z):\n"
        "临时规避：降低 max_num_seqs 并关闭 chunked prefill 可缓解。"
    ),
    created_at="2026-06-20T00:00:00Z",
    status="open",
    labels=["bug", "pd"],
    version_span=VersionSpan(min="0.18.0"),
    component="vllm-ascend",
    component_versions={"vllm": "0.12.1", "cann": "8.1.RC2"},
)

ASC_RESOLVED = KbDocument(
    source_type="github_issue",
    source_id="github:vllm-project-vllm-ascend:issue:9002",
    url="https://github.com/vllm-project/vllm-ascend/issues/9002",
    title="[Bug]: GLM5.1 PD 分离调度器偶发挂死",
    body=(
        "GLM5.1 PD 分离下 scheduler 偶发挂死，已定位为 PD 心跳超时，修复见 PR。\n"
        "### Your current environment\n- **vllm-ascend version**: 0.18.0"
    ),
    created_at="2026-05-01T00:00:00Z",
    resolved_at="2026-05-20T00:00:00Z",
    status="closed",
    labels=["bug", "pd"],
    version_span=VersionSpan(min="0.18.0"),
    component="vllm-ascend",
)

VLLM_ISSUE = KbDocument(
    source_type="github_issue",
    source_id="github:vllm-project-vllm:issue:9003",
    url="https://github.com/vllm-project/vllm/issues/9003",
    title="[Bug]: PD 分离时 EngineCore 挂死 no heartbeat",
    body=(
        "PD 分离部署下 EngineCore 不发送心跳导致对端判定挂死。\n"
        "### Your current environment\n- **vLLM version**: 0.12.1"
    ),
    created_at="2026-04-01T00:00:00Z",
    resolved_at="2026-04-15T00:00:00Z",
    status="closed",
    labels=["bug", "pd"],
    version_span=VersionSpan(min="0.12.1"),
    component="vllm",
)

MATRIX = {
    "rows": [
        {
            "vllm-ascend": "0.18.0",
            "vllm": "0.12.1",
            "cann": "8.1.RC2",
            "pytorch": "2.6.0",
            "pytorch-ascend": "2.6.0.post1",
            "npu-driver": "",
            "notes": "",
        }
    ]
}


def make_cfg(tmp: Path, resolved_min_sim: float = 0.5) -> AppConfig:
    matrix_path = tmp / "matrix.json"
    matrix_path.write_text(json.dumps(MATRIX), encoding="utf-8")
    return AppConfig.model_validate(
        {
            "embedding": {"provider": "echo", "dimensions": 1024, "batch_size": 4},
            "chunking": {"max_chunk_chars": 3000, "overlap_chars": 100},
            "storage": {
                "vector_backend": "python",
                "lancedb_path": str(tmp / "vec.json"),
                "sqlite_path": str(tmp / "kb.sqlite3"),
                "canonical_file": str(tmp / "canonical.jsonl"),
                "companion_file": str(matrix_path),
                "release_calendar": "",
            },
            "retrieval": {
                "final_top_k": 10,
                "vector_top_k": 50,
                "fts_top_k": 50,
                "resolved_min_similarity": resolved_min_sim,
                "prefer_unresolved_without_resolved": True,
            },
        }
    )


class TestCompanionSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.store = PythonVectorStore(self.cfg.resolve(self.cfg.storage.lancedb_path + "_py.json"))
        self.embed = EmbeddingClient(self.cfg.embedding)

    def tearDown(self):
        self.tmp.cleanup()

    def _ingest(self, docs):
        ingest_docs(self.cfg, docs, self.embed, self.store)

    def test_component_query_parsed_and_companion_expanded(self):
        """'vllm-ascend:0.18.0 ...' 查询：vllm 文档按配套版本 0.12.1 打分（w_ver=1.0）。"""
        self._ingest([ASC_UNRESOLVED, VLLM_ISSUE])
        engine = SearchEngine(self.cfg)
        try:
            results = engine.search("vllm-ascend:0.18.0 GLM5.1 PD 分离 P 节点挂死", top_k=10)
            # 配套上下文已展开
            self.assertEqual(engine.last_context["component"], "vllm-ascend")
            self.assertEqual(engine.last_context["companions"].get("vllm"), ["0.12.1"])
            # 主组件 issue 命中
            self.assertEqual(results[0].doc_id, ASC_UNRESOLVED.source_id)
            # vllm 文档：版本参考来自配套（0.12.1），w_ver 应为 1.0 而非默认 0.5
            vllm_hit = next((r for r in results if r.doc_id == VLLM_ISSUE.source_id), None)
            self.assertIsNotNone(vllm_hit, "vllm 配套文档应被召回")
            self.assertEqual(vllm_hit.version_ref, "0.12.1")
            self.assertAlmostEqual(vllm_hit.confidence.version_weight, 1.0)
        finally:
            engine.close()

    def test_component_filter(self):
        """严格过滤用 filters={'component': ...}；component 参数仅用于版本解析（不排除配套文档）。"""
        self._ingest([ASC_UNRESOLVED, VLLM_ISSUE])
        engine = SearchEngine(self.cfg)
        try:
            results = engine.search(
                "PD 分离 挂死", component="vllm-ascend", version="0.18.0",
                filters={"component": "vllm-ascend"}, top_k=10,
            )
            self.assertTrue(all(r.component == "vllm-ascend" for r in results))
            self.assertNotIn(VLLM_ISSUE.source_id, [r.doc_id for r in results])
        finally:
            engine.close()

    def test_component_param_keeps_companion_docs(self):
        """component 参数只做版本解析：配套组件（vllm）文档仍会被召回（反向配套关联）。"""
        self._ingest([ASC_UNRESOLVED, VLLM_ISSUE])
        engine = SearchEngine(self.cfg)
        try:
            results = engine.search(
                "PD 分离 挂死", component="vllm-ascend", version="0.18.0", top_k=10
            )
            doc_ids = [r.doc_id for r in results]
            self.assertIn(ASC_UNRESOLVED.source_id, doc_ids)
            self.assertIn(VLLM_ISSUE.source_id, doc_ids)  # 配套文档不被排除
        finally:
            engine.close()

    def test_unresolved_leads_when_no_strong_resolved(self):
        """无强匹配已解决问题时，未解决问题（含规避方案）优先。"""
        self._ingest([ASC_UNRESOLVED, ASC_RESOLVED])
        engine = SearchEngine(self.cfg)
        try:
            # 阈值 0.9：已解决结果相似度必然低于阈值 -> 未解决优先
            engine.cfg.retrieval.resolved_min_similarity = 0.9
            results = engine.search("vllm-ascend:0.18.0 GLM5.1 PD 分离 P 节点挂死", top_k=10)
            self.assertTrue(results)
            self.assertFalse(results[0].resolved, "无强匹配已解决时应优先未解决（含规避方案）")
            self.assertEqual(results[0].doc_id, ASC_UNRESOLVED.source_id)
            # 未解决结果带 resolved=False 标记，供 agent 识别
            tags = {r.doc_id: r.resolved for r in results}
            self.assertFalse(tags[ASC_UNRESOLVED.source_id])
        finally:
            engine.close()

    def test_resolved_leads_when_strong_match(self):
        """存在强匹配已解决问题时，已解决优先（w_rel 更高）。"""
        self._ingest([ASC_UNRESOLVED, ASC_RESOLVED])
        engine = SearchEngine(self.cfg)
        try:
            engine.cfg.retrieval.resolved_min_similarity = 0.1  # 已解决结果相似度高于阈值
            results = engine.search("vllm-ascend:0.18.0 GLM5.1 PD 分离 P 节点挂死", top_k=10)
            self.assertTrue(results)
            # 两个问题都很相关：已解决（w_rel=0.6）应排在未解决（0.4）之前
            self.assertTrue(results[0].resolved)
        finally:
            engine.close()

    def test_kind_downweight_flows_through_search(self):
        """[Doc]/[Feature] 反馈类 issue 的 kind 降权应穿透到检索结果（w_rel 现场重算）。"""
        doc_issue = ASC_UNRESOLVED.model_copy(
            update={"source_id": "github:vllm-project-vllm-ascend:issue:9101",
                    "extra": {"kind": "doc"}}
        )
        bug_issue = ASC_UNRESOLVED.model_copy(
            update={"source_id": "github:vllm-project-vllm-ascend:issue:9102",
                    "extra": {"kind": "bug"}}
        )
        self._ingest([doc_issue, bug_issue])
        engine = SearchEngine(self.cfg)
        try:
            results = engine.search("vllm-ascend:0.18.0 GLM5.1 PD 分离 P 节点挂死", top_k=10)
            by_id = {r.doc_id: r for r in results}
            # open issue 基础 w_rel=0.4；doc 类 ×0.6 -> 0.24，bug 类不降 -> 0.4
            self.assertAlmostEqual(by_id[bug_issue.source_id].confidence.reliability, 0.4)
            self.assertAlmostEqual(by_id[doc_issue.source_id].confidence.reliability, 0.24)
            # 降权后 doc 反馈类排在 bug 之后
            idx = {r.doc_id: i for i, r in enumerate(results)}
            self.assertLess(idx[bug_issue.source_id], idx[doc_issue.source_id])
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
