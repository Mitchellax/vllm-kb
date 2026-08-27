"""离线全链路测试：echo embedding + 纯 Python 向量后端 + 临时目录。

覆盖：入库 -> 检索 -> 置信度分解 -> 过滤 -> 幂等重建。
不依赖任何外部 API 与网络。
"""
import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from vllm_kb.config import AppConfig
from vllm_kb.embed import EmbeddingClient
from vllm_kb.ingest import ingest_docs
from vllm_kb.models import KbDocument, VersionSpan
from vllm_kb.search import SearchEngine
from vllm_kb.vectorstore import PythonVectorStore

CUDA_ISSUE = KbDocument(
    source_type="github_issue",
    source_id="github:issue:10001",
    url="https://github.com/vllm-project/vllm/issues/10001",
    title="CUDA illegal memory access during inference",
    body=(
        "When running vLLM inference with tensor parallel, I get "
        "CUDA error: an illegal memory access was encountered. "
        "It happens after a few steps of generation with llama 70b on 2 GPUs.\n\n"
        "---\n\n### commenter (2024-03-01T00:00:00Z):\n"
        "Try reducing max_num_seqs or disabling chunked prefill as a workaround."
    ),
    created_at="2024-02-20T00:00:00Z",
    resolved_at="2024-03-10T00:00:00Z",
    status="closed",
    labels=["bug", "bug: v0.5.x"],
    version_span=VersionSpan(min="0.5.0", max="0.5.4"),
)

OOM_ISSUE = KbDocument(
    source_type="github_issue",
    source_id="github:issue:10002",
    url="https://github.com/vllm-project/vllm/issues/10002",
    title="Not enough memory to allocate paged attention blocks",
    body=(
        "ValueError: Not enough memory to allocate paged attention blocks. "
        "The model requires 12GB but only 8GB is available. "
        "This happens when max_model_len is too large for the GPU.\n\n"
        "---\n\n### maintainer (2024-04-05T00:00:00Z):\n"
        "Reduce max_model_len or use gpu_memory_utilization=0.9."
    ),
    created_at="2024-04-01T00:00:00Z",
    resolved_at="2024-04-10T00:00:00Z",
    status="closed",
    labels=["bug", "oom"],
    version_span=VersionSpan(min="0.5.0", max="0.6.1"),
)

FEATURE_ISSUE = KbDocument(
    source_type="github_issue",
    source_id="github:issue:10003",
    url="https://github.com/vllm-project/vllm/issues/10003",
    title="Feature request: support fp8 quantization for mixtral",
    body="It would be nice to support fp8 quantization for mixtral models. We plan to add this soon.",
    created_at="2024-05-01T00:00:00Z",
    status="open",
    labels=["feature-request"],
)


def make_cfg(tmp: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"name": "test", "data_root": "data"},
            "github": {"token": "", "token_env": "GITHUB_TOKEN"},
            "embedding": {"provider": "echo", "dimensions": 1024, "batch_size": 4},
            "chunking": {"max_chunk_chars": 3000, "overlap_chars": 100},
            "storage": {
                "vector_backend": "python",
                "lancedb_path": str(tmp / "vec.json"),
                "sqlite_path": str(tmp / "kb.sqlite3"),
                "canonical_file": str(tmp / "canonical.jsonl"),  # 必须隔离到临时目录，避免覆盖真实 canonical
                "release_calendar": "",
            },
            "retrieval": {
                "final_top_k": 5,
                "vector_top_k": 20,
                "fts_top_k": 20,
                "default_target_version": "0.6.0",
                "resolved_min_similarity": 0.2,  # echo 向量相似度整体偏低，放宽触发阈值
            },
        }
    )


class CountingStore:
    """记录 add/delete 调用次数与批大小的假向量存储（验证攒批行为）。"""

    def __init__(self):
        self.add_calls = 0
        self.delete_calls = 0
        self.added = 0
        self.deleted: list[str] = []

    def add_items(self, items):
        self.add_calls += 1
        self.added += len(items)

    def delete_docs(self, doc_ids):
        self.delete_calls += 1
        self.deleted.extend(doc_ids)

    def delete_doc(self, doc_id):
        self.delete_calls += 1
        self.deleted.append(doc_id)

    def update_doc_meta(self, doc_id, meta):
        pass


class TestOfflinePipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.store = PythonVectorStore(self.cfg.resolve(self.cfg.storage.lancedb_path + "_py.json"))
        self.embed = EmbeddingClient(self.cfg.embedding)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ingest_and_search_relevant_first(self):
        ingest_docs(self.cfg, [CUDA_ISSUE, OOM_ISSUE, FEATURE_ISSUE], self.embed, self.store)
        engine = SearchEngine(self.cfg)
        try:
            # 用特征性强的短语：echo 向量 + FTS 都应优先命中 CUDA issue
            results = engine.search("illegal memory access", target_version="0.6.0")
            self.assertTrue(results)
            top = results[0]
            self.assertEqual(top.doc_id, CUDA_ISSUE.source_id)
            # 置信度分解存在
            self.assertGreater(top.confidence.time_weight, 0)
            self.assertGreater(top.confidence.reliability, 0)
            self.assertGreaterEqual(top.final, 0)
        finally:
            engine.close()

    def test_oom_phrase_hits_oom_first(self):
        ingest_docs(self.cfg, [CUDA_ISSUE, OOM_ISSUE], self.embed, self.store)
        engine = SearchEngine(self.cfg)
        try:
            results = engine.search("not enough memory to allocate paged attention blocks", target_version="0.6.0")
            self.assertTrue(results)
            self.assertEqual(results[0].doc_id, OOM_ISSUE.source_id)
        finally:
            engine.close()

    def test_fts_fallback_hits_keyword(self):
        ingest_docs(self.cfg, [CUDA_ISSUE, OOM_ISSUE], self.embed, self.store)
        engine = SearchEngine(self.cfg)
        try:
            # echo 向量对短查询召回有限，FTS 精确关键词应能命中
            results = engine.search("paged attention blocks", target_version="0.6.0")
            self.assertTrue(results)
            self.assertEqual(results[0].doc_id, OOM_ISSUE.source_id)
        finally:
            engine.close()

    def test_filters(self):
        ingest_docs(self.cfg, [CUDA_ISSUE, OOM_ISSUE, FEATURE_ISSUE], self.embed, self.store)
        engine = SearchEngine(self.cfg)
        try:
            results = engine.search("vLLM", filters={"status": "open"})
            self.assertTrue(all(r.meta.get("status") == "open" for r in results))
            self.assertEqual(results[0].doc_id, FEATURE_ISSUE.source_id)
        finally:
            engine.close()

    def test_idempotent_reingest(self):
        ingest_docs(self.cfg, [CUDA_ISSUE, OOM_ISSUE], self.embed, self.store)
        n1 = self.store.count()
        # 重复入库同一批 -> 条数不变（内容哈希跳过，不重新嵌入）
        stats2 = ingest_docs(self.cfg, [CUDA_ISSUE, OOM_ISSUE], self.embed, self.store)
        n2 = self.store.count()
        self.assertEqual(n1, n2)
        self.assertGreater(n1, 0)
        self.assertGreaterEqual(stats2.get("skipped_unchanged", 0), 2)
        self.assertEqual(stats2.get("embedded", -1), 0)  # 未变化 -> 不重新嵌入

    def test_incremental_reingest_only_changes(self):
        """同一批文档，只有变更的那条被重新嵌入。"""
        ingest_docs(self.cfg, [CUDA_ISSUE, OOM_ISSUE], self.embed, self.store)
        changed = CUDA_ISSUE.model_copy(update={"body": CUDA_ISSUE.body + "\n\n### NEW (2026-08-01T00:00:00Z):\nupdated root cause"})
        stats = ingest_docs(self.cfg, [changed, OOM_ISSUE], self.embed, self.store)
        self.assertEqual(stats.get("docs", -1), 1)  # 只处理变更的一条
        self.assertEqual(stats.get("skipped_unchanged", 0), 1)
        self.assertEqual(stats.get("embedded", -1), 1)

    def test_meta_only_change_no_reembed(self):
        """仅元数据变化（版本区间）不重嵌：meta_refresh，向量条数不变，检索元数据已刷新。"""
        ingest_docs(self.cfg, [CUDA_ISSUE], self.embed, self.store)
        n1 = self.store.count()
        changed = CUDA_ISSUE.model_copy(update={"version_span": VersionSpan(min="0.5.0", max="0.6.4")})
        stats = ingest_docs(self.cfg, [changed], self.embed, self.store)
        self.assertEqual(stats.get("meta_refresh", 0), 1)
        self.assertEqual(stats.get("embedded", -1), 0)
        self.assertEqual(self.store.count(), n1)  # 向量未变
        engine = SearchEngine(self.cfg)
        try:
            results = engine.search("illegal memory access", target_version="0.6.4", top_k=3)
            hit = next((r for r in results if r.doc_id == CUDA_ISSUE.source_id), None)
            self.assertIsNotNone(hit)
            # min 仍入库；max 一律不返回（历史列/派生值不泄露，查询期按仓库日历现算打分）
            self.assertEqual(hit.meta.get("version_span_min"), "0.5.0")
            self.assertIsNone(hit.meta.get("version_span_max"))
        finally:
            engine.close()

    def test_old_db_backfill_skips_without_reembed(self):
        """旧库（embed_hash 为空）且 content_hash 相同 -> 回填 embed_hash 并跳过，不重嵌。"""
        import sqlite3

        ingest_docs(self.cfg, [CUDA_ISSUE], self.embed, self.store)
        sqlite_path = self.cfg.resolve(self.cfg.storage.sqlite_path)
        conn = sqlite3.connect(str(sqlite_path))
        conn.execute("UPDATE docs SET embed_hash = NULL")
        conn.commit()
        conn.close()
        stats = ingest_docs(self.cfg, [CUDA_ISSUE], self.embed, self.store)
        self.assertEqual(stats.get("skipped_unchanged", 0), 1)
        self.assertEqual(stats.get("embedded", -1), 0)

    def test_dedupe_by_doc(self):
        """长 issue 切出多个 chunk，检索结果里同一 doc 只出现一次（保留最高分）。"""
        para = ("paged attention blocks allocation failed " + "x" * 1200) * 4  # 长正文 -> 多 chunk
        long_doc = KbDocument(
            source_type="github_issue",
            source_id="github:issue:20001",
            url="https://github.com/vllm-project/vllm/issues/20001",
            title="paged attention OOM long thread",
            body=para,
            created_at="2026-01-01T00:00:00Z",
            status="closed",
            labels=["bug"],
        )
        ingest_docs(self.cfg, [long_doc, CUDA_ISSUE], self.embed, self.store)
        self.assertGreater(self.store.count(), 2)  # long_doc 确实被切成多块
        engine = SearchEngine(self.cfg)
        try:
            results = engine.search("paged attention blocks", target_version="0.26.0", top_k=10)
            long_hits = [r for r in results if r.doc_id == long_doc.source_id]
            self.assertEqual(len(long_hits), 1, "同一 doc 的多个 chunk 应去重为一条")
            self.assertTrue(long_hits, "长 issue 应出现在结果中")
        finally:
            engine.close()

    def test_version_decay_visible(self):
        """修复落地上界（查询期按仓库日历现算）-> 超出上界的目标版本降权。"""
        ingest_docs(self.cfg, [CUDA_ISSUE], self.embed, self.store)
        engine = SearchEngine(self.cfg)
        try:
            # resolved 2024-03-10 -> 上界 v0.5.4（mock 仓库日历）
            with mock.patch.object(
                SearchEngine, "_calendar_for",
                return_value={
                    "v0.5.4": datetime(2024, 3, 5, tzinfo=timezone.utc),
                    "v0.6.1": datetime(2024, 4, 5, tzinfo=timezone.utc),
                },
            ):
                r_new = engine.search("CUDA illegal memory access", target_version="0.5.4")
                r_old = engine.search("CUDA illegal memory access", target_version="0.8.0")
            # 0.8.0 超出修复版本上界 0.5.4 很远 -> 版本权重低 -> final 分更低
            self.assertGreater(r_new[0].final, r_old[0].final)
            self.assertLess(r_old[0].confidence.version_weight, 0.5)
        finally:
            engine.close()

    def test_canonical_roundtrip(self):
        """统一 canonical 文件写读往返（--rebuild 依赖此路径）。"""
        canonical_path = self.cfg.resolve(self.cfg.storage.canonical_file)
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        with open(canonical_path, "w", encoding="utf-8") as f:
            for doc in [CUDA_ISSUE, OOM_ISSUE]:
                f.write(doc.model_dump_json() + "\n")
        from vllm_kb.github_pull import load_canonical

        docs = load_canonical(canonical_path)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].source_id, CUDA_ISSUE.source_id)
        self.assertEqual(json.loads(docs[1].model_dump_json())["status"], "closed")

    def test_canonical_roundtrip_with_line_separators(self):
        """回归：正文含 U+2028/U+2029/NEL 时，canonical 单文件仍保持一行一条（曾因此被拆行）。"""
        from vllm_kb.github_pull import load_canonical
        from vllm_kb.models import doc_to_json

        weird = KbDocument(
            source_type="github_issue",
            source_id="github:test:90001",
            url="https://example.com/90001",
            title="line separator issue",
            body="first line\u2028second paragraph\u2029third\x85fourth",
            created_at="2026-01-01T00:00:00Z",
            status="open",
        )
        canonical_path = self.cfg.resolve(self.cfg.storage.canonical_file)
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        with open(canonical_path, "w", encoding="utf-8") as f:
            f.write(doc_to_json(weird) + "\n")
        # 文件行数 == 文档数（不被 U+2028 等拆行）
        n_lines = len(canonical_path.read_text(encoding="utf-8").splitlines())
        self.assertEqual(n_lines, 1)
        docs = load_canonical(canonical_path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].body, weird.body)

    def test_ingest_batches_vector_writes(self):
        """回归：向量写必须攒批（LanceDB 单条 add/delete 随表增长退化 30~40 倍）。

        用 mock store 记录调用次数：多文档入库时 add_items/delete_docs
        应各只调用 1 次（或远少于文档数），而非每条文档一次。
        """
        from unittest import mock

        store = CountingStore()
        # 制造 > 200 条 chunk 的文档（触发多次 flush），验证 flush 次数远小于文档数
        docs = []
        for i in range(30):
            body = ("paged attention blocks allocation failed " * 200)  # 每条约 8 个 chunk
            docs.append(
                KbDocument(
                    source_type="github_issue",
                    source_id=f"github:issue:9{i:04d}",
                    url=f"https://github.com/vllm-project/vllm/issues/9{i:04d}",
                    title=f"bulk ingest test {i}",
                    body=body,
                    created_at="2026-01-01T00:00:00Z",
                    status="closed",
                    labels=["bug"],
                )
            )
        stats = ingest_docs(self.cfg, docs, self.embed, store)
        self.assertGreater(stats["embedded"], 30, "应产生多 chunk")
        self.assertLess(store.add_calls, 30, f"add 应攒批调用（实际 {store.add_calls} 次）")
        self.assertEqual(store.added, stats["embedded"])
        self.assertEqual(sorted(store.deleted), sorted(d.source_id for d in docs))

    def test_ingest_batches_embedding_calls(self):
        """回归：嵌入必须跨文档攒批（每文档一次 API 调用 = 5 万次网络往返）。

        用 mock embedding client 统计调用次数与每次入参条数：
        多文档入库时调用次数应远小于文档数，且每次入参 ≥ 2 条文本。
        """
        class CountingEmbed:
            def __init__(self):
                self.calls = 0
                self.total_texts = 0

            def embed_texts(self, texts):
                self.calls += 1
                self.total_texts += len(texts)
                # echo 风格确定性向量（无需真实 API）
                from vllm_kb.embed import EmbeddingClient
                from vllm_kb.config import EmbeddingCfg
                fake = EmbeddingClient(EmbeddingCfg(provider="echo", dimensions=1024))
                return fake.embed_texts(texts)

        docs = []
        for i in range(12):
            body = ("paged attention blocks allocation failed " * 200)  # 每条约 8 个 chunk
            docs.append(
                KbDocument(
                    source_type="github_issue",
                    source_id=f"github:issue:8{i:04d}",
                    url=f"https://github.com/vllm-project/vllm/issues/8{i:04d}",
                    title=f"embed batching test {i}",
                    body=body,
                    created_at="2026-01-01T00:00:00Z",
                    status="closed",
                    labels=["bug"],
                )
            )
        emb = CountingEmbed()
        store = CountingStore()
        stats = ingest_docs(self.cfg, docs, emb, store)
        # 12 文档 × ~3 chunk = 36 chunks：应合并为 1 次 embedding 调用（而非 12 次）
        self.assertGreater(stats["embedded"], 30)
        self.assertLess(emb.calls, 12, f"embedding 应攒批调用（实际 {emb.calls} 次）")
        self.assertEqual(emb.total_texts, stats["embedded"])

    def test_search_degrades_to_fts_without_embedding(self):
        """只读检索不依赖密钥：embedding 失败时降级为全文检索，不抛异常（回归）。"""
        from unittest import mock

        from vllm_kb.search import SearchEngine

        # 正常入库（echo embedding）
        ingest_docs(self.cfg, [CUDA_ISSUE], self.embed, self.store)
        engine = SearchEngine(self.cfg, read_only=True)
        try:
            # mock embedding 抛错（模拟无 key / API 401）
            with mock.patch.object(engine.embed, "embed", side_effect=RuntimeError("embedding API 401: Token is invalid")):
                results = engine.search("illegal memory access", target_version="0.6.0")
            self.assertTrue(results, "降级后仍应返回 FTS 结果")
            self.assertIsNotNone(engine._embed_error)
            self.assertIn("401", engine._embed_error or "")
        finally:
            engine.close()

    def test_search_component_filter_applies(self):
        """--component 应同时过滤结果（FTS 降级路径也生效）。"""
        from unittest import mock

        from vllm_kb.search import SearchEngine

        ingest_docs(self.cfg, [CUDA_ISSUE, OOM_ISSUE], self.embed, self.store)
        engine = SearchEngine(self.cfg, read_only=True)
        try:
            with mock.patch.object(engine.embed, "embed", side_effect=RuntimeError("no key")):
                results = engine.search(
                    "illegal memory access", target_version="0.6.0",
                    filters={"component": "vllm"},
                )
            # CUDA_ISSUE 组件为 vllm（默认），应保留；全库无其他组件时仍返回
            self.assertTrue(all(r.component == "vllm" for r in results))
        finally:
            engine.close()

    def test_embedding_circuit_breaker(self):
        """熔断器：连续失败后打开（跳过 embed 调用零等待降级），到期自动探测恢复。"""
        from unittest import mock

        from vllm_kb.search import SearchEngine

        ingest_docs(self.cfg, [CUDA_ISSUE], self.embed, self.store)
        engine = SearchEngine(self.cfg, read_only=True)
        try:
            # 连续失败达到阈值 -> 熔断打开
            with mock.patch.object(engine.embed, "embed", side_effect=RuntimeError("svc down")):
                for _ in range(engine._EMBED_CIRCUIT_FAIL_THRESHOLD):
                    engine.search("illegal memory access", target_version="0.6.0")
                engine.search("illegal memory access", target_version="0.6.0")
            self.assertFalse(engine._embed_available(), "连续失败后熔断应打开")
            self.assertIn("熔断", engine._embed_error or "")

            # 熔断打开期间：embed 不应再被调用（零等待降级）
            with mock.patch.object(engine.embed, "embed", side_effect=AssertionError("熔断期不应调用 embed")) as m:
                engine.search("illegal memory access", target_version="0.6.0")
                m.assert_not_called()

            # 熔断到期：半开探测，成功即恢复
            engine._embed_circuit_open_until = time.time() - 1  # 手动拨快时钟
            with mock.patch.object(engine.embed, "embed", return_value=[0.1] * 8) as m:
                engine.search("illegal memory access", target_version="0.6.0")
                m.assert_called_once()
            self.assertTrue(engine._embed_available(), "探测成功后熔断应关闭")
            self.assertIsNone(engine._embed_error)
        finally:
            engine.close()


class TestCanonicalUpsert(unittest.TestCase):
    """统一 canonical upsert：新增 / 更新（修改的 PDF 回写）/ 跳过（幂等）/ 不覆盖其他来源。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def path(self):
        return Path(self.cfg.storage.canonical_file)

    def test_add_skip_update(self):
        from vllm_kb.pipeline import upsert_unified_canonical

        # 新增 2 条
        r = upsert_unified_canonical(self.cfg, [CUDA_ISSUE, OOM_ISSUE])
        self.assertEqual(r, {"added": 2, "updated": 0, "skipped": 0})
        lines = self.path().read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        # 幂等：内容相同 → 全部跳过（不重写）
        r2 = upsert_unified_canonical(self.cfg, [CUDA_ISSUE, OOM_ISSUE])
        self.assertEqual(r2, {"added": 0, "updated": 0, "skipped": 2})
        self.assertEqual(len(self.path().read_text(encoding="utf-8").splitlines()), 2)
        # 更新：标题变化（模拟修改过的 PDF 重新解析）→ updated，行内容回写
        changed = CUDA_ISSUE.model_copy(update={"title": "修改后的标题"})
        r3 = upsert_unified_canonical(self.cfg, [changed])
        self.assertEqual(r3, {"added": 0, "updated": 1, "skipped": 0})
        first = json.loads(self.path().read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(first["title"], "修改后的标题")

    def test_preserves_other_sources(self):
        """逐来源处理：upsert 本来源时，canonical 中其他来源的行必须保留。"""
        from vllm_kb.pipeline import upsert_unified_canonical

        self.path().write_text(
            '{"source_id": "github:issue:999", "source_type": "github_issue", "title": "旧", "body": "x"}\n',
            encoding="utf-8",
        )
        r = upsert_unified_canonical(self.cfg, [FEATURE_ISSUE])
        self.assertEqual(r["added"], 1)
        lines = self.path().read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)  # 旧行 + 新行
        ids = {json.loads(x)["source_id"] for x in lines}
        self.assertEqual(ids, {"github:issue:999", "github:issue:10003"})


if __name__ == "__main__":
    unittest.main()
