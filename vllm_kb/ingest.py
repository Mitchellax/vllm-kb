"""入库：canonical 文档 -> 向量库（LanceDB/Python）+ SQLite（docs 元数据 + FTS5 全文）。

幂等 + 增量断点续传（双哈希）：
- embed_hash：由 source_id+title+body 决定 —— 真正影响嵌入向量的内容；
- meta_hash：整篇 canonical（含版本区间/组件/kind 等元数据）。
分类（预扫描打印统计）：
- 两哈希均未变        -> 跳过（不重新嵌入，崩溃续传也按此粒度恢复）；
- embed 未变、meta 变 -> 只刷新元数据（docs 行 + 向量 meta），不重新嵌入；
- embed 变了/新文档    -> 全量重嵌。
换 embedding 模型仍用 --rebuild 全量重建。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Any

from .chunking import chunk_doc
from .confidence import reliability_score
from .config import AppConfig
from .embed import EmbeddingClient
from .models import KbDocument
from .tagging import merge_final
from .vectorstore import BaseVectorStore, VectorItem

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
  source_id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  url TEXT,
  title TEXT,
  created_at TEXT,
  resolved_at TEXT,
  status TEXT,
  labels TEXT,
  version_span_min TEXT,
  version_span_max TEXT,
  reliability REAL,
  component TEXT,
  content_hash TEXT,
  embed_hash TEXT,
  extra TEXT,
  tags TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED, doc_id UNINDEXED, text
);
CREATE TABLE IF NOT EXISTS chunks_meta (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  seq INTEGER,
  section TEXT
);
-- 文档级标签覆盖层（人工治理状态）：
--   auto_snapshot = 最近一次入库的自动标签快照（审核页展示，随入库刷新）；
--   excluded = 人工排除的自动标签（可恢复）；manual = 人工添加的标签。
-- 最终标签 = (auto − excluded) ∪ manual（见 tagging.merge_final，ingest 与 build_graph 共用）。
CREATE TABLE IF NOT EXISTS doc_tags (
  source_id TEXT PRIMARY KEY,
  auto_snapshot TEXT,
  excluded TEXT,
  manual TEXT,
  updated_at TEXT,
  reviewer TEXT
);
"""


def _connect(sqlite_path: Path) -> sqlite3.Connection:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sqlite_path))
    conn.executescript(_SCHEMA)
    # 旧库迁移：补缺失列（老版本建的表没有）
    cols = [r[1] for r in conn.execute("PRAGMA table_info(docs)")]
    for col, ddl in (
        ("content_hash", "ALTER TABLE docs ADD COLUMN content_hash TEXT"),
        ("embed_hash", "ALTER TABLE docs ADD COLUMN embed_hash TEXT"),
        ("component", "ALTER TABLE docs ADD COLUMN component TEXT"),
        ("tags", "ALTER TABLE docs ADD COLUMN tags TEXT"),
    ):
        if col not in cols:
            conn.execute(ddl)
    mcols = [r[1] for r in conn.execute("PRAGMA table_info(chunks_meta)")]
    if "section" not in mcols:
        conn.execute("ALTER TABLE chunks_meta ADD COLUMN section TEXT")
    conn.commit()
    return conn


def _hashes(doc: KbDocument) -> tuple[str, str]:
    """返回 (embed_hash, meta_hash)。

    embed_hash 只覆盖 source_id+title+body（嵌入输入）；meta_hash 覆盖整篇，
    因此版本/组件/kind 等元数据变化不会触发重嵌。
    """
    from .models import doc_to_json

    meta_hash = hashlib.sha256(doc_to_json(doc).encode("utf-8")).hexdigest()
    embed_src = json.dumps(
        {"id": doc.source_id, "title": doc.title, "body": doc.body},
        ensure_ascii=True,
        sort_keys=True,
    )
    embed_hash = hashlib.sha256(embed_src.encode("utf-8")).hexdigest()
    return embed_hash, meta_hash


def _log_progress(
    processed: int,
    total: int,
    start_ts: float,
    stats: dict,
    step: int,
    samples: deque[tuple[float, int]] | None = None,
) -> None:
    """打印入库进度（标准库实现，无第三方依赖）。

    节流：第 1 条、每 step 条、最后一条打印；step 默认 total//100（约 100 行）。
    嵌入是最慢步骤，进度按文档数计，附带已嵌入 chunk 数与 ETA。

    ETA 用近 WINDOW_SECS 秒滑动窗口的实测速率线性外推（对速率波动更敏感），
    窗口内样本不足（<2 个或窗口时长过短）时回退整体平均速率，避免初期抖动。
    """
    if total <= 0:
        return
    if processed < total and processed % step != 0 and processed != 1:
        return
    elapsed = time.time() - start_ts
    rate = _window_rate(samples, processed, elapsed) if samples else 0.0
    eta_s = (total - processed) / rate if rate > 0 else 0.0
    pct = processed / total * 100
    print(
        f"[ingest] {processed}/{total} ({pct:.1f}%) | "
        f"embedded={stats['embedded']} chunks={stats['chunks']} "
        f"skipped={stats['skipped_unchanged']} meta_refresh={stats['meta_refresh']} | "
        f"{elapsed:.0f}s | ETA {eta_s / 60:.1f}min",
        flush=True,
    )


# 滑动窗口时长：ETA 外推基于最近这么多秒的实测速率
_WINDOW_SECS = 60.0


def _window_rate(
    samples: deque[tuple[float, int]], processed: int, elapsed: float
) -> float:
    """近 _WINDOW_SECS 秒的滑动窗口速率（docs/s）。

    取窗口内最早样本与当前(processed, now)的差分；样本不足或窗口太短
    （<10s）时回退整体平均（processed/elapsed），避免早期 ETA 剧烈跳动。
    """
    now = time.time()
    while samples and now - samples[0][0] > _WINDOW_SECS:
        samples.popleft()
    if len(samples) >= 2:
        t0, p0 = samples[0]
        window_dt = now - t0
        if window_dt >= 10.0:
            rate = (processed - p0) / window_dt
            if rate > 0:
                return rate
    return processed / elapsed if elapsed > 0 else 0.0


def chunk_meta(doc: KbDocument, reliability: float, tags: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "doc_id": doc.source_id,
        "source_type": doc.source_type,
        "url": doc.url,
        "title": doc.title,
        "created_at": doc.created_at,
        "resolved_at": doc.resolved_at,
        "status": doc.status,
        "labels": doc.labels,
        "tags": list(tags if tags is not None else doc.tags),
        # version_span_max 不再写入（历史列含旧版日历推导的跨仓库错配值；修复落地上界
        # 一律查询期按仓库日历现算，不落库、不随 meta/API 返回）
        "version_span_min": doc.version_span.min,
        "component": doc.component,
        "kind": doc.extra.get("kind", ""),
        "reliability": reliability,
    }


def _upsert_docs_row(conn: sqlite3.Connection, doc: KbDocument, rel: float,
                     embed_hash: str, meta_hash: str,
                     tags: Optional[list[str]] = None) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO docs
           (source_id, source_type, url, title, created_at, resolved_at, status,
            labels, version_span_min, version_span_max, reliability, component,
            content_hash, embed_hash, extra, tags)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            doc.source_id,
            doc.source_type,
            doc.url,
            doc.title,
            doc.created_at,
            doc.resolved_at,
            doc.status,
            json.dumps(doc.labels, ensure_ascii=False),
            doc.version_span.min,
            doc.version_span.max,
            rel,
            doc.component,
            meta_hash,
            embed_hash,
            json.dumps(doc.extra, ensure_ascii=False),
            json.dumps(list(tags if tags is not None else doc.tags), ensure_ascii=False),
        ),
    )


def ingest_docs(
    cfg: AppConfig,
    docs: list[KbDocument],
    embed_client: EmbeddingClient,
    vector_store: BaseVectorStore,
    sqlite_path: Path | None = None,
) -> dict:
    """入库（幂等 + 增量断点续传）。返回统计。"""
    sqlite_path = sqlite_path or cfg.resolve(cfg.storage.sqlite_path)
    conn = _connect(sqlite_path)
    # 人工标签覆盖层（doc_tags：excluded/manual）——最终标签 = (auto − excluded) ∪ manual
    from .review import load_doc_tags_conn, upsert_auto_snapshot_conn

    overlay = load_doc_tags_conn(conn)
    stats = {"docs": 0, "chunks": 0, "embedded": 0, "skipped_unchanged": 0, "meta_refresh": 0, "skipped_empty": 0}

    total = len(docs)
    step = max(1, total // 100)
    start_ts = time.time()
    processed = 0

    # ---- 批量写缓冲：LanceDB 单条 add/delete 都要 commit 版本文件（大表下极慢），
    #      攒批后一次性写入（add 与 delete 各自攒批，flush 时先删后加）。 ----
    _FLUSH_BATCH = 200  # 攒够 200 条就 flush 一次
    pending_adds: list[VectorItem] = []
    pending_deletes: list[str] = []

    def _flush_vector() -> None:
        if pending_deletes:
            vector_store.delete_docs(pending_deletes)
            pending_deletes.clear()
        if pending_adds:
            vector_store.add_items(pending_adds)
            pending_adds.clear()

    # ---- 批量嵌入缓冲：embedding API 每次调用都有网络往返，跨文档攒批减少调用次数。
    #      攒够 _EMBED_FLUSH_CHUNKS 个 chunk 就一次 embed_texts（内部再按 batch_size 分批）。 ----
    _EMBED_FLUSH_CHUNKS = 64  # 64 个 chunk / 32 batch = 2 次 API 调用一批
    pending_embed: list = []  # [(KbDocument, list[KbChunk])]

    def _flush_embed() -> None:
        """嵌入缓冲 -> 向量/SQLite 写库（向量写仍走攒批）。"""
        if not pending_embed:
            return
        all_chunks = [c for _, chs in pending_embed for c in chs]
        vectors = embed_client.embed_texts([c.text for c in all_chunks])
        if len(vectors) != len(all_chunks):
            raise RuntimeError(f"embedding 返回数 {len(vectors)} != chunk 数 {len(all_chunks)}")
        idx = 0
        for doc, chunks in pending_embed:
            doc_vecs = vectors[idx : idx + len(chunks)]
            idx += len(chunks)
            eh, mh = _hashes(doc)
            rel = reliability_score(
                doc.source_type, doc.status, doc.resolved_at, doc.reliability, cfg.confidence,
                kind=doc.extra.get("kind", ""),
            )
            # 最终标签 = (自动 − 排除) ∪ 人工（覆盖层见 doc_tags 表）
            ov = overlay.get(doc.source_id, {})
            final_tags = merge_final(doc.tags or [], ov.get("excluded", []), ov.get("manual", []))
            # 幂等：先删旧（攒批，flush 时先删后加）
            pending_deletes.append(doc.source_id)
            conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc.source_id,))
            conn.execute("DELETE FROM chunks_meta WHERE doc_id = ?", (doc.source_id,))
            # 写向量（攒批）：chunk meta 带 section（所属章节标题）+ tags（最终标签）
            base_meta = chunk_meta(doc, rel, tags=final_tags)
            items = []
            for c, v in zip(chunks, doc_vecs):
                m = dict(base_meta)
                if c.section:
                    m["section"] = c.section
                items.append(VectorItem(id=c.chunk_id, vector=v, meta=m, text=c.text))
            pending_adds.extend(items)
            # 写 SQLite
            _upsert_docs_row(conn, doc, rel, eh, mh, tags=final_tags)
            upsert_auto_snapshot_conn(conn, doc.source_id, doc.tags or [])
            for c in chunks:
                conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, doc_id, text) VALUES (?,?,?)",
                    (c.chunk_id, doc.source_id, c.text),
                )
                conn.execute(
                    "INSERT INTO chunks_meta (chunk_id, doc_id, seq, section) VALUES (?,?,?,?)",
                    (c.chunk_id, doc.source_id, c.seq, c.section),
                )
            stats["docs"] += 1
            stats["chunks"] += len(chunks)
            stats["embedded"] += len(doc_vecs)
            if len(pending_adds) >= _FLUSH_BATCH or len(pending_deletes) >= _FLUSH_BATCH:
                _flush_vector()
        pending_embed.clear()

    # ---- 预扫描：分类 跳过 / 仅刷新元数据 / 需嵌入 ----
    existing: dict[str, tuple[str, str]] = {}
    for sid, mh, eh in conn.execute("SELECT source_id, content_hash, embed_hash FROM docs"):
        existing[sid] = (mh, eh)
    plan = {"embed": 0, "skip": 0, "refresh": 0}
    for doc in docs:
        eh, mh = _hashes(doc)
        stored = existing.get(doc.source_id)
        if stored:
            sm, se = stored
            if se is None and sm == mh:
                se = eh  # 旧库兼容：content_hash 相同视为已嵌入，回填 embed_hash
                conn.execute("UPDATE docs SET embed_hash = ? WHERE source_id = ?", (eh, doc.source_id))
            if se == eh:
                plan["skip" if sm == mh else "refresh"] += 1
            else:
                plan["embed"] += 1
        else:
            plan["embed"] += 1
    print(
        f"[ingest] 预扫描 {total} 条：需嵌入 {plan['embed']}，跳过 {plan['skip']}（内容未变），"
        f"仅刷新元数据 {plan['refresh']}（版本/组件/kind 变化，不重嵌）",
        flush=True,
    )

    samples: deque[tuple[float, int]] = deque()  # (时间戳, 已处理文档数)，ETA 滑动窗口用
    for doc in docs:
        processed += 1
        try:
            eh, mh = _hashes(doc)
            stored = existing.get(doc.source_id)
            if stored:
                sm, se = stored
                if se is None and sm == mh:
                    se = eh
                if se == eh:
                    if sm == mh:
                        stats["skipped_unchanged"] += 1
                        continue
                    # 仅元数据变化：刷新 docs 行 + 向量 meta，不重新嵌入
                    rel = reliability_score(
                        doc.source_type, doc.status, doc.resolved_at, doc.reliability,
                        cfg.confidence, kind=doc.extra.get("kind", ""),
                    )
                    ov = overlay.get(doc.source_id, {})
                    final_tags = merge_final(doc.tags or [], ov.get("excluded", []),
                                             ov.get("manual", []))
                    _upsert_docs_row(conn, doc, rel, eh, mh, tags=final_tags)
                    vector_store.update_doc_meta(doc.source_id, chunk_meta(doc, rel, tags=final_tags))
                    upsert_auto_snapshot_conn(conn, doc.source_id, doc.tags or [])
                    stats["meta_refresh"] += 1
                    continue

            # ---- 全量路径：版本补全 + 可靠度 + 分块（嵌入与写库攒批） ----
            rel = reliability_score(
                doc.source_type, doc.status, doc.resolved_at, doc.reliability, cfg.confidence,
                kind=doc.extra.get("kind", ""),
            )
            chunks = chunk_doc(doc, cfg.chunking.max_chunk_chars, cfg.chunking.overlap_chars)
            if not chunks:
                # 空正文文档：仍记录哈希（含元数据），下次预扫描直接跳过，不重扫
                ov = overlay.get(doc.source_id, {})
                final_tags = merge_final(doc.tags or [], ov.get("excluded", []),
                                         ov.get("manual", []))
                _upsert_docs_row(conn, doc, rel, eh, mh, tags=final_tags)
                upsert_auto_snapshot_conn(conn, doc.source_id, doc.tags or [])
                stats["skipped_empty"] += 1
                continue
            pending_embed.append((doc, chunks))
            if sum(len(c) for _, c in pending_embed) >= _EMBED_FLUSH_CHUNKS:
                _flush_embed()
        finally:
            samples.append((time.time(), processed))
            _log_progress(processed, total, start_ts, stats, step, samples)

    _flush_embed()   # 收尾：嵌入 + 写库残余缓冲
    _flush_vector()  # 收尾：清空残余批量写缓冲
    conn.commit()
    conn.close()
    return stats


def rebuild(cfg: AppConfig) -> dict:
    """全量重建：清空向量库与 SQLite，从统一 canonical.jsonl 重新入库（换 embedding 模型后使用）。"""
    from .embed import EmbeddingClient
    from .github_pull import load_canonical
    from .vectorstore import build_vector_store

    print("[rebuild] 清空向量库 ...", flush=True)
    vector_store = build_vector_store(cfg)
    vector_store.clear()
    sqlite_path = cfg.resolve(cfg.storage.sqlite_path)
    if sqlite_path.exists():
        sqlite_path.unlink()
    print("[rebuild] 加载统一 canonical ...", flush=True)
    docs = load_canonical(cfg.resolve(cfg.storage.canonical_file))
    print(f"[rebuild] canonical {len(docs)} 条，开始入库（嵌入为最慢步骤，见 [ingest] 进度）", flush=True)
    embed_client = EmbeddingClient(cfg.embedding)
    stats = ingest_docs(cfg, docs, embed_client, vector_store, sqlite_path)
    stats["total_docs"] = len(docs)
    print(f"[rebuild] 完成: {stats}", flush=True)
    return stats
