"""检索：混合召回（向量 + FTS5）-> 组件配套扩展 -> 置信度重排 -> 证据输出。

- 向量召回：语义 top-k（LanceDB / Python 兜底）；
- 全文召回：SQLite FTS5（BM25），覆盖关键词精确匹配；
- 组件查询："vllm-ascend:0.18.0 GLM5.1 PD分离P节点挂死" 解析出 (组件, 版本, 语义词)，
  语义词用于嵌入；按组件配套矩阵反向展开（expand），把其他组件文档（如 vllm）
  按其配套版本参与打分——vllm 知识只记 vllm 自己的版本，通过配套关联到 vllm-ascend 提问；
- 重排：sim^gamma * confidence^(1-gamma)，每个文档按其生效版本参考动态计算置信度；
- 未解决兜底：无强匹配已解决问题时，优先列接近版本的未解决问题（含工程规避方案）。
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .companion import CompanionMatrix, parse_component_query
from .confidence import ConfidenceBreakdown, compute_confidence, final_score, version_weight
from .config import AppConfig
from .embed import EmbeddingClient
from .vectorstore import BaseVectorStore, ReadOnlyVectorStore, build_vector_store

_FTS_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    title: str
    url: str
    similarity: float
    confidence: ConfidenceBreakdown
    final: float
    source: str  # vector | fts | both
    component: str = ""
    resolved: bool = True  # 是否已解决（closed/merged 且有 resolved_at）
    version_ref: str = ""  # 该文档打分时使用的生效版本参考（查询组件版本或配套版本）
    meta: dict[str, Any] = field(default_factory=dict)


class SearchEngine:
    def __init__(self, cfg: AppConfig, read_only: bool = False):
        """read_only=True：结构只读 —— SQLite mode=ro 连接 + 向量库只读包装，
        任何写路径（INSERT/UPDATE/向量写入）硬失败，供检索 API 使用。"""
        self.cfg = cfg
        self.read_only = read_only
        self.embed = EmbeddingClient(cfg.embedding)
        store = build_vector_store(cfg)
        self.vector_store: BaseVectorStore = ReadOnlyVectorStore(store) if read_only else store
        self.sqlite_path: Path = cfg.resolve(cfg.storage.sqlite_path)
        self.companion: Optional[CompanionMatrix] = CompanionMatrix.load(
            cfg.storage.companion_file
        )
        self._conn: Optional[sqlite3.Connection] = None
        self._warned_no_version = False
        self.last_context: dict[str, Any] = {}  # 最近一次查询的组件上下文（verify/agent 展示用）
        self._embed_error: Optional[str] = None  # 查询时 embedding 失败原因（降级提示）

    @property
    def conn(self) -> sqlite3.Connection:
        if self.read_only:
            raise RuntimeError("只读模式下禁止使用可写连接，请用 _ro_conn()")
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.sqlite_path))
        return self._conn

    def _ro_conn(self) -> sqlite3.Connection:
        """URI 级只读 SQLite 连接：文件缺失或任何写操作都会抛错。"""
        uri = f"file:{self.sqlite_path.as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---------------- 检索 ----------------

    def search(
        self,
        query: str,
        target_version: Optional[str] = None,
        now: Optional[datetime] = None,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,  # 如 {"status": "closed", "component": "vllm-ascend", "min_created": "..."}
        component: Optional[str] = None,
        version: Optional[str] = None,
    ) -> list[SearchResult]:
        """检索。

        - 组件查询：query 形如 "vllm-ascend:0.18.0 GLM5.1 PD分离P节点挂死"，
          或显式传 component/version（优先于 query 前缀解析）；
        - target_version：普通（无组件）查询的版本参数，沿用旧语义。
        """
        parsed_comp, parsed_ver, semantic = parse_component_query(query)
        comp = component or parsed_comp
        ver = version or parsed_ver
        semantic = semantic or query

        # 生效的目标版本：组件查询用组件版本（每个文档按配套关系取自己的参考版本）；
        # 普通查询沿用 target_version 参数或配置默认。
        if comp is None:
            target_version = target_version or self.cfg.retrieval.default_target_version or None
            if target_version is None and not self._warned_no_version:
                self._warned_no_version = True
                print(
                    "[warn] 未指定目标版本/组件版本（w_ver 取默认值）："
                    "建议用 'vllm-ascend:0.18.0 问题描述' 或 target_version='0.26.0' 查询"
                )
        final_top_k = top_k or self.cfg.retrieval.final_top_k

        # 组件配套反向展开（"组件:版本" -> 其他组件 -> 配套版本列表）
        companion_ctx: dict[str, list[str]] = {}
        if comp and ver and self.companion:
            companion_ctx = self.companion.expand(comp, ver)
            if companion_ctx:
                detail = ", ".join(f"{k} {v}" for k, v in companion_ctx.items())
                print(f"[companion] {comp}:{ver} -> {detail}")
        self.last_context = {
            "component": comp,
            "version": ver,
            "semantic_query": semantic,
            "companions": companion_ctx,
        }

        # 查询向量：embedding 失败（无 key/网络）时优雅降级为 FTS-only 检索，
        # 不依赖密钥即可完成查询（只读检索不应被更新所需的密钥阻塞）。
        try:
            q_vec = self.embed.embed(semantic)
            embed_ok = True
        except Exception as e:
            embed_ok = False
            self._embed_error = str(e)[:200]
            print(f"[warn] embedding 不可用（{type(e).__name__}: {str(e)[:80]}），降级为全文检索")

        # 只读模式：本次查询用独立的只读 SQLite 连接（线程安全 + 写必败）
        conn = self._ro_conn() if self.read_only else self.conn

        # 1) 向量召回（embedding 可用时）
        vector_hits = {}
        if embed_ok:
            vector_hits = {
                h.id: h
                for h in self.vector_store.search(q_vec, self.cfg.retrieval.vector_top_k)
            }
        # 2) 全文召回（BM25），并与向量结果合并（只读连接仅在 FTS 阶段使用，用后即关）
        try:
            fts_hits = self._fts_search(semantic, self.cfg.retrieval.fts_top_k, conn=conn)
            merged: dict[str, dict] = {}
            for hid, hit in vector_hits.items():
                m = dict(hit.meta or {})
                # 向量路径的 chunk meta（入库时嵌入）不含 docs.extra 的验证状态——
                # 用 SQLite 补全 verification（expert/tested/unverified 参与置信度）
                try:
                    fm = self._doc_meta_from_fts(hit.text, hid, conn=conn)
                    if fm and fm.get("verification"):
                        m["verification"] = fm["verification"]
                except Exception:
                    pass
                merged[hid] = {"sim": hit.score, "meta": m, "text": hit.text, "src": "vector"}
            for hid, (sim, text) in fts_hits.items():
                if hid in merged:
                    merged[hid]["src"] = "both"
                    merged[hid]["sim"] = max(merged[hid]["sim"], sim)
                else:
                    merged[hid] = {"sim": sim, "meta": self._doc_meta_from_fts(text, hid, conn=conn), "text": text, "src": "fts"}
        finally:
            if self.read_only:
                conn.close()

        # 3) 过滤
        if filters:
            merged = {
                hid: m for hid, m in merged.items() if self._pass_filters(m["meta"], filters)
            }

        # 4) 置信度重排（每个文档按其生效版本参考计算）
        c_cfg = self.cfg.confidence
        results = []
        for hid, m in merged.items():
            if m["sim"] < self.cfg.retrieval.min_similarity:
                continue
            meta = m["meta"]
            doc_comp = meta.get("component", "")
            version_ref, w_ver = self._resolve_version_ref(
                doc_comp, meta.get("version_span_min"), meta.get("version_span_max"),
                comp, ver, companion_ctx,
            )
            if version_ref is None:
                version_ref = target_version
            conf = compute_confidence(
                created_at=meta.get("created_at"),
                resolved_at=meta.get("resolved_at"),
                status=meta.get("status", "open"),
                source_type=meta.get("source_type", "github_issue"),
                span_min=meta.get("version_span_min"),
                span_max=meta.get("version_span_max"),
                target_version=version_ref,
                now=now,
                cfg=c_cfg,
                # 不信任入库时存储的 reliability（可能早于 kind 规则生成）：
                # 查询时按字段 + kind 现场重算，保证 kind 降权始终生效
                kind=meta.get("kind"),
                # 验证状态（维度 B）：expert/tested/unverified 作为可靠度下限提升
                verification=meta.get("verification"),
            )
            resolved = meta.get("status") in ("closed", "merged") and bool(meta.get("resolved_at"))
            results.append(
                SearchResult(
                    chunk_id=hid,
                    doc_id=meta.get("doc_id", ""),
                    text=m["text"],
                    title=meta.get("title", ""),
                    url=meta.get("url", ""),
                    similarity=round(m["sim"], 4),
                    confidence=conf,
                    final=round(final_score(m["sim"], conf.score, c_cfg.gamma), 4),
                    source=m["src"],
                    component=doc_comp,
                    resolved=resolved,
                    version_ref=version_ref or "",
                    meta=meta,
                )
            )

        results.sort(key=lambda r: r.final, reverse=True)
        # 未解决兜底：没有强匹配的已解决问题时，优先列接近版本的未解决问题（含规避方案）
        if self.cfg.retrieval.prefer_unresolved_without_resolved:
            resolved_list = [r for r in results if r.resolved]
            unresolved_list = [r for r in results if not r.resolved]
            if resolved_list:
                best_resolved_sim = max(r.similarity for r in resolved_list)
                if best_resolved_sim < self.cfg.retrieval.resolved_min_similarity:
                    results = (
                        sorted(unresolved_list, key=lambda r: r.final, reverse=True)
                        + sorted(resolved_list, key=lambda r: r.final, reverse=True)
                    )
        if self.cfg.retrieval.dedupe_by_doc:
            # 同一文档的多个 chunk 只保留最高分的一条，避免占满结果位（长 issue 可能切出多块）
            seen: set[str] = set()
            deduped: list[SearchResult] = []
            for r in results:
                if r.doc_id in seen:
                    continue
                seen.add(r.doc_id)
                deduped.append(r)
            results = deduped
        return results[:final_top_k]

    def _resolve_version_ref(
        self,
        doc_component: str,
        span_min: Optional[str],
        span_max: Optional[str],
        query_component: Optional[str],
        query_version: Optional[str],
        companion_ctx: dict[str, list[str]],
    ) -> tuple[Optional[str], Optional[float]]:
        """确定该文档打分时的生效版本参考，返回 (version_ref, w_ver)。

        - 文档组件 == 查询组件：参考版本 = 查询版本；
        - 文档组件在配套矩阵中：参考版本 = 使 w_ver 最大的配套版本（反向配套关联）；
        - 否则：无参考（w_ver 取默认值）。
        """
        c = self.cfg.confidence
        if not query_component or not query_version:
            return None, None
        if doc_component == query_component:
            return query_version, version_weight(
                span_min, span_max, query_version, c.version_sigma, c.unknown_version_weight
            )
        candidates = companion_ctx.get(doc_component, [])
        if candidates:
            best = max(
                candidates,
                key=lambda cv: version_weight(
                    span_min, span_max, cv, c.version_sigma, c.unknown_version_weight
                ),
            )
            return best, version_weight(
                span_min, span_max, best, c.version_sigma, c.unknown_version_weight
            )
        return None, None

    # ---------------- FTS ----------------

    def _fts_search(self, query: str, limit: int, conn: Optional[sqlite3.Connection] = None) -> dict[str, tuple[float, str]]:
        """FTS5 短语匹配；失败则退化为单 token 前缀。返回 {chunk_id: (bm25_sim, text)}。"""
        conn = conn or self.conn
        toks = _FTS_TOKEN_RE.findall(query.lower())
        if not toks:
            return {}
        phrase = '"' + " ".join(toks) + '"'
        rows = self._fts_query(phrase, limit, conn=conn)
        if not rows:
            # 退化为 token 前缀 OR 匹配
            fallback = " OR ".join(f'"{t}"*' for t in toks[:8])
            rows = self._fts_query(fallback, limit, conn=conn)
        return {r[0]: (r[1], r[2]) for r in rows}

    def _fts_query(self, match_expr: str, limit: int, conn: Optional[sqlite3.Connection] = None) -> list[tuple[str, float, str]]:
        conn = conn or self.conn
        try:
            cur = conn.execute(
                """SELECT chunk_id, bm25(chunks_fts) AS s, text
                   FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?""",
                (match_expr, limit),
            )
            out = []
            for chunk_id, s, text in cur.fetchall():
                # FTS5 bm25() 返回负值，越负（|s| 越大）匹配越好（ORDER BY s ASC）。
                # 映射为 0..1 相似度：随 |s| 单调递增的饱和函数；
                # 上限 0.5：FTS 仅是兜底信号，不应压过真实的向量语义命中。
                b = abs(float(s))
                sim = min(0.5, b / (b + 3.0))
                out.append((chunk_id, sim, text))
            return out
        except sqlite3.OperationalError:
            return []

    def _doc_meta_from_fts(self, text: str, chunk_id: str, conn: Optional[sqlite3.Connection] = None) -> dict[str, Any]:
        conn = conn or self.conn
        doc_id = conn.execute(
            "SELECT doc_id FROM chunks_meta WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if not doc_id:
            return {}
        row = conn.execute(
            """SELECT source_type, url, title, created_at, resolved_at, status,
                      labels, version_span_min, version_span_max, reliability, component, extra
               FROM docs WHERE source_id = ?""",
            (doc_id[0],),
        ).fetchone()
        if not row:
            return {}
        import json as _json

        extra = _json.loads(row[11] or "{}")
        meta = {
            "doc_id": doc_id[0],
            "source_type": row[0],
            "url": row[1],
            "title": row[2],
            "created_at": row[3],
            "resolved_at": row[4],
            "status": row[5],
            "labels": _json.loads(row[6] or "[]"),
            "version_span_min": row[7],
            "version_span_max": row[8],
            "reliability": row[9],
            "component": row[10] or "",
            "kind": extra.get("kind", ""),
            "verification": extra.get("verification", ""),  # unverified | tested | expert（质量分级维度 B）
        }
        # FTS 路径补 section（chunks_meta.section，PDF 手册章节标题）
        sec = conn.execute(
            "SELECT section FROM chunks_meta WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if sec and sec[0]:
            meta["section"] = sec[0]
        return meta

    @staticmethod
    def _pass_filters(meta: dict, filters: dict) -> bool:
        if not meta:
            return False
        if "status" in filters and meta.get("status") != filters["status"]:
            return False
        if "source_type" in filters and meta.get("source_type") != filters["source_type"]:
            return False
        if "component" in filters and meta.get("component") != filters["component"]:
            return False
        if "min_created" in filters:
            mc = filters["min_created"]
            c = meta.get("created_at") or ""
            if c < mc:
                return False
        if "max_created" in filters:
            c = meta.get("created_at") or ""
            if c > filters["max_created"]:
                return False
        return True


# ---------------- 标题精确检索（能力2） ----------------

@dataclass
class TitleHit:
    doc_id: str
    title: str
    url: str
    component: str
    resolved: bool
    resolved_at: Optional[str] = None


def title_search(
    conn: sqlite3.Connection,
    keyword: str,
    component: Optional[str] = None,
    limit: int = 20,
    match: str = "contains",
) -> list[TitleHit]:
    """标题子串精确检索（SQL LIKE）。

    用于"已知现象找 issue"：报错里出现专有名词（vector core / rtDeviceSynchronize 等）
    时，标题含该词的 issue 是最直接的线索——语义检索对英文描述正文的召回常不足。

    match: contains（默认，子串）| prefix（标题前缀）。
    """
    kw = keyword.strip()
    if not kw:
        return []
    sql = "SELECT source_id, title, url, component, resolved_at FROM docs WHERE title LIKE ?"
    params: list = [f"%{kw}%" if match == "contains" else f"{kw}%"]
    if component:
        sql += " AND component = ?"
        params.append(component)
    sql += " ORDER BY length(title) LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    hits = []
    for sid, title, url, comp, resolved_at in rows:
        hits.append(
            TitleHit(
                doc_id=sid,
                title=title or "",
                url=url or "",
                component=comp or "",
                resolved=bool(resolved_at),
                resolved_at=resolved_at,
            )
        )
    return hits
