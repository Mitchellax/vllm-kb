"""社区与文档检索路由（community）：/search、/signature-search、/title、/doc、/graph/*、/tags/*。

社区检索（issues/pr，经 SearchEngine 走 SQLite+向量库+Kùzu）与文档检索（PDF/MD，同一引擎）
共用存储，仅在路由层分清边界——存储合一是设计取舍（双引擎无收益）。
从 api.py 拆出，行为不变。注册函数接收 ctx（create_app 构建的共享上下文）。
"""
from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .api import _AppContext  # noqa: F401


# ---------------- 请求模型（模块级：FastAPI 闭包内局部模型 + 延迟注解会解析失败） ----------------

class SearchRequest(BaseModel):
    query: str
    target_version: Optional[str] = None
    component: Optional[str] = None
    version: Optional[str] = None
    top_k: Optional[int] = None
    filters: dict[str, Any] = Field(default_factory=dict)


class SignatureRequest(BaseModel):
    text: str  # 原始报错/描述，现场提取签名
    component: Optional[str] = None
    top_k: Optional[int] = 15


class TagMatchRequest(BaseModel):
    text: str  # 问题描述（context 命令后端：问题→标签匹配）


def register(app, ctx) -> None:
    from fastapi import HTTPException

    engine = ctx.engine
    cfg = ctx.cfg
    _out = ctx.out

    @app.post("/search")
    def search(req: SearchRequest):
        filters = dict(req.filters or {})
        # component 参数同时作为结果过滤（组件查询的 companion 展开 + 过滤）
        if req.component and "component" not in filters:
            filters["component"] = req.component
        results = engine.search(
            query=req.query,
            target_version=req.target_version,
            component=req.component,
            version=req.version,
            top_k=req.top_k,
            filters=filters or None,
        )
        return {
            "context": engine.last_context,
            "results": [
                {
                    "doc_id": r.doc_id,
                    "title": _out(r.title),
                    "url": r.url,
                    "component": r.component,
                    "resolved": r.resolved,
                    "section": r.meta.get("section", ""),  # PDF 手册等：所属章节标题
                    "kind": r.meta.get("kind", ""),
                    "status": r.meta.get("status", ""),
                    "verification": r.meta.get("verification", ""),
                    "tags": r.meta.get("tags", []),  # 文档最终标签（两级分类）
                    "version_ref": r.version_ref,
                    "version_span": [r.meta.get("version_span_min"), r.meta.get("version_span_max")],
                    "similarity": r.similarity,
                    "final": r.final,
                    "confidence": {
                        "score": r.confidence.score,
                        "w_time": r.confidence.time_weight,
                        "w_ver": r.confidence.version_weight,
                        "w_rel": r.confidence.reliability,
                        "target_version": r.confidence.target_version,
                        "verification": r.confidence.extras.get("verification", ""),
                        "w_hist": r.confidence.w_hist,
                        "n_eff": r.confidence.n_eff,
                        "history_flag": r.confidence.history_flag,
                        "sigma": r.confidence.sigma,
                    },
                    "snippet": _out(r.text[:500]),
                }
                for r in results
            ],
            "degraded": engine._embed_error or None,  # embedding 不可用时降级为全文检索的提示
        }

    @app.get("/doc/{source_id}")
    def doc(source_id: str):
        """返回整篇文档全文（只读，从 SQLite FTS 分块按序拼装）。

        extra 经 _sanitize_extra 白名单清理——不返回任何服务器路径（安全约束）。
        """
        conn = ctx.readonly_sqlite(engine.sqlite_path)
        try:
            row = conn.execute(
                "SELECT title, url, created_at, resolved_at, status, component, extra, tags "
                "FROM docs WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="文档不存在")
            chunks = conn.execute(
                "SELECT f.text FROM chunks_meta m JOIN chunks_fts f ON f.chunk_id = m.chunk_id "
                "WHERE m.doc_id = ? ORDER BY m.seq",
                (source_id,),
            ).fetchall()
            return {
                "source_id": source_id,
                "title": _out(row[0]),
                "url": row[1],
                "created_at": row[2],
                "resolved_at": row[3],
                "status": row[4],
                "component": row[5],
                "tags": json.loads(row[7]) if row[7] else [],
                "extra": ctx.sanitize_extra(json.loads(row[6] or "{}")),
                "body": _out("\n\n".join(c[0] for c in chunks)),
            }
        finally:
            conn.close()

    # ---------------- 签名精确检索 ----------------

    @app.post("/signature-search")
    def signature_search(req: SignatureRequest):
        """从原始报错文本现场提取签名，做精确检索（FTS 短语 + 加权聚合）。

        返回聚合视图：提取签名 + 命中的社区高频信号词 + 精确命中 + 标题精确命中。
        """
        from .signature import extract_signatures, format_signatures, signature_search

        # 加载源码符号表（三层提取的优先层）；失败则退回基础正则
        try:
            from .symbol_table import load_symbol_table, load_signal_words, match_signal_words

            table = load_symbol_table(cfg)
            signal_words = load_signal_words(cfg)
            phrase_signals = [
                {"word": e.name, "weight": e.weight}
                for e in table.entries if e.kind == "phrase"
            ]
        except Exception:
            table = None
            signal_words = []
            phrase_signals = []
        sigs = extract_signatures(req.text, symbol_table=table)
        signal_hits = match_signal_words(signal_words, req.text, phrase_signals=phrase_signals) \
            if (signal_words or phrase_signals) else []
        conn = ctx.readonly_sqlite(engine.sqlite_path)
        try:
            hits = signature_search(
                conn, sigs, top_k=req.top_k or 15, component=req.component,
            )
            # 标题精确命中（信号词/签名在标题里出现的最直接线索）
            from .search import title_search

            title_hits: list[dict] = []
            for sw in signal_hits[:5]:
                for th in title_search(conn, sw["word"], component=req.component, limit=5):
                    if not any(t["doc_id"] == th.doc_id for t in title_hits):
                        title_hits.append({
                            "doc_id": th.doc_id, "title": th.title, "url": th.url,
                            "component": th.component, "resolved": th.resolved,
                            "signal": sw["word"],
                        })
        finally:
            conn.close()
        return {
            "signatures": [
                {"text": s.text, "kind": s.kind, "weight": s.weight, "origin": s.origin}
                for s in sigs
            ],
            "signatures_text": format_signatures(sigs),
            "signal_words": signal_hits[:10],
            "results": [
                {
                    "doc_id": h.doc_id,
                    "title": _out(h.title),
                    "url": h.url,
                    "hit_signatures": h.hit_signatures,
                    "score": h.score,
                }
                for h in hits
            ],
            "title_hits": [
                {
                    "doc_id": th.doc_id, "title": _out(th.title), "url": th.url,
                    "component": th.component, "resolved": th.resolved,
                    "signal": th.signal,
                }
                for th in title_hits[:10]
            ],
        }

    @app.get("/title")
    def title(keyword: str, component: Optional[str] = None, limit: int = 20, match: str = "contains"):
        """标题子串精确检索（SQL LIKE）：已知现象找 issue 的最快路径。"""
        from .search import title_search

        conn = ctx.readonly_sqlite(engine.sqlite_path)
        try:
            hits = title_search(conn, keyword, component=component, limit=limit, match=match)
        finally:
            conn.close()
        return {
            "keyword": keyword,
            "component": component,
            "results": [
                {
                    "doc_id": h.doc_id,
                    "title": _out(h.title),
                    "url": h.url,
                    "component": h.component,
                    "resolved": h.resolved,
                    "resolved_at": h.resolved_at,
                }
                for h in hits
            ],
        }

    # ---------------- Phase 2：图存储检索（Kùzu，懒加载；只读查询，不触发建图） ----------------

    _graph_state: dict = {"builder": None}

    def _graph():
        """懒加载图访问器（只读查询）。图未构建时 is_built()=False，端点返回提示。"""
        from .graph import GraphBuilder, default_graph_path

        b = _graph_state["builder"]
        if b is None:
            b = GraphBuilder(default_graph_path(cfg))
            _graph_state["builder"] = b
        return b

    def _require_graph():
        b = _graph()
        if not b.is_built():
            raise HTTPException(status_code=503, detail="图未构建：运行 python scripts/build_graph.py")
        return b

    @app.get("/graph/stats")
    def graph_stats():
        b = _graph()
        if not b.is_built():
            return {"built": False, "note": "图未构建：运行 python scripts/build_graph.py"}
        s = b.stats()
        return {"built": True, "nodes": s.nodes, "rels": s.rels,
                "summary": s.summary()}

    def _sanitize_graph(d: Any) -> Any:
        """图检索结果出口脱敏：递归脱敏所有 title 字段（内部原文检索，出口统一脱敏）。"""
        if isinstance(d, dict):
            return {k: (_out(v) if k == "title" and isinstance(v, str) else _sanitize_graph(v))
                    for k, v in d.items()}
        if isinstance(d, list):
            return [_sanitize_graph(x) for x in d]
        return d

    @app.get("/graph/chain")
    def graph_chain(doc: str):
        """issue → 修复 PR → 落地 release 链路（doc=完整 source_id，如 github:vllm-project-vllm:issue:10700）。"""
        return _sanitize_graph(_require_graph().chain_issue(doc))

    @app.get("/graph/fixes")
    def graph_fixes(doc: str):
        """PR 视角：该 PR 修复的 issues + 落地 release。"""
        return _sanitize_graph(_require_graph().fixes_pr(doc))

    @app.get("/graph/sig")
    def graph_sig(sig: str, limit: int = 10):
        """签名实体（算子/错误码/模型/版本）→ 提及它的 issue/PR。"""
        return _sanitize_graph(_require_graph().sig_lookup(sig, limit=limit))

    @app.get("/graph/doc")
    def graph_doc(doc: str):
        """文档邻接视图：MENTIONS 实体（调试/详情）。"""
        return _require_graph().doc_neighbors(doc)

    @app.get("/graph/tags")
    def graph_tags(tag: str, limit: int = 50):
        """标签 → 打标文档（Doc/Issue/PR）——图侧标签查询。"""
        return _sanitize_graph(_require_graph().tags_lookup(tag, limit=limit))

    @app.get("/graph/evidence")
    def graph_evidence(doc: str):
        """文档互证（Evidence）：与目标文档共享 ≥2 个实体的其他文档（多来源互证）。"""
        return _sanitize_graph(_require_graph().evidence_for(doc))

    # ---------------- 文档级标签（能力目录 / 标签检索 / 问题匹配） ----------------

    def _tag_counts() -> dict[str, int]:
        """docs.tags（最终标签）→ 文档数聚合。"""
        conn = ctx.readonly_sqlite(engine.sqlite_path)
        try:
            try:
                rows = conn.execute(
                    "SELECT tags FROM docs WHERE tags IS NOT NULL"
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        finally:
            conn.close()
        counts: dict[str, int] = {}
        for (tags_json,) in rows:
            try:
                for t in json.loads(tags_json or "[]"):
                    counts[t] = counts.get(t, 0) + 1
            except (TypeError, json.JSONDecodeError):
                continue
        return counts

    @app.get("/tags")
    def tags_catalog():
        """能力目录：按 tier 分组 {name, docs}（docs>0；registry-only 标签不出现避免噪音）。

        这是"知识能力目录"（有哪些主题/作用类文档可提供知识），非文件枚举——
        agent 据此知道可查哪些文档类别，再按标签检索。
        """
        from .tagging import TagRegistry

        registry = TagRegistry.load(cfg)
        groups: dict[str, list] = {"domain": [], "purpose": []}
        counts = _tag_counts()
        for name, n in sorted(counts.items()):
            tier = registry.tier(name)
            groups.setdefault(tier, []).append({"name": name, "docs": n})
        return {
            "groups": groups,
            "total_tags": len(counts),
            "note": "主题/领域类 domain=这是什么领域的知识；具体作用类 purpose=文档能帮我做什么。"
                    "按标签检索: /tags/{tag}/docs",
        }

    @app.get("/tags/{tag}/docs")
    def tags_docs(tag: str, limit: int = 50):
        """标签过滤检索：该标签下文档的标题/文档 id/验证状态/片段（上限 200，精确匹配最终标签）。"""
        conn = ctx.readonly_sqlite(engine.sqlite_path)
        try:
            try:
                rows = conn.execute(
                    "SELECT source_id, source_type, title, url, extra, tags "
                    "FROM docs WHERE tags IS NOT NULL"
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        finally:
            conn.close()
        out = []
        for sid, st, title, url, extra, tags_json in rows:
            try:
                tags = json.loads(tags_json or "[]")
            except (TypeError, json.JSONDecodeError):
                continue
            if tag not in tags:
                continue
            try:
                ex = json.loads(extra or "{}")
            except (TypeError, json.JSONDecodeError):
                ex = {}
            out.append({
                "doc_id": sid,
                "source_type": st,
                "title": _out(title or ""),
                "url": url or "",
                "verification": ex.get("verification", ""),
                "tags": tags,
            })
            if len(out) >= min(limit, 200):
                break
        return {"tag": tag, "docs": out, "count": len(out),
                "note": "标签过滤的检索结果（非文件枚举）；读取全文用 doc <id>"}

    @app.post("/tags/match")
    def tags_match(req: TagMatchRequest):
        """问题→标签自动匹配（context 命令后端）：从问题文本命中词典标签。

        domain 组做范围圈定、purpose 组做能力提示、两维交集的文档排前——
        agent 据此发现"知识库有哪些文档能帮上这个问题"（如 HCCL 超时 →
        命中 HCCL(domain) + 超时排查/命令参考(purpose) 及其文档线索）。
        """
        import re as _re

        from .tagging import TagEntry, TagRegistry

        registry = TagRegistry.load(cfg)
        lowered = (req.text or "").lower()
        counts = _tag_counts()
        # 1) 词典子串命中；2) 无命中时回退：问题拉丁 token 与已用标签（docs>0）词级匹配
        matched: list[TagEntry] = []
        seen: set[str] = set()
        for e in registry.entries:
            if e.name and e.name.lower() in lowered and e.name not in seen:
                matched.append(e)
                seen.add(e.name)
        if not matched:
            tokens = {t for t in _re.findall(r"[A-Za-z][A-Za-z0-9_]{1,}", lowered)}
            for name in counts:
                if name and name.lower() in tokens and name not in seen:
                    matched.append(TagEntry(name=name, tier=registry.tier(name)))
                    seen.add(name)
        if not matched:
            return {"matched": [], "count": 0,
                    "note": "问题文本未命中任何标签（可尝试 tags 查看全部能力目录）"}

        conn = ctx.readonly_sqlite(engine.sqlite_path)
        try:
            try:
                rows = conn.execute(
                    "SELECT source_id, source_type, title, url, extra, tags "
                    "FROM docs WHERE tags IS NOT NULL"
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        finally:
            conn.close()
        docs_by_tag: dict[str, list[dict]] = {e.name: [] for e in matched}
        doc_tags_map: dict[str, set[str]] = {}
        for sid, st, title, url, extra, tags_json in rows:
            try:
                tags = set(json.loads(tags_json or "[]"))
            except (TypeError, json.JSONDecodeError):
                continue
            if not tags:
                continue
            doc_tags_map[sid] = tags
            for e in matched:
                if e.name in tags and len(docs_by_tag[e.name]) < 3:
                    try:
                        ex = json.loads(extra or "{}")
                    except (TypeError, json.JSONDecodeError):
                        ex = {}
                    docs_by_tag[e.name].append({
                        "doc_id": sid, "title": _out(title or ""),
                        "verification": ex.get("verification", ""), "url": url or "",
                    })
        # 交集排序：同时命中 domain+purpose 标签的文档线索排前
        matched_names = [e.name for e in matched]
        hits: list[dict] = []
        for e in matched:
            top = docs_by_tag[e.name]
            # 同标签文档按"是否同时命中其他标签"排序
            top.sort(key=lambda d: -len(doc_tags_map.get(d["doc_id"], set()) & set(matched_names)))
            hits.append({
                "name": e.name,
                "tier": e.tier,
                "docs": counts.get(e.name, 0),
                "top": top,
            })
        return {
            "matched": hits,
            "count": len(hits),
            "note": "匹配的标签：domain=领域范围，purpose=能力（能帮我做什么）；"
                    "top=各标签下代表性文档线索（读取全文用 doc <id>）",
        }
