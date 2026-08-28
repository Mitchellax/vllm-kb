"""只读检索 API（FastAPI）。结构上杜绝一切写操作，不依赖提示词约束：

1. SQLite 以 URI `mode=ro` 打开 —— 任何 INSERT/UPDATE/CREATE 在连接层必然失败；
2. 向量库经 ReadOnlyVectorStore 包装 —— add/delete/update/clear 一律抛 ReadOnlyError；
3. 本模块不导入任何可写模块（ingest/github_pull/pipeline/sources），不含写端点、
   不打开文件写 —— tests/test_api_readonly.py 的源码级审计兜底；
4. 运行前可执行 scripts/check_readonly.py 验证只读姿态。

启动（需先 pip install fastapi uvicorn）：
    python scripts/serve_api.py [--port 8000]
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from pydantic import BaseModel, Field

from .config import AppConfig
from .search import SearchEngine


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


class CodeSearchRequest(BaseModel):
    keyword: str
    version: Optional[str] = None
    limit: Optional[int] = 20
    repo: Optional[str] = None  # vllm-ascend | vllm
    path: Optional[str] = None  # 限定文件路径子串（如 worker/model_runner_v1.py）
    per_version: Optional[bool] = False  # 每个版本各自收集命中（对比版本差异用）
    kind: Optional[str] = None  # def | op | env | msg（msg=报错字面量 LIKE 子串检索）


class TagMatchRequest(BaseModel):
    text: str  # 问题描述（context 命令后端：问题→标签匹配）


def _readonly_sqlite(path) -> sqlite3.Connection:
    """URI 级只读 SQLite 连接：文件缺失或任何写操作都会抛错。"""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _sanitize_extra(extra: Any) -> dict:
    """出口白名单（纵深防御）：剥离 asset/evidence 中可能含服务器路径的字段。

    - 只保留 verification/quality/structure/kind/tag_candidates（知识性字段）；
    - evidence 只保留无路径形态（kind/asset_id/sha256/source_ref 仅 http(s) URL）——
      即使存量库残留历史路径也不外泄（安全约束：skill 响应不含服务器路径）。
    """
    if not isinstance(extra, dict):
        return {}
    out: dict = {}
    for k in ("verification", "quality", "structure", "kind"):
        if k in extra:
            out[k] = extra[k]
    if "tag_candidates" in extra and isinstance(extra["tag_candidates"], list):
        out["tag_candidates"] = [
            {"name": c.get("name"), "tier": c.get("tier")}
            for c in extra["tag_candidates"] if isinstance(c, dict)
        ]
    if "evidence" in extra and isinstance(extra["evidence"], list):
        safe = []
        for e in extra["evidence"]:
            if not isinstance(e, dict):
                continue
            item = {k: e[k] for k in ("kind", "asset_id", "sha256") if k in e}
            sr = e.get("source_ref", "")
            if isinstance(sr, str) and sr.startswith(("http://", "https://")):
                item["source_ref"] = sr
            safe.append(item)
        if safe:
            out["evidence"] = safe
    return out


def create_app(config_path: Optional[str] = None):
    """构建 FastAPI 应用（fastapi 懒加载：未安装时本模块仍可导入）。"""
    from fastapi import FastAPI, HTTPException

    cfg = AppConfig.load(config_path, require_keys=False)
    engine = SearchEngine(cfg, read_only=True)

    # 出口脱敏：内部检索用原文（库/向量/FTS 存原文），返回给 agent 的正文/标题字段统一脱敏
    # （config.sanitize 白名单；改配置即时生效、无需重嵌）。serve_api 保持结构只读——纯函数，不写文件。
    from .sanitize import sanitize_text as _sanitize_text

    def _out(s: Any) -> Any:
        """出口脱敏：非字符串原样返回；字符串按 config.sanitize 白名单脱敏 IP/路径。"""
        if not isinstance(s, str) or not s:
            return s
        return _sanitize_text(s, keep_paths=cfg.sanitize.keep_paths,
                              keep_ips=cfg.sanitize.keep_ips)

    # 版本化代码仓（懒加载：未预存时端点返回可用版本提示，不崩溃）
    try:
        from .code_index import VersionedCode

        code_index = VersionedCode(cfg)
    except Exception:
        code_index = None

    app = FastAPI(
        title="vllm-kb 只读检索 API",
        version="0.1.0",
        description="vLLM / vllm-ascend 故障知识库检索接口。结构只读：无写端点，"
                    "SQLite mode=ro，向量库写操作抛错。数据更新由用户运行流水线触发。",
    )

    @app.get("/health")
    def health():
        embed_state = "ok"
        if engine._embed_error:
            embed_state = "degraded" if not engine._embed_available() else "degraded-retrying"
        return {
            "status": "ok",  # 服务可用（embedding 不可用时检索自动降级为全文）
            "read_only": True,
            "chunks": engine.vector_store.count(),
            "embedding": embed_state,
            "embedding_note": engine._embed_error or None,
        }

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
        conn = _readonly_sqlite(engine.sqlite_path)
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
                "extra": _sanitize_extra(json.loads(row[6] or "{}")),
                "body": _out("\n\n".join(c[0] for c in chunks)),
            }
        finally:
            conn.close()

    @app.get("/components")
    def components():
        conn = _readonly_sqlite(engine.sqlite_path)
        try:
            rows = conn.execute(
                "SELECT component, count(*) c FROM docs GROUP BY component ORDER BY c DESC"
            ).fetchall()
            return {"components": [{"component": r[0], "docs": r[1]} for r in rows]}
        finally:
            conn.close()

    @app.get("/companion")
    def companion(component: str, version: str):
        m = engine.companion
        if m is None:
            return {"component": component, "version": version, "companions": {},
                    "note": "配套矩阵未配置或为空（运行 scripts/build_companion_matrix.py）"}
        return {"component": component, "version": version, "companions": m.expand(component, version)}

    @app.get("/matrix")
    def matrix():
        m = engine.companion
        if m is None:
            return {"rows": []}
        return {"rows": [r.model_dump(by_alias=True) for r in m.rows]}

    @app.get("/stats")
    def stats():
        conn = _readonly_sqlite(engine.sqlite_path)
        try:
            total = conn.execute("SELECT count(*) FROM docs").fetchone()[0]
            with_version = conn.execute(
                "SELECT count(*) FROM docs WHERE version_span_min IS NOT NULL"
            ).fetchone()[0]
            return {"docs": total, "docs_with_version": with_version,
                    "chunks": engine.vector_store.count()}
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
        conn = _readonly_sqlite(engine.sqlite_path)
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

    @app.get("/version")
    def version_info(version: str, repo: Optional[str] = None):
        """版本形态判断：正式 release / rc / pre / unknown（基于版本日历）。

        repo: vllm-project/vllm-ascend（默认）| vllm-project/vllm；也接受简名 vllm-ascend/vllm。
        """
        from .confidence import load_release_meta, version_kind

        repo = repo or "vllm-project/vllm-ascend"
        # 简名兼容：vllm-ascend -> vllm-project/vllm-ascend
        if repo in ("vllm-ascend", "ascend"):
            repo = "vllm-project/vllm-ascend"
        elif repo == "vllm":
            repo = "vllm-project/vllm"
        repo_slug = repo.replace("/", "-")
        meta = load_release_meta(cfg.resolve(f"data/compatibility/release_calendar.{repo_slug}.json"))
        kind = version_kind(meta, version)
        info = None
        if meta:
            v = version.lower()
            for tag, m in meta.items():
                if tag.lower() == v or tag.lower().lstrip("v") == v.lstrip("v"):
                    info = {"tag": tag, **m}
                    break
        return {
            "version": version,
            "repo": repo,
            "kind": kind,
            "calendar_loaded": meta is not None,
            "release": info,
            "note": "kind: release=正式版 rc=预发布 pre=早期 pre 版 unknown=日历中无此版本",
        }

    @app.get("/title")
    def title(keyword: str, component: Optional[str] = None, limit: int = 20, match: str = "contains"):
        """标题子串精确检索（SQL LIKE）：已知现象找 issue 的最快路径。"""
        from .search import title_search

        conn = _readonly_sqlite(engine.sqlite_path)
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

    # ---------------- 版本化代码仓检索 ----------------

    def _code_index_for(repo: Optional[str]):
        """按 repo 取代码仓访问器：vllm-ascend（默认）| vllm。"""
        if repo not in (None, "", "vllm-ascend", "vllm"):
            return None
        repo = "vllm-ascend" if repo in (None, "", "vllm-ascend") else "vllm"
        try:
            from .code_index import VersionedCode

            return VersionedCode(cfg, repo=repo)
        except Exception:
            return None

    def _code_call(fn, *args, **kwargs):
        """统一代码仓调用异常处理：
        - 版本未预存（客户端请求问题）→ 404，带可用版本与预存指引；
        - 符号索引未构建/其他意外（服务端状态）→ 503，不向客户端泄漏堆栈。
        """
        from .code_index import CodeIndexError

        try:
            return fn(*args, **kwargs)
        except CodeIndexError as e:
            if "未预存" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            print(f"[api] code 调用异常（{type(e).__name__}: {e}）", flush=True)
            raise HTTPException(status_code=503, detail="代码仓检索暂时不可用（详见服务端日志）")

    @app.get("/code/versions")
    def code_versions(repo: Optional[str] = None):
        ci = _code_index_for(repo)
        if ci is None:
            return {"repo": repo or "vllm-ascend", "versions": [],
                    "note": "code_index 未初始化（检查 config.storage.code_root）"}
        return {"repo": repo or "vllm-ascend", "versions": ci.available_versions,
                "note": "预存版本源码快照；未列的版本请先运行 scripts/build_code_snapshots.py 或 build_vllm_snapshots.py"}

    @app.post("/code/search")
    def code_search(req: CodeSearchRequest):
        """代码仓检索：符号索引精确命中 → 关键词全文兜底；kind=msg 走报错字面量 LIKE 检索。

        - kind=msg：报错字面量索引（raise/assert/logger.error 字符串参数）子串检索，
          定位"报错文本来自哪段代码"（无需全文 grep）；
        - path：限定文件路径子串（--in-file）；per_version：每个版本各自收集命中，
          输出各版本行号便于对比"哪个版本引入/移动了该代码"。
        """
        repo = req.repo
        ci = _code_index_for(repo)
        if ci is None:
            raise HTTPException(status_code=503, detail="code_index 未初始化（运行 scripts/build_code_snapshots.py）")
        if req.kind == "msg":
            hits = _code_call(ci.search_messages, req.keyword, req.version, limit=req.limit or 20)
            return {"mode": "message_index", "symbol": req.keyword, "repo": repo or "vllm-ascend",
                    "version": req.version, "hits": hits}
        symbols = _code_call(ci.search_symbols, req.keyword, req.version, limit=req.limit or 20,
                             kind=req.kind)
        if symbols and not req.per_version:
            return {"mode": "symbol_index", "symbol": req.keyword, "repo": repo or "vllm-ascend",
                    "version": req.version, "hits": symbols}
        greps = _code_call(ci.grep, req.keyword, req.version, limit=req.limit or 20,
                           path_sub=req.path, per_version=bool(req.per_version))
        mode = "grep" if not req.per_version else "grep_per_version"
        return {"mode": mode, "symbol": req.keyword, "repo": repo or "vllm-ascend",
                "version": req.version, "hits": greps}

    @app.get("/code/file")
    def code_file(version: str, path: str, max_chars: int = 20000, repo: Optional[str] = None):
        """读取指定版本的源码文件片段（按需解压；截断时末尾带明确标记）。"""
        ci = _code_index_for(repo)
        if ci is None:
            raise HTTPException(status_code=503, detail="code_index 未初始化")
        text = _code_call(ci.read_file, version, path, max_chars)
        if text is None:
            raise HTTPException(status_code=404, detail=f"{version}:{path} 不存在（repo={repo or 'vllm-ascend'}）")
        return {"version": version, "repo": repo or "vllm-ascend", "path": path, "content": text}

    @app.get("/code/diff")
    def code_diff(version1: str, version2: str, path: str,
                  keyword: Optional[str] = None, context: int = 3,
                  repo: Optional[str] = None):
        """跨版本精确 diff：同一文件在两个版本快照间的 unified diff。

        定位"哪个版本引入/修改了某代码"（新增行 = 修复引入点）；--keyword 只显示含关键词的差异行。
        """
        import difflib

        ci = _code_index_for(repo)
        if ci is None:
            raise HTTPException(status_code=503, detail="code_index 未初始化")
        p1 = _code_call(ci.find_file, version1, path)
        p2 = _code_call(ci.find_file, version2, path)
        missing = [v for v, p in ((version1, p1), (version2, p2)) if p is None]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"版本 {missing} 未预存文件 {path}（repo={repo or 'vllm-ascend'}）",
            )
        t1 = p1.read_text(encoding="utf-8", errors="replace").splitlines()
        t2 = p2.read_text(encoding="utf-8", errors="replace").splitlines()
        diff_lines = list(difflib.unified_diff(
            t1, t2, fromfile=f"{version1}:{path}", tofile=f"{version2}:{path}",
            n=context, lineterm=""))
        out: list[str] = []
        shown = 0
        for line in diff_lines:
            if line.startswith(("+++", "---", "@@")):
                out.append(line)  # 头部/块标记始终显示
            elif keyword and keyword.lower() not in line.lower():
                continue
            else:
                out.append(line)
                shown += 1
        return {
            "path": path, "v1": version1, "v2": version2, "repo": repo or "vllm-ascend",
            "lines1": len(t1), "lines2": len(t2), "keyword": keyword, "context": context,
            "diff": "\n".join(out),
            "note": f"无包含关键词 '{keyword}' 的差异行（v1/v2 该文件可能无差异）"
                    if shown == 0 and keyword else None,
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
        conn = _readonly_sqlite(engine.sqlite_path)
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
        conn = _readonly_sqlite(engine.sqlite_path)
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

        conn = _readonly_sqlite(engine.sqlite_path)
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

    return app
