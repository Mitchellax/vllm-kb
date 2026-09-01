"""辅助检索路由（meta）：/health、/stats、/components、/companion、/matrix、/version。

从 api.py 拆出，行为不变。路由注册函数接收 ctx（create_app 构建的共享上下文），
仅注册不返回 app——组装在 api.py。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import _AppContext  # noqa: F401


def register(app, ctx) -> None:
    from fastapi import HTTPException

    engine = ctx.engine
    cfg = ctx.cfg

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

    @app.get("/components")
    def components():
        conn = ctx.readonly_sqlite(engine.sqlite_path)
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
        conn = ctx.readonly_sqlite(engine.sqlite_path)
        try:
            total = conn.execute("SELECT count(*) FROM docs").fetchone()[0]
            with_version = conn.execute(
                "SELECT count(*) FROM docs WHERE version_span_min IS NOT NULL"
            ).fetchone()[0]
            return {"docs": total, "docs_with_version": with_version,
                    "chunks": engine.vector_store.count()}
        finally:
            conn.close()

    @app.get("/version")
    def version_info(version: str, repo: str | None = None):
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
