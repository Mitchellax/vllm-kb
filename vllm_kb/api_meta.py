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

    def _ocr_state() -> dict:
        """OCR 配置状态（**不主动探测服务**——连通性由审核工作台「测试连通」显式触发）。

        如实反映配置，便于排查「端点 404 / 400 / 503」时先区分"没配 OCR"、"配了但不能
        用于请求期"还是"配好了但服务挂了"。
        """
        from .ocr import ocr_config_from_cfg

        try:
            oc = ocr_config_from_cfg(cfg)
        except Exception as e:
            return {"state": "unknown", "note": str(e)}
        if oc is None:
            return {"state": "unconfigured", "note": "未启用 image source"}
        note = f"provider={oc.provider} mode={oc.mode}"
        if oc.model:
            note += f" model={oc.model}"
        if oc.provider == "none":
            return {"state": "disabled", "note": "ocr_provider=none（明确跳过 OCR）"}
        if oc.provider == "api" and not oc.api_base:
            return {"state": "unconfigured", "note": "ocr_provider=api 但未配置 ocr_api_base"}
        if oc.provider == "ask" and not oc.api_base:
            return {"state": "unconfigured",
                    "note": "ocr_provider=ask 且无 ocr_api_base（导入时询问本地/跳过）"}
        return {"state": "configured", "endpoint": "/ocr" if oc.provider == "api" else None,
                "note": note}

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
            "ocr": _ocr_state(),
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
        """全量配套矩阵（含 commit 溯源字段：image_created / vllm_commit / vllm_commit_date）。"""
        m = engine.companion
        if m is None:
            return {"rows": [], "generated_at": ""}
        return {"generated_at": m.generated_at,
                "rows": [r.model_dump(by_alias=True) for r in m.rows]}

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
