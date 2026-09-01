"""只读检索 API（FastAPI）入口与组装。结构上杜绝一切写操作，不依赖提示词约束：

1. SQLite 以 URI `mode=ro` 打开 —— 任何 INSERT/UPDATE/CREATE 在连接层必然失败；
2. 向量库经 ReadOnlyVectorStore 包装 —— add/delete/update/clear 一律抛 ReadOnlyError；
3. 本模块不导入任何可写模块（ingest/github_pull/pipeline/sources），不含写端点、
   不打开文件写 —— tests/test_api_readonly.py 的源码级审计兜底；
4. 运行前可执行 scripts/check_readonly.py 验证只读姿态。

路由按检索域拆分（存算分离的"算"侧分层）：
- api_meta.py      辅助：/health /stats /components /companion /matrix /version
- api_community.py 社区（issues/pr）+ 文档检索：/search /signature-search /title
                  /doc /graph/* /tags/*（共用 SearchEngine，存储合一）
- api_code.py      版本化代码仓符号索引：/code/* /code-versions（本地快照，存储独立）
- api_code_graph.py 代码图谱检索（gh-puller 接入）：/code-graph/*（见该模块）

启动（需先 pip install fastapi uvicorn）：
    python scripts/serve_api.py [--port 8000]
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from .config import AppConfig
from .search import SearchEngine

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _AppContext:
    """create_app 构建的共享上下文，传给各路由注册函数。

    各检索域路由共用同一 SearchEngine（社区/文档）与同一只读 SQLite 连接工厂，
    避免重复构造；出口脱敏/extra 白名单清理也统一在此提供。
    """
    cfg: AppConfig
    engine: SearchEngine
    out: Callable[[Any], Any]            # 出口脱敏（字符串按 sanitize 白名单）
    readonly_sqlite: Callable[["Path"], sqlite3.Connection]  # URI mode=ro 连接
    sanitize_extra: Callable[[Any], dict]  # extra 字段白名单清理


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
    from fastapi import FastAPI

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

    ctx = _AppContext(
        cfg=cfg,
        engine=engine,
        out=_out,
        readonly_sqlite=_readonly_sqlite,
        sanitize_extra=_sanitize_extra,
    )

    app = FastAPI(
        title="vllm-kb 只读检索 API",
        version="0.1.0",
        description="vLLM / vllm-ascend 故障知识库检索接口。结构只读：无写端点，"
                    "SQLite mode=ro，向量库写操作抛错。数据更新由用户运行流水线触发。",
    )

    # 版本化代码仓（懒加载占位：api_code 各端点按需自建 VersionedCode，此处不再单例）
    from . import api_meta, api_community, api_code

    api_meta.register(app, ctx)
    api_community.register(app, ctx)
    api_code.register(app, ctx)

    # 代码图谱检索（gh-puller 接入）：配置启用时注册，未配置跳过（端点不存在比 503 更干净）
    from .config import CodeGraphCfg

    if getattr(cfg, "code_graph", None) and cfg.code_graph.enabled:
        from . import api_code_graph

        api_code_graph.register(app, ctx)

    return app
