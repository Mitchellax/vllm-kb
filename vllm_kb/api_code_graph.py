"""代码图谱检索路由（gh-puller 接入）：/code-graph/*。

与 /code/*（本地版本化符号索引）并列、不替换、不回退——能力互补不重叠
（见 code_graph.py 模块 docstring）。gh-puller 不可达时本组端点直接 503 +
引导用 code 命令查本地索引（本地无等价图谱能力，回退无意义）。

端点对齐 gh-puller-mcp 工具面（REST 路径按可读短横线，契约清单到手后改
code_graph.py 默认即可）：
- POST /code-graph/search       search_graph（BM25/正则/语义三模搜函数/类/路由）
- POST /code-graph/code-search  search_code（grep + 图增强）
- POST /code-graph/trace        trace_path（调用链/数据流/跨服务路径）
- POST /code-graph/query        query_graph（Cypher 多跳/聚合）
- GET  /code-graph/architecture get_architecture（架构总览/聚类/边界）
- POST /code-graph/changes      detect_changes（git diff → 影响面）
- GET  /code-graph/health       探测 gh-puller 可达性
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from .api import _AppContext  # noqa: F401


# ---------------- 请求模型（模块级：FastAPI 闭包内局部模型 + 延迟注解会解析失败） ----------------

class GraphSearchRequest(BaseModel):
    query: Optional[str] = None  # 自然语言/BM25 全文（与 name_pattern 至少传一个）
    name_pattern: Optional[str] = None  # 正则精确匹配
    label: Optional[str] = None  # 节点标签过滤（Function/Class/Route/...）
    repo: Optional[str] = None  # vllm-ascend（默认）| vllm，映射见 config.code_graph.repo_project_map
    limit: int = 10
    offset: int = 0


class CodeSearchGraphRequest(BaseModel):
    pattern: str  # grep 文本
    mode: str = "compact"  # compact | full | files
    path_filter: Optional[str] = None  # 结果文件路径正则过滤
    repo: Optional[str] = None
    limit: int = 10


class TraceRequest(BaseModel):
    function_name: str
    direction: str = "both"  # inbound | outbound | both
    depth: int = 3
    limit: int = 100
    cursor: Optional[str] = None  # 翻页 token（响应 next 传回）
    mode: str = "calls"  # calls | data_flow | cross_service
    repo: Optional[str] = None


class GraphQueryRequest(BaseModel):
    query: str  # Cypher
    max_rows: Optional[int] = None
    repo: Optional[str] = None


class ArchitectureRequest(BaseModel):
    aspects: Optional[list[str]] = None  # all|overview|structure|dependencies|routes|...
    path: Optional[str] = None  # 目录前缀限定
    repo: Optional[str] = None


class DetectChangesRequest(BaseModel):
    diff: str  # git diff 文本
    scope: str = "impact"  # files | impact
    direction: str = "inbound"  # inbound | outbound | both
    depth: int = 2
    limit: int = 20
    repo: Optional[str] = None


def register(app, ctx) -> None:
    from fastapi import HTTPException

    from .code_graph import CodeGraphClient, CodeGraphToolError, CodeGraphUnavailable

    client = CodeGraphClient(ctx.cfg.code_graph)

    def _project(repo: Optional[str]) -> str:
        p = client.resolve_project(repo)
        if p is None:
            raise HTTPException(status_code=400, detail=f"未知 repo: {repo}（映射见 config.code_graph.repo_project_map）")
        return p

    def _call(fn, *args, **kwargs):
        """统一图谱调用异常：工具级错误 → 400（行动建议）；不可达 → 503 + 引导。

        不向客户端泄漏堆栈。"""
        try:
            return fn(*args, **kwargs)
        except CodeGraphToolError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except CodeGraphUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    @app.get("/code-graph/health")
    def code_graph_health():
        """探测 gh-puller 可达性（不触发熔断计数，仅展示）。"""
        return client.health()

    @app.post("/code-graph/search")
    def code_graph_search(req: GraphSearchRequest):
        """search_graph：BM25/正则/语义三模搜函数/类/路由（建议优先用此而非 grep）。"""
        if not req.query and not req.name_pattern:
            raise HTTPException(status_code=400, detail="query 与 name_pattern 至少传一个")
        return _call(client.search_graph,
                     project=_project(req.repo), query=req.query, name_pattern=req.name_pattern,
                     label=req.label, limit=req.limit, offset=req.offset)

    @app.post("/code-graph/code-search")
    def code_graph_code_search(req: CodeSearchGraphRequest):
        """search_code：grep + 图增强（去重到函数、按结构重要性排序）。"""
        return _call(client.search_code,
                     project=_project(req.repo), pattern=req.pattern, mode=req.mode,
                     path_filter=req.path_filter, limit=req.limit)

    @app.post("/code-graph/trace")
    def code_graph_trace(req: TraceRequest):
        """trace_path：调用链/数据流/跨服务路径追踪（替代手写 grep 找调用关系）。"""
        return _call(client.trace_path,
                     project=_project(req.repo), function_name=req.function_name,
                     direction=req.direction, depth=req.depth, limit=req.limit,
                     cursor=req.cursor, mode=req.mode)

    @app.post("/code-graph/query")
    def code_graph_query(req: GraphQueryRequest):
        """query_graph：执行 Cypher 查知识图谱（多跳/聚合/跨服务分析）。"""
        return _call(client.query_graph,
                     project=_project(req.repo), query=req.query, max_rows=req.max_rows)

    @app.post("/code-graph/architecture")
    def code_graph_architecture(req: ArchitectureRequest):
        """get_architecture：架构总览（聚类/边界/热点/层次/依赖/路由）。"""
        return _call(client.get_architecture,
                     project=_project(req.repo), aspects=req.aspects, path=req.path)

    @app.post("/code-graph/changes")
    def code_graph_changes(req: DetectChangesRequest):
        """detect_changes：git diff → 影响面（blast radius，变更波及的调用方）。"""
        return _call(client.detect_changes,
                     project=_project(req.repo), diff=req.diff, scope=req.scope,
                     direction=req.direction, depth=req.depth, limit=req.limit)
