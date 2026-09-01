"""代码检索路由（code）：/code/search、/code/file、/code/diff、/code/versions。

版本化代码仓符号索引（本地快照），与社区/文档检索存储独立（data/code）。
从 api.py 拆出，行为不变。注册函数接收 ctx（create_app 构建的共享上下文）。

代码图谱检索（gh-puller 接入）见 api_code_graph.py，与本组并列、不替换。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from .api import _AppContext  # noqa: F401


class CodeSearchRequest(BaseModel):
    keyword: str
    version: Optional[str] = None
    limit: Optional[int] = 20
    repo: Optional[str] = None  # vllm-ascend | vllm
    path: Optional[str] = None  # 限定文件路径子串（如 worker/model_runner_v1.py）
    per_version: Optional[bool] = False  # 每个版本各自收集命中（对比版本差异用）
    kind: Optional[str] = None  # def | op | env | msg（msg=报错字面量 LIKE 子串检索）


def register(app, ctx) -> None:
    from fastapi import HTTPException

    cfg = ctx.cfg

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
