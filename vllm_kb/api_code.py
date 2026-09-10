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
    repo: Optional[str] = None  # vllm-ascend | vllm | fork:{model} | img:{tag}
    path: Optional[str] = None  # 限定文件路径子串（如 worker/model_runner_v1.py）
    per_version: Optional[bool] = False  # 每个版本各自收集命中（对比版本差异用）
    kind: Optional[str] = None  # def | op | env | msg（msg=报错字面量 LIKE 子串检索）


def register(app, ctx) -> None:
    from fastapi import HTTPException

    cfg = ctx.cfg

    def _code_index_for(repo: Optional[str]):
        """按 repo 取代码仓访问器：vllm-ascend（默认）| vllm | fork:{model} | img:{tag}。

        - fork:{model}：0day 开发分支的 **vllm 源码**快照（data/code/forks/{model}/）；
        - img:{tag}：0day 镜像内 **vllm-ascend 插件源码**（data/code/images/{tag}/）；
        - img（无 tag）：聚合视图，只用于 /code/versions 列举（其余端点需带 tag）。

        两个命名空间都与官方 rc/release 版本物理隔离——必须显式传前缀才会命中，
        默认检索永不混入 0day 代码。
        """
        from .code_index import CodeIndexError, VersionedCode

        r = repo or "vllm-ascend"
        if r not in ("vllm-ascend", "vllm") and not r.startswith(("fork:", "img:")):
            return None
        try:
            return VersionedCode(cfg, repo=r)
        except CodeIndexError as e:
            if "非法" in str(e):
                raise HTTPException(status_code=400, detail=str(e))
            return None
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
        import json as _json

        r = repo or "vllm-ascend"
        # img（无 tag）：聚合列举已提取镜像——agent 发现 img: 前缀的入口
        if r == "img":
            from .code_index import list_image_snapshots

            images = list_image_snapshots(cfg.resolve(cfg.storage.code_root))
            return {
                "repo": "img",
                "versions": [i["tag"] for i in images],
                "images": images,
                "note": ("已提取的 0day 镜像插件源码（repo=img:{tag} 检索；版本键 = 镜像 tag）。"
                         "未列的镜像：python scripts/build_image_snapshots.py --tag <tag>"),
            }
        ci = _code_index_for(repo)
        if ci is None:
            return {"repo": r, "versions": [],
                    "note": "code_index 未初始化（检查 config.storage.code_root）"}
        payload = {"repo": r, "versions": ci.available_versions,
                   "note": "预存版本源码快照；未列的版本请先运行 scripts/build_code_snapshots.py 或 build_vllm_snapshots.py"}
        meta_path = ci.root / "meta.json"
        if r.startswith(("fork:", "img:")) and meta_path.exists():
            try:
                payload["meta"] = _json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if r.startswith("fork:"):
            payload["note"] = ("fork 快照版本 = 镜像锁定 commit（SHA 前 12 位）；"
                               "预存请运行 scripts/build_fork_snapshots.py")
        elif r.startswith("img:"):
            payload["note"] = ("镜像插件源码快照（版本键 = 镜像 tag = "
                               f"{r[4:]}）；预存请运行 scripts/build_image_snapshots.py --tag {r[4:]}")
        return payload

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

    def _split_repo_version(spec: str, default_repo: Optional[str]) -> tuple[Optional[str], str]:
        """拆分跨命名空间 diff 的版本参数。

        支持形态（镜像 tag 不含冒号，故不会误拆普通版本号）：
        - `vllm-ascend:{版本}` / `vllm:{版本}` → (repo, 版本)
        - `img:{tag}`            → (img:{tag}, {tag})（该命名空间的版本键就是镜像 tag）
        - `fork:{model}@{sha12}` → (fork:{model}, {sha12})
        - `fork:{model}`         → (fork:{model}, '')：留空表示"用该命名空间唯一版本"，
                                   多于一个版本时由调用方提示补 @{sha12}
        - 其它                   → (default_repo, 原样版本号)
        """
        for ns in ("vllm-ascend", "vllm"):
            if spec.startswith(ns + ":"):
                return ns, spec[len(ns) + 1:]
        if spec.startswith("img:"):
            return spec, spec[4:]
        if spec.startswith("fork:"):
            body = spec[5:]
            if "@" in body:
                model, ver = body.split("@", 1)
                return f"fork:{model}", ver
            return spec, ""
        return default_repo, spec

    @app.get("/code/diff")
    def code_diff(version1: str, version2: str, path: str,
                  keyword: Optional[str] = None, context: int = 3,
                  repo: Optional[str] = None):
        """精确 diff：同一文件在两个快照间的 unified diff（可跨命名空间）。

        - 同仓跨版本：`diff v0.22.1rc1 v0.23.0rc1 <path>` —— 定位"哪个版本引入/修改"；
        - 跨命名空间：版本参数带前缀，如 `diff img:glm5.2 vllm-ascend:0.23.0 <path>`
          —— "这个 0day 镜像的插件代码相对官方同基线改了什么"；
          `fork:{model}@{sha12}` 亦可与官方版本对比；
        - `--keyword` 只显示含关键词的差异行。
        """
        import difflib

        repo1, ver1 = _split_repo_version(version1, repo)
        repo2, ver2 = _split_repo_version(version2, repo)
        ci1 = _code_index_for(repo1)
        if ci1 is None:
            raise HTTPException(status_code=503, detail=f"code_index 未初始化（repo={repo1}）")
        ci2 = ci1 if repo2 == repo1 else _code_index_for(repo2)
        if ci2 is None:
            raise HTTPException(status_code=503, detail=f"code_index 未初始化（repo={repo2}）")
        # fork:{model} 未给 @sha：命名空间唯一版本时自动取用（0day 镜像通常只锁一个 SHA）
        for ci, rp, ver in ((ci1, repo1, ver1), (ci2, repo2, ver2)):
            if rp and rp.startswith("fork:") and not ver:
                vs = ci.available_versions
                if len(vs) == 1:
                    if ci is ci1:
                        repo1, ver1 = rp, vs[0]
                    else:
                        repo2, ver2 = rp, vs[0]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{rp} 有多个版本 {vs}：请用 fork:{{model}}@{{sha12}} 指定",
                    )
        p1 = _code_call(ci1.find_file, ver1, path)
        p2 = _code_call(ci2.find_file, ver2, path)
        missing = [f"{r}:{v}" for r, v, p in ((repo1, ver1, p1), (repo2, ver2, p2)) if p is None]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"未预存文件 {path}（缺失：{missing}）；"
                       f"repo={repo1 or 'vllm-ascend'} / {repo2 or 'vllm-ascend'}",
            )
        t1 = p1.read_text(encoding="utf-8", errors="replace").splitlines()
        t2 = p2.read_text(encoding="utf-8", errors="replace").splitlines()
        diff_lines = list(difflib.unified_diff(
            t1, t2, fromfile=f"{repo1 or 'vllm-ascend'}:{ver1}:{path}",
            tofile=f"{repo2 or 'vllm-ascend'}:{ver2}:{path}",
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
            "path": path, "v1": version1, "v2": version2,
            "repo": repo1 or "vllm-ascend", "repo2": repo2 or "vllm-ascend",
            "lines1": len(t1), "lines2": len(t2), "keyword": keyword, "context": context,
            "diff": "\n".join(out),
            "note": f"无包含关键词 '{keyword}' 的差异行（v1/v2 该文件可能无差异）"
                    if shown == 0 and keyword else None,
        }
