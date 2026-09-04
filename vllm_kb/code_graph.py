"""代码图谱检索（接入 gh-puller 的代码知识图谱）：MCP Streamable HTTP 客户端 + 熔断。

与 code_index（本地版本化符号快照）的关系——互补不重叠，故不走回退：
- 本地强在：版本化定位（按部署版本）、报错字面量索引、离线、跨版本 diff；
- 图谱强在：跨函数调用链/数据流、变更影响面（blast radius）、架构聚类、跨仓边、语义搜索。

gh-puller 经 **MCP Streamable HTTP** 暴露工具桌（stateless 单端点，免 initialize 握手，
每次 POST 回纯 JSON，无 SSE 流）。协议是 MCP JSON-RPC：
    POST {base_url}{path}  (path 默认 /gh-puller/graph)
    body  {"jsonrpc":"2.0","id":1,"method":"tools/call",
           "params":{"name":"<tool>","arguments":{...}}}
    resp  {"jsonrpc":"2.0","id":1,"result":{
             "content":[{"type":"text","text":"..."}],
             "structuredContent":{...},  # 结构化结果（优先取）
             "isError":false}}
本客户端面向该契约：单端点 _call_tool，各工具方法是入参透传 + 结果解包。

不可达语义：gh-puller 不可达时本模块抛 CodeGraphUnavailable，api_code_graph
转 503 + 引导用 code 命令查本地索引——不回退本地（本地无等价图谱能力，回退无意义）。
工具级错误（isError：未知函数/参数错）抛 CodeGraphToolError，api 层转 400 +
行动建议——服务健康时不再误报 503（agent 会误判服务挂了而放弃，实际只需
换个函数名形态重试）。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from .config import CodeGraphCfg


class CodeGraphUnavailable(Exception):
    """gh-puller 代码图谱服务不可达（连接失败/超时/5xx/熔断打开）。

    消息可直接展示给 agent：含上游错误 + 引导用 code 命令查本地版本化索引。
    """


class CodeGraphToolError(Exception):
    """gh-puller 工具级错误（isError=true：未知函数/参数错/预检零建议）。

    与不可达分开：服务本身健康，问题在请求参数——API 层转 400（含行动
    建议），不触发熔断计数（工具级错误不代表服务故障）。
    """


class CodeGraphClient:
    """gh-puller MCP Streamable HTTP 客户端（带熔断器：连续失败 N 次熔断 M 秒，期间零等待直接抛）。

    熔断器模式复用 search.py embedding 降级的思路（连续失败阈值→打开→到期探测），
    但语义不同：embedding 熔断→降级全文；code_graph 熔断→直接 503（无本地等价能力）。
    """

    _CIRCUIT_OPEN_SECS = 60.0  # 熔断打开时长

    def __init__(self, cfg: CodeGraphCfg):
        self.cfg = cfg
        self._circuit_open_until: float = 0.0  # 熔断打开截止时间戳（0=未打开）
        self._consecutive_fails: int = 0  # 连续失败计数（成功即清零）
        self._next_id = 0  # JSON-RPC id 自增（stateless 协议下 id 无状态含义，仅满足规范）

    # ---------------- 熔断 ----------------

    def _circuit_open(self) -> bool:
        """熔断是否打开：打开期间直接抛 CodeGraphUnavailable（零等待，不发包）。"""
        return time.time() < self._circuit_open_until

    def _on_fail(self, err: CodeGraphUnavailable) -> CodeGraphUnavailable:
        """记录失败；连续达阈值则打开熔断窗口。返回原异常（供调用方抛出）。"""
        self._consecutive_fails += 1
        threshold = max(2, self.cfg.max_retries + 1)
        if self._consecutive_fails >= threshold:
            self._circuit_open_until = time.time() + self._CIRCUIT_OPEN_SECS
        return err

    def _on_success(self) -> None:
        """成功清零连续失败计数（关闭熔断）。"""
        self._consecutive_fails = 0
        self._circuit_open_until = 0.0

    # ---------------- project 映射 ----------------

    def resolve_project(self, repo: Optional[str]) -> Optional[str]:
        """vllm-kb 内部简名（vllm-ascend/vllm）→ gh-puller project 标识。

        repo=None 时取 vllm-ascend（默认主仓）；映射表见 config.code_graph.repo_project_map。
        """
        repo = repo or "vllm-ascend"
        return self.cfg.repo_project_map.get(repo, repo)

    # ---------------- MCP JSON-RPC 调用 ----------------

    def _call_tool(self, name: str, arguments: dict) -> Any:
        """发 tools/call JSON-RPC 到 gh-puller 单端点，解包返回 structuredContent（优先）或 text。

        不可达/超时/非 2xx/非 JSON 转 CodeGraphUnavailable（触发熔断计数）；
        工具级错误（isError）转 CodeGraphToolError（不计数——服务健康，参数问题）。
        """
        if self._circuit_open():
            raise CodeGraphUnavailable(
                "代码图谱服务熔断中（连续失败，零等待降级）；引导用 `code <符号> --version <版本>` 查本地版本化索引"
            )
        if not self.cfg.base_url:
            raise CodeGraphUnavailable(
                "代码图谱服务未配置（config.code_graph.base_url 为空）；引导用 `code <符号> --version <版本>` 查本地版本化索引"
            )
        url = self.cfg.base_url.rstrip("/") + self.cfg.path
        self._next_id += 1
        body = json.dumps({
            "jsonrpc": "2.0", "id": self._next_id, "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_seconds) as r:
                resp = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                raw = e.read().decode("utf-8")
                j = json.loads(raw)
                if isinstance(j, dict) and j.get("detail"):
                    detail = j["detail"] if isinstance(j["detail"], str) else json.dumps(j["detail"], ensure_ascii=False)
                elif isinstance(j, dict) and j.get("error"):
                    detail = str(j["error"])
            except Exception:
                detail = raw[:200] if raw else ""
            raise self._on_fail(CodeGraphUnavailable(
                f"代码图谱服务 HTTP {e.code}（{name}）: {detail or e.reason}；引导用 `code <符号>` 查本地索引"
            )) from e
        except (urllib.error.URLError, OSError) as e:
            raise self._on_fail(CodeGraphUnavailable(
                f"代码图谱服务不可达（{e}）；引导用 `code <符号> --version <版本>` 查本地版本化索引"
            )) from e
        except ValueError as e:  # json.JSONDecodeError
            raise self._on_fail(CodeGraphUnavailable(
                f"代码图谱服务返回非 JSON（{name}）: {e}；引导用 `code <符号>` 查本地索引"
            )) from e

        # JSON-RPC 错误（服务端报错，非工具 isError）
        if isinstance(resp, dict) and resp.get("error"):
            err = resp["error"]
            raise self._on_fail(CodeGraphUnavailable(
                f"代码图谱服务 JSON-RPC 错误（{name}）: {err}；引导用 `code <符号>` 查本地索引"
            ))

        result = resp.get("result") if isinstance(resp, dict) else None
        if not isinstance(result, dict):
            raise self._on_fail(CodeGraphUnavailable(
                f"代码图谱服务响应缺 result 信封（{name}）；引导用 `code <符号>` 查本地索引"
            ))

        # 工具级错误（isError=true：未知工具/未知函数/参数错）——服务健康，参数问题
        if result.get("isError"):
            text = ""
            content = result.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                text = content[0].get("text", "")
            sc = result.get("structuredContent")
            detail = ""
            if isinstance(sc, dict) and sc.get("error"):
                detail = str(sc["error"])
            raise CodeGraphToolError(
                f"代码图谱工具 {name} 报错: {detail or text or 'isError'}"
            )

        self._on_success()
        # 优先返回结构化结果；无则退回 content[0].text（纯文本工具结果）
        sc = result.get("structuredContent")
        if sc is not None:
            return sc
        content = result.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            return content[0].get("text", "")
        return None

    # ---------------- 工具调用（对齐 gh-puller-mcp 工具名） ----------------

    def search_graph(self, *, project: str, query: Optional[str] = None,
                     name_pattern: Optional[str] = None, label: Optional[str] = None,
                     limit: int = 10, offset: int = 0) -> Any:
        """search_graph：BM25/正则/语义三模搜函数/类/路由。query 与 name_pattern 至少传一个。"""
        args: dict[str, Any] = {"project": project, "limit": limit, "offset": offset}
        if query is not None:
            args["query"] = query
        if name_pattern is not None:
            args["name_pattern"] = name_pattern
        if label is not None:
            args["label"] = label
        return self._call_tool("search_graph", args)

    def search_code(self, *, project: str, pattern: str, mode: str = "compact",
                    path_filter: Optional[str] = None, limit: int = 10) -> Any:
        """search_code：grep + 图增强（去重到函数、按结构重要性排序）。mode: compact|full|files。"""
        args: dict[str, Any] = {"project": project, "pattern": pattern, "mode": mode, "limit": limit}
        if path_filter is not None:
            args["path_filter"] = path_filter
        return self._call_tool("search_code", args)

    _NOT_FOUND_MARKERS = ("not found", "unknown function")  # 上游零命中错误特征（大小写不敏感）

    def trace_path(self, *, project: str, function_name: str, direction: str = "both",
                   depth: int = 3, limit: int = 100, cursor: Optional[str] = None,
                   mode: str = "calls") -> Any:
        """trace_path：调用链/数据流/跨服务路径追踪。mode: calls|data_flow|cross_service。

        function_name 双形态（上游 CBM ≥0.10.8 均支持）：裸短名（bare name 列匹配）
        或完整 qn（`{index_name}.{module}.{symbol}`，bare name 未命中时按
        project+qn 精确回退——`{index_name}.` 前缀是 CBM 规范 qn 的一部分，勿剥；
        剥前缀的 module-relative 名与末两段 partial 形态均不可解析）。

        原样透传策略（保 cursor 翻页参数一致——上游要求 cursor 与原参数完全相同）：
        - dotted 输入（完整 qn / module.func）：不改写不预检（qn 自身已精确）；
        - 裸短名：先 search_graph 唯一性预检——多命中 → ambiguous 候选（200，
          候选含完整 qn，agent 拿 qn 精确重试）；唯一/零命中/预检不可达 → 原样透传；
        - 翻页（cursor）跳过预检。

        零命中错误增强：上游 function not found（属性/descriptor 节点不可直接追踪
        ——上游仅匹配函数/方法节点）→ 补一次轻量预检取节点 label → 400 可读错误
        （property/field/attribute 判定属性节点 + 引导改 trace 宿主类方法）。
        """
        args: dict[str, Any] = {"project": project, "function_name": function_name,
                                "direction": direction, "depth": depth, "limit": limit, "mode": mode}
        if cursor is not None:
            args["cursor"] = cursor
            return self._call_tool("trace_path", args)

        pre: Optional[list] = None
        prechecked = False
        if "." not in function_name:
            prechecked = True
            pre = self._precheck_unique(project, function_name)
            if pre and len(pre) > 1:
                return {
                    "status": "ambiguous",
                    "function_name": function_name,
                    "matched": len(pre),
                    "candidates": pre[:20],
                    "hint": "同名多个节点；从候选确认目标（file/lines 定位），用其完整 qn 精确重试",
                }
        try:
            return self._call_tool("trace_path", args)
        except CodeGraphToolError as e:
            if not any(m in str(e).lower() for m in self._NOT_FOUND_MARKERS):
                raise  # 参数错等非零命中错误：原样抛出
            # 零命中：dotted 输入未预检过，补一次轻量预检取节点 label（裸短名复用已有结果）
            if not prechecked:
                pre = self._precheck_unique(project, function_name.rsplit(".", 1)[-1])
            label = str(pre[0].get("label", "")) if pre else ""
            raise self._trace_not_found_error(function_name, label) from e

    @staticmethod
    def _trace_not_found_error(original: str, label: str) -> "CodeGraphToolError":
        """零命中可读错误：属性节点判定 + 行动引导（API 层转 400 detail 展示给 agent）。"""
        if label and any(k in label.lower() for k in ("property", "field", "attribute")):
            return CodeGraphToolError(
                f"节点 {original}（label={label}）为属性/descriptor 节点——上游 trace_path 当前"
                f"仅匹配函数/方法节点。建议：改 trace 宿主类或其方法（code-graph search 查"
                f"label=Function/Method 节点）；属性节点解析依赖 gh-puller 侧适配"
                f"（见 docs/gh-puller-integration-checklist.md 待适配项）"
            )
        return CodeGraphToolError(
            f"函数 {original} 上游 trace 零命中。可能为属性/descriptor 节点（上游仅匹配"
            f"函数/方法）或名称形态不匹配；建议 code-graph search 确认目标节点后用短名或"
            f"完整 qn 重试"
        )

    _PRECHECK_LIMIT = 51  # >50 视为大量候选，按多命中处理

    def _precheck_unique(self, project: str, short: str):
        """search_graph 预检短名唯一性。

        返回 name 精确等于短名的候选行列表；None = 跳过预检（预检不可达/
        工具级错/返回非预期结构/无精确同名行）——调用方按旧行为透传原输入，
        预检不引入新的失败模式（无精确同名行时替换函数名反而可能错配）。
        """
        import re as _re

        try:
            pre = self.search_graph(project=project,
                                    name_pattern="^" + _re.escape(short) + "$",
                                    limit=self._PRECHECK_LIMIT)
        except (CodeGraphUnavailable, CodeGraphToolError):
            return None
        rows = self._search_rows(pre)
        if not rows:
            return None
        exact = [r for r in rows if r.get("name") == short]
        return exact or None

    @staticmethod
    def _search_rows(pre: Any):
        """search_graph 结果提取候选行（rows/groups 两种归一形态）。

        行字段 name/label/qn/file/lines（分组树 flatten 后 name=短名、qn=完整）。
        非预期结构返回 None。
        """
        if not isinstance(pre, dict):
            return None
        rows = pre.get("rows")
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
        groups = pre.get("groups")
        out = []
        if isinstance(groups, list):
            for g in groups:
                if not isinstance(g, dict):
                    continue
                prefix = g.get("group") or g.get("prefix") or ""
                for m in g.get("members") or []:
                    if isinstance(m, dict):
                        row = dict(m)
                        row.setdefault("name", m.get("name") or "")
                        if prefix and not row.get("qn"):
                            row["qn"] = f"{prefix}.{m.get('name', '')}"
                        out.append(row)
        return out or None

    def query_graph(self, *, project: str, query: str, max_rows: Optional[int] = None) -> Any:
        """query_graph：执行 Cypher 查询知识图谱（多跳/聚合/跨服务分析）。"""
        args: dict[str, Any] = {"project": project, "query": query}
        if max_rows is not None:
            args["max_rows"] = max_rows
        return self._call_tool("query_graph", args)

    def get_architecture(self, *, project: str, aspects: Optional[list[str]] = None,
                         path: Optional[str] = None) -> Any:
        """get_architecture：架构总览（聚类/边界/热点/层次/依赖）。aspects: all|overview|structure|..."""
        args: dict[str, Any] = {"project": project}
        if aspects is not None:
            args["aspects"] = aspects
        if path is not None:
            args["path"] = path
        return self._call_tool("get_architecture", args)

    def detect_changes(self, *, project: str, diff: str, scope: str = "impact",
                       direction: str = "inbound", depth: int = 2, limit: int = 20) -> Any:
        """detect_changes：git diff → 影响面（blast radius）。scope: files|impact；direction: inbound|outbound|both。"""
        args: dict[str, Any] = {"project": project, "diff": diff, "scope": scope,
                                "direction": direction, "depth": depth, "limit": limit}
        return self._call_tool("detect_changes", args)

    # ---------------- 健康 ----------------

    def health(self) -> dict:
        """探测 gh-puller 可达性（不发熔断计数副作用——探测结果仅供 /code-graph/health 展示）。

        熔断打开时返回明确状态，不打断调用方展示。
        """
        if self._circuit_open():
            return {"status": "circuit_open", "note": "连续失败熔断中，端点暂不可用"}
        if not self.cfg.base_url:
            return {"status": "not_configured", "note": "config.code_graph.base_url 未配置"}
        try:
            # 轻量探测：tools/list（空 cursor，stateless），验证端点 + 协议可达
            body = json.dumps({"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}}).encode("utf-8")
            req = urllib.request.Request(
                self.cfg.base_url.rstrip("/") + self.cfg.path, data=body,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                resp = json.loads(r.read().decode("utf-8"))
            ok = isinstance(resp, dict) and "result" in resp
            return {"status": "ok" if ok else "bad_response", "upstream": resp.get("result", {}) if ok else resp}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)[:200]}
