"""vllm-kb 只读检索客户端（标准库实现，无第三方依赖）。

只调用只读 API 端点；本文件不含任何写入逻辑（知识库数据只能由用户运行流水线修改）。

服务端地址解析（存算分离：数据在远程服务器，本 client 只发 HTTP 请求）：
1. 命令行 --base 显式指定（最高优先）；
2. 环境变量 VLLM_KB_BASE（远程部署推荐）；
3. 默认 http://127.0.0.1:8000（本地）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.parse


DEFAULT_BASE = os.environ.get("VLLM_KB_BASE", "http://127.0.0.1:8000")


def _force_utf8_stdio() -> None:
    """强制 stdout/stderr 以 UTF-8 输出（Python 3.7+）。

    检索结果含中文（标题/URL），而 Windows PowerShell 下 Python 默认按代码页
    （GBK/cp936）编码 stdout——中文会乱码或抛 UnicodeEncodeError，调用方
    （agent/管道）不得不额外设 PYTHONIOENCODING=utf-8。这里在进程内直接
    reconfigure，任何环境（PowerShell/重定向/容器）输出一致 UTF-8，无需外部设置。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass  # 非 TTY / 被替换的流 / 老版本：保持原样


class ClientError(Exception):
    """客户端可预期的失败（连接失败/HTTP 错误/非 JSON 响应），消息可直接展示给 agent。"""


def _error_detail(err: "urllib.error.HTTPError") -> str:
    """从 HTTPError 响应体提取 FastAPI 风格 detail（{"detail": "..."}）；解析失败退回原文截断。"""
    try:
        body = err.read().decode("utf-8")
    except Exception:
        return ""
    try:
        j = json.loads(body)
        if isinstance(j, dict) and j.get("detail"):
            d = j["detail"]
            return d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)
    except Exception:
        pass
    return body.strip()[:200]


def _request(url: str, timeout: int, payload: dict | None = None) -> dict:
    """发只读请求并统一错误处理（调用方无需再处理 HTTP/连接异常）：

    - HTTP 4xx/5xx：抛 ClientError，含状态码 + 服务端 detail（如"版本未预存/图未构建"）；
    - 连接失败/超时：抛 ClientError，说明服务地址与启动方式；
    - 200 但非 JSON 响应：抛 ClientError。
    """
    if payload is not None:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
    else:
        req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # 4xx/5xx（HTTPError 是 URLError 子类，须先捕获）
        raise ClientError(f"API 错误 {e.code}（{e.url}）: {_error_detail(e) or e.reason}") from e
    except urllib.error.URLError as e:
        raise ClientError(
            f"无法连接 API（{e.reason}）：请确认服务已启动（python scripts/serve_api.py）"
            f"且 --base 地址正确（当前 {url.split('?')[0]}）"
        ) from e
    except OSError as e:  # socket.timeout / 连接重置等
        raise ClientError(f"请求失败（{e}）：请检查网络与服务状态") from e
    except ValueError as e:  # json.JSONDecodeError
        raise ClientError(f"API 返回非 JSON 响应（HTTP 200）：{e}") from e


def _get(base: str, path: str, params: dict | None = None) -> dict:
    url = base.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return _request(url, timeout=30)


def _post(base: str, path: str, payload: dict) -> dict:
    return _request(base.rstrip("/") + path, timeout=60, payload=payload)


def fmt_search(data: dict) -> str:
    lines = []
    ctx = data.get("context") or {}
    if ctx.get("component"):
        comps = ctx.get("companions") or {}
        detail = ", ".join(f"{k} {v}" for k, v in comps.items()) or "(无配套记录)"
        lines.append(f"组件: {ctx['component']}:{ctx.get('version')} | 配套: {detail}")
    if data.get("degraded"):
        lines.append(f"[提示] embedding 不可用，已降级为全文检索（向量召回跳过）：{data['degraded'][:100]}")
    for i, r in enumerate(data.get("results", []), 1):
        c = r["confidence"]
        tag = "已解决" if r["resolved"] else "未解决"
        section = r.get("section") or ""
        sec_txt = f"  章节: {section}" if section else ""
        lines.append(
            f"[{i}] final={r['final']:.3f} sim={r['similarity']:.3f} conf={c['score']:.3f} "
            f"[{tag}] 组件={r['component'] or '-'} 版本参考={r['version_ref'] or '-'}"
        )
        lines.append(f"    w_time={c['w_time']:.3f} w_ver={c['w_ver']:.3f} w_rel={c['w_rel']:.3f}")
        lines.append(f"    {r['title']}{sec_txt}")
        ver = r.get("verification") or ""
        ver_txt = f"  验证={ver}" if ver else ""
        lines.append(f"    {r['url']}  status={r['status']}{ver_txt}  version_span={r['version_span']}")
        lines.append(f"    ...{r['snippet'][:180]}...")
    return "\n".join(lines)


def fmt_version(data: dict) -> str:
    rel = data.get("release")
    if not data.get("calendar_loaded"):
        return f"版本 {data['version']}: 日历未加载（运行 scripts/build_release_calendar.py）"
    if rel:
        return (f"版本 {data['version']} = {data['kind']}（正式版/预发布判断）\n"
                f"  tag={rel['tag']}  发布日期={rel['date'][:10]}  kind={rel['kind']}")
    return f"版本 {data['version']}: 日历中无此版本（kind={data['kind']}）"


def fmt_signature(data: dict) -> str:
    lines = ["===== 提取的签名 ====="]
    lines.append(data.get("signatures_text") or "(未提取到签名)")
    sw = data.get("signal_words") or []
    if sw:
        lines.append("\n===== 命中的社区高频信号词（agent 判断）=====")
        for w in sw:
            lines.append(f"  {w['word']}  (count={w['count']}, score={w['score']})")
    lines.append("\n===== 精确检索命中 =====")
    results = data.get("results", [])
    if not results:
        lines.append("(无精确命中)")
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] score={r['score']:.2f} 命中: {', '.join(r['hit_signatures'][:6])}")
        lines.append(f"    {r['title']}")
        lines.append(f"    {r['url']}")
    th = data.get("title_hits") or []
    if th:
        lines.append("\n===== 标题精确命中（信号词→标题）=====")
        for r in th:
            tag = "已解决" if r["resolved"] else "未解决"
            lines.append(f"  [{tag}] ({r['signal']}) {r['title'][:90]}")
            lines.append(f"    {r['url']}")
    return "\n".join(lines)


def fmt_title(data: dict) -> str:
    lines = [f"标题含 '{data.get('keyword')}' 的文档（component={data.get('component') or '全部'}）:"]
    results = data.get("results", [])
    if not results:
        lines.append("(无命中)")
    for r in results:
        tag = "已解决" if r["resolved"] else "未解决"
        lines.append(f"  [{tag}] {r['title']}")
        lines.append(f"    {r['url']}  comp={r['component']}")
    return "\n".join(lines)


def fmt_matrix(data: dict, limit: int = 100) -> str:
    rows = data.get("rows") or []
    if not rows:
        return "(配套矩阵未构建或为空 —— 运行 scripts/build_companion_matrix.py)"
    shown = rows[:limit]
    lines = [f"配套矩阵共 {len(rows)} 行（调试/管理用；日常查询用 companion <组件> <版本>）:"]
    for r in shown:
        parts = "  ".join(
            f"{k}={v}" for k, v in r.items() if v and k not in ("notes", "source")
        )
        notes = f"  notes: {r.get('notes')}" if r.get("notes") else ""
        src = f"  source: {r.get('source')}" if r.get("source") else ""
        lines.append(f"  {parts}{notes}{src}")
    if len(rows) > limit:
        lines.append(f"  … 共 {len(rows)} 行（显示前 {limit}；用 --limit 调整）")
    return "\n".join(lines)


def fmt_code_versions(data: dict) -> str:
    versions = data.get("versions", [])
    repo = data.get("repo", "vllm-ascend")
    if not versions:
        return f"[{repo}] (未预存版本) {data.get('note', '')}"
    return f"[{repo}] 可用版本:\n" + "\n".join(f"  {v}" for v in versions) + \
           f"\n提示: {data.get('note', '')}"


def fmt_code_hits(data: dict) -> str:
    lines = [f"检索模式: {data.get('mode')}  symbol: {data.get('symbol')}  version: {data.get('version') or '(全部)'}"]
    hits = data.get("hits", [])
    if not hits:
        lines.append("(无命中)")
    elif data.get("mode") == "grep_per_version":
        # 按版本分组，便于对比"哪个版本引入/移动了该代码"
        by_ver: dict[str, list[dict]] = {}
        for h in hits:
            by_ver.setdefault(h.get("version", ""), []).append(h)
        for v, vh in by_ver.items():
            lines.append(f"  [{v}] {len(vh)} 处命中:")
            for h in vh[:8]:
                lines.append(f"    {h.get('file')}:{h.get('line')}  {(h.get('snippet') or '')[:100]}")
            if len(vh) > 8:
                lines.append(f"    … 共 {len(vh)} 处")
    else:
        for h in hits[:20]:
            v = h.get("version", "")
            f = h.get("file", "")
            ln = h.get("line", "")
            sn = (h.get("snippet") or "")[:120]
            lines.append(f"  [{v}] {f}:{ln}  {sn}")
    return "\n".join(lines)


def fmt_code_file(data: dict) -> str:
    return f"===== {data['version']}:{data['path']} =====\n" + (data.get("content") or "")


def fmt_diff(data: dict) -> str:
    lines = [f"===== {data['path']}  {data['v1']} ({data.get('lines1')} 行) → {data['v2']} ({data.get('lines2')} 行) ====="]
    diff = data.get("diff") or ""
    if diff:
        lines.append(diff)
    else:
        lines.append("(无差异)")
    if data.get("note"):
        lines.append(f"[diff] {data['note']}")
    return "\n".join(lines)


def _resolve_graph_doc(arg: str, kind: str) -> str:
    """图查询的 doc 参数解析：完整 source_id 或 repo#number 简写。

    kind: issue | pr（简写时决定 source_id 的类型段）。
    source_id 的 repo 段按 github_pull 规则 '/' -> '-'（github:vllm-project-vllm:issue:50237）。
    """
    if arg.startswith("github:"):
        return arg
    if "#" in arg:
        repo, num = arg.rsplit("#", 1)
        repo = {"vllm": "vllm-project/vllm", "vllm-ascend": "vllm-project/vllm-ascend",
                "ascend": "vllm-project/vllm-ascend"}.get(repo, repo)
        return f"github:{repo.replace('/', '-')}:{kind}:{num}"
    return arg


def fmt_graph_chain(data: dict) -> str:
    if not data.get("found"):
        return f"[graph] issue 不存在于图中: {data.get('issue_id')}"
    lines = [
        f"issue {data['number']} [{data['status']}]  {data['title']}",
        f"  {data['url']}   resolved_at={data.get('resolved_at') or '-'}",
    ]
    fixes = data.get("fixes") or []
    if not fixes:
        lines.append("修复链路: (图中无 FIXES 边——无 PR 声明修复此 issue)")
    for f in fixes:
        rel = f.get("release_tag")
        rel_txt = f"→ 落地 release: {rel} ({f.get('release_date') or '-'})" if rel else "→ 未发布（PR 已合并但日历中尚无对应 tag）"
        lines.append(f"修复 PR {f['pr_id']} [{f['status']}]  {f['title']}")
        lines.append(f"  merged_at={f.get('merged_at') or '-'}")
        lines.append(f"  {rel_txt}")
    if fixes and not data.get("released"):
        lines.append("结论: 有修复 PR 但尚未进入任何 release（需升级到修复合并后的版本或等发布）")
    return "\n".join(lines)


def fmt_graph_fixes(data: dict) -> str:
    if not data.get("found"):
        return f"[graph] PR 不存在于图中: {data.get('pr_id')}"
    lines = [f"PR {data['number']} [{data['status']}]  {data['title']}", f"  {data['url']}"]
    rels = data.get("releases") or []
    if rels:
        lines.append("落地 release: " + ", ".join(f"{r['tag']} ({r['date']})" for r in rels))
    else:
        lines.append("落地 release: (无 —— 尚未进入任何 release)")
    fixes = data.get("fixes") or []
    if not fixes:
        lines.append("修复的 issues: (无 FIXES 边)")
    for i in fixes:
        lines.append(f"  修复 issue {i['issue_id']} [{i['status']}]  {i['title']}")
        lines.append(f"    {i['url']}")
    return "\n".join(lines)


def fmt_graph_sig(data: dict) -> str:
    ent = data.get("entity_id")
    ent_txt = f"，实际实体: {ent}" if ent else ""
    lines = [f"签名 '{data['signature']}'（实体类型: {data.get('entity_type') or '未命中'}{ent_txt}）"]
    docs = data.get("docs") or []
    if not docs:
        lines.append("(图中无提及此签名的 issue/PR —— 实体可能未被任何文档提及，或数据未积累)")
    for d in docs:
        lines.append(f"  [{d['status']}] {d['doc_id']}  {d['title']}")
    lines.append(f"提示: {data.get('note', '')}")
    return "\n".join(lines)


def fmt_graph_stats(data: dict) -> str:
    if not data.get("built"):
        return f"[graph] 未构建: {data.get('note')}"
    nodes = data.get("nodes") or {}
    rels = data.get("rels") or {}
    return (f"图已构建: 节点 {sum(nodes.values())}（Issue {nodes.get('Issue', 0)} / PR {nodes.get('PR', 0)} / "
            f"Release {nodes.get('Release', 0)} / 实体 {sum(v for k, v in nodes.items() if k not in ('Issue', 'PR', 'Release'))}），"
            f"边 {sum(rels.values())}（FIXES {rels.get('FIXES', 0)} / MERGED_IN {rels.get('MERGED_IN', 0)} / "
            f"MENTIONS {rels.get('MENTIONS', 0)}）")


def fmt_graph_tags(data: dict) -> str:
    lines = [f"标签 '{data['tag']}'（tier={data.get('tier') or '-'}）→ 图内 {data['count']} 个节点:"]
    if not data.get("docs"):
        lines.append("(图中无打标文档——标签可能仅在词典中，重建图后生效)")
    for d in data.get("docs", []):
        lines.append(f"  [{d['doc_type']}] {d['doc_id']}  {d['title']}")
    lines.append(f"提示: {data.get('note', '')}")
    return "\n".join(lines)


def fmt_graph_evidence(data: dict) -> str:
    if not data.get("found"):
        return f"[graph] 文档不存在于图中（或不是业务文档）: {data.get('doc_id')}"
    lines = [f"文档互证（Evidence）: {data['title']} → {data['count']} 篇佐证文档:"]
    if not data.get("corroborated_by"):
        lines.append("(无共享 ≥2 个实体的其他文档——多来源互证尚未建立，可导入更多同主题手册)")
    for d in data.get("corroborated_by", []):
        lines.append(f"  {d['doc_id']}  {d['title']}")
        lines.append(f"    共享实体: {', '.join(d['shared'])}")
    lines.append(f"提示: {data.get('note', '')}")
    return "\n".join(lines)


def fmt_tags(data: dict) -> str:
    lines = ["文档标签能力目录（主题/领域类=有什么知识可查；具体作用类=文档能帮我做什么）:"]
    for tier, label in (("domain", "主题/领域类"), ("purpose", "具体作用类")):
        items = (data.get("groups") or {}).get(tier) or []
        if items:
            lines.append(f"[{label}] " + "  ".join(f"{x['name']}({x['docs']}篇)" for x in items))
        else:
            lines.append(f"[{label}] (暂无)")
    lines.append("按标签检索文档: tags docs <标签>；问题→标签匹配: context <问题描述>")
    return "\n".join(lines)


def fmt_tags_docs(data: dict) -> str:
    lines = [f"标签 '{data['tag']}' 下的文档（{data['count']} 篇）:"]
    if not data.get("docs"):
        lines.append("(无文档——标签可能仅在词典中)")
    for d in data.get("docs", []):
        ver = f"  验证={d['verification']}" if d.get("verification") else ""
        lines.append(f"  {d['doc_id']}  {d['title']}{ver}")
        if d.get("url"):
            lines.append(f"    {d['url']}")
    lines.append(f"提示: {data.get('note', '')}")
    return "\n".join(lines)


def fmt_tags_match(data: dict) -> str:
    lines = ["知识领域命中（文档能力发现——先读相关文档，再结合 issue/代码下结论）:"]
    matched = data.get("matched") or []
    if not matched:
        lines.append("(未命中任何标签：知识库暂无对应主题文档——走 signature/search/code 流程)")
        return "\n".join(lines)
    for m in matched:
        tier_label = "领域" if m.get("tier") == "domain" else "作用"
        n = m.get("docs", 0)
        zero_note = "" if n else "（词典已注册、暂无打标文档——可 search 语义检索该主题确认）"
        lines.append(f"[{tier_label}] {m['name']}（{n} 篇）{zero_note}")
        for d in m.get("top", []):
            lines.append(f"    → {d['doc_id']}  {d['title']}（验证={d.get('verification') or '-'}）")
    lines.append("提示: 同时命中领域×作用的文档交集最相关；读取全文 doc <id>；按标签列表 tags docs <标签>")
    return "\n".join(lines)


def main() -> None:
    _force_utf8_stdio()
    ap = argparse.ArgumentParser(description="vllm-kb 只读检索客户端")
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"API 地址（默认 {DEFAULT_BASE}）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("search", help="检索（支持 组件:版本 问题 格式）")
    p.add_argument("query")
    p.add_argument("--version", default=None, help="目标版本（普通查询用）")
    p.add_argument("--component", default=None)
    p.add_argument("--version-of-component", dest="comp_version", default=None, help="组件版本")
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--tag", action="append", default=None, metavar="TAG",
                   help="按文档标签过滤（可多次，全部包含才保留，如 --tag HCCL --tag 超时排查）")

    p = sub.add_parser("signature", help="签名精确检索：从原始报错提取签名并精确匹配")
    p.add_argument("text", help="原始报错文本/日志片段")
    p.add_argument("--component", default=None)
    p.add_argument("--top", type=int, default=15)

    p = sub.add_parser("title", help="标题子串精确检索（已知现象找 issue 最快路径）")
    p.add_argument("keyword")
    p.add_argument("--component", default=None)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--match", default="contains", choices=["contains", "prefix"])

    p = sub.add_parser("version", help="版本形态判断（正式 release / rc / pre）")
    p.add_argument("version")
    p.add_argument("--repo", default="vllm-ascend", help="仓库（vllm-ascend 默认 | vllm）")

    p = sub.add_parser("code", help="版本化代码仓检索")
    p.add_argument("keyword", nargs="?", default=None,
                   help="符号/关键词（如 DispatchFFNCombine、halMemCreate；--file 读取文件模式可省略）")
    p.add_argument("--version", default=None, help="限定版本（默认全部已预存版本）")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--repo", default="vllm-ascend", help="仓库：vllm-ascend（默认）| vllm")
    p.add_argument("--file", dest="code_file", metavar="PATH", default=None,
                   help="直接读取指定版本源码文件（需配合 --version；此时忽略 keyword）")
    p.add_argument("--max-chars", dest="code_max_chars", type=int, default=20000,
                   help="--file 读取的最大字符数（默认 20000；函数体较长时可调大，截断会明确标注）")
    p.add_argument("--in-file", dest="in_file", metavar="SUBSTR", default=None,
                   help="限定 grep 的文件路径子串（如 worker/model_runner_v1.py）")
    p.add_argument("--per-version", action="store_true",
                   help="每个版本各自收集命中（输出各版本行号，对比哪个版本引入/移动了该代码）")
    p.add_argument("--kind", choices=["def", "op", "env", "msg"], default=None,
                   help="限定符号类型：msg=报错字面量子串检索（raise/assert/logger.error 的字符串参数，"
                        "定位'报错文本来自哪段代码'）；默认全部类型")

    p = sub.add_parser("code-versions", help="列出已预存的代码仓版本")
    p.add_argument("--repo", default="vllm-ascend", help="仓库：vllm-ascend（默认）| vllm")

    p = sub.add_parser("diff", help="跨版本精确 diff：对比两个版本同一文件的 unified diff")
    p.add_argument("v1", help="旧版本（如 v0.22.1rc1）")
    p.add_argument("v2", help="新版本（如 v0.23.0rc1）")
    p.add_argument("path", help="文件路径（相对仓库根，如 vllm_ascend/worker/model_runner_v1.py）")
    p.add_argument("--keyword", default=None, help="只显示包含该关键词的差异行（定位修复代码）")
    p.add_argument("--context", type=int, default=3, help="diff 上下文行数")
    p.add_argument("--repo", default="vllm-ascend", help="仓库：vllm-ascend（默认）| vllm")

    p = sub.add_parser("doc", help="读取整篇文档")
    p.add_argument("doc_id")

    sub.add_parser("health", help="健康检查")
    sub.add_parser("components", help="组件分布")
    sub.add_parser("stats", help="知识库统计")

    p = sub.add_parser("companion", help="配套反向展开")
    p.add_argument("component")
    p.add_argument("version")

    p = sub.add_parser("matrix", help="全量配套矩阵（调试/管理用；日常用 companion）")
    p.add_argument("--limit", type=int, default=100, help="最多显示行数（默认 100）")

    # Phase 2：图检索
    p = sub.add_parser("graph", help="Phase 2 图检索（关系追溯）")
    gsub = p.add_subparsers(dest="graph_cmd", required=True)
    g = gsub.add_parser("chain", help="核心链路：issue→修复PR→落地release")
    g.add_argument("doc", help="source_id 或 repo#number（如 vllm-ascend#10700）")
    g = gsub.add_parser("fixes", help="PR 视角：该 PR 修复的 issues + 落地 release")
    g.add_argument("doc", help="source_id 或 repo#number（如 vllm#50241）")
    g = gsub.add_parser("sig", help="签名实体（算子/错误码/模型/版本）→ 提及它的 issue/PR")
    g.add_argument("sig", help="如 DispatchFFNCombine、561000、GLM-5.1")
    g.add_argument("--limit", type=int, default=10)
    g = gsub.add_parser("doc", help="文档邻接视图（MENTIONS 实体，调试用）")
    g.add_argument("doc", help="source_id 或 repo#number")
    g = gsub.add_parser("tags", help="标签 → 打标文档（Doc/Issue/PR）")
    g.add_argument("tag", help="标签名（如 HCCL、超时排查）")
    g = gsub.add_parser("evidence", help="文档互证（Evidence）：共享实体的其他文档")
    g.add_argument("doc", help="业务文档 source_id（如 pdf:xxx）")
    gsub.add_parser("stats", help="图统计")

    # 文档标签（能力目录 / 标签检索 / 问题匹配）
    p = sub.add_parser("tags", help="文档标签能力目录（两级分类）与标签检索")
    tsub = p.add_subparsers(dest="tags_cmd", required=True)
    tsub.add_parser("list", help="能力目录：主题/领域 + 具体作用，各标签文档数")
    g = tsub.add_parser("docs", help="按标签检索文档")
    g.add_argument("tag", help="标签名（如 HCCL、超时排查）")

    p = sub.add_parser("context", help="问题→标签匹配（文档能力发现：知识库有哪些文档能帮上这个问题）")
    p.add_argument("text", help="问题描述（如 vllm-ascend HCCL 超时）")

    args = ap.parse_args()
    base = args.base

    if args.cmd == "search":
        payload = {"query": args.query, "top_k": args.top}
        if args.version:
            payload["target_version"] = args.version
        if args.component:
            payload["component"] = args.component
        if args.comp_version:
            payload["version"] = args.comp_version
        if args.tag:
            payload["filters"] = {"tags": list(dict.fromkeys(args.tag))}
        data = _post(base, "/search", payload)
        print(fmt_search(data))
    elif args.cmd == "signature":
        payload = {"text": args.text, "top_k": args.top}
        if args.component:
            payload["component"] = args.component
        data = _post(base, "/signature-search", payload)
        print(fmt_signature(data))
    elif args.cmd == "title":
        params = {"keyword": args.keyword, "limit": args.limit, "match": args.match}
        if args.component:
            params["component"] = args.component
        data = _get(base, "/title", params)
        print(fmt_title(data))
    elif args.cmd == "version":
        data = _get(base, "/version", {"version": args.version, "repo": args.repo})
        print(fmt_version(data))
    elif args.cmd == "code":
        if args.code_file:
            if not args.version:
                print("[client] 读取文件需指定 --version")
                sys.exit(2)
            data = _get(base, "/code/file",
                        {"version": args.version, "path": args.code_file, "repo": args.repo,
                         "max_chars": args.code_max_chars})
            print(fmt_code_file(data))
        else:
            if not args.keyword:
                print("[client] code 检索需要关键词：code <关键词> [选项]；"
                      "读取文件用 code --file <路径> --version <版本>")
                sys.exit(2)
            payload = {"keyword": args.keyword, "version": args.version,
                       "limit": args.limit, "repo": args.repo,
                       "path": args.in_file, "per_version": args.per_version}
            if args.kind:
                payload["kind"] = args.kind
            data = _post(base, "/code/search", payload)
            print(fmt_code_hits(data))
    elif args.cmd == "code-versions":
        print(fmt_code_versions(_get(base, "/code/versions", {"repo": args.repo})))
    elif args.cmd == "diff":
        params = {"version1": args.v1, "version2": args.v2, "path": args.path,
                  "repo": args.repo, "context": args.context}
        if args.keyword:
            params["keyword"] = args.keyword
        print(fmt_diff(_get(base, "/code/diff", params)))
    elif args.cmd == "doc":
        data = _get(base, f"/doc/{urllib.parse.quote(args.doc_id, safe='')}")
        print(f"# {data['title']}")
        print(f"url={data['url']}  status={data['status']}  component={data['component']}")
        print(f"created={data['created_at']}  resolved={data['resolved_at']}")
        print("\n" + data["body"])
    elif args.cmd == "health":
        print(json.dumps(_get(base, "/health"), ensure_ascii=False, indent=2))
    elif args.cmd == "components":
        print(json.dumps(_get(base, "/components"), ensure_ascii=False, indent=2))
    elif args.cmd == "stats":
        print(json.dumps(_get(base, "/stats"), ensure_ascii=False, indent=2))
    elif args.cmd == "companion":
        data = _get(base, "/companion", {"component": args.component, "version": args.version})
        print(json.dumps(data, ensure_ascii=False, indent=2))
    elif args.cmd == "matrix":
        print(fmt_matrix(_get(base, "/matrix"), limit=args.limit))
    elif args.cmd == "graph":
        if args.graph_cmd == "stats":
            print(fmt_graph_stats(_get(base, "/graph/stats")))
        elif args.graph_cmd == "chain":
            doc = _resolve_graph_doc(args.doc, "issue")
            print(fmt_graph_chain(_get(base, "/graph/chain", {"doc": doc})))
        elif args.graph_cmd == "fixes":
            doc = _resolve_graph_doc(args.doc, "pr")
            print(fmt_graph_fixes(_get(base, "/graph/fixes", {"doc": doc})))
        elif args.graph_cmd == "sig":
            print(fmt_graph_sig(_get(base, "/graph/sig", {"sig": args.sig, "limit": args.limit})))
        elif args.graph_cmd == "doc":
            print(json.dumps(_get(base, "/graph/doc", {"doc": args.doc}), ensure_ascii=False, indent=2))
        elif args.graph_cmd == "tags":
            print(fmt_graph_tags(_get(base, "/graph/tags", {"tag": args.tag})))
        elif args.graph_cmd == "evidence":
            print(fmt_graph_evidence(_get(base, "/graph/evidence", {"doc": args.doc})))
    elif args.cmd == "tags":
        if args.tags_cmd == "list":
            print(fmt_tags(_get(base, "/tags")))
        elif args.tags_cmd == "docs":
            print(fmt_tags_docs(_get(base, f"/tags/{urllib.parse.quote(args.tag, safe='')}/docs")))
    elif args.cmd == "context":
        data = _post(base, "/tags/match", {"text": args.text})
        print(fmt_tags_match(data))


if __name__ == "__main__":
    try:
        main()
    except ClientError as e:
        print(f"[client] {e}")
        sys.exit(1)
    except urllib.error.URLError as e:  # 兜底（正常路径已被 _request 转成 ClientError）
        print(f"[client] 无法连接只读 API（{e.reason}）：请先运行 python scripts/serve_api.py")
        sys.exit(1)
