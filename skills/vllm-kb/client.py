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


def _get(base: str, path: str, params: dict | None = None) -> dict:
    url = base.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(base: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


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
    lines = [f"签名 '{data['signature']}'（实体类型: {data.get('entity_type') or '未命中'}）"]
    docs = data.get("docs") or []
    if not docs:
        lines.append("(图中无提及此签名的 issue/PR —— 可试小写/大小写变体，或用 search 语义检索)")
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
    p.add_argument("keyword", help="符号/关键词（如 DispatchFFNCombine、halMemCreate）")
    p.add_argument("--version", default=None, help="限定版本（默认全部已预存版本）")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--repo", default="vllm-ascend", help="仓库：vllm-ascend（默认）| vllm")
    p.add_argument("--file", dest="code_file", metavar="PATH", default=None,
                   help="直接读取指定版本源码文件（需配合 --version）")
    p.add_argument("--in-file", dest="in_file", metavar="SUBSTR", default=None,
                   help="限定 grep 的文件路径子串（如 worker/model_runner_v1.py）")
    p.add_argument("--per-version", action="store_true",
                   help="每个版本各自收集命中（输出各版本行号，对比哪个版本引入/移动了该代码）")

    p = sub.add_parser("code-versions", help="列出已预存的代码仓版本")
    p.add_argument("--repo", default="vllm-ascend", help="仓库：vllm-ascend（默认）| vllm")

    p = sub.add_parser("doc", help="读取整篇文档")
    p.add_argument("doc_id")

    sub.add_parser("health", help="健康检查")
    sub.add_parser("components", help="组件分布")
    sub.add_parser("stats", help="知识库统计")

    p = sub.add_parser("companion", help="配套反向展开")
    p.add_argument("component")
    p.add_argument("version")

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
    gsub.add_parser("stats", help="图统计")

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
                        {"version": args.version, "path": args.code_file, "repo": args.repo})
            print(fmt_code_file(data))
        else:
            data = _post(base, "/code/search",
                         {"keyword": args.keyword, "version": args.version,
                          "limit": args.limit, "repo": args.repo,
                          "path": args.in_file, "per_version": args.per_version})
            print(fmt_code_hits(data))
    elif args.cmd == "code-versions":
        print(fmt_code_versions(_get(base, "/code/versions", {"repo": args.repo})))
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


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"[client] 无法连接只读 API（{e.reason}）：请先运行 python scripts/serve_api.py")
        sys.exit(1)
