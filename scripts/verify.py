"""验收脚本：对真实故障报错查询知识库，打印排序结果与置信度分解。

用法：python scripts/verify.py [--config config.json] [--query "自定义报错"] [--version 0.6.0]
默认查询来自 config.json 的 verify.queries。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402
from vllm_kb.search import SearchEngine  # noqa: E402


def show(results, query, target_version, context=None) -> None:
    label = target_version or "(未指定，w_ver 取默认值)"
    print(f"\n===== 查询: {query}  (目标版本: {label}) =====")
    if context:
        comp = context.get("component")
        if comp:
            comps = context.get("companions") or {}
            detail = ", ".join(f"{k} {v}" for k, v in comps.items()) or "(无配套记录)"
            print(f"      组件: {comp}:{context.get('version')} | 配套反向展开 -> {detail}")
    if not results:
        print("  (无结果)")
        return
    for i, r in enumerate(results, 1):
        c = r.confidence
        tag = "已解决" if r.resolved else "未解决"
        comp = r.component or "-"
        print(f"\n[{i}] final={r.final:.3f} sim={r.similarity:.3f} conf={c.score:.3f} src={r.source} "
              f"[{tag}] 组件={comp} 版本参考={r.version_ref or '-'}")
        print(f"    w_time={c.time_weight:.3f} w_ver={c.version_weight:.3f} w_rel={c.reliability:.3f}")
        print(f"    {r.title}")
        print(f"    {r.url}  status={r.meta.get('status')} resolved={r.meta.get('resolved_at')}")
        span_min = r.meta.get("version_span_min")
        span_max = r.meta.get("version_span_max")
        span_txt = f"[{span_min or '-'}]" if span_max is None else f"[{span_min or '-'} .. {span_max}]"
        print(f"    version_span={span_txt}")
        snippet = r.text[:180].replace("\n", " ")
        print(f"    ...{snippet}...")


def main() -> None:
    # 控制台输出容错：正文含 emoji/生僻字符时（GBK 控制台）不崩溃，缺失字符用 ? 代替
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="vllm-kb 验收查询")
    ap.add_argument("--config", default=None)
    ap.add_argument("--query", action="append", default=None, help="自定义查询（可多次）")
    ap.add_argument("--version", default=None, help="目标部署版本，默认取 config.retrieval.default_target_version")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    engine = SearchEngine(cfg)
    # 生效的目标版本：命令行优先，其次 config.retrieval.default_target_version
    effective_version = args.version or cfg.retrieval.default_target_version or None
    queries = args.query or cfg.verify.queries or ["vLLM 报错"]
    for q in queries:
        results = engine.search(q, target_version=effective_version, top_k=args.top)
        show(results, q, effective_version, context=engine.last_context)
    engine.close()


if __name__ == "__main__":
    main()
