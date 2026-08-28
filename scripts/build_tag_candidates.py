"""正文 TF-IDF 标签候选导出（jieba）——**不自动写入 config**，输出到文件供人工审阅后手动同步。

用法：
    python scripts/build_tag_candidates.py                 # 输出到 data/tag_candidates_manual.json
    python scripts/build_tag_candidates.py --top 10        # 每篇文档候选数（默认 15）
    python scripts/build_tag_candidates.py --min-count 2   # 候选至少被 N 篇文档提及才保留（默认 1）
    python scripts/build_tag_candidates.py --out path      # 自定义输出路径
    python scripts/build_tag_candidates.py --config path   # 指定 config.json

背景：自动标签当前只从文件名 + 内部标题提取（词典 `config.tags.registry` 驱动）；
正文的高频主题词（如"拓扑"、"固件"、"带宽"）未纳入候选。本脚本用 jieba TF-IDF
从文档正文提取关键词，与现有标签/词典对比后输出**未收录候选**——人工审阅文件后，
把采纳的词手动写入 config.json 的 `tags.registry`（不自动同步、不自动打标，
保证词典由人把关；写入后下次入库/建图自动生效）。

输出格式（JSON，ensure_ascii=False）：
[
  {"doc_id": "pdf:xxx", "title": "...", "candidates": [
      {"name": "拓扑", "score": 0.18, "tier": "domain"},
      {"name": "固件升级", "score": 0.11, "tier": "purpose"}
  ]}
]

依赖：jieba（未安装时提示安装并退出）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig
from vllm_kb.fts_tokenizer import register_words
from vllm_kb.tagging import TagRegistry, tier_for


def _load_canonical(cfg: AppConfig) -> list[dict]:
    """读统一 canonical，返回有正文的文档（doc_* 与 github 均可，正文 TF-IDF 提取标签候选）。"""
    p = cfg.resolve(cfg.storage.canonical_file)
    if not p.exists():
        print(f"[tag-candidates] canonical 不存在: {p}（先运行 scripts/build_kb.py）")
        sys.exit(1)
    docs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (d.get("body") or "").strip():
            docs.append(d)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="正文 TF-IDF 标签候选导出（jieba；输出到文件，人工写入 config.tags.registry）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--top", type=int, default=15, help="每篇文档候选数（默认 15）")
    ap.add_argument("--min-count", type=int, default=1, help="候选至少被 N 篇文档提及才保留（默认 1）")
    ap.add_argument("--out", default=None, help="输出路径（默认 data/tag_candidates_manual.json）")
    ap.add_argument("--include-github", action="store_true",
                    help="也处理 github issue/PR（默认只处理业务文档 doc_*）")
    args = ap.parse_args()

    try:
        import jieba.analyse  # noqa: F401
    except ImportError:
        print("[tag-candidates] 需要 jieba：pip install jieba（离线 wheel 可装）")
        sys.exit(1)

    cfg = AppConfig.load(args.config, require_keys=False)
    registry = TagRegistry.load(cfg)
    register_words([e.name for e in registry.entries])  # 标签词防拆分，稳定 TF-IDF 候选
    stopwords = {s.lower() for s in (cfg.tags.stopwords or [])}
    docs = _load_canonical(cfg)
    if not args.include_github:
        docs = [d for d in docs if str(d.get("source_type", "")).startswith("doc_")]
    if not docs:
        print("[tag-candidates] canonical 无业务文档（doc_*）；加 --include-github 处理全部")
        return

    out_path = Path(args.out) if args.out else cfg.resolve("data/tag_candidates_manual.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agg: dict[str, int] = {}  # 候选词 -> 提及文档数
    rows: list[dict] = []
    for d in docs:
        body = d.get("body") or ""
        title = d.get("title") or ""
        doc_tags = set(d.get("tags") or [])
        cands = []
        for word, weight in jieba.analyse.extract_tags(body, topK=args.top, withWeight=True):
            name = word.strip()
            if not name or len(name) < 2 or len(name) > 20:
                continue
            if name.lower() in stopwords or name.isdigit():
                continue
            if registry.contains(name) or name in doc_tags:
                continue  # 已收录/已打标：不重复推荐
            cands.append({"name": name, "score": round(float(weight), 4),
                          "tier": tier_for(name)})
            agg[name] = agg.get(name, 0) + 1
        if cands:
            rows.append({"doc_id": d.get("source_id", ""), "title": title,
                         "candidates": cands})
    # 按 min-count 过滤（跨文档频率）
    if args.min_count > 1:
        for r in rows:
            r["candidates"] = [c for c in r["candidates"] if agg.get(c["name"], 0) >= args.min_count]
        rows = [r for r in rows if r["candidates"]]

    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(len(r["candidates"]) for r in rows)
    print(f"[tag-candidates] {len(docs)} 篇文档 → {len(rows)} 篇含候选，共 {total} 个未收录候选")
    print(f"[tag-candidates] 输出: {out_path}")
    print("[tag-candidates] 人工审阅后，把采纳的词写入 config.json 的 tags.registry"
          "（不自动同步；写入后下次入库/建图自动生效）")


if __name__ == "__main__":
    main()
