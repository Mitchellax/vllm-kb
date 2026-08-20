"""采集入库流水线：多来源 -> 统一 canonical -> 增量入库（逐来源处理）。

设计要点：
- 数据源可配置（github / markdown / excel ...），见 config.json 的 sources 与 vllm_kb/sources.py；
- **逐来源处理**：每个来源 拉取 -> canonical -> 追加统一 canonical -> 入库 完成后才处理下一个，
  这样全量马拉松（vllm 数万条、数十小时）期间，先完成的来源（vllm-ascend）立即可用；
- 每个来源的原始数据独立分目录存储（data/raw/{source_id}/...）；canonical 统一单文件；
- 拉取只落原始数据 + checkpoint（断点续传）；canonical 可再生成（换逻辑无需重拉）；
- 入库双哈希增量（见 ingest）：内容未变跳过、仅元数据变化不重嵌。

用法（在项目根）：
    python -m vllm_kb.pipeline                # 全流程（拉取 + 再生 + 增量入库）
    python -m vllm_kb.pipeline --skip-pull    # 不拉取，只用现有原始数据
    python -m vllm_kb.pipeline --limit 100    # 本次拉取条数上限（作用于 github 来源）
    python -m vllm_kb.pipeline --recanonicalize  # 只再生 canonical 并重新入库
    python -m vllm_kb.pipeline --rebuild      # 清库后全量重建（换 embedding 模型时）
"""
from __future__ import annotations

import argparse
import json

from .config import AppConfig
from .embed import EmbeddingClient
from .ingest import ingest_docs
from .models import KbDocument
from .sources import BaseSource, GithubSource, build_sources
from .vectorstore import build_vector_store


def collect_docs(cfg: AppConfig, pull: bool, limit: int | None) -> list[KbDocument]:
    """遍历生效数据源：pull（可选）+ canonicalize，合并所有来源的文档。"""
    sources = build_sources(cfg)
    all_docs: list[KbDocument] = []
    for src in sources:
        try:
            if pull:
                if isinstance(src, GithubSource):
                    n = src.pull(max_issues=limit)
                else:
                    n = src.pull()
                print(f"[build] 来源 {src.id} ({src.type}) 拉取新增 {n} 条")
            docs = src.canonicalize()
            print(f"[build] 来源 {src.id} ({src.type}) canonical {len(docs)} 条")
            all_docs.extend(docs)
        except NotImplementedError as e:
            print(f"[warn] 来源 {src.id} ({src.type}) 未实现，跳过：{e}")
    return all_docs


def write_unified_canonical(cfg: AppConfig, docs: list[KbDocument]) -> int:
    """把多来源文档合并写入统一 canonical 单文件（按 source_id 去重，覆盖写）。"""
    from .models import doc_to_json

    path = cfg.resolve(cfg.storage.canonical_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            if doc.source_id in seen:
                continue
            seen.add(doc.source_id)
            f.write(doc_to_json(doc) + "\n")
    print(f"[build] 统一 canonical {len(seen)} 条 -> {path}")
    return len(seen)


def append_unified_canonical(cfg: AppConfig, docs: list[KbDocument]) -> int:
    """把单来源文档去重追加到统一 canonical（逐来源处理时用，不覆盖已有来源）。"""
    from .models import doc_to_json

    path = cfg.resolve(cfg.storage.canonical_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line).get("source_id", ""))
            except Exception:
                pass
    added = 0
    with open(path, "a", encoding="utf-8") as f:
        for doc in docs:
            if doc.source_id in seen:
                continue
            seen.add(doc.source_id)
            f.write(doc_to_json(doc) + "\n")
            added += 1
    print(f"[build] canonical 追加 {added} 条（累计 {len(seen)}）-> {path}")
    return added


def process_source(src: BaseSource, cfg: AppConfig, pull: bool, limit: int | None) -> dict:
    """处理单个来源：拉取(可选) -> canonical -> 追加统一 canonical -> 增量入库。"""
    if pull:
        if isinstance(src, GithubSource):
            n = src.pull(max_issues=limit)
        else:
            n = src.pull()
        print(f"[build] 来源 {src.id} ({src.type}) 拉取新增 {n} 条")
    docs = src.canonicalize()
    print(f"[build] 来源 {src.id} ({src.type}) canonical {len(docs)} 条")
    if not docs:
        return {"pulled": 0, "docs": 0}
    append_unified_canonical(cfg, docs)
    embed_client = EmbeddingClient(cfg.embedding)
    vector_store = build_vector_store(cfg)
    stats = ingest_docs(cfg, docs, embed_client, vector_store)
    stats["chunks_in_store"] = vector_store.count()
    print(f"[build] 来源 {src.id} 入库完成: {stats}")
    return stats


def run_build(cfg: AppConfig, pull: bool = True, limit: int | None = None) -> dict:
    """逐来源处理（先配置在前的来源）。全量马拉松期间，先完成的来源立即可用。"""
    grand = {"pulled": 0, "ingested_docs": 0}
    for src in build_sources(cfg):
        try:
            stats = process_source(src, cfg, pull, limit)
            grand["pulled"] += stats.get("pulled", 0)
            grand["ingested_docs"] += stats.get("docs", 0)
        except NotImplementedError as e:
            print(f"[warn] 来源 {src.id} ({src.type}) 未实现，跳过：{e}")
    return grand


def main() -> None:
    ap = argparse.ArgumentParser(description="vllm-kb 采集入库流水线（多数据源，逐来源处理）")
    ap.add_argument("--config", default=None, help="config.json 路径（默认项目根 config.json）")
    ap.add_argument("--skip-pull", action="store_true", help="跳过拉取，用现有原始数据再生并入库")
    ap.add_argument("--limit", type=int, default=None, help="本次拉取条数上限（覆盖 github 来源配置）")
    ap.add_argument("--rebuild", action="store_true", help="清空向量库与 SQLite 后全量重建")
    ap.add_argument("--recanonicalize", action="store_true",
                    help="跳过拉取，从原始 JSON 再生统一 canonical 并重新入库（逐来源）")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    if args.rebuild:
        from .ingest import rebuild

        stats = rebuild(cfg)
        print(f"[build] 全量重建完成: {stats}")
        return
    grand = run_build(cfg, pull=not (args.skip_pull or args.recanonicalize), limit=args.limit)
    print(f"[build] 本轮汇总: {grand}")
    print("[build] 提示：中途 Ctrl-C 后重跑同一命令即断点续传；逐来源处理，先完成的来源已可用。")


if __name__ == "__main__":
    main()
