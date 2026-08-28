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
import sys

from .config import AppConfig
from .embed import EmbeddingClient
from .ingest import ingest_docs
from .models import KbDocument
from .sources import BaseSource, GithubSource, build_sources
from .vectorstore import build_vector_store


def collect_docs(cfg: AppConfig, pull: bool, limit: int | None,
                 incremental: bool = False) -> list[KbDocument]:
    """遍历生效数据源：pull（可选）+ canonicalize，合并所有来源的文档。"""
    sources = build_sources(cfg)
    all_docs: list[KbDocument] = []
    for src in sources:
        try:
            if pull:
                if isinstance(src, GithubSource):
                    n = src.pull(max_issues=limit, incremental=incremental)
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


def upsert_unified_canonical(cfg: AppConfig, docs: list[KbDocument]) -> dict:
    """把单来源文档按 source_id **upsert** 到统一 canonical（逐来源处理时用，不覆盖其他来源）。

    - source_id 不存在 → 追加（新增）；
    - 行内容与旧行相同 → 跳过（幂等，不重写）；
    - 行内容不同（如**修改过的 PDF**）→ 覆盖更新——canonical 是 --rebuild 的唯一事实源，
      修改必须回写，否则全量重建会回退旧内容。
    返回 {"added": int, "updated": int, "skipped": int}。
    """
    from .models import doc_to_json

    path = cfg.resolve(cfg.storage.canonical_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}  # source_id -> 行内容（保留原行序，新增追加末尾）
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                sid = json.loads(line).get("source_id", "")
            except Exception:
                sid = ""
            if sid:
                existing[sid] = line
    added = updated = 0
    changed: list[str] = []
    for doc in docs:
        new_line = doc_to_json(doc)
        sid = doc.source_id
        old = existing.get(sid)
        if old is None:
            existing[sid] = new_line
            added += 1
        elif old != new_line:
            existing[sid] = new_line
            updated += 1
            changed.append(sid)
        # 内容相同：跳过（幂等）
    with open(path, "w", encoding="utf-8") as f:
        for line in existing.values():
            f.write(line + "\n")
    skipped = max(0, len(docs) - added - updated)
    note = ""
    if changed:
        shown = ", ".join(changed[:5])
        note = f"（更新 {len(changed)} 条: {shown}{'…' if len(changed) > 5 else ''}）"
    print(f"[build] canonical upsert：新增 {added} / 更新 {updated} / 跳过 {skipped}"
          f"{note}（累计 {len(existing)}）-> {path}")
    return {"added": added, "updated": updated, "skipped": skipped}


def process_source(src: BaseSource, cfg: AppConfig, pull: bool, limit: int | None,
                   incremental: bool = False) -> dict:
    """处理单个来源：拉取(可选) -> canonical -> 追加统一 canonical -> 增量入库。"""
    if pull:
        if isinstance(src, GithubSource):
            n = src.pull(max_issues=limit, incremental=incremental)
        else:
            n = src.pull()
        print(f"[build] 来源 {src.id} ({src.type}) 拉取新增 {n} 条")
    docs = src.canonicalize()
    print(f"[build] 来源 {src.id} ({src.type}) canonical {len(docs)} 条")
    if not docs:
        return {"pulled": 0, "docs": 0}
    upsert_unified_canonical(cfg, docs)
    embed_client = EmbeddingClient(cfg.embedding)
    vector_store = build_vector_store(cfg)
    stats = ingest_docs(cfg, docs, embed_client, vector_store)
    stats["chunks_in_store"] = vector_store.count()
    print(f"[build] 来源 {src.id} 入库完成: {stats}")
    return stats


def run_build(cfg: AppConfig, pull: bool = True, limit: int | None = None,
              incremental: bool = False) -> dict:
    """逐来源处理（先配置在前的来源）。全量马拉松期间，先完成的来源立即可用。"""
    grand = {"pulled": 0, "ingested_docs": 0}
    for src in build_sources(cfg):
        try:
            stats = process_source(src, cfg, pull, limit, incremental)
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
    ap.add_argument("--incremental", action="store_true",
                    help="GitHub 增量拉取：已拉取完成（done）后仍从头拉取社区新增 issue/PR"
                         "（跳过已有，连续 3 页无新增停止）；默认 done 后跳过拉取")
    ap.add_argument("--rebuild", action="store_true",
                    help="高危：清空向量库与 SQLite 后全量重建（需交互确认，或加 --yes）")
    ap.add_argument("--yes", action="store_true",
                    help="跳过 --rebuild 的确认提示（自动化/无人值守用）")
    ap.add_argument("--recanonicalize", action="store_true",
                    help="跳过拉取，从原始 JSON 再生统一 canonical 并重新入库（逐来源）")
    from .net import DEFAULT_GITHUB_BASE

    ap.add_argument("--insecure", action="store_true",
                    help="跳过 SSL 证书校验（内网自签证书/SSL 被禁；亦可用环境变量 VLLM_KB_INSECURE=1）")
    ap.add_argument("--github-base", default=None,
                    help=f"GitHub API 镜像前缀（默认 {DEFAULT_GITHUB_BASE}；亦可用环境变量 VLLM_KB_GITHUB_BASE）")
    args = ap.parse_args()
    # CLI 参数 > 环境变量（与 net 统一入口语义一致）：注入环境变量，让后续构造的
    # GithubPuller（读 insecure_from_env / VLLM_KB_GITHUB_BASE）生效。
    if args.insecure:
        os.environ["VLLM_KB_INSECURE"] = "1"
        print("[build] --insecure：跳过 SSL 证书校验（内网模式）")
    if args.github_base:
        os.environ["VLLM_KB_GITHUB_BASE"] = args.github_base

    cfg = AppConfig.load(args.config)
    if args.rebuild:
        from .ingest import rebuild

        _confirm_rebuild(args.yes)
        stats = rebuild(cfg)
        print(f"[build] 全量重建完成: {stats}")
        return
    grand = run_build(cfg, pull=not (args.skip_pull or args.recanonicalize), limit=args.limit,
                      incremental=args.incremental)
    print(f"[build] 本轮汇总: {grand}")
    print("[build] 提示：中途 Ctrl-C 后重跑同一命令即断点续传；逐来源处理，先完成的来源已可用。")


def _confirm_rebuild(yes: bool) -> None:
    """--rebuild 高危确认：清空向量库 + 删除 kb.sqlite3 后全量重嵌。

    - TTY 交互：输入 y/yes 确认，否则中止；
    - 非 TTY（agent/CI/管道）：必须 --yes，否则拒绝执行——避免无人值守时
      误触发几十小时的全量重嵌入。
    """
    print(
        "[告警] --rebuild 是高危操作：将清空向量库（lancedb）并删除 kb.sqlite3，\n"
        "      然后从 canonical.jsonl 全量重新分块 + 嵌入（66K 文档约数小时）。\n"
        "      canonical/raw/图/审核库不受影响；中断后重跑仍会先清空再重建。"
    )
    if yes:
        print("[build] --yes：跳过确认，开始重建 …")
        return
    if not sys.stdin.isatty():
        print("[build] 非交互环境（agent/管道）执行 --rebuild 需要 --yes，已中止。")
        sys.exit(3)
    try:
        ans = input("确认清空并全量重建？输入 y 继续，其余中止: ").strip().lower()
    except EOFError:
        ans = ""
    if ans not in ("y", "yes"):
        print("[build] 已取消（未确认）。")
        sys.exit(3)


if __name__ == "__main__":
    main()
