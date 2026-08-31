"""只再生 canonical（不入库、不拉取）：遍历生效来源 canonicalize() → 追加/更新统一 canonical.jsonl。

与 build_kb.py 的关系（分工）：
- `scripts/build_kb.py`：数据流水线**全路径**（拉取 → canonical → 入库），`--skip-pull` 半路径；
- 本脚本：**只处理 canonical 中间产物**（复用 pipeline.upsert_unified_canonical / sources.canonicalize），
  不触碰 kb.sqlite3 / 向量库——提取逻辑（版本/kind/组件/标签规则）升级后只需更新 canonical，
  再跑 build_graph.py 用新 canonical 建图即可，无需重嵌向量。

用法：
    python scripts/build_canonical.py                 # 逐来源再生并 upsert canonical（幂等）
    python scripts/build_canonical.py --config path   # 指定 config.json

后续（按需）：
    python scripts/build_graph.py                     # canonical → Kùzu 图（先停 serve_api）
    python scripts/build_kb.py --skip-pull            # 若同时要把变更重入库（重嵌向量）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402
from vllm_kb.pipeline import upsert_unified_canonical  # noqa: E402
from vllm_kb.sources import build_sources  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="只再生统一 canonical（不入库）：提取逻辑升级后更新 canonical 供建图/重建")
    ap.add_argument("--config", default=None, help="config.json 路径（默认项目根）")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config, require_keys=False)
    grand_added = grand_updated = 0
    for src in build_sources(cfg):
        try:
            docs = src.canonicalize()
        except NotImplementedError as e:
            print(f"[warn] 来源 {src.id} ({src.type}) 未实现，跳过：{e}")
            continue
        stats = upsert_unified_canonical(cfg, docs)
        grand_added += stats["added"]
        grand_updated += stats["updated"]
        print(f"[canonical] 来源 {src.id} ({src.type}) canonical {len(docs)} 条"
              f"：新增 {stats['added']} / 更新 {stats['updated']} / 跳过 {stats['skipped']}")
    print(f"[canonical] 完成：累计新增 {grand_added} / 更新 {grand_updated} 条 -> "
          f"{cfg.resolve(cfg.storage.canonical_file)}")
    print("[canonical] 后续：python scripts/build_graph.py 用新 canonical 建图（先停 serve_api）；"
          "如需重入库（重嵌向量）再跑 build_kb.py --skip-pull")


if __name__ == "__main__":
    main()
