"""Phase 2 图构建入口（Kùzu）。

用法：
    python scripts/build_graph.py                 # 全量重建图（DROP + 重建 + 统计）
    python scripts/build_graph.py --limit 1000    # 试跑：只处理前 N 条 canonical
    python scripts/build_graph.py --stats         # 只打印现有图统计（不重建）
    python scripts/build_graph.py --config path   # 指定 config.json
    python scripts/build_graph.py --graph-dir X   # 覆盖图目录（默认 config.storage.graph_path）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig
from vllm_kb.graph import GraphBuilder, default_graph_path
from vllm_kb.tagging import TagRegistry


def main() -> None:
    ap = argparse.ArgumentParser(description="vllm-kb 图存储构建（Kùzu）")
    ap.add_argument("--config", default=None, help="config.json 路径（默认项目根）")
    ap.add_argument("--graph-dir", default=None, help="图目录（默认 config.storage.graph_path = data/graph）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条 canonical（试跑）")
    ap.add_argument("--stats", action="store_true", help="只打印现有图统计，不重建")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config, require_keys=False)
    graph_dir = Path(args.graph_dir) if args.graph_dir else default_graph_path(cfg)

    builder = GraphBuilder(graph_dir)
    try:
        if args.stats:
            s = builder.stats()
            print(s.summary())
            return
        print(f"[graph] 重建图（{graph_dir}）…")
        builder.create_schema(drop_existing=True)
        canonical = cfg.resolve(cfg.storage.canonical_file)
        # parsed 目录用于提取文档表格（错误码表 → ErrorCode 节点 + DOCUMENTS 边）
        parsed_root = cfg.resolve("data/parsed")
        # kb.sqlite3 提供人工标签覆盖层（doc_tags）；registry 提供词典全量 Tag 节点 + 标题标签提取
        kb_path = cfg.resolve(cfg.storage.sqlite_path)
        registry = TagRegistry.load(cfg)
        s = builder.build_from_canonical(canonical, limit=args.limit, parsed_root=parsed_root,
                                         kb_path=kb_path, registry=registry)
        print("[graph] 构建完成：")
        print(s.summary())
    finally:
        builder.close()


if __name__ == "__main__":
    main()
