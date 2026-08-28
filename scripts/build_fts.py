"""重建 FTS5 全文索引（jieba 中文分词版）——**不重嵌向量、不动 docs/审核数据**。

用法：
    python scripts/build_fts.py                 # 全量重建（读现有 chunk 原文重新分词）
    python scripts/build_fts.py --limit 1000    # 试跑：只处理前 N 个 chunk
    python scripts/build_fts.py --config path   # 指定 config.json

原理：读现有 kb.sqlite3 `chunks_fts` 的 (chunk_id, doc_id, text 原文)——chunk_id 与
向量库严格一致（**不重新分块**，向量/嵌入结果不受影响）——对原文 jieba 分词后
DROP 重建 chunks_fts（新 schema：indexed_text 索引列 + text 原文列），按相同 chunk_id 写回。

适用：升级 jieba 或分词规则后重建全文索引；旧库（无 indexed_text 列）升级后的
首次重建。普通增量入库时新文档自动分词写入，无需每次运行本脚本。
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig
from vllm_kb.fts_tokenizer import register_words, tokenize_text
from vllm_kb.tagging import TagRegistry

_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED, doc_id UNINDEXED,
  indexed_text,
  text UNINDEXED
);
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="vllm-kb FTS5 全文索引重建（jieba 中文分词，不重嵌向量）")
    ap.add_argument("--config", default=None, help="config.json 路径（默认项目根）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 chunk（试跑）")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config, require_keys=False)
    kb_path = cfg.resolve(cfg.storage.sqlite_path)
    if not kb_path.exists():
        print(f"[fts] kb.sqlite3 不存在: {kb_path}（先运行 scripts/build_kb.py 入库）")
        sys.exit(1)
    # 标签词典词注册进 jieba（复合标签不拆散；jieba 未装时降级原文，本脚本仍幂等）
    register_words([e.name for e in TagRegistry.load(cfg).entries])

    conn = sqlite3.connect(str(kb_path))
    try:
        # 1) 读现有 chunk 原文（旧/新 schema 的 text 列都是原文）
        try:
            rows = conn.execute("SELECT chunk_id, doc_id, text FROM chunks_fts").fetchall()
        except sqlite3.OperationalError:
            print("[fts] chunks_fts 表不存在，无可重建内容")
            return
        total = len(rows)
        print(f"[fts] 读取 {total} 个 chunk 原文，分词重建 …")
        # 2) 分词（进度节流）
        tokenized: list[tuple[str, str, str, str]] = []
        start_ts = time.time()
        step = max(1, total // 100)
        for i, (chunk_id, doc_id, text) in enumerate(rows, 1):
            tokenized.append((chunk_id, doc_id, tokenize_text(text or ""), text or ""))
            if i % step == 0:
                rate = i / max(0.001, time.time() - start_ts)
                print(f"[fts] 分词 {i}/{total} ({100.0 * i / total:.0f}%) "
                      f"| {rate:.0f} chunk/s | ETA {(total - i) / rate / 60:.1f}min",
                      flush=True)
        # 3) 重建表并写回（同 chunk_id，与向量库严格一致）
        print(f"[fts] 分词完成（{time.time() - start_ts:.0f}s），重建 chunks_fts …")
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.executescript(_FTS_DDL)
        conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, doc_id, indexed_text, text) VALUES (?,?,?,?)",
            tokenized[:args.limit] if args.limit else tokenized,
        )
        conn.commit()
        done = min(args.limit, total) if args.limit else total
        print(f"[fts] 重建完成：{done}/{total} chunk（{time.time() - start_ts:.0f}s）。"
              "向量库未改动——中文词（如'超时'）现在可命中含'超时排查'的文本。")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
