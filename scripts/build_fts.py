"""重建 FTS5 全文索引（jieba 中文分词版）——**不重嵌向量、不动 docs/审核数据**。

用法：
    python scripts/build_fts.py                 # 全量重建（从向量库取权威原文重新分词）
    python scripts/build_fts.py --limit 1000    # 试跑：只处理前 N 个 chunk
    python scripts/build_fts.py --config path   # 指定 config.json

原理（**2026-10 业务实测修正**：重建源是向量库，不是 chunks_fts 自身）：
    1. 权威 chunk 清单 = `chunks_meta`（与向量库同源写入，FTS 缺行时它仍是全集）；
    2. 权威原文 = 向量库（LanceDB `texts_by_id()`，每条 chunk 的 text 完整拷贝）——
       旧实现从 `chunks_fts` 自身读原文，**缺行的表重建后仍缺行**，历史缺口永远找不回
       （业务库曾 meta 150202 行 / fts 29078 行，重建后依旧 29078）；
    3. 对原文 jieba 分词后 DROP 重建 chunks_fts（新 schema：indexed_text 索引列 +
       text 原文列），按相同 chunk_id 写回——chunk_id 与向量库严格一致，
       **不重新分块**，向量/嵌入结果不受影响。

适用：FTS 缺行回填（历史 DROP 后未回填的存量库，跑一次即全量找回）；升级 jieba 或
分词规则后重建。普通增量入库时新文档自动分词写入，无需每次运行本脚本。
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig
from vllm_kb.fts_tokenizer import register_words, tokenize_text
from vllm_kb.tagging import TagRegistry
from vllm_kb.vectorstore import build_vector_store

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

    # 权威原文来自向量库（只读用途，不写）
    store = build_vector_store(cfg)

    conn = sqlite3.connect(str(kb_path))
    try:
        # 1) 权威 chunk 清单：chunks_meta 与向量库同源写入——FTS 缺行时它仍是全集
        meta_rows = conn.execute(
            "SELECT chunk_id, doc_id FROM chunks_meta ORDER BY doc_id, seq").fetchall()
        if not meta_rows:
            print("[fts] chunks_meta 为空——先运行 scripts/build_kb.py 入库")
            return
        # 2) 向量库原文映射（不加载向量列）
        texts = store.texts_by_id()
        print(f"[fts] chunks_meta {len(meta_rows)} chunk / 向量库 {len(texts)} chunk，"
              "取原文分词重建 …")
        tokenized: list[tuple[str, str, str, str]] = []
        missing: list[str] = []
        start_ts = time.time()
        step = max(1, len(meta_rows) // 100)
        for i, (chunk_id, doc_id) in enumerate(meta_rows, 1):
            text = texts.get(chunk_id)
            if text is None:
                missing.append(chunk_id)   # 向量库也没有 → 两层都缺（不应发生，如实暴露）
                continue
            tokenized.append((chunk_id, doc_id, tokenize_text(text), text))
            if i % step == 0:
                rate = i / max(0.001, time.time() - start_ts)
                print(f"[fts] 分词 {i}/{len(meta_rows)} ({100.0 * i / len(meta_rows):.0f}%) "
                      f"| {rate:.0f} chunk/s | ETA {(len(meta_rows) - i) / rate / 60:.1f}min",
                      flush=True)
        if missing:
            print(f"[fts] ⚠️ {len(missing)} 个 chunk 在向量库中无原文（示例: "
                  f"{missing[:3]}）——将按现有数据重建；若占比异常请检查向量库完整性")
        # 3) 显式事务内重建（DROP→CREATE→INSERT 原子：中途失败不破坏现有索引）
        print(f"[fts] 分词完成（{time.time() - start_ts:.0f}s），重建 chunks_fts …")
        done = tokenized[:args.limit] if args.limit else tokenized
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.execute(_FTS_DDL)
        conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, doc_id, indexed_text, text) VALUES (?,?,?,?)",
            done,
        )
        conn.commit()
        print(f"[fts] 重建完成：{len(done)}/{len(meta_rows)} chunk"
              f"（{time.time() - start_ts:.0f}s）。"
              "向量库未改动——中文词（如'超时'）现在可命中含'超时排查'的文本。")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
