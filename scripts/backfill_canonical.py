"""从 kb.sqlite3 回填 canonical.jsonl 缺失文档（修复 kb↔canonical 不同步）。

背景：canonical.jsonl 是 --rebuild 与 build_graph 的唯一事实源。若 kb.sqlite3 含有
canonical 缺失的文档（历史旧版流程 / --recanonicalize 时 raw 不全等导致漂移），
图（从 canonical 建）与全量重建（从 canonical 重嵌）都会丢这些文档——
典型症状：title/search 能命中（kb 有），graph chain/fixes 查不到（图里没有）。

本脚本用 kb.sqlite3（docs + chunks_meta/chunks_fts + doc_tags）重建缺失的 canonical 行：
- body：chunks_meta → chunks_fts 按 seq 拼回（与 /doc 端点同法；chunk 间 overlap 有少量重复，
  对图构建的关系提取（FIXES/MENTIONS）无影响）；
- tags：优先 doc_tags.auto_snapshot（canonical 语义=入库时自动标签），无则回退 docs.tags；
- extra 原样带出（图构建依赖 extra.repo/github_number/merged_at 建 FIXES/MERGED_IN 边）；
- 已知限制：component_versions、updated_at 在 docs 表未存，回填行该字段为空（影响极小）。

用法：
    python scripts/backfill_canonical.py                  # dry-run：只打印差集清单，不写
    python scripts/backfill_canonical.py --write          # 把缺失文档回填 canonical.jsonl（幂等）
    python scripts/backfill_canonical.py --doc github:vllm-project-vllm-ascend:pr:9749 --write
    python scripts/backfill_canonical.py --config path    # 指定 config.json

回填后：
    1. python scripts/build_kb.py --skip-pull   # 回填文档重新入库（body 为 chunks 拼回，
                                                  embed_hash 变化会重嵌，幂等可重跑）
    2. 停 serve_api（Kùzu 单写者）→ python scripts/build_graph.py → 重启 serve_api
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402
from vllm_kb.models import KbDocument, VersionSpan, doc_to_json  # noqa: E402


def _read_canonical_index(path: Path) -> dict[str, str]:
    """canonical source_id -> 原始行（保留原行序与内容）。损坏行不参与索引。"""
    idx: dict[str, str] = {}
    if not path.exists():
        return idx
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            sid = json.loads(line).get("source_id", "")
        except json.JSONDecodeError:
            sid = ""
        if sid:
            idx[sid] = line
    return idx


def _reconstruct_body(conn: sqlite3.Connection, source_id: str) -> str:
    """从 chunks_meta + chunks_fts 按 seq 拼回正文（与 api.py /doc 端点同法）。"""
    rows = conn.execute(
        "SELECT f.text FROM chunks_meta m JOIN chunks_fts f ON f.chunk_id = m.chunk_id "
        "WHERE m.doc_id = ? ORDER BY m.seq",
        (source_id,),
    ).fetchall()
    return "\n\n".join(r[0] for r in rows if r[0])


def _reconstruct_doc(conn: sqlite3.Connection, source_id: str) -> KbDocument | None:
    row = conn.execute(
        "SELECT source_type, url, title, created_at, resolved_at, status, labels, "
        "version_span_min, version_span_max, reliability, component, extra, tags "
        "FROM docs WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        return None
    (st, url, title, created_at, resolved_at, status, labels_json,
     vs_min, vs_max, reliability, component, extra_json, tags_json) = row

    # canonical.tags 语义 = 入库时自动提取的标签：优先 doc_tags.auto_snapshot
    trow = conn.execute("SELECT auto_snapshot FROM doc_tags WHERE source_id = ?",
                        (source_id,)).fetchone()
    auto_tags: list[str] = []
    if trow and trow[0]:
        try:
            auto_tags = json.loads(trow[0])
        except (TypeError, json.JSONDecodeError):
            auto_tags = []
    if not auto_tags:
        try:
            auto_tags = json.loads(tags_json or "[]")
        except (TypeError, json.JSONDecodeError):
            auto_tags = []
    try:
        labels = json.loads(labels_json or "[]")
    except (TypeError, json.JSONDecodeError):
        labels = []
    try:
        extra = json.loads(extra_json or "{}")
    except (TypeError, json.JSONDecodeError):
        extra = {}

    return KbDocument(
        source_type=st or "",
        source_id=source_id,
        url=url or "",
        title=title or "",
        body=_reconstruct_body(conn, source_id),
        created_at=created_at,
        resolved_at=resolved_at,
        status=status or "open",
        labels=labels,
        tags=auto_tags,
        version_span=VersionSpan(min=vs_min, max=vs_max),
        component=component or "",
        reliability=reliability,
        extra=extra,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="从 kb.sqlite3 回填 canonical.jsonl 缺失文档")
    ap.add_argument("--config", default=None, help="config.json 路径（默认项目根）")
    ap.add_argument("--write", action="store_true",
                    help="写回 canonical.jsonl（默认 dry-run 只打印差集清单）")
    ap.add_argument("--doc", action="append", default=None, metavar="SOURCE_ID",
                    help="只处理指定 source_id（可多次）；缺省 = 全部缺失")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config, require_keys=False)
    kb_path = cfg.resolve(cfg.storage.sqlite_path)
    canon_path = cfg.resolve(cfg.storage.canonical_file)
    if not kb_path.exists():
        print(f"[backfill] kb.sqlite3 不存在: {kb_path}")
        sys.exit(1)

    conn = sqlite3.connect(f"file:{kb_path.as_posix()}?mode=ro", uri=True)
    try:
        kb_ids = {r[0] for r in conn.execute("SELECT source_id FROM docs")}
    finally:
        conn.close()
    canon_idx = _read_canonical_index(canon_path)

    missing = sorted(kb_ids - set(canon_idx))
    if args.doc:
        want = set(args.doc)
        missing = [s for s in missing if s in want]
        absent = [s for s in want if s not in kb_ids]
        already = sorted(s for s in want if s in canon_idx)
        if absent:
            print(f"[backfill] --doc 指定但 kb 中不存在: {absent}")
        if already:
            print(f"[backfill] --doc 指定但 canonical 已有（无需回填）: {already}")

    print(f"[backfill] kb docs {len(kb_ids)} / canonical {len(canon_idx)} / 缺失 {len(missing)}")
    if not missing:
        print("[backfill] 无缺失：kb 与 canonical 一致。")
        return

    for sid in missing:
        conn = sqlite3.connect(f"file:{kb_path.as_posix()}?mode=ro", uri=True)
        try:
            doc = _reconstruct_doc(conn, sid)
        finally:
            conn.close()
        if doc is None:
            print(f"  - {sid}  (docs 行缺失，跳过)")
            continue
        print(f"  - {sid}  [{doc.source_type}] {doc.title[:60]}  body={len(doc.body)} 字符")

    if not args.write:
        print("[backfill] dry-run：以上为缺失清单。加 --write 回填 canonical.jsonl。")
        return

    added = 0
    conn = sqlite3.connect(f"file:{kb_path.as_posix()}?mode=ro", uri=True)
    try:
        for sid in missing:
            doc = _reconstruct_doc(conn, sid)
            if doc is None:
                continue
            canon_idx[sid] = doc_to_json(doc)
            added += 1
    finally:
        conn.close()
    canon_path.parent.mkdir(parents=True, exist_ok=True)
    with open(canon_path, "w", encoding="utf-8") as f:
        for line in canon_idx.values():
            f.write(line + "\n")
    print(f"[backfill] 已回填 {added} 条 -> {canon_path}")
    print("[backfill] 后续：python scripts/build_kb.py --skip-pull（重入库，body 拼回会触发重嵌）→ "
          "停 serve_api 后 python scripts/build_graph.py 重建图 → 重启 serve_api")


if __name__ == "__main__":
    main()
