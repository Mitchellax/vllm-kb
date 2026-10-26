"""旧行级 excel 文档 → 文件级迁移工具。

背景（2026-10 业务反馈"表格的文档打标还是以行为单位"）：ExcelSource 重构后一个
文件一篇文档（`source_id = excel:<文件名>`，与 md/word/pdf 同构，打标/认证/删除
都是文件粒度）；**旧版构建产物是逐行一篇**（`excel:<文件名>:<Sheet>:<行>`）——文档
打标/认证只能逐行做。本工具一次性迁移：

  1) --plan        干跑：列出旧行级文档（按文件分组），收集每组人工标签/excluded
                   并集，存 JSON 存档（不修改任何数据）
  2) --delete      从 kb/向量库删除行级文档并登记墓碑（构建不再复活；含关联候选清理）
  3) （用户）重跑构建（build_kb.py 或审核台启动）——新格式 excel:<文件名> 一篇/文件
  4) --apply-tags  构建完成后：把存档的人工标签/excluded 写到新文档（doc_tags 覆盖层
                   + docs.tags final 同步；词典命中的 auto 标签构建时自然重现，不迁移）

用法（业务环境，按顺序）：
  python scripts/migrate_excel_rowdocs.py --plan
  python scripts/migrate_excel_rowdocs.py --delete
  python build_kb.py            # 或重启审核台触发增量构建
  python scripts/migrate_excel_rowdocs.py --apply-tags
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_kb.config import AppConfig            # noqa: E402
from vllm_kb.review import (                    # noqa: E402
    collect_rowdoc_tags,
    default_review_path,
    delete_external_doc,
    find_excel_rowdocs,
    prune_tag_candidates_for_doc,
    record_tombstones,
    set_doc_tags_conn,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json", help="配置文件路径（默认 config.json）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="干跑：列出行级文档+标签收集存档")
    g.add_argument("--delete", action="store_true", help="删除行级文档+登记墓碑")
    g.add_argument("--apply-tags", action="store_true", help="构建后写回人工标签")
    ap.add_argument("--reviewer", default="excel-migration", help="操作人（墓碑/覆盖层审计）")
    ap.add_argument("--force", action="store_true", help="delete 跳过确认")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config, require_keys=False)
    kb_path = cfg.resolve(cfg.storage.sqlite_path)
    if not kb_path.exists():
        print(f"[migrate] kb.sqlite3 不存在：{kb_path}")
        return 1
    review_db = default_review_path(cfg)
    import sqlite3

    conn = sqlite3.connect(str(kb_path))
    try:
        rowdocs = find_excel_rowdocs(conn)
        groups = collect_rowdoc_tags(conn)
    finally:
        conn.close()

    if not rowdocs:
        print("[migrate] 未发现旧行级 excel 文档——无需迁移（已是文件级格式）")
        return 0
    files = {d["source_id"].split(":", 2)[1] for d in rowdocs}
    print(f"[migrate] 旧行级文档 {len(rowdocs)} 行，归属 {len(files)} 个文件：")
    for f in sorted(files):
        n = sum(1 for d in rowdocs if d["source_id"].split(":", 2)[1] == f)
        tags = groups.get(f"excel:{f}") or {}
        manual = tags.get("manual") or []
        excluded = tags.get("excluded") or []
        extra = f"（人工标签 {len(manual)}：{','.join(manual[:5])}{'…' if len(manual) > 5 else ''}"
        extra += f"；排除 {len(excluded)}）" if excluded else "）"
        print(f"  excel:{f}  —— {n} 行{extra}")

    if args.plan:
        out = Path(args.config).resolve().parent / "migrate_excel_rowtags.json"
        out.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[migrate] 干跑完成，标签存档已写 {out}（--delete 不改数据；--apply-tags 构建后用）")
        return 0

    if args.delete:
        if not args.force:
            ans = input(f"[migrate] 将删除 {len(rowdocs)} 行级文档并登记墓碑（不可逆，"
                        f"重跑构建后按文件级重建）。确认输入 yes：")
            if ans.strip().lower() != "yes":
                print("[migrate] 已取消")
                return 1
        # 先把标签存档落盘（删了就没有了）
        out = Path(args.config).resolve().parent / "migrate_excel_rowtags.json"
        out.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            from vllm_kb.vectorstore import build_vector_store

            try:
                vector_store = build_vector_store(cfg)
            except Exception as e:
                print(f"[migrate] 向量库不可用（仅删 SQLite，向量幽灵由墓碑屏蔽）: {e}")
                vector_store = None
        except ImportError:
            vector_store = None
        n = 0
        for d in rowdocs:
            try:
                delete_external_doc(kb_path, d["source_id"], vector_store=vector_store)
                prune_tag_candidates_for_doc(review_db, d["source_id"])
                n += 1
            except ValueError as e:  # 已不存在
                print(f"[migrate] 跳过 {d['source_id']}: {e}")
        record_tombstones(review_db, [d["source_id"] for d in rowdocs], reviewer=args.reviewer)
        print(f"[migrate] 已删除 {n} 行级文档，墓碑 {len(rowdocs)} 条，标签存档 {out}。"
              f"下一步：重跑构建（build_kb.py 或重启审核台），完成后 --apply-tags")
        return 0

    if args.apply_tags:
        archive = Path(args.config).resolve().parent / "migrate_excel_rowtags.json"
        if not archive.exists():
            print(f"[migrate] 找不到标签存档 {archive}（先跑 --plan/--delete 生成）")
            return 1
        groups = json.loads(archive.read_text(encoding="utf-8"))
        conn = sqlite3.connect(str(kb_path))
        applied = missing = 0
        try:
            for file_key, tags in groups.items():
                row = conn.execute("SELECT 1 FROM docs WHERE source_id=?", (file_key,)).fetchone()
                if row is None:
                    missing += 1
                    print(f"[migrate] 新格式文档不存在：{file_key}（先重跑构建）")
                    continue
                set_doc_tags_conn(conn, file_key,
                                  excluded=tags.get("excluded") or None,
                                  manual=tags.get("manual") or None,
                                  reviewer=args.reviewer)
                applied += 1
            conn.commit()
        finally:
            conn.close()
        print(f"[migrate] 人工标签写回 {applied} 个文件级文档（缺失 {missing}）。"
              f"检索侧已即时生效（docs.tags 已同步）。")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
