"""mirror 来源对账（本地原始数据优先）：快照 + 宽限期 + 自动回撤删除。

设计：docs/DESIGN-audited-git-source.md（审计驱动 git 文档来源，PR 合入落盘 /
回撤 PR 删文件）。`deletion_policy: keep`（默认）的来源**完全不经此路径**。

- 快照 `data/source_state/<source_id>.json`：`{source_id: 连续缺席构建次数}`；
- 损坏/缺失视为**首轮**：只建快照不删除（宁漏删不误删）；
- 连续 `GRACE_BUILDS`（2）次构建缺席 → 自动 `mark_deleted`（备份 doc_row →
  kb 四层删 → 登记墓碑 → 生成"待实际删除"审核项，可撤回）；
- 已墓碑 sid 重新出现 → **解除墓碑**（本地原始数据优先：文件在场即应存在；
  人工标签/认证覆盖层残留会在重新入库时自动恢复）。
"""
import json
from pathlib import Path
from typing import Optional

GRACE_BUILDS = 2


def _state_path(cfg, source_id: str) -> Path:
    return cfg.resolve(f"data/source_state/{source_id}.json")


def load_snapshot(path: Path) -> Optional[dict]:
    """读快照；文件缺失/损坏 → None（首轮）。"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"[sync] 快照损坏，视为首轮（本轮不删除）: {path}", flush=True)
        return None
    if not isinstance(data, dict):
        print(f"[sync] 快照格式异常，视为首轮（本轮不删除）: {path}", flush=True)
        return None
    out: dict[str, int] = {}
    for k, v in data.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            out[str(k)] = 0
    return out


def save_snapshot(path: Path, snap: dict) -> None:
    """原子写快照（tmp + replace，防中断出半个文件）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _find_item_id(store, category: str, item_ref: str) -> Optional[int]:
    for it in store.list_items(category=category):
        if it["item_ref"] == item_ref:
            return it["id"]
    return None


def reconcile_mirror_source(cfg, source_id: str, discovered: list[str],
                            kb_path, review_db,
                            grace: int = GRACE_BUILDS) -> dict:
    """mirror 来源对账。`discovered` = 本轮该来源发现的全部 source_id。

    - 缺席 sid：宽限计数；达 `grace` → 自动回撤删除（建 source_withdraw 审核项 →
      mark_deleted：备份 doc_row + kb 四层删 + 登记墓碑；kb 行已不在则只补墓碑）；
    - 已墓碑 sid 重新出现：解除墓碑（后续 ingest 正常入库，覆盖层自动恢复人工成果）；
    - 快照原子更新。
    返回 {"withdrawn": [...], "grace": [...], "untombstoned": [...], "first_round": bool}。
    """
    from .review import ReviewStore, load_tombstones, record_tombstones

    path = _state_path(cfg, source_id)
    snap = load_snapshot(path)
    first_round = snap is None
    snap = dict(snap or {})
    disc = list(dict.fromkeys(discovered))
    out: dict = {"withdrawn": [], "grace": [], "untombstoned": [],
                 "first_round": first_round}
    if not first_round:
        tombs = load_tombstones(review_db)
        store = ReviewStore(review_db)
        for sid in [s for s in snap if s not in disc]:
            absent = int(snap.get(sid, 0)) + 1
            if absent < grace:
                snap[sid] = absent
                out["grace"].append(sid)
                continue
            # 达宽限 → 自动回撤删除（幂等：同 sid 已有审核项则复用）
            store.add_item("source_withdraw", sid, {
                "source_id": sid, "title": sid,
                "note": f"mirror 来源 {source_id} 回撤自动删除（连续 {absent} 次构建缺席）"})
            item_id = _find_item_id(store, "source_withdraw", sid)
            try:
                if item_id is not None:
                    store.mark_deleted(item_id, "source-sync", kb_path,
                                       note=f"来源 {source_id} 回撤自动删除")
                else:
                    record_tombstones(review_db, [sid], "source-sync")
            except ValueError:
                # kb 里已无此行（此前被人工删除等）——只补墓碑保持一致
                record_tombstones(review_db, [sid], "source-sync")
            out["withdrawn"].append(sid)
            snap.pop(sid, None)
        # 本地原始数据优先：文件在场覆盖墓碑（撤稿后重新合入 → 自动恢复入库资格）
        back = [s for s in disc if s in tombs]
        if back:
            store.untombstone(back)
            out["untombstoned"] = back
    for sid in disc:
        snap[sid] = 0
    save_snapshot(path, snap)
    return out
