"""mirror 来源对账（本地原始数据优先）：快照 + 宽限期 + 自动回撤删除。

设计：docs/DESIGN-audited-git-source.md（审计驱动 git 文档来源，PR 合入落盘 /
回撤 PR 删文件）。`deletion_policy: keep`（默认）的来源**完全不经此路径**。

- 快照 `data/source_state/<source_id>.json`：`{source_id: 连续缺席构建次数}`；
- 损坏/缺失视为**首轮**：首轮做**全量对账**——库里该 kind 前缀的存量文档若本轮
  未发现，视为已缺席 1 次计入宽限（覆盖"开启 mirror 前源文件就已缺失"的历史
  滞留；连续缺席达宽限即自动回撤，宁慢勿误）；
- 连续 `GRACE_BUILDS`（2）次构建缺席 → 自动 `mark_deleted`（备份 doc_row →
  kb 四层删 → 登记墓碑 → 生成"待实际删除"审核项，可撤回）；
- 已墓碑 sid 重新出现 → **解除墓碑**（本地原始数据优先：文件在场即应存在；
  人工标签/认证覆盖层残留会在重新入库时自动恢复）。

前提：mirror 语义要求该 kind 的导入路径**唯一**（sid 前缀按 kind 划分，
同 kind 多来源无法区分文档归属）。
"""
import json
import sqlite3
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
                            kb_path, review_db, *, sid_prefix: str,
                            grace: int = GRACE_BUILDS) -> dict:
    """mirror 来源对账。`discovered` = 本轮该来源发现的全部 source_id。

    - **首轮**（快照缺失/损坏）：全量对账——库里 `sid_prefix` 前缀的存量文档
      本轮未发现 → 计缺席 1 次入快照（历史滞留进入正常宽限轨道）；
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
    store = ReviewStore(review_db)
    tombs = load_tombstones(review_db)
    if first_round:
        # 首轮全量对账：库内存量 vs 本轮发现。历史缺失的文档从未进过快照，
        # 只有在这里补记缺席，才能进入宽限轨道被自愈。
        stored: list[str] = []
        try:
            conn = sqlite3.connect(str(kb_path))
            try:
                stored = [r[0] for r in conn.execute(
                    "SELECT source_id FROM docs WHERE source_id LIKE ?",
                    (sid_prefix + "%",))]
            finally:
                conn.close()
        except sqlite3.Error as e:
            print(f"[sync:{source_id}] 首轮存量对账失败（仅建快照，不计时序）: {e}")
        seeded = [s for s in stored if s not in set(disc)]
        for sid in seeded:
            snap[sid] = 1  # 本轮缺席第 1 次；下次构建再缺席即达宽限 → 自动回撤
            out["grace"].append(sid)
        if stored:
            print(f"[sync:{source_id}] 首轮全量对账: 库内存量 {len(stored)} · "
                  f"本轮发现 {len(disc)} · 历史缺席 {len(seeded)}"
                  f"（计宽限，连续缺席达 {grace} 次自动回撤）", flush=True)
    else:
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
