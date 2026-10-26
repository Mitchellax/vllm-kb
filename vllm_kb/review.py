"""审核工作台数据层：审核队列（review.sqlite3）+ 审核项生成 + API 配置中心。

- **审核队列**：所有需要人工确认的位置共用（verification_pending / case_title_flag /
  ocr_mismatch / low_confidence_ocr / equivalence_candidate / table_join_candidate），
  独立于只读 kb.sqlite3（检索 API 不碰审核库）；
- **审核项生成（seed，幂等）**：导入/OCR 时由调用方 add_item，或运行 seed_* 扫描现有库补单：
  * seed_verification_pending: kb.sqlite3 中 verification=unverified 的文档 → 待补标；
  * seed_case_title_flags: 案例标题含"待审核/待修改" → 待确认；
- **API 配置中心**：集中展示 embedding / OCR / GitHub 等 API 配置状态（**key 一律脱敏**，
  只显示"已配置/未配置"），供工作台统一管理入口。
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .doc_kinds import EXTERNAL_DOC_TYPES

if TYPE_CHECKING:
    from .config import AppConfig

# 审核类别（与布局文档 §10 一致）
CATEGORIES = (
    "verification_pending",   # 未验证文档待补标（wiki/无标记案例/工程师记录）
    "case_title_flag",        # 案例标题含"待审核/待修改"
    "ocr_mismatch",           # 图文互证不一致（正文签名 ↔ OCR 签名）
    "low_confidence_ocr",     # C 档低质量 OCR 签名待核对
    "equivalence_candidate",  # 跨来源疑似同一问题（EQUIVALENT_TO 候选）
    "table_join_candidate",   # 表格行引用 GitHub 编号 join 候选
    "tag_candidate",          # 自动提取的未收录标签候选（采纳→入词典+打标；忽略不记录）
    "source_withdraw",        # mirror 来源回撤自动删除（待实际删除列表可撤回；见 DESIGN-audited-git-source.md）
)

# 审核项状态机（审核人员无编辑权限，只有三类动作）：
#   pending   -> approved   认证：文档有效，不再提示
#   pending   -> suspected  存疑：重新进入队列，排在未审核（pending）之后
#   pending   -> deleted    删除：只删 kb.sqlite3 数据库记录，原始资产文件保留
#                           （人工到"待实际删除"列表本地删除，可撤回）
#   deleted   -> pending    撤回：恢复数据库记录，重新进入队列
STATUSES = ("pending", "approved", "suspected", "deleted")
# 队列排序权重：未审核最先，存疑其次，已处理最后
_STATUS_ORDER = {"pending": 0, "suspected": 1, "approved": 2, "deleted": 3}

# 资产注册表（review.sqlite3，管理员侧）：资产路径**不进 canonical/检索库**（安全约束）；
# 审核页显示/预览/待删除列表经 asset_id → rel_path 找回文件；检索 API 全程不碰审核库。
_ASSET_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS asset_registry (
  asset_id TEXT PRIMARY KEY,
  rel_path TEXT NOT NULL,
  sha256 TEXT,
  size INTEGER,
  source_type TEXT
);
"""

# 文档级标签覆盖层（kb.sqlite3，ingest 建表；本模块读写函数独立可用时幂等补建）
_DOC_TAGS_DDL = """
CREATE TABLE IF NOT EXISTS doc_tags (
  source_id TEXT PRIMARY KEY,
  auto_snapshot TEXT,
  excluded TEXT,
  manual TEXT,
  updated_at TEXT,
  reviewer TEXT
);
"""


def _ensure_doc_tags(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("SELECT 1 FROM doc_tags LIMIT 1")
    except sqlite3.OperationalError:
        conn.executescript(_DOC_TAGS_DDL)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    item_ref TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT,
    reviewed_at TEXT,
    reviewer TEXT,
    result TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_items(status, category);
CREATE TABLE IF NOT EXISTS deleted_docs (
    source_id TEXT PRIMARY KEY,
    deleted_at TEXT,
    reviewer TEXT
);
""" + _ASSET_REGISTRY_DDL


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------- kb.sqlite3 docs 记录读写（删除/撤回用） ----------------

def _doc_columns(kb_path: str | Path) -> list[str]:
    conn = sqlite3.connect(str(kb_path))
    try:
        return [c[1] for c in conn.execute("PRAGMA table_info(docs)").fetchall()]
    finally:
        conn.close()


def _read_doc_row(kb_path: str | Path, source_id: str) -> Optional[dict]:
    """读 docs 完整行（extra 解析为 dict）。"""
    conn = sqlite3.connect(str(kb_path))
    try:
        cols = _doc_columns(kb_path)
        row = conn.execute("SELECT * FROM docs WHERE source_id=?", (source_id,)).fetchone()
        if not row:
            return None
        d = dict(zip(cols, row))
        try:
            d["extra"] = json.loads(d.get("extra") or "{}")
        except (TypeError, json.JSONDecodeError):
            d["extra"] = {}
        return d
    finally:
        conn.close()


def _delete_doc_row(kb_path: str | Path, source_id: str) -> None:
    conn = sqlite3.connect(str(kb_path), timeout=15)
    try:
        conn.execute("DELETE FROM docs WHERE source_id=?", (source_id,))
        conn.commit()
    finally:
        conn.close()


def _delete_doc_chunks(kb_path: str | Path, source_id: str) -> None:
    """删除该文档的全部 chunk（chunks_fts + chunks_meta），并返回 chunk_id 列表。

    FTS 是 contentless 表（text 在 chunks_fts_content），chunks_fts 只存索引，
    直接 DELETE FROM chunks_fts 即可；chunks_meta 单独删。
    """
    conn = sqlite3.connect(str(kb_path), timeout=15)
    try:
        ids = [r[0] for r in conn.execute(
            "SELECT chunk_id FROM chunks_meta WHERE doc_id=?", (source_id,)).fetchall()]
        conn.execute("DELETE FROM chunks_fts WHERE doc_id=?", (source_id,))
        conn.execute("DELETE FROM chunks_meta WHERE doc_id=?", (source_id,))
        conn.commit()
        return ids
    finally:
        conn.close()


def _restore_doc_row(kb_path: str | Path, doc_row: dict) -> None:
    """用备份恢复 docs 行（extra dict → JSON 字符串）。"""
    conn = sqlite3.connect(str(kb_path))
    try:
        cols = _doc_columns(kb_path)
        data = dict(doc_row)
        if isinstance(data.get("extra"), dict):
            data["extra"] = json.dumps(data["extra"], ensure_ascii=False)
        row = {c: data.get(c) for c in cols}
        placeholders = ",".join("?" for _ in cols)
        conn.execute(
            f"INSERT INTO docs ({','.join(cols)}) VALUES ({placeholders})",
            [row[c] for c in cols],
        )
        conn.commit()
    finally:
        conn.close()


class ReviewStore:
    """审核队列存储（sqlite3，独立文件，可写）。

    每次操作使用短连接（FastAPI 线程池跨线程复用长连接会触发
    "SQLite objects created in a thread..." 错误）。
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def close(self) -> None:
        pass  # 短连接模式，无需持有

    def add_item(self, category: str, item_ref: str, payload: Optional[dict] = None,
                 dedupe: bool = True) -> bool:
        """新增审核项。dedupe=True 时同一 (category, item_ref) 已存在（**任意状态**）则跳过——
        确认/拒绝/修改过的项不会被自动补单再次打扰；如需重新审，先删除记录。
        返回是否新增。"""
        if category not in CATEGORIES:
            raise ValueError(f"未知审核类别: {category}（支持 {CATEGORIES}）")
        conn = self._connect()
        try:
            if dedupe:
                row = conn.execute(
                    "SELECT id FROM review_items WHERE category=? AND item_ref=?",
                    (category, item_ref),
                ).fetchone()
                if row:
                    return False
            conn.execute(
                "INSERT INTO review_items(category, item_ref, payload, status, created_at) "
                "VALUES(?,?,?,?,?)",
                (category, item_ref, json.dumps(payload or {}, ensure_ascii=False), "pending", _now()),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def delete_item(self, item_id: int) -> bool:
        """彻底删除审核项（重新审/误报清理用）。返回是否删除。"""
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM review_items WHERE id=?", (int(item_id),))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def cleanup_old_tag_candidates(self) -> int:
        """删除旧版 tag_candidate 格式残留（item_ref = 'source_id::name'，无 'tag:' 前缀）。

        聚合版 item_ref 为 'tag:{name}'；旧格式项在新版 seed 前删除，未处理的候选
        由 seed 按聚合格式重建（已忽略/采纳的因其词已收录或 dedupe 不再打扰）。
        返回删除条数。
        """
        conn = self._connect()
        try:
            cur = conn.execute(
                "DELETE FROM review_items WHERE category='tag_candidate' "
                "AND item_ref NOT LIKE 'tag:%'"
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def cleanup_legacy_row_items(self) -> int:
        """删除旧版**逐行** excel 补标项（item_ref 形如 `excel:文件:sheet:行`，pending）。

        表格审核已升级为**文件粒度**（`excelfile:<文件>:<sha8>`），逐行 pending 项被取代；
        已处理（approved/suspected）的保留作历史。当前格式 `excel:<文件名>`（恰好一个
        冒号）不受影响。返回删除条数。
        """
        conn = self._connect()
        try:
            cur = conn.execute(
                "DELETE FROM review_items WHERE category='verification_pending' "
                "AND status='pending' AND item_ref GLOB 'excel:*:*:*'"
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def clear_category(self, category: str) -> int:
        """清空某类审核项（**全状态**，含已处理历史）——供"提取规则升级后重播"用。

        典型场景：候选名规范化/标题策略等**生成侧规则**变更后，存量项的 payload 是
        旧规则快照（seed 对已存在 item_ref 不更新），清空重补单才能拿到新 payload。
        仅接受已知类别，防误删。返回删除条数。
        """
        if category not in CATEGORIES:
            raise ValueError(f"未知审核类别: {category}（支持 {CATEGORIES}）")
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM review_items WHERE category=?", (category,))
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def list_items(self, category: Optional[str] = None, status: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> list[dict]:
        """审核队列：未审核（pending）优先、存疑（suspected）其次，已处理最后。"""
        sql = ("SELECT id, category, item_ref, payload, status, created_at, reviewed_at, "
               "reviewer, result FROM review_items")
        conds, args = [], []
        if category:
            conds.append("category=?")
            args.append(category)
        if status:
            if status not in STATUSES:
                raise ValueError(f"非法状态: {status}（支持 {STATUSES}）")
            conds.append("status=?")
            args.append(status)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += (" ORDER BY CASE status WHEN 'pending' THEN 0 WHEN 'suspected' THEN 1 "
                "WHEN 'approved' THEN 2 ELSE 3 END, id DESC LIMIT ? OFFSET ?")
        args += [int(limit), int(offset)]
        conn = self._connect()
        try:
            rows = conn.execute(sql, args).fetchall()
        finally:
            conn.close()
        return [self._row_dict(r) for r in rows]

    def get_item(self, item_id: int) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, category, item_ref, payload, status, created_at, reviewed_at, reviewer, result "
                "FROM review_items WHERE id=?", (int(item_id),),
            ).fetchone()
        finally:
            conn.close()
        return self._row_dict(row) if row else None

    @staticmethod
    def _row_dict(r) -> dict:
        return {
            "id": r[0], "category": r[1], "item_ref": r[2],
            "payload": json.loads(r[3] or "{}"),
            "status": r[4], "created_at": r[5], "reviewed_at": r[6],
            "reviewer": r[7],
            "result": json.loads(r[8]) if r[8] else None,
        }

    def update_payload(self, item_id: int, payload: dict) -> bool:
        """覆盖审核项 payload（seed 队列校正用；不影响状态/审计字段）。"""
        conn = self._connect()
        try:
            cur = conn.execute("UPDATE review_items SET payload=? WHERE id=?",
                               (json.dumps(payload, ensure_ascii=False), int(item_id)))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def tombstone(self, source_ids: list[str], reviewer: str = "review-ui") -> int:
        """登记删除墓碑（deleted_docs）：审核台删除 = **一等删除**——任何构建路径
        （增量/资产回退/GitHub 缓存/--rebuild）都不得复活这些 sid（ingest/upsert/
        建图读取墓碑跳过）；撤回删除（undo_delete）解除。"""
        ids = [s for s in source_ids if s]
        if not ids:
            return 0
        conn = self._connect()
        try:
            now = _now()
            conn.executemany(
                "INSERT OR REPLACE INTO deleted_docs(source_id, deleted_at, reviewer) "
                "VALUES(?,?,?)",
                [(s, now, reviewer) for s in ids])
            conn.commit()
            return len(ids)
        finally:
            conn.close()

    def untombstone(self, source_ids: list[str]) -> int:
        """解除删除墓碑（撤回删除用）。"""
        ids = [s for s in source_ids if s]
        if not ids:
            return 0
        conn = self._connect()
        try:
            conn.executemany("DELETE FROM deleted_docs WHERE source_id=?",
                             [(s,) for s in ids])
            conn.commit()
            return len(ids)
        finally:
            conn.close()

    def review(self, item_id: int, action: str, reviewer: str,
               result: Optional[dict] = None) -> bool:
        """提交标注（**不改原始内容**）：action ∈ approved（认证）| suspected（存疑）。
        存疑项保留在队列、排未审核之后；删除走 mark_deleted（需 kb 路径）。"""
        if action not in ("approved", "suspected"):
            raise ValueError(f"非法标注动作: {action}（支持 approved | suspected；删除走 mark_deleted）")
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE review_items SET status=?, reviewed_at=?, reviewer=?, result=? WHERE id=?",
                (action, _now(), reviewer, json.dumps(result or {}, ensure_ascii=False), int(item_id)),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def reopen(self, item_id: int, reviewer: str, result: Optional[dict] = None) -> bool:
        """审核项重开（回 pending）：认证撤销（verification 重置为 unverified）专用——
        撤销认证后文档回到待补标状态，队列项必须同步回 pending，否则用户失去重审入口
        （add_item 的 (category, item_ref) 去重铁律会挡住 seed 重新入队）。"""
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE review_items SET status='pending', reviewed_at=?, reviewer=?, result=? "
                "WHERE id=?",
                (_now(), reviewer, json.dumps(result or {}, ensure_ascii=False), int(item_id)),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def mark_deleted(self, item_id: int, reviewer: str, kb_path: str | Path,
                     note: str = "") -> bool:
        """标记删除：**只删除 kb.sqlite3 的数据库记录**，原始资产文件保留
        （由人员到"待实际删除"列表手动本地删除）。删除前把完整 docs 行存入
        payload.doc_row，供撤回恢复。

        **文件级表格项**（payload.rows 聚合多行）：备份并删除文件内全部行
        （payload.doc_rows 列表），撤回整体恢复。
        """
        it = self.get_item(item_id)
        if it is None:
            return False
        rows = [r.get("source_id") for r in ((it["payload"] or {}).get("rows") or [])
                if isinstance(r, dict) and r.get("source_id")]
        payload = dict(it["payload"] or {})
        deleted_sids: list[str] = []
        if rows:
            # 文件级：批量备份 + 删除（单行缺失不阻断——可能已被单独处理）
            doc_rows = []
            for sid in rows:
                d = _read_doc_row(kb_path, sid)
                if d is not None:
                    doc_rows.append(d)
            if not doc_rows:
                raise ValueError("文件内已无可删除的文档记录（可能已全部删除）")
            for d in doc_rows:
                _delete_doc_row(kb_path, d["source_id"])
            deleted_sids = [d["source_id"] for d in doc_rows]
            payload["doc_rows"] = doc_rows
            payload["asset"] = (doc_rows[0].get("extra") or {}).get("asset", {})
            payload["deleted_note"] = note
        else:
            doc_row = _read_doc_row(kb_path, it["item_ref"])
            if doc_row is None:
                raise ValueError(f"kb.sqlite3 中不存在 {it['item_ref']}（可能已删除）")
            _delete_doc_row(kb_path, it["item_ref"])
            deleted_sids = [it["item_ref"]]
            payload["doc_row"] = doc_row
            payload["asset"] = (doc_row.get("extra") or {}).get("asset", {})
            payload["deleted_note"] = note
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE review_items SET status='deleted', reviewed_at=?, reviewer=?, "
                "payload=?, result=? WHERE id=?",
                (_now(), reviewer, json.dumps(payload, ensure_ascii=False),
                 json.dumps({"action": "deleted", "note": note}, ensure_ascii=False), int(item_id)),
            )
            conn.commit()
            # 墓碑：删除的 sid 不得被任何构建路径复活（撤回时解除）
            self.tombstone(deleted_sids, reviewer)
            return cur.rowcount > 0
        finally:
            conn.close()

    def undo_delete(self, item_id: int, kb_path: str | Path) -> bool:
        """撤回删除：用 payload.doc_row / doc_rows 恢复 kb.sqlite3 记录，审核项回到 pending。"""
        it = self.get_item(item_id)
        if it is None or it["status"] != "deleted":
            raise ValueError(f"审核项 {item_id} 不存在或不是 deleted 状态（当前 {it['status'] if it else '?'}）")
        doc_rows = (it.get("payload") or {}).get("doc_rows") or []
        doc_row = (it.get("payload") or {}).get("doc_row")
        if doc_rows:
            for d in doc_rows:
                _restore_doc_row(kb_path, d)
        elif doc_row:
            _restore_doc_row(kb_path, doc_row)
        else:
            raise ValueError("该审核项缺少 doc_row 备份，无法撤回（可能由旧版本生成）")
        payload = dict(it["payload"] or {})
        payload.pop("doc_row", None)
        payload.pop("doc_rows", None)
        payload.pop("deleted_note", None)
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE review_items SET status='pending', reviewed_at=NULL, reviewer=NULL, "
                "payload=?, result=NULL WHERE id=?",
                (json.dumps(payload, ensure_ascii=False), int(item_id)),
            )
            conn.commit()
            # 解除墓碑：撤回 = 恢复复活资格
            self.untombstone([d["source_id"] for d in doc_rows] if doc_rows
                             else ([doc_row["source_id"]] if doc_row else []))
            return cur.rowcount > 0
        finally:
            conn.close()

    def stats(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT category, status, count(*) FROM review_items GROUP BY category, status"
            ).fetchall()
        finally:
            conn.close()
        for cat, st, n in rows:
            d = out.setdefault(cat, {"pending": 0, "suspected": 0, "total": 0})
            d["total"] += n
            if st == "pending":
                d["pending"] += n
            elif st == "suspected":
                d["suspected"] += n
        return out


def default_review_path(cfg: Optional["AppConfig"] = None) -> Path:
    if cfg is not None:
        return cfg.resolve(getattr(cfg.storage, "review_path", "data/review.sqlite3"))
    return Path("data/review.sqlite3")


def record_tombstones(review_db: str | Path, source_ids: list[str],
                      reviewer: str = "review-ui") -> None:
    """登记删除墓碑（模块级便捷入口，端点用；Store.tombstone 同义，含建表）。"""
    ids = [s for s in source_ids if s]
    if not ids:
        return
    conn = sqlite3.connect(str(review_db))
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS deleted_docs (
            source_id TEXT PRIMARY KEY, deleted_at TEXT, reviewer TEXT)""")
        conn.executemany(
            "INSERT OR REPLACE INTO deleted_docs(source_id, deleted_at, reviewer) VALUES(?,?,?)",
            [(s, _now(), reviewer) for s in ids])
        conn.commit()
    finally:
        conn.close()


def list_withdrawn(review_db: str | Path) -> list[dict]:
    """已回撤文档（删除墓碑）列表——审核台"已回撤"痕迹视图的数据源。

    返回 [{source_id, deleted_at, reviewer, undo_item_id}]；undo_item_id = 可撤回的
    审核项 id（mirror 自动回撤生成；人工审核台删除无审核项 → None，不可从此处撤回）。
    """
    p = Path(review_db)
    if not p.exists():
        return []
    conn = sqlite3.connect(str(p))
    try:
        try:
            rows = conn.execute(
                "SELECT d.source_id, d.deleted_at, d.reviewer, MIN(i.id) FROM deleted_docs d "
                "LEFT JOIN review_items i ON i.item_ref = d.source_id AND i.status='deleted' "
                "GROUP BY d.source_id ORDER BY d.deleted_at DESC").fetchall()
        except sqlite3.OperationalError:
            return []  # 旧库尚无 deleted_docs 表
    finally:
        conn.close()
    return [{"source_id": r[0], "deleted_at": r[1], "reviewer": r[2],
             "undo_item_id": r[3]} for r in rows]


def load_tombstones(review_db: str | Path) -> set[str]:
    """已删除文档墓碑集合（deleted_docs；表不存在/库缺失 → 空集）。

    ingest / upsert_unified_canonical / 建图读取墓碑跳过已删 sid——审核台删除是一等
    删除，资产层回退 / GitHub 采集缓存 / canonical 旧行都不得复活（2026-10 用户实测：
    删文件 + 删记录后 --skip-pull 仍重新入库）。撤回删除（undo_delete）解除墓碑。
    """
    p = Path(review_db)
    if not p.exists():
        return set()
    conn = sqlite3.connect(str(p))
    try:
        try:
            return {r[0] for r in conn.execute("SELECT source_id FROM deleted_docs").fetchall()}
        except sqlite3.OperationalError:
            return set()  # 旧库尚无 deleted_docs 表
    finally:
        conn.close()


# ---------------- 外源文档管理（审核工作台"文档管理"页签） ----------------

# 外源文档 = 导入的业务来源（PDF / Markdown / Word / 表格 / OCR 等），由文档种类声明表派生。
# 原先这里是一份**没人引用**的字面量（实际查询用的是 `NOT LIKE 'github%'` 这个 hack）——
# 声明看着存在却不生效；现在真的被 list_external_docs 使用。
_EXTERNAL_SOURCE_TYPES = EXTERNAL_DOC_TYPES


def list_external_docs(kb_path: str | Path, review_db: Optional[str | Path] = None,
                       limit: int = 200, offset: int = 0,
                       q: Optional[str] = None) -> list[dict]:
    """列出 kb.sqlite3 中的外源文档（source_type 在 `_EXTERNAL_SOURCE_TYPES` 内）。

    q：子串过滤 source_id/title（大小写不敏感，LIKE 转义）——73k+ 文档远超单页窗口，
    没有搜索时用户只能看到排序前 500 条，删除后窗口"补位"极易误读为删除失败（2026-10）。

    返回 {source_id, source_type, title, component, url, verification,
    asset_id, asset_path（管理员侧 rel_path，经 asset_registry）,
    rel_src（来源 id）/ rel_dir（imports 第一层子目录；根级 ""；资产回退无 rel_dir、
    网络来源两者皆缺 → None）,
    tags: {auto, excluded, manual, final}, duplicate（同 stem 重名告警）}。
    """
    from .tagging import merge_final

    conn = sqlite3.connect(str(kb_path))
    try:
        where = f"source_type IN ({','.join('?' * len(_EXTERNAL_SOURCE_TYPES))})"
        params: list = [*_EXTERNAL_SOURCE_TYPES]
        if q and q.strip():
            like = f"%{_like_escape(q.strip())}%"
            where += " AND (source_id LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\')"
            params += [like, like]
        rows = conn.execute(
            f"""SELECT source_id, source_type, title, component, url, extra
                FROM docs WHERE {where}
                ORDER BY source_type, source_id LIMIT ? OFFSET ?""",
            (*params, int(limit), int(offset)),
        ).fetchall()
        overlay = load_doc_tags_conn(conn)
        # 人工补标覆盖层：verification 本身已在 extra（ingest 合并后的生效值），
        # 这里额外给出**是否人工所加 + 谁加的**，供审计/排查"这文档为什么是 tested"
        verif_overlay = {sid: (v, rev) for sid, v, rev in
                         conn.execute("SELECT source_id, verification, reviewer "
                                      "FROM doc_verification").fetchall()} \
            if _has_table(conn, "doc_verification") else {}
        try:
            tags_by_id = {
                sid: json.loads(raw) if raw else []
                for sid, raw in conn.execute("SELECT source_id, tags FROM docs").fetchall()
            }
        except sqlite3.OperationalError:
            tags_by_id = {}  # 旧库无 tags 列
    finally:
        conn.close()
    assets = list_assets(review_db) if review_db else {}
    # 同 stem 重名统计（pdf:/md: 前缀后的部分；`--<sha8>` 是自动消歧后缀，不算 stem）
    stems: dict[str, int] = {}
    for sid, *_ in rows:
        stem = _stem_of(sid)
        stems[stem] = stems.get(stem, 0) + 1
    out = []
    for sid, st, title, comp, url, extra in rows:
        try:
            ex = json.loads(extra or "{}")
        except (TypeError, json.JSONDecodeError):
            ex = {}
        asset = ex.get("asset") or {}
        asset_id = str(asset.get("asset_id", "") or "")
        rel_path = (assets.get(asset_id) or {}).get("rel_path", "")
        ov = overlay.get(sid, {"excluded": [], "manual": [], "auto_snapshot": []})
        auto = list(ov.get("auto_snapshot") or [])
        if not auto:
            auto = list(tags_by_id.get(sid, []))  # 旧库无快照：以最终标签近似
        final = merge_final(auto, ov.get("excluded", []), ov.get("manual", []))
        stem = _stem_of(sid)
        out.append({
            "source_id": sid,
            "source_type": st,
            "title": title or "",
            "component": comp or "",
            "url": url or "",
            "verification": ex.get("verification", ""),
            "verification_manual": (verif_overlay.get(sid) or (None, None))[0],
            "verification_reviewer": (verif_overlay.get(sid) or (None, None))[1],
            "asset_id": asset_id,
            "asset_path": rel_path,
            "rel_src": ex.get("rel_src"),
            "rel_dir": ex.get("rel_dir"),
            "tags": {"auto": auto, "excluded": ov.get("excluded", []),
                     "manual": ov.get("manual", []), "final": final},
            "duplicate": stems.get(stem, 0) > 1,
        })
    return out


def delete_external_doc(kb_path: str | Path, source_id: str,
                        vector_store=None) -> dict:
    """从数据库彻底删除外源文档（本地资产文件不动）。

    - kb.sqlite3：docs 行 + chunks_fts/chunks_meta（该文档全部 chunk）；
    - 向量库：lancedb 删除该文档全部 chunk 向量（vector_store 传入时）。
    返回 {"chunks_deleted": n}。下次增量入库时若本地文件仍在，会重新入库。
    """
    doc_row = _read_doc_row(kb_path, source_id)
    if doc_row is None:
        raise ValueError(f"kb.sqlite3 中不存在 {source_id}（可能已删除）")
    chunk_ids = _delete_doc_chunks(kb_path, source_id)
    _delete_doc_row(kb_path, source_id)
    if vector_store is not None:
        try:
            vector_store.delete_doc(source_id)
        except Exception as e:
            print(f"[review] 向量删除失败（不影响 SQLite 删除）: {e}")
    return {"chunks_deleted": len(chunk_ids)}


# ---------------- 文档级标签覆盖层（doc_tags 表，kb.sqlite3，ingest 建表） ----------------
# 最终标签 = (auto − excluded) ∪ manual（tagging.merge_final，ingest 与 build_graph 共用）。
# 覆盖层函数统一接受 sqlite3.Connection：ingest 传自己的写连接，审核页开短连接传入。

def load_doc_tags_conn(conn: sqlite3.Connection) -> dict[str, dict]:
    """批量读取覆盖层：{source_id: {"excluded": [...], "manual": [...], "auto_snapshot": [...]}}
    （ingest/建图预读 excluded/manual；审核页列表展示用 auto_snapshot）。"""
    out: dict[str, dict] = {}
    _ensure_doc_tags(conn)
    try:
        rows = conn.execute(
            "SELECT source_id, auto_snapshot, excluded, manual FROM doc_tags"
        ).fetchall()
    except sqlite3.OperationalError:
        return out  # 旧库尚无 doc_tags 表（ingest 首次写入时创建）
    for sid, auto, excluded, manual in rows:
        out[sid] = {
            "auto_snapshot": json.loads(auto) if auto else [],
            "excluded": json.loads(excluded) if excluded else [],
            "manual": json.loads(manual) if manual else [],
        }
    return out


def get_doc_tags_conn(conn: sqlite3.Connection, source_id: str) -> dict:
    """单篇覆盖层（审核页展示用）。"""
    _ensure_doc_tags(conn)
    row = conn.execute(
        "SELECT auto_snapshot, excluded, manual, updated_at, reviewer "
        "FROM doc_tags WHERE source_id=?", (source_id,),
    ).fetchone()
    if row is None:
        return {"source_id": source_id, "auto_snapshot": [], "excluded": [], "manual": [],
                "updated_at": None, "reviewer": None}
    return {
        "source_id": source_id,
        "auto_snapshot": json.loads(row[0]) if row[0] else [],
        "excluded": json.loads(row[1]) if row[1] else [],
        "manual": json.loads(row[2]) if row[2] else [],
        "updated_at": row[3],
        "reviewer": row[4],
    }


def upsert_auto_snapshot_conn(conn: sqlite3.Connection, source_id: str, auto_tags: list[str]) -> None:
    """回写自动标签快照（入库时刷新；不影响 excluded/manual/reviewer）。"""
    _ensure_doc_tags(conn)
    row = conn.execute("SELECT excluded, manual, reviewer FROM doc_tags WHERE source_id=?",
                       (source_id,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO doc_tags(source_id, auto_snapshot, excluded, manual, updated_at, reviewer) "
            "VALUES(?,?,?,?,?,?)",
            (source_id, json.dumps(auto_tags, ensure_ascii=False), "[]", "[]", _now(), ""),
        )
    else:
        conn.execute(
            "UPDATE doc_tags SET auto_snapshot=?, updated_at=? WHERE source_id=?",
            (json.dumps(auto_tags, ensure_ascii=False), _now(), source_id),
        )


def set_doc_tags_conn(conn: sqlite3.Connection, source_id: str,
                      excluded: Optional[list[str]] = None,
                      manual: Optional[list[str]] = None,
                      reviewer: Optional[str] = None) -> dict:
    """更新覆盖层（excluded/manual 整体替换；None=保持不变），并**同步最终标签到 docs.tags**
    （检索侧立即生效；图侧重建时按同一公式再算，两处结果一致）。审核页调用。"""
    from .tagging import merge_final

    _ensure_doc_tags(conn)
    _ensure_docs_tags_col(conn)
    row = conn.execute("SELECT excluded, manual FROM doc_tags WHERE source_id=?",
                       (source_id,)).fetchone()
    cur_ex = json.loads(row[0]) if row and row[0] else []
    cur_ma = json.loads(row[1]) if row and row[1] else []
    if excluded is not None:
        cur_ex = [t for t in excluded if t]
    if manual is not None:
        cur_ma = [t for t in manual if t]
    if row is None:
        conn.execute(
            "INSERT INTO doc_tags(source_id, auto_snapshot, excluded, manual, updated_at, reviewer) "
            "VALUES(?,?,?,?,?,?)",
            (source_id, "[]", json.dumps(cur_ex, ensure_ascii=False),
             json.dumps(cur_ma, ensure_ascii=False), _now(), reviewer or ""),
        )
    else:
        conn.execute(
            "UPDATE doc_tags SET excluded=?, manual=?, updated_at=?, reviewer=? WHERE source_id=?",
            (json.dumps(cur_ex, ensure_ascii=False), json.dumps(cur_ma, ensure_ascii=False),
             _now(), reviewer or "", source_id),
        )
    # 同步最终标签到 docs.tags：基准 = auto_snapshot（无则回退当前 docs.tags）
    auto_row = conn.execute("SELECT auto_snapshot FROM doc_tags WHERE source_id=?",
                            (source_id,)).fetchone()
    auto = json.loads(auto_row[0]) if auto_row and auto_row[0] else []
    if not auto:
        trow = conn.execute("SELECT tags FROM docs WHERE source_id=?", (source_id,)).fetchone()
        if trow and trow[0]:
            auto = json.loads(trow[0])
    final = merge_final(auto, cur_ex, cur_ma)
    _ensure_docs_tags_col(conn)
    conn.execute("UPDATE docs SET tags=? WHERE source_id=?",
                 (json.dumps(final, ensure_ascii=False), source_id))
    return {"source_id": source_id, "excluded": cur_ex, "manual": cur_ma, "final": final}


def _ensure_docs_tags_col(conn: sqlite3.Connection) -> None:
    """旧库 docs 表补 tags 列（与 ingest 迁移一致；审核页标签编辑兼容老库）。"""
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(docs)")]
    except sqlite3.OperationalError:
        return
    if "tags" not in cols:
        conn.execute("ALTER TABLE docs ADD COLUMN tags TEXT")


# ---------------- 文档级验证状态覆盖层（doc_verification 表，kb.sqlite3，ingest 建表） ----------------
# 人工补标（unverified → tested/expert）**必须存在独立表里，不能直接改 docs.extra**：
# ingest 是 INSERT OR REPLACE，下次构建会用 canonical（verification=unverified）覆盖回去，
# 人工认证就白做了（这正是"认证了但 w_rel 一直 0.5"的根因）。
# ingest 读本表并把结果并进 docs.extra.verification 与 w_rel；无行 = 沿用 canonical 自带值。
# 与 doc_tags 同构（覆盖层 + 入库合并）。

# 合法验证级别（与 config.confidence.verification_weights 的键一致）
VERIFICATION_LEVELS = ("unverified", "tested", "expert")


def _ensure_doc_verification(conn: sqlite3.Connection) -> None:
    """幂等建表（老库 / 审核页自建连接时）。"""
    try:
        conn.execute("SELECT 1 FROM doc_verification LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS doc_verification (
                 source_id TEXT PRIMARY KEY,
                 verification TEXT NOT NULL,
                 canonical_verification TEXT,
                 content_sha TEXT,
                 reviewer TEXT,
                 updated_at TEXT
               )"""
        )
    else:
        # 老行可能没有 canonical_verification 列（记下覆盖前的 canonical 值，
        # 撤销时要回退到它——否则撤销会读到"上次写入的覆盖值"而撤不掉）
        cols = [r[1] for r in conn.execute("PRAGMA table_info(doc_verification)")]
        if "canonical_verification" not in cols:
            conn.execute("ALTER TABLE doc_verification ADD COLUMN canonical_verification TEXT")
        if "content_sha" not in cols:
            # 内容指纹：认证时文档的内容 sha256。内容变更后旧认证不适用
            # （seed 清过期覆盖层重新入队；ingest 合并时跳过不匹配的覆盖——见 effective_overlay）
            conn.execute("ALTER TABLE doc_verification ADD COLUMN content_sha TEXT")


def _like_escape(s: str) -> str:
    """LIKE 通配符转义（配合 ESCAPE '\\'，防用户输入 %/_ 全表扫）。"""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    """表是否存在（旧库可能还没建覆盖层表）。"""
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None
    except sqlite3.OperationalError:
        return False


def load_doc_verification_conn(conn: sqlite3.Connection) -> dict[str, str]:
    """批量读取覆盖层：{source_id: verification}（兼容旧调用方/测试；ingest 用带指纹版）。"""
    return {sid: ov["verification"]
            for sid, ov in load_doc_verification_map_conn(conn).items()}


def load_doc_verification_map_conn(conn: sqlite3.Connection) -> dict[str, dict]:
    """批量读取覆盖层（含内容指纹）：{source_id: {"verification", "content_sha"}}。

    ingest 合并时用 `effective_overlay` 判定指纹是否仍匹配——不匹配 = 内容已变，
    旧认证不适用（按 canonical 入库，seed 会重新生成审核项）。
    """
    out: dict[str, dict] = {}
    _ensure_doc_verification(conn)
    try:
        rows = conn.execute(
            "SELECT source_id, verification, content_sha FROM doc_verification").fetchall()
    except sqlite3.OperationalError:
        return out  # 旧库尚无该表
    for sid, v, csha in rows:
        if v in VERIFICATION_LEVELS:
            out[sid] = {"verification": v, "content_sha": csha or ""}
    return out


def effective_overlay(overlay: Optional[dict], doc_asset_sha: Optional[str]) -> Optional[str]:
    """覆盖层是否仍适用：认证时记录的内容指纹与当前文档内容一致（或覆盖层无指纹=旧数据）。

    内容变了 → 返回 None（按 canonical 入库），由 seed_verification_pending 清过期
    覆盖层并重新生成审核项——"表格内容有变动需要重新审核"的执行点。
    """
    if not overlay:
        return None
    stamped = str(overlay.get("content_sha") or "")
    if stamped and doc_asset_sha and stamped != str(doc_asset_sha):
        return None
    return overlay.get("verification") or None


def get_doc_verification_conn(conn: sqlite3.Connection, source_id: str) -> dict:
    """单篇覆盖层（审核页展示"人工补标为 X"用）。"""
    _ensure_doc_verification(conn)
    row = conn.execute(
        "SELECT verification, canonical_verification, content_sha, reviewer, updated_at "
        "FROM doc_verification WHERE source_id=?",
        (source_id,),
    ).fetchone()
    if row is None:
        return {"source_id": source_id, "verification": None, "canonical_verification": None,
                "content_sha": None, "reviewer": None, "updated_at": None}
    return {"source_id": source_id, "verification": row[0], "canonical_verification": row[1],
            "content_sha": row[2], "reviewer": row[3], "updated_at": row[4]}


def set_doc_verification_conn(conn: sqlite3.Connection, source_id: str, verification: str,
                              reviewer: Optional[str] = None,
                              canonical: Optional[str] = None,
                              content_sha: Optional[str] = None) -> None:
    """写覆盖层（整体替换）。verification 必须是合法级别。

    `canonical` 只在**首次**覆盖时记录（覆盖前的 canonical 值），供撤销回退；
    已有覆盖行时不覆盖它——否则连续两次认证会把第一次的覆盖值当成 canonical 存下来。
    `content_sha` 每次认证都刷新为**当时**的内容指纹（重新认证 = 对新内容认证）。
    """
    if verification not in VERIFICATION_LEVELS:
        raise ValueError(f"非法验证级别: {verification}（支持 {VERIFICATION_LEVELS}）")
    _ensure_doc_verification(conn)
    row = conn.execute("SELECT canonical_verification FROM doc_verification WHERE source_id=?",
                       (source_id,)).fetchone()
    canon = row[0] if row is not None else canonical
    conn.execute(
        "INSERT OR REPLACE INTO doc_verification"
        "(source_id, verification, canonical_verification, content_sha, reviewer, updated_at) "
        "VALUES(?,?,?,?,?,?)",
        (source_id, verification, canon, content_sha or "", reviewer or "", _now()),
    )


def clear_doc_verification_conn(conn: sqlite3.Connection, source_id: str) -> Optional[str]:
    """撤销人工补标，返回**覆盖前的 canonical 值**（无覆盖行 → None，调用方保持原值）。"""
    _ensure_doc_verification(conn)
    row = conn.execute("SELECT canonical_verification FROM doc_verification WHERE source_id=?",
                       (source_id,)).fetchone()
    if row is None:
        return None
    conn.execute("DELETE FROM doc_verification WHERE source_id=?", (source_id,))
    return row[0]


def _apply_verification_conn(conn: sqlite3.Connection, source_id: str,
                             verification: Optional[str], reviewer: str,
                             confidence, *, clear: bool = False) -> dict:
    """apply_verification 的连接内实现（单连接批量复用）。

    认证时把文档**当前**内容指纹（extra.asset.sha256）写进覆盖层 content_sha——
    "内容变更需重新审核"的凭证；撤销（clear）回到覆盖前 canonical 值。
    """
    from .confidence import reliability_score

    row = conn.execute(
        "SELECT source_type, status, resolved_at, extra FROM docs WHERE source_id=?",
        (source_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"kb.sqlite3 中不存在文档 {source_id}")
    source_type, status, resolved_at, extra_raw = row
    extra = json.loads(extra_raw or "{}")
    cur_sha = str(((extra.get("asset") or {}).get("sha256")) or "")
    if clear:
        # 撤销：回到**覆盖前**的 canonical 值（存在覆盖层里；无覆盖行则保持原值）
        prev = clear_doc_verification_conn(conn, source_id)
        effective = prev if prev is not None else extra.get("verification", "")
    else:
        # 首次覆盖时把当前（canonical）值记进覆盖层，供撤销回退；内容指纹每次刷新
        set_doc_verification_conn(conn, source_id, verification or "", reviewer,
                                  canonical=extra.get("verification", ""),
                                  content_sha=cur_sha or None)
        effective = verification or ""
    extra["verification"] = effective
    rel = reliability_score(source_type, status, resolved_at, None, confidence,
                            kind=extra.get("kind", ""), verification=effective or None)
    conn.execute("UPDATE docs SET extra=?, reliability=? WHERE source_id=?",
                 (json.dumps(extra, ensure_ascii=False), rel, source_id))
    return {"source_id": source_id, "verification": effective, "reliability": rel}


def apply_verification(cfg: "AppConfig", source_id: str, verification: Optional[str],
                       reviewer: str = "", *, clear: bool = False) -> dict:
    """人工补标落库：写覆盖层 + **立即刷新 docs 行**（不必等下次构建）。

    刷新是必要的：查询期 `search._doc_meta_from_fts` 从 `docs.extra.verification` 取该值
    （向量路径也会用它补全 meta），不刷的话人工认证要等到下次 build_kb 才影响排序——
    正是"点了认证但 w_rel 还是 0.5"的观感。

    `docs.reliability` 按**查询期口径**重算（`compute_confidence` 不传 explicit_reliability，
    会现场按字段+kind+verification 重算），因此这里也用 explicit=None，与检索实际得分一致。
    **不动 content_hash**：下次 ingest 会因 meta_hash 不符走"仅刷新元数据"路径
    （embed_hash 未变 → 不重嵌），结果一致且自愈。
    """
    kb = cfg.resolve(cfg.storage.sqlite_path)
    if not kb.exists():
        raise ValueError(f"kb.sqlite3 不存在：{kb}（先跑 build_kb.py 入库）")
    conn = sqlite3.connect(str(kb))
    try:
        r = _apply_verification_conn(conn, source_id, verification, reviewer,
                                     cfg.confidence, clear=clear)
        conn.commit()
    finally:
        conn.close()
    return r


def apply_verification_many(cfg: "AppConfig", source_ids: list[str],
                            verification: Optional[str], reviewer: str, *,
                            clear: bool = False) -> dict:
    """批量人工补标（**文件级表格审核**用）：单连接逐行写覆盖层 + 刷新 docs。

    - 认证：每行覆盖层的 content_sha 记**该行当时**的内容指纹（同一文件的行共享文件
      sha；个别行缺 asset 也能处理）；
    - 撤销（clear=True）：逐行清除覆盖层、回 canonical；
    - 单行失败（文档已被删等）不阻断整批，计入 failed。
    """
    from .confidence import reliability_score  # noqa: F401  # _apply_verification_conn 内用

    if not clear and verification not in VERIFICATION_LEVELS:
        raise ValueError(f"非法验证级别: {verification}（支持 {VERIFICATION_LEVELS}）")
    kb = cfg.resolve(cfg.storage.sqlite_path)
    if not kb.exists():
        raise ValueError(f"kb.sqlite3 不存在：{kb}（先跑 build_kb.py 入库）")
    verified = 0
    failed: list[dict] = []
    conn = sqlite3.connect(str(kb))
    try:
        for sid in source_ids:
            try:
                _apply_verification_conn(conn, sid, verification, reviewer,
                                         cfg.confidence, clear=clear)
                verified += 1
            except ValueError as e:
                failed.append({"source_id": sid, "reason": str(e)})
        conn.commit()
    finally:
        conn.close()
    return {"verified": verified, "failed": failed}


def review_verification(cfg: "AppConfig", store: ReviewStore, item_id: int, action: str,
                        reviewer: str, result: Optional[dict] = None) -> dict:
    """补标审核动作（verification_pending 专用）：写覆盖层 + 刷新 docs + 记录审核项。

    action=approved → 覆盖层写 `tested`（`result.verification="expert"` 可升级为专家认证）；
    action=suspected → **清除覆盖层**（回到 unverified，可反悔；不是写 unverified，
    因为 canonical 可能自带更高值如 PDF 的 expert）。

    **文件级表格审核项**（payload.rows 聚合了同一文件的多行/多篇）：认证/存疑**批量**
    应用到全部行——一次审核整表生效；行内容指纹逐行记入覆盖层，表格内容变更后由
    seed 重新入队（见 seed_verification_pending）。
    """
    it = store.get_item(item_id)
    if it is None:
        raise ValueError(f"审核项不存在: {item_id}")
    if it["category"] != "verification_pending":
        raise ValueError(f"审核项 {item_id} 类别为 {it['category']}，不是补标项")
    rows = [r.get("source_id") for r in ((it["payload"] or {}).get("rows") or [])
            if isinstance(r, dict) and r.get("source_id")]
    if rows:
        # 文件级（表格）：批量应用到全部行
        if action == "approved":
            want = (result or {}).get("verification") or "tested"
            if want == "unverified":
                # 重置：清除人工覆盖层，生效值回 canonical（unverified）——认证撤销语义。
                # 不直接写 "unverified" 覆盖层：canonical 可能自带更高值（如 PDF expert），
                # 清层后按 canonical 生效与 seed 的过期清理逻辑一致。
                r = apply_verification_many(cfg, rows, None, reviewer, clear=True)
                store.reopen(item_id, reviewer,
                             result={"action": "verification", "verification": "unverified",
                                     "rows_count": len(rows), "rows_verified": r["verified"],
                                     "rows_failed": len(r["failed"]), **(result or {})})
                return {"ok": True, "source_id": rows[0], "aggregate": "excel_file",
                        "reopened": True, "rows_count": len(rows),
                        "rows_verified": r["verified"], "rows_failed": len(r["failed"]),
                        "failed": r["failed"]}
            elif want not in ("tested", "expert"):
                raise ValueError(f"approved 的 verification 只能是 tested/expert/unverified，收到 {want!r}")
            else:
                r = apply_verification_many(cfg, rows, want, reviewer)
        elif action == "suspected":
            r = apply_verification_many(cfg, rows, None, reviewer, clear=True)
        else:
            raise ValueError(f"非法标注动作: {action}（支持 approved | suspected）")
        store.review(item_id, action, reviewer,
                     result={"action": "verification", "verification": (result or {}).get("verification") or "tested",
                             "rows_count": len(rows), "rows_verified": r["verified"],
                             "rows_failed": len(r["failed"]), **(result or {})})
        return {"ok": True, "source_id": rows[0], "aggregate": "excel_file",
                "rows_count": len(rows), "rows_verified": r["verified"],
                "rows_failed": len(r["failed"]), "failed": r["failed"]}
    # 单文档项
    source_id = it["item_ref"]
    if action == "approved":
        want = (result or {}).get("verification") or "tested"
        if want == "unverified":
            r = apply_verification(cfg, source_id, None, reviewer, clear=True)
            store.reopen(item_id, reviewer,
                         result={"action": "verification", "verification": "unverified",
                                 **(result or {})})
            return {"ok": True, "source_id": source_id, "reopened": True, **r}
        elif want not in ("tested", "expert"):
            raise ValueError(f"approved 的 verification 只能是 tested/expert/unverified，收到 {want!r}")
        else:
            r = apply_verification(cfg, source_id, want, reviewer)
    elif action == "suspected":
        r = apply_verification(cfg, source_id, None, reviewer, clear=True)
    else:
        raise ValueError(f"非法标注动作: {action}（支持 approved | suspected）")
    store.review(item_id, action, reviewer,
                 result={"action": "verification", "verification": r["verification"],
                         "reliability": r["reliability"], **(result or {})})
    return {"ok": True, "source_id": source_id, **r}


# ---------------- 资产注册表（asset_registry，review.sqlite3，管理员侧） ----------------
# 表结构见顶部 _ASSET_REGISTRY_DDL（ReviewStore._SCHEMA 与 register_asset 共用）。

def _stem_of(source_id: str) -> str:
    """`pdf:x` / `md:x` → `x`；并剥掉自动消歧后缀 `--<sha8>`（sources 走 source_files.path_tag 加的）。

    消歧后 `md:same--d7878d1f` 的 stem 仍是 `same`，否则"同 stem 重名"提示会因
    后缀不同而永不触发（消歧把原本能触发的告警抵消掉了）。
    """
    stem = source_id.split(":", 1)[1] if ":" in source_id else source_id
    return re.sub(r"--[0-9a-f]{8}$", "", stem)


def register_assets(db_path: str | Path, items: list[tuple]) -> int:
    """批量注册资产映射（**单连接**，幂等 upsert），返回写入条数。

    items: [(rel_path, sha256, source_type[, size])]，asset_id 取 sha256 前 16 位
    （与 register_asset 一致）。逐条调 register_asset 会为每个资产开一次连接——
    图片动辄数百张，故提供批量入口（md/pdf/excel/图片资产统一走这里）。
    """
    if not items:
        return 0
    conn = sqlite3.connect(str(db_path))
    n = 0
    try:
        conn.executescript(_ASSET_REGISTRY_DDL)
        for it in items:
            rel, sha = it[0], (it[1] or "")
            stype = it[2] if len(it) > 2 else ""
            size = it[3] if len(it) > 3 else 0
            asset_id = sha[:16]
            if not asset_id or not rel:
                continue
            row = conn.execute(
                "SELECT sha256, size, source_type FROM asset_registry WHERE asset_id=?",
                (asset_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO asset_registry(asset_id, rel_path, sha256, size, source_type) "
                    "VALUES(?,?,?,?,?)",
                    (asset_id, rel, sha, int(size or 0), stype or ""),
                )
            else:
                conn.execute(
                    "UPDATE asset_registry SET rel_path=?, sha256=?, size=?, source_type=? "
                    "WHERE asset_id=?",
                    (rel, sha or row[0] or "", int(size or row[1] or 0),
                     stype or row[2] or "", asset_id),
                )
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def register_asset(db_path: str | Path, asset_id: str, rel_path: str,
                   sha256: str = "", size: int = 0, source_type: str = "") -> None:
    """注册单个资产映射（幂等 upsert，已有记录保留未提供的字段）。db_path=review.sqlite3。

    批量场景请用 `register_assets`（单连接）；本函数保留给单条调用与测试。
    """
    if not asset_id or not rel_path:
        return
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_ASSET_REGISTRY_DDL)
        row = conn.execute(
            "SELECT sha256, size, source_type FROM asset_registry WHERE asset_id=?",
            (asset_id,),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO asset_registry(asset_id, rel_path, sha256, size, source_type) "
                "VALUES(?,?,?,?,?)",
                (asset_id, rel_path, sha256 or "", int(size or 0), source_type or ""),
            )
        else:
            conn.execute(
                "UPDATE asset_registry SET rel_path=?, sha256=?, size=?, source_type=? WHERE asset_id=?",
                (rel_path, sha256 or row[0] or "", int(size or row[1] or 0),
                 source_type or row[2] or "", asset_id),
            )
        conn.commit()
    finally:
        conn.close()


def list_assets(db_path: str | Path) -> dict[str, dict]:
    """全部资产映射：{asset_id: {rel_path, sha256, size, source_type}}（审核页用）。"""
    out: dict[str, dict] = {}
    if not Path(db_path).exists():
        return out
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT asset_id, rel_path, sha256, size, source_type FROM asset_registry"
        ).fetchall()
    except sqlite3.OperationalError:
        return out
    finally:
        conn.close()
    for aid, rel, sha, size, stype in rows:
        out[aid] = {"rel_path": rel, "sha256": sha or "", "size": size or 0,
                    "source_type": stype or ""}
    return out


# ---------------- 文档标签视图（审核页数据层） ----------------

def doc_tags_view(kb_path: str | Path, source_id: str) -> dict:
    """文档标签视图：auto（自动快照）/ excluded / manual / final（合并结果）。"""
    conn = sqlite3.connect(f"file:{Path(kb_path).as_posix()}?mode=ro", uri=True)
    try:
        ov = get_doc_tags_conn(conn, source_id)
        row = conn.execute("SELECT tags FROM docs WHERE source_id=?", (source_id,)).fetchone()
        final = json.loads(row[0]) if row and row[0] else []
        auto = ov["auto_snapshot"]
        if not auto and row:
            auto = list(final)  # 旧库无快照：以最终标签为近似自动标签
        return {
            "source_id": source_id,
            "auto": auto,
            "excluded": ov["excluded"],
            "manual": ov["manual"],
            "final": final,
        }
    finally:
        conn.close()


def update_doc_tags(kb_path: str | Path, source_id: str, action: str, tag: str,
                    reviewer: str, config_path: Optional[str | Path] = None) -> dict:
    """审核页标签编辑：exclude（排除自动）/ restore（恢复排除）/ add（人工添加）/ remove（删除人工）。
    人工添加的标签不在词典时自动同步进词典（config.tags.registry）。"""
    from .tagging import TagRegistry, save_registry_to_config

    tag = (tag or "").strip()
    if not tag:
        raise ValueError("标签不能为空")
    conn = sqlite3.connect(str(kb_path))
    try:
        ov = get_doc_tags_conn(conn, source_id)
        excluded = list(ov["excluded"])
        manual = list(ov["manual"])
        if action == "exclude":
            if tag not in excluded:
                excluded.append(tag)
        elif action == "restore":
            excluded = [t for t in excluded if t != tag]
        elif action == "add":
            if tag not in manual:
                manual.append(tag)
        elif action == "remove":
            manual = [t for t in manual if t != tag]
        else:
            raise ValueError(f"未知标签动作: {action}")
        r = set_doc_tags_conn(conn, source_id, excluded=excluded, manual=manual, reviewer=reviewer)
        conn.commit()
    finally:
        conn.close()
    # 人工添加的新标签同步词典（config.tags.registry；审核页是唯一入口）
    if action == "add":
        try:
            registry = TagRegistry.load_from_config_file(config_path)
            if not registry.contains(tag):
                registry.add(tag)
                save_registry_to_config(_cfg_from_kb(kb_path), registry, config_path=config_path)
        except Exception as e:
            print(f"[review] 词典同步失败（标签已打标，下次 build_graph 后入图）: {e}")
    return r


def _cfg_from_kb(kb_path: str | Path):
    """从 kb 路径回推 AppConfig（审核页标签编辑时同步词典用；失败返回 None）。"""
    from .config import AppConfig

    try:
        return AppConfig.load(require_keys=False)
    except Exception:
        return None


# ---------------- 标签词典管理（registry，config.json 为唯一事实源） ----------------

def tag_dict(cfg: "AppConfig", kb_path: str | Path,
             config_path: Optional[str | Path] = None) -> dict:
    """词典视图：按 tier 分组 + 文档计数 + 统计（审核页标签管理 tab）。

    词典以配置文件为准（load_from_config_file——多次编辑后内存 cfg 可能滞后）。
    """
    from .tagging import TagRegistry

    registry = TagRegistry.load_from_config_file(config_path)
    counts: dict[str, int] = {}
    if Path(kb_path).exists():
        conn = sqlite3.connect(f"file:{Path(kb_path).as_posix()}?mode=ro", uri=True)
        try:
            try:
                tag_rows = conn.execute(
                    "SELECT tags FROM docs WHERE tags IS NOT NULL"
                ).fetchall()
            except sqlite3.OperationalError:
                tag_rows = []  # 旧库无 tags 列
            for (tags_json,) in tag_rows:
                try:
                    for t in json.loads(tags_json or "[]"):
                        counts[t] = counts.get(t, 0) + 1
                except (TypeError, json.JSONDecodeError):
                    continue
        finally:
            conn.close()
    groups = {"domain": [], "purpose": []}
    for e in sorted(registry.entries, key=lambda x: (x.tier, x.name)):
        groups.setdefault(e.tier, []).append({"name": e.name, "docs": counts.get(e.name, 0)})
    return {
        "groups": groups,
        "stats": {
            "domain": len(groups["domain"]),
            "purpose": len(groups["purpose"]),
            "tagged_docs": sum(1 for c in counts.values() if c > 0) if False else len(counts),
        },
        "note": "词典为 config.json tags.registry 唯一事实源；新增/改名/改 tier 同步配置，重建图后入图",
    }


def add_tag_to_registry(cfg: "AppConfig", name: str, tier: Optional[str] = None,
                        config_path: Optional[str | Path] = None) -> dict:
    """新增词典标签（同步 config.json）。以配置文件为准（内存 cfg 可能滞后）。"""
    from .tagging import TagRegistry, save_registry_to_config

    registry = TagRegistry.load_from_config_file(config_path)
    entry = registry.add(name, tier)
    save_registry_to_config(cfg, registry, config_path=config_path)
    return {"name": entry.name, "tier": entry.tier}


def rename_tag(cfg: "AppConfig", old: str, new: str, kb_path: str | Path,
               config_path: Optional[str | Path] = None) -> dict:
    """词典改名：registry + 全库替换 docs.tags / doc_tags.excluded+manual。返回受影响文档数。"""
    from .tagging import TagRegistry, save_registry_to_config

    registry = TagRegistry.load_from_config_file(config_path)
    if not registry.rename(old, new):
        raise ValueError(f"词典中不存在标签 {old}")
    save_registry_to_config(cfg, registry, config_path=config_path)
    n = 0
    if Path(kb_path).exists():
        conn = sqlite3.connect(str(kb_path))
        try:
            n += _replace_json_list_tag(conn, "docs", "tags", old, new)
            n += _replace_json_list_tag(conn, "doc_tags", "excluded", old, new)
            n += _replace_json_list_tag(conn, "doc_tags", "manual", old, new)
            conn.commit()
        finally:
            conn.close()
    return {"renamed": {"old": old, "new": new}, "docs_updated": n}


def _replace_json_list_tag(conn: sqlite3.Connection, table: str, column: str,
                           old: str, new: str) -> int:
    """把 table.column（JSON 数组）中的 old 替换为 new（同名去重）。返回受影响行数。"""
    changed = 0
    try:
        rows = conn.execute(f"SELECT rowid, {column} FROM {table} WHERE {column} IS NOT NULL").fetchall()
    except sqlite3.OperationalError:
        return 0
    for rowid, raw in rows:
        if not raw:
            continue
        try:
            vals = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if old not in vals:
            continue
        vals = [new if v == old else v for v in vals]
        seen, dedup = set(), []
        for v in vals:
            if v not in seen:
                seen.add(v)
                dedup.append(v)
        conn.execute(f"UPDATE {table} SET {column}=? WHERE rowid=?", (json.dumps(dedup, ensure_ascii=False), rowid))
        changed += 1
    return changed


def set_tag_tier(cfg: "AppConfig", name: str, tier: str,
                 config_path: Optional[str | Path] = None) -> dict:
    """修改标签层级（domain/purpose，全局生效；tier 是标签固有属性）。"""
    from .tagging import TagRegistry, save_registry_to_config

    if tier not in ("domain", "purpose"):
        raise ValueError(f"非法 tier: {tier}")
    registry = TagRegistry.load_from_config_file(config_path)
    if not registry.set_tier(name, tier):
        raise ValueError(f"词典中不存在标签 {name}")
    save_registry_to_config(cfg, registry, config_path=config_path)
    return {"name": name, "tier": tier}


def delete_tag(cfg: "AppConfig", name: str,
               config_path: Optional[str | Path] = None) -> dict:
    """词典删除：仅移出词典（不动已打标文档上的该标签）。"""
    from .tagging import TagRegistry, save_registry_to_config

    registry = TagRegistry.load_from_config_file(config_path)
    if not registry.remove(name):
        raise ValueError(f"词典中不存在标签 {name}")
    save_registry_to_config(cfg, registry, config_path=config_path)
    return {"deleted": name}


# ---------------- 审核项生成（seed，幂等） ----------------

def _excel_file_of(source_id: str) -> str:
    """excel source_id → 文件身份。

    当前格式 `excel:<文件名>`（一个文件一篇）；旧逐行格式 `excel:<文件名>:<sheet>:<行>`
    取文件段——同一登记表的新旧行聚合到同一个文件级审核项。
    """
    parts = source_id.split(":")
    return parts[1] if len(parts) > 1 else source_id


def seed_verification_pending(cfg: "AppConfig", store: ReviewStore) -> int:
    """扫描 kb.sqlite3 中**生效** verification=unverified 的文档，生成补标审核项（幂等）。

    读的是生效值（`docs.extra.verification`，ingest 已把人工补标覆盖层合并进去），
    所以补标过的文档不会再进队列。LIKE 只作**预筛**（键名不受 json 分隔符影响），
    真正判定用 JSON 精确比对——否则 json.dumps 的格式一变，这里会静默地什么都捞不到。

    **表格按文件粒度**：doc_excel 的待补标文档按源文件聚成**一条**审核项
    （item_ref = `excelfile:<文件名>:<sha8>`）——认证一次应用到文件内全部行，
    不逐行审；`payload.rows` 带行清单供批量应用与展示。

    **内容变更重审**：认证时的内容指纹记录在覆盖层 content_sha；文档内容变更后
    旧认证不适用——这里清掉过期覆盖层（docs 回 canonical），文档随即进入下面的
    待补标集合重新入队。item_ref 带 sha8，内容变了 = 新 ref，dedupe 挡不住新内容。

    **兼容**：旧版逐行格式的 pending excel 项（`excel:文件:sheet:行`）被文件级项取代，
    自动清理（已处理的保留作历史）。
    """
    from .confidence import reliability_score

    kb = cfg.resolve(cfg.storage.sqlite_path)
    if not kb.exists():
        return 0
    conn = sqlite3.connect(str(kb))   # 可写：清过期覆盖层要改 docs/覆盖层
    added = 0
    try:
        _ensure_doc_verification(conn)
        docs_cols = {r[1] for r in conn.execute("PRAGMA table_info(docs)")}
        # 最小测试库/旧库可能缺列：缺 source_type 按非表格处理，缺 reliability 不刷该列
        has_stype = "source_type" in docs_cols
        sel_cols = "source_id, title, url, extra" if not has_stype else \
            "source_id, source_type, title, url, extra"
        # ---- 过期覆盖层：内容指纹与当前文档不符 → 清除 + docs 回 canonical ----
        cur_sha: dict[str, str] = {}
        for sid, extra_raw in conn.execute("SELECT source_id, extra FROM docs").fetchall():
            try:
                ex = json.loads(extra_raw or "{}")
            except ValueError:
                continue
            cur_sha[sid] = str((((ex.get("asset") or {}).get("sha256"))) or "")
        stale: dict[str, str] = {}
        for sid, _v, canonical, csha in conn.execute(
                "SELECT source_id, verification, canonical_verification, content_sha "
                "FROM doc_verification").fetchall():
            if csha and cur_sha.get(sid, "") and csha != cur_sha[sid]:
                conn.execute("DELETE FROM doc_verification WHERE source_id=?", (sid,))
                stale[sid] = canonical or "unverified"
        if stale:
            reset_cols = ["extra"] + [c for c in ("source_type", "status", "resolved_at")
                                      if c in docs_cols]
            for sid, canon in stale.items():
                row = conn.execute(
                    f"SELECT {','.join(reset_cols)} FROM docs WHERE source_id=?",
                    (sid,)).fetchone()
                if row is None:
                    continue
                vals = dict(zip(reset_cols, row))
                try:
                    ex = json.loads(vals.get("extra") or "{}")
                except ValueError:
                    ex = {}
                ex["verification"] = canon
                rel = reliability_score(vals.get("source_type") or "",
                                        vals.get("status") or "",
                                        vals.get("resolved_at"), None, cfg.confidence,
                                        kind=ex.get("kind", ""), verification=canon or None)
                if "reliability" in docs_cols:
                    conn.execute("UPDATE docs SET extra=?, reliability=? WHERE source_id=?",
                                 (json.dumps(ex, ensure_ascii=False), rel, sid))
                else:
                    conn.execute("UPDATE docs SET extra=? WHERE source_id=?",
                                 (json.dumps(ex, ensure_ascii=False), sid))
            conn.commit()
        # ---- 待补标集合：单文档逐条；表格按文件聚合 ----
        groups: dict[tuple[str, str], dict] = {}
        for row in conn.execute(f"SELECT {sel_cols} FROM docs").fetchall():
            source_id, title, url, extra_raw = (row[0], row[1], row[2], row[3]) if not has_stype \
                else (row[0], row[2], row[3], row[4])
            stype = row[1] if has_stype else ""
            try:
                ex = json.loads(extra_raw or "{}")
            except ValueError:
                continue
            if ex.get("verification") != "unverified":
                continue
            sha = str((((ex.get("asset") or {}).get("sha256"))) or "")
            if has_stype and stype == "doc_excel":
                key = (_excel_file_of(source_id), sha)
                g = groups.setdefault(key, {"title": title or key[0], "url": url or "",
                                            "rows": []})
                g["rows"].append({"source_id": source_id, "title": title or source_id})
            else:
                if store.add_item("verification_pending", source_id,
                                  {"source_id": source_id, "title": title, "url": url,
                                   "content_sha": sha}):
                    added += 1
        for (file_key, sha), g in sorted(groups.items()):
            ref = f"excelfile:{file_key}:{sha[:8]}" if sha else f"excelfile:{file_key}"
            payload = {"source_id": g["rows"][0]["source_id"], "title": g["title"],
                       "url": g["url"], "file": file_key, "file_sha": sha,
                       "aggregate": "excel_file", "row_count": len(g["rows"]),
                       "rows": g["rows"][:200]}  # 展示/批量上限，超出走文件身份全量应用
            if store.add_item("verification_pending", ref, payload):
                added += 1
    finally:
        conn.close()
    # 旧逐行 pending 项被文件级项取代（已处理的保留作历史）
    removed = store.cleanup_legacy_row_items()
    if removed:
        print(f"[review] 清理旧版逐行 excel 审核项 {removed} 条（已由文件级项取代）")
    return added


_CASE_TITLE_FLAG_RE = None


def seed_case_title_flags(cfg: "AppConfig", store: ReviewStore) -> int:
    """案例标题含"待审核"/"待修改" → 生成待确认审核项（幂等）。"""
    import re

    global _CASE_TITLE_FLAG_RE
    if _CASE_TITLE_FLAG_RE is None:
        _CASE_TITLE_FLAG_RE = re.compile(r"待审核|待修改")
    kb = cfg.resolve(cfg.storage.sqlite_path)
    if not kb.exists():
        return 0
    conn = sqlite3.connect(f"file:{kb.as_posix()}?mode=ro", uri=True)
    added = 0
    try:
        rows = conn.execute("SELECT source_id, title, url FROM docs").fetchall()
        for source_id, title, url in rows:
            if title and _CASE_TITLE_FLAG_RE.search(title):
                if store.add_item("case_title_flag", source_id,
                                  {"source_id": source_id, "title": title, "url": url}):
                    added += 1
    finally:
        conn.close()
    return added


def seed_tag_candidates(cfg: "AppConfig", store: ReviewStore) -> int:
    """扫描 kb.sqlite3 中 extra.tag_candidates（未收录强候选）→ tag_candidate 审核项（幂等）。

    **按候选词聚合**：同一候选词在多篇文档出现只生成一条审核项（item_ref = `tag:{name}`），
    payload 带提及文档数 + 文档列表——人工审一次 = 全部提及文档生效。
    词典已收录的候选不生成；忽略/认证过的候选因 item_ref 已存在（任意状态）不再重复打扰。

    兼容升级：旧版按 (source_id::name) 生成，升级后自动清理旧格式残留（其词已收录/忽略的
    不重建，未处理的按新聚合格式重建）。
    """
    from .tagging import TagRegistry, normalize_tag, tag_key

    kb = cfg.resolve(cfg.storage.sqlite_path)
    if not kb.exists():
        return 0
    # 清理旧格式（source_id::name）残留：item_ref 含 "::" 且非 "tag:" 前缀
    store.cleanup_old_tag_candidates()
    registry = TagRegistry.load(cfg)
    conn = sqlite3.connect(f"file:{kb.as_posix()}?mode=ro", uri=True)
    # key -> {display: 首见原形, docs: [(source_id, title)], tier}
    # 聚合键 tag_key（大小写/分隔符/CJK 空格归一）：存量 extras 里 "Atlas A3"/"atlas_a3"、
    # "超时 排查"/"超时排查" 是同一词的排版变体——收敛为一条候选；display 保留首见
    # normalize 形式（专有名词可读性），采纳入词典时 add() 同 key 收敛到已有词条。
    agg: dict[str, dict] = {}
    try:
        rows = conn.execute("SELECT source_id, title, extra FROM docs").fetchall()
        for source_id, title, extra in rows:
            try:
                ex = json.loads(extra or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            cands = ex.get("tag_candidates") or []
            for c in cands:
                raw = c.get("name") if isinstance(c, dict) else str(c)
                display = normalize_tag(raw)
                key = tag_key(raw or "")
                if not key or not display or registry.contains(display):
                    continue
                tier = c.get("tier") if isinstance(c, dict) else None
                entry = agg.setdefault(key, {"display": display, "docs": [], "tier": tier})
                entry["docs"].append((source_id, title or source_id))
                if tier and not entry["tier"]:
                    entry["tier"] = tier
    finally:
        conn.close()
    added = 0
    for key, info in sorted(agg.items()):
        docs = info["docs"]
        if store.add_item("tag_candidate", f"tag:{key}", {
            "candidate": info["display"],
            "suggested_tier": info["tier"],
            "doc_count": len(docs),
            "docs": [{"source_id": s, "title": t} for s, t in docs[:20]],  # 展示用，截断
            # 注意：不再写首文档 source_id/title 兼容字段——曾把列表标题"压"成首文档标题
            #（多手册同产品名时 4 个不同候选全显示同一个名字）；旧条目的兼容读取在
            # adopt/showItem 的 docs||[payload] 回退里，新条目一律有 docs。
        }):
            added += 1
    # 存量变体合并（tag_key 引入后的一次性收敛，幂等）：旧 item_ref=tag:{normalize原形}，
    # 同 key 的 "Atlas A3"/"atlas_a3" 等旧条目不会因 add_item 去重自动合并——
    # 同 key 组内保留最早 id，docs 并入首条，其余标记 approved（merged_variant）。
    merged_variants = _merge_tag_candidate_variants(store)
    if merged_variants:
        print(f"[candidates] 变体合并: {merged_variants} 条旧条目并入同 key 首条", flush=True)
    # 以 kb 现状校正 pending 队列（2026-10 用户实测缺口）：任何删库路径（审核台删除/
    # canonical 删行 + rebuild/换库）之后，kb 里文档没了但队列项会永久滞留 pending——
    # 全悬空自动忽略、部分悬空刷新引用，下次 seed（审核台启动）自动自愈。
    ignored, refreshed = _sweep_tag_candidates(store, agg, registry)
    if ignored or refreshed:
        print(f"[candidates] 队列校正: 自动忽略 {ignored} 条（提及文档已不存在/词典已收录）、"
              f"刷新 {refreshed} 条的引用", flush=True)
    return added


def _merge_tag_candidate_variants(store: "ReviewStore") -> int:
    """存量 tag_candidate 按 tag_key 合并变体（幂等）：同 key 组内保留最早 id（队列位置
    最老、可能已被人工看到），其余条目的 docs/doc_count 并入首条后标记 approved
    （result.action=merged_variant，审计可查）；approved/suspected 历史条目不动。
    返回被合并掉的条数。"""
    from .tagging import tag_key

    groups: dict[str, list[dict]] = {}
    for it in store.list_items(category="tag_candidate", status="pending"):
        ref = it["item_ref"] or ""
        if not ref.startswith("tag:"):
            continue
        groups.setdefault(tag_key(ref[4:]), []).append(it)
    merged = 0
    for items in groups.values():
        if len(items) < 2:
            continue
        items.sort(key=lambda i: i["id"])
        keep, rest = items[0], items[1:]
        kp = dict(keep["payload"] or {})
        kdocs = {d.get("source_id"): d for d in (kp.get("docs") or [])
                 if isinstance(d, dict) and d.get("source_id")}
        for it in rest:
            p = it["payload"] or {}
            for d in (p.get("docs") or []):
                if isinstance(d, dict) and d.get("source_id") and d["source_id"] not in kdocs:
                    kdocs[d["source_id"]] = d
            store.review(it["id"], "approved", "system",
                         result={"action": "merged_variant", "into_item": keep["id"],
                                 "candidate": p.get("candidate", "")})
            merged += 1
        kp["docs"] = list(kdocs.values())[:20]
        kp["doc_count"] = len(kdocs)
        store.update_payload(keep["id"], kp)
    return merged


def _sweep_tag_candidates(store: "ReviewStore", agg: dict[str, dict],
                          registry: "TagRegistry") -> tuple[int, int]:
    """以 kb 现状校正 pending tag_candidate 队列。只动 pending（已审项是历史记录）。

    - 词已无任何 kb 文档提及（agg 无此词）→ 自动忽略（approved，result 记原因）；
    - 词已在词典（人工加过，agg 因此不含）→ 自动忽略（词已生效，候选无意义）；
    - 词仍在 kb 但提及文档变了 → payload.docs/doc_count 刷新为现状（引用修正）。
    返回 (auto_ignored, refreshed)。
    """
    ignored = refreshed = 0
    from .tagging import tag_key

    for it in store.list_items(category="tag_candidate", status="pending"):
        p = it["payload"] or {}
        name = p.get("candidate", "")
        if not name:
            continue
        # agg 键是 tag_key（聚合键）——候选原形（payload.candidate 首见形式）须过同款归一再查
        fresh = agg.get(tag_key(name))
        if fresh is None:
            reason = ("词典已收录，候选已生效" if registry.contains(name)
                      else "提及文档已不存在（重建后 kb 中不再提及）")
            store.review(it["id"], "approved", "system",
                         result={"action": "auto-ignore", "reason": reason})
            ignored += 1
            continue
        docs = [{"source_id": s, "title": t} for s, t in fresh["docs"][:20]]
        cur_ids = [d.get("source_id") for d in (p.get("docs") or [])]
        if cur_ids != [d["source_id"] for d in docs]:
            p["docs"] = docs
            p["doc_count"] = len(fresh["docs"])
            store.update_payload(it["id"], p)
            refreshed += 1
    return ignored, refreshed


def adopt_tag_candidate(cfg: "AppConfig", store: ReviewStore, item_id: int,
                        reviewer: str, tier: Optional[str] = None,
                        config_path: Optional[str | Path] = None) -> dict:
    """采纳候选：① 入词典（config.tags.registry，tier 取审核选择或启发式）
    ② 写入提及该候选的**全部文档** manual（立即生效，无需等下次入库）；③ 审核项标记 approved。

    聚合后一条审核项对应多篇提及文档（payload.docs）；兼容旧数据单文档（payload.source_id）。
    """
    from .tagging import TagRegistry, save_registry_to_config

    it = store.get_item(item_id)
    if it is None:
        raise ValueError(f"审核项不存在: {item_id}")
    if it["category"] != "tag_candidate":
        raise ValueError(f"不是 tag_candidate 审核项: {it['category']}")
    payload = it["payload"] or {}
    source_id = payload.get("source_id", "")
    name = payload.get("candidate", "")
    if not name:
        raise ValueError("审核项缺少 candidate")
    registry = TagRegistry.load_from_config_file(config_path)
    # add() 同 key 收敛：候选是词典词条的排版变体（"atlas_a3" vs 词典 "Atlas A3"）时
    # 返回已有词条——manual 写词典正式名（标签体系统一），不产生第二个词条。
    entry = registry.add(name, tier)  # tier=None → 启发式；已有词条 tier 仅显式覆盖
    tag_name = entry.name
    save_registry_to_config(cfg, registry, config_path=config_path)
    # 提及该候选的全部文档（聚合）；旧单文档审核项回退到 source_id
    docs = [d["source_id"] for d in (payload.get("docs") or []) if d.get("source_id")]
    if not docs and source_id:
        docs = [source_id]
    # 存在性过滤：候选的 docs 是入库时快照，源文档可能已从 kb 删除——不过滤会给
    # 不存在的 sid 写孤儿 doc_tags 覆盖层（doc_tags 按 source_id 无外键）。
    kb = cfg.resolve(cfg.storage.sqlite_path)
    existing: set[str] = set()
    if docs and Path(kb).exists():
        conn = sqlite3.connect(str(kb))
        try:
            qmarks = ",".join("?" * len(docs))
            existing = {r[0] for r in conn.execute(
                f"SELECT source_id FROM docs WHERE source_id IN ({qmarks})", docs).fetchall()}
        finally:
            conn.close()
    missing = [s for s in docs if s not in existing]
    conn = sqlite3.connect(str(kb))
    try:
        for sid in docs:
            if sid not in existing:
                continue
            ov = get_doc_tags_conn(conn, sid)
            manual = list(ov["manual"])
            if tag_name not in manual:
                manual.append(tag_name)
            set_doc_tags_conn(conn, sid, manual=manual, reviewer=reviewer)
        conn.commit()
    finally:
        conn.close()
    store.review(item_id, "approved", reviewer,
                 result={"action": "adopt", "tag": tag_name, "tier": entry.tier,
                         **({"variant_of": tag_name} if tag_name != name else {}),
                         "docs_count": len(existing),
                         **({"docs_missing": missing} if missing else {})})
    return {"ok": True, "tag": tag_name, "tier": entry.tier, "source_id": source_id,
            "docs_count": len(existing), "docs_missing": len(missing)}


def prune_tag_candidates_for_doc(review_db: str | Path, source_id: str) -> dict:
    """删除文档后修剪 pending tag_candidate 项的**悬空引用**（用户确认缺口 2026-10）。

    此前链路：磁盘文件删除 → kb 记录不自动删（防误删清库，设计如此）→ 人工删 kb 记录后
    队列项不级联清理 → payload.docs 悬空引用残留；全部提及文档删光的项永久滞留 pending
    （seed_tag_candidates 幂等：item_ref 已存在不重建也不清理）→ 采纳时给不存在的 sid
    写孤儿 doc_tags。

    修剪规则（只动 pending 项；已审项是历史记录不改）：
    - payload.docs 移除该 source_id（doc_count 同步减）；
    - 修剪后快照为空 → 项自动标记 approved（= 忽略，result 记原因）——词典不受影响，
      若其他文档日后重新提及，seed 会因 item_ref 已存在而…… 不，approved 后 add_item
      幂等也不会重建；极端场景（删光后又想采纳）走文档管理页人工添加即可。
    返回 {"pruned": 引用修剪的项数, "auto_ignored": 自动忽略的项数}。
    """
    conn = sqlite3.connect(str(review_db))
    n_pruned = n_ignored = 0
    try:
        rows = conn.execute(
            "SELECT id, payload FROM review_items "
            "WHERE category='tag_candidate' AND status='pending'").fetchall()
        for iid, raw in rows:
            try:
                p = json.loads(raw or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            docs = [d for d in (p.get("docs") or []) if isinstance(d, dict)]
            hit = any(d.get("source_id") == source_id for d in docs)
            if not hit:
                # 旧单文档格式（payload.source_id，无 docs）兼容
                if p.get("source_id") != source_id:
                    continue
                docs = [{"source_id": source_id}]
                hit = True
            new_docs = [d for d in docs if d.get("source_id") != source_id]
            p["docs"] = new_docs
            p["doc_count"] = max(0, int(p.get("doc_count") or len(docs)) - 1)
            if new_docs:
                conn.execute("UPDATE review_items SET payload=? WHERE id=?",
                             (json.dumps(p, ensure_ascii=False), iid))
                n_pruned += 1
            else:
                conn.execute(
                    "UPDATE review_items SET status='approved', reviewed_at=?, reviewer=?, "
                    "result=?, payload=? WHERE id=?",
                    (_now(), "system",
                     json.dumps({"action": "auto-ignore",
                                 "reason": f"提及文档已删除（{source_id}），无剩余提及文档"},
                                ensure_ascii=False),
                     json.dumps(p, ensure_ascii=False), iid))
                n_ignored += 1
        conn.commit()
    finally:
        conn.close()
    return {"pruned": n_pruned, "auto_ignored": n_ignored}


def seed_low_confidence_ocr(cfg: "AppConfig", store: ReviewStore) -> int:
    """扫描 kb.sqlite3 的 `extra.evidence[].ocr` → 低置信/自报异常图片审核项（幂等）。

    只针对**未进正文**的图片 OCR（`text_included=false`）——即置信度低于
    `ocr_min_confidence`，或模型自报置信度异常（缺失/不可解析/越界/格式污染）。
    **按图片 sha256 聚合**（item_ref = `ocr:{sha256前16位}`）：同一张图被多篇文档引用只生成
    一条审核项，人工核对一次即可。reason 区分 low（分低）/ anomaly（自报异常）/
    no_confidence（无置信度）。

    已认证/存疑/删除过的项因 item_ref 已存在（任意状态）不再重复打扰。
    """
    kb = cfg.resolve(cfg.storage.sqlite_path)
    if not kb.exists():
        return 0
    # 图片资产反查（asset_id → rel_path，管理员侧）：让审核台能直接预览原图
    assets = list_assets(cfg.resolve(cfg.storage.review_path))
    conn = sqlite3.connect(f"file:{kb.as_posix()}?mode=ro", uri=True)
    added = 0
    try:
        rows = conn.execute("SELECT source_id, title, url, extra FROM docs").fetchall()
        for source_id, title, url, extra in rows:
            try:
                ex = json.loads(extra or "{}")
            except Exception:
                continue
            for ev in (ex.get("evidence") or []):
                if not isinstance(ev, dict):
                    continue
                ocr = ev.get("ocr")
                if not isinstance(ocr, dict) or ocr.get("text_included"):
                    continue
                sha = str(ev.get("sha256", "") or "")
                ref = f"ocr:{sha[:16]}" if sha else f"ocr:{source_id}"
                conf = ocr.get("confidence")
                anomaly = str(ocr.get("anomaly", "") or "")
                if anomaly:
                    reason = "anomaly"
                elif conf is None:
                    reason = "no_confidence"
                else:
                    reason = "low"
                aid = str(ev.get("asset_id", "") or "")
                if store.add_item("low_confidence_ocr", ref, {
                    "source_id": source_id, "title": title, "url": url,
                    "asset_id": aid, "sha256": sha,
                    "kind": ev.get("kind", ""), "reason": reason,
                    "confidence": conf,
                    "confidence_source": ocr.get("confidence_source", ""),
                    "anomaly": anomaly,
                    "signatures": ocr.get("signatures") or [],
                    # 审核侧路径（管理员可见）：assets 静态挂载在审核端口，可直接预览
                    "asset_path": (assets.get(aid) or {}).get("rel_path", ""),
                }):
                    added += 1
    finally:
        conn.close()
    return added


def sweep_stale_verification_pending(store: "ReviewStore", kb_path: str | Path) -> int:
    """verification_pending 兜底 sweep（2026-10 用户实测缺口）：引用文档已从 kb 删除的
    pending 补标项自动忽略——此前 _sweep_tag_candidates 只覆盖 tag_candidate。

    item_ref = 文档 source_id（单冒号形态）；表格等文件级聚合项（excelfile: 前缀/
    多冒号 item_ref）不动。每次 seed（审核台启动）执行。返回自动忽略条数。
    """
    conn = sqlite3.connect(f"file:{Path(kb_path).as_posix()}?mode=ro", uri=True)
    try:
        kb_docs = {r[0] for r in conn.execute("SELECT source_id FROM docs").fetchall()}
    finally:
        conn.close()
    ignored = 0
    for it in store.list_items(category="verification_pending", status="pending"):
        ref = it["item_ref"]
        if ref.count(":") != 1 or ref in kb_docs:
            continue  # 非 doc sid 形态（表格/文件级聚合项）或文档仍在库
        store.review(it["id"], "approved", "system",
                     result={"action": "auto-ignore", "reason": "引用文档已删除（kb 中不存在）"})
        ignored += 1
    return ignored


def normalize_tag_variants(cfg: "AppConfig", config_path: Optional[str | Path] = None) -> dict:
    """已认证 tag 的同义变体归一（幂等，2026-10）：存量词典里同一 tag_key 的多个词条
    （"CANN版本说明" 与 "CANN 版本说明" 都被人工认证入典）收敛为首条原形，并对全库做
    标签替换（docs.tags / doc_tags.manual+excluded）——**已认证的 tag 同样适用 tag_key
    规则**，不需要删掉重加。变体组为空时零开销（每次 seed 前跑）。

    返回 {"merged_variants": 收敛对数, "docs_updated": 全库替换受影响文档数}。
    """
    import json as _json

    from .config import PROJECT_ROOT
    from .tagging import tag_key

    p = Path(config_path) if config_path else PROJECT_ROOT / "config.json"
    if not p.exists():
        return {"merged_variants": 0, "docs_updated": 0}
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return {"merged_variants": 0, "docs_updated": 0}
    raw_entries = (data.get("tags") or {}).get("registry") or []
    names: list[str] = []
    for item in raw_entries:
        name = (item.get("name") if isinstance(item, dict) else str(item)) or ""
        if name.strip():
            names.append(name.strip())
    # 按 tag_key 分组找变体对（保持 config 出现顺序；组内首条 = canonical）
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(tag_key(n), []).append(n)
    variants = [(g[1], g[0]) for g in groups.values() if len(g) > 1]
    if not variants:
        return {"merged_variants": 0, "docs_updated": 0}
    variant_names = {old for old, _ in variants}
    # 词典层：直接从 config 删变体词条（TagRegistry 加载/contains/add 均按 key 收敛，
    # 词典不留变体；load_from_config_file 已收敛所以这里不能用 rename——找不到旧词条）
    def _keep(item) -> bool:
        name = (item.get("name") if isinstance(item, dict) else str(item)) or ""
        return name.strip() not in variant_names

    data["tags"]["registry"] = [e for e in raw_entries if _keep(e)]
    p.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    merged_docs = 0
    kb_path = cfg.resolve(cfg.storage.sqlite_path)
    for old, new in variants:
        if Path(kb_path).exists():
            conn = sqlite3.connect(str(kb_path))
            try:
                n = (_replace_json_list_tag(conn, "docs", "tags", old, new)
                     + _replace_json_list_tag(conn, "doc_tags", "excluded", old, new)
                     + _replace_json_list_tag(conn, "doc_tags", "manual", old, new))
                conn.commit()
            finally:
                conn.close()
            merged_docs += n
    return {"merged_variants": len(variants), "docs_updated": merged_docs}


def seed_all(cfg: "AppConfig", store: ReviewStore) -> dict[str, int]:
    """运行全部 seed（幂等），返回各 seed 新增数。"""
    # 已认证 tag 的变体归一先于一切 seed：词典先收敛，候选的"词典已收录"判定才准确
    norm = normalize_tag_variants(cfg)
    if norm.get("merged_variants"):
        print(f"[seed] 词典变体归一: {norm['merged_variants']} 组同 key 词条收敛，"
              f"全库替换影响 {norm['docs_updated']} 行", flush=True)
    out = {
        "verification_pending": seed_verification_pending(cfg, store),
        "case_title_flag": seed_case_title_flags(cfg, store),
        "low_confidence_ocr": seed_low_confidence_ocr(cfg, store),
        "tag_candidate": seed_tag_candidates(cfg, store),
    }
    # 引用文档已删除的 pending 补标项自动忽略（verification_pending 侧的队列自愈）
    swept = sweep_stale_verification_pending(store, cfg.resolve(cfg.storage.sqlite_path))
    if swept:
        print(f"[seed] verification_pending 队列校正: 自动忽略 {swept} 条（引用文档已删除）",
              flush=True)
    out["verification_pending_swept"] = swept
    return out


# ---------------- API 配置中心（key 脱敏） ----------------

def api_configs(cfg: "AppConfig") -> list[dict]:
    """集中展示所有 API 配置状态（**key 一律脱敏**，只显示已配置/未配置）。"""
    out: list[dict] = []
    # embedding
    e = cfg.embedding
    out.append({
        "name": "embedding",
        "provider": e.provider,
        "base_url": e.base_url or "",
        "model": e.model or "",
        "key_configured": bool(e.effective_api_key),
        "status": "configured" if (e.provider == "echo" or e.effective_api_key) else "missing",
        "note": "echo=离线占位（无语义）；openai_compatible 需 base_url + key",
    })
    # OCR（image source）
    for sc in cfg.effective_sources():
        if sc.type == "image":
            api_base = str(sc.get("ocr_api_base", "") or "")
            key = str(sc.get("ocr_api_key", "") or os.environ.get("OCR_API_KEY", ""))
            out.append({
                "name": "ocr",
                "provider": str(sc.get("ocr_provider", "ask")),
                "base_url": api_base,
                "model": str(sc.get("ocr_api_model", "") or ""),
                "mode": str(sc.get("ocr_api_mode", "custom") or "custom"),
                "min_confidence": str(sc.get("ocr_min_confidence", "") or ""),
                "key_configured": bool(key),
                "status": "configured" if (api_base or sc.get("ocr_provider") == "paddle") else "ask",
                "note": "ask=无 API 时询问本地/跳过；api 需 ocr_api_base；mode: custom=自研/ocr 协议, "
                        "openai=OpenAI 兼容（如 DeepSeek-OCR，需 ocr_api_model）；"
                        "min_confidence=进正文阈值（默认 0.6，低于则只留签名并进审核队列）",
            })
            break
    else:
        out.append({"name": "ocr", "provider": "none", "base_url": "",
                    "key_configured": False, "status": "not_configured",
                    "note": "未启用 image source（config.json sources 中配置 images）"})
    # GitHub 采集
    gh_token = bool(os.environ.get("GITHUB_TOKEN", ""))
    gh_in_cfg = any(
        s.type == "github" and (s.get("token") or os.environ.get(s.get("token_env", "GITHUB_TOKEN"), ""))
        for s in cfg.effective_sources()
    )
    out.append({
        "name": "github",
        "provider": "rest",
        "base_url": "https://api.github.com",
        "key_configured": gh_token or gh_in_cfg,
        "status": "configured" if (gh_token or gh_in_cfg) else "missing",
        "note": "采集需要；检索不需要（离线）",
    })
    # 代码图谱检索（gh-puller 接入）——与 embedding/OCR 同级管理（连通性测试）
    cg = cfg.code_graph
    out.append({
        "name": "code_graph",
        "provider": "gh-puller",
        "base_url": cg.base_url or "",
        "path": cg.path,
        "enabled": cg.enabled,
        "key_configured": True,  # 业务环境无鉴权（暂定）
        "status": "configured" if (cg.enabled and cg.base_url) else
                  ("disabled" if cg.base_url else "missing"),
        "note": "代码图谱检索（调用链/影响面/架构）；enabled=False 时 /code-graph/* 端点不注册",
    })
    return out


# ---------------- API 配置编辑（审核工作台填写） ----------------

# 1x1 纯色 PNG（PIL 不可用时的 fallback 测试图，仅验证 HTTP/鉴权可达）
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def probe_ocr_connectivity(cfg: "AppConfig") -> dict:
    """OCR 连通性测试：用内置测试图走**真实 OCR 链路**（HTTP/鉴权/模型一次验证）。

    返回 {"ok": bool, "detail": str}。配置缺失（无 image source / provider=none /
    ask 且无 api_base）抛 ValueError（调用方转 400）；引擎失败返回 ok=False（转 502）。
    """
    import tempfile

    from .ocr import OcrApiError, OcrUnavailable, ocr_config_from_cfg, ocr_image

    oc = ocr_config_from_cfg(cfg)  # 与请求期 /ocr 端点同源，避免两处解析漂移
    if oc is None:
        raise ValueError("未启用 image source（config.json sources 中配置 images 条目）")
    provider, api_base, api_key, model, mode = (
        oc.provider, oc.api_base, oc.api_key, oc.model, oc.mode)
    if provider == "none":
        raise ValueError("ocr_provider=none（明确跳过 OCR），无需测试")
    if provider == "ask" and not api_base:
        raise ValueError("ocr_provider=ask 且未配置 ocr_api_base：无 API，连通性不适用（运行时会询问本地/跳过）")
    if provider == "api" and mode == "openai" and not model:
        raise ValueError("openai 模式的 OCR 需要 ocr_api_model（如 deepseek-ai/DeepSeek-OCR）")

    # 内置测试图：优先 PIL 生成含文字图（验证模型真实识别）；无 PIL 用 1x1 fallback（只验证可达）
    fd, tmp_name = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img_path = Path(tmp_name)
    try:
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (140, 40), "white")
            ImageDraw.Draw(img).text((8, 10), "vllm-kb OCR 123", fill="black")
            img.save(img_path)
        except Exception:
            img_path.write_bytes(base64.b64decode(_TINY_PNG_B64))
        text, conf = ocr_image(img_path, provider, api_base=api_base, api_key=api_key,
                               model=model, mode=mode)
    except OcrApiError as e:
        return {"ok": False, "detail": f"OCR API 不可达/失败: {e}"}
    except OcrUnavailable as e:
        return {"ok": False, "detail": f"本地 OCR 不可用: {e}"}
    finally:
        try:
            img_path.unlink()
        except OSError:
            pass
    return {"ok": True, "detail": f"连通（provider={provider} mode={mode}，识别 '{text[:40]}'，conf={conf:.2f}）"}


def update_config_json(cfg: "AppConfig", section: str, fields: dict,
                       config_path: Optional[str | Path] = None) -> None:
    """更新 config.json 的**非密钥**字段。

    section: embedding（顶层 embedding 段）| ocr（sources 中 type=image 条目）|
             github（sources 中 github 条目的非密钥字段）|
             code_graph（顶层 code_graph 段：enabled/base_url/path）。
    密钥一律不写 config.json（走 secrets 文件，见 vllm_kb/secrets.py）。
    修改对已运行服务生效需重启。
    """
    from .config import PROJECT_ROOT

    cfg_path = Path(config_path) if config_path else PROJECT_ROOT / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json 不存在: {cfg_path}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    fields = {k: v for k, v in fields.items() if v is not None}
    if section == "embedding":
        data.setdefault("embedding", {}).update(fields)
    elif section == "ocr":
        for sc in data.get("sources", []):
            if sc.get("type") == "image":
                sc.update(fields)
                break
        else:
            raise ValueError("config.json sources 中没有 image 来源（先添加 images 条目）")
    elif section == "github":
        for sc in data.get("sources", []):
            if sc.get("type") == "github":
                sc.update(fields)
                break
    elif section == "code_graph":
        # 顶层 code_graph 段（gh-puller 接入）：UI 只编辑 enabled/base_url/path
        # （repo_project_map 等高级字段手工编辑 config.json）
        data.setdefault("code_graph", {}).update(fields)
    else:
        raise ValueError(f"未知配置段: {section}（支持 embedding | ocr | github | code_graph）")
    cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
