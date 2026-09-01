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
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

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
    conn = sqlite3.connect(str(kb_path))
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
    conn = sqlite3.connect(str(kb_path))
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

    def mark_deleted(self, item_id: int, reviewer: str, kb_path: str | Path,
                     note: str = "") -> bool:
        """标记删除：**只删除 kb.sqlite3 的数据库记录**，原始资产文件保留
        （由人员到"待实际删除"列表手动本地删除）。删除前把完整 docs 行存入
        payload.doc_row，供撤回恢复。"""
        it = self.get_item(item_id)
        if it is None:
            return False
        doc_row = _read_doc_row(kb_path, it["item_ref"])
        if doc_row is None:
            raise ValueError(f"kb.sqlite3 中不存在 {it['item_ref']}（可能已删除）")
        _delete_doc_row(kb_path, it["item_ref"])
        payload = dict(it["payload"] or {})
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
            return cur.rowcount > 0
        finally:
            conn.close()

    def undo_delete(self, item_id: int, kb_path: str | Path) -> bool:
        """撤回删除：用 payload.doc_row 恢复 kb.sqlite3 记录，审核项回到 pending 重新入队。"""
        it = self.get_item(item_id)
        if it is None or it["status"] != "deleted":
            raise ValueError(f"审核项 {item_id} 不存在或不是 deleted 状态（当前 {it['status'] if it else '?'}）")
        doc_row = (it.get("payload") or {}).get("doc_row")
        if not doc_row:
            raise ValueError("该审核项缺少 doc_row 备份，无法撤回（可能由旧版本生成）")
        _restore_doc_row(kb_path, doc_row)
        payload = dict(it["payload"] or {})
        payload.pop("doc_row", None)
        payload.pop("deleted_note", None)
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE review_items SET status='pending', reviewed_at=NULL, reviewer=NULL, "
                "payload=?, result=NULL WHERE id=?",
                (json.dumps(payload, ensure_ascii=False), int(item_id)),
            )
            conn.commit()
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


# ---------------- 外源文档管理（审核工作台"文档管理"页签） ----------------

# 外源文档 = 非 GitHub 采集来源（导入的 PDF / Markdown / 表格 / OCR 等）
_EXTERNAL_SOURCE_TYPES = ("doc_pdf", "doc_markdown", "doc_excel", "doc_other")


def list_external_docs(kb_path: str | Path, review_db: Optional[str | Path] = None,
                       limit: int = 200, offset: int = 0) -> list[dict]:
    """列出 kb.sqlite3 中的外源文档（source_type 以 doc_ 开头，排除 github_*）。

    返回 {source_id, source_type, title, component, url, verification,
    asset_id, asset_path（管理员侧 rel_path，经 asset_registry）,
    tags: {auto, excluded, manual, final}, duplicate（同 stem 重名告警）}。
    """
    from .tagging import merge_final

    conn = sqlite3.connect(str(kb_path))
    try:
        rows = conn.execute(
            """SELECT source_id, source_type, title, component, url, extra
               FROM docs WHERE source_type NOT LIKE 'github%'
               ORDER BY source_type, source_id LIMIT ? OFFSET ?""",
            (int(limit), int(offset)),
        ).fetchall()
        overlay = load_doc_tags_conn(conn)
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
    # 同 stem 重名统计（pdf:/md: 前缀后的部分）
    stems: dict[str, int] = {}
    for sid, *_ in rows:
        stem = sid.split(":", 1)[1] if ":" in sid else sid
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
        stem = sid.split(":", 1)[1] if ":" in sid else sid
        out.append({
            "source_id": sid,
            "source_type": st,
            "title": title or "",
            "component": comp or "",
            "url": url or "",
            "verification": ex.get("verification", ""),
            "asset_id": asset_id,
            "asset_path": rel_path,
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


# ---------------- 资产注册表（asset_registry，review.sqlite3，管理员侧） ----------------
# 表结构见顶部 _ASSET_REGISTRY_DDL（ReviewStore._SCHEMA 与 register_asset 共用）。

def register_asset(db_path: str | Path, asset_id: str, rel_path: str,
                   sha256: str = "", size: int = 0, source_type: str = "") -> None:
    """注册资产映射（幂等 upsert，已有记录保留未提供的字段）。db_path=review.sqlite3。"""
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

def seed_verification_pending(cfg: "AppConfig", store: ReviewStore) -> int:
    """扫描 kb.sqlite3 中 verification=unverified 的文档，生成补标审核项（幂等）。"""
    kb = cfg.resolve(cfg.storage.sqlite_path)
    if not kb.exists():
        return 0
    conn = sqlite3.connect(f"file:{kb.as_posix()}?mode=ro", uri=True)
    added = 0
    try:
        rows = conn.execute(
            "SELECT source_id, title, url, extra FROM docs WHERE extra LIKE '%\"verification\": \"unverified\"%'"
        ).fetchall()
        for source_id, title, url, extra in rows:
            if store.add_item("verification_pending", source_id,
                              {"source_id": source_id, "title": title, "url": url}):
                added += 1
    finally:
        conn.close()
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
    from .tagging import TagRegistry

    kb = cfg.resolve(cfg.storage.sqlite_path)
    if not kb.exists():
        return 0
    # 清理旧格式（source_id::name）残留：item_ref 含 "::" 且非 "tag:" 前缀
    store.cleanup_old_tag_candidates()
    registry = TagRegistry.load(cfg)
    conn = sqlite3.connect(f"file:{kb.as_posix()}?mode=ro", uri=True)
    # candidate -> {docs: [(source_id, title)], tier}
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
                name = c.get("name") if isinstance(c, dict) else str(c)
                if not name or registry.contains(name):
                    continue
                tier = c.get("tier") if isinstance(c, dict) else None
                entry = agg.setdefault(name, {"docs": [], "tier": tier})
                entry["docs"].append((source_id, title or source_id))
                if tier and not entry["tier"]:
                    entry["tier"] = tier
    finally:
        conn.close()
    added = 0
    for name, info in sorted(agg.items()):
        docs = info["docs"]
        if store.add_item("tag_candidate", f"tag:{name}", {
            "candidate": name,
            "suggested_tier": info["tier"],
            "doc_count": len(docs),
            "docs": [{"source_id": s, "title": t} for s, t in docs[:20]],  # 展示用，截断
            # 兼容旧单文档展示：首个文档的 title/source_id
            "source_id": docs[0][0],
            "title": docs[0][1],
        }):
            added += 1
    return added


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
    entry = registry.add(name, tier)  # tier=None → 启发式
    save_registry_to_config(cfg, registry, config_path=config_path)
    # 提及该候选的全部文档（聚合）；旧单文档审核项回退到 source_id
    docs = [d["source_id"] for d in (payload.get("docs") or []) if d.get("source_id")]
    if not docs and source_id:
        docs = [source_id]
    kb = cfg.resolve(cfg.storage.sqlite_path)
    conn = sqlite3.connect(str(kb))
    try:
        for sid in docs:
            ov = get_doc_tags_conn(conn, sid)
            manual = list(ov["manual"])
            if name not in manual:
                manual.append(name)
            set_doc_tags_conn(conn, sid, manual=manual, reviewer=reviewer)
        conn.commit()
    finally:
        conn.close()
    store.review(item_id, "approved", reviewer,
                 result={"action": "adopt", "tag": name, "tier": entry.tier,
                         "docs_count": len(docs)})
    return {"ok": True, "tag": name, "tier": entry.tier, "source_id": source_id,
            "docs_count": len(docs)}


def seed_all(cfg: "AppConfig", store: ReviewStore) -> dict[str, int]:
    """运行全部 seed（幂等），返回各 seed 新增数。"""
    return {
        "verification_pending": seed_verification_pending(cfg, store),
        "case_title_flag": seed_case_title_flags(cfg, store),
        "tag_candidate": seed_tag_candidates(cfg, store),
    }


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
                "key_configured": bool(key),
                "status": "configured" if (api_base or sc.get("ocr_provider") == "paddle") else "ask",
                "note": "ask=无 API 时询问本地/跳过；api 需 ocr_api_base；mode: custom=自研/ocr 协议, "
                        "openai=OpenAI 兼容（如 DeepSeek-OCR，需 ocr_api_model）",
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
        "key_configured": True,  # 内网无鉴权（暂定）
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


def test_ocr_connectivity(cfg: "AppConfig") -> dict:
    """OCR 连通性测试：用内置测试图走**真实 OCR 链路**（HTTP/鉴权/模型一次验证）。

    返回 {"ok": bool, "detail": str}。配置缺失（无 image source / provider=none /
    ask 且无 api_base）抛 ValueError（调用方转 400）；引擎失败返回 ok=False（转 502）。
    """
    import tempfile

    from .ocr import OcrApiError, OcrUnavailable, ocr_image

    sc = next((s for s in cfg.effective_sources() if s.type == "image"), None)
    if sc is None:
        raise ValueError("未启用 image source（config.json sources 中配置 images 条目）")
    provider = str(sc.get("ocr_provider", "ask") or "ask").lower()
    api_base = str(sc.get("ocr_api_base", "") or "")
    api_key = str(sc.get("ocr_api_key", "") or os.environ.get("OCR_API_KEY", ""))
    model = str(sc.get("ocr_api_model", "") or "")
    mode = str(sc.get("ocr_api_mode", "custom") or "custom")
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
             github（sources 中 github 条目的非密钥字段）。
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
    else:
        raise ValueError(f"未知配置段: {section}（支持 embedding | ocr | github）")
    cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
