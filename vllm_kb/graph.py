"""Kùzu 图存储（Phase 2 图 + 向量双存储的图侧）。

节点表（V1）：
- Issue    {id(PK), repo, number, title, status, created_at, resolved_at, url}
- PR       {id(PK), repo, number, title, status, merged_at, url}
- Release  {id(PK), repo, tag, date, kind}
- Operator {id(PK)}  ErrorCode {id(PK)}  Model {id(PK)}  Version {id(PK)}

关系表：
- FIXES     (PR → Issue)
- MERGED_IN (PR → Release)
- MENTIONS  (Issue/PR → Operator/ErrorCode/Model/Version)  多 label 关系表

构建方式：canonical.jsonl 逐条确定性抽取（graph_rels）→ 节点/边 CSV → Kùzu COPY（批量最快）。

兼容性保证：**图构建只依赖 canonical 与版本日历**——任何来源只要产出 canonical
（含 source_id/body/version_span/extra）即可入图；业务来源的扩展字段（asset/quality/
verification/evidence）无需改图侧，节点 id 一律用 source_id 命名空间，天然多来源兼容。

PART_OF（chunk→doc）V1 不入图：chunk 归属在 SQLite/FTS 侧已有一对一映射，图内重复建
12 万+ chunk 节点成本高、查询价值低；图专注"跨文档关系追溯"（issue→PR→release→实体）。
"""
from __future__ import annotations

import csv
import io
import json
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import kuzu

from .graph_rels import (
    extract_doc_relations,
    load_release_calendars,
    map_merged_to_release,
)

if TYPE_CHECKING:
    from .config import AppConfig

# ---------------- Schema（Kùzu DDL） ----------------

_NODE_DDL = [
    "CREATE NODE TABLE Issue(id STRING, repo STRING, number INT64, title STRING, "
    "status STRING, created_at STRING, resolved_at STRING, url STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE PR(id STRING, repo STRING, number INT64, title STRING, "
    "status STRING, merged_at STRING, url STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE Release(id STRING, repo STRING, tag STRING, date STRING, "
    "kind STRING, PRIMARY KEY(id))",
    # 通用文档节点（doc_pdf / doc_markdown / 其他非 github 来源）：接口指南、案例、wiki 等
    "CREATE NODE TABLE Doc(id STRING, source_type STRING, title STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE Operator(id STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE ErrorCode(id STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE Model(id STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE Version(id STRING, PRIMARY KEY(id))",
    # 接口/命令（如 hccn_tool）：从手册表格/正文提取
    "CREATE NODE TABLE Interface(id STRING, PRIMARY KEY(id))",
]

_REL_DDL = [
    "CREATE REL TABLE FIXES(FROM PR TO Issue)",
    "CREATE REL TABLE MERGED_IN(FROM PR TO Release)",
    "CREATE REL TABLE MENTIONS(FROM Issue TO Operator, FROM Issue TO ErrorCode, "
    "FROM Issue TO Model, FROM Issue TO Version, FROM PR TO Operator, FROM PR TO ErrorCode, "
    "FROM PR TO Model, FROM PR TO Version, FROM Doc TO Operator, FROM Doc TO ErrorCode, "
    "FROM Doc TO Model, FROM Doc TO Version)",
    # 文档化关系：接口指南/手册等文档记录某错误码/接口（可回答"这个错误码在哪个手册定义"）
    "CREATE REL TABLE DOCUMENTS(FROM Doc TO ErrorCode, FROM Doc TO Interface)",
]

# 实体类型 → 节点表名
_ENTITY_TABLE = {
    "operator": "Operator",
    "error_code": "ErrorCode",
    "model": "Model",
    "version": "Version",
}
# 节点表名 → 规范 kind（label() 返回表名，输出统一为小写 kind）
_ENTITY_KIND_BY_TABLE = {v: k for k, v in _ENTITY_TABLE.items()}

_REPO = "repo"
_NUMBER = "number"


def _doc_id(repo: str, kind: str, number: int) -> str:
    """按 github_pull.source_id 的命名空间规则构造端点 id：
    repo 的 '/' 替换为 '-'（github:vllm-project-vllm:issue:50237）。
    extra.repo 存的是带 '/' 的原名，必须 slug 化才能与 source_id 匹配。
    """
    return f"github:{repo.replace('/', '-')}:{kind}:{number}"


# 表格/正文中的 ACL 错误码（与 error_parse._ACL_CODE_RE 同源）
_ERRCODE_IN_TEXT_RE = re.compile(r"\b(E\d{4,6}|10[7-9]\d{3}|50\d{4}|56\d{4}|0x[0-9a-fA-F]{5,8})\b")

# 命令格式段：工具名 + 可选属性子命令（-xxx，排除方括号内的参数块与 -g/-s/-h 动作）
_CMD_FORMAT_RE = re.compile(
    r"命令格式\s*\n\s*(?P<tool>[a-z][a-z0-9_-]{1,31})"
    r"(?:\s*\[[^\]]*\])*(?:\s+-(?P<sub>[a-z][a-z0-9_-]*))?"
)
# 属性类子命令（-bandwidth/-roce_test/-ip...）；-g 查询/-s 设置/-h 帮助是动作，不作为接口标识
_CMD_ACTION_FLAGS = {"g", "s", "h", "i"}


def _extract_doc_interfaces(doc: dict) -> set[str]:
    """从文档正文的"命令格式"段提取 Interface（工具.子命令级）。

    只提取**文档结构化定义的命令**（命令格式段首行 = 工具名 + 属性 flag），
    且只产出**子命令级**（如 hccn_tool.bandwidth）——纯工具名（如 hccn_tool -h）
    无接口信息量，跳过；正文随机出现的工具名/日志示例不提取，保证节点纯度。
    返回 {"hccn_tool.bandwidth", "hccn_tool.roce_test", ...}。
    """
    out: set[str] = set()
    body = doc.get("body") or ""
    for m in _CMD_FORMAT_RE.finditer(body):
        tool = m.group("tool")
        sub = (m.group("sub") or "").strip()
        if not tool or not sub:
            continue
        if sub.lower() in _CMD_ACTION_FLAGS or len(sub) <= 1:
            continue
        out.add(f"{tool}.{sub.lower()}")
    return out


def _extract_doc_table_codes(doc: dict, parsed_root: Path) -> set[str]:
    """从文档的 parsed 表格产物（data/parsed/pdf/<name>.tables.json）提取错误码。

    接口指南的错误码表 → ErrorCode 节点 + DOCUMENTS 边（"这个错误码在哪个手册定义"）。
    无表格产物时返回空集。
    """
    codes: set[str] = set()
    asset = (doc.get("extra") or {}).get("asset") or {}
    stem = Path(str(asset.get("path", ""))).stem
    if not stem:
        return codes
    for name in (f"{stem}.tables.json", f"{stem}.table.json"):
        p = parsed_root / "pdf" / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for tab in data.get("tables") or []:
            for row in tab.get("rows") or []:
                for cell in row:
                    for m in _ERRCODE_IN_TEXT_RE.finditer(str(cell)):
                        codes.add(m.group(1))
    return codes


@dataclass
class GraphStats:
    nodes: dict[str, int] = field(default_factory=dict)
    rels: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        n = sum(self.nodes.values())
        r = sum(self.rels.values())
        return (f"节点 {n}（Issue {self.nodes.get('Issue', 0)} / PR {self.nodes.get('PR', 0)} / "
                f"Release {self.nodes.get('Release', 0)} / 实体 {n - self.nodes.get('Issue', 0) - self.nodes.get('PR', 0) - self.nodes.get('Release', 0)}），"
                f"边 {r}（FIXES {self.rels.get('FIXES', 0)} / MERGED_IN {self.rels.get('MERGED_IN', 0)} / "
                f"MENTIONS {self.rels.get('MENTIONS', 0)}）")


class GraphBuilder:
    """Kùzu 图构建：schema + 从 canonical 批量建图（COPY）。"""

    def __init__(self, graph_dir: str | os.PathLike, buffer_pool_size: int = 0):
        self.graph_dir = Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        # Kùzu 要求路径不存在（自建）；已存在目录时用子目录区分
        db_path = self.graph_dir / "db"
        self.db = kuzu.Database(str(db_path), buffer_pool_size=buffer_pool_size) if buffer_pool_size else kuzu.Database(str(db_path))
        self.conn = kuzu.Connection(self.db)

    # ---------- schema ----------

    def create_schema(self, drop_existing: bool = False) -> None:
        if drop_existing:
            for name in ("DOCUMENTS", "MENTIONS", "MERGED_IN", "FIXES", "Interface",
                         "Version", "Model", "ErrorCode", "Operator", "Doc", "Release", "PR", "Issue"):
                try:
                    self.conn.execute(f"DROP TABLE {name}")
                except Exception:
                    pass
        for ddl in _NODE_DDL:
            self.conn.execute(ddl)
        for ddl in _REL_DDL:
            self.conn.execute(ddl)

    # ---------- 构建 ----------

    def build_from_canonical(
        self,
        canonical_path: str | os.PathLike,
        calendars: Optional[dict[str, list]] = None,
        limit: int = 0,
        parsed_root: Optional[str | os.PathLike] = None,
    ) -> GraphStats:
        """遍历 canonical.jsonl 建图。limit>0 时只处理前 limit 条（试跑）。

        calendars: {repo: [ReleaseInfo]}，None 时自动从 data/compatibility 加载。
        parsed_root: data/parsed 目录（None 时跳过文档表格 → ErrorCode 的 DOCUMENTS 提取）。
        """
        calendars = calendars if calendars is not None else load_release_calendars()
        canonical_path = Path(canonical_path)
        parsed_root = Path(parsed_root) if parsed_root else None

        # 收集节点与边
        issue_rows: dict[str, dict] = {}
        pr_rows: dict[str, dict] = {}
        doc_rows: dict[str, dict] = {}
        release_rows: dict[str, dict] = {}
        entity_vals: dict[str, set[str]] = {k: set() for k in _ENTITY_TABLE}
        interface_vals: set[str] = set()
        fixes_edges: set[tuple[str, str]] = set()      # (pr_id, issue_id)
        merged_edges: set[tuple[str, str]] = set()     # (pr_id, release_id)
        mention_edges: set[tuple[str, str, str]] = set()  # (doc_id, entity_kind, value)
        documents_edges: set[tuple[str, str, str]] = set()  # (doc_id, target_kind, target_id)

        with canonical_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._ingest_doc(doc, issue_rows, pr_rows, doc_rows, release_rows, entity_vals,
                                 interface_vals, fixes_edges, merged_edges, mention_edges,
                                 documents_edges, calendars, parsed_root)

        # 过滤端点不存在的边（引用编号可能不在库中/编号实际是 PR 等）：Kùzu COPY 强制端点存在
        issue_ids = set(issue_rows)
        pr_ids = set(pr_rows)
        fixes_edges = {(a, b) for a, b in fixes_edges if a in pr_ids and b in issue_ids}
        merged_edges = {(a, b) for a, b in merged_edges if a in pr_ids}
        doc_ids = issue_ids | pr_ids | set(doc_rows)
        mention_edges = {(a, k, v) for a, k, v in mention_edges if a in doc_ids}
        documents_edges = {
            (a, k, v) for a, k, v in documents_edges
            if a in doc_ids and (k == "error_code" and v in entity_vals["error_code"]
                                 or k == "interface" and v in interface_vals)
        }

        # 写入 CSV 并 COPY
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._copy_nodes(tmp_path, issue_rows, pr_rows, doc_rows, release_rows,
                             entity_vals, interface_vals)
            self._copy_rels(tmp_path, fixes_edges, merged_edges, mention_edges, documents_edges)

        return self.stats()

    def _ingest_doc(self, doc: dict, issue_rows, pr_rows, doc_rows, release_rows, entity_vals,
                    interface_vals, fixes_edges, merged_edges, mention_edges, documents_edges,
                    calendars, parsed_root: Optional[Path] = None) -> None:
        st = doc.get("source_type", "")
        sid = doc.get("source_id", "")
        if not sid:
            return
        extra = doc.get("extra") or {}
        repo = extra.get("repo") or doc.get("component", "")
        number = extra.get("github_number") or 0
        title = (doc.get("title") or "").replace("\n", " ").replace("\r", " ")
        url = doc.get("url", "")
        status = doc.get("status", "")
        vs = doc.get("version_span") or {}

        ex = extract_doc_relations(
            sid, repo, number, st,
            doc.get("body") or "",
            version_span_min=vs.get("min"),
            version_span_max=vs.get("max"),
        )

        if st == "github_issue":
            issue_rows[sid] = {
                "id": sid, "repo": repo, "number": number, "title": title,
                "status": status, "created_at": doc.get("created_at") or "",
                "resolved_at": doc.get("resolved_at") or "", "url": url,
            }
            # 反向引用：PR(repo:n) -> 本 issue
            for trepo, n in ex.fixed_by:
                fixes_edges.add((_doc_id(trepo, "pr", n), sid))
        elif st == "github_pr":
            merged_at = extra.get("merged_at") or ""
            pr_rows[sid] = {
                "id": sid, "repo": repo, "number": number, "title": title,
                "status": status, "merged_at": merged_at, "url": url,
            }
            # 正向引用：本 PR -> Issue(repo:n)
            for trepo, n in ex.fixes:
                fixes_edges.add((sid, _doc_id(trepo, "issue", n)))
            # MERGED_IN
            if merged_at and repo in calendars:
                tag = map_merged_to_release(merged_at, calendars[repo])
                if tag:
                    rel_id = f"{repo}:{tag}"
                    release_rows.setdefault(rel_id, {
                        "id": rel_id, "repo": repo, "tag": tag,
                        "date": next((r.date for r in calendars[repo] if r.tag == tag), ""),
                        "kind": next((r.kind for r in calendars[repo] if r.tag == tag), "release"),
                    })
                    merged_edges.add((sid, rel_id))
        else:
            # 通用文档节点（doc_pdf / doc_markdown / 其他非 github 来源）
            doc_rows[sid] = {"id": sid, "source_type": st, "title": title}
            # 文档表格 → ErrorCode 的 DOCUMENTS 边（接口指南错误码表）
            if parsed_root is not None:
                for code in _extract_doc_table_codes(doc, parsed_root):
                    entity_vals["error_code"].add(code)
                    documents_edges.add((sid, "error_code", code))
            # 文档"命令格式"段 → Interface 的 DOCUMENTS 边（手册定义的接口/命令）
            for iface in _extract_doc_interfaces(doc):
                interface_vals.add(iface)
                documents_edges.add((sid, "interface", iface))

        # MENTIONS：任何 doc → 实体
        for kind, values in (ex.mentions or {}).items():
            table = _ENTITY_TABLE.get(kind)
            if table is None:
                continue
            for v in values:
                if not v:
                    continue
                entity_vals[kind].add(v)
                mention_edges.add((sid, kind, v))

    # ---------- CSV + COPY ----------

    def _write_csv(self, tmp: Path, name: str, header: list[str], rows) -> str:
        p = tmp / name
        with p.open("w", newline="", encoding="utf-8") as f:
            # QUOTE_ALL：任何字段都引号包裹，避免字段含分隔符/引号/控制字符时
            # QUOTE_MINIMAL 需要 escapechar 的场景；Kùzu COPY（PARALLEL=FALSE）支持 RFC4180 引号字段。
            w = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar="\\")
            w.writerow(header)
            for r in rows:
                w.writerow(r)
        return str(p).replace("\\", "/")

    def _copy_nodes(self, tmp: Path, issue_rows, pr_rows, doc_rows, release_rows,
                    entity_vals, interface_vals) -> None:
        def _node_csv(name, rows):
            if not rows:
                return None
            header = list(next(iter(rows.values())).keys())
            return self._write_csv(tmp, name, header, (r.values() for r in rows.values()))

        for table, rows in (("Issue", issue_rows), ("PR", pr_rows), ("Doc", doc_rows),
                            ("Release", release_rows)):
            p = _node_csv(f"{table}.csv", rows)
            if p:
                self.conn.execute(f"COPY {table} FROM '{p}' (HEADER=true, PARALLEL=FALSE)")
        if interface_vals:
            p = self._write_csv(tmp, "Interface.csv", ["id"], ([v] for v in sorted(interface_vals)))
            self.conn.execute(f"COPY Interface FROM '{p}' (HEADER=true, PARALLEL=FALSE)")
        for kind, table in _ENTITY_TABLE.items():
            vals = entity_vals[kind]
            if not vals:
                continue
            p = self._write_csv(tmp, f"{table}.csv", ["id"], ([v] for v in sorted(vals)))
            self.conn.execute(f"COPY {table} FROM '{p}' (HEADER=true, PARALLEL=FALSE)")

    def _copy_rels(self, tmp: Path, fixes_edges, merged_edges, mention_edges,
                   documents_edges) -> None:
        if fixes_edges:
            p = self._write_csv(tmp, "FIXES.csv", ["PR_id", "Issue_id"], fixes_edges)
            self.conn.execute(f"COPY FIXES FROM '{p}' (HEADER=true, PARALLEL=FALSE)")
        if merged_edges:
            p = self._write_csv(tmp, "MERGED_IN.csv", ["PR_id", "Release_id"], merged_edges)
            self.conn.execute(f"COPY MERGED_IN FROM '{p}' (HEADER=true, PARALLEL=FALSE)")
        if documents_edges:
            # DOCUMENTS 多 label：按目标类型拆分（ErrorCode / Interface）
            err_edges = [(a, c) for a, k, c in documents_edges if k == "error_code"]
            iface_edges = [(a, c) for a, k, c in documents_edges if k == "interface"]
            if err_edges:
                p = self._write_csv(tmp, "DOCUMENTS_E.csv", ["Doc_id", "ErrorCode_id"], err_edges)
                self.conn.execute(
                    f"COPY DOCUMENTS FROM '{p}' (HEADER=true, PARALLEL=FALSE, FROM='Doc', TO='ErrorCode')")
            if iface_edges:
                p = self._write_csv(tmp, "DOCUMENTS_I.csv", ["Doc_id", "Interface_id"], iface_edges)
                self.conn.execute(
                    f"COPY DOCUMENTS FROM '{p}' (HEADER=true, PARALLEL=FALSE, FROM='Doc', TO='Interface')")
        if mention_edges:
            # MENTIONS 多 label：按目标实体类型拆分 CSV（from 列名固定 Issue_id/PR_id）
            by_kind: dict[str, list[tuple[str, str]]] = {}
            for doc_id, kind, value in mention_edges:
                by_kind.setdefault(kind, []).append((doc_id, value))
            for kind, edges in by_kind.items():
                table = _ENTITY_TABLE[kind]
                # from 端点可能是 Issue / PR / Doc：从 source_id 前缀判断
                issue_edges = [(a, b) for a, b in edges if ":issue:" in a]
                pr_edges = [(a, b) for a, b in edges if ":pr:" in a]
                doc_edges = [(a, b) for a, b in edges
                             if ":issue:" not in a and ":pr:" not in a]
                if issue_edges:
                    p = self._write_csv(tmp, f"MENTIONS_I_{table}.csv", ["Issue_id", f"{table}_id"], issue_edges)
                    self.conn.execute(f"COPY MENTIONS FROM '{p}' (HEADER=true, PARALLEL=FALSE, FROM='Issue', TO='{table}')")
                if pr_edges:
                    p = self._write_csv(tmp, f"MENTIONS_P_{table}.csv", ["PR_id", f"{table}_id"], pr_edges)
                    self.conn.execute(f"COPY MENTIONS FROM '{p}' (HEADER=true, PARALLEL=FALSE, FROM='PR', TO='{table}')")
                if doc_edges:
                    p = self._write_csv(tmp, f"MENTIONS_D_{table}.csv", ["Doc_id", f"{table}_id"], doc_edges)
                    self.conn.execute(f"COPY MENTIONS FROM '{p}' (HEADER=true, PARALLEL=FALSE, FROM='Doc', TO='{table}')")

    # ---------- 查询与统计 ----------

    def query(self, cypher: str, params: Optional[dict] = None) -> list[list]:
        """执行只读查询，返回行列表（每行 list 值）。"""
        result = self.conn.execute(cypher, params or {})
        return [list(row) for row in result]

    def is_built(self) -> bool:
        """图库是否已构建（存在且含表）。

        Kùzu 在 Windows 上把库存为单文件（data/graph/db），Linux 上为目录——两种形态都处理。
        注意：kuzu.Database() 懒加载即创建空库文件，仅凭文件存在会误判——用已知表探测
        （schema 固定含 Issue 表；空库查询不存在的表会抛错 → False）。
        """
        db_path = self.graph_dir / "db"
        if not db_path.exists():
            return False
        if db_path.is_dir() and not any(db_path.iterdir()):
            return False
        try:
            rows = self.query("MATCH (i:Issue) RETURN 1 LIMIT 1")
            return bool(rows)
        except Exception:
            return False

    def chain_issue(self, issue_id: str) -> dict:
        """核心链路：issue → 修复它的 PR → 落地 release。

        回答"这问题是否已修复、修复在哪个版本提供"。PR 无 MERGED_IN = 修复合并但尚未发布。
        """
        head = self.query(
            "MATCH (i:Issue {id: $id}) RETURN i.title, i.repo, i.number, i.status, i.url, i.resolved_at",
            {"id": issue_id},
        )
        if not head:
            return {"issue_id": issue_id, "found": False}
        h = head[0]
        fixes = self.query(
            "MATCH (i:Issue {id: $id})<-[:FIXES]-(pr:PR) "
            "OPTIONAL MATCH (pr)-[:MERGED_IN]->(r:Release) "
            "RETURN pr.id, pr.title, pr.status, pr.merged_at, r.tag, r.date, r.repo ORDER BY r.date",
            {"id": issue_id},
        )
        fix_list = [
            {
                "pr_id": f[0], "title": f[1], "status": f[2], "merged_at": f[3],
                "release_tag": f[4], "release_date": f[5], "release_repo": f[6],
            }
            for f in fixes
        ]
        return {
            "issue_id": issue_id,
            "found": True,
            "title": h[0], "repo": h[1], "number": h[2], "status": h[3],
            "url": h[4], "resolved_at": h[5],
            "fixes": fix_list,
            "released": any(f["release_tag"] for f in fix_list),
        }

    def fixes_pr(self, pr_id: str) -> dict:
        """给定 PR：它修复的 issues + 落地 release。"""
        head = self.query(
            "MATCH (p:PR {id: $id}) RETURN p.title, p.repo, p.number, p.status, p.merged_at, p.url",
            {"id": pr_id},
        )
        if not head:
            return {"pr_id": pr_id, "found": False}
        h = head[0]
        issues = self.query(
            "MATCH (p:PR {id: $id})-[:FIXES]->(i:Issue) RETURN i.id, i.title, i.status, i.url",
            {"id": pr_id},
        )
        rels = self.query(
            "MATCH (p:PR {id: $id})-[:MERGED_IN]->(r:Release) RETURN r.tag, r.date, r.kind",
            {"id": pr_id},
        )
        return {
            "pr_id": pr_id,
            "found": True,
            "title": h[0], "repo": h[1], "number": h[2], "status": h[3],
            "merged_at": h[4], "url": h[5],
            "fixes": [
                {"issue_id": i[0], "title": i[1], "status": i[2], "url": i[3]} for i in issues
            ],
            "releases": [{"tag": r[0], "date": r[1], "kind": r[2]} for r in rels],
        }

    def sig_lookup(self, sig: str, limit: int = 10) -> dict:
        """签名实体（算子/错误码/模型/版本）→ 提及它的 issue/PR + 实体类型。

        通用查询不限定实体 label：MENTIONS 的 TO 端多 label，WHERE e.id 过滤即可。
        """
        limit = max(1, min(int(limit), 100))
        rows = self.query(
            "MATCH (d)-[:MENTIONS]->(e) WHERE e.id = $s "
            "RETURN label(e) AS etype, d.id, d.title, d.status ORDER BY d.id LIMIT " + str(limit),
            {"s": sig},
        )
        docs = [
            {"entity_type": _ENTITY_KIND_BY_TABLE.get(r[0], r[0]), "doc_id": r[1],
             "title": r[2], "status": r[3]} for r in rows
        ]
        return {
            "signature": sig,
            "entity_type": docs[0]["entity_type"] if docs else None,
            "docs": docs,
            "count": len(docs),
            "note": "实体类型: operator=算子 error_code=错误码 model=模型 version=版本；"
                    "doc 状态 open/closed/merged",
        }

    def doc_neighbors(self, doc_id: str) -> dict:
        """文档邻接视图（调试/详情）：MENTIONS 实体 + DOCUMENTS（手册定义了什么）。"""
        mentions = self.query(
            "MATCH (d {id: $id})-[:MENTIONS]->(e) RETURN label(e) AS etype, e.id",
            {"id": doc_id},
        )
        documents = self.query(
            "MATCH (d:Doc {id: $id})-[:DOCUMENTS]->(e) RETURN label(e) AS etype, e.id",
            {"id": doc_id},
        )
        out = {
            "doc_id": doc_id,
            "mentions": [{"entity_type": _ENTITY_KIND_BY_TABLE.get(r[0], r[0]), "value": r[1]}
                         for r in mentions],
        }
        if documents:
            out["documents"] = [
                {"entity_type": _ENTITY_KIND_BY_TABLE.get(r[0]) or str(r[0]).lower(),
                 "value": r[1]}
                for r in documents
            ]
        return out

    def docs_for_error_code(self, code: str) -> list[dict]:
        """错误码 → 定义它的文档（DOCUMENTS 反向）："这个错误码在哪个手册定义"。"""
        rows = self.query(
            "MATCH (d:Doc)-[:DOCUMENTS]->(e:ErrorCode {id: $c}) RETURN d.id, d.source_type, d.title",
            {"c": code},
        )
        return [{"doc_id": r[0], "source_type": r[1], "title": r[2]} for r in rows]

    def stats(self) -> GraphStats:
        s = GraphStats()
        for row in self.query("MATCH (n) RETURN label(n), count(*) ORDER BY count(*) DESC"):
            s.nodes[str(row[0])] = int(row[1])
        for rel in ("FIXES", "MERGED_IN", "MENTIONS", "DOCUMENTS"):
            try:
                s.rels[rel] = int(self.query(f"MATCH ()-[:{rel}]->() RETURN count(*)")[0][0])
            except Exception:
                s.rels[rel] = 0
        return s

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass


def default_graph_path(cfg: Optional["AppConfig"] = None) -> Path:
    """默认图目录：cfg.storage.graph_path（经 VLLM_KB_DATA_ROOT 重定向）或 data/graph。"""
    if cfg is not None:
        return cfg.resolve(getattr(cfg.storage, "graph_path", "data/graph"))
    return Path("data/graph")
