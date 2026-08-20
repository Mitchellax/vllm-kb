"""错误签名提取与精确检索：从用户原始输入（报错日志/描述）提取关键签名，
用签名做精确匹配（FTS 短语 + 关键词），与语义检索解耦。

设计要点：
- 签名从**本次输入**现场提取（不预存、不依赖历史），复用知识库的 FTS 精确匹配能力；
- 提取规则按 vllm/vllm-ascend 故障特征定制：
  * 算子/内核名（kernel_name=、aclnnXxx、dispatch_ffn_combine 等）
  * ACL 错误码（drvRetCode=6、error code 107020、507014 等）
  * 专有错误短语（timeout or trap error、out of memory、SMMU 等）
  * 环境变量/特性开关（VLLM_ASCEND_*、enable_fused_mc2 等）
  * 模型名（GLM-5.1、DeepSeek-V4-Flash、Qwen3.5 等）
- 精确检索：每个签名做 FTS5 短语查询，命中文档按"命中签名数 + 权重"聚合排序；
- 与语义检索（search.py）独立：本模块只做"签名 → 精确命中"，语义兜底由调用方决定。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------- 签名提取规则 ----------------

# 内核/算子名：kernel_name=xxx、kName=xxx、aclnnXxx、npuXxx、_C_ascend::xxx
_KERNEL_RE = re.compile(
    r"(?:kernel_name|kName|kname)\s*=\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)"
)
_ACLNN_RE = re.compile(r"\b(aclnn[A-Z][A-Za-z0-9_]*|npu[A-Z][A-Za-z0-9_]*|_C_ascend::[A-Za-z0-9_]+)")
# 下划线小写算子（dispatch_ffn_combine、mega_moe、npu_moe_distribute_* 等）
_OP_RE = re.compile(r"\b(dispatch_ffn_combine|mega_moe|moe_[a-z0-9_]+|[a-z]+_[a-z0-9_]+_combine|"
                    r"[a-z]+_[a-z0-9_]+_dispatch)\b")

# ACL 错误码
_ERRCODE_RE = re.compile(
    r"(?:drvRetCode|retCode)\s*[=: ]\s*(\d{1,7})|"
    r"(?:error code|errCode|err_code|error_code)\s*[=: ]\s*(\d{3,7})"
)
_ERRNO_RE = re.compile(r"\b(?:ACL_|aclnn)?(?:error|errno)?\s*(?:code|number)\s*[=:]\s*(\d{6})\b")
# 裸 6 位错误码（如 107020 / 507014 / 561000）——限定在错误上下文附近
_RAW_ERRCODE_RE = re.compile(r"(?i)\b(?:failed|error|exception)[^\n]{0,40}\b(107\d{3}|50[0-9]\d{3}|56\d{4}|0x[0-9a-f]{5,8})\b")

# 专有错误短语
_PHRASES = [
    "timeout or trap error", "trap error", "out of memory", "out pf memory",
    "illegal memory access", "smmu fault", "smmu", "aivector error", "aivec error",
    "mt3 error", "mte address", "vector core exception", "hccl failure",
    "suspected remote error", "workspace", "halmemcreate",
]

# 环境变量 / 特性开关
_ENV_RE = re.compile(r"\b(VLLM_ASCEND_[A-Z0-9_]+|ASCEND_[A-Z0-9_]+|HCCL_[A-Z0-9_]+|"
                     r"enable_[a-z0-9_]+|FUSED_MC2|MLAPO|PCP|DCP|EPLB)\b")

# 模型名（vllm-ascend 常见）
_MODEL_RE = re.compile(
    r"\b(GLM-?5(?:\.\d+)?|DeepSeek-?V4[-A-Za-z0-9]*|Qwen3(?:\.\d+)?[-A-Za-z0-9]*|"
    r"MiniMax-?M[0-9.]+|Kimi-?K[0-9.]+|Bailing[-\w]*)\b"
)

# 版本号（部署版本信号）
_VERSION_RE = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?(?:rc\d+)?)\b")

# 单字符过滤：太短的签名没有区分度
_MIN_SIG_LEN = 3


@dataclass
class Signature:
    text: str          # 签名原文（小写归一）
    kind: str          # kernel | errcode | phrase | env | model | version | op
    weight: float = 1.0
    origin: str = ""   # 提取来源（如 "kernel_name="）


def _dedupe(sigs: list[Signature]) -> list[Signature]:
    """去重：先按 (kind, text) 完全去重；同 kind 下短签名是长签名子串时只保留长者。

    注意：errcode 类不做子串去重（错误码 "6" 与 "561000" 是不同错误码，不是变体）。
    """
    seen: set[tuple[str, str]] = set()
    uniq: list[Signature] = []
    for s in sigs:
        key = (s.kind, s.text)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    by_kind: dict[str, list[Signature]] = {}
    for s in uniq:
        by_kind.setdefault(s.kind, []).append(s)
    out: list[Signature] = []
    for kind, lst in by_kind.items():
        if kind == "errcode":
            # 错误码不按子串去重：6 与 561000 是独立错误码
            out.extend(lst)
            continue
        lst.sort(key=lambda s: -len(s.text))
        kept: list[Signature] = []
        for s in lst:
            if any(s.text in k.text for k in kept):
                continue
            kept.append(s)
        out.extend(kept)
    return out


def extract_signatures(text: str, symbol_table=None, signal_words: Optional[list[str]] = None) -> list[Signature]:
    """从原始输入提取错误签名（三层：源码符号表 > 结构解析 > 通用短语兜底）。

    - symbol_table：源码符号表（SymbolTable），算子/特性自动匹配——**优先层**；
    - signal_words：社区高频信号词（由 agent 判断用不用，本函数只透传不筛选）；
    - 结构解析（error_parse）：Python 堆栈函数、键值对（kernel_name=/errorStr=）、ACL 错误码、
      HDK/CANN 版本——解析**结构**而非内容；
    - 通用短语：少量专有错误短语兜底（timeout or trap error 等）。
    """
    if not text:
        return []
    sigs: list[Signature] = []

    def add(s: str, kind: str, weight: float, origin: str = "") -> None:
        s = s.strip().strip("\"'")
        # errcode 短编码（如 drvRetCode=6）也保留；其余签名需 ≥ 最小长度
        if len(s) < _MIN_SIG_LEN and kind != "errcode":
            return
        sigs.append(Signature(text=s.lower(), kind=kind, weight=weight, origin=origin))

    # ---- 第 1 层：源码符号表匹配（算子 > 特性 > 模型 > 版本，按你给的优先级）----
    if symbol_table is not None:
        from .symbol_table import match_symbols

        for name, kind, weight in match_symbols(symbol_table, text):
            sig_kind = {"kernel": "kernel", "feature": "env", "model": "model",
                        "version": "version"}.get(kind, "op")
            add(name, sig_kind, weight, f"symbol[{kind}]")
    else:
        # 无符号表时的基础兜底（保持旧能力与 API 兼容）
        for m in _KERNEL_RE.finditer(text):
            add(m.group(1), "kernel", 3.5, "kernel_name=")
        for m in _ACLNN_RE.finditer(text):
            add(m.group(1), "kernel", 3.0, "aclnn")
        for m in _OP_RE.finditer(text):
            add(m.group(1), "op", 2.5, "op")
        for m in _ERRCODE_RE.finditer(text):
            code = m.group(1) or m.group(2)
            if code:
                w = 1.0 if len(code) == 1 else 3.0
                add(code, "errcode", w, "errcode")
        for m in _ERRNO_RE.finditer(text):
            add(m.group(1), "errcode", 2.5, "errno")
        for m in _RAW_ERRCODE_RE.finditer(text):
            add(m.group(1), "errcode", 2.5, "raw")
        for m in _ENV_RE.finditer(text):
            add(m.group(1), "env", 1.2, "env")
        for m in _MODEL_RE.finditer(text):
            add(m.group(1), "model", 1.0, "model")

    # ---- 第 2 层：结构解析（堆栈函数 / 键值对 / ACL 错误码 / 版本形态）----
    from .error_parse import parse_error_text

    for t in parse_error_text(text):
        kind_map = {
            "stack_func": "stack_func", "stack_file": "module", "exc_type": "phrase",
            "kv": "op", "module": "module", "acl_code": "errcode", "version": "version",
        }
        add(t.text, kind_map.get(t.kind, "phrase"), t.weight, t.origin)

    # ---- 第 3 层：通用短语兜底（少量，仅覆盖结构解析抓不到的自然语言错误）----
    lowered = text.lower()
    for p in _PHRASES:
        if p in lowered:
            add(p, "phrase", 2.0, "phrase")

    # ---- 社区高频信号词：只透传标注（agent 判断），不参与过滤/加权 ----
    if signal_words:
        for w in signal_words:
            if w.lower() in lowered:
                add(w, "signal", 0.6, "signal[kb高频]")

    return _dedupe(sigs)


def format_signatures(sigs: list[Signature]) -> str:
    if not sigs:
        return "(未提取到签名)"
    lines = []
    for s in sigs:
        lines.append(f"  [{s.kind}] {s.text}  (w={s.weight:.1f}{f', {s.origin}' if s.origin else ''})")
    return "\n".join(lines)


# ---------------- 精确检索 ----------------

@dataclass
class SignatureHit:
    doc_id: str
    title: str
    url: str
    hit_signatures: list[str] = field(default_factory=list)
    score: float = 0.0


def _fts_phrase_sql(conn, sig: str, limit: int) -> list[tuple[str, str, str]]:
    """FTS5 短语精确查询：命中返回 (chunk_id, doc_id, text)。"""
    # 转义 FTS 特殊字符，作为短语查询
    esc = sig.replace('"', '""')
    try:
        rows = conn.execute(
            "SELECT chunk_id, doc_id, text FROM chunks_fts "
            "WHERE chunks_fts MATCH ? LIMIT ?",
            (f'"{esc}"', limit),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]
    except Exception:
        return []


def signature_search(
    conn,
    signatures: list[Signature],
    top_k: int = 15,
    per_sig_limit: int = 30,
    component: Optional[str] = None,
) -> list[SignatureHit]:
    """用签名做精确检索：FTS 短语命中，按命中签名加权聚合。

    conn: 只读 SQLite 连接（含 chunks_fts）。
    评分策略：
    - 签名命中 chunk 加权分（signature.weight）；
    - **标题命中加成**：同一 doc 的标题文本若包含该签名，额外 +50%——标题含算子的
      issue 是"直接相关问题"，正文提到算子名的只是"提及"（噪音多）；
    - 返回按 score 排序的文档命中列表。
    """
    if not signatures:
        return []

    # 预取每个 doc 的标题（用于标题命中加成）
    doc_titles: dict[str, str] = {}
    for row in conn.execute("SELECT source_id, title FROM docs"):
        doc_titles[row[0]] = (row[1] or "").lower()

    # doc_id -> {chunk_id, sigs:set, score}
    agg: dict[str, dict] = {}
    # 每个签名对同一 doc 只计一次分（防正文多次贴报错虚高）；标题命中加成
    for sig in signatures:
        sig_doc_score: dict[str, float] = {}
        for chunk_id, doc_id, text in _fts_phrase_sql(conn, sig.text, per_sig_limit):
            if doc_id not in agg:
                agg[doc_id] = {"chunk_id": chunk_id, "sigs": set(), "score": 0.0}
            agg[doc_id]["sigs"].add(sig.text)
            add = sig.weight
            if sig.text in doc_titles.get(doc_id, ""):
                add *= 1.5
            sig_doc_score[doc_id] = max(sig_doc_score.get(doc_id, 0.0), add)
        for doc_id, sc in sig_doc_score.items():
            agg[doc_id]["score"] += sc

    # 查元数据（title/url/component）
    hits: list[SignatureHit] = []
    for doc_id, info in agg.items():
        row = conn.execute(
            "SELECT title, url, component FROM docs WHERE source_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            continue
        if component and row[2] != component:
            continue
        hits.append(
            SignatureHit(
                doc_id=doc_id,
                title=row[0] or "",
                url=row[1] or "",
                hit_signatures=sorted(info["sigs"]),
                score=round(info["score"], 3),
            )
        )
    hits.sort(key=lambda h: (-h.score, h.doc_id))
    return hits[:top_k]


def format_hits(hits: list[SignatureHit]) -> str:
    if not hits:
        return "(精确检索无命中)"
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] score={h.score:.2f} 命中签名: {', '.join(h.hit_signatures[:6])}")
        lines.append(f"    {h.title}")
        lines.append(f"    {h.url}")
    return "\n".join(lines)
