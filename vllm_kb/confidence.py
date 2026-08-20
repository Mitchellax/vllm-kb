"""置信度模型（查询时动态计算，无后台任务）。

confidence = w_time * (alpha * w_ver + beta * w_rel)
final      = sim^gamma * confidence^(1-gamma)

三个因子全部由元数据（时间戳、版本区间、状态）+ 查询参数（目标版本、当前时间）现场算出：
- w_time  时间衰退：半衰期模型，修复类知识从 resolved_at 起算，floor 保底；
- w_ver   版本相关性：知识版本区间 [min, max] 与目标部署版本 V 的距离衰减；
- w_rel   来源可靠度：已修复/官方文档 > 已关闭 > open 讨论。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import ConfidenceCfg

_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_version(v: Optional[str]) -> Optional[tuple[int, int, int]]:
    """'v0.6.4' / '0.6.4.post1' / '0.6' -> (0, 6, 4)。"""
    if not v:
        return None
    m = _VERSION_RE.search(str(v))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def version_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """版本距离（单位：小版本数）。0.6.4 与 0.6.1 差 0.3 -> 0.3；0.6.x 与 0.7.x 差 1。"""
    return (
        abs(a[0] - b[0]) * 10.0
        + abs(a[1] - b[1])
        + abs(a[2] - b[2]) * 0.1
    )


def load_release_calendar(path: Optional[str | Path]) -> Optional[dict[str, datetime]]:
    """版本日历：{"v0.6.4": "2024-08-01T00:00:00Z", ...}（tag -> 发布日期）。

    兼容两种格式：
    1. 旧格式：{"tag": "iso", ...}（dict 直存）；
    2. 新格式（build_release_calendar.py 生成）：{"releases": [{"tag","date","prerelease","kind"}...]}。
    Phase 1 由 GitHub Releases API 生成；Phase 0 可留空（返回 None）。
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    cal = {}
    if isinstance(data, dict) and "releases" in data:
        for rel in data["releases"]:
            dt = parse_iso(rel.get("date"))
            if dt and rel.get("tag"):
                cal[rel["tag"]] = dt
    else:
        for tag, iso in data.items():
            if tag == "generated_at" or tag == "repo":
                continue
            dt = parse_iso(iso)
            if dt:
                cal[tag] = dt
    return cal or None


def load_release_meta(path: Optional[str | Path]) -> Optional[dict]:
    """加载完整版本日历元数据（含 kind/prerelease），用于版本形态判断。

    返回 {"tag": {"date": iso, "kind": "release|rc|pre", "prerelease": bool}, ...}。
    文件不存在或格式不支持时返回 None。
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or "releases" not in data:
        return None
    meta = {}
    for rel in data["releases"]:
        tag = rel.get("tag")
        if tag:
            meta[tag] = {
                "date": rel.get("date", ""),
                "kind": rel.get("kind", "release"),
                "prerelease": bool(rel.get("prerelease")),
            }
    return meta or None


def version_kind(meta: Optional[dict], version: str) -> str:
    """版本形态判断：release（正式）| rc（预发布）| pre | unknown。

    用于故障分析：部署在正式版 vs rc 版，影响"该修复是否 backport 到我的版本"的判断。
    """
    if not meta:
        return "unknown"
    v = version.lower()
    # 精确匹配 tag
    for tag, info in meta.items():
        if tag.lower() == v:
            return info.get("kind", "unknown")
    # 去掉 v 前缀后匹配
    vv = v.lstrip("v")
    for tag, info in meta.items():
        if tag.lower().lstrip("v") == vv:
            return info.get("kind", "unknown")
    return "unknown"


def version_at_date(calendar: Optional[dict[str, datetime]], dt: datetime) -> Optional[str]:
    """给定日期，返回该日期之前最近已发布的版本 tag（用于把 resolved_at 映射为版本上界）。"""
    if not calendar:
        return None
    best_tag, best_dt = None, None
    for tag, release_dt in calendar.items():
        if release_dt <= dt and (best_dt is None or release_dt > best_dt):
            best_tag, best_dt = tag, release_dt
    return best_tag


def time_weight(
    created_at: Optional[str],
    resolved_at: Optional[str],
    now: Optional[datetime] = None,
    half_life_days: float = 365.0,
    floor: float = 0.15,
) -> float:
    """时间衰退：w = floor + (1-floor) * 2^(-dt/HL)。

    t_ref 取 resolved_at（结论确定日起衰退），无则取 created_at，都无则返回 1.0。
    """
    t_ref = parse_iso(resolved_at) or parse_iso(created_at)
    if t_ref is None:
        return 1.0
    now = now or datetime.now(timezone.utc)
    days = max(0.0, (now - t_ref).total_seconds() / 86400.0)
    return floor + (1.0 - floor) * (2.0 ** (-days / half_life_days))


def version_weight(
    span_min: Optional[str],
    span_max: Optional[str],
    target_version: Optional[str],
    sigma: float = 1.5,
    unknown_weight: float = 0.5,
) -> float:
    """版本相关性 w_ver。

    区间语义：
      [min, max] 均给出 -> V 在区间内 1.0；V<min 或 V>max 按版本距离衰减；
      只有 min      -> V >= min 视为适用 1.0；V < min 衰减；
      只有 max      -> V <= max 视为适用 1.0；V > max 衰减（修复已落地，目标版本可能已无此问题）；
      都无          -> unknown_weight。
    """
    tv = parse_version(target_version)
    if tv is None:
        return unknown_weight
    lo = parse_version(span_min)
    hi = parse_version(span_max)
    if lo is None and hi is None:
        return unknown_weight

    if lo is not None and tv < lo:
        return _exp_decay(version_distance(tv, lo), sigma)
    if hi is not None and tv > hi:
        return _exp_decay(version_distance(tv, hi), sigma)
    return 1.0


def _exp_decay(dist: float, sigma: float) -> float:
    return 2.0 ** (-dist / sigma) if sigma > 0 else 0.0


def reliability_score(
    source_type: str,
    status: str,
    resolved_at: Optional[str],
    explicit: Optional[float] = None,
    cfg: Optional[ConfidenceCfg] = None,
    kind: Optional[str] = None,
    verification: Optional[str] = None,
) -> float:
    """来源可靠度 w_rel。explicit（文档自带 reliability）优先，否则按类型/状态规则。

    kind（issue 类型，github_pull 按标题前缀识别）：故障知识库中 bug/fix 权威，
    doc/feature/rfc 反馈类降权（文档反馈、需求讨论对"解决当前故障"参考价值低）。

    verification（验证状态，维度 B）：expert/tested/unverified 作为**可靠度下限提升**
    （max 融合）——官方手册/专家认证文档即使 status=open 也应高可靠。
    """
    if explicit is not None:
        return explicit
    table = cfg.reliability if cfg else ConfidenceCfg().reliability
    if source_type in ("doc", "official_doc", "release_note"):
        base = table.get("official_doc", 0.85)
    elif source_type == "wiki":
        base = table.get("wiki", 0.7)
    elif source_type == "discussion":
        base = table.get("discussion", 0.5)
    elif source_type in ("github_issue", "github_pr"):
        # Phase 0 近似：closed 且有结论时间 -> 0.6；Phase 2 接入图后，
        # 检测到"已修复(有合并 PR)"时用 merged_fix=0.9 覆盖。
        if status == "merged":
            base = table.get("merged_fix", 0.9)
        elif status == "closed" and resolved_at:
            base = table.get("closed", 0.6)
        else:
            base = table.get("open", 0.4)
        # 反馈类 issue（文档反馈/需求/RFC）降权：对故障排查参考价值低
        adjust = _KIND_ADJUST.get(kind or "other", 1.0)
        base = base * adjust
    else:
        base = table.get("open", 0.4)
    # 验证状态因子：expert/tested 提升，unverified 小幅保底（不降权已有规则分）
    if verification:
        vf = (cfg.verification_weights if cfg else ConfidenceCfg().verification_weights).get(verification)
        if vf is not None:
            return max(base, vf)
    return base


# 反馈类 issue 降权系数（bug/fix 为故障知识金标准，不降权）
_KIND_ADJUST = {"bug": 1.0, "fix": 1.0, "other": 1.0, "doc": 0.6, "feature": 0.7, "rfc": 0.7}


@dataclass
class ConfidenceBreakdown:
    score: float
    time_weight: float
    version_weight: float
    reliability: float
    target_version: str
    now: str
    extras: dict = field(default_factory=dict)


def compute_confidence(
    created_at: Optional[str],
    resolved_at: Optional[str],
    status: str,
    source_type: str,
    span_min: Optional[str],
    span_max: Optional[str],
    target_version: Optional[str],
    now: Optional[datetime] = None,
    cfg: Optional[ConfidenceCfg] = None,
    explicit_reliability: Optional[float] = None,
    kind: Optional[str] = None,
    verification: Optional[str] = None,
) -> ConfidenceBreakdown:
    c = cfg or ConfidenceCfg()
    w_t = time_weight(created_at, resolved_at, now, c.half_life_days, c.time_floor)
    w_v = version_weight(span_min, span_max, target_version, c.version_sigma, c.unknown_version_weight)
    w_r = reliability_score(source_type, status, resolved_at, explicit_reliability, c, kind,
                            verification=verification)
    score = w_t * (c.alpha * w_v + c.beta * w_r)
    now_str = (now or datetime.now(timezone.utc)).isoformat()
    extras = {}
    if verification:
        extras["verification"] = verification
    return ConfidenceBreakdown(
        score=score,
        time_weight=round(w_t, 4),
        version_weight=round(w_v, 4),
        reliability=round(w_r, 4),
        target_version=target_version or "(未指定)",
        now=now_str,
        extras=extras,
    )


def final_score(similarity: float, confidence: float, gamma: float = 0.6) -> float:
    """检索排序分 = sim^gamma * confidence^(1-gamma)。"""
    sim = max(0.0, min(1.0, similarity))
    conf = max(0.0, min(1.0, confidence))
    if sim == 0 and conf == 0:
        return 0.0
    return (sim ** gamma) * (conf ** (1.0 - gamma))
