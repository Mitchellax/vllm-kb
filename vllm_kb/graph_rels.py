"""图关系确定性抽取（零 LLM，可重放）。

从 canonical 记录 + 版本日历抽取图节点/边，全部为确定性规则：
- FIXES (PR → Issue)：正文 "fixes/closes/resolves #N"（同 repo 编号），支持 "owner/repo#N" 跨 repo；
- MERGED_IN (PR → Release)：PR.merged_at 落在版本日历区间 → 合并后首次发布的 tag（含 rc/pre）；
- MENTIONS (Issue/PR → Operator/ErrorCode/Model/Version)：复用 signature.py 三层签名提取
  的 kernel/op/errcode/model/version 类签名；另补 version_span 与正文组件版本。

设计约束：
- 纯函数、无 IO（除 load_release_calendars），便于单元测试；
- 与来源类型解耦：任何来源只要产出 canonical（含 source_id/body/extra）即可入图。
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import AppConfig

# ---------------- FIXES：issue 引用抽取 ----------------

# 同 repo：fixes/closes/resolves/addresses + #N（可带 issue/bug 前缀）
_FIX_VERB = r"(?:fix(?:es|ed)?|close(?:s|d)?|resolve(?:s|d)?|address(?:es|ed)?)"
_REF_SAME_REPO = re.compile(
    rf"(?i)\b{_FIX_VERB}\s+(?:issue\s+|bug\s+)?#(\d+)\b"
)
# 跨 repo：fixes owner/repo#N（含完整 URL 形式）
_REF_CROSS_REPO = re.compile(
    rf"(?i)\b{_FIX_VERB}\s+(?:https?://github\.com/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)#(\d+)\b"
)
# 反向引用（issue 侧："fixed by #N" / "will be fixed in #N"）——同样收进 FIXES（方向反转为 PR→Issue）
_REF_REVERSE_RE = re.compile(
    rf"(?i)(?:fixed|fix|closed|resolved)\s+by\s+#(\d+)\b"
)

# ---------------- MERGED_IN：PR 合并时间 → 发布版本 ----------------

_ISO_DATE = "%Y-%m-%dT%H:%M:%S"


@dataclass
class ReleaseInfo:
    """版本日历中的一条发布（按 repo 分组使用）。"""

    tag: str
    date: str  # ISO8601
    kind: str  # release | rc | pre
    prerelease: bool = False


@dataclass
class GraphExtract:
    """一条 canonical 记录抽取出的图关系（供建图脚本消费）。"""

    source_id: str = ""
    repo: str = ""
    number: int = 0
    source_type: str = ""  # github_issue | github_pr | ...
    fixes: list[tuple[str, int]] = field(default_factory=list)  # 正向：本 PR 修复的目标 (repo, number)
    fixed_by: list[tuple[str, int]] = field(default_factory=list)  # 反向：修复本文档的 PR (repo, number)
    mentions: dict[str, set[str]] = field(default_factory=dict)  # kind -> 实体值集合
    merged_at: str = ""  # PR 用


def extract_fix_refs(text: str, repo: str) -> list[tuple[str, int]]:
    """抽取**正向**修复引用：当前文档修复的目标 (target_repo, number)。

    - 同 repo："fixes #12345" -> (repo, 12345)
    - 跨 repo："fixes vllm-project/vllm#12345" -> ("vllm-project/vllm", 12345)
    典型出现于 PR body。返回去重列表。
    """
    out: set[tuple[str, int]] = set()
    for m in _REF_SAME_REPO.finditer(text):
        out.add((repo, int(m.group(1))))
    for m in _REF_CROSS_REPO.finditer(text):
        out.add((m.group(1), int(m.group(2))))
    return sorted(out)


def extract_fixed_by_refs(text: str, repo: str) -> list[tuple[str, int]]:
    """抽取**反向**修复引用：修复当前文档的 PR 编号 (repo, number)。

    典型出现于 issue body（"fixed by #9876" / "will be fixed in #9876"）。
    返回去重列表。
    """
    out: set[tuple[str, int]] = set()
    for m in _REF_REVERSE_RE.finditer(text):
        out.add((repo, int(m.group(1))))
    return sorted(out)


def map_merged_to_release(merged_at: str, releases: list[ReleaseInfo]) -> Optional[str]:
    """PR 合并时间 → 合并后**首次发布**的 tag（含 rc/pre）；晚于所有发布 → None（尚未发布）。

    releases 按 date 升序传入。语义：修复合并进 main 后，随下一个 tag 提供给用户。
    """
    if not merged_at or not releases:
        return None
    target = merged_at[: len(_ISO_DATE)]  # 截到秒，忽略时区后缀差异（UTC 均以 Z 结尾）
    for r in releases:
        if r.date[: len(_ISO_DATE)] >= target:
            return r.tag
    return None


def load_release_calendars(cfg: Optional["AppConfig"] = None) -> dict[str, list[ReleaseInfo]]:
    """加载全部版本日历（按 repo 分组）。

    扫描 storage.release_calendar 同目录下的 release_calendar*.json（含 per-repo 文件）。
    返回 {repo: [ReleaseInfo(date 升序)]}。
    """
    if cfg is not None:
        calendar_path = cfg.resolve(cfg.storage.release_calendar or "data/compatibility/release_calendar.json")
    else:
        calendar_path = Path("data/compatibility/release_calendar.json")
    calendar_dir = calendar_path.parent
    calendars: dict[str, list[ReleaseInfo]] = {}
    for f in sorted(calendar_dir.glob("release_calendar*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        repo = data.get("repo", "")
        if not repo or not data.get("releases"):
            continue
        rels = [ReleaseInfo(tag=r["tag"], date=r["date"], kind=r.get("kind", "release"),
                            prerelease=bool(r.get("prerelease", False))) for r in data["releases"]]
        rels.sort(key=lambda r: r.date)
        calendars[repo] = rels
    return calendars


# ---------------- MENTIONS：签名 → 实体 ----------------

_ENTITY_KIND_MAP = {
    "kernel": "operator",
    "op": "operator",
    "errcode": "error_code",
    "model": "model",
    "version": "version",
}

# operator 实体只收纯标识符形态（dispatch_ffn_combine / halMemCreate / aclnnXxx）；
# 排除 kv→op 的日志片段（"ez9999: inner error!"、"get reginfo failed" 等含空格/冒号/标点）。
_OP_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{2,63}$")

# 实体 id 禁用字符：Kùzu COPY/查询对反斜杠、逗号、引号敏感，一律替换为下划线
_BAD_ENTITY_CHARS = str.maketrans({'\\': '_', ',': '_', '"': '_', "'": '_'})


def _clean_entity(v: str) -> str:
    """实体值规范化：去首尾空白/引号，内部空白（含换行）归一为单空格，
    反斜杠/逗号/引号替换为下划线（Kùzu CSV 与错误消息安全）。

    error_parse 的 _VER_RE 用 \\s 匹配组件版本，可能跨行捕获
    （如 "vllm-ascend\\n0.23.0"）；_KV_RE 可能捕获含字面 \\n 的日志片段
    （如 "561103.\\n\\n[error] ..."）——实体节点 id 必须为单行且无特殊字符。
    """
    v = v.strip().strip("\"'")
    v = " ".join(v.split())
    return v.translate(_BAD_ENTITY_CHARS)


def signature_mentions(text: str, symbol_table=None, signal_words: Optional[list] = None) -> dict[str, set[str]]:
    """复用三层签名提取，按实体类型归类 MENTIONS 目标。

    返回 {entity_kind: {实体值}}，entity_kind ∈ operator | error_code | model | version。
    phrase/env 类签名 V1 不入图（可作 FTS 检索信号，图内价值低）。
    """
    from .signature import extract_signatures

    mentions: dict[str, set[str]] = {}
    for sig in extract_signatures(text, symbol_table=symbol_table, signal_words=signal_words):
        kind = _ENTITY_KIND_MAP.get(sig.kind)
        if kind is None or len(sig.text) < 3:
            continue
        v = _clean_entity(sig.text)
        # operator 只收合法标识符（日志片段/键值对值不是算子实体）
        if kind == "operator" and not _OP_IDENT_RE.match(v):
            continue
        mentions.setdefault(kind, set()).add(v)
    return mentions


def extract_doc_relations(
    source_id: str,
    repo: str,
    number: int,
    source_type: str,
    body: str,
    version_span_min: Optional[str] = None,
    version_span_max: Optional[str] = None,
    symbol_table=None,
    signal_words: Optional[list] = None,
) -> GraphExtract:
    """一条 canonical 记录的完整关系抽取（FIXES + MENTIONS）。"""
    ex = GraphExtract(source_id=source_id, repo=repo, number=number, source_type=source_type)
    if body:
        ex.fixes = extract_fix_refs(body, repo)
        ex.fixed_by = extract_fixed_by_refs(body, repo)
        ex.mentions = signature_mentions(body, symbol_table=symbol_table, signal_words=signal_words)
        # 正文显式组件版本（"vllm 0.18.0" / "CANN 8.5.1"）→ version 实体
        try:
            from .components import extract_component_versions

            for comp, ver in extract_component_versions(body).items():
                if ver:
                    ex.mentions.setdefault("version", set()).add(_clean_entity(ver))
        except Exception:
            pass
    # version_span 区间端点 → version 实体（该 doc 适用的版本）
    for v in (version_span_min, version_span_max):
        if v:
            ex.mentions.setdefault("version", set()).add(_clean_entity(v))
    return ex
