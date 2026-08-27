"""文档级标签（两级分类）确定性提取核心。

- **词典**（config.tags.registry）为全局唯一事实源：条目 {name, tier: domain|purpose}；
- **两级分类**：
  * 主题/领域类（domain）：回答"这是什么领域的知识"（HCCL、网络、NPU、CANN、算子…），用于过滤/圈定范围；
  * 具体作用类（purpose）：回答"这篇文档能帮我做什么"（超时排查、命令参考、错误码表…），用于能力/动作匹配；
- **提取输入**：文件名 stem + 内部标题列表（PDF 编号标题 / Markdown `#` 标题由来源侧收集后传入）；
- **提取规则**（确定性、可重放，零 LLM）：
  1. 词典子串命中（文件名/标题含词典词）→ 命中即标签，tier 取词典值；
  2. 文件名/标题中的拉丁词 token（`[A-Za-z][A-Za-z0-9_.-]{1,}`，过滤停用词/版本号）→ 候选；
  3. 短标题本身（≤ heading_max_chars 字符、清洗后）→ 候选；
- **tier 判定**：词典命中 → 词典 tier；未收录候选 → 启发式（含 purpose 信号词 → purpose，否则 domain）；
- **输出**：(确定标签, 未收录强候选)——候选供审核队列 tag_candidate 人工采纳（采纳后入词典）；
- **合并公式唯一实现**：`final = (auto − excluded) ∪ manual`（ingest 与 build_graph 共用本模块，杜绝两处分叉）；
- **jieba 可选**：已安装时对中文做分词增强并注册词典词防拆分；未安装自动回退规则模式，主链路不受影响。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import AppConfig

TIER_DOMAIN = "domain"  # 主题/领域类
TIER_PURPOSE = "purpose"  # 具体作用类

# 拉丁词 token：HCCL / npu / hccn_tool / GLM5.1 等（文件名与标题中的组件/型号名）
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{0,63}")
# 版本号形态（v1.0 / v0.23.0rc1 / 2024.1）不作为标签
_VERSION_RE = re.compile(r"^v?\d+(\.\d+){1,3}[a-z0-9]*$", re.I)
# PDF 编号标题行："2.34 获取network版本号信息"（与 chunking 同源规则）。
# 编号限制 1~3 位数字每段（"2.34"），排除错误码/表格行（如 "507014 device busy" 的 5-6 位纯数字）。
_PDF_SECTION_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3}){0,3})\s+(\S.{0,60}?)\s*$")
# Markdown 标题行：^#{1,6} 标题（可带结尾 #）
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
# 标题内行内元素清洗：链接 [x](url) → x；行内代码 `x` → x
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_CODE_RE = re.compile(r"`([^`]*)`")

# purpose 启发式信号词（未收录候选含这些词 → 具体作用类）
_DEFAULT_PURPOSE_SIGNALS = (
    "排查", "定位", "调优", "优化", "命令", "参考", "指南", "部署", "配置",
    "故障", "安装", "测试", "监控", "评估", "说明", "设计", "清单", "流程",
    "步骤", "规范", "手册", "诊断", "恢复", "升级", "运维", "验收", "导出", "导入",
    "错误码", "码表", "对照表", "参数表",
)
_PURPOSE_SIGNAL_RE = re.compile("|".join(_DEFAULT_PURPOSE_SIGNALS))


@dataclass
class TagEntry:
    """词典条目：标签名 + 层级（domain | purpose）。"""

    name: str
    tier: str = TIER_DOMAIN


def normalize_tag(name: str) -> str:
    """标签规范化：去首尾空白/包裹符号，内部空白归一，截断超长。"""
    name = (name or "").strip().strip("\"'`*#").strip()
    name = re.sub(r"\s+", " ", name)
    if len(name) > 64:
        name = name[:64]
    return name


def tier_for(name: str) -> str:
    """启发式 tier 判定：含 purpose 信号词 → purpose，否则 domain。"""
    if _PURPOSE_SIGNAL_RE.search(name or ""):
        return TIER_PURPOSE
    return TIER_DOMAIN


# ---------------- 标题提取（来源侧共用，按来源选择实现） ----------------

def headings_from_pdf(text: str) -> list[str]:
    """PDF 编号标题行（'2.34 获取network版本号信息'），返回去编号的标题文本。

    排除目录点线填充行与超长行（正文标题行短）；目录连续标题不在这里去重
    （标签提取只需标题集合，去重由上层候选合并处理）。
    """
    out: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if len(s) > 80 or "...." in s or "…" in s or "·" in s:
            continue
        m = _PDF_SECTION_RE.match(s)
        if m:
            out.append(m.group(2).strip())
    return out


def headings_from_markdown(text: str) -> list[str]:
    """Markdown 标题行（'## 命令格式'），清洗行内链接与行内代码。"""
    out: list[str] = []
    for line in (text or "").splitlines():
        m = _MD_HEADING_RE.match(line)
        if not m:
            continue
        h = m.group(1).strip()
        h = _MD_LINK_RE.sub(r"\1", h)
        h = _MD_CODE_RE.sub(r"\1", h)
        h = h.strip()
        if h:
            out.append(h)
    return out


# ---------------- 词典（registry） ----------------

class TagRegistry:
    """标签词典：config.tags.registry 的加载/查询/修改（保存到 config 由调用方负责）。

    registry 为全局唯一事实源；tier 修改后全库文档的展示层级随标签全局生效。
    """

    def __init__(self, entries: Optional[list[TagEntry]] = None):
        self.entries: list[TagEntry] = list(entries or [])

    # ---------- 构造 ----------

    @classmethod
    def load(cls, cfg: Optional["AppConfig"] = None) -> "TagRegistry":
        """从 config.tags.registry 加载（无 tags 配置段时为空词典）。

        兼容两种条目形态：{"name": "...", "tier": "..."} 与裸字符串（tier 走启发式）。
        """
        entries: list[TagEntry] = []
        raw: list = []
        if cfg is not None:
            raw = list(getattr(cfg.tags, "registry", []) or [])
        for item in raw:
            if isinstance(item, str):
                name = normalize_tag(item)
                if name:
                    entries.append(TagEntry(name=name, tier=tier_for(name)))
            elif isinstance(item, dict):
                name = normalize_tag(str(item.get("name", "")))
                if name:
                    tier = str(item.get("tier", "") or "")
                    if tier not in (TIER_DOMAIN, TIER_PURPOSE):
                        tier = tier_for(name)
                    entries.append(TagEntry(name=name, tier=tier))
        # 同名去重（后出现的覆盖）
        merged: dict[str, TagEntry] = {}
        for e in entries:
            merged[e.name] = e
        return cls(list(merged.values()))

    # ---------- 查询 ----------

    def by_name(self) -> dict[str, TagEntry]:
        return {e.name: e for e in self.entries}

    def contains(self, name: str) -> bool:
        return normalize_tag(name) in self.by_name()

    def tier(self, name: str) -> str:
        return self.by_name().get(normalize_tag(name), TagEntry(name="", tier=TIER_DOMAIN)).tier

    def match(self, text: str) -> list[TagEntry]:
        """子串命中：文本（文件名+标题拼接）含词典词 → 返回命中条目（大小写不敏感）。"""
        lowered = (text or "").lower()
        hits: list[TagEntry] = []
        for e in self.entries:
            if e.name and e.name.lower() in lowered:
                hits.append(e)
        return hits

    # ---------- 修改 ----------

    def add(self, name: str, tier: Optional[str] = None) -> TagEntry:
        """新增或更新词典条目（同名更新 tier）。返回条目。"""
        name = normalize_tag(name)
        if not name:
            raise ValueError("标签名不能为空")
        if tier not in (TIER_DOMAIN, TIER_PURPOSE):
            tier = tier_for(name)
        merged = self.by_name()
        merged[name] = TagEntry(name=name, tier=tier)
        self.entries = list(merged.values())
        return merged[name]

    def rename(self, old: str, new: str) -> Optional[TagEntry]:
        """改名（保留 tier；新名已存在时后者覆盖）。返回新条目。"""
        old = normalize_tag(old)
        new = normalize_tag(new)
        if not old or not new or old == new:
            return None
        merged = self.by_name()
        entry = merged.pop(old, None)
        if entry is None:
            return None
        entry.name = new
        merged[new] = entry
        self.entries = list(merged.values())
        return entry

    def set_tier(self, name: str, tier: str) -> bool:
        if tier not in (TIER_DOMAIN, TIER_PURPOSE):
            raise ValueError(f"非法 tier: {tier}")
        entry = self.by_name().get(normalize_tag(name))
        if entry is None:
            return False
        entry.tier = tier
        return True

    def remove(self, name: str) -> bool:
        merged = self.by_name()
        if normalize_tag(name) not in merged:
            return False
        merged.pop(normalize_tag(name))
        self.entries = list(merged.values())
        return True

    # ---------- 序列化 ----------

    def to_config_list(self) -> list[dict]:
        return [{"name": e.name, "tier": e.tier} for e in self.entries]


def save_registry_to_config(cfg: "AppConfig", registry: TagRegistry,
                            config_path: Optional[str | Path] = None) -> None:
    """把词典写回 config.json 的 tags.registry（审核页新增/改名/改 tier 后同步）。"""
    from .config import PROJECT_ROOT

    p = Path(config_path) if config_path else PROJECT_ROOT / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"config.json 不存在: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    tags = data.setdefault("tags", {})
    tags["registry"] = registry.to_config_list()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------- 自动提取 ----------------

def extract_tags(
    stem: str,
    headings: Optional[list[str]] = None,
    registry: Optional[TagRegistry] = None,
    stopwords: Optional[list[str]] = None,
    min_len: int = 2,
    heading_max_chars: int = 12,
) -> tuple[list[TagEntry], list[TagEntry]]:
    """从文件名 stem + 内部标题提取 (确定标签, 未收录强候选)。

    - 确定标签：词典子串命中（tier 取词典值）；
    - 未收录候选：拉丁词 token + 短标题（tier 启发式）——进审核队列 tag_candidate 人工采纳。
    返回顺序稳定（词典顺序 → 候选按出现顺序），便于测试与重放。
    """
    registry = registry or TagRegistry()
    stopwords = {s.lower() for s in (stopwords or [])}
    headings = [normalize_tag(h) for h in (headings or []) if normalize_tag(h)]
    # 词典子串匹配用原始文件名（hccn_tool 等连写词不被拆分）；token 提取用拆分后的文件名
    match_text = " ".join([stem or ""] + headings)
    token_text = " ".join([re.sub(r"[_\-\.]+", " ", stem or "")] + headings)

    matched: list[TagEntry] = []
    matched_names: set[str] = set()
    for e in registry.match(match_text):
        if e.name not in matched_names:
            matched.append(e)
            matched_names.add(e.name)

    candidates: list[TagEntry] = []
    cand_seen: set[str] = set()

    def _add_candidate(name: str, tier: Optional[str] = None) -> None:
        name = normalize_tag(name)
        if not name or name in matched_names or name in cand_seen:
            return
        if len(name) < min_len:
            return
        cand_seen.add(name)
        candidates.append(TagEntry(name=name, tier=tier or tier_for(name)))

    # 拉丁词 token（文件名拆分 + 标题）
    for t in _LATIN_TOKEN_RE.findall(token_text):
        t = t.strip("._-")
        if not t or len(t) < min_len or t.lower() in stopwords:
            continue
        if _VERSION_RE.match(t):
            continue
        _add_candidate(t)
    # 短标题本身
    for h in headings:
        if 0 < len(h) <= heading_max_chars:
            _add_candidate(h)

    return matched, candidates


# ---------------- 合并公式（唯一实现） ----------------

def merge_final(auto: list[str], excluded: list[str], manual: list[str]) -> list[str]:
    """最终标签 = (自动 − 排除) ∪ 人工。

    保序去重：自动标签按原顺序，人工标签追加在尾部（同名不重复）。
    ingest 与 build_graph 均调用本函数，杜绝合并逻辑两处分叉。
    """
    excluded_set = set(excluded or [])
    out: list[str] = []
    seen: set[str] = set()
    for t in (auto or []):
        if t in excluded_set or t in seen:
            continue
        seen.add(t)
        out.append(t)
    for t in (manual or []):
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out
