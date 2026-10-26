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

# 拉丁停用词默认集（候选降噪 2026-10：真实积压里 for/apt-get 类无检索价值；cfg.stopwords 可追加）
_DEFAULT_LATIN_STOPWORDS = frozenset((
    "for", "of", "and", "the", "with", "from", "to", "in", "on", "or", "by", "as", "at",
    "is", "are", "via", "use", "using", "based", "new", "guide", "manual", "doc", "docs",
    "case", "report", "log", "test", "demo",
))
# 中文整词停用词（2026-10 业务积压：候选里大量"排查/说明/常见问题"类标题模式词——
# **整词等于才滤**："HCCL 超时排查"含"排查"但是有价值短语，不受影响；cfg.stopwords 同样生效）
_DEFAULT_CJK_STOPWORDS = frozenset((
    "排查", "问题", "常见问题", "说明", "介绍", "汇总", "记录", "日志", "分析", "总结",
    "笔记", "注意事项", "方案", "方法", "步骤", "示例", "演示", "教程", "学习", "实践",
    "参考", "手册", "指南", "使用说明", "操作说明", "解决方案", "处理方法", "问题记录",
))
# 纯数字不是标签（版本片段/日期/序号）；带字母前缀的型号（A2/910B）不受影响
_PURE_NUM_RE = re.compile(r"^\d+$")


@dataclass
class TagEntry:
    """词典条目：标签名 + 层级（domain | purpose）。"""

    name: str
    tier: str = TIER_DOMAIN


# 首尾句读标点（中西文）：heading 提取的候选词常带句尾标点（"复位芯片。"）——不剥则
# 与 "复位芯片" 生成两个 key、两个候选。只剥首尾（中间标点保留，版本号 v0.11.0 不伤）。
_EDGE_SENT_PUNCT_RE = re.compile(r"^[\s。．,，、;；:：!！?？·・.]+|[\s。．,，、;；:：!！?？·・.]+$")


def normalize_tag(name: str) -> str:
    """标签规范化（**显示级**，保留原形大小写）：去首尾空白/包裹符号/句读标点，内部
    空白归一，CJK↔拉丁边界空格剥离，截断超长。

    CJK↔拉丁边界空格剥离的原因：PDF/Word 标题里该空格不稳定（"配置pip 源"/"配置pip源"、
    "升级NPU 驱动"/"升级NPU驱动" 会生成两个候选词、占两个队列项）——归一后变体收敛，
    词典命中同步稳定（词典条目加载时同样过本函数）。纯 CJK 间/纯拉丁间的空格保留
    （"方式二 构建容器镜像" 不受影响）。
    首尾句读剥离（2026-10 业务实测："复位芯片。" 与 "复位芯片" 两个 tag）——heading
    以句子结尾时句号是文法产物不是标签的一部分。
    """
    name = (name or "").strip().strip("\"'`*#").strip()
    name = _EDGE_SENT_PUNCT_RE.sub("", name)
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[A-Za-z0-9])|(?<=[A-Za-z0-9]) (?=[\u4e00-\u9fff])",
                  "", name)
    if len(name) > 64:
        name = name[:64]
    return name


# 聚合/匹配键（**不用于显示**）：大小写、分隔符、CJK 间空格的变体在此收敛。
# 业务积压里 "Atlas A3"/"atlas_a3"/"ATLAS A3"、"超时 排查"/"超时排查" 是同一词的
# 排版变体——聚合键归一后自动合并为一条候选；词典正式名/候选显示名仍保留首见原形
# （Atlas A3、HCCL 等专有名词全小写会损害可读性）。
_CJK_KEY_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")


def tag_key(name: str) -> str:
    """标签聚合键：normalize_tag → 全角空格半角化 → `_`→`-` → 删 CJK 字符间空格 →
    剩余空白（拉丁词间）→`-` → 小写 → 压缩连字符。

    用于：候选聚合去重（extract_tags/seed_tag_candidates）、词典查重（contains/add）、
    存量候选变体合并。显示与词典 name 一律保留原形，不经本函数。
    空格/连字符/下划线三种分隔在 key 层同义（"Atlas A3"=="atlas_a3"=="Atlas-A3"）；
    首尾句读已由 normalize_tag 剥离（"复位芯片。"=="复位芯片"）；点保留（版本号 v0.11.0）。
    """
    name = normalize_tag(name)
    name = name.replace("\u3000", " ").replace("_", "-")
    name = _CJK_KEY_SPACE_RE.sub("", name)
    name = re.sub(r"\s+", "-", name.strip().lower())
    return re.sub(r"-{2,}", "-", name).strip("-")


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

    def __init__(self, entries: Optional[list[TagEntry]] = None,
                 devices: Optional[DeviceGlossary] = None):
        self.entries: list[TagEntry] = list(entries or [])
        self.devices: DeviceGlossary = devices or DeviceGlossary()

    # ---------- 构造 ----------

    @classmethod
    def _from_raw(cls, raw: list, devices_raw: Optional[list] = None) -> "TagRegistry":
        """从原始 registry 列表构造（兼容 dict 与裸字符串条目）。

        devices_raw（设备词表）一并构造：设备 name 补入词典（registry 显式条目优先），
        别名命中由 match() 归一成 canonical 设备标签。
        """
        entries: list[TagEntry] = []
        for item in raw or []:
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
        # 同 key 去重（**首见保留**，2026-10 同义变体收敛）：存量 config 里
        # "CANN版本说明" 与 "CANN 版本说明" 可能都已被人工认证入典——加载即收敛为
        # 一条（首见原形），contains/match/adopt 不再产生第二词条；config 文件本身
        # 不动，下次 save 自然写收敛后的列表（全库标签替换走 normalize_tag_variants）
        merged: dict[str, TagEntry] = {}
        for e in entries:
            merged.setdefault(tag_key(e.name), e)
        devices = DeviceGlossary._from_raw(devices_raw or [])
        # 设备 name 补入词典（registry 已有同 key → registry tier 优先）
        for de in devices.entries:
            merged.setdefault(tag_key(de.name), TagEntry(name=de.name, tier=de.tier))
        reg = cls(list(merged.values()), devices)
        reg.devices.warnings.extend(devices.warnings)
        return reg

    @classmethod
    def load(cls, cfg: Optional["AppConfig"] = None) -> "TagRegistry":
        """从 config.tags.registry 加载（无 tags 配置段时为空词典）。

        兼容两种条目形态：{"name": "...", "tier": "..."} 与裸字符串（tier 走启发式）。
        config.tags.devices（设备词表）同步加载：name 入词典 + 别名匹配可用。
        """
        raw: list = []
        devices_raw: Optional[list] = None
        if cfg is not None:
            raw = list(getattr(cfg.tags, "registry", []) or [])
            devices_raw = list(getattr(cfg.tags, "devices", []) or [])
        return cls._from_raw(raw, devices_raw)

    @classmethod
    def load_from_config_file(cls, config_path: Optional[str | Path] = None) -> "TagRegistry":
        """直接从 config.json 文件读取词典（审核页多次编辑后内存 cfg 可能滞后，以文件为准）。"""
        from .config import PROJECT_ROOT

        p = Path(config_path) if config_path else PROJECT_ROOT / "config.json"
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        tags = data.get("tags") or {}
        return cls._from_raw(tags.get("registry", []) or [], tags.get("devices", []) or [])

    # ---------- 查询 ----------

    def by_name(self) -> dict[str, TagEntry]:
        return {e.name: e for e in self.entries}

    def by_key(self) -> dict[str, TagEntry]:
        """按聚合键索引（同 key 变体 "Atlas A3"/"atlas_a3" 收敛到首条）。"""
        out: dict[str, TagEntry] = {}
        for e in self.entries:
            out.setdefault(tag_key(e.name), e)
        return out

    def contains(self, name: str) -> bool:
        return tag_key(name) in self.by_key()

    def tier(self, name: str) -> str:
        return self.by_key().get(tag_key(name), TagEntry(name="", tier=TIER_DOMAIN)).tier

    def match(self, text: str) -> list[TagEntry]:
        """子串命中：文本（文件名+标题拼接）含词典词 → 返回命中条目（大小写+分隔符不敏感）。

        归一层 tag_key 的子串版：文本与词条都做 lower + `_`→`-` 后再匹配——正文里
        "hccn_tool 使用" 能命中词典 "hccn-tool"（排版变体不应漏命中）。
        设备词表命中在词典之后：**别名命中归一为 canonical 设备标签**（文本含
        "800I A2" → A2 标签；含 910B → 910B 标签，但**不反推机型**——见 DeviceGlossary）。
        """
        lowered = (text or "").lower().replace("_", "-")
        hits: list[TagEntry] = []
        seen: set[str] = set()
        for e in self.entries:
            if e.name and e.name.lower().replace("_", "-") in lowered and e.name not in seen:
                hits.append(e)
                seen.add(e.name)
        for de in self.devices.match(text):
            if de.name not in seen:
                hits.append(TagEntry(name=de.name, tier=de.tier))
                seen.add(de.name)
        return hits

    # ---------- 修改 ----------

    def add(self, name: str, tier: Optional[str] = None) -> TagEntry:
        """新增或更新词典条目。**同 key 已有词条（"Atlas A3" vs "atlas_a3"）→ 保留已有
        原形词条**（变体不产生第二个词条；显式传 tier 时仅更新 tier）。返回条目。"""
        name = normalize_tag(name)
        if not name:
            raise ValueError("标签名不能为空")
        key = tag_key(name)
        existing = self.by_key().get(key)
        if existing is not None:
            if tier in (TIER_DOMAIN, TIER_PURPOSE) and tier != existing.tier:
                self.set_tier(existing.name, tier)
                existing = self.by_key()[key]
            return existing
        if tier not in (TIER_DOMAIN, TIER_PURPOSE):
            tier = tier_for(name)
        self.entries.append(TagEntry(name=name, tier=tier))
        return self.entries[-1]

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


# ---------------- 设备词表（config.tags.devices） ----------------

_DEVICE_KINDS = ("machine", "chip")


@dataclass
class DeviceEntry:
    """设备词表条目：机型/芯片别名归一 + 机器→芯片单向关系。

    - aliases：同义/变体归一到 name（800I A2 / 800T A2 / Atlas 800I A2 → A2——规格细微
      差异、本质同机，2026-10 领域确认）；
    - uses_chips：机器使用的芯片（A2 → 910B）。**单向**：检索同向扩展安全；芯片绝不
      反推机型（300I A2 同用 910B 但拓扑差异巨大）——反推逻辑天然不存在；
    - note：人工判断备注（为什么不能反推、规格差异点），词表是唯一承载处。
    """

    name: str
    tier: str = TIER_DOMAIN
    kind: str = "machine"
    aliases: list[str] = field(default_factory=list)
    uses_chips: list[str] = field(default_factory=list)
    note: str = ""


class DeviceGlossary:
    """设备词表：config.tags.devices 的加载/校验/别名匹配。

    为什么单独维护（而不是通用规则）：机型/芯片分型是故障检索核心轴，正则与通用
    NER 表达不了"同芯片不同拓扑"这类业务判断，只能人工定义——但只需定一次、
    几十条规模。词表词（name/alias）在候选降噪中**永久豁免**（A2/A3 不是版本号）。
    """

    def __init__(self, entries: Optional[list[DeviceEntry]] = None,
                 warnings: Optional[list[str]] = None):
        self.entries: list[DeviceEntry] = list(entries or [])
        self.warnings: list[str] = list(warnings or [])

    # ---------- 构造与校验 ----------

    @classmethod
    def _from_raw(cls, raw: list) -> "DeviceGlossary":
        """从原始 devices 列表构造。校验只告警不致命（词表错不阻塞入库）。"""
        entries: list[DeviceEntry] = []
        warnings: list[str] = []
        alias_owner: dict[str, str] = {}
        for i, item in enumerate(raw or []):
            if not isinstance(item, dict):
                warnings.append(f"devices[{i}] 非 dict 条目，已跳过")
                continue
            name = normalize_tag(str(item.get("name", "")))
            if not name:
                warnings.append(f"devices[{i}] 缺 name，已跳过")
                continue
            kind = str(item.get("kind", "") or "machine")
            if kind not in _DEVICE_KINDS:
                warnings.append(f"devices[{i}] {name}: kind={kind!r} 非法（machine|chip），按 machine 处理")
                kind = "machine"
            tier = str(item.get("tier", "") or "")
            if tier not in (TIER_DOMAIN, TIER_PURPOSE):
                tier = tier_for(name)
            aliases = [normalize_tag(a) for a in (item.get("aliases") or []) if normalize_tag(a)]
            uses_chips = [normalize_tag(c) for c in (item.get("uses_chips") or []) if normalize_tag(c)]
            if name in {e.name for e in entries}:
                warnings.append(f"devices[{i}] name 重复: {name}（后者覆盖）")
                entries = [e for e in entries if e.name != name]
            for a in [name] + aliases:
                owner = alias_owner.get(a)
                if owner is not None and owner != name:
                    warnings.append(f"devices[{i}] 别名冲突: {a} 已属 {owner}，先入先得")
                else:
                    alias_owner[a] = name
            entries.append(DeviceEntry(name=name, tier=tier, kind=kind, aliases=aliases,
                                       uses_chips=uses_chips,
                                       note=str(item.get("note", "") or "")))
        names = {e.name for e in entries}
        for e in entries:
            for c in e.uses_chips:
                if c not in names:
                    warnings.append(f"{e.name}.uses_chips 引用未定义设备: {c}")
        return cls(entries, warnings)

    @classmethod
    def load(cls, cfg: Optional["AppConfig"] = None) -> "DeviceGlossary":
        raw: list = []
        if cfg is not None:
            raw = list(getattr(cfg.tags, "devices", []) or [])
        return cls._from_raw(raw)

    @classmethod
    def load_from_config_file(cls, config_path: Optional[str | Path] = None) -> "DeviceGlossary":
        from .config import PROJECT_ROOT

        p = Path(config_path) if config_path else PROJECT_ROOT / "config.json"
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        return cls._from_raw((data.get("tags") or {}).get("devices", []) or [])

    # ---------- 匹配 ----------

    def by_alias(self) -> dict[str, DeviceEntry]:
        """alias(lower) → 条目（name 也是别名）；同别名先入先得（冲突已在构造时告警）。"""
        out: dict[str, DeviceEntry] = {}
        for e in self.entries:
            for a in [e.name] + e.aliases:
                out.setdefault(a.lower(), e)
        return out

    def match(self, text: str) -> list[DeviceEntry]:
        """别名子串命中（大小写不敏感），**长别名优先**（800I A2 整体命中优先于 A2）。"""
        lowered = (text or "").lower()
        hits: list[DeviceEntry] = []
        seen: set[str] = set()
        for a, e in sorted(self.by_alias().items(), key=lambda kv: -len(kv[0])):
            if a and a in lowered and e.name not in seen:
                hits.append(e)
                seen.add(e.name)
        return hits


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
    stopwords = {s.lower() for s in (stopwords or [])} | _DEFAULT_LATIN_STOPWORDS
    # 中文整词停用（tag_key 比对，覆盖大小写/全半角变体；cfg.stopwords 里的中文词同效）
    cjk_stop = {tag_key(s) for s in (stopwords or [])} | {tag_key(s) for s in _DEFAULT_CJK_STOPWORDS}
    headings = [normalize_tag(h) for h in (headings or []) if normalize_tag(h)]
    # 词典子串匹配用原始文件名（hccn_tool 等连写词不被拆分）；token 提取用拆分后的文件名
    match_text = " ".join([stem or ""] + headings)
    token_text = " ".join([re.sub(r"[_\-\.]+", " ", stem or "")] + headings)

    matched: list[TagEntry] = []
    matched_names: set[str] = set()
    matched_keys: set[str] = set()
    for e in registry.match(match_text):
        if e.name not in matched_names:
            matched.append(e)
            matched_names.add(e.name)
            matched_keys.add(tag_key(e.name))

    # 命中的设备别名串：其组成 token（"800I"/"A2"）不再各自成为候选（已归一为设备标签）
    lowered_match = match_text.lower()
    hit_aliases = [a for a in registry.devices.by_alias() if a and a in lowered_match]

    candidates: list[TagEntry] = []
    cand_seen: set[str] = set()

    def _add_candidate(name: str, tier: Optional[str] = None) -> None:
        name = normalize_tag(name)
        if not name:
            return
        key = tag_key(name)
        # 去重/排除按聚合键：词典命中词、已收候选的变体（"Atlas A3" vs "atlas_a3"）不再重复出现
        if key in matched_keys or key in cand_seen:
            return
        if key in cjk_stop:
            return
        if len(name) < min_len:
            return
        cand_seen.add(key)
        candidates.append(TagEntry(name=name, tier=tier or tier_for(name)))

    # 拉丁词 token（文件名拆分 + 标题）
    for t in _LATIN_TOKEN_RE.findall(token_text):
        t = t.strip("._-")
        if not t or len(t) < min_len:
            continue
        tl = t.lower()
        if tl in stopwords:
            continue
        if _PURE_NUM_RE.match(t):
            continue
        if _VERSION_RE.match(t):
            continue
        if any(tl in ha for ha in hit_aliases):
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
