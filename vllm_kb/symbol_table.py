"""源码符号表：从版本化代码仓自动提取"高区分度故障信号词"，替代手写正则。

核心思想：**不维护"报错→关键词"规则，而是从源码生成"符号表"，提取报错时查表匹配**。
新算子/新特性只要重新索引代码仓（scripts/build_code_snapshots.py）就自动覆盖。

符号分类（对应故障检索优先级）：
- kernel/op：算子名（aclnnXxx、npuXxx、dispatch_*_combine、mega_moe ...）—— 优先级最高
- feature：推理特性（enable_*、VLLM_ASCEND_*、FUSED_MC2、DSA_CP ...）
- errcode：ACL 错误码（E39999、107020、507014 ...）—— 从知识库 issue 统计
- model：模型名（GLM-5.1、DeepSeek-V4-Flash ...）—— 从知识库 issue 标题统计
- version：HDK/CANN 版本形态（CANN 9.1.0、HDK 26.1 ...）

生成与存储：
- 构建时（build_code_snapshots.py）从每个版本的源码快照提取；
- 汇总到 data/code/symbols.json（跨版本并集 + 出现版本数），供 signature 提取时加载；
- 只读：本模块不写 API 可触达的任何状态。

用法：
    from vllm_kb.symbol_table import load_symbol_table, match_symbols
    table = load_symbol_table(cfg)
    hits = match_symbols(table, "kernel_name=DispatchFFNCombine ...")  # -> [Signature...]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import AppConfig

# ---------------- 源码提取规则（结构化，非"报错内容"正则） ----------------

# 算子名：aclnnXxx / npuXxx（C++ 注册 + python 调用）
_OP_SYMBOL_RE = re.compile(r"\b(aclnn[A-Z][A-Za-z0-9_]*|npu[A-Z][A-Za-z0-9_]*)\b")
# 下划线算子：dispatch_xxx_combine / xxx_dispatch / xxx_combine_xxx / mega_moe / moe_xxx
_OP_UNDERSCORE_RE = re.compile(
    r"\b((?:dispatch|distribute|token|moe|init|sort|gather|mega|router)[a-z0-9_]*_"
    r"(?:combine|dispatch|gather|scatter|reduce|route|permute|moe)[a-z0-9_]*)\b"
)
# 通用小写下划线复合词（特性名候选）：限定已知特性前缀，避免收进普通变量名
_OP_GENERIC_RE = re.compile(
    r"\b((?:mega|dsa|mla|sfa|dcp|pcp|eplb|dp|tp|ep|cp|fc|kv|swa|gqa|mtp|flashcomm|"
    r"mooncake|layerwise|ascendstore|recompute|balance|prefill|decode)[a-z0-9_]*"
    r"_(?:moe|cp|dp|ep|tp|mc2|kv|cache|offload|sched|routing|mask)[a-z0-9_]*)\b"
)
# 特性：enable_xxx（配置项）、VLLM_ASCEND_* / ASCEND_* / HCCL_*（环境变量）
_FEATURE_ENV_RE = re.compile(r"\b(VLLM_ASCEND_[A-Z0-9_]+|ASCEND_[A-Z0-9_]+|HCCL_[A-Z0-9_]+)\b")
_FEATURE_ENABLE_RE = re.compile(r"\b(enable_[a-z0-9_]+)\b")
# 特性缩写（全大写，3-8 字符，非通用词）
_FEATURE_ABBR_RE = re.compile(r"\b(FUSED_MC2|MC2|MLAPO|PCP|DCP|EPLB|SFA|DSA|MTP|C8|NZ|MXFP|SP|PP|TP|DP)\b")

# 通用词黑名单（源码里高频但无故障区分度）
_STOPWORDS = {
    "init", "process", "compute", "params", "switch", "reset", "align", "while",
    "constexpr", "context", "copyout", "copyin", "ceildiv", "roundup",
    "opdef", "blockepilogue", "makelayout", "posttiling", "prefixtiling",
    "getplatforminfo", "getworkspacesize", "gettilingkey", "alloc", "free",
    "copy", "move", "write", "read", "update", "apply", "check", "validate",
    "print", "log", "trace", "debug", "info", "warn", "error", "exception",
}


@dataclass
class SymbolEntry:
    name: str            # 符号原文（大小写保留）
    kind: str            # kernel | feature | model | version
    weight: float = 1.0
    versions: list[str] = field(default_factory=list)  # 出现过的版本


class SymbolTable:
    def __init__(self) -> None:
        # name_lower -> SymbolEntry（跨版本合并）
        self._by_name: dict[str, SymbolEntry] = {}
        # 归一化形 -> 原始小写（变体匹配用）
        self._by_norm: dict[str, str] = {}

    def add(self, name: str, kind: str, version: str, weight: float = 1.0) -> None:
        if not isinstance(weight, (int, float)):
            raise TypeError(f"weight 必须是数值，got {type(weight).__name__}: {weight!r}")
        key = name.lower()
        e = self._by_name.get(key)
        if e is None:
            self._by_name[key] = SymbolEntry(name=name, kind=kind, weight=weight, versions=[version])
        else:
            if version not in e.versions:
                e.versions.append(version)
        # 归一化键（驼峰/下划线同算子）
        norm = _normalize(name)
        if norm and norm not in self._by_norm:
            self._by_norm[norm] = key

    @property
    def entries(self) -> list[SymbolEntry]:
        return sorted(self._by_name.values(), key=lambda e: -e.weight)

    def get(self, name: str) -> Optional[SymbolEntry]:
        return self._by_name.get(name.lower())

    def get_by_norm(self, norm: str) -> Optional[SymbolEntry]:
        key = self._by_norm.get(norm)
        return self._by_name.get(key) if key else None


def _extract_from_source(text: str, kind: str, version: str, table: SymbolTable) -> None:
    """从一段源码提取该类型的符号进表。"""
    if kind in ("kernel", "feature"):
        for m in _OP_SYMBOL_RE.finditer(text):
            table.add(m.group(1), "kernel", version, 3.0)
        for m in _OP_UNDERSCORE_RE.finditer(text):
            name = m.group(1)
            if name.lower() not in _STOPWORDS:
                table.add(name, "kernel", version, 2.5)
        for m in _OP_GENERIC_RE.finditer(text):
            name = m.group(1)
            if name.lower() not in _STOPWORDS and not name.endswith("_meta"):
                table.add(name, "kernel", version, 2.0)
        for m in _FEATURE_ENV_RE.finditer(text):
            table.add(m.group(1), "feature", version, 2.0)
        for m in _FEATURE_ENABLE_RE.finditer(text):
            name = m.group(1)
            if name.lower() not in _STOPWORDS:
                table.add(name, "feature", version, 1.8)
        for m in _FEATURE_ABBR_RE.finditer(text):
            table.add(m.group(1), "feature", version, 1.5)


def build_symbol_table_from_snapshots(
    snapshots_dir: Path,
    versions: Optional[list[str]] = None,
) -> SymbolTable:
    """扫描已解压快照，生成符号表（跨版本并集）。

    只扫故障相关目录（csrc/、vllm_ascend/），跳过 tests/docs。
    """
    table = SymbolTable()
    if not snapshots_dir.is_dir():
        return table
    version_dirs = versions or [d.name for d in snapshots_dir.iterdir() if d.is_dir()]
    for v in version_dirs:
        snap = snapshots_dir / v
        # 兼容 zip 顶层目录
        root = snap
        subdirs = [d for d in snap.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and not (snap / "csrc").exists() and not (snap / "vllm_ascend").exists():
            root = subdirs[0]
        for base in ("csrc", "vllm_ascend"):
            d = root / base
            if not d.is_dir():
                continue
            for p in d.rglob("*"):
                if not p.is_file() or p.suffix not in (".py", ".cpp", ".hpp", ".h", ".cc", ".cxx"):
                    continue
                rel = p.relative_to(root).as_posix()
                if any(f"/{x}/" in f"/{rel}" for x in ("tests", "third_party")):
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                _extract_from_source(text, "kernel", v, table)
                _extract_from_source(text, "feature", v, table)
    return table


def _extract_models_and_versions_from_issues(issue_dir: Path, table: SymbolTable) -> None:
    """从知识库 issue 标题提取模型名与 HDK/CANN 版本形态（信号词，供 agent 判断）。"""
    if not issue_dir.is_dir():
        return
    import json as _json

    model_re = re.compile(
        r"\b(GLM-?5(?:\.\d+)?|DeepSeek-?V4[-A-Za-z0-9]*|Qwen3(?:\.\d+)?[-A-Za-z0-9]*|"
        r"MiniMax-?M[0-9.]+|Kimi-?K[0-9.]+|Bailing[-\w]*)\b",
        re.I,
    )
    version_re = re.compile(r"\b(CANN\s*\d+\.\d+(?:\.\d+)?|HDK\s*\d+\.\d+(?:\.\d+)?|"
                            r"torch-?npu\s*\d+\.\d+(?:\.\d+)?|vllm[-_]ascend\s*v?\d+\.\d+(?:\.\d+)?)\b", re.I)
    for p in issue_dir.glob("*.json"):
        try:
            d = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        title = d.get("title") or ""
        for m in model_re.finditer(title):
            table.add(m.group(1), "model", "kb", 1.0)
        for m in version_re.finditer(title):
            table.add(m.group(1), "version", "kb", 0.8)


def load_symbol_table(cfg: Optional[AppConfig] = None) -> SymbolTable:
    """加载/构建符号表：优先读缓存 symbols.json，无则从快照现建。"""
    cfg = cfg or AppConfig.load()
    cache = cfg.resolve(cfg.storage.code_root) / "symbols.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            table = SymbolTable()
            for rec in data.get("symbols", []):
                e = SymbolEntry(**rec)
                table._by_name[e.name.lower()] = e
                norm = _normalize(e.name)
                if norm and norm not in table._by_norm:
                    table._by_norm[norm] = e.name.lower()
            return table
        except Exception:
            pass
    code_root = cfg.resolve(cfg.storage.code_root)
    table = build_symbol_table_from_snapshots(code_root / "snapshots")
    issue_dir = cfg.resolve("data/raw/vllm-ascend/issues")
    _extract_models_and_versions_from_issues(issue_dir, table)
    learn_error_phrases_from_issues(issue_dir, table)
    return table


def save_symbol_table(table: SymbolTable, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_from": "source snapshots + kb issues",
        "symbols": [
            {"name": e.name, "kind": e.kind, "weight": e.weight, "versions": e.versions}
            for e in table.entries
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _normalize(name: str) -> str:
    """算子名归一化：驼峰/下划线/大小写变体映射到同一规范形。

    DispatchFFNCombine / dispatch_ffn_combine / DispatchFFNCombineW4A8
    -> dispatch_ffn_combine（去掉 GetWorkspaceSize/Inner/W4A8 等后缀修饰）。
    用于符号表匹配时识别同一算子的不同写法。
    """
    s = name.strip()
    # 去掉常见修饰后缀（递归）
    for suf in ("getworkspacesize", "inner", "_meta", "w4a8", "w8a8", "bf16", "v2", "v3"):
        if s.lower().endswith(suf):
            s = s[: -len(suf)]
    # 驼峰 -> 下划线
    s1 = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    s1 = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s1)
    return s1.lower()


def match_symbols(table: SymbolTable, text: str) -> list[tuple[str, str, float]]:
    """在报错文本里查符号表：返回 [(name, kind, weight), ...]（按长度降序，长符号优先）。

    匹配策略：对报错文本中的每个候选 token（驼峰词/下划线词/大写词），
    归一化后与符号表的规范化键比对——识别同一算子的不同写法。
    """
    lowered = text.lower()
    # 直接子串匹配（快速路径）
    direct: list[tuple[str, str, float]] = []
    for key, e in table._by_name.items():
        if key in lowered:
            direct.append((e.name, e.kind, e.weight))
    # 变体归一化匹配（驼峰/下划线）：从文本提取候选 token，逐一归一化查表
    candidates = set()
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_]*(?:_[A-Za-z0-9_]+)*", text):
        tok = m.group(0)
        if 3 <= len(tok) <= 48:
            candidates.add(tok)
    variant: list[tuple[str, str, float]] = []
    for tok in candidates:
        norm = _normalize(tok)
        e = table.get_by_norm(norm) if norm else None
        if e is not None and norm != tok.lower():
            variant.append((tok, e.kind, e.weight * 0.9))
    # 合并去重（同名不同形态只留一个）
    seen: set[tuple[str, str]] = set()
    merged: list[tuple[str, str, float]] = []
    for hit in direct + variant:
        key = (hit[0].lower(), hit[1])
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    merged.sort(key=lambda t: (-len(t[0]), -t[2]))
    return merged


# ---------------- 社区高频信号词（能力3：参与标注，agent 判断） ----------------

def load_signal_words(cfg: Optional[AppConfig] = None) -> list[dict]:
    """加载社区高频信号词表（build_signal_words.py 生成）。

    返回 [{word, count, docs, idf, score}, ...]（按 score 降序）。
    信号词只用于**标注**（告诉 agent"社区里这个词高频出现"），不自动过滤。
    """
    cfg = cfg or AppConfig.load()
    path = cfg.resolve(cfg.storage.code_root) / "signal_words.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("words", [])
    except Exception:
        return []


def match_signal_words(signal_words: list[dict], text: str, top_n: int = 15,
                       min_word_len: int = 4, phrase_signals: Optional[list[dict]] = None) -> list[dict]:
    """在报错文本里命中信号词：返回命中的词（按长度降序，长词优先），供 agent 判断。

    - 双词短语（Vector Core）比单词（GLM/Out）具体得多，优先返回；
    - phrase_signals：已学习的错误短语（symbols.json 的 phrase 类，如 vector core timeout）
      ——罕见但高价值，不经过统计截断，直接作为信号源；
    - min_word_len：过滤太短的泛词（GLM/Out/CI 等，命中标题全是噪音）；
    - 只标注不筛选（agent 判断用不用）。
    """
    lowered = text.lower()
    hits = [
        w for w in signal_words
        if len(w["word"]) >= min_word_len and w["word"].lower() in lowered
    ]
    for ps in phrase_signals or []:
        if ps["word"].lower() in lowered:
            hits.append({"word": ps["word"], "count": ps.get("count", 0),
                         "docs": ps.get("docs", 0), "idf": ps.get("idf", 0),
                         "score": ps.get("score", ps.get("weight", 1.0))})
    # 长词优先（双词短语 > 长单词 > 短单词），同长度按 score
    seen: set[str] = set()
    uniq: list[dict] = []
    for w in hits:
        if w["word"].lower() in seen:
            continue
        seen.add(w["word"].lower())
        uniq.append(w)
    uniq.sort(key=lambda w: (-len(w["word"]), -w["score"]))
    return uniq[:top_n]


# ---------------- 错误短语自动学习（能力1） ----------------

# 错误上下文行：从 issue 正文提取"报错行"（含 failed/error/exception/reason 等信号）
_ERR_LINE_RE = re.compile(
    r"(?im)^\s*(?:\[?ERROR\]?|\[?Error\]?|>>>?\s*)?"
    r"(?=[^\[\]\n]*(?:failed|error|exception|timeout|trap|abort|crash|assert|out of memory)"
    r"[^\[\]\n]*)[^\n]{8,160}$"
)
# 报错行里的可读短语（去堆栈帧/路径/时间戳噪音）
_ERR_PHRASE_RE = re.compile(
    r"(?:reason|errStr|errorStr|detail|message)\s*[=:]\s*[\"']?([A-Za-z][A-Za-z0-9 _\-]{5,80})"
    r"|([A-Za-z][A-Za-z0-9 ]{5,60}?(?:failed|error|exception|timeout|trap|abort|out of memory))"
)
# 短语停用（太泛，无区分度）
_PHRASE_STOP = {
    "error occurred", "an error occurred", "error message", "error log", "error info",
    "failed to", "the error", "this error", "error code", "error is", "error while",
    "please check", "check the", "see the", "the following error", "following error",
}


def _learn_phrases_from_text(text: str, table: SymbolTable) -> None:
    """从单段文本学习错误短语（只统计不筛选，agent 判断）。"""
    from collections import Counter

    counter: Counter = Counter()
    for m in _ERR_LINE_RE.finditer(text):
        line = m.group(0).strip()
        for pm in _ERR_PHRASE_RE.finditer(line):
            ph = (pm.group(1) or pm.group(2) or "").strip().strip("'\"")
            ph_l = ph.lower()
            if len(ph) < 6 or len(ph) > 90:
                continue
            if ph_l in _PHRASE_STOP:
                continue
            # 只收"看起来像错误描述"的（含信号词，或专有名词形态）
            if not any(k in ph_l for k in ("fail", "error", "exception", "timeout", "trap",
                                           "abort", "crash", "memory", "invalid", "out of",
                                           "synchroniz", "all_gather", "allgather", "vector core",
                                           "hccl", "smmu", "mte", "aiv", "aicore")):
                if not re.match(r"^[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+$", ph):
                    continue
            counter[ph_l] += 1

    for ph, n in counter.items():
        if n >= 3:  # 至少出现在 3 个片段里才进表
            table.add(ph, "phrase", "kb", min(1.5, 0.8 + 0.1 * n))


def learn_error_phrases_from_issues(issue_dir: Path, table: SymbolTable) -> int:
    """从 issue 正文学习高频错误短语（覆盖驱动层报错：vector core timeout 等源码没有的词）。

    返回学习的短语数。只统计出现 ≥3 次的，避免单条 issue 噪音。
    """
    if not issue_dir.is_dir():
        return 0
    import json as _json

    n_before = sum(1 for e in table.entries if e.kind == "phrase")
    for p in issue_dir.glob("*.json"):
        try:
            d = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        body = d.get("body") or ""
        if body:
            _learn_phrases_from_text(body, table)
    n_after = sum(1 for e in table.entries if e.kind == "phrase")
    return n_after - n_before
