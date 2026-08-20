"""社区高频信号词统计：从知识库 issue 标题提取高频专有词（TF-IDF 风格），
生成 data/code/signal_words.json 供 **agent 判断**使用（不参与代码层过滤）。

设计（对应你的要求"高频信号词置信度可能不高，按社区高频关键字形式落到 skill，让 agent 判断"）：
- 统计对象：issue 标题（比正文更聚焦，正文重复贴报错会虚高）；
- 候选词：专有名词形态（驼峰/下划线/数字混合），排除通用英文词；
- 打分：TF-IDF 风格 —— 词频高 且 出现在少数 issue（分布集中）= 高区分度；
- 输出：按 (出现次数, 区分度) 排序的信号词列表，agent 可据此判断"该词是否值得作为检索签名"。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

# 候选词形态：驼峰（aXxxB / XxxYyy）、下划线（a_b_c）、全大写（NPU、MTP）、首字母大写单词（Vector、Core）
# 注意：按空白拆分后逐个匹配，避免 "Vector Core Exception" 被黏成一个驼峰 token
_TOKEN_RE = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+"
    r"|[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+"
    r"|[A-Z]{2,}[0-9]*"
    r"|[A-Z][a-z]{2,})"
)


def _tokenize(title: str) -> list[str]:
    """按空白 + 标点拆分，返回标题里的专有 token（驼峰/下划线/全大写）。"""
    out = []
    for chunk in re.split(r"[\s,;:()\[\]{}<>/|&+=]+", title):
        m = _TOKEN_RE.fullmatch(chunk)
        if m:
            out.append(m.group(0))
    return out
# 通用英文停用词（无故障区分度）
_STOP = {
    "the", "and", "for", "with", "from", "this", "that", "when", "after", "before",
    "issue", "bug", "error", "failed", "fail", "deploy", "deployment", "service",
    "model", "models", "node", "nodes", "version", "vllm", "ascend", "using", "used",
    "use", "set", "setting", "enable", "disable", "running", "run", "start", "startup",
    "报错", "问题", "部署", "模型", "版本", "开启", "关闭", "使用", "运行", "出现", "异常",
    "test", "tests", "testing", "support", "supported", "does", "not", "work", "works",
    "one", "two", "multi", "single", "dual", "config", "configuration", "output",
    "input", "request", "response", "time", "first", "last", "current", "new", "old",
    "open", "closed", "fixed", "fix", "fixes", "fixing", "log", "logs", "info", "detail",
    "please", "help", "need", "want", "how", "why", "what", "have", "has", "had",
    "can", "could", "would", "should", "may", "might", "also", "still", "yet", "all",
    "some", "any", "each", "every", "other", "another", "about", "into", "onto", "over",
    "under", "between", "among", "through", "during", "before", "while", "because",
    "such", "only", "very", "just", "even", "more", "most", "much", "many", "few",
    "than", "then", "there", "here", "where", "which", "whose", "who", "whom",
}


def collect_signal_words(issue_dir: Path, top_n: int = 300) -> list[dict]:
    """从 issue 标题统计信号词（单词 + 双词短语）。返回 [{word, count, docs, idf, score}, ...] 降序。

    双词短语（如 Vector Core / out of memory）对"标题含专有名词"的 issue 检索价值高，
    单靠单词统计会拆散（Vector / Core 各自泛化）。
    """
    if not issue_dir.is_dir():
        return []
    doc_freq: Counter = Counter()  # 词 -> 出现过的 issue 数
    total_docs = 0
    for p in issue_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        title = d.get("title") or ""
        words = [w for w in _tokenize(title) if w.lower() not in _STOP and len(w) >= 3]
        # 双词短语：连续两个非停用词（保留原文大小写归一）
        bigrams = set()
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            if a.lower() in _STOP or b.lower() in _STOP or len(a) < 2 or len(b) < 2:
                continue
            # 双大写词组合（Vector Core / Out Of Memory 这类）
            if a[0].isupper() and b[0].isupper():
                bigrams.add(f"{a} {b}")
        tokens = set(w for w in words) | bigrams
        if not tokens:
            continue
        total_docs += 1
        for w in tokens:
            doc_freq[w] += 1

    if total_docs == 0:
        return []
    # TF-IDF 风格：score = count * log(N / df)（df 越小越集中，区分度越高）
    out = []
    for w, df in doc_freq.items():
        idf = math.log(max(total_docs, 1) / max(df, 1)) + 1.0
        out.append({"word": w, "count": df, "docs": df,
                    "idf": round(idf, 3), "score": round(df * idf, 2)})
    out.sort(key=lambda x: -x["score"])
    return out[:top_n]


def main() -> None:
    import sys
    from pathlib import Path as P

    sys.path.insert(0, str(P(__file__).resolve().parent.parent))
    from vllm_kb.config import AppConfig

    cfg = AppConfig.load()
    issue_dir = cfg.resolve("data/raw/vllm-ascend/issues")
    words = collect_signal_words(issue_dir)
    out = cfg.resolve(cfg.storage.code_root) / "signal_words.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"note": "社区高频信号词（供 agent 判断，非自动过滤）",
                    "total_issues": len(list(issue_dir.glob('*.json'))) if issue_dir.is_dir() else 0,
                    "words": words}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[signal] 统计 {len(words)} 个信号词 -> {out}")
    for w in words[:30]:
        print(f"  {w['word']:40s} count={w['count']:4d} score={w['score']}")


if __name__ == "__main__":
    main()
