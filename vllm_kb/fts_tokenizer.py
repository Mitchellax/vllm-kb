"""FTS5 中文分词（jieba 可选，未装降级原文）。

背景：SQLite FTS5 默认 unicode61 tokenizer 把连续中文整段当一个 token——
"超时排查" 无法被 "超时" 命中（中文 FTS 基本等于整段匹配）。用 jieba 在
**入库侧**对 chunk 文本分词（空格连接）写入 FTS 索引、**查询侧**对查询串分词构造
MATCH——中文词独立索引后，"超时" 可命中含 "超时排查" 的文本。

设计：
- 分词只作用于 FTS 索引列（indexed_text）与查询串；向量库（原文嵌入）与
  snippet 展示（原文）不受影响——**无需重建向量库**，仅需重建 FTS 索引
  （scripts/build_fts.py，基于现有 chunk 原文重新分词）；
- 降级：jieba 未安装时 tokenize_text 返回原文（FTS 行为与旧版一致，无需重建索引）；
- 词典词防拆分：标签词典词（config.tags.registry）注册进 jieba 自定义词典，
  复合标签（如"超时排查"）不被拆散。
"""
from __future__ import annotations

import re
from typing import Optional

_jieba = None  # None=未初始化 False=不可用 模块=可用


def _ensure_jieba():
    """懒加载 jieba（首次 import 建词典约 0.3s，仅分词路径触发）。"""
    global _jieba
    if _jieba is None:
        try:
            import jieba

            _jieba = jieba
        except ImportError:
            _jieba = False
    return _jieba or None


def register_words(words) -> None:
    """把词注册进 jieba 词典（防拆分），如标签词典词（HCCL/超时排查 不拆散）。"""
    jb = _ensure_jieba()
    if jb is None:
        return
    for w in words or []:
        w = (w or "").strip()
        if w and re.fullmatch(r"[\w\u4e00-\u9fff]+", w):
            try:
                jb.add_word(w)
            except Exception:
                pass


# CJK↔拉丁边界空格剥离（与 tagging.normalize_tag 的空格规则同源）：用户查询里
# "芯片MAC 地址" 与文档里 "芯片MAC地址" 必须等价——该空格是排版噪音，两侧分词前
# 统一剥离，jieba 切分才同构、FTS 短语才可命中。纯 CJK 间/纯拉丁间空格保留
# （"HCCL 超时排查"、"kernel NULL pointer" 的词边界不动）。
_CJK_LATIN_SPACE_RE = re.compile(
    r"(?<=[\u4e00-\u9fff]) +(?=[A-Za-z0-9])|(?<=[A-Za-z0-9]) +(?=[\u4e00-\u9fff])")


def strip_cjk_latin_spaces(text: str) -> str:
    """查询/索引归一：全角空格→半角、压缩连续空白、删 CJK↔拉丁边界空格。"""
    if not text:
        return text or ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return _CJK_LATIN_SPACE_RE.sub("", text)


def tokenize_text(text: str) -> str:
    """文本 → 空格连接的分词结果（FTS 索引写入用）；jieba 不可用时返回归一原文。

    入口先做边界空格剥离：索引侧与查询侧同规则，"芯片MAC 地址" 类排版差异
    不产生不同 token 序（改索引侧需重建 FTS——build_fts.py）。
    """
    if not text:
        return ""
    text = strip_cjk_latin_spaces(text)
    jb = _ensure_jieba()
    if jb is None:
        return text
    words = jb.lcut(text)
    return " ".join(w for w in words if w and not w.isspace())


# 降级路径的 token 提取（与 search._FTS_TOKEN_RE 同源：中英 token，剔除标点）
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")


def query_tokens(text: str) -> list[str]:
    """查询串 → token 列表（构造 FTS MATCH 用）。

    入口先做 CJK↔拉丁边界空格剥离（与索引侧 tokenize_text 同规则）——"芯片MAC 地址"
    与 "芯片MAC地址" 分词同构。
    jieba 可用：分词结果（中文词独立，"超时"可命中"超时排查"）；
    不可用：降级为中英 token 提取（原 unicode61 行为，整段中文一个 token）。
    """
    if not text:
        return []
    text = strip_cjk_latin_spaces(text)
    jb = _ensure_jieba()
    if jb is not None:
        return [t for t in (w for w in jb.lcut(text)) if t and not t.isspace()]
    return _TOKEN_RE.findall(text.lower())
