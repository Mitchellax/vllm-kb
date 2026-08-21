"""分块：KbDocument（讨论线）-> 多个 KbChunk。

策略（Phase 0 简单版）：
- 按空行分段（段落），累积到 max_chunk_chars 切块；
- 单段超长按 max_chunk_chars 硬切；
- 相邻块间用前一 block 尾部 overlap_chars 字符做重叠，保留上下文；
- **PDF 手册（doc_pdf）带章节结构**：识别章节标题（如 "2.34 获取network版本号信息"），
  每个 chunk 记录所属 section 并在文本前缀注入标题行——命中正文可同时看到所属章节，
  标题文本也参与向量/FTS 匹配。
"""
from __future__ import annotations

import re

from .models import KbChunk, KbDocument

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")

# PDF 手册章节标题：编号 + 空格 + 标题（如 "2.34 获取network版本号信息"、"1 用户指南"）。
# 排除目录页的"....."点线填充行与超长行（正文标题行短）。
_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(\S.{0,60}?)\s*$")


def _is_section_line(line: str) -> bool:
    """判断一行是否为章节标题（PDF 手册编号模式）。"""
    if len(line) > 80 or "...." in line or "…" in line:
        return False
    return bool(_SECTION_RE.match(line))


def _split_sections(text: str) -> list[tuple[str, str]]:
    """按章节标题把正文切成 (section, 段落文本) 段；无标题前置内容归入 ''。

    行级扫描：PDF 文字层标题行可能与正文同段（无空行分隔），
    也常与目录连续排列——只认"编号 + 短标题"的行，其余行累积到当前章节。
    """
    sections: list[tuple[str, str]] = []
    cur_title = ""
    cur_lines: list[str] = []
    # 目录区启发：目录里标题后常跟 '.....页码'，正文标题后跟正文——
    # 对每个候选标题行，若下一行仍是标题行，视为目录连续标题（跳过，不切新章节）
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _is_section_line(stripped):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt and _is_section_line(nxt):
                # 目录区：连续标题行，跳过（不污染正文章节）
                continue
            if cur_lines:
                sections.append((cur_title, "\n".join(cur_lines)))
            cur_title = stripped
            cur_lines = []
        else:
            cur_lines.append(stripped)
    if cur_lines:
        sections.append((cur_title, "\n".join(cur_lines)))
    return sections


def chunk_doc(
    doc: KbDocument,
    max_chunk_chars: int = 4000,
    overlap_chars: int = 200,
) -> list[KbChunk]:
    text = (doc.body or "").strip()
    if not text:
        return []

    # PDF 手册：带章节结构切分；其他来源保持原逻辑（section 为空）
    if doc.source_type == "doc_pdf":
        return _chunk_sections(doc, text, max_chunk_chars, overlap_chars)

    if len(text) <= max_chunk_chars:
        return [KbChunk(chunk_id=f"{doc.source_id}#0", doc_id=doc.source_id, seq=0, text=text)]

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    raw_chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for para in paragraphs:
        # 单段超长：先硬切
        while len(para) > max_chunk_chars:
            if cur:
                raw_chunks.append("\n".join(cur))
                cur, cur_len = [], 0
            raw_chunks.append(para[:max_chunk_chars])
            para = para[max_chunk_chars:]
        if not para:
            continue
        if cur_len + len(para) + 2 > max_chunk_chars and cur:
            raw_chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(para)
        cur_len += len(para) + 2
    if cur:
        raw_chunks.append("\n".join(cur))

    chunks: list[KbChunk] = []
    prev_tail = ""
    for i, c in enumerate(raw_chunks):
        text_c = c
        if overlap_chars > 0 and prev_tail and i > 0:
            text_c = prev_tail + "\n" + c
        chunks.append(KbChunk(chunk_id=f"{doc.source_id}#{i}", doc_id=doc.source_id, seq=i, text=text_c))
        prev_tail = c[-overlap_chars:] if overlap_chars > 0 else ""
    return chunks


def _chunk_sections(
    doc: KbDocument,
    text: str,
    max_chunk_chars: int,
    overlap_chars: int,
) -> list[KbChunk]:
    """带章节的 PDF 手册分块：每段正文按章节分组，chunk 注入所属标题。"""
    chunks: list[KbChunk] = []
    seq = 0
    for section, section_text in _split_sections(text):
        if not section_text:
            continue
        # 章节内分块：前缀注入标题（标题参与匹配，命中即知所属章节）
        body = section_text
        prefix = f"【{section}】\n" if section else ""
        para_chunks = _chunk_plain(body, max_chunk_chars, overlap_chars)
        for c in para_chunks:
            text_c = (prefix + c) if section else c
            chunks.append(KbChunk(
                chunk_id=f"{doc.source_id}#{seq}",
                doc_id=doc.source_id,
                seq=seq,
                text=text_c,
                section=section,
            ))
            seq += 1
    if not chunks and text:
        # 退化：没有识别到章节，按普通文本分块
        for i, c in enumerate(_chunk_plain(text, max_chunk_chars, overlap_chars)):
            chunks.append(KbChunk(
                chunk_id=f"{doc.source_id}#{i}", doc_id=doc.source_id, seq=i, text=c,
            ))
    return chunks


def _chunk_plain(text: str, max_chunk_chars: int, overlap_chars: int) -> list[str]:
    """普通文本分块（段落累积 + 重叠），返回原始文本块列表。"""
    if len(text) <= max_chunk_chars:
        return [text]
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    raw_chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for para in paragraphs:
        while len(para) > max_chunk_chars:
            if cur:
                raw_chunks.append("\n".join(cur))
                cur, cur_len = [], 0
            raw_chunks.append(para[:max_chunk_chars])
            para = para[max_chunk_chars:]
        if not para:
            continue
        if cur_len + len(para) + 2 > max_chunk_chars and cur:
            raw_chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(para)
        cur_len += len(para) + 2
    if cur:
        raw_chunks.append("\n".join(cur))
    out: list[str] = []
    prev_tail = ""
    for i, c in enumerate(raw_chunks):
        text_c = c
        if overlap_chars > 0 and prev_tail and i > 0:
            text_c = prev_tail + "\n" + c
        out.append(text_c)
        prev_tail = c[-overlap_chars:] if overlap_chars > 0 else ""
    return out
