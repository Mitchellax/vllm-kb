"""分块：KbDocument（讨论线）-> 多个 KbChunk。

策略（Phase 0 简单版）：
- 按空行分段（段落），累积到 max_chunk_chars 切块；
- 单段超长按 max_chunk_chars 硬切；
- 相邻块间用前一 block 尾部 overlap_chars 字符做重叠，保留上下文。
"""
from __future__ import annotations

import re

from .models import KbChunk, KbDocument

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def chunk_doc(
    doc: KbDocument,
    max_chunk_chars: int = 4000,
    overlap_chars: int = 200,
) -> list[KbChunk]:
    text = (doc.body or "").strip()
    if not text:
        return []

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
