"""Canonical 数据模型：所有数据源（GitHub issue/PR/discussion/wiki/doc/...）归一为 KbDocument。

任何新数据源只需写一个 adapter 产出 KbDocument，入库流水线完全复用。
"""
from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field


class VersionSpan(BaseModel):
    """知识适用的版本区间（语义见 confidence.py 的 w_ver）。"""

    min: Optional[str] = None  # 知识产生时对应的版本（如 issue 创建时最近的发布版/标签声明）
    max: Optional[str] = None  # 修复/结论落地版本（如修复 PR 合并后发布的版本）


class KbDocument(BaseModel):
    """统一知识文档（一条 issue/PR/讨论/wiki 页面等）。"""

    source_type: str  # github_issue | github_pr | discussion | wiki | doc | code_chunk
    source_id: str  # 全局唯一：如 "github:vllm-project-vllm:issue:12345"
    url: str
    title: str
    body: str  # 讨论线：issue 正文 + 评论按时间序拼接（分块输入）
    created_at: Optional[str] = None  # ISO8601
    updated_at: Optional[str] = None
    resolved_at: Optional[str] = None  # closed_at / 修复合并时间；None = 未解决
    status: str = "open"  # open | closed | merged | archived
    labels: list[str] = Field(default_factory=list)
    version_span: VersionSpan = Field(default_factory=VersionSpan)
    component: str = ""  # 主组件：vllm | vllm-ascend | cann | pytorch | pytorch-ascend | npu-driver | 其他
    component_versions: dict[str, str] = Field(default_factory=dict)  # 正文提到的其他组件版本
    reliability: Optional[float] = None  # 显式可靠度；None -> 按 status/source_type 规则计算
    extra: dict[str, Any] = Field(default_factory=dict)


class KbChunk(BaseModel):
    """分块结果：一条 KbDocument 可能切成多个 chunk，每个 chunk 一条 embedding。"""

    chunk_id: str  # f"{doc.source_id}#{seq}"
    doc_id: str
    seq: int
    text: str
    section: str = ""  # 所属章节标题（PDF 手册等结构化文档；无章节则空）


def doc_to_json(doc: KbDocument) -> str:
    """序列化为单行 JSON（canonical 单文件用）。

    用 Python json 而非 pydantic 的 model_dump_json：确保 U+2028/U+2029 等字符
    一律转义为 \\uXXXX，避免 str.splitlines() 把一行 canonical 拆成多行
    （真实 vLLM issue 正文里出现过 U+2028，曾导致 canonical 行被截断）。
    """
    return json.dumps(doc.model_dump(mode="json"), ensure_ascii=True)
