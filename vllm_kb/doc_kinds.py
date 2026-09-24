"""文档种类声明表（唯一事实源）。

## 为什么单独一张表

同一个"文档种类"的知识原先散落在四处：来源类里的 `source_type` / 资产层子目录 /
`source_id` 前缀 / 接受的后缀，`review` 的"哪些算外源文档"，`chunking` 的"按什么切章节"。
新增一种文档来源要改四个文件，**漏一处就静默不一致**（典型症状：新来源的 chunk 永远
没有 section、或它在审核台「文档管理」页里看不见）。

更糟的是 `review._EXTERNAL_SOURCE_TYPES` 原先只是一份**没人引用**的字面量——
"中央声明"看着存在，实际不生效。这里改成：声明一次，各处派生，并且**真的被使用**。

## 刻意不做的事

`confidence.py` 的可靠度基线（`github_issue`/`doc_pdf` …）与 `graph.py` 的节点类型映射
**不**由本表驱动——那是**评分语义**与**图语义**，不是"文件型来源"的属性。硬塞进来会让
这张表退化成万能配置表，反而把不相关的层耦合到一起。这两处各有自己的规则，保持独立。

## 依赖

本模块**零依赖**（不 import 项目内任何模块），因此 `sources` / `chunking` / `review`
都能安全地 import 它而不产生环。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DocKind:
    """一种"文件型文档来源"的全部声明。

    - `source_type`：`KbDocument.source_type`（入库/检索/图都按它区分）；
    - `source`：config `sources[].type`，与 `sources._REGISTRY` 的 key 一致；
    - `suffixes`：**裸后缀**（`.pdf`），glob 由 `patterns` 派生——原先 word 存 `*.docx`、
      excel 存 `.xlsx`，两种写法混用，这里统一成裸后缀；
    - `section_splitter`：分块时按什么切章节（`""` = 不切，`"pdf"` = 编号标题，
      `"markdown"` = `#` 标题且同标题正文合成一段，`"lines"` = `##` 标题分组但**每行各自成块**）；
    - `external_doc`：是否算"外源文档"（审核台「文档管理」页收录）；
    - `default_verification`：该来源入库时的默认验证状态（人工可经审核台补标覆盖，
      见 `review.review_verification`）。
    """

    source_type: str
    source: str
    label: str
    asset_sub: str
    sid_prefix: str
    suffixes: tuple[str, ...]
    section_splitter: str
    external_doc: bool
    default_verification: str

    @property
    def patterns(self) -> tuple[str, ...]:
        """文件发现的 glob 形态（`(".pdf",)` → `("*.pdf",)`）。"""
        return tuple(f"*{s}" for s in self.suffixes)


KINDS: tuple[DocKind, ...] = (
    DocKind(source_type="doc_pdf", source="pdf", label="PDF", asset_sub="pdf",
            sid_prefix="pdf", suffixes=(".pdf",), section_splitter="pdf",
            external_doc=True, default_verification="expert"),
    DocKind(source_type="doc_markdown", source="markdown", label="Markdown", asset_sub="md",
            sid_prefix="md", suffixes=(".md", ".markdown"), section_splitter="markdown",
            external_doc=True, default_verification="unverified"),
    DocKind(source_type="doc_word", source="word", label="Word", asset_sub="word",
            sid_prefix="word", suffixes=(".docx", ".docm"), section_splitter="markdown",
            external_doc=True, default_verification="unverified"),
    DocKind(source_type="doc_excel", source="excel", label="Excel", asset_sub="excel",
            sid_prefix="excel", suffixes=(".xlsx", ".xlsm"), section_splitter="lines",
            external_doc=True, default_verification="unverified"),
)

# 历史/预留种类：没有任何来源类产出它，但旧库里可能存在——保留在"外源文档"集合里，
# 否则那些文档会从审核台「文档管理」页凭空消失。
LEGACY_EXTERNAL_TYPES: tuple[str, ...] = ("doc_other",)

BY_TYPE: dict[str, DocKind] = {k.source_type: k for k in KINDS}
BY_SOURCE: dict[str, DocKind] = {k.source: k for k in KINDS}

# 审核台「文档管理」页收录的 source_type（原先是一份没人引用的字面量，现在真的被使用）
EXTERNAL_DOC_TYPES: tuple[str, ...] = (
    tuple(k.source_type for k in KINDS if k.external_doc) + LEGACY_EXTERNAL_TYPES
)

# 分块策略名 → 实现函数由 chunking 注册（本模块不 import chunking，避免环）
SPLITTER_PDF = "pdf"
SPLITTER_MARKDOWN = "markdown"
SPLITTER_LINES = "lines"


def section_splitter_for(source_type: str) -> str:
    """该 source_type 用哪个章节切分策略（未声明的种类 → `""`，即不切）。"""
    kind = BY_TYPE.get(source_type)
    return kind.section_splitter if kind else ""
