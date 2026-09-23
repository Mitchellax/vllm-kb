"""Markdown 图片引用扫描与重写（供 `MarkdownSource` 收集图片 + 保证正文不含服务器路径）。

设计目标有两条，且**第二条是硬约束**：

1. **尽量认得出**：支持常见与常见误写形态，让图片真的被收集到资产层；
2. **一条都不漏**：任何图片语法里的目标都不能以原文留在正文/canonical 里——
   否则服务器目录结构会随正文进检索库（既定安全约束：正文与 canonical 不含路径）。

支持的形态：

| 形态 | 示例 |
|---|---|
| 行内（无空格） | `![alt](imgs/shot.png)`、`![alt](shot.png "标题")` |
| 行内（含空格，非标准但常见） | `![alt](my file.png)` |
| 行内（尖括号包裹） | `![alt](<my file.png>)` |
| 行内（目标含括号） | `![alt](img(1).png)` |
| 行内（跨行目标） | `![alt](\n  shot.png\n)` |
| 引用式 | `![alt][r1]` + `[r1]: shot.png "标题"`、`![alt][]`（标签=alt） |
| 快捷引用 | `![alt]`（**仅当存在同名定义**时才视为图片，否则按 CommonMark 当字面文本） |
| HTML | `<img src="shot.png" alt="x" width="200">`（单/双引号/无引号） |
| 兜底 | 未闭合或无法解析的 `![alt](...)` → 整段占位，绝不保留原文 |

**代码感知**：``` / ~~~ 围栏块与行内 `` `code` `` 内的图片语法**不处理**
（否则会破坏代码示例并产生虚假的 unresolved 记录）。
未闭合的围栏块按 CommonMark 语义吞到文末。

**已知边界**（不处理，且不会造成路径泄漏）：
- 4 空格缩进代码块：与列表续行无法可靠区分，故不识别；其中的图片语法会被当正文处理；
- 裸快捷引用 `![alt]`（无同名定义）：按 CommonMark 是字面文本，且不含目标，无泄漏风险。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

# 围栏块：行首 ≤3 空格 + ``` 或 ~~~
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")

# 引用式定义：[label]: <dest> "title"（行首 ≤3 空格）
# 目标用非贪婪 `.+?` 而非 `\S+`：容忍 `[r1]: my file.png` 这种含空格的写法（否则定义不被
# 识别 → 路径留在正文 → 泄漏）。尾部 title 由可选组吸收。
_DEF_RE = re.compile(
    r"^ {0,3}\[([^\]]+)\]:[ \t]*(?:<([^>\n]*)>|(.+?))"
    r"(?:[ \t]+(?:\"[^\"]*\"|'[^']*'|\([^()]*\)))?[ \t]*$",
    re.MULTILINE,
)

_IMG_START_RE = re.compile(r"!\[|<img\b", re.I)
_SRC_RE = re.compile(r"""\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.I)
_ALT_RE = re.compile(r"""\balt\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.I)

# 目标尾部的 title（"..." / '...' / (...)）——剥离后才是真正的目标
_TITLE_TAIL_RE = re.compile(r"""^(?P<d>.*?)\s+(?:"[^"]*"|'[^']*'|\([^()]*\))$""", re.S)


@dataclass
class ImageRef:
    """正文中的一处图片引用（span 相对于**原始正文**）。"""
    start: int
    end: int
    alt: str = ""
    dest: str = ""
    form: str = "inline"          # inline | reference | html | unterminated
    def_dest_span: Optional[tuple[int, int]] = None  # 引用式定义里目标的 span（需一并去路径）


def normalize_label(label: str) -> str:
    """引用式标签归一（CommonMark：大小写不敏感 + 空白折叠）。"""
    return " ".join(label.split()).lower()


# ---------------- 代码区识别 ----------------

def _inline_code_spans(line: str) -> list[tuple[int, int]]:
    """行内 code span（反引号串配对，长度必须相等）。返回行内 [start,end) 列表。"""
    out: list[tuple[int, int]] = []
    i, n = 0, len(line)
    while i < n:
        if line[i] != "`":
            i += 1
            continue
        j = i
        while j < n and line[j] == "`":
            j += 1
        run = j - i
        k, close = j, -1
        while k < n:
            if line[k] != "`":
                k += 1
                continue
            m = k
            while m < n and line[m] == "`":
                m += 1
            if m - k == run:      # 长度必须相等（更长的不闭合）
                close = k
                break
            k = m
        if close < 0:
            i = j
            continue
        out.append((i, close + run))
        i = close + run
    return out


def code_spans(text: str) -> list[tuple[int, int]]:
    """Markdown 代码区（围栏块 + 行内 code span）的 [start,end) 区间，按位置排序。"""
    spans: list[tuple[int, int]] = []
    lines = text.split("\n")
    offs: list[int] = []
    pos = 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1

    i, n = 0, len(lines)
    prose_lines: list[int] = []
    while i < n:
        m = _FENCE_OPEN_RE.match(lines[i])
        if not m:
            prose_lines.append(i)
            i += 1
            continue
        fence_char, fence_len = m.group(1)[0], len(m.group(1))
        start = offs[i]
        j, end = i + 1, None
        while j < n:
            cm = _FENCE_CLOSE_RE.match(lines[j])
            if cm and cm.group(1)[0] == fence_char and len(cm.group(1)) >= fence_len:
                end = offs[j] + len(lines[j])
                break
            j += 1
        spans.append((start, len(text) if end is None else end))
        i = j + 1

    for idx in prose_lines:
        base, line = offs[idx], lines[idx]
        for s, e in _inline_code_spans(line):
            spans.append((base + s, base + e))
    spans.sort()
    return spans


def _prose_segments(text: str) -> list[tuple[int, str]]:
    """代码区的补集：[(绝对起点, 片段)]。"""
    out: list[tuple[int, str]] = []
    pos = 0
    for s, e in code_spans(text):
        if s > pos:
            out.append((pos, text[pos:s]))
        pos = max(pos, e)
    if pos < len(text):
        out.append((pos, text[pos:]))
    return out


# ---------------- 目标解析 ----------------

def _strip_title(dest: str) -> str:
    m = _TITLE_TAIL_RE.match(dest)
    return (m.group("d") if m else dest).strip()


def _unwrap_angle(dest: str) -> str:
    """剥掉目标两侧的尖括号（`<my file.png>` → `my file.png`）。"""
    d = dest.strip()
    if d.startswith("<") and d.endswith(">") and len(d) >= 2:
        return d[1:-1].strip()
    return d


def _scan_balanced_dest(s: str, i: int) -> Optional[tuple[str, int]]:
    """从 `(` 起平衡扫描到配对的 `)`。返回 (原始目标, `)` 之后的下标)；未闭合返回 None。

    平衡计数让 `img(1).png` 正确收尾；不因空格提前中止（容忍 `my file.png` 这种非标准写法）；
    遇空行（段落边界）即放弃，避免跨段吞掉大段正文。
    """
    n, j, depth = len(s), i + 1, 0
    while j < n:
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "\n":
            if j + 1 < n and s[j + 1] == "\n":
                return None
            j += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                return s[i + 1:j], j + 1
            depth -= 1
        j += 1
    return None


def _collect_definitions(text: str) -> dict[str, tuple[str, int, int]]:
    """引用式定义：label → (dest, dest_start, dest_end)。定义行本身在代码区内的忽略。"""
    blocked = code_spans(text)
    out: dict[str, tuple[str, int, int]] = {}
    for m in _DEF_RE.finditer(text):
        if any(s <= m.start() < e for s, e in blocked):
            continue
        if m.group(2) is not None:
            dest, ds, de = m.group(2), m.start(2), m.end(2)
        else:
            dest, ds, de = m.group(3) or "", m.start(3), m.end(3)
        out[normalize_label(m.group(1))] = (dest, ds, de)
    return out


# ---------------- 扫描 ----------------

def find_image_refs(text: str) -> list[ImageRef]:
    """扫描正文中的图片引用（跳过代码区），按位置排序、互不重叠。"""
    defs = _collect_definitions(text)
    refs: list[ImageRef] = []
    for base, chunk in _prose_segments(text):
        refs.extend(_scan_chunk(chunk, base, defs))
    refs.sort(key=lambda r: r.start)
    return refs


def _scan_chunk(chunk: str, base: int, defs: dict) -> list[ImageRef]:
    refs: list[ImageRef] = []
    for m in _IMG_START_RE.finditer(chunk):
        start = m.start()
        if chunk[start] != "!":
            refs.extend(_scan_html(chunk, base, start))
            continue
        k = chunk.find("]", start + 2)
        if k < 0:
            continue
        alt = chunk[start + 2:k]
        nxt = k + 1
        if nxt < len(chunk) and chunk[nxt] == "(":
            parsed = _scan_balanced_dest(chunk, nxt)
            if parsed is None:
                # 未闭合：只吞到**本行行尾**（不吞掉后续引用），仍占位——绝不把原文留在正文。
                # 遇下一个 `![` 提前收尾，避免把同一行后面的正常引用一起吃掉。
                eol = chunk.find("\n", nxt)
                if eol < 0:
                    eol = len(chunk)
                nxt_img = chunk.find("![", nxt)
                if 0 <= nxt_img < eol:
                    eol = nxt_img
                raw = _unwrap_angle(chunk[nxt + 1:eol])
                refs.append(ImageRef(base + start, base + eol, alt,
                                     _strip_title(raw), "unterminated"))
                continue
            raw, end = parsed
            refs.append(ImageRef(base + start, base + end, alt,
                                 _strip_title(_unwrap_angle(raw)), "inline"))
        elif nxt < len(chunk) and chunk[nxt] == "[":
            k2 = chunk.find("]", nxt + 1)
            if k2 < 0:
                continue
            label = chunk[nxt + 1:k2].strip() or alt.strip()
            d = defs.get(normalize_label(label))
            refs.append(ImageRef(
                base + start, base + k2 + 1, alt, d[0] if d else "", "reference",
                def_dest_span=(base + d[1], base + d[2]) if d else None))
        else:
            # 快捷引用：有同名定义才算图片（否则 CommonMark 视作字面文本）
            d = defs.get(normalize_label(alt.strip()))
            if d:
                refs.append(ImageRef(base + start, base + k, alt, d[0], "reference",
                                     def_dest_span=(base + d[1], base + d[2])))
    return refs


def _scan_html(chunk: str, base: int, start: int) -> list[ImageRef]:
    gt = chunk.find(">", start)
    if gt < 0:
        return []
    tag = chunk[start:gt + 1]
    src = _SRC_RE.search(tag)
    alt_m = _ALT_RE.search(tag)
    dest = ""
    if src:
        dest = src.group(1) or src.group(2) or src.group(3) or ""
    alt = ""
    if alt_m:
        alt = alt_m.group(1) or alt_m.group(2) or alt_m.group(3) or ""
    return [ImageRef(base + start, base + gt + 1, alt, dest, "html")]


# ---------------- 重写 ----------------

def rewrite_images(text: str, resolve: Callable[[ImageRef], str]) -> tuple[str, list[ImageRef]]:
    """把每处图片引用替换为 `resolve(ref)` 的返回值，返回 (新正文, 引用列表)。

    `resolve` 负责资产收集/OCR 并返回替换文本（通常是占位符，高置信 OCR 时追加文本）。
    引用式定义里的目标同时被替换为 `[图片]`（去路径）——同一标签只替换一次。
    """
    refs = find_image_refs(text)
    if not refs:
        return text, []
    edits: list[tuple[int, int, str]] = []
    seen_defs: set[tuple[int, int]] = set()
    for r in refs:
        edits.append((r.start, r.end, resolve(r)))
        if r.def_dest_span and r.def_dest_span not in seen_defs:
            seen_defs.add(r.def_dest_span)
            edits.append((r.def_dest_span[0], r.def_dest_span[1], "[图片]"))
    edits.sort(key=lambda e: e[0], reverse=True)
    out = text
    for s, e, rep in edits:
        out = out[:s] + rep + out[e:]
    return out, refs
