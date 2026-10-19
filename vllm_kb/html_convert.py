"""HTML → Markdown 转换（HtmlSource 解析层，混合引擎：trafilatura 正文 + lxml 表格）。

引擎分工（2026 定版，路径 C）：
- **表格先行收割**：lxml.html 遍历原始 DOM，**抽取前**把 `<table>` 从树上移除——
  * 内容表（≥2×2 或有 th/caption）→ 结构化 rows（错误码进图的输入，tables.json 供
    建图消费），正文统一由本模块以 pipe table 补回（`table_to_markdown` 单一渲染实现，
    pdf/word 同源引用）——trafilatura 的 markdown 输出实测会把表格退化成纯文本行，
    与其赌它退化成什么样，不如一棵树上只走一条渲染路径；
  * 布局表（<2×2 且无 th/caption）：整表文字 ≥80 字符视为"套壳正文"（常见单格代码框）
    → 原地展开成 `<p>`（文字保留、结构降级）；否则整体丢弃（导航/菜单碎片），丢弃数
    进 quality 元数据不静默；
- **图片同样先收割后占位**：`//img` 按文档序收集 (src, alt) 并**原位替换为占位 token**
  `[[KBIMG:i]]`（先于表格收割，表格内的截图也不丢）→ 正文抽取后**原位**还原为
  markdown 图片语法（图文位置不再脱节；trafilatura 丢掉整个占位段落时退化为文末补回），
  资产化/占位/OCR 统一走 `BaseSource._resolve_images`——**trafilatura 实测会把纯图段落
  整段丢弃**，不收割必丢图；占位是**纯文本**，正文引擎不再丢弃它；
  显式 `width/height ≤2px` 的装饰图（1x1 埋点 gif 等）收割时直接丢弃；
- **正文**：trafilatura.extract(output_format="markdown", include_images/links,
  include_formatting)——自带去噪，适配对外案例 / wiki 文章页。已知边界（实测）：
  正文不含 markdown 标题（h1-h6 抽成普通行）→ doc_html 的 chunk 无 section
  （markdown 切分器无锚点，chunking 有段落级兜底切块，检索粒度略粗）；
- **页面骨架修剪**：script/style 与 nav/aside/footer 在抽取前整棵移除（小页面会触发
  trafilatura 的召回兜底、把导航噪音留进正文，修剪先于引擎；script 文本否则会被
  baseline 当正文捞回来）；
- **图片**：src 原样保留（相对路径 / file:// / data:URI / 远程 URL），资产化/占位
  统一走 `BaseSource._resolve_images`——本模块不做任何文件写入。- **链接**：只保留 http/https/mailto 目标（与 WordSource 超链接同一安全规则），
  相对路径 / file:// 目标剥掉只留链接文字——"正文不含服务器路径"对任意输入成立。

编码（内网保存页常见 GB2312/GBK）：BOM → meta charset 声明 → utf-8 严格试解 →
gb18030 严格试解 → utf-8 errors=replace 兜底，返回实际使用的 charset 供 quality 元数据。
（utf-8 必须排在 gb18030 之前——gb18030 硬解 utf-8 中文常"成功"出乱码。）

原始 URL：审计元数据 `<meta name="x-source-url">` 优先，其次浏览器"另存为"残留的
`<!-- saved from url=(NNNN)… -->` 注释 → 文档 url 字段（phase2 设计"保留 URL"；无则空）。

审计元数据：`<meta name="x-*">`（x-created/x-updated/x-status/x-author/x-categories/
x-expiry 等）整体收割进 `ConvertResult.meta` → HtmlSource 落 `extra["meta"]`——
审计流程的元信息不再被丢弃。

降级链（每级醒目 warning，调用方落 quality 元数据，不静默）：
  trafilatura 未安装 → lxml 全文兜底（无去噪，表格已收割不受影响）；
  lxml 也未安装 → stdlib 正则剥标签（无结构、无表格）；
  trafilatura 抽取失败/过短 → trafilatura.baseline() 全文兜底。
"""
from __future__ import annotations

import codecs
import html as _html
import re
from dataclasses import dataclass, field
from typing import Optional

# <meta charset="gbk"> / <meta http-equiv="Content-Type" content="text/html; charset=gbk">
_META_CHARSET_RE = re.compile(
    r"""<meta[^>]+charset\s*=\s*(?:"[^"]*?([\w-]+)"|'[^']*?([\w-]+)'|([\w-]+))""", re.I)
# 浏览器"另存为"残留注释：<!-- saved from url=(0021)https://... -->
_SAVED_FROM_URL_RE = re.compile(r"<!--\s*saved from url=\(\d{4}\)(\S+?)\s*-->")
# markdown 行内链接（!前缀的图片形态由 lookbehind 排除；图片 dest 由 _resolve_images 处理）
_MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(\s*([^)\s]+)[^)]*\)")

# 页面骨架（非正文）：先于引擎整棵移除——小页面会触发 trafilatura 的召回兜底，
# 不修剪则导航噪音必然进正文
_CHROME_XPATH = "//nav|//aside|//footer|//noscript"

_MIN_BODY_CHARS = 40        # 低于此长度视为"抽取失败"，走 baseline 兜底
_LAYOUT_MIN_ROWS = 2        # 布局表启发式：行×列 < 2×2 且无 th/caption
_LAYOUT_MIN_COLS = 2
_UNWRAP_MIN_CHARS = 80      # 布局表文字达到此长度 → 展开 <p> 保留（单格代码框等）

# 收割占位 token：纯文本（无括号配对歧义、不被 markdown 链接正则命中），
# 抽取后原位还原为 markdown——正文引擎把纯文本带上来，位置即原文位置
_IMG_TOKEN = "[[KBIMG:{i}]]"
_TBL_TOKEN = "[[KBTBL:{i}]]"


@dataclass
class ConvertResult:
    body: str
    title: str
    tables: list[dict]           # 收割的内容表（tables.json 用；正文已按 pipe table 补回）
    charset: str
    source_url: str              # x-source-url meta 优先，其次 saved-from-url 注释
    extractor: str               # trafilatura | trafilatura_baseline | lxml_fallback | stdlib_fallback
    warnings: list[str] = field(default_factory=list)
    layout_tables_dropped: int = 0   # 丢弃的布局/碎片表数（进 quality 元数据）
    tables_unwrapped: int = 0        # 展开成段落的"套壳正文"表数
    meta: dict = field(default_factory=dict)  # <meta name="x-*"> 审计元数据（extra["meta"]）


def table_to_markdown(rows: list[list]) -> str:
    """二维表 → Markdown 表格（第一行作表头；列数取最长行并补齐）。

    全库唯一的表格渲染实现（pdf/word 经 sources 别名引用、html 直用）。
    """
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in norm[0]) + " |"]
    lines.append("|" + "---|" * width)
    for r in norm[1:]:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
    return "\n".join(lines)


def decode_html(raw: bytes) -> tuple[str, str]:
    """字节 → 文本（BOM → meta 声明 → utf-8 → gb18030 → replace）。返回 (text, charset)。

    meta 声明优先（浏览器保存页一定带）；严格试解，解码失败换下一个候选；
    最后 errors=replace 兜底：宁可出占位符也不抛异常中断整批导入。
    """
    for bom, enc in ((codecs.BOM_UTF8, "utf-8-sig"),
                     (codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
                     (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16")):
        if raw.startswith(bom):
            try:
                return raw.decode(enc), enc
            except (UnicodeDecodeError, LookupError):
                break  # BOM 命中但解码失败：落到 meta/试探链
    head = raw[:4096].decode("latin-1", errors="ignore")
    m = _META_CHARSET_RE.search(head)
    candidates: list[str] = []
    if m:
        declared = next((g for g in m.groups() if g), "")
        if declared:
            candidates.append(declared)
    candidates += ["utf-8", "gb18030"]
    for enc in candidates:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


def _saved_from_url(text: str) -> str:
    m = _SAVED_FROM_URL_RE.search(text)
    return m.group(1) if m else ""


def _classify_table(tab) -> tuple[Optional[dict], str]:
    """单个 `<table>` → (结构化 dict | None, "content" | "layout" | "empty")。

    - 嵌套表只取最外层（外层单元格 text 已含内层文字，与 WordSource 同一约定），
      行归属按 ancestor::table 判定，内层表的行不重复计入外层；
    - 单元格文本压缩空白（HTML 缩进/换行不进库）。
    """
    ths = tab.xpath(".//th")
    caps = tab.xpath("./caption")
    caption = " ".join("".join(caps[0].itertext()).split()) if caps else ""
    rows: list[list[str]] = []
    for tr in tab.xpath(".//tr"):
        if any(a is not tab for a in tr.xpath("ancestor::table")):
            continue  # 嵌套表的行，归属内层表
        cells = [" ".join("".join(cell.itertext()).split())
                 for cell in tr.xpath("./td|./th")]
        if any(cells):
            rows.append(cells)
    if not rows:
        return None, "empty"
    ncols = max(len(r) for r in rows)
    if len(rows) < _LAYOUT_MIN_ROWS or ncols < _LAYOUT_MIN_COLS:
        if not ths and not caption:
            return None, "layout"
    return {"index": 0, "caption": caption, "rows": rows}, "content"


def _harvest_tables(tree) -> tuple[list[dict], int, int]:
    """收割内容表并**原位替换为占位 token**（布局/空表照旧移除）。

    内容表占位 `<p>[[KBTBL:i]]</p>`：正文引擎把 token 文本带上 → 抽取后还原为
    pipe table（图文/表文位置保持原文位置）。返回 (tables, layout_dropped, unwrapped)。
    """
    from lxml import html as lhtml

    tables: list[dict] = []
    layout_dropped = 0
    unwrapped = 0
    for tab in tree.xpath("//table[not(ancestor::table)]"):
        t, kind = _classify_table(tab)
        parent = tab.getparent()
        if kind == "content":
            t["index"] = len(tables)
            tables.append(t)
            if parent is not None:
                p = lhtml.Element("p")
                p.text = _TBL_TOKEN.format(i=t["index"])
                parent.replace(tab, p)
            else:
                tab.drop_tree()
        elif kind == "empty":
            tab.drop_tree()
        else:  # layout
            text = " ".join(tab.text_content().split())
            if len(text) >= _UNWRAP_MIN_CHARS and parent is not None:
                p = lhtml.Element("p")
                p.text = text
                parent.replace(tab, p)
                unwrapped += 1
            else:
                tab.drop_tree()
                layout_dropped += 1
    return tables, layout_dropped, unwrapped


def _harvest_images(tree) -> list[tuple[str, str]]:
    """收割 `//img` → [(src, alt)]（文档序）并**原位替换为占位 token**。

    **trafilatura 实测会把纯图段落整段丢弃**——占位是纯文本，正文引擎会把它带上来，
    抽取后原位还原为 markdown 图片语法。先于表格收割调用（表格内的截图也要收）；
    无 src 的装饰图跳过；显式 `width/height ≤2px` 的装饰图（1x1 埋点 gif/logo）直接丢弃。
    """
    from lxml import html as lhtml

    out: list[tuple[str, str]] = []
    for img in tree.xpath("//img"):
        src = (img.get("src") or "").strip()
        parent = img.getparent()
        if not src:
            if parent is not None:
                img.drop_tree()
            continue
        try:
            w = int(str(img.get("width") or "").strip() or "0")
            h = int(str(img.get("height") or "").strip() or "0")
        except ValueError:
            w = h = 0
        if 0 < w <= 2 and 0 < h <= 2:
            if parent is not None:
                img.drop_tree()
            continue
        alt = " ".join((img.get("alt") or "").split())
        out.append((src, alt))
        if parent is not None:
            span = lhtml.Element("span")
            span.text = _IMG_TOKEN.format(i=len(out) - 1)
            parent.replace(img, span)
    return out


def _inline_tokens(body: str, images: list[tuple[str, str]],
                   tables: list[dict]) -> str:
    """占位 token → 原位还原：图片还原 markdown 语法、内容表还原 pipe table。

    表格单元格里的图片占位一并还原（后续 _resolve_images 会统一资产化/OCR）。
    **文末兜底**：正文引擎把占位段落整段丢弃时（纯图段落中招，token 不在抽取
    原文里）→ 缺席的图/表按文档序文末补回（旧策略），保证资产不丢、图表不缺席。
    判断基准是**替换前**的抽取原文——在场 token 此时已被原位还原，不能再看替换后的 body。
    """
    def _img_md(i: int) -> str:
        src, alt = images[i]
        return f"![{alt}]({src})" if alt else f"![]({src})"

    def _cell(c: str) -> str:
        for j in range(len(images)):
            c = c.replace(_IMG_TOKEN.format(i=j), _img_md(j))
        return c

    def _tbl_md(t: dict) -> str:
        rows = [[_cell(c) for c in row] for row in t["rows"]]
        return ((f"**{t['caption']}**\n\n" if t.get("caption") else "")
                + table_to_markdown(rows))

    extracted = body  # 抽取原文：占位是否被引擎整段丢弃以此为准
    for i in range(len(images)):
        tok = _IMG_TOKEN.format(i=i)
        if tok in body:
            body = body.replace(tok, _img_md(i))
    for t in tables:
        tok = _TBL_TOKEN.format(i=t["index"])
        if tok in body:
            body = body.replace(tok, "\n\n" + _tbl_md(t) + "\n\n")

    def _img_missing(i: int) -> bool:
        tok = _IMG_TOKEN.format(i=i)
        if tok in extracted:
            return False
        # 只在"存活表格"的单元格里出现的占位不算缺席（已随 pipe table 原位还原）
        return not any(tok in c
                       for t in tables
                       if _TBL_TOKEN.format(i=t["index"]) in extracted
                       for row in t["rows"] for c in row)

    missing = [_img_md(i) for i in range(len(images)) if _img_missing(i)]
    missing += [_tbl_md(t) for t in tables
                if _TBL_TOKEN.format(i=t["index"]) not in extracted]
    if missing:
        body = (body.rstrip() + "\n\n" if body.strip() else "") + "\n\n".join(missing)
    return body


def _extract_meta(tree) -> dict:
    """收割 `<meta name="x-*">` 审计元数据（x-source-url/x-created/x-updated/x-status/
    x-author/x-categories/x-expiry 等）→ ConvertResult.meta（HtmlSource 落 extra["meta"]）。
    同名重复（多个 x-categories）→ 逗号连接。"""
    out: dict[str, str] = {}
    for m in tree.xpath('//meta[starts-with(translate(@name, "X", "x"), "x-")]'):
        name = (m.get("name") or "").strip().lower()
        content = " ".join((m.get("content") or "").split())
        if not name or not content:
            continue
        out[name] = f"{out[name]}, {content}" if name in out else content
    return out


def _sanitize_links(md: str) -> str:
    """行内链接安全化：只留 http/https/mailto 目标，其余剥掉只留文字（图片不动）。"""
    def _repl(m: "re.Match[str]") -> str:
        dest = m.group(2).strip()
        if dest.startswith(("http://", "https://", "mailto:")):
            return m.group(0)
        return m.group(1)
    return _MD_LINK_RE.sub(_repl, md)


def _page_title(tree) -> str:
    for xp in ('//meta[@property="og:title"]/@content',
               '//meta[@name="title"]/@content',
               '//title/text()',
               "(//h1)[1]//text()"):
        for g in tree.xpath(xp):
            t = " ".join(str(g).split())
            if t:
                return t
    return ""


def _blocks_to_text(tree) -> str:
    """lxml 全文兜底：去 script/style 后 text_content（无结构，保文字可检索）。

    只取 `//body`——`<title>` 不算正文（整树 text_content 会把标题文本捞进来，
    纯脚本页会凭空产出一篇只有标题的"文档"）。
    """
    for bad in tree.xpath("//script|//style"):
        bad.drop_tree()
    roots = tree.xpath("//body") or [tree]
    lines = [ln.strip() for ln in roots[0].text_content().splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _strip_html_fallback(text: str) -> str:
    """stdlib 兜底（连 lxml 都没有时）：正则剥标签。只求文字可检索，不求结构。"""
    txt = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1\s*>", " ", text)
    txt = re.sub(r"(?is)<!--.*?-->", " ", txt)
    txt = re.sub(r"(?i)<br\s*/?>", "\n", txt)
    txt = re.sub(r"(?i)</(p|div|h[1-6]|li|tr|table|pre|blockquote|section|article)>", "\n\n", txt)
    txt = re.sub(r"<[^>]+>", "", txt)
    txt = _html.unescape(txt)
    lines = [ln.strip() for ln in txt.splitlines()]
    return "\n\n".join(ln for ln in lines if ln)


def _import_trafilatura():
    try:
        import trafilatura
        return trafilatura
    except ImportError:
        return None


def html_to_markdown(raw: bytes, *, fallback_title: str = "") -> ConvertResult:
    """HTML 字节 → ConvertResult（正文 markdown + 结构化表格 + 元数据）。

    纯函数：不做任何文件写入/资产落盘；图片资产化留给 BaseSource._resolve_images。
    """
    text, charset = decode_html(raw)
    warnings: list[str] = []

    tree = None
    if text.strip():
        try:
            from lxml import html as lhtml

            tree = lhtml.document_fromstring(text)
        except ImportError:
            warnings.append("lxml 未安装——stdlib 粗提取（无去噪/无表格），建议 pip install trafilatura")

    tables: list[dict] = []
    images: list[tuple[str, str]] = []
    layout_dropped = 0
    unwrapped = 0
    meta: dict = {}
    body = ""
    title = ""
    extractor = "stdlib_fallback"

    if tree is not None:
        extractor = "lxml_fallback"
        for bad in tree.xpath("//script|//style"):
            bad.drop_tree()
        for bad in tree.xpath(_CHROME_XPATH):
            bad.drop_tree()
        meta = _extract_meta(tree)
        source_url = meta.get("x-source-url") or _saved_from_url(text)
        images = _harvest_images(tree)
        tables, layout_dropped, unwrapped = _harvest_tables(tree)
        trafilatura = _import_trafilatura()
        if trafilatura is not None:
            main = None
            try:
                main = trafilatura.extract(
                    tree, output_format="markdown", include_tables=True,
                    include_images=True, include_links=True, include_formatting=True,
                    url=source_url or None)
            except Exception as e:
                warnings.append(f"trafilatura 抽取异常（{e}）")
            if main and len(main.strip()) >= _MIN_BODY_CHARS:
                body, extractor = main, "trafilatura"
            else:
                try:
                    b = trafilatura.baseline(tree)[1] or ""
                except Exception:
                    b = ""
                if b and len(b.strip()) >= _MIN_BODY_CHARS:
                    body, extractor = b, "trafilatura_baseline"
                    warnings.append("正文抽取为空/过短——baseline 全文兜底")
                elif main:
                    body, extractor = main, "trafilatura"
                    warnings.append("正文抽取过短且 baseline 不可用——保留原抽取")
                else:
                    warnings.append("trafilatura 抽取失败——lxml 全文兜底")
        else:
            warnings.append("trafilatura 未安装——lxml 全文兜底（无去噪；表格已收割不受影响）")
        if extractor == "lxml_fallback":
            body = _blocks_to_text(tree)
        title = _page_title(tree)
    else:
        body = _strip_html_fallback(text)
        source_url = _saved_from_url(text)

    body = _sanitize_links(body or "")
    body = _inline_tokens(body, images, tables)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return ConvertResult(
        body=body,
        title=title.strip() or fallback_title,
        tables=tables,
        charset=charset,
        source_url=source_url,
        extractor=extractor,
        warnings=warnings,
        layout_tables_dropped=layout_dropped,
        tables_unwrapped=unwrapped,
        meta=meta,
    )
