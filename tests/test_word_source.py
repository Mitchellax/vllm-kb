"""Word（.docx）来源测试：标题层级→section、列表、表格→Markdown+JSON、身份与消歧、
缓存、脱敏、分块复用。

测试用 python-docx **现场生成** docx（不塞二进制夹具）：夹具二进制无法 review，
且生成方式本身就是"我们支持哪些写法"的可执行文档。
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vllm_kb.config import AppConfig, SourceCfg
from vllm_kb.sources import WordSource


def make_docx(path: Path, *, headings=True, table=True, lists=True,
              note="现场：训练任务在第 300 步卡住。") -> None:
    """生成含标题/正文/列表/表格/空段落的测试 docx（覆盖中英样式名）。

    `headings=False` 时仍写一段正文——空文档会被来源跳过，用它构造"改过内容"的版本。
    """
    import docx

    d = docx.Document()
    if headings:
        d.add_heading("HCCL 超时排查案例", level=1)
        d.add_paragraph(note)
        d.add_heading("现象", level=2)
        d.add_paragraph("节点 10.0.0.5 报错，日志在 /home/user/logs/hccl.log。")
    else:
        d.add_paragraph(note)
    if lists:
        d.add_heading("排查步骤", level=2)
        d.add_paragraph("检查网卡状态", style="List Bullet")
        d.add_paragraph("确认 rank 表一致", style="List Number")
    if table:
        d.add_heading("错误码表", level=2)
        t = d.add_table(rows=3, cols=3)
        data = [["错误码", "含义", "处理"],
                ["561000", "aclnn 执行失败", "检查算子支持"],
                ["107020", "dispatch_ffn_combine 失败", "升级驱动"]]
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                t.cell(r, c).text = val
    d.add_paragraph("")          # 空段落应跳过
    path.parent.mkdir(parents=True, exist_ok=True)
    d.save(str(path))


class _WordCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.import_dir = self.root / "data" / "imports" / "word"
        self.import_dir.mkdir(parents=True)
        self.cfg = SourceCfg(id="cases", type="word", path="data/imports/word", enabled=True)

    def tearDown(self):
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def _src(self, **kw) -> WordSource:
        return WordSource(self.cfg, project_root=self.root, **kw)

    def _write(self, name: str, **kw) -> Path:
        p = self.import_dir / name
        make_docx(p, **kw)
        return p

    def _docs(self, **kw):
        """构建：导入目录还没有 docx 时自动放一篇默认案例（多数用例只关心解析结果）。"""
        if self.import_dir.exists() and not any(self.import_dir.rglob("*.docx")):
            self._write("案例.docx", **kw)
        src = self._src()
        src.pull()
        return src.canonicalize()


class TestWordSource(_WordCase):
    def test_pull_copies_to_assets(self):
        self._write("案例.docx")
        src = self._src()
        self.assertEqual(src.pull(), 1)
        self.assertTrue((self.root / "data" / "assets" / "word" / "案例.docx").exists())
        self.assertEqual(src.pull(), 0)          # 幂等

    def test_headings_become_markdown_levels(self):
        """标题样式 → `#` 层级（这是分块/标签共用的结构来源）。"""
        docs = self._docs()
        self.assertEqual(len(docs), 1)
        body = docs[0].body
        self.assertIn("# HCCL 超时排查案例", body)
        self.assertIn("## 现象", body)
        self.assertIn("## 排查步骤", body)
        # 一级标题作 title（不是文件名）
        self.assertEqual(docs[0].title, "HCCL 超时排查案例")

    def test_lists_get_prefix(self):
        """列表项加 `- `/`1. ` 前缀——否则条目会粘成一段，FTS 命中粒度差。"""
        body = self._docs()[0].body
        self.assertIn("- 检查网卡状态", body)
        self.assertIn("1. 确认 rank 表一致", body)

    def test_table_into_body_and_json(self):
        """表格既拼进正文（可 FTS 检索）又落结构化 JSON（供图/查询）。"""
        docs = self._docs()
        body = docs[0].body
        self.assertIn("| 错误码 | 含义 | 处理 |", body)
        self.assertIn("| 561000 | aclnn 执行失败 | 检查算子支持 |", body)
        self.assertEqual(docs[0].extra["asset"]["tables"], 1)
        rel = docs[0].extra["structure"]["tables"]
        self.assertEqual(len(rel), 1)
        # 落盘文件名以 asset_id 命名（不暴露文件名/路径）
        aid = docs[0].extra["asset"]["asset_id"]
        self.assertEqual(rel[0], f"parsed/word/{aid}.tables.json")
        tpath = self.root / "data" / rel[0]
        self.assertTrue(tpath.exists())
        data = json.loads(tpath.read_text(encoding="utf-8"))
        self.assertEqual(data["tables"][0]["rows"][1][0], "561000")
        self.assertEqual(data["source"], f"word:{aid}")

    def test_identity_and_metadata(self):
        docs = self._docs()
        d = docs[0]
        self.assertEqual(d.source_id, "word:案例")
        self.assertEqual(d.source_type, "doc_word")
        self.assertEqual(d.extra["asset"]["format"], "word")
        self.assertEqual(d.extra["quality"]["parsed_with"], "python-docx")
        self.assertEqual(d.extra["quality"]["text_source"], "text_layer")
        # 与 markdown/excel 统一路径：先入库，审核工作台补标
        self.assertEqual(d.extra["verification"], "unverified")
        self.assertGreater(d.extra["asset"]["paragraphs"], 0)

    def test_entities_reuse_existing_pipeline(self):
        """表格里的错误码复用现有提取线路（canonical → MENTIONS）。"""
        from vllm_kb.graph_rels import extract_doc_relations

        d = self._docs()[0]
        ex = extract_doc_relations(d.source_id, "", 0, d.source_type, d.body)
        self.assertIn("561000", ex.mentions.get("error_code", set()))

    def test_empty_document_skipped(self):
        """无正文（只含空段落）→ 跳过，不产出空文档。"""
        import docx

        d = docx.Document()
        d.add_paragraph("")
        (self.import_dir / "空.docx").parent.mkdir(parents=True, exist_ok=True)
        d.save(str(self.import_dir / "空.docx"))
        self.assertEqual(self._docs(), [])

    def test_corrupt_file_skipped_without_crash(self):
        """非 OOXML（伪装的 .docx）→ 打印原因并跳过，不影响其它文件。"""
        self._write("好的.docx")
        (self.import_dir / "坏的.docx").write_bytes(b"\xd0\xcf\x11\xe0 not a zip")
        docs = self._docs()
        self.assertEqual([d.source_id for d in docs], ["word:好的"])

    def test_import_dir_wins_over_assets(self):
        """优先读导入目录：改源文件后重跑，仍是同一篇（不会因资产层副本变成两篇）。"""
        p = self._write("案例.docx")
        self._docs()
        # 改内容 → 资产层会另存 案例.<sha12>.docx（不可变层），但 canonicalize 读导入目录
        make_docx(p, headings=False, table=False, lists=False)
        docs = self._docs()
        self.assertEqual([d.source_id for d in docs], ["word:案例"])
        self.assertNotIn("## 现象", docs[0].body)

    def test_fallback_to_assets_collapses_versions(self):
        """导入目录缺失 → 回退资产层，并按版本族收敛（只取最新一份，id 取族名）。"""
        self._write("案例.docx")
        self._docs()
        shutil.rmtree(self.import_dir)
        assets = self.root / "data" / "assets" / "word"
        # 伪造一个"更新版本"（_copy_asset 的同名异内容命名）
        newer = assets / "案例.abcdef123456.docx"
        make_docx(newer, headings=True, table=False, lists=False)
        os.utime(assets / "案例.docx", (1000, 1000))
        os.utime(newer, (2000, 2000))
        docs = self._docs()
        self.assertEqual([d.source_id for d in docs], ["word:案例"])   # 收敛为一篇
        self.assertIn("# HCCL 超时排查案例", docs[0].body)             # 取最新版本

    def test_same_stem_different_dir_disambiguated(self):
        """同名不同目录 → 相对路径指纹消歧（导入目录保留目录树的意义所在）。"""
        make_docx(self.import_dir / "a" / "案例.docx", headings=False, table=False, lists=False)
        make_docx(self.import_dir / "b" / "案例.docx", headings=False, table=False, lists=False)
        ids = sorted(d.source_id for d in self._docs())
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(i.startswith("word:案例--") for i in ids), ids)
        self.assertNotEqual(ids[0], ids[1])

    def test_parse_cache_hit(self):
        """解析中间产物按 asset_id 缓存；内容不变时复用（并写 extract.json）。"""
        self._write("案例.docx")
        self._docs()
        aid = None
        for f in (self.root / "data" / "parsed" / "word").glob("*.extract.json"):
            aid = f.stem.split(".")[0]
            data = json.loads(f.read_text(encoding="utf-8"))
            self.assertEqual(data["asset_id"], aid)
        self.assertIsNotNone(aid, "未写 extract.json 缓存")
        # 二次解析：结果一致（缓存命中路径与首次共用 _doc_from_extract）
        first = self._docs()
        second = self._docs()
        self.assertEqual(first[0].body, second[0].body)
        self.assertEqual(first[0].source_id, second[0].source_id)

    def test_raw_text_stored_and_sanitize_log_collected(self):
        """后置脱敏：body 原文入库（原文检索）；会被脱敏的 IP/路径落维护日志。"""
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data")
        cfg = AppConfig.model_validate({})
        self._write("案例.docx")
        src = self._src(app_cfg=cfg)
        src.pull()
        docs = src.canonicalize()
        body = docs[0].body
        self.assertIn("10.0.0.5", body)                       # 原文入库
        self.assertIn("/home/user/logs/hccl.log", body)
        log = self.root / "data" / "sanitize_log.json"
        self.assertTrue(log.exists())
        data = json.loads(log.read_text(encoding="utf-8"))
        self.assertIn("10.0.0.5", data["ips"])
        self.assertIn("/home/user/logs/hccl.log", data["paths"])

    def test_asset_registry_registration(self):
        """资产注册到 asset_registry（审核台据此反查路径/下载原件）。"""
        from vllm_kb.review import list_assets

        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data")
        cfg = AppConfig.model_validate({})
        self._write("案例.docx")
        src = self._src(app_cfg=cfg)
        src.pull()
        assets = list_assets(self.root / "data" / "review.sqlite3")
        self.assertEqual(len(assets), 1)
        reg = next(iter(assets.values()))
        self.assertEqual(reg["rel_path"], "assets/word/案例.docx")
        self.assertEqual(reg["source_type"], "doc_word")
        self.assertGreater(reg["size"], 0)

    def test_word_missing_dependency_degrades(self):
        """未装 python-docx → 提示安装并返回空（与 openpyxl/pymupdf 同样的降级风格）。"""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "docx":
                raise ImportError("no module named docx")
            return real_import(name, *a, **kw)

        self._write("案例.docx")
        with mock.patch.object(builtins, "__import__", side_effect=fake_import):
            self.assertEqual(self._src().canonicalize(), [])


class TestWordChunking(_WordCase):
    def test_chunks_carry_section(self):
        """doc_word 复用 markdown 章节切分：chunk 带 section（命中能看到所属章节）。"""
        from vllm_kb.chunking import chunk_doc

        d = self._docs()[0]
        chunks = chunk_doc(d)
        sections = [c.section for c in chunks]
        self.assertIn("现象", sections)
        self.assertIn("排查步骤", sections)
        # section 前缀注入正文（检索命中直接可读）
        self.assertTrue(any("【现象】" in c.text for c in chunks))


class TestWordHelpers(unittest.TestCase):
    """样式识别与文档顺序遍历的单测（不依赖真实 Word 文件）。"""

    def test_heading_level_english_and_chinese(self):
        import docx

        from vllm_kb.sources import _word_heading_level

        d = docx.Document()
        h1 = d.add_heading("一", level=1)
        h3 = d.add_heading("三", level=3)
        plain = d.add_paragraph("正文")
        self.assertEqual(_word_heading_level(h1), 1)
        self.assertEqual(_word_heading_level(h3), 3)
        self.assertEqual(_word_heading_level(plain), 0)
        # 中文界面样式名（"标题 1"）+ 无 style_id 的兜底路径
        with mock.patch.object(type(h1.style), "name", "标题 2"):
            self.assertEqual(_word_heading_level(h1), 2)

    def test_title_style_is_level_1(self):
        import docx

        from vllm_kb.sources import _word_heading_level

        d = docx.Document()
        p = d.add_paragraph("文档标题", style="Title")
        self.assertEqual(_word_heading_level(p), 1)

    def test_list_prefix(self):
        import docx

        from vllm_kb.sources import _word_list_prefix

        d = docx.Document()
        self.assertEqual(_word_list_prefix(d.add_paragraph("正文")), "")
        self.assertEqual(_word_list_prefix(d.add_paragraph("项", style="List Bullet")), "- ")
        self.assertEqual(_word_list_prefix(d.add_paragraph("项", style="List Number")), "1. ")

    def test_iter_docx_blocks_preserves_order(self):
        """段落与表格必须按文档顺序产出（否则表格会全被挪到正文末尾、section 归属错）。"""
        import docx

        from vllm_kb.sources import _iter_docx_blocks

        d = docx.Document()
        d.add_paragraph("前")
        t = d.add_table(rows=1, cols=1)
        t.cell(0, 0).text = "表"
        d.add_paragraph("后")
        kinds = ["table" if hasattr(b, "rows") else b.text for b in _iter_docx_blocks(d)]
        self.assertEqual(kinds, ["前", "table", "后"])

    def test_discover_prefers_import_dir(self):
        """发现逻辑：导入目录有文件就不看资产层（且返回 from_imports=True）。"""
        from vllm_kb.config import SourceCfg
        from vllm_kb.sources import _discover_source_files, WordSource

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = WordSource(SourceCfg(id="c", type="word", path="data/imports/word"),
                             project_root=root)
            (root / "data" / "imports" / "word").mkdir(parents=True)
            (root / "data" / "assets" / "word").mkdir(parents=True)
            (root / "data" / "imports" / "word" / "a.docx").write_bytes(b"x")
            (root / "data" / "assets" / "word" / "b.docx").write_bytes(b"y")
            files, fallback = _discover_source_files(
                src, ("*.docx",), src.import_dir, src._assets_dir())
            self.assertFalse(fallback)
            self.assertEqual([p.name for p, _, _ in files], ["a.docx"])
            self.assertTrue(all(fi for _, fi, _ in files))
            # 导入目录空 → 回退
            (root / "data" / "imports" / "word" / "a.docx").unlink()
            files, fallback = _discover_source_files(
                src, ("*.docx",), src.import_dir, src._assets_dir())
            self.assertTrue(fallback)
            self.assertEqual([p.name for p, _, _ in files], ["b.docx"])
            self.assertFalse(any(fi for _, fi, _ in files))
            # 导入目录不存在、资产层也不存在 → 空
            shutil.rmtree(root / "data" / "assets")
            self.assertEqual(
                _discover_source_files(src, ("*.docx",), src.import_dir, src._assets_dir()),
                ([], False))


if __name__ == "__main__":
    unittest.main()
