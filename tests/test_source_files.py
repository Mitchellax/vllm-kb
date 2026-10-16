"""文件发现接缝（source_files）契约测试。

这一层是"文件型来源"的共用骨架：**导入路径优先 + 版本族收敛 + 族名即身份 + 同名消歧**。
四种来源（pdf/markdown/word/excel）都走它，所以它的不变量必须被钉住：
1. 导入路径有匹配文件时，资产层**不参与文档集**（只参与诊断）；
2. 回退到资产层时按版本族收敛，且 **stem 取族名**（否则 source_id 会漂移）；
3. 同族名多份 → `duplicate` 标记 + `sid()` 加路径指纹；
4. 资产层有、导入路径没有的族 → **必须告警**（否则"文件不再被刷新"是静默偏离）；
5. 导入路径可以是**文件**（excel 的 `path` 就是单文件）。
"""
import tempfile
import unittest
from pathlib import Path

from vllm_kb.source_files import (DiscoverSpec, SourceFile, discover,
                                  latest_versions, path_tag, rel_dirs_by_sid)


def spec(tmp: Path, *, patterns=("*.md",), import_path=None, sid_prefix="md",
         label="Markdown") -> DiscoverSpec:
    return DiscoverSpec(
        patterns=patterns,
        import_path=import_path or (tmp / "data" / "imports" / "md"),
        assets_dir=tmp / "data" / "assets" / "md",
        sid_prefix=sid_prefix,
        label=label,
        log_prefix="[t]",
    )


def write(p: Path, text: str = "x", mtime: float | None = None) -> Path:
    import os

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


class TestDiscover(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # ---------- 导入路径优先 ----------

    def test_imports_win_over_assets(self):
        write(self.root / "data" / "imports" / "md" / "a.md")
        write(self.root / "data" / "assets" / "md" / "b.md")
        d = discover(spec(self.root))
        self.assertFalse(d.fallback)
        self.assertEqual([f.path.name for f in d.files], ["a.md"])
        self.assertTrue(all(f.from_imports for f in d.files))
        # 资产层多出来的族必须出现在诊断里（"不再纳管"要可见）
        self.assertEqual(d.stray, ("b.md",))

    def test_no_stray_when_families_match(self):
        write(self.root / "data" / "imports" / "md" / "a.md")
        write(self.root / "data" / "assets" / "md" / "a.md")
        self.assertEqual(discover(spec(self.root)).stray, ())

    def test_stray_ignores_version_copies_of_covered_family(self):
        """`a.<sha12>.md` 是 a 族的历史版本 → 导入路径有 a.md 就不算 stray。"""
        write(self.root / "data" / "imports" / "md" / "a.md")
        write(self.root / "data" / "assets" / "md" / "a.md")
        write(self.root / "data" / "assets" / "md" / "a.0123456789ab.md")
        self.assertEqual(discover(spec(self.root)).stray, ())

    # ---------- 回退与版本族收敛 ----------

    def test_fallback_converges_to_latest_and_keeps_family_name(self):
        write(self.root / "data" / "assets" / "md" / "a.md", "旧", mtime=100)
        write(self.root / "data" / "assets" / "md" / "a.0123456789ab.md", "新", mtime=200)
        d = discover(spec(self.root))
        self.assertTrue(d.fallback)
        self.assertEqual(len(d.files), 1)
        f = d.files[0]
        self.assertEqual(f.stem, "a")                       # 族名，不是 a.0123456789ab
        self.assertEqual(f.path.name, "a.0123456789ab.md")  # 但内容取最新那份
        self.assertFalse(f.from_imports)
        self.assertEqual(f.sid("md", self.root / "x", self.root / "y"), "md:a")

    def test_fallback_when_import_path_missing(self):
        write(self.root / "data" / "assets" / "md" / "a.md")
        d = discover(spec(self.root, import_path=self.root / "nope"))
        self.assertTrue(d.fallback)
        self.assertEqual([f.stem for f in d.files], ["a"])

    def test_empty_everywhere(self):
        d = discover(spec(self.root))
        self.assertEqual(d.files, ())
        self.assertTrue(d.fallback)

    def test_fallback_note_appended(self):
        import contextlib
        import io

        write(self.root / "data" / "assets" / "md" / "a.md")
        s = DiscoverSpec(patterns=("*.md",), import_path=self.root / "nope",
                         assets_dir=self.root / "data" / "assets" / "md",
                         fallback_note="图片相对路径已失锚")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            discover(s)
        self.assertIn("回退到资产层副本", buf.getvalue())
        self.assertIn("图片相对路径已失锚", buf.getvalue())

    # ---------- 同名消歧 ----------

    def test_duplicate_flagged_and_sid_disambiguated(self):
        write(self.root / "data" / "imports" / "md" / "sub_a" / "same.md")
        write(self.root / "data" / "imports" / "md" / "sub_b" / "same.md")
        write(self.root / "data" / "imports" / "md" / "only.md")
        d = discover(spec(self.root))
        by_name = {f.path.parent.name: f for f in d.files if f.stem == "same"}
        self.assertTrue(all(f.duplicate for f in by_name.values()))
        self.assertFalse(next(f for f in d.files if f.stem == "only").duplicate)
        import_path = self.root / "data" / "imports" / "md"
        sids = {f.sid("md", import_path, self.root / "assets") for f in by_name.values()}
        self.assertEqual(len(sids), 2, "同名必须消歧成两个 id")
        self.assertTrue(all(s.startswith("md:same--") for s in sids))
        # 指纹不可读：不得含目录名
        for s in sids:
            self.assertNotIn("sub_a", s)
            self.assertNotIn("sub_b", s)

    def test_duplicate_warning_printed(self):
        import contextlib
        import io

        write(self.root / "data" / "imports" / "md" / "sub_a" / "same.md")
        write(self.root / "data" / "imports" / "md" / "sub_b" / "same.md")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            discover(spec(self.root))
        out = buf.getvalue()
        self.assertIn("同名 Markdown", out)
        self.assertIn("md:<stem>--<sha8>", out)

    # ---------- 导入路径是文件 ----------

    def test_import_path_can_be_a_file(self):
        f = write(self.root / "data" / "imports" / "engineer" / "登记表.xlsx")
        d = discover(DiscoverSpec(patterns=("*.xlsx", "*.xlsm"), import_path=f,
                                  assets_dir=self.root / "data" / "assets" / "excel",
                                  sid_prefix="excel", label="Excel", log_prefix="[t]"))
        self.assertEqual([x.path for x in d.files], [f])
        self.assertFalse(d.fallback)

    def test_import_path_file_with_wrong_suffix_warns_and_skips(self):
        import contextlib
        import io

        f = write(self.root / "data" / "imports" / "engineer" / "旧表.xls")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            d = discover(DiscoverSpec(patterns=("*.xlsx",), import_path=f,
                                      assets_dir=self.root / "data" / "assets" / "excel",
                                      label="Excel", log_prefix="[t]"))
        self.assertEqual(d.files, ())
        self.assertFalse(d.fallback, "配置写错时不该静默改读资产层")
        self.assertIn("后缀不匹配", buf.getvalue())


class TestLatestVersions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_family_convergence(self):
        paths = [write(self.d / n, n, mtime=mt) for n, mt in
                 [("x.md", 10), ("x.aaaaaaaaaaaa.md", 20), ("y.md", 30),
                  ("y.bbbbbbbbbbbb.md", 30)]]
        kept = {p.name: base for p, base in latest_versions(paths)}
        self.assertEqual(set(kept), {"x.aaaaaaaaaaaa.md", "y.md"})
        self.assertEqual(set(kept.values()), {"x", "y"})

    def test_non_version_names_are_separate_families(self):
        paths = [write(self.d / "a.md"), write(self.d / "b.md"),
                 write(self.d / "c.zzzzzzzzzzzz.md")]  # 12 位非 hex → 独立族
        self.assertEqual({b for _p, b in latest_versions(paths)},
                         {"a", "b", "c.zzzzzzzzzzzz"})

    def test_missing_file_skipped(self):
        self.assertEqual(latest_versions([self.d / "gone.md"]), [])


class TestSourceFileSid(unittest.TestCase):
    def test_no_suffix_when_unique(self):
        f = SourceFile(path=Path("a.md"), from_imports=True, stem="a")
        self.assertEqual(f.sid("md", Path("i"), Path("s")), "md:a")

    def test_rel_key_falls_back_to_name(self):
        f = SourceFile(path=Path("/other/a.md"), from_imports=False, stem="a", duplicate=True)
        self.assertEqual(f.rel_key(Path("/base")), "a.md")

    def test_path_tag_stable_and_normalized(self):
        self.assertEqual(path_tag("a/b.md"), path_tag("a\\b.md"))
        self.assertEqual(len(path_tag("x")), 8)
        self.assertNotEqual(path_tag("sub_a/same.md"), path_tag("sub_b/same.md"))


class TestRelDirsBySid(unittest.TestCase):
    """rel_dir = imports **第一层子目录**（审核台文件夹分组的数据源）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_level_dir_root_and_missing(self):
        s = spec(self.root)
        write(self.root / "data" / "imports" / "md" / "故障" / "a.md")
        write(self.root / "data" / "imports" / "md" / "根文件.md")
        m = rel_dirs_by_sid(s)
        self.assertEqual(m["md:a"], "故障")
        self.assertEqual(m["md:根文件"], "")

    def test_deep_nesting_collapses_to_first_level(self):
        write(self.root / "data" / "imports" / "md" / "x" / "y" / "deep.md")
        m = rel_dirs_by_sid(spec(self.root))
        self.assertEqual(m["md:deep"], "x")

    def test_assets_fallback_files_not_mapped(self):
        """回退资产层的文件（imports 缺失）没有目录归属——分组视图显示"未分组"。"""
        write(self.root / "data" / "assets" / "md" / "a.md")
        s = DiscoverSpec(patterns=("*.md",), import_path=self.root / "nope",
                         assets_dir=self.root / "data" / "assets" / "md",
                         sid_prefix="md", label="Markdown", log_prefix="[t]")
        self.assertEqual(rel_dirs_by_sid(s), {})

    def test_duplicate_sids_match_discover(self):
        """重名消歧文件的 sid（带 --sha8）必须与 discover/canonicalize 的一致，
        否则映射对不上、文档分错组。"""
        write(self.root / "data" / "imports" / "md" / "sub_a" / "same.md")
        write(self.root / "data" / "imports" / "md" / "sub_b" / "same.md")
        m = rel_dirs_by_sid(spec(self.root))
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            d = discover(spec(self.root))
        sids = {f.sid("md", self.root / "data" / "imports" / "md",
                      self.root / "data" / "assets" / "md") for f in d.files}
        self.assertEqual(set(m), sids)
        self.assertEqual(sorted(m.values()), ["sub_a", "sub_b"])

    def test_empty_when_no_files(self):
        self.assertEqual(rel_dirs_by_sid(spec(self.root)), {})


if __name__ == "__main__":
    unittest.main()
