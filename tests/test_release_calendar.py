"""版本日历测试：版本形态判断（正式版/rc）、tag->日期映射、生成脚本的丢数据防护。"""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_release_calendar as brc  # noqa: E402

from vllm_kb.confidence import (
    load_release_calendar,
    load_release_meta,
    version_at_date,
    version_kind,
)


def make_calendar(path: Path) -> None:
    data = {
        "generated_at": "2026-08-18T00:00:00Z",
        "repo": "vllm-project/vllm-ascend",
        "releases": [
            {"tag": "v0.18.0", "date": "2026-04-30T00:00:00Z", "prerelease": False, "kind": "release"},
            {"tag": "v0.20.2rc1", "date": "2026-06-03T00:00:00Z", "prerelease": True, "kind": "rc"},
            {"tag": "v0.23.0", "date": "2026-08-16T00:00:00Z", "prerelease": False, "kind": "release"},
            {"tag": "v0.23.0rc1", "date": "2026-07-19T00:00:00Z", "prerelease": True, "kind": "rc"},
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class TestReleaseCalendar(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "release_calendar.json"
        make_calendar(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_calendar_new_format(self):
        cal = load_release_calendar(self.path)
        self.assertIsNotNone(cal)
        self.assertIn("v0.18.0", cal)
        self.assertIn("v0.23.0rc1", cal)
        self.assertIsInstance(cal["v0.18.0"], datetime)

    def test_version_at_date(self):
        cal = load_release_calendar(self.path)
        dt = datetime(2026, 5, 10, tzinfo=timezone.utc)
        # 2026-05-10 之前最近发布 = v0.18.0
        self.assertEqual(version_at_date(cal, dt), "v0.18.0")

    def test_version_kind_release(self):
        meta = load_release_meta(self.path)
        self.assertEqual(version_kind(meta, "0.18.0"), "release")
        self.assertEqual(version_kind(meta, "v0.18.0"), "release")

    def test_version_kind_rc(self):
        meta = load_release_meta(self.path)
        self.assertEqual(version_kind(meta, "v0.23.0rc1"), "rc")
        self.assertEqual(version_kind(meta, "0.23.0rc1"), "rc")

    def test_version_kind_unknown(self):
        meta = load_release_meta(self.path)
        self.assertEqual(version_kind(meta, "v9.9.9"), "unknown")

    def test_version_kind_no_meta(self):
        self.assertEqual(version_kind(None, "0.18.0"), "unknown")

    def test_load_meta_returns_none_missing(self):
        self.assertIsNone(load_release_meta(Path(self.tmp.name) / "nope.json"))


class TestCalendarNoDataLoss(unittest.TestCase):
    """生成脚本：抓取失败/空结果绝不能把好日历覆盖掉（实测踩过：代理掐 TLS 后
    一次 deploy 就把带 v0.23.0 的日历写成 releases: []，w_ver 与图关联整体消失）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "release_calendar.vllm-project-vllm-ascend.json"
        make_calendar(self.path)
        self.cfg = mock.Mock()
        self.cfg.resolve.side_effect = lambda p: Path(self.tmp.name) / Path(p).name

    def tearDown(self):
        self.tmp.cleanup()

    def _before(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_fetch_raises_on_first_page_failure(self):
        # 拉取走 requests（get_session，代理掐 TLS 自动直连兜底）；mock 掉 session 验证
        # fetch_releases 对拉取失败的"不返回残缺列表"契约
        session = mock.MagicMock()
        session.get.side_effect = OSError("proxy closed TLS tunnel")
        with mock.patch("vllm_kb.net.get_session", return_value=session):
            with self.assertRaises(brc.CalendarFetchError) as cm:
                brc.fetch_releases("vllm-project/vllm-ascend")
        self.assertEqual(cm.exception.page, 1)
        self.assertEqual(cm.exception.partial, [])
        self.assertIn("proxy closed TLS tunnel", str(cm.exception))

    def test_fetch_raises_on_later_page_failure_not_partial_result(self):
        """第 2 页失败 → 抛错（不返回残缺列表，否则日历会比原来更小）。"""
        page1 = [
            {"tag_name": f"v0.{i}.0", "published_at": "2026-01-01T00:00:00Z", "prerelease": False}
            for i in range(100)
        ]
        resp = mock.MagicMock()
        resp.json.return_value = page1
        session = mock.MagicMock()
        session.get.side_effect = [resp, OSError("boom")]
        with mock.patch("vllm_kb.net.get_session", return_value=session):
            with self.assertRaises(brc.CalendarFetchError) as cm:
                brc.fetch_releases("vllm-project/vllm-ascend")
        self.assertEqual(cm.exception.page, 2)
        self.assertEqual(len(cm.exception.partial), 100)

    def test_write_refuses_empty_over_existing(self):
        before = self._before()
        empty = {"generated_at": "x", "repo": "vllm-project/vllm-ascend", "releases": []}
        with self.assertRaises(brc.CalendarWriteRefused) as cm:
            brc.write_calendar(self.cfg, empty)
        self.assertIn("拒绝覆盖", str(cm.exception))
        self.assertEqual(self._before(), before, "现有日历必须原样保留")

    def test_write_allows_first_creation_empty(self):
        """首次生成（无现有文件）允许写空——那才是"该仓库真没有 release"。"""
        fresh = Path(self.tmp.name) / "fresh.json"
        empty = {"generated_at": "x", "repo": "vllm-project/vllm-ascend", "releases": []}
        out = brc.write_calendar(self.cfg, empty, path=fresh)
        self.assertEqual(out, fresh)
        self.assertEqual(json.loads(fresh.read_text(encoding="utf-8"))["releases"], [])

    def test_write_force_overrides_guard(self):
        before = self._before()
        self.assertTrue(before["releases"], "前置条件：现有日历非空")
        empty = {"generated_at": "x", "repo": "vllm-project/vllm-ascend", "releases": []}
        brc.write_calendar(self.cfg, empty, force=True)
        self.assertEqual(self._before()["releases"], [])

    def test_write_normal_result_replaces(self):
        cal = {"generated_at": "x", "repo": "vllm-project/vllm-ascend",
               "releases": [{"tag": "v9.9.9", "date": "2026-09-01T00:00:00Z",
                             "prerelease": False, "kind": "release"}]}
        brc.write_calendar(self.cfg, cal)
        self.assertEqual([r["tag"] for r in self._before()["releases"]], ["v9.9.9"])

    def test_would_drop_releases_ignores_corrupt_existing(self):
        """现有文件损坏 → 不拦（否则永远修不回来）。"""
        self.path.write_text("{ not json", encoding="utf-8")
        empty = {"generated_at": "x", "repo": "r", "releases": []}
        self.assertEqual(brc.would_drop_releases(self.path, empty), "")


if __name__ == "__main__":
    unittest.main()
