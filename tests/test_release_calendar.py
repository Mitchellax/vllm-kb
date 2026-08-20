"""版本日历测试：版本形态判断（正式版/rc）、tag->日期映射。"""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
