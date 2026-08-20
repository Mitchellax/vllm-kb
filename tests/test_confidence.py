"""置信度模型单元测试。"""
import unittest
from datetime import datetime, timedelta, timezone

from vllm_kb.confidence import (
    compute_confidence,
    final_score,
    parse_iso,
    parse_version,
    reliability_score,
    time_weight,
    version_distance,
    version_weight,
)


class TestParse(unittest.TestCase):
    def test_parse_version(self):
        self.assertEqual(parse_version("v0.6.4"), (0, 6, 4))
        self.assertEqual(parse_version("0.6.4.post1"), (0, 6, 4))
        self.assertEqual(parse_version("0.6"), (0, 6, 0))
        self.assertIsNone(parse_version("abc"))
        self.assertIsNone(parse_version(None))

    def test_parse_iso(self):
        dt = parse_iso("2024-03-01T10:00:00Z")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertIsNone(parse_iso("garbage"))
        self.assertIsNone(parse_iso(None))

    def test_version_distance(self):
        self.assertAlmostEqual(version_distance((0, 6, 4), (0, 6, 1)), 0.3)
        self.assertAlmostEqual(version_distance((0, 6, 0), (0, 7, 0)), 1.0)


class TestTimeWeight(unittest.TestCase):
    def test_recent_is_high(self):
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(days=7)).isoformat()
        w = time_weight(recent, None, now=now, half_life_days=365, floor=0.15)
        self.assertGreater(w, 0.5)

    def test_decays_monotonically(self):
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=365)).isoformat()
        new = (now - timedelta(days=30)).isoformat()
        w_old = time_weight(old, None, now=now, half_life_days=365, floor=0.15)
        w_new = time_weight(new, None, now=now, half_life_days=365, floor=0.15)
        self.assertLess(w_old, w_new)

    def test_floor_keeps_old_knowledge(self):
        now = datetime.now(timezone.utc)
        very_old = (now - timedelta(days=3650)).isoformat()
        w = time_weight(very_old, None, now=now, half_life_days=365, floor=0.15)
        self.assertAlmostEqual(w, 0.15, places=2)

    def test_resolved_at_is_reference(self):
        # 已修复知识：从 resolved_at 起衰退；未修复：从 created_at 起衰退
        now = datetime.now(timezone.utc)
        created = (now - timedelta(days=1000)).isoformat()
        resolved = (now - timedelta(days=30)).isoformat()
        w_resolved = time_weight(created, resolved, now=now)
        w_open = time_weight(created, None, now=now)
        self.assertGreater(w_resolved, w_open)

    def test_no_dates_returns_one(self):
        self.assertEqual(time_weight(None, None), 1.0)


class TestVersionWeight(unittest.TestCase):
    def test_inside_interval(self):
        self.assertEqual(version_weight("0.5.0", "0.6.4", "0.6.0"), 1.0)

    def test_below_min_decays(self):
        w = version_weight("0.6.0", None, "0.4.0", sigma=1.5)
        self.assertLess(w, 1.0)
        self.assertGreater(w, 0.0)

    def test_above_max_decays(self):
        # 修复已落地到 0.6.4，目标 0.7.0 大概率已无此问题 -> 权重下降
        w = version_weight("0.5.0", "0.6.4", "0.7.0", sigma=1.5)
        self.assertLess(w, 1.0)

    def test_unknown_returns_default(self):
        self.assertEqual(version_weight(None, None, "0.6.0", unknown_weight=0.5), 0.5)
        self.assertEqual(version_weight("0.5.0", None, None, unknown_weight=0.5), 0.5)

    def test_closer_higher(self):
        w1 = version_weight(None, "0.6.4", "0.6.3", sigma=1.5)
        w2 = version_weight(None, "0.6.4", "0.8.0", sigma=1.5)
        self.assertGreater(w1, w2)


class TestReliability(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(reliability_score("github_issue", "open", None), 0.4)
        self.assertEqual(reliability_score("github_issue", "closed", "2024-01-01T00:00:00Z"), 0.6)
        self.assertEqual(reliability_score("github_pr", "merged", None), 0.9)
        self.assertEqual(reliability_score("official_doc", "closed", None), 0.85)
        self.assertEqual(reliability_score("wiki", "open", None), 0.7)

    def test_explicit_wins(self):
        self.assertEqual(reliability_score("github_issue", "open", None, explicit=0.95), 0.95)

    def test_kind_adjustment(self):
        """反馈类 issue（doc/feature/rfc）降权；bug/fix 不降。"""
        base = reliability_score("github_issue", "open", None)
        self.assertEqual(reliability_score("github_issue", "open", None, kind="bug"), base)
        self.assertLess(reliability_score("github_issue", "open", None, kind="doc"), base)
        self.assertLess(reliability_score("github_issue", "open", None, kind="feature"), base)
        self.assertLess(reliability_score("github_issue", "open", None, kind="rfc"), base)
        self.assertEqual(reliability_score("github_issue", "open", None, kind="fix"), base)

    def test_verification_factor_lifts_reliability(self):
        """验证状态因子（维度 B）：expert/tested/unverified 作为可靠度下限提升。"""
        # 官方手册：status=open → 0.4，但 verification=expert → 0.95
        self.assertEqual(
            reliability_score("doc_pdf", "open", None, verification="expert"), 0.95)
        # tested → 0.85
        self.assertEqual(
            reliability_score("doc_pdf", "open", None, verification="tested"), 0.85)
        # unverified → 0.5（小幅保底，不覆盖已有规则分）
        self.assertEqual(
            reliability_score("doc_pdf", "open", None, verification="unverified"), 0.5)
        # 已有高分不被拉低：merged_fix 0.9 + expert 0.95 → max=0.95
        self.assertEqual(
            reliability_score("github_pr", "merged", None, verification="expert"), 0.95)
        # closed 0.6 + tested 0.85 → 0.85
        self.assertEqual(
            reliability_score("github_issue", "closed", "2024-01-01T00:00:00Z",
                              verification="tested"), 0.85)
        # 无 verification → 规则分不变
        self.assertEqual(reliability_score("doc_pdf", "open", None), 0.4)
        # compute_confidence 透传 verification 到 breakdown
        from vllm_kb.confidence import compute_confidence
        b = compute_confidence(None, None, "open", "doc_pdf", None, None, None,
                               verification="expert")
        self.assertEqual(b.reliability, 0.95)
        self.assertEqual(b.extras.get("verification"), "expert")


class TestCombine(unittest.TestCase):
    def test_compute_confidence_breakdown(self):
        br = compute_confidence(
            created_at="2024-01-01T00:00:00Z",
            resolved_at="2024-02-01T00:00:00Z",
            status="closed",
            source_type="github_issue",
            span_min="0.5.0",
            span_max="0.6.4",
            target_version="0.6.0",
            now=datetime(2024, 3, 1, tzinfo=timezone.utc),
        )
        self.assertGreater(br.score, 0)
        self.assertLessEqual(br.score, 1.0)
        self.assertGreater(br.time_weight, 0)
        self.assertAlmostEqual(br.version_weight, 1.0)
        self.assertEqual(br.reliability, 0.6)

    def test_final_score_blends(self):
        self.assertAlmostEqual(final_score(1.0, 1.0), 1.0)
        self.assertAlmostEqual(final_score(0.0, 1.0), 0.0)
        self.assertGreater(final_score(0.8, 0.5), final_score(0.3, 0.5))

    def test_final_score_clips_negative_sim(self):
        self.assertGreaterEqual(final_score(-0.5, 0.9), 0.0)


if __name__ == "__main__":
    unittest.main()
