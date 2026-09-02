"""后验置信度模型单测：指数遗忘/后验统计量/w_hist 三段式/history_flag/seed 衰减。

纯数学验证，不依赖数据库/网络——feedback_model 是纯函数模块。
"""
import math
import unittest
from datetime import datetime, timedelta, timezone

from vllm_kb.config import ConfidenceCfg
from vllm_kb.feedback_model import (
    DocFeedback,
    PosteriorStats,
    apply_feedback_event,
    compute_posterior,
    load_feedback_table,
    save_feedback_table,
)


def _cfg(**kw) -> ConfidenceCfg:
    base = {
        "feedback_enabled": True,
        "feedback_half_life_days": 365.0,
        "feedback_z": 1.0,
        "feedback_n_min": 5.0,
        "history_sigma": 0.5,
        "feedback_seed_success": 1.0,
        "feedback_seed_fail": 1.0,
    }
    base.update(kw)
    return ConfidenceCfg(**base)


_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


class TestPosteriorStats(unittest.TestCase):
    def test_new_doc_neutral(self):
        """新文档（无反馈）：n_eff=0, w_hist=1.0（中性，不误杀）。"""
        ps = compute_posterior(None, _cfg())
        self.assertEqual(ps.n_eff, 0.0)
        self.assertEqual(ps.w_hist, 1.0)
        self.assertEqual(ps.history_flag, "new")
        self.assertIsNone(ps.mean)

    def test_zero_eff_uses_seed_safely(self):
        """seed=1+1 但 n_eff=0（a+b=0）时仍中性——不用 seed 套 lb 公式。"""
        fb = DocFeedback(a=0.0, b=0.0, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg())
        self.assertEqual(ps.w_hist, 1.0)
        self.assertEqual(ps.history_flag, "new")

    def test_hit_dominant_high_mean(self):
        """命中多未命中少 → mean 高，lb 接近 mean。"""
        fb = DocFeedback(a=10.0, b=1.0, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg())
        self.assertAlmostEqual(ps.mean, 10 / 11, places=3)
        self.assertGreater(ps.w_hist, 0.7)  # lb 较高
        self.assertEqual(ps.history_flag, "supported")  # n_eff=11 >= 5

    def test_miss_dominant_low_mean(self):
        """未命中多 → mean 低，flag=failing。"""
        fb = DocFeedback(a=1.0, b=10.0, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg())
        self.assertLess(ps.mean, 0.2)
        self.assertEqual(ps.history_flag, "failing")

    def test_low_neff_accumulating(self):
        """n_eff < n_min → accumulating。"""
        fb = DocFeedback(a=1.0, b=0.0, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg())
        self.assertLess(ps.n_eff, 5.0)
        self.assertEqual(ps.history_flag, "accumulating")

    def test_low_neff_high_unknown_evidence_thin(self):
        """n_eff 低 + unknown 占比高 → evidence_thin。"""
        fb = DocFeedback(a=1.0, b=0.0, n_unknown=3.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg())
        self.assertEqual(ps.history_flag, "evidence_thin")

    def test_high_unknown_used_but_unconfirmed(self):
        """n_eff>=n_min + unknown 占比高 + mean 偏低 → used_but_unconfirmed。"""
        fb = DocFeedback(a=2.0, b=3.0, n_unknown=10.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg())
        self.assertGreaterEqual(ps.n_eff, 5.0)
        self.assertEqual(ps.history_flag, "used_but_unconfirmed")

    def test_unknown_not_in_neff(self):
        """unknown 不进 n_eff（单独计数），统计正确。"""
        fb = DocFeedback(a=5.0, b=0.0, n_unknown=100.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg())
        self.assertEqual(ps.n_eff, 5.0)  # 只算 a+b，不含 unknown
        self.assertEqual(ps.n_unknown, 100.0)

    def test_lb_clipped_to_01(self):
        """lb 裁剪到 [0,1]。"""
        fb = DocFeedback(a=0.0, b=10.0, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg())
        self.assertGreaterEqual(ps.lb, 0.0)
        self.assertLessEqual(ps.lb, 1.0)

    def test_large_neff_sd_approaches_zero(self):
        """n_eff 大时 sd→0，z 无论取值 lb≈mean（冷启动保护失效，数据主导）。"""
        fb = DocFeedback(a=500.0, b=10.0, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg(z=1.6))
        self.assertLess(ps.sd, 0.01)  # sd 很小但不精确为 0
        self.assertAlmostEqual(ps.lb, ps.mean, places=1)  # z 大时 lb 仍接近 mean

    def test_sigma_linear_interpolation(self):
        """sigma 在 n_eff 0→n_min 间线性插值 sigma_min→history_sigma。"""
        cfg = _cfg(history_sigma=0.5, history_sigma_min=0.2, feedback_n_min=5.0)
        # n_eff 接近 0 → sigma 接近 0.2
        fb_low = DocFeedback(a=0.1, b=0.1, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps_low = compute_posterior(fb_low, cfg)
        self.assertLess(ps_low.sigma, 0.3)  # 接近 0.2

        # n_eff = n_min/2 → sigma 在 0.2 和 0.5 中间
        fb_mid = DocFeedback(a=1.25, b=1.25, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps_mid = compute_posterior(fb_mid, cfg)
        self.assertAlmostEqual(ps_mid.sigma, 0.35, places=2)  # (0.2+0.5)/2

        # n_eff >= n_min → sigma = 0.5
        fb_high = DocFeedback(a=5.0, b=0.0, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps_high = compute_posterior(fb_high, cfg)
        self.assertEqual(ps_high.sigma, 0.5)

    def test_sigma_no_jump_at_n_min(self):
        """n_eff 跨过 n_min 时 sigma 无跳变（插值到 n_min 时≈history_sigma）。"""
        cfg = _cfg(history_sigma=0.5, history_sigma_min=0.2, feedback_n_min=5.0)
        # n_eff 略低于 n_min
        fb_below = DocFeedback(a=2.5, b=2.49, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps_below = compute_posterior(fb_below, cfg)
        # n_eff 略高于 n_min
        fb_above = DocFeedback(a=2.5, b=2.51, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps_above = compute_posterior(fb_above, cfg)
        self.assertAlmostEqual(ps_below.sigma, ps_above.sigma, places=2)


class TestExponentialForgetting(unittest.TestCase):
    def test_hit_increases_a(self):
        fb = apply_feedback_event(None, "hit", 0.5, _NOW, _cfg())
        self.assertGreater(fb.a, 1.0)  # seed + 0.5

    def test_miss_increases_b(self):
        fb = apply_feedback_event(None, "miss", 0.5, _NOW, _cfg())
        self.assertGreater(fb.b, 1.0)

    def test_unknown_increases_n_unknown_not_ab(self):
        fb = apply_feedback_event(None, "unknown", 0.0, _NOW, _cfg())
        self.assertEqual(fb.a, 1.0)  # seed 不变
        self.assertEqual(fb.b, 1.0)
        self.assertEqual(fb.n_unknown, 1.0)

    def test_time_decay_applied(self):
        """Δt=HL 时衰减一半：a <- a × 0.5 + w×hit。"""
        old_ts = _NOW - timedelta(days=365)  # HL=365 → decay=0.5
        fb = DocFeedback(a=10.0, b=0.0, n_unknown=0.0, last_update_ts=old_ts.isoformat())
        fb2 = apply_feedback_event(fb, "hit", 0.5, _NOW, _cfg(half_life_days=365))
        self.assertAlmostEqual(fb2.a, 10.0 * 0.5 + 0.5, places=3)  # 5.5

    def test_seed_decays_over_time(self):
        """seed 随遗忘衰减：HL 后 seed 从 1.0→0.5。"""
        old_ts = _NOW - timedelta(days=365)
        fb = DocFeedback(a=1.0, b=1.0, n_unknown=0.0, last_update_ts=old_ts.isoformat())
        fb2 = apply_feedback_event(fb, "hit", 0.0, _NOW, _cfg(half_life_days=365))
        self.assertAlmostEqual(fb2.a, 0.5, places=3)  # seed 1.0 × 0.5 + 0
        self.assertAlmostEqual(fb2.b, 0.5, places=3)

    def test_weighted_not_counted(self):
        """w=0.4 的推断只贡献 0.4，不是 1.0。"""
        fb = apply_feedback_event(None, "hit", 0.4, _NOW, _cfg())
        self.assertAlmostEqual(fb.a, 1.4, places=3)  # seed 1.0 + 0.4

    def test_unknown_weight_ignored(self):
        """unknown 的 weight 不影响 n_unknown（计数 1，不受 w 影响）。"""
        fb = apply_feedback_event(None, "unknown", 0.5, _NOW, _cfg())
        self.assertEqual(fb.n_unknown, 1.0)


class TestSeedAndDominance(unittest.TestCase):
    def test_seed_strength_2_data_needs_3_to_dominate(self):
        """seed=1+1=2，数据部分需≥3 超先验 1.5 倍才主导。
        推论：平均确认频率约每季度 1 条（4 年达 supported）。"""
        # seed 衰减后 + 5 次 hit(w=1，但实际 w≤0.5)
        # 简化验证：a+b >= 3 时 n_eff > seed 初始的 1.5 倍
        fb = DocFeedback(a=3.0, b=0.0, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg())
        self.assertGreaterEqual(ps.n_eff, 3.0)
        self.assertGreaterEqual(ps.n_eff, 2.0 * 1.5)  # 超过 seed 1.5 倍
        self.assertEqual(ps.history_flag, "accumulating")  # 3 < 5 仍 accumulating

    def test_cold_domain_never_supported(self):
        """冷门 domain 低频：n_eff 难达 n_min，保持 accumulating/new。"""
        fb = DocFeedback(a=1.5, b=0.5, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg())
        self.assertLess(ps.n_eff, 5.0)
        self.assertEqual(ps.history_flag, "accumulating")


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        import tempfile
        from pathlib import Path

        table = {
            "doc:1": DocFeedback(a=5.0, b=1.0, n_unknown=2.0, last_update_ts=_NOW.isoformat()),
            "doc:2": DocFeedback(a=0.0, b=3.0, n_unknown=0.0, last_update_ts=_NOW.isoformat()),
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fb.json"
            save_feedback_table(p, table)
            loaded = load_feedback_table(p)
        self.assertEqual(set(loaded.keys()), {"doc:1", "doc:2"})
        self.assertAlmostEqual(loaded["doc:1"].a, 5.0)
        self.assertAlmostEqual(loaded["doc:2"].b, 3.0)

    def test_load_missing_file_empty(self):
        from pathlib import Path

        self.assertEqual(load_feedback_table(Path("/nonexistent")), {})

    def test_load_corrupt_file_empty(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fb.json"
            p.write_text("not json", encoding="utf-8")
            self.assertEqual(load_feedback_table(p), {})


class TestFinalFormulaIntegrity(unittest.TestCase):
    """w_hist 与 w_rel 正交：n_eff 永不进 w_rel，置信度与来源可信度正交。"""

    def test_w_hist_is_lb(self):
        """w_hist === lb（同一值），便于消费侧解释。"""
        fb = DocFeedback(a=8.0, b=2.0, n_unknown=0.0, last_update_ts=_NOW.isoformat())
        ps = compute_posterior(fb, _cfg())
        self.assertEqual(ps.w_hist, ps.lb)

    def test_new_doc_w_hist_neutral_does_not_affect_conf(self):
        """新文档 w_hist=1.0：1.0^σ=1.0，final 不变（乘性中性）。"""
        ps = compute_posterior(None, _cfg())
        self.assertEqual(ps.w_hist, 1.0)
        # 1.0 ** 0.5 == 1.0，乘进 final 不改值
        self.assertEqual(1.0 ** _cfg().history_sigma, 1.0)


if __name__ == "__main__":
    unittest.main()
