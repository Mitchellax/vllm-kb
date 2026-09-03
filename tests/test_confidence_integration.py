"""置信度打分集成单测：w_hist 并入 final + API 响应暴露 + 中性不变。

验证：
- final = sim^γ × conf^(1-γ) × w_hist^σ（乘性，w_hist=1 时不变，w_hist=0 时 final=0）
- compute_confidence 输出 w_hist/n_eff/history_flag
- feedback_enabled=False 时 w_hist=1.0（中性，不改 score）
- API /search 响应含 w_hist/n_eff/history_flag
"""
import math
import unittest
from datetime import datetime, timezone
from unittest import mock

from vllm_kb.config import ConfidenceCfg
from vllm_kb.confidence import compute_confidence, final_score


class TestFinalScoreWithWHist(unittest.TestCase):
    def test_neutral_w_hist_unchanged(self):
        """w_hist=1.0 时 final 不变（1.0^σ=1.0）。"""
        base = final_score(0.8, 0.7, gamma=0.6)
        with_hist = final_score(0.8, 0.7, gamma=0.6, w_hist=1.0, sigma=0.5)
        self.assertAlmostEqual(base, with_hist)

    def test_low_w_hist_reduces_final(self):
        """w_hist<1 时 final 降低。"""
        base = final_score(0.8, 0.7, gamma=0.6)
        low = final_score(0.8, 0.7, gamma=0.6, w_hist=0.3, sigma=0.5)
        self.assertLess(low, base)

    def test_zero_w_hist_zero_final(self):
        """w_hist=0 时 final=0（历史全负不被相似度翻盘）。"""
        self.assertEqual(final_score(0.9, 0.9, gamma=0.6, w_hist=0.0, sigma=0.5), 0.0)

    def test_zero_sim_zero_final(self):
        """sim=0 时 final=0（无匹配不被历史翻盘）。"""
        self.assertEqual(final_score(0.0, 0.9, gamma=0.6, w_hist=0.9, sigma=0.5), 0.0)

    def test_disabled_feedback_neutral(self):
        """feedback_enabled=False → w_hist=1.0, n_eff=0, flag=new。"""
        cfg = ConfidenceCfg(feedback_enabled=False)
        cb = compute_confidence(
            created_at="2024-01-01T00:00:00Z", resolved_at="2024-06-01T00:00:00Z",
            status="closed", source_type="github_issue",
            span_min=None, span_max=None, target_version="0.6.1",
            cfg=cfg, doc_id="doc:1",
        )
        self.assertEqual(cb.w_hist, 1.0)
        self.assertEqual(cb.n_eff, 0.0)
        self.assertEqual(cb.history_flag, "new")

    def test_enabled_no_feedback_file_neutral(self):
        """feedback_enabled=True 但无反馈文件 → w_hist=1.0（新文档中性）。"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            cfg = ConfidenceCfg(
                feedback_enabled=True,
                feedback_path=str(Path(d) / "nonexistent.json"),
            )
            # 清缓存
            from vllm_kb import confidence as conf_mod
            conf_mod._feedback_table_cache.clear()
            cb = compute_confidence(
                created_at="2024-01-01T00:00:00Z", resolved_at=None,
                status="open", source_type="github_issue",
                span_min=None, span_max=None, target_version=None,
                cfg=cfg, doc_id="doc:new",
            )
            self.assertEqual(cb.w_hist, 1.0)
            self.assertEqual(cb.history_flag, "new")

    def test_with_feedback_data_w_hist_nontrivial(self):
        """有反馈数据时 w_hist = lb（非 1.0）。"""
        import tempfile
        from pathlib import Path

        from vllm_kb import confidence as conf_mod
        from vllm_kb.feedback_model import DocFeedback, save_feedback_table

        with tempfile.TemporaryDirectory() as d:
            fb_path = Path(d) / "fb.json"
            save_feedback_table(fb_path, {
                "doc:1": DocFeedback(a=10.0, b=1.0, n_unknown=0.0,
                                     last_update_ts="2025-01-01T00:00:00+00:00"),
            })
            cfg = ConfidenceCfg(
                feedback_enabled=True, feedback_path=str(fb_path),
            )
            conf_mod._feedback_table_cache.clear()
            cb = compute_confidence(
                created_at="2024-01-01T00:00:00Z", resolved_at=None,
                status="open", source_type="github_issue",
                span_min=None, span_max=None, target_version=None,
                cfg=cfg, doc_id="doc:1",
            )
            self.assertLess(cb.w_hist, 1.0)  # lb < 1（有 b=1 的未命中）
            self.assertGreater(cb.n_eff, 0)
            self.assertEqual(cb.history_flag, "supported")
            conf_mod._feedback_table_cache.clear()


@unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi"),
    "fastapi 未安装",
)
class TestApiSearchExposesHistFields(unittest.TestCase):
    def test_search_response_has_hist_fields(self):
        import os
        from pathlib import Path

        from fastapi.testclient import TestClient

        from vllm_kb.api import create_app
        from vllm_kb.config import PROJECT_ROOT

        # 部署冒烟：/search 需真实 KB。config.json 被 gitignore——fresh clone/CI 跳过。
        cfg_path = PROJECT_ROOT / "config.json"
        if not cfg_path.exists():
            self.skipTest("config.json 不存在（fresh clone/CI）——/search 冒烟仅在部署机运行")

        old_e = os.environ.get("EMBEDDING_API_KEY")
        old_g = os.environ.get("GITHUB_TOKEN")
        os.environ["EMBEDDING_API_KEY"] = "dummy"
        os.environ["GITHUB_TOKEN"] = "dummy"
        try:
            client = TestClient(create_app(str(cfg_path)))
            r = client.post("/search", json={"query": "test", "top_k": 1})
            self.assertEqual(r.status_code, 200)
            results = r.json().get("results", [])
            if results:
                conf = results[0].get("confidence", {})
                self.assertIn("w_hist", conf)
                self.assertIn("n_eff", conf)
                self.assertIn("history_flag", conf)
        finally:
            for name, old in (("EMBEDDING_API_KEY", old_e), ("GITHUB_TOKEN", old_g)):
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old


if __name__ == "__main__":
    unittest.main()
