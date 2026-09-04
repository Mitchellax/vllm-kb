"""build_feedback.py 离线推断单测：会话重建+行为推断三态+缺口判决+后验更新。

用造的遥测事件（不依赖真实 serve_api）验证推断规则正确性。
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_feedback as bf
from vllm_kb.config import AppConfig
from vllm_kb.feedback_model import DocFeedback, apply_feedback_event, load_feedback_table


def _cfg(tmpdir):
    """合成最小配置（不读本机 config.json——gitignored 文件不能作测试夹具，
    且本机 confidence 值会让测试行为机器间漂移）。"""
    cfg = AppConfig.model_validate({
        "embedding": {"provider": "echo"},
        "storage": {"sqlite_path": str(Path(tmpdir) / "kb.sqlite3")},
        "confidence": {
            "feedback_enabled": True,
            "telemetry_path": str(Path(tmpdir) / "telemetry.sqlite3"),
            "feedback_path": str(Path(tmpdir) / "confidence_feedback.json"),
        },
    })
    return cfg


def _event(session, ts, endpoint, doc_ids=None, count=0, sig_hash="", sig_entities="",
           query_norm="q", component="vllm-ascend"):
    return {
        "session_id": session, "ts": ts, "endpoint": endpoint, "method": "POST",
        "result_doc_ids": json.dumps(doc_ids or []), "result_count": count,
        "signature_hash": sig_hash, "signature_entities": sig_entities,
        "signature_text": "", "query_normalized": query_norm, "component": component,
    }


def _ts(minute):
    """生成 UTC iso 时间戳，minute 为相对偏移。"""
    return (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)).isoformat()


class TestProbeExclusion(unittest.TestCase):
    """探索/测试打标（probe≠0）事件不进反馈推断；旧库无 probe 列全量进（兼容）。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = _cfg(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _seed_db(self, events, with_probe_col=True):
        db = self.cfg.resolve(self.cfg.confidence.telemetry_path)
        cols = ("session_id TEXT NOT NULL, ts TEXT NOT NULL, endpoint TEXT NOT NULL, "
                "method TEXT, query_hash TEXT, query_normalized TEXT, "
                "signature_hash TEXT, signature_entities TEXT, signature_text TEXT, "
                "result_doc_ids TEXT, result_count INTEGER, component TEXT, "
                "target_version TEXT, repo TEXT"
                + (", probe INTEGER DEFAULT 0" if with_probe_col else ""))
        conn = sqlite3.connect(str(db))
        conn.execute(f"CREATE TABLE query_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
        if events:
            colnames = ",".join(events[0].keys())
            marks = ",".join("?" * len(events[0]))
            conn.executemany(
                f"INSERT INTO query_events ({colnames}) VALUES ({marks})",
                [tuple(e.values()) for e in events])
        conn.commit()
        conn.close()
        return db

    def test_probe_events_excluded_from_inference(self):
        """probe=1/2 事件被 _load_events 排除，probe=0 保留。"""
        db = self._seed_db([
            {"session_id": "s1", "ts": _ts(0), "endpoint": "/search",
             "query_normalized": "真实查询", "probe": 0},
            {"session_id": "s1", "ts": _ts(1), "endpoint": "/search",
             "query_normalized": "test", "probe": 2},
            {"session_id": "s1", "ts": _ts(2), "endpoint": "/search",
             "query_normalized": "另一真实", "probe": 1},
        ])
        events = bf._load_events(Path(db))
        norms = [e["query_normalized"] for e in events]
        self.assertEqual(norms, ["真实查询"])

    def test_legacy_db_without_probe_column_kept(self):
        """旧库无 probe 列：全部视为 probe=0 进推断（与打标机制引入前行为一致）。"""
        db = self._seed_db([
            {"session_id": "s1", "ts": _ts(0), "endpoint": "/search",
             "query_normalized": "test"},  # 无 probe 键
        ], with_probe_col=False)
        events = bf._load_events(Path(db))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["query_normalized"], "test")


class TestInferFeedback(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = _cfg(self.tmpdir)

    def test_signature_hit_then_end_weak_positive(self):
        """signature 命中后会话直接结束 → 弱正 hit。"""
        events = [
            _event("s1", _ts(0), "/signature-search", doc_ids=["doc:a"], count=1,
                   sig_hash="h1", sig_entities='[{"kind":"kernel","text":"foo"}]'),
        ]
        ev = bf.infer_feedback(events, self.cfg)
        hits = [e for e in ev if e["state"] == "hit"]
        self.assertTrue(any(h["doc_id"] == "doc:a" for h in hits))

    def test_search_then_pull_doc_no_research_weak_positive(self):
        """search 命中后拉 doc 且不重查 → 弱正 hit。"""
        events = [
            _event("s1", _ts(0), "/search", doc_ids=["doc:a"], count=1, query_norm="q1"),
            _event("s1", _ts(1), "/doc/doc:a"),
        ]
        ev = bf.infer_feedback(events, self.cfg)
        hits = [e for e in ev if e["state"] == "hit" and e["doc_id"] == "doc:a"]
        self.assertTrue(hits)

    def test_rephrase_within_60s_weak_negative(self):
        """60s 内改述重查 → 弱负 miss。"""
        events = [
            _event("s1", _ts(0), "/search", doc_ids=["doc:a"], count=1, query_norm="q1"),
            _event("s1", _ts(1), "/search", doc_ids=["doc:b"], count=1, query_norm="q2"),
        ]
        ev = bf.infer_feedback(events, self.cfg)
        misses = [e for e in ev if e["state"] == "miss" and e["doc_id"] == "doc:a"]
        self.assertTrue(misses)

    def test_code_after_doc_hit_medium_positive(self):
        """有 doc 命中后调 code → 中正 hit。"""
        events = [
            _event("s1", _ts(0), "/search", doc_ids=["doc:a"], count=1, query_norm="q1"),
            _event("s1", _ts(2), "/code/search", query_norm="keyword"),
        ]
        ev = bf.infer_feedback(events, self.cfg)
        hits = [e for e in ev if e["state"] == "hit" and e["doc_id"] == "doc:a"]
        self.assertTrue(hits)
        self.assertTrue(any(h["weight"] == 0.5 for h in hits))

    def test_code_without_doc_hit_no_evidence(self):
        """无 doc 命中后调 code → 无 doc 级证据（进缺口检测，不进后验）。"""
        events = [
            _event("s1", _ts(0), "/search", doc_ids=[], count=0, query_norm="q1"),
            _event("s1", _ts(2), "/code/search", query_norm="keyword"),
        ]
        ev = bf.infer_feedback(events, self.cfg)
        # 无命中时不产生 doc 级证据（没有目标 doc 可标）
        self.assertEqual(len(ev), 0)

    def test_zero_action_unknown(self):
        """search 命中后零后续 → unknown（不进 n_eff）。"""
        events = [
            _event("s1", _ts(0), "/search", doc_ids=["doc:a"], count=1, query_norm="q1"),
        ]
        ev = bf.infer_feedback(events, self.cfg)
        unknowns = [e for e in ev if e["state"] == "unknown" and e["doc_id"] == "doc:a"]
        self.assertTrue(unknowns)

    def test_unknown_does_not_increate_neff(self):
        """unknown 证据不进 a/b/n_eff（feedback_model 处理）。"""
        events = [
            _event("s1", _ts(0), "/search", doc_ids=["doc:a"], count=1, query_norm="q1"),
        ]
        ev = bf.infer_feedback(events, self.cfg)
        table = bf.update_posteriors(ev, {}, self.cfg)
        fb = table["doc:a"]
        # unknown 只增 n_unknown，a/b 保持 seed（1.0）
        self.assertEqual(fb.n_unknown, 1.0)
        self.assertAlmostEqual(fb.a, 1.0, places=2)  # seed 衰减后≈1（Δt 小）
        self.assertAlmostEqual(fb.b, 1.0, places=2)


class TestGapDetection(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = _cfg(self.tmpdir)

    def test_strong_sig_zero_hits_hard_gap(self):
        """强签名（kind 白名单 + weight≥2.5）零命中跨≥3会话 → hard_gap。"""
        sig_entities = json.dumps([{"kind": "kernel", "text": "DispatchFFN", "weight": 3.5}])
        events = [
            _event("s1", _ts(0), "/signature-search", count=0, sig_hash="h1",
                   sig_entities=sig_entities),
            _event("s2", _ts(100), "/signature-search", count=0, sig_hash="h1",
                   sig_entities=sig_entities),
            _event("s3", _ts(200), "/signature-search", count=0, sig_hash="h1",
                   sig_entities=sig_entities),
        ]
        gaps = bf.detect_gaps(events, self.cfg)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["gap_type"], "hard_gap")

    def test_short_errcode_weight_low_not_hard_gap(self):
        """短码（weight<2.5 如 drvRetCode=6 weight=1.0）不判 hard_gap → quality_gap。"""
        sig_entities = json.dumps([{"kind": "errcode", "text": "6", "weight": 1.0}])
        events = [
            _event(f"s{i}", _ts(i * 100), "/signature-search", count=0, sig_hash="h1",
                   sig_entities=sig_entities) for i in range(4)
        ]
        gaps = bf.detect_gaps(events, self.cfg)
        self.assertTrue(gaps)
        self.assertEqual(gaps[0]["gap_type"], "quality_gap")

    def test_weak_sig_not_hard_gap(self):
        """弱签名（无 kernel/op/errcode）零命中 → quality_gap 非 hard_gap。"""
        sig_entities = json.dumps([{"kind": "phrase", "text": "timeout"}])
        events = [
            _event(f"s{i}", _ts(i * 100), "/signature-search", count=0, sig_hash="h1",
                   sig_entities=sig_entities) for i in range(4)
        ]
        gaps = bf.detect_gaps(events, self.cfg)
        self.assertTrue(gaps)
        self.assertEqual(gaps[0]["gap_type"], "quality_gap")

    def test_hits_no_resolved_soft_gap(self):
        """命中但全无 resolved 结论 → soft_gap（mock _count_resolved_docs 返回 0）。"""
        sig_entities = json.dumps([{"kind": "errcode", "text": "561000", "weight": 3.0}])
        events = [
            _event(f"s{i}", _ts(i * 100), "/signature-search", doc_ids=["doc:x"],
                   count=1, sig_hash="h1", sig_entities=sig_entities) for i in range(4)
        ]
        with mock.patch("build_feedback._count_resolved_docs", return_value=0):
            gaps = bf.detect_gaps(events, self.cfg)
        self.assertTrue(gaps)
        self.assertEqual(gaps[0]["gap_type"], "soft_gap")

    def test_hits_with_resolved_no_gap(self):
        """命中且有 resolved → 不算缺口（有结论不缺知识）。"""
        sig_entities = json.dumps([{"kind": "errcode", "text": "561000", "weight": 3.0}])
        events = [
            _event(f"s{i}", _ts(i * 100), "/signature-search", doc_ids=["doc:x"],
                   count=1, sig_hash="h1", sig_entities=sig_entities) for i in range(4)
        ]
        with mock.patch("build_feedback._count_resolved_docs", return_value=1):
            gaps = bf.detect_gaps(events, self.cfg)
        self.assertEqual(len(gaps), 0)

    def test_less_than_3_sessions_no_gap(self):
        """<3 会话不记缺口。"""
        sig_entities = json.dumps([{"kind": "kernel", "text": "foo"}])
        events = [
            _event("s1", _ts(0), "/signature-search", count=0, sig_hash="h1",
                   sig_entities=sig_entities),
            _event("s2", _ts(100), "/signature-search", count=0, sig_hash="h1",
                   sig_entities=sig_entities),
        ]
        gaps = bf.detect_gaps(events, self.cfg)
        self.assertEqual(len(gaps), 0)


class TestPosteriorUpdate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = _cfg(self.tmpdir)

    def test_hit_increases_a(self):
        ev = [{"doc_id": "doc:a", "state": "hit", "weight": 0.3, "ts": _ts(0)}]
        table = bf.update_posteriors(ev, {}, self.cfg)
        self.assertGreater(table["doc:a"].a, 1.0)  # seed + 0.3

    def test_miss_increases_b(self):
        ev = [{"doc_id": "doc:a", "state": "miss", "weight": 0.5, "ts": _ts(0)}]
        table = bf.update_posteriors(ev, {}, self.cfg)
        self.assertGreater(table["doc:a"].b, 1.0)

    def test_existing_table_updated(self):
        """已有后验表追加新证据。"""
        existing = {"doc:a": DocFeedback(a=2.0, b=1.0, n_unknown=0.0, last_update_ts=_ts(0))}
        ev = [{"doc_id": "doc:a", "state": "hit", "weight": 0.5, "ts": _ts(1)}]
        table = bf.update_posteriors(ev, existing, self.cfg)
        self.assertGreater(table["doc:a"].a, 2.0)  # 衰减后 + 0.5

    def test_save_and_reload(self):
        ev = [{"doc_id": "doc:a", "state": "hit", "weight": 0.5, "ts": _ts(0)}]
        table = bf.update_posteriors(ev, {}, self.cfg)
        save_path = Path(self.tmpdir) / "fb.json"
        bf.save_feedback_table(save_path, table)
        loaded = load_feedback_table(save_path)
        self.assertIn("doc:a", loaded)
        self.assertGreater(loaded["doc:a"].a, 1.0)


if __name__ == "__main__":
    unittest.main()
