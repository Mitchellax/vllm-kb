"""OCR 双模型审评数据层测试（ocr_reviews 表 / 锁定 / 孤儿 / ocr.json review 字段）。"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_kb.config import AppConfig
from vllm_kb.ocr import OcrArtifact
from vllm_kb.review import (apply_locked_ocr, delete_ocr_review, get_ocr_review,
                            list_auto_consistent, list_ocr_review_orphans,
                            list_ocr_reviews, ocr_json_path_for_sha, seed_low_confidence_ocr,
                            set_ocr_json_review, upsert_ocr_review)


def make_cfg(root: Path) -> AppConfig:
    os.environ["VLLM_KB_DATA_ROOT"] = str(root / "data")
    return AppConfig.model_validate({
        "project": {"name": "t", "data_root": str(root / "data")},
        "embedding": {"provider": "echo", "dimensions": 8},
        "sources": [],
    })


class TestOcrReviewData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.addCleanup(os.environ.pop, "VLLM_KB_DATA_ROOT", None)
        (self.root / "data" / "parsed" / "images").mkdir(parents=True)
        (self.root / "data" / "assets" / "images").mkdir(parents=True)

    def _write_ocr_json(self, sha: str, text: str = "引擎输出"):
        p = self.root / "data" / "parsed" / "images" / f"img_{sha[:16]}.ocr.json"
        p.write_text(json.dumps({"sha256": sha, "text": text, "confidence": 0.4},
                                ensure_ascii=False), encoding="utf-8")
        return p

    def _write_asset(self, sha: str):
        (self.root / "data" / "assets" / "images" / f"img_{sha[:16]}.png").write_bytes(b"x")

    def test_crud_and_decision_validation(self):
        sha = "a" * 64
        upsert_ocr_review(self.cfg, sha, "最终文本", "pp", vl_text="VL", pp_text="PP",
                          agree=0.6, reviewer="tester")
        rev = get_ocr_review(self.cfg, sha)
        self.assertEqual(rev["status"], "approved")
        self.assertEqual(rev["final_text"], "最终文本")
        self.assertEqual(rev["decision"], "pp")
        # upsert 覆盖（编辑后再保存）
        upsert_ocr_review(self.cfg, sha, "改后文本", "edited", reviewer="t2")
        self.assertEqual(get_ocr_review(self.cfg, sha)["final_text"], "改后文本")
        self.assertEqual(len(list_ocr_reviews(self.cfg)), 1)
        with self.assertRaises(ValueError):
            upsert_ocr_review(self.cfg, "b" * 64, "x", "bad")

    def test_apply_locked_ocr(self):
        sha = "c" * 64
        art = OcrArtifact(sha256=sha, text="旧", confidence=0.3, anomaly="no_confidence",
                          min_confidence=0.6)
        self.assertFalse(apply_locked_ocr(art, sha, self.cfg))   # 无锁
        self.assertEqual(art.review_reason, "anomaly")           # anomaly 优先于 no_confidence
        upsert_ocr_review(self.cfg, sha, "人工核对文本", "edited")
        self.assertTrue(apply_locked_ocr(art, sha, self.cfg))
        self.assertEqual(art.text, "人工核对文本")
        self.assertEqual(art.confidence_source, "manual")
        self.assertEqual(art.anomaly, "")
        self.assertTrue(art.high_confidence)                     # property 实时判定成立
        self.assertEqual(art.review_reason, "")

    def test_orphan_semantics(self):
        sha1, sha2 = "d" * 64, "e" * 64
        upsert_ocr_review(self.cfg, sha1, "t1", "vl")
        upsert_ocr_review(self.cfg, sha2, "t2", "pp")
        self._write_asset(sha1)                                   # sha1 资产在场
        self.assertEqual([r["sha256"] for r in list_ocr_review_orphans(self.cfg)],
                         [sha2])                                  # sha2 文件没了 → 孤儿
        # 图片替换场景：sha2 的文件重新出现 → 不再孤儿（数据不撤回）
        self._write_asset(sha2)
        self.assertEqual(list_ocr_review_orphans(self.cfg), [])

    def test_delete_keeps_files(self):
        sha = "f" * 64
        ocr_p = self._write_ocr_json(sha)
        asset_p = self.root / "data" / "assets" / "images" / f"img_{sha[:16]}.png"
        asset_p.write_bytes(b"x")
        upsert_ocr_review(self.cfg, sha, "t", "vl")
        self.assertTrue(set_ocr_json_review(self.cfg, sha, {"decision": "vl"}))
        self.assertEqual(json.loads(ocr_p.read_text(encoding="utf-8"))["review"]["decision"], "vl")
        # 删除：db 行消失 + ocr.json review 字段清（API 层同时调两步）+ **两个文件都还在**
        self.assertTrue(delete_ocr_review(self.cfg, sha))
        self.assertTrue(set_ocr_json_review(self.cfg, sha, None))
        self.assertIsNone(get_ocr_review(self.cfg, sha))
        self.assertEqual(json.loads(ocr_p.read_text(encoding="utf-8")).get("review"), None)
        self.assertTrue(ocr_p.is_file() and asset_p.is_file())
        self.assertFalse(delete_ocr_review(self.cfg, sha))        # 幂等

    def test_ocr_json_path_and_missing(self):
        sha = "0" * 64
        self.assertIsNone(ocr_json_path_for_sha(self.cfg, sha))
        self.assertFalse(set_ocr_json_review(self.cfg, sha, {"decision": "vl"}))
        self._write_ocr_json(sha)
        self.assertEqual(ocr_json_path_for_sha(self.cfg, sha),
                         self.root / "data" / "parsed" / "images" / f"img_{sha[:16]}.ocr.json")
        self.assertTrue(set_ocr_json_review(self.cfg, sha, None))  # 清不存在的字段也 OK
        self.assertIsNone(json.loads(ocr_json_path_for_sha(self.cfg, sha).read_text(
            encoding="utf-8")).get("review"))


from unittest import mock  # noqa: E402

from vllm_kb.ocr import enrich_ocr_artifact  # noqa: E402
from vllm_kb.review import ReviewStore  # noqa: E402


class TestEnrichCrossConfirm(unittest.TestCase):
    """enrich_ocr_artifact：第二意见复核 + 低置信翻案矩阵（PP mock）。"""

    def _art(self, conf=None, anomaly="", text="VL文本"):
        return OcrArtifact(sha256="a" * 64, text=text, confidence=conf,
                           confidence_source="model" if conf is not None else "none",
                           anomaly=anomaly, min_confidence=0.6)

    def test_low_conf_agree_overrides(self):
        art = self._art(conf=0.4)
        with mock.patch("vllm_kb.ocr.ppocr_second_opinion", return_value="VL文本"):
            changed = enrich_ocr_artifact(art, "x.png", min_confidence=0.6)
        self.assertTrue(changed)
        self.assertEqual(art.confidence_source, "cross_confirmed")
        self.assertTrue(art.high_confidence)          # 翻案 → 正文注入
        self.assertEqual(art.review_reason, "")       # 不进待办
        self.assertEqual(art.second_opinion["agree"], 1.0)

    def test_low_conf_diverge_stays_pending(self):
        art = self._art(conf=0.4, text="VL文本")
        with mock.patch("vllm_kb.ocr.ppocr_second_opinion", return_value="完全不同的字"):
            enrich_ocr_artifact(art, "x.png", min_confidence=0.6)
        self.assertEqual(art.confidence_source, "model")   # 不翻案
        self.assertFalse(art.high_confidence)
        self.assertEqual(art.review_reason, "low")
        self.assertLess(art.second_opinion["agree"], 0.85)  # 分歧已记录（UI diff 用）

    def test_structural_anomaly_never_overrides(self):
        art = self._art(conf=0.9, anomaly="truncated", text="截断")
        with mock.patch("vllm_kb.ocr.ppocr_second_opinion", return_value="截断"):
            enrich_ocr_artifact(art, "x.png", min_confidence=0.6)
        self.assertEqual(art.anomaly, "truncated")          # 结构异常保留 → 人工
        self.assertFalse(art.high_confidence)

    def test_high_conf_unaffected_by_pp(self):
        art = self._art(conf=0.95, text="正文")
        with mock.patch("vllm_kb.ocr.ppocr_second_opinion", return_value="全然不同布局文本"):
            enrich_ocr_artifact(art, "x.png", min_confidence=0.6)
        self.assertEqual(art.confidence, 0.95)              # 高置信不吃 PP 判定
        self.assertEqual(art.confidence_source, "model")
        self.assertTrue(art.high_confidence)                # 版面分歧是预期误报

    def test_idempotent_and_missing_pp(self):
        art = self._art(conf=0.4)
        art.second_opinion = {"engine": "PP-OCRv5(local)", "text": "VL文本", "agree": 0.9}
        calls = []
        with mock.patch("vllm_kb.ocr.ppocr_second_opinion",
                        side_effect=lambda p: calls.append(p) or "VL文本"):
            changed = enrich_ocr_artifact(art, "x.png", min_confidence=0.6)
        self.assertEqual(calls, [])                          # 已有复核 → 不重跑
        self.assertTrue(changed)                             # 但翻案仍生效
        art2 = self._art(conf=0.4)
        with mock.patch("vllm_kb.ocr.ppocr_second_opinion", return_value=None):
            changed2 = enrich_ocr_artifact(art2, "x.png", min_confidence=0.6)
        self.assertFalse(changed2)                           # PP 不可用 → 无变更（不写盘）
        self.assertEqual(art2.review_reason, "low")


class TestSeedThreeTiers(unittest.TestCase):
    """seed_low_confidence_ocr 三档分流：复核一致不入队 / diverged 带双文本 / 锁定跳过。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.root / "data")
        self.addCleanup(os.environ.pop, "VLLM_KB_DATA_ROOT", None)
        self.cfg = AppConfig.model_validate({
            "project": {"name": "t", "data_root": str(self.root / "data")},
            "embedding": {"provider": "echo", "dimensions": 8},
            "sources": [],
        })
        (self.root / "data" / "parsed" / "images").mkdir(parents=True)
        kb = self.root / "data" / "kb.sqlite3"
        conn = sqlite3.connect(str(kb))
        conn.execute("CREATE TABLE docs (source_id TEXT, title TEXT, url TEXT, extra TEXT)")
        conn.commit()
        self.conn = conn
        self.store = ReviewStore(self.root / "data" / "review.sqlite3")
        self.addCleanup(self.conn.close)

    def _add_doc(self, evidences):
        self.conn.execute("INSERT INTO docs VALUES(?,?,?,?)",
                          ("s1", "t", "u", json.dumps({"evidence": evidences},
                                                      ensure_ascii=False)))
        self.conn.commit()

    @staticmethod
    def _ev(sha, conf=0.4, anomaly="", so=None, review=None, included=False):
        ocr = {"confidence": conf, "confidence_source": "model", "anomaly": anomaly,
               "text_included": included, "signatures": []}
        if so is not None:
            ocr["second_opinion"] = so
        if review is not None:
            ocr["review"] = review
        return {"kind": "image", "sha256": sha, "asset_id": f"as_{sha[:6]}", "ocr": ocr}

    def _seed(self):
        seed_low_confidence_ocr(self.cfg, self.store)
        return [dict(i) for i in self.store.list_items(category="low_confidence_ocr")]

    def test_auto_consistent_not_queued(self):
        self._add_doc([self._ev("a" * 64, conf=0.4,
                                so={"engine": "PP-OCRv5(local)", "agree": 0.92})])
        self.assertEqual(self._seed(), [])   # 复核一致：不进待办

    def test_diverged_queued_with_both_texts(self):
        sha = "b" * 64
        (self.root / "data" / "parsed" / "images" / f"img_{sha[:16]}.ocr.json").write_text(
            json.dumps({"sha256": sha, "text": "VL全文", "confidence": 0.4,
                        "second_opinion": {"engine": "PP-OCRv5(local)",
                                           "text": "PP全文", "agree": 0.5}},
                       ensure_ascii=False), encoding="utf-8")
        self._add_doc([self._ev(sha, conf=0.4,
                                so={"engine": "PP-OCRv5(local)", "agree": 0.5})])
        items = self._seed()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["payload"]["reason"], "diverged")
        self.assertEqual(items[0]["payload"]["vl_text"], "VL全文")
        self.assertEqual(items[0]["payload"]["pp_text"], "PP全文")
        self.assertEqual(items[0]["payload"]["agree"], 0.5)

    def test_structural_anomaly_queued_even_if_agree(self):
        self._add_doc([self._ev("c" * 64, conf=0.9, anomaly="truncated",
                                so={"engine": "PP", "agree": 0.95})])
        items = self._seed()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["payload"]["reason"], "anomaly")  # 不翻案

    def test_locked_skipped(self):
        self._add_doc([self._ev("d" * 64, conf=0.4,
                                review={"decision": "pp"})])
        self.assertEqual(self._seed(), [])   # 已人工锁定：不再打扰

    def test_unreviewed_low_still_queued(self):
        self._add_doc([self._ev("e" * 64, conf=0.4)])
        items = self._seed()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["payload"]["reason"], "low")

    def test_auto_consistent_listing(self):
        sha = "f" * 64
        (self.root / "data" / "parsed" / "images" / f"img_{sha[:16]}.ocr.json").write_text(
            json.dumps({"sha256": sha, "text": "VL", "confidence": 0.85,
                        "confidence_source": "cross_confirmed",
                        "second_opinion": {"engine": "PP", "text": "PP", "agree": 0.9}},
                       ensure_ascii=False), encoding="utf-8")
        rows = list_auto_consistent(self.cfg)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sha256"], sha)
        self.assertEqual(rows[0]["pp_text"], "PP")


if __name__ == "__main__":
    unittest.main()
