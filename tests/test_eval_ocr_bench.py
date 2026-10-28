"""OCR 基准集评估工具测试（scripts/eval_ocr_bench.py 的可 import 核心）。"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_ocr_bench import (cmd_gen_transforms, cmd_pick_from_assets,
                                    norm_text, score_pair, summarize)


class TestScorePair(unittest.TestCase):
    def test_identical_and_whitespace_insensitive(self):
        self.assertEqual(score_pair("复位芯片", "复位芯片"), 1.0)
        self.assertEqual(score_pair("复 位\n芯片", "复位芯片"), 1.0)   # 空白/换行不扣分

    def test_partial_and_empty(self):
        self.assertLess(score_pair("复位芯片", "复位心片"), 1.0)
        self.assertGreater(score_pair("复位芯片", "复位芯片。"), 0.85)  # 单字符差异（4/5 字符）
        self.assertEqual(score_pair("", ""), 1.0)
        self.assertEqual(score_pair("复位芯片", ""), 0.0)


class TestSummarize(unittest.TestCase):
    def test_per_scene_and_overall(self):
        rows = [
            {"image": "normal_01", "scene": "normal", "score": 1.0, "pass": True,
             "confidence": 0.9, "anomaly": ""},
            {"image": "tilt_01", "scene": "tilt", "score": 0.5, "pass": False,
             "confidence": None, "anomaly": "truncated"},
            {"image": "tilt_02", "scene": "tilt", "score": 0.7, "pass": False,
             "confidence": 0.8, "anomaly": ""},
        ]
        s = summarize(rows)
        self.assertEqual(s["normal"]["n"], 1)
        self.assertEqual(s["tilt"]["n"], 2)
        self.assertEqual(s["tilt"]["pass_rate"], 0.0)
        self.assertEqual(s["tilt"]["avg_score"], 0.6)
        self.assertEqual(s["tilt"]["avg_confidence"], 0.8)      # None 不计入均值
        self.assertEqual(s["tilt"]["anomaly_n"], 1)
        self.assertEqual(s["overall"]["n"], 3)


class TestGenTransforms(unittest.TestCase):
    def test_tilt_and_persp_generated(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy 不可用")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bench = Path(tmp.name)
        (bench / "images").mkdir(parents=True)
        # 用纯 Python 画一张 40x20 白底红条 PNG 当源图
        from PIL import Image

        Image.new("RGB", (40, 20), (255, 255, 255)).save(bench / "images" / "normal_01.png")
        labels = {"version": 1, "items": [
            {"image": "normal_01.png", "scene": "normal", "expect": "x"},
            {"image": "tilt_01.png", "scene": "tilt", "expect": "x",
             "derived_from": "normal_01.png", "param": {"rotate": 5}},
            {"image": "persp_01.png", "scene": "persp", "expect": "x",
             "derived_from": "normal_01.png", "param": {}},
        ]}
        (bench / "labels.json").write_text(json.dumps(labels, ensure_ascii=False),
                                           encoding="utf-8")
        n = cmd_gen_transforms(bench)
        self.assertEqual(n, 2)
        self.assertTrue((bench / "images" / "tilt_01.png").exists())
        self.assertTrue((bench / "images" / "persp_01.png").exists())
        # 幂等：已存在不重建
        self.assertEqual(cmd_gen_transforms(bench), 0)


class TestPickFromAssets(unittest.TestCase):
    def test_buckets_and_prefilled_labels(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        assets = root / "assets"
        assets.mkdir(parents=True)
        from PIL import Image

        # 高置信 normal 源 ×2 + 低置信 blurry 源 ×2（conf 0.95/0.9/0.4/0.3）
        for name, conf in (("a", 0.95), ("b", 0.90), ("c", 0.40), ("d", 0.30)):
            Image.new("RGB", (30, 15), (255, 255, 255)).save(assets / f"{name}.png")
            (assets / f"{name}.ocr.json").write_text(json.dumps({
                "text": f"文本{name}", "confidence": conf}), encoding="utf-8")
        bench = root / "ocr_bench"
        n = cmd_pick_from_assets(assets, bench, 4)
        self.assertEqual(n, 4)
        labels = json.loads((bench / "labels.json").read_text(encoding="utf-8"))
        by_scene = {}
        for it in labels["items"]:
            by_scene.setdefault(it["scene"], []).append(it)
        self.assertEqual(len(by_scene["normal"]), 2)            # conf 0.95/0.90
        self.assertEqual(len(by_scene["blurry"]), 2)            # conf 0.40/0.30
        self.assertEqual(by_scene["normal"][0]["expect"], "文本a")  # expect 预填
        self.assertTrue((bench / "images" / "normal_01.png").exists())


if __name__ == "__main__":
    unittest.main()
