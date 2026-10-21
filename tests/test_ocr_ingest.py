"""P1 图片 OCR 入库链路测试：正文注入、evidence 摘要、引擎指纹/阈值、出口白名单、审核 seed。

覆盖：
- 高置信 OCR 文本注入 Markdown 正文（进 FTS + 向量）；低置信 / 自报异常不注入（只留签名线索）；
- `extra.evidence[].ocr` 摘要（无路径）与 ocr.json 产物；
- 引擎指纹幂等（换模型重算 / 调阈值不重算）；
- `api._sanitize_extra` 出口白名单放行 ocr 摘要；
- `review.seed_low_confidence_ocr` 按图片 sha256 聚合生成审核项。
"""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vllm_kb.config import AppConfig, SourceCfg
from vllm_kb.ocr import OcrApiError, OcrResult
from vllm_kb.sources import MarkdownSource


def make_png(path: Path, text: str = "test image") -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 60), "white")
    ImageDraw.Draw(img).text((10, 20), text, fill="black")
    img.save(path)


OCR_TEXT = "error code 107020, dispatch_ffn_combine failed, GLM-5.1"


class _MarkdownOcrBase(unittest.TestCase):
    """合成 AppConfig（数据根重定向到临时目录）+ 一个含截图的 Markdown。"""

    MIN_CONF = 0.6

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_root = self.root / "data"
        md_dir = self.data_root / "imports" / "md"
        md_dir.mkdir(parents=True)
        make_png(md_dir / "shot.png")
        (md_dir / "doc.md").write_text(
            "# 报错记录\n\n截图: ![err](shot.png)\n", encoding="utf-8")
        self._old_root = os.environ.get("VLLM_KB_DATA_ROOT")
        os.environ["VLLM_KB_DATA_ROOT"] = str(self.data_root)
        self.app_cfg = self._app_config(self.MIN_CONF)

    def tearDown(self):
        if self._old_root is None:
            os.environ.pop("VLLM_KB_DATA_ROOT", None)
        else:
            os.environ["VLLM_KB_DATA_ROOT"] = self._old_root
        self.tmp.cleanup()

    def _app_config(self, min_conf) -> AppConfig:
        return AppConfig.model_validate({
            "embedding": {"provider": "echo", "dimensions": 8},
            "storage": {"vector_backend": "python"},
            "sources": [
                {"id": "wiki", "type": "markdown", "path": "data/imports/md",
                 "title_pattern": r"^#\s+(.+)", "enabled": True},
                {"id": "images", "type": "image", "enabled": True,
                 "ocr_provider": "paddle", "ocr_min_confidence": min_conf},
            ],
        })

    def _run(self, result, app_cfg=None, side_effect=None):
        cfg = SourceCfg(id="wiki", type="markdown", path="data/imports/md",
                        title_pattern=r"^#\s+(.+)", enabled=True)
        src = MarkdownSource(cfg, project_root=self.root,
                             app_cfg=app_cfg or self.app_cfg)
        kwargs = {"side_effect": side_effect} if side_effect is not None else {"return_value": result}
        with mock.patch("vllm_kb.ocr.ocr_image_detail", **kwargs) as m:
            docs = src.canonicalize()
        return docs[0], m

    def _ocr_json(self) -> dict:
        p = self.data_root / "parsed" / "images" / "shot.ocr.json"
        return json.loads(p.read_text(encoding="utf-8"))


class TestMarkdownOcrInjection(_MarkdownOcrBase):
    def test_high_confidence_text_injected(self):
        doc, m = self._run(OcrResult(text=OCR_TEXT, confidence=0.92,
                                     confidence_source="model", provider="api"))
        self.assertEqual(m.call_count, 1)
        self.assertIn("[图片:err]", doc.body)
        self.assertIn(OCR_TEXT, doc.body)               # 进正文 → FTS + 向量
        self.assertIn("OCR 置信度 0.92", doc.body)
        # 安全约束：正文仍不含任何服务器路径
        self.assertNotIn("assets/images/", doc.body)
        self.assertNotIn("shot.png", doc.body)
        ev = doc.extra["evidence"][0]
        self.assertEqual(ev["kind"], "local")
        self.assertTrue(ev["ocr"]["text_included"])
        self.assertEqual(ev["ocr"]["confidence"], 0.92)
        self.assertEqual(ev["ocr"]["confidence_source"], "model")
        self.assertNotIn("path", ev)
        kinds = {s["kind"] for s in ev["ocr"]["signatures"]}
        self.assertTrue(kinds & {"errcode", "op", "model"}, kinds)

    def test_low_confidence_not_injected(self):
        doc, _ = self._run(OcrResult(text=OCR_TEXT, confidence=0.3,
                                     confidence_source="model", provider="api"))
        self.assertIn("[图片:err]", doc.body)
        self.assertNotIn(OCR_TEXT, doc.body)            # 低置信不进正文
        ev = doc.extra["evidence"][0]
        self.assertFalse(ev["ocr"]["text_included"])
        self.assertEqual(ev["ocr"]["confidence"], 0.3)
        self.assertTrue(ev["ocr"]["signatures"])        # 签名线索仍保留

    def test_anomaly_not_injected(self):
        doc, _ = self._run(OcrResult(text=OCR_TEXT, confidence=None,
                                     confidence_source="model", anomaly="missing",
                                     provider="api"))
        self.assertNotIn(OCR_TEXT, doc.body)
        ev = doc.extra["evidence"][0]
        self.assertFalse(ev["ocr"]["text_included"])
        self.assertEqual(ev["ocr"]["anomaly"], "missing")
        self.assertIsNone(ev["ocr"]["confidence"])

    def test_ocr_failure_does_not_block_import(self):
        doc, _ = self._run(None, side_effect=OcrApiError("service down"))
        self.assertIn("[图片:err]", doc.body)
        self.assertNotIn(OCR_TEXT, doc.body)
        ev = doc.extra["evidence"][0]
        self.assertIsNone(ev["ocr"])                    # 无 OCR 结果：保留字段位但为空
        self.assertFalse((self.data_root / "parsed" / "images" / "shot.ocr.json").exists())

    def test_threshold_from_image_source_config(self):
        """阈值取自 image source 的 ocr_min_confidence（与 ImageSource 同源）。"""
        strict = self._app_config(0.95)
        doc, _ = self._run(OcrResult(text=OCR_TEXT, confidence=0.9,
                                     confidence_source="model", provider="api"),
                           app_cfg=strict)
        self.assertNotIn(OCR_TEXT, doc.body)            # 0.9 < 0.95
        loose = self._app_config(0.5)
        doc2, _ = self._run(OcrResult(text=OCR_TEXT, confidence=0.9,
                                      confidence_source="model", provider="api"),
                            app_cfg=loose)
        self.assertIn(OCR_TEXT, doc2.body)              # 0.9 ≥ 0.5

    def test_ocr_json_artifact_written_with_fingerprint(self):
        self._run(OcrResult(text=OCR_TEXT, confidence=0.92,
                            confidence_source="model", provider="api"))
        meta = self._ocr_json()
        self.assertTrue(meta["engine_fingerprint"])
        self.assertEqual(meta["provider"], "paddle")
        self.assertEqual(meta["confidence"], 0.92)
        self.assertTrue(meta["text_included"])
        self.assertEqual(meta["image"], "assets/images/shot.png")

    def test_no_ocr_config_keeps_placeholder_only(self):
        """无 image source（未配置 OCR）→ 只占位，不报错。"""
        cfg_only_md = AppConfig.model_validate({
            "embedding": {"provider": "echo", "dimensions": 8},
            "storage": {"vector_backend": "python"},
            "sources": [{"id": "wiki", "type": "markdown", "path": "data/imports/md",
                         "title_pattern": r"^#\s+(.+)", "enabled": True}],
        })
        doc, m = self._run(OcrResult(text=OCR_TEXT, confidence=0.99),
                           app_cfg=cfg_only_md)
        self.assertEqual(m.call_count, 0)
        self.assertIn("[图片:err]", doc.body)
        self.assertNotIn(OCR_TEXT, doc.body)


class TestSanitizeExtraOcr(unittest.TestCase):
    """出口白名单：ocr 摘要放行（纯知识字段），路径类字段仍被剥离。"""

    def test_ocr_summary_passes_through(self):
        from vllm_kb.api import _sanitize_extra

        out = _sanitize_extra({
            "evidence": [{
                "kind": "local", "asset_id": "abc", "sha256": "f" * 64,
                "path": "/secret/server/path.png",
                "ocr": {"confidence": 0.9, "confidence_source": "model", "anomaly": "",
                        "text_included": True,
                        "signatures": [{"text": "107020", "kind": "errcode"}],
                        "raw_path": "/secret/ocr.json"},
            }],
        })
        ev = out["evidence"][0]
        self.assertNotIn("path", ev)
        self.assertEqual(ev["ocr"]["confidence"], 0.9)
        self.assertTrue(ev["ocr"]["text_included"])
        self.assertEqual(ev["ocr"]["signatures"], [{"text": "107020", "kind": "errcode"}])
        self.assertNotIn("raw_path", ev["ocr"])          # ocr 子字段同样白名单

    def test_non_dict_ocr_ignored(self):
        from vllm_kb.api import _sanitize_extra

        out = _sanitize_extra({"evidence": [{"kind": "local", "ocr": "nonsense"}]})
        self.assertNotIn("ocr", out["evidence"][0])


class TestOcrMinConfidenceConfig(unittest.TestCase):
    """ocr_min_confidence 配置校验：非法值在加载时报错，而非静默回默认。"""

    def _cfg(self, value):
        cfg = AppConfig.model_validate({
            "embedding": {"provider": "echo", "dimensions": 8},
            "storage": {"vector_backend": "python"},
            "sources": [{"id": "images", "type": "image", "enabled": True,
                         "ocr_min_confidence": value}],
        })
        cfg.validate_runtime(require_keys=False)
        return cfg

    def test_valid_values(self):
        self.assertEqual(self._cfg(0.75).sources[0].ocr_min_confidence, 0.75)
        self._cfg("")     # 空串 = 用默认值
        self._cfg(0)
        self._cfg(1)

    def test_invalid_values_raise(self):
        for bad in ("abc", -0.1, 1.5):
            with self.assertRaises(ValueError, msg=f"{bad!r} 应报错"):
                self._cfg(bad)


class TestSeedLowConfidenceOcr(unittest.TestCase):
    """审核 seed：未进正文的图片 OCR → low_confidence_ocr 审核项（按图片 sha 聚合）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "kb.sqlite3"
        conn = sqlite3.connect(str(self.db))
        conn.execute("CREATE TABLE docs (source_id TEXT PRIMARY KEY, title TEXT, "
                     "url TEXT, extra TEXT)")
        conn.commit()
        conn.close()
        self.cfg = AppConfig.model_validate({
            "embedding": {"provider": "echo", "dimensions": 8},
            "storage": {"vector_backend": "python", "sqlite_path": str(self.db)},
        })

    def tearDown(self):
        self.tmp.cleanup()

    def _add_doc(self, source_id, ocr, sha="a" * 64, title="t"):
        conn = sqlite3.connect(str(self.db))
        conn.execute("INSERT INTO docs VALUES (?,?,?,?)", (
            source_id, title, f"http://x/{source_id}",
            json.dumps({"evidence": [{"kind": "local", "asset_id": sha[:16],
                                      "sha256": sha, "ocr": ocr}]}, ensure_ascii=False)))
        conn.commit()
        conn.close()

    def test_low_confidence_and_anomaly_seeded(self):
        from vllm_kb.review import ReviewStore, seed_low_confidence_ocr

        self._add_doc("wiki:1", {"confidence": 0.3, "confidence_source": "model",
                                 "anomaly": "", "text_included": False,
                                 "signatures": [{"text": "107020", "kind": "errcode"}]})
        self._add_doc("wiki:2", {"confidence": None, "confidence_source": "model",
                                 "anomaly": "missing", "text_included": False},
                      sha="b" * 64)
        self._add_doc("wiki:3", {"confidence": 0.95, "confidence_source": "model",
                                 "anomaly": "", "text_included": True}, sha="c" * 64)
        store = ReviewStore(self.root / "review.sqlite3")
        added = seed_low_confidence_ocr(self.cfg, store)
        self.assertEqual(added, 2)                       # 高置信那条不生成
        items = store.list_items(category="low_confidence_ocr")
        by_ref = {i["item_ref"]: i for i in items}
        self.assertIn("ocr:" + "a" * 16, by_ref)
        self.assertEqual(by_ref["ocr:" + "a" * 16]["payload"]["reason"], "low")
        self.assertEqual(by_ref["ocr:" + "b" * 16]["payload"]["reason"], "anomaly")
        # 幂等：重复 seed 不再新增
        self.assertEqual(seed_low_confidence_ocr(self.cfg, store), 0)

    def test_same_image_in_two_docs_aggregates(self):
        from vllm_kb.review import ReviewStore, seed_low_confidence_ocr

        self._add_doc("wiki:1", {"confidence": 0.2, "text_included": False}, sha="d" * 64)
        self._add_doc("wiki:2", {"confidence": 0.2, "text_included": False}, sha="d" * 64)
        store = ReviewStore(self.root / "review.sqlite3")
        self.assertEqual(seed_low_confidence_ocr(self.cfg, store), 1)  # 同 sha 聚合一条


class TestOcrPathConsistency(unittest.TestCase):
    """未设 VLLM_KB_DATA_ROOT（AppConfig.resolve 走 PROJECT_ROOT 分支）时，资产化→OCR
    的路径必须与**写盘位置**一致。

    业务实测（2026-10，html 导入大量刷屏）：旧代码 `_ocr_artifact_for(self.resolve(rel))`
    —— rel 形如 assets/images/x.png，未设 DATA_ROOT 时按 PROJECT_ROOT 拼接丢掉 data/
    一层 → OCR 读 /proj/assets/…（写入在 /proj/data/assets/…）→ 每张图报"图片不存在"，
    且 asset_registry 登记同样坏路径。设了 DATA_ROOT 的部署会剥 data/ 前缀恰好自洽，
    本地未配 OCR 从未走到该行——既有测试全设 DATA_ROOT 故未暴露。

    隔离手段：patch `config.PROJECT_ROOT` 到临时目录（未设 env 时 resolve 以它为根），
    逐字复现业务环境的解析分支。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        md_dir = self.root / "data" / "imports" / "md"
        md_dir.mkdir(parents=True)
        make_png(md_dir / "shot.png")
        (md_dir / "doc.md").write_text(
            "# 报错记录\n\n截图: ![err](shot.png)\n", encoding="utf-8")
        self._old_root = os.environ.pop("VLLM_KB_DATA_ROOT", None)   # 关键：未设
        self.app_cfg = AppConfig.model_validate({
            "embedding": {"provider": "echo", "dimensions": 8},
            "storage": {"vector_backend": "python"},
            "sources": [
                {"id": "wiki", "type": "markdown", "path": "data/imports/md",
                 "title_pattern": r"^#\s+(.+)", "enabled": True},
                {"id": "images", "type": "image", "enabled": True,
                 "ocr_provider": "api", "ocr_api_base": "http://127.0.0.1:9"},
            ],
        })

    def tearDown(self):
        if self._old_root is not None:
            os.environ["VLLM_KB_DATA_ROOT"] = self._old_root
        self.tmp.cleanup()

    def test_ocr_receives_written_location(self):
        # patch 覆盖**构造+canonicalize 全程**：import_dir 在 __init__ 即解析
        with mock.patch("vllm_kb.config.PROJECT_ROOT", self.root):
            cfg = SourceCfg(id="wiki", type="markdown", path="data/imports/md",
                            title_pattern=r"^#\s+(.+)", enabled=True)
            src = MarkdownSource(cfg, project_root=self.root, app_cfg=self.app_cfg)
            with mock.patch("vllm_kb.ocr.ocr_image_detail",
                            return_value=OcrResult(text=OCR_TEXT, confidence=0.9,
                                                   confidence_source="model", provider="api")) as m:
                docs = src.canonicalize()
        self.assertEqual(m.call_count, 1)
        asset_path = m.call_args[0][0]                     # OCR 收到的路径
        self.assertTrue(asset_path.is_file(), asset_path)  # 必须是真实写盘位置
        self.assertIn("data/assets/images",
                      str(asset_path).replace("\\", "/"))  # 且带 data/ 一层
        self.assertIn(OCR_TEXT, docs[0].body)              # OCR 拿到真文件 → 文本注入


class TestPaddleNdarrayInput(unittest.TestCase):
    """paddle 引擎收 ndarray（内存解码），不收文件路径——绕开 cv2 非 ASCII 路径限制。"""

    def test_paddle_feeds_ndarray_not_path(self):
        import io
        import sys
        import types

        import numpy as np
        from PIL import Image

        from vllm_kb.ocr import ocr_image_detail

        buf = io.BytesIO()
        Image.new("RGB", (10, 10), "white").save(buf, format="PNG")
        with tempfile.TemporaryDirectory() as td:
            cdir = Path(td) / "中文目录"
            cdir.mkdir()
            img = cdir / "图.png"                 # 非 ASCII 路径
            img.write_bytes(buf.getvalue())
            calls = []

            class _Stub:
                def ocr(self, arr, cls=True):
                    calls.append(arr)
                    return [[([0, 0], ("hello", 0.9))]]

            fake_cv2 = types.ModuleType("cv2")
            fake_cv2.imdecode = lambda b, flag: np.zeros((4, 4, 3), dtype=np.uint8)
            fake_cv2.IMREAD_COLOR = 1
            with mock.patch.dict(sys.modules, {"cv2": fake_cv2}), \
                 mock.patch("vllm_kb.ocr._get_paddle", return_value=_Stub()):
                res = ocr_image_detail(img, provider="paddle")
        self.assertEqual(res.text, "hello")
        self.assertEqual(res.provider, "paddle")
        self.assertEqual(len(calls), 1)
        self.assertTrue(hasattr(calls[0], "shape"))   # ndarray，非文件路径


if __name__ == "__main__":
    unittest.main()
