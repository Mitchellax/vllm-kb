"""Markdown 图片收集 + ImageSource OCR 链路测试。

覆盖：md 图片引用各形态（相对/子目录/绝对/base64/URL）的资产化与正文重写、
evidence 记录、ImageSource OCR 引擎决策（ask/api/paddle/none）与幂等、签名导向提取。
"""
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vllm_kb.config import SourceCfg
from vllm_kb.ocr import OcrApiError
from vllm_kb.sources import ImageSource, MarkdownSource


def make_png(path: Path, text: str = "test image") -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 60), "white")
    ImageDraw.Draw(img).text((10, 20), text, fill="black")
    img.save(path)


class TestMarkdownImageCollection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        md_dir = self.root / "data" / "imports" / "md"
        md_dir.mkdir(parents=True)
        # 相对路径图片（md 旁）+ 子目录图片
        make_png(md_dir / "local.png")
        (md_dir / "imgs").mkdir()
        make_png(md_dir / "imgs" / "sub.png")
        # base64 图片
        buf = io.BytesIO()
        from PIL import Image

        Image.new("RGB", (10, 10), "red").save(buf, format="PNG")
        self.b64 = base64.b64encode(buf.getvalue()).decode()
        # 绝对路径图片
        self.abs_png = self.root / "abs.png"
        make_png(self.abs_png)
        md = (
            "# 测试文档\n\n"
            "本地图: ![a](local.png)\n"
            "子目录图: ![b](imgs/sub.png)\n"
            "绝对图: ![c](%s)\n"
            "内嵌图: ![d](data:image/png;base64,%s)\n"
            "外链图: ![e](https://example.com/remote.png)\n"
        ) % (str(self.abs_png).replace("\\", "/"), self.b64)
        (md_dir / "doc.md").write_text(md, encoding="utf-8")
        self.cfg = SourceCfg(id="wiki", type="markdown", path="data/imports/md",
                             title_pattern=r"^#\s+(.+)", enabled=True)
        self.src = MarkdownSource(self.cfg, project_root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_images_collected_and_body_rewritten(self):
        self.src.pull()
        docs = self.src.canonicalize()
        self.assertEqual(len(docs), 1)
        body = docs[0].body
        # 本地/base64 图片引用重写为资产路径
        self.assertIn("assets/images/local.png", body)
        self.assertIn("assets/images/sub.png", body)
        self.assertIn("assets/images/abs.png", body)
        self.assertIn("assets/images/doc_img1.png", body)  # base64 解码
        # 外链保留原引用
        self.assertIn("https://example.com/remote.png", body)
        # 资产层存在图片文件
        images = self.root / "data" / "assets" / "images"
        names = {p.name for p in images.glob("*.png")}
        self.assertIn("local.png", names)
        self.assertIn("sub.png", names)
        self.assertIn("abs.png", names)
        self.assertIn("doc_img1.png", names)

    def test_evidence_records(self):
        self.src.pull()
        d = self.src.canonicalize()[0]
        ev = d.extra["evidence"]
        kinds = {e["kind"] for e in ev}
        self.assertEqual(kinds, {"local", "base64", "remote"})
        local_ev = [e for e in ev if e["kind"] == "local"]
        self.assertTrue(all(e["path"].startswith("assets/images/") for e in local_ev))
        remote_ev = [e for e in ev if e["kind"] == "remote"]
        self.assertEqual(remote_ev[0]["path"], None)  # URL 不下载

    def test_unresolved_image_kept(self):
        # 引用不存在的本地图片：保留原引用，标记 unresolved
        md_dir = self.root / "data" / "imports" / "md"
        (md_dir / "ghost.md").write_text("![x](missing.png)\n", encoding="utf-8")
        self.src.pull()
        docs = self.src.canonicalize()
        ghost = [d for d in docs if "ghost" in d.source_id][0]
        self.assertIn("missing.png", ghost.body)
        self.assertEqual(ghost.extra["evidence"][0]["kind"], "unresolved")


class TestImageSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        images = self.root / "data" / "assets" / "images"
        images.mkdir(parents=True)
        make_png(images / "log.png", "halMemCreate failed drvRetCode=6")
        self.images = images

    def tearDown(self):
        self.tmp.cleanup()

    def make_src(self, **extra):
        return ImageSource(SourceCfg(id="images", type="image", enabled=True, **extra),
                           project_root=self.root)

    def ocr_path(self):
        return self.root / "data" / "parsed" / "images" / "log.ocr.json"

    def test_ocr_none_skips(self):
        """provider=none：明确跳过，不写 OCR 产物。"""
        self.make_src(ocr_provider="none").canonicalize()
        self.assertFalse(self.ocr_path().exists())

    @mock.patch("vllm_kb.ocr.ocr_image", return_value=(
        "error code 107020, dispatch_ffn_combine failed, GLM-5.1", 0.92))
    def test_ocr_paddle_mode(self, _mock):
        """provider=paddle（明确本地）：不询问，直接 OCR + 签名提取。"""
        self.make_src(ocr_provider="paddle").canonicalize()
        meta = json.loads(self.ocr_path().read_text(encoding="utf-8"))
        kinds = {s["kind"] for s in meta["signatures"]}
        self.assertIn("errcode", kinds)
        self.assertIn("op", kinds)
        self.assertIn("model", kinds)
        self.assertEqual(meta["confidence"], 0.92)

    @mock.patch("vllm_kb.ocr.ocr_image", return_value=("", 0.0))
    @mock.patch("builtins.input", return_value="y")
    def test_ask_no_api_yes_local(self, _in, _ocr):
        """ask + 无 api_base：询问 → y → 本地 paddle 运行。"""
        self.make_src(ocr_provider="ask").canonicalize()
        self.assertTrue(self.ocr_path().exists())

    @mock.patch("builtins.input", return_value="n")
    def test_ask_no_api_no_skip(self, _in):
        """ask + 无 api_base：询问 → n → 跳过 OCR（无产物）。"""
        self.make_src(ocr_provider="ask").canonicalize()
        self.assertFalse(self.ocr_path().exists())

    @mock.patch("vllm_kb.sources.ImageSource._ask_local_ocr", return_value=False)
    def test_ask_noninteractive_skips(self, _ask):
        """ask + 非交互终端：默认跳过。"""
        self.make_src(ocr_provider="ask").canonicalize()
        self.assertFalse(self.ocr_path().exists())

    @mock.patch("vllm_kb.ocr.ocr_image",
                side_effect=OcrApiError("connection refused"))
    @mock.patch("builtins.input", return_value="n")
    def test_api_failure_ask_no_skip(self, _in, _ocr):
        """provider=api 调用失败 → 询问 → n → 跳过。"""
        self.make_src(ocr_provider="api", ocr_api_base="http://127.0.0.1:9999").canonicalize()
        self.assertFalse(self.ocr_path().exists())

    @mock.patch("vllm_kb.ocr.ocr_image",
                side_effect=[OcrApiError("down"), ("halMemCreate failed", 0.8)])
    @mock.patch("builtins.input", return_value="y")
    def test_api_failure_ask_yes_local_retry(self, _in, _ocr):
        """provider=api 失败 → 询问 → y → 本地重试成功。"""
        self.make_src(ocr_provider="api", ocr_api_base="http://127.0.0.1:9999").canonicalize()
        self.assertTrue(self.ocr_path().exists())
        meta = json.loads(self.ocr_path().read_text(encoding="utf-8"))
        self.assertEqual(meta["provider"], "paddle")  # 已切换本地

    def test_ocr_idempotent(self):
        self.make_src(ocr_provider="none").canonicalize()  # 无产物
        # 用 paddle mock 生成产物后，重跑应跳过（sha 一致）
        with mock.patch("vllm_kb.ocr.ocr_image", return_value=("x", 0.5)):
            self.make_src(ocr_provider="paddle").canonicalize()
        mtime1 = self.ocr_path().stat().st_mtime_ns
        with mock.patch("vllm_kb.ocr.ocr_image", return_value=("x2", 0.6)):
            self.make_src(ocr_provider="paddle").canonicalize()
        self.assertEqual(self.ocr_path().stat().st_mtime_ns, mtime1)


class TestOcrApiModel(unittest.TestCase):
    """OCR API 的 model 字段透传（可选，服务端默认）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.png = self.root / "img.png"
        make_png(self.png)

    def tearDown(self):
        self.tmp.cleanup()

    def _fake_urlopen(self, captured):
        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"text": "hi", "confidence": 0.9}).encode()

        def fake(req, timeout=60):
            captured["body"] = json.loads(req.data)
            return FakeResp()

        return fake

    def test_model_passthrough_when_set(self):
        from vllm_kb.ocr import ocr_image

        captured = {}
        with mock.patch("vllm_kb.ocr.urllib_request.urlopen", self._fake_urlopen(captured)):
            text, conf = ocr_image(self.png, "api", api_base="http://ocr:8000", model="table")
        self.assertEqual(captured["body"]["model"], "table")
        self.assertEqual(text, "hi")
        self.assertEqual(conf, 0.9)

    def test_model_absent_when_not_set(self):
        from vllm_kb.ocr import ocr_image

        captured = {}
        with mock.patch("vllm_kb.ocr.urllib_request.urlopen", self._fake_urlopen(captured)):
            ocr_image(self.png, "api", api_base="http://ocr:8000")
        self.assertNotIn("model", captured["body"])

    def test_imagesource_passes_api_model(self):
        """ImageSource 配置 ocr_api_model → 传给 ocr_image。"""
        cfg = SourceCfg(id="images", type="image", ocr_provider="api",
                        ocr_api_base="http://ocr:8000", ocr_api_model="log", enabled=True)
        images = self.root / "data" / "assets" / "images"
        images.mkdir(parents=True)
        make_png(images / "x.png")
        src = ImageSource(cfg, project_root=self.root)
        with mock.patch("vllm_kb.ocr.ocr_image", return_value=("", 0.0)) as m:
            src.canonicalize()
            kwargs = m.call_args.kwargs
            self.assertEqual(kwargs.get("model"), "log")
            self.assertEqual(kwargs.get("api_base"), "http://ocr:8000")
            self.assertEqual(kwargs.get("mode"), "custom")

    def test_imagesource_passes_api_mode(self):
        """ImageSource 配置 ocr_api_mode=openai → 传给 ocr_image。"""
        cfg = SourceCfg(id="images", type="image", ocr_provider="api",
                        ocr_api_base="http://ocr:8000", ocr_api_mode="openai",
                        ocr_api_model="deepseek-ai/DeepSeek-OCR", enabled=True)
        images = self.root / "data" / "assets" / "images"
        images.mkdir(parents=True)
        make_png(images / "x.png")
        src = ImageSource(cfg, project_root=self.root)
        with mock.patch("vllm_kb.ocr.ocr_image", return_value=("", 0.0)) as m:
            src.canonicalize()
            self.assertEqual(m.call_args.kwargs.get("mode"), "openai")
            self.assertEqual(m.call_args.kwargs.get("model"), "deepseek-ai/DeepSeek-OCR")


class TestOcrOpenAiMode(unittest.TestCase):
    """OpenAI 兼容 OCR 模式（siliconflow DeepSeek-OCR 等）：/chat/completions 请求结构。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.png = self.root / "img.png"
        make_png(self.png)

    def tearDown(self):
        self.tmp.cleanup()

    def _fake_urlopen(self, captured):
        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "识别出的文字"}}]}).encode()

        def fake(req, timeout=60):
            captured["body"] = json.loads(req.data)
            captured["url"] = req.full_url
            return FakeResp()

        return fake

    def test_openai_payload_structure(self):
        from vllm_kb.ocr import ocr_image

        captured = {}
        with mock.patch("vllm_kb.ocr.urllib_request.urlopen", self._fake_urlopen(captured)):
            text, conf = ocr_image(self.png, "api", api_base="https://api.example.com/v1",
                                   api_key="k", model="deepseek-ai/DeepSeek-OCR", mode="openai")
        self.assertEqual(captured["url"], "https://api.example.com/v1/chat/completions")
        body = captured["body"]
        self.assertEqual(body["model"], "deepseek-ai/DeepSeek-OCR")
        content = body["messages"][0]["content"]
        self.assertTrue(any(c["type"] == "image_url" for c in content))
        img_url = next(c["image_url"]["url"] for c in content if c["type"] == "image_url")
        self.assertTrue(img_url.startswith("data:image/png;base64,"))
        self.assertEqual(text, "识别出的文字")
        self.assertEqual(conf, 0.0)  # OpenAI 兼容无置信度

    def test_openai_requires_model(self):
        from vllm_kb.ocr import OcrApiError, ocr_image

        with self.assertRaises(OcrApiError) as ctx:
            ocr_image(self.png, "api", api_base="https://x/v1", mode="openai")
        self.assertIn("ocr_api_model", str(ctx.exception))

    def test_openai_bad_response(self):
        from vllm_kb.ocr import OcrApiError, ocr_image

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"unexpected": True}).encode()

        def fake(req, timeout=60):
            return FakeResp()

        with mock.patch("vllm_kb.ocr.urllib_request.urlopen", fake):
            with self.assertRaises(OcrApiError):
                ocr_image(self.png, "api", api_base="https://x/v1",
                          model="m", mode="openai")


if __name__ == "__main__":
    unittest.main()
