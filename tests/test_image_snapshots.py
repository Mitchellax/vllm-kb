"""镜像插件层提取测试（不触网）：层定位 / tar 安全解包 / digest 锚 / meta 与索引。

镜像 = quay ascend/vllm-ascend 的 0day 模型 tag；只提取
`COPY . /vllm-workspace/vllm-ascend/` 那一层（插件源码），二进制层不拉。
"""
import gzip
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_image_snapshots as bis  # noqa: E402

from vllm_kb.code_index import VersionedCode  # noqa: E402
from vllm_kb.config import AppConfig  # noqa: E402

TAG = "glm5.2"
DIGEST = "sha256:" + "d" * 64
LAYER_DIGEST = "sha256:" + "e" * 64


def make_cfg(tmp: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"name": "test", "data_root": "data"},
            "embedding": {"provider": "echo", "dimensions": 64},
            "storage": {
                "vector_backend": "python",
                "lancedb_path": str(tmp / "vec.json"),
                "sqlite_path": str(tmp / "kb.sqlite3"),
                "canonical_file": str(tmp / "canonical.jsonl"),
                "code_root": str(tmp / "code"),
                "companion_file": str(tmp / "compat.json"),
            },
        }
    )


def make_layer_bytes(entries: dict[str, str], symlink: str | None = None) -> bytes:
    """构造 docker 层形态的 tar.gz（gzip tar），条目名按镜像内绝对路径给。"""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tf:
        for name, content in entries.items():
            data = content.encode("utf-8")
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        if symlink:
            ti = tarfile.TarInfo(symlink)
            ti.type = tarfile.SYMTYPE
            ti.linkname = "/etc/passwd"
            tf.addfile(ti)
    return gzip.compress(raw.getvalue())


def layer_history() -> list[dict]:
    return [
        {"created_by": "COPY /usr/local/Ascend /usr/local/Ascend # buildkit",
         "empty_layer": False},
        {"created_by": "ARG VLLM_TAG=v0.23.0", "empty_layer": True},
        {"created_by": "RUN |2 /bin/bash -c pip install -e /vllm-workspace/vllm # buildkit",
         "empty_layer": False},
        {"created_by": "COPY . /vllm-workspace/vllm-ascend/ # buildkit", "empty_layer": False},
        {"created_by": "RUN |4 /bin/bash -c python setup.py build # buildkit", "empty_layer": False},
    ]


class TestLayerLocate(unittest.TestCase):
    def test_locate_plugin_layer(self):
        layers = ["sha256:a", "sha256:b", "sha256:c", "sha256:d"]
        # 非空 history 条目 = 3 条（Ascend/pip/COPY/build）→ COPY 是第 2 个非空 → layers[2]
        hist = layer_history()
        self.assertEqual(bis.locate_plugin_layer(hist, layers), 2)

    def test_missing_layer_returns_minus_one(self):
        hist = [{"created_by": "RUN something", "empty_layer": False}]
        self.assertEqual(bis.locate_plugin_layer(hist, ["sha256:a"]), -1)

    def test_index_beyond_layers_rejected(self):
        hist = layer_history()
        self.assertEqual(bis.locate_plugin_layer(hist, ["sha256:a"]), -1)


class TestSafeRel(unittest.TestCase):
    def test_strips_prefix(self):
        self.assertEqual(
            bis._safe_rel("vllm-workspace/vllm-ascend/vllm_ascend/x.py", bis._PLUGIN_PREFIX),
            "vllm_ascend/x.py")
        self.assertEqual(
            bis._safe_rel("./vllm-workspace/vllm-ascend/csrc/a.cpp", bis._PLUGIN_PREFIX),
            "csrc/a.cpp")
        self.assertEqual(
            bis._safe_rel("vllm-workspace/vllm-ascend/.git/shallow", bis._PLUGIN_PREFIX),
            ".git/shallow")

    def test_rejects_unsafe(self):
        for bad in ("etc/passwd", "vllm-workspace/vllm-ascend/../../etc/passwd",
                    "/vllm-workspace/vllm-ascend/x.py", "vllm-workspace/vllm-ascend/",
                    "vllm-workspace/other/x.py"):
            self.assertIsNone(bis._safe_rel(bad, bis._PLUGIN_PREFIX), bad)


class TestExtractPluginTree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "snap"

    def tearDown(self):
        self.tmp.cleanup()

    def _blob(self):
        return io.BytesIO(make_layer_bytes(
            {
                "vllm-workspace/vllm-ascend/vllm_ascend/platform.py":
                    "def patched():\n    return 1\n",
                "vllm-workspace/vllm-ascend/csrc/ops/a.cpp": "void op_a() {}\n",
                "vllm-workspace/vllm-ascend/.git/shallow": "f" * 40 + "\n",
                "etc/passwd": "root:x:0:0\n",                       # 前缀外：丢弃
                "vllm-workspace/vllm-ascend/../escape.py": "evil\n",  # 路径穿越：丢弃
            },
            symlink="vllm-workspace/vllm-ascend/link.py",
        ))

    def test_extract_only_plugin_prefix(self):
        info = bis.extract_plugin_tree(self._blob(), self.dest)
        self.assertTrue((self.dest / "vllm_ascend" / "platform.py").exists())
        self.assertTrue((self.dest / "csrc" / "ops" / "a.cpp").exists())
        # 前缀外与路径穿越都不落盘
        self.assertFalse((self.dest / "etc").exists())
        self.assertFalse((self.dest.parent / "escape.py").exists())
        # symlink 跳过
        self.assertFalse((self.dest / "link.py").exists())
        # .git 不入快照，但识别出并读出 commit
        self.assertFalse((self.dest / ".git").exists())
        self.assertTrue(info["has_git"])
        self.assertEqual(info["plugin_commit"], "f" * 40)
        self.assertEqual(info["files"], 2)

    def test_no_git_means_empty_commit(self):
        blob = io.BytesIO(make_layer_bytes({
            "vllm-workspace/vllm-ascend/vllm_ascend/platform.py": "x = 1\n"}))
        info = bis.extract_plugin_tree(blob, self.dest)
        self.assertFalse(info["has_git"])
        self.assertEqual(info["plugin_commit"], "")


class TestDownloadLayer(unittest.TestCase):
    """层体下载：流式读取 + 整层重试（quay CDN 偶发读超时）。"""

    class _Resp:
        def __init__(self, chunks, fail_after=None):
            self._chunks = chunks
            self._fail_after = fail_after

        def raise_for_status(self):
            return None

        def iter_content(self, n):
            for i, c in enumerate(self._chunks):
                if self._fail_after is not None and i >= self._fail_after:
                    raise OSError("read timeout")
                yield c

    class _Sess:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls = 0

        def get(self, url, **kw):
            self.calls += 1
            return self._responses.pop(0)

    def test_retries_whole_layer_on_timeout(self):
        sess = self._Sess([self._Resp([b"x" * 10], fail_after=0),
                           self._Resp([b"y" * 10])])
        meta = {"session": sess, "v2": "v2", "token": "t"}
        with mock.patch("build_image_snapshots.time.sleep"):
            buf = bis.download_layer(meta, "sha256:l", 10)
        self.assertEqual(sess.calls, 2)
        self.assertEqual(buf.read(), b"y" * 10)

    def test_raises_after_all_retries(self):
        sess = self._Sess([self._Resp([b"x"], fail_after=0) for _ in range(3)])
        meta = {"session": sess, "v2": "v2", "token": "t"}
        with mock.patch("build_image_snapshots.time.sleep"), \
                self.assertRaises(RuntimeError):
            bis.download_layer(meta, "sha256:l", 10, retries=3)
        self.assertEqual(sess.calls, 3)


class _FakeResp:
    """最小 HTTP 响应：状态码 + json 载荷 + raise_for_status 行为。"""

    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # (url, headers)

    def get(self, url, **kw):
        self.calls.append((url, kw.get("headers") or {}))
        return self._responses.pop(0)


class TestFetchImageLayersHardening(unittest.TestCase):
    """在线路径加固：HTTP 错误态绝不流入解析；blob 401 刷新匿名 token 重试。

    背景（真实事故）：匿名 token 有效期短，首个镜像下载重试耗掉有效期后，
    后续 config blob 请求全部 401——旧代码不查状态码，错误 JSON 被解析成
    空 history，整批镜像被误报"未找到插件层，跳过"。
    """

    MANIFEST = {"config": {"digest": "sha256:" + "c" * 64},
                "layers": [{"digest": "sha256:" + "e" * 64, "size": 123}]}
    CONFIG = {"history": [{"created_by": "COPY . /vllm-workspace/vllm-ascend/ # buildkit",
                           "empty_layer": False}]}

    def _run(self, session, token="expired"):
        with mock.patch("vllm_kb.net.get_session", return_value=session), \
                mock.patch("build_image_snapshots.time.sleep"):
            return bis.fetch_image_layers({"manifest_digest": "sha256:" + "d" * 64},
                                          token, insecure=False)

    def test_blob_401_refreshes_token_and_retries(self):
        sess = _FakeSession([
            _FakeResp(200, {"manifest_data": json.dumps(self.MANIFEST)}),  # manifest
            _FakeResp(401, {"errors": [{"code": "UNAUTHORIZED"}]}),        # blob：token 过期
            _FakeResp(200, self.CONFIG),                                   # 刷新后成功
        ])
        with mock.patch.object(bis.bcm, "get_quay_token", return_value="fresh") as gt:
            meta = self._run(sess)
        self.assertTrue(gt.called)
        self.assertEqual(meta["token"], "fresh")
        self.assertEqual(meta["history"], self.CONFIG["history"])
        self.assertEqual(sess.calls[2][1].get("Authorization"), "Bearer fresh")

    def test_deterministic_4xx_raises_without_retry(self):
        sess = _FakeSession([_FakeResp(404, {"error": "not found"})])
        with self.assertRaises(requests.exceptions.HTTPError):
            self._run(sess)
        self.assertEqual(len(sess.calls), 1)  # 404 确定性错误：不重试

    def test_429_still_retries_then_raises(self):
        sess = _FakeSession([_FakeResp(429, {}) for _ in range(4)])
        with self.assertRaises(requests.exceptions.HTTPError):
            self._run(sess)
        self.assertEqual(len(sess.calls), 4)


class TestCandidates(unittest.TestCase):
    def test_only_non_version_groups(self):
        tags = [
            {"name": "v0.23.0", "manifest_digest": "sha256:1"},
            {"name": "glm5.2", "manifest_digest": "sha256:2"},
            {"name": "glm5.2-a3", "manifest_digest": "sha256:3"},
            {"name": "kimi-k3", "manifest_digest": "sha256:4"},
            {"name": "nightly-main", "manifest_digest": "sha256:5"},  # 看护规则排除
        ]
        cands = bis.candidate_images(tags)
        by_tag = {c["tag"]: c for c in cands}
        self.assertEqual(set(by_tag), {"glm5.2", "kimi-k3"})
        # 组内平台变体归组，代表 tag 取无后缀原名
        self.assertEqual(by_tag["glm5.2"]["group"], "glm5.2")
        self.assertEqual(sorted(by_tag["glm5.2"]["variants"]), ["glm5.2", "glm5.2-a3"])

    def test_only_filter_by_tag_or_group(self):
        tags = [{"name": "glm5.2", "manifest_digest": "sha256:2"},
                {"name": "glm5.2-a3", "manifest_digest": "sha256:3"},
                {"name": "kimi-k3", "manifest_digest": "sha256:4"}]
        self.assertEqual([c["tag"] for c in bis.candidate_images(tags, only=["glm5.2"])],
                         ["glm5.2"])
        # 平台变体 tag 也指向同一组（取该组代表 tag）
        self.assertEqual([c["tag"] for c in bis.candidate_images(tags, only=["glm5.2-a3"])],
                         ["glm5.2"])
        self.assertEqual(sorted(c["tag"] for c in bis.candidate_images(tags, only=["kimi-k3"])),
                         ["kimi-k3"])


class TestExtractOne(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.cand = {"group": TAG, "tag": TAG, "variants": [TAG],
                     "tag_info": {"name": TAG, "manifest_digest": DIGEST,
                                  "last_modified": "Mon, 27 Jul 2026 15:39:42 -0000"}}
        self.matrix = {TAG: {"vllm-ascend": TAG, "vllm": "0.23.0",
                             "vllm_commit": "c" * 40,
                             "vllm_commit_date": "2026-06-15T03:35:17Z",
                             "image_created": "2026-07-27T15:39:42Z"}}
        self.layers = {"layers": ["sha256:ascend", LAYER_DIGEST],
                       "layer_sizes": [100, 200],
                       "history": [
                           {"created_by": "COPY /usr/local/Ascend /usr/local/Ascend",
                            "empty_layer": False},
                           {"created_by": "COPY . /vllm-workspace/vllm-ascend/ # buildkit",
                            "empty_layer": False}],
                       "session": None, "v2": "v2", "token": "t"}
        self.blob = make_layer_bytes({
            "vllm-workspace/vllm-ascend/vllm_ascend/platform.py":
                "def glm52_platform():\n    return 'patched'\n",
            "vllm-workspace/vllm-ascend/csrc/a.cpp": "void a() {}\n"})

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, **kw):
        with mock.patch.object(bis, "fetch_image_layers", return_value=self.layers), \
                mock.patch.object(bis, "download_layer",
                                  return_value=io.BytesIO(self.blob)) as dl:
            r = bis.extract_one(self.cand, self.cfg, "token", self.matrix, **kw)
        return r, dl

    def test_extract_index_and_meta(self):
        r, dl = self._run()
        self.assertEqual(r["status"], "ok")
        self.assertTrue(dl.called)
        root = bis.image_dir(TAG, self.cfg.resolve(self.cfg.storage.code_root))
        self.assertTrue((root / "snapshots" / TAG / "vllm_ascend" / "platform.py").exists())
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["image_digest"], DIGEST)
        self.assertEqual(meta["layer_digest"], LAYER_DIGEST)
        self.assertEqual(meta["vllm_commit"], "c" * 40)
        self.assertEqual(meta["vllm_baseline"], "0.23.0")
        self.assertEqual(meta["image_created"], "2026-07-27T15:39:42Z")
        self.assertEqual(meta["plugin_layer_index"], 1)
        self.assertGreater(meta["files"], 0)
        # 索引可用（检索侧）
        code = VersionedCode(self.cfg, repo=f"img:{TAG}")
        self.assertTrue(code.search_symbols("glm52_platform", TAG))

    def test_digest_anchor_skips_reextract(self):
        self._run()
        # 第二次：digest 未变 → 不触网（fetch_image_layers 若被调用即失败）
        with mock.patch.object(bis, "fetch_image_layers",
                               side_effect=AssertionError("digest 锚命中不应触网")):
            r = bis.extract_one(self.cand, self.cfg, "token", self.matrix)
        self.assertEqual(r["status"], "ok")

    def test_refresh_forces_reextract(self):
        self._run()
        r, dl = self._run(refresh=True)
        self.assertEqual(r["status"], "ok")
        self.assertTrue(dl.called)

    def test_no_plugin_layer(self):
        self.layers["history"] = [{"created_by": "RUN x", "empty_layer": False}]
        with mock.patch.object(bis, "fetch_image_layers", return_value=self.layers), \
                mock.patch.object(bis, "download_layer",
                                  side_effect=AssertionError("无插件层不应下载")):
            r = bis.extract_one(self.cand, self.cfg, "token", self.matrix)
        self.assertEqual(r["status"], "no-layer")

    def test_index_only_without_snapshot_skips(self):
        with mock.patch.object(bis, "fetch_image_layers",
                               side_effect=AssertionError("--index-only 不应触网")):
            r = bis.extract_one(self.cand, self.cfg, "token", self.matrix, index_only=True)
        self.assertEqual(r["status"], "skip")

    def test_matrix_commit_absent_falls_back_to_tag_lookup(self):
        self.matrix = {TAG: {"vllm-ascend": TAG, "vllm": "0.23.0"}}  # 无 vllm_commit
        with mock.patch.object(bis.bcm, "_github_token", return_value="tok"), \
                mock.patch.object(bis.bcm, "fetch_tag_commit",
                                  return_value={"sha": "a" * 40, "date": "2026-01-01T00:00:00Z",
                                                "src": "github"}) as ft:
            r, _ = self._run()
        self.assertTrue(ft.called)
        self.assertEqual(r["meta"]["vllm_commit"], "a" * 40)

    def test_without_token_no_tag_lookup(self):
        self.matrix = {TAG: {"vllm-ascend": TAG, "vllm": "0.23.0"}}
        with mock.patch.object(bis.bcm, "_github_token", return_value=""), \
                mock.patch.object(bis.bcm, "fetch_tag_commit",
                                  side_effect=AssertionError("无 token 不应触网")):
            r, _ = self._run()
        self.assertEqual(r["meta"]["vllm_commit"], "")


if __name__ == "__main__":
    unittest.main()
