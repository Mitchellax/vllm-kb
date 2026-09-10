"""img:{tag} 命名空间测试（不触网）：镜像插件源码检索 + 跨命名空间 diff。

img = 0day 镜像内 vllm-ascend 插件源码（data/code/images/{tag}/，从镜像插件层提取），
版本键 = 镜像 tag；与官方 rc/release 版本、fork 快照三者物理隔离。
"""
import json
import tempfile
import unittest
from pathlib import Path

from vllm_kb.code_index import CodeIndexError, VersionedCode
from vllm_kb.config import AppConfig

TAG = "glm5.2"
SHA = "9ab939da68de3acd6acd40365d4e1bc25ae15d79"


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
            },
        }
    )


def write_plugin_tree(code: VersionedCode, tag: str) -> None:
    """造最小插件快照：vllm_ascend/ + csrc/ + tests/（不应被索引）。"""
    snap = code.snapshots_dir / tag
    (snap / "vllm_ascend" / "worker").mkdir(parents=True, exist_ok=True)
    (snap / "vllm_ascend" / "worker" / "model_runner.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def glm52_patch_attention(x):\n"
        "    logger.error('GLM52 attention patch failed: unexpected shape')\n"
        "    return x\n",
        encoding="utf-8",
    )
    (snap / "csrc" / "ops").mkdir(parents=True, exist_ok=True)
    (snap / "csrc" / "ops" / "dispatch_ffn_combine.cpp").write_text(
        "void dispatch_ffn_combine_custom() {\n}\n", encoding="utf-8")
    (snap / "tests").mkdir(parents=True, exist_ok=True)
    (snap / "tests" / "test_x.py").write_text("def should_not_be_indexed():\n    pass\n",
                                              encoding="utf-8")


class TestImageVersionedCode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.img = VersionedCode(self.cfg, repo=f"img:{TAG}")
        write_plugin_tree(self.img, TAG)
        # 同时放官方版本与 fork 快照，验证三者隔离
        self.asc = VersionedCode(self.cfg, repo="vllm-ascend")
        zp = self.asc.zips_dir / "v0.23.0.zip"
        zp.parent.mkdir(parents=True, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("vllm-ascend-v0.23.0/vllm_ascend/platform.py",
                        "def official_only():\n    pass\n")
        self.fork = VersionedCode(self.cfg, repo="fork:hy4")
        fz = self.fork.zips_dir / f"{SHA[:12]}.zip"
        fz.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(fz, "w") as zf:
            zf.writestr(f"vllm-{SHA[:12]}/vllm/fork_only.py", "def fork_only():\n    pass\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_root_mapping(self):
        self.assertEqual(self.img.root,
                         self.cfg.resolve(self.cfg.storage.code_root) / "images" / TAG)

    def test_invalid_tags_rejected(self):
        for bad in ("img:../evil", "img:a/b", "img:a\\b", "img:", "img:."):
            with self.assertRaises(CodeIndexError, msg=bad):
                VersionedCode(self.cfg, repo=bad)

    def test_uppercase_tag_allowed(self):
        # quay 实际有 DeepSeekV4-flash-0731 这类大写 tag
        v = VersionedCode(self.cfg, repo="img:DeepSeekV4-flash-0731")
        self.assertEqual(v.root.name, "DeepSeekV4-flash-0731")

    def test_namespace_isolation(self):
        self.assertEqual(self.img.available_versions, [TAG])
        self.assertNotIn(TAG, self.asc.available_versions)
        self.assertNotIn(TAG, self.fork.available_versions)

    def test_index_plugin_dirs_only(self):
        n = self.img.build_index_for_version(TAG)
        self.assertGreaterEqual(n, 2)
        hits = self.img.search_symbols("glm52_patch_attention", TAG)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["file"], "vllm_ascend/worker/model_runner.py")
        # tests/ 不索引
        self.assertFalse(self.img.search_symbols("should_not_be_indexed", TAG))

    def test_message_literal_and_read(self):
        self.img.build_index_for_version(TAG)
        hits = self.img.search_messages("GLM52 attention patch failed", TAG)
        self.assertTrue(hits)
        self.assertTrue(hits[0]["file"].startswith("vllm_ascend/"))
        text = self.img.read_file(TAG, "csrc/ops/dispatch_ffn_combine.cpp")
        self.assertIn("dispatch_ffn_combine_custom", text or "")

    def test_preset_hint_mentions_image_script(self):
        with self.assertRaises(CodeIndexError) as cm:
            self.img.ensure_snapshot("no-such-tag")
        self.assertIn("build_image_snapshots.py", str(cm.exception))

    def test_missing_image_lists_extracted_ones(self):
        # 未提取的镜像：错误信息给出已提取清单（agent 自我纠正路径）
        other = VersionedCode(self.cfg, repo="img:kimi-k3")
        write_plugin_tree(other, "kimi-k3")
        with self.assertRaises(CodeIndexError) as cm:
            VersionedCode(self.cfg, repo="img:not-extracted").ensure_snapshot("not-extracted")
        msg = str(cm.exception)
        self.assertIn("已提取", msg)
        self.assertIn("glm5.2", msg)
        self.assertIn("kimi-k3", msg)

    def test_list_image_snapshots_exposes_group_and_variants(self):
        (self.img.root).mkdir(parents=True, exist_ok=True)
        (self.img.root / "meta.json").write_text(json.dumps({
            "tag": TAG, "group": "hy4", "variants": ["hy4-a3", "hy4"],
            "image_digest": "sha256:" + "b" * 64, "vllm_baseline": "0.23.0",
        }), encoding="utf-8")
        from vllm_kb.code_index import list_image_snapshots

        got = list_image_snapshots(self.cfg.resolve(self.cfg.storage.code_root))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["group"], "hy4")          # 用户可能按组名找（hy4 镜像）
        self.assertEqual(got[0]["variants"], ["hy4-a3", "hy4"])
        self.assertTrue(got[0]["extracted"])


@unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi"),
    "fastapi 未安装（pip install fastapi uvicorn）",
)
class TestImageApiRoutes(unittest.TestCase):
    def setUp(self):
        import os

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_kb import api_code

        self._old_key = os.environ.get("EMBEDDING_API_KEY")
        os.environ["EMBEDDING_API_KEY"] = "dummy-for-test"
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        img = VersionedCode(self.cfg, repo=f"img:{TAG}")
        write_plugin_tree(img, TAG)
        img.build_index_for_version(TAG)
        (img.root / "meta.json").write_text(json.dumps({
            "tag": TAG, "group": TAG, "image_digest": "sha256:" + "a" * 64,
            "image_created": "2026-07-27T15:39:42Z",
            "vllm_commit": "c" * 40, "vllm_commit_date": "2026-06-15T03:35:17Z",
        }), encoding="utf-8")
        # 官方同名文件（跨命名空间 diff 用）
        asc = VersionedCode(self.cfg, repo="vllm-ascend")
        import zipfile

        zp = asc.zips_dir / "v0.23.0.zip"
        zp.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("vllm-ascend-v0.23.0/vllm_ascend/worker/model_runner.py",
                        "def glm52_patch_attention(x):\n    return x\n")
        asc.ensure_snapshot("v0.23.0")

        from types import SimpleNamespace

        app = FastAPI()
        api_code.register(app, SimpleNamespace(cfg=self.cfg))
        self.client = TestClient(app)

    def tearDown(self):
        import os

        if self._old_key is None:
            os.environ.pop("EMBEDDING_API_KEY", None)
        else:
            os.environ["EMBEDDING_API_KEY"] = self._old_key
        self.tmp.cleanup()

    def test_versions_aggregation_for_img(self):
        r = self.client.get("/code/versions", params={"repo": "img"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["versions"], [TAG])
        self.assertEqual(body["images"][0]["tag"], TAG)
        self.assertEqual(body["images"][0]["vllm_commit"], "c" * 40)
        self.assertTrue(body["images"][0]["indexed"])

    def test_versions_with_tag_returns_meta(self):
        r = self.client.get("/code/versions", params={"repo": f"img:{TAG}"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["versions"], [TAG])
        self.assertEqual(body["meta"]["image_digest"], "sha256:" + "a" * 64)
        self.assertIn("build_image_snapshots.py", body["note"])

    def test_invalid_img_tag_400(self):
        r = self.client.get("/code/versions", params={"repo": "img:../evil"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("非法", r.json()["detail"])

    def test_search_and_file_on_image(self):
        r = self.client.post("/code/search", json={
            "keyword": "glm52_patch_attention", "version": TAG, "repo": f"img:{TAG}"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["hits"])
        r2 = self.client.get("/code/file", params={
            "version": TAG, "path": "vllm_ascend/worker/model_runner.py", "repo": f"img:{TAG}"})
        self.assertEqual(r2.status_code, 200)
        self.assertIn("glm52_patch_attention", r2.json()["content"])

    def test_msg_kind_on_image(self):
        r = self.client.post("/code/search", json={
            "keyword": "unexpected shape", "version": TAG, "repo": f"img:{TAG}", "kind": "msg"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["mode"], "message_index")
        self.assertTrue(r.json()["hits"])

    def test_cross_namespace_diff(self):
        # 镜像插件代码 vs 官方同基线版本：直接看出 0day 定制
        r = self.client.get("/code/diff", params={
            "version1": f"img:{TAG}", "version2": "vllm-ascend:v0.23.0",
            "path": "vllm_ascend/worker/model_runner.py"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["repo"], f"img:{TAG}")
        self.assertEqual(body["repo2"], "vllm-ascend")
        self.assertIn("+", body["diff"])
        self.assertIn("logger.error", body["diff"])

    def test_same_repo_diff_still_works(self):
        # 同命名空间两版本（此处同一 tag）→ 无差异，但请求成立（不因前缀解析而 404）
        r = self.client.get("/code/diff", params={
            "version1": TAG, "version2": TAG, "repo": f"img:{TAG}",
            "path": "vllm_ascend/worker/model_runner.py"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["diff"], "")

    def test_diff_missing_file_lists_both_sides(self):
        r = self.client.get("/code/diff", params={
            "version1": f"img:{TAG}", "version2": "vllm-ascend:v0.23.0",
            "path": "vllm_ascend/nope.py"})
        self.assertEqual(r.status_code, 404)
        self.assertIn("img:glm5.2", r.json()["detail"])

    def test_fork_prefix_auto_unique_version(self):
        fork = VersionedCode(self.cfg, repo="fork:hy4")
        import zipfile

        z = fork.zips_dir / f"{SHA[:12]}.zip"
        z.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr(f"vllm-{SHA[:12]}/vllm/x.py", "def fork_x():\n    pass\n")
        fork.ensure_snapshot(SHA[:12])
        # fork:{model} 未带 @sha，命名空间唯一版本 → 自动取用（与官方同名路径比会 404，
        # 但错误信息应说明两侧，证明解析走到了 fork 命名空间）
        r = self.client.get("/code/diff", params={
            "version1": "fork:hy4", "version2": "vllm-ascend:v0.23.0",
            "path": "vllm/x.py"})
        self.assertEqual(r.status_code, 404)
        self.assertIn("fork:hy4", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
