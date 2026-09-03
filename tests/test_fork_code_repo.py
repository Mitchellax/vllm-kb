"""fork 命名空间检索测试（不触网）：VersionedCode fork 映射/隔离 + api 路由白名单。

fork = 0day 开发分支快照（data/code/forks/{model}/），版本键 = 镜像锁定 SHA 前 12 位，
与官方 rc/release 版本物理隔离（默认检索 repo 缺省时永不混入）。
"""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from vllm_kb.code_index import CodeIndexError, VersionedCode
from vllm_kb.config import AppConfig

SHA = "9ab939da68de3acd6acd40365d4e1bc25ae15d79"
V = SHA[:12]


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


def make_fork_zip(code: VersionedCode, model: str, sha: str) -> None:
    """造最小 fork zip（codeload 形态：vllm-{sha12}/vllm/...）。"""
    zpath = code.zips_dir / f"{sha[:12]}.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "def hy4_custom_kernel(x):\n"
        "    return x + 1\n"
    )
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(f"vllm-{sha[:12]}/vllm/hy4_ops.py", content)
        zf.writestr(f"vllm-{sha[:12]}/vllm/__init__.py", "")


class TestForkVersionedCode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.fork = VersionedCode(self.cfg, repo="fork:hy4")
        make_fork_zip(self.fork, "hy4", SHA)
        self.fork.ensure_snapshot(V)
        # 同时放一个官方 vllm 版本，验证隔离
        self.vllm = VersionedCode(self.cfg, repo="vllm")
        zp = self.vllm.zips_dir / "0.22.1.zip"
        zp.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("vllm-0.22.1/vllm/utils.py", "def get_ip():\n    return '127.0.0.1'\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_fork_root_mapping(self):
        self.assertEqual(self.fork.root, self.cfg.resolve(self.cfg.storage.code_root) / "forks" / "hy4")
        self.assertEqual(self.vllm.root.name, "vllm")

    def test_invalid_model_names_rejected(self):
        for bad in ("fork:../evil", "fork:a/b", "fork:a\\b", "fork:", "fork:."):
            with self.assertRaises(CodeIndexError, msg=bad):
                VersionedCode(self.cfg, repo=bad)

    def test_isolation_from_official_versions(self):
        # fork 命名空间只见自己的 SHA，不见官方版本；反之亦然
        self.assertEqual(self.fork.available_versions, [V])
        self.assertIn("0.22.1", self.vllm.available_versions)
        self.assertNotIn(V, self.vllm.available_versions)
        # 默认仓（vllm-ascend）也不含 fork 版本
        asc = VersionedCode(self.cfg, repo="vllm-ascend")
        self.assertNotIn(V, asc.available_versions)

    def test_symbol_and_file_access(self):
        self.fork.build_index_for_version(V)
        hits = self.fork.search_symbols("hy4_custom_kernel", V)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["file"], "vllm/hy4_ops.py")
        text = self.fork.read_file(V, "vllm/hy4_ops.py")
        self.assertIn("hy4_custom_kernel", text or "")
        g = self.fork.grep("hy4_custom_kernel", V)
        self.assertTrue(g)

    def test_fork_index_uses_vllm_subdir(self):
        # fork 快照是 vllm 源码树：索引只扫 vllm/ 子目录（与 repo=vllm 相同规则）
        n = self.fork.build_index_for_version(V)
        self.assertGreaterEqual(n, 1)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi"),
    "fastapi 未安装（pip install fastapi uvicorn）",
)
class TestForkApiRoutes(unittest.TestCase):
    def setUp(self):
        import os

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_kb import api_code

        self._old_key = os.environ.get("EMBEDDING_API_KEY")
        self._old_gh = os.environ.get("GITHUB_TOKEN")
        os.environ["EMBEDDING_API_KEY"] = "dummy-for-test"
        os.environ["GITHUB_TOKEN"] = "dummy-for-test"

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        fork = VersionedCode(self.cfg, repo="fork:hy4")
        make_fork_zip(fork, "hy4", SHA)
        fork.ensure_snapshot(V)
        fork.build_index_for_version(V)
        (fork.root / "meta.json").write_text(json.dumps({
            "model": "hy4", "repo": "voidvelocity/vllm", "ref": "dev_hy4",
            "base": "0.23.0", "sha": SHA, "image_digest": "sha256:x",
        }), encoding="utf-8")

        from types import SimpleNamespace

        app = FastAPI()
        api_code.register(app, SimpleNamespace(cfg=self.cfg))
        self.client = TestClient(app)

    def tearDown(self):
        import os

        for name, old in (("EMBEDDING_API_KEY", self._old_key),
                          ("GITHUB_TOKEN", self._old_gh)):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        self.tmp.cleanup()

    def test_versions_with_meta(self):
        r = self.client.get("/code/versions", params={"repo": "fork:hy4"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["versions"], [V])
        self.assertEqual(body["meta"]["repo"], "voidvelocity/vllm")
        self.assertEqual(body["meta"]["base"], "0.23.0")
        self.assertIn("锁定", body["note"])

    def test_default_repo_excludes_fork(self):
        # 默认（repo 缺省）绝不混入 fork 版本
        r = self.client.get("/code/versions")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn(V, r.json()["versions"])

    def test_invalid_fork_model_400(self):
        r = self.client.get("/code/versions", params={"repo": "fork:../evil"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("非法", r.json()["detail"])

    def test_unknown_repo_empty(self):
        r = self.client.get("/code/versions", params={"repo": "bogus"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["versions"], [])

    def test_search_and_file_on_fork(self):
        r = self.client.post("/code/search", json={
            "keyword": "hy4_custom_kernel", "version": V, "repo": "fork:hy4"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["hits"])

        r2 = self.client.get("/code/file", params={
            "version": V, "path": "vllm/hy4_ops.py", "repo": "fork:hy4"})
        self.assertEqual(r2.status_code, 200)
        self.assertIn("hy4_custom_kernel", r2.json()["content"])


if __name__ == "__main__":
    unittest.main()
