"""数据源抽象层测试：注册表、构建、禁用、未实现来源、旧配置折叠。"""
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig, PROJECT_ROOT, SourceCfg
from vllm_kb.sources import (
    BaseSource,
    ExcelSource,
    GithubSource,
    MarkdownSource,
    build_sources,
    create_source,
    register_source,
)
from vllm_kb.models import KbDocument


class TestSources(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)  # 所有构造统一临时 root，路径解析不落真实仓库

    def tearDown(self):
        self._tmp.cleanup()

    def test_registry_types(self):
        self.assertIsInstance(create_source(SourceCfg(id="a", type="github"), self.root), GithubSource)
        self.assertIsInstance(create_source(SourceCfg(id="a", type="markdown"), self.root), MarkdownSource)
        self.assertIsInstance(create_source(SourceCfg(id="a", type="excel"), self.root), ExcelSource)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            create_source(SourceCfg(id="a", type="nope"), self.root)

    def test_custom_registration(self):
        class MySource(BaseSource):
            type = "custom"

            def pull(self) -> int:
                return 0

            def canonicalize(self) -> list[KbDocument]:
                return []

        register_source("custom", MySource)
        try:
            self.assertIsInstance(create_source(SourceCfg(id="c", type="custom"), self.root), MySource)
        finally:
            from vllm_kb import sources as _s

            _s._REGISTRY.pop("custom")

    def test_unimplemented_source_raises_not_implemented(self):
        # markdown/pdf/excel 已实现；导入目录不存在时返回 0/空（schema-free，不抛错）
        # 临时目录作 project_root：导入路径与资产层（data/assets/*）都不落真实仓库
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        try:
            root = Path(tmp.name)
            nonexistent = str(root / "nonexist")
            src = create_source(SourceCfg(id="xlsx", type="excel", path=nonexistent), root)
            self.assertEqual(src.pull(), 0)
            self.assertEqual(src.canonicalize(), [])
            md = create_source(SourceCfg(id="md", type="markdown", path=nonexistent), root)
            self.assertEqual(md.pull(), 0)
            self.assertEqual(md.canonicalize(), [])
        finally:
            tmp.cleanup()

    def test_github_source_defaults_raw_dir_per_source(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        try:
            root = Path(tmp.name)
            src = create_source(SourceCfg(id="vllm-ascend", type="github"), root)
            self.assertEqual(src.raw_dir.name, "vllm-ascend")
            self.assertEqual(src.raw_dir.parent.name, "raw")
            self.assertEqual(src.puller.repo, "vllm-project/vllm")
        finally:
            tmp.cleanup()

    def test_build_sources_skips_disabled_and_unknown(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        try:
            cfg = AppConfig.model_validate(
                {
                    "sources": [
                        {"id": "a", "type": "github", "enabled": False},
                        {"id": "b", "type": "markdown"},
                        {"id": "c", "type": "ghost-type"},
                    ],
                    "embedding": {"provider": "echo"},
                }
            )
            sources = build_sources(cfg, Path(tmp.name))
            self.assertEqual([s.id for s in sources], ["b"])
        finally:
            tmp.cleanup()

    def test_legacy_github_folding(self):
        """旧版顶层 github 段自动折叠为等效 source（向后兼容）。"""
        cfg = AppConfig.model_validate(
            {
                "github": {"repo": "vllm-project/vllm", "max_issues": 123},
                "embedding": {"provider": "echo"},
            }
        )
        eff = cfg.effective_sources()
        self.assertEqual(len(eff), 1)
        self.assertEqual(eff[0].type, "github")
        self.assertEqual(eff[0].id, "vllm")
        self.assertEqual(eff[0].get("repo"), "vllm-project/vllm")
        self.assertEqual(eff[0].get("max_issues"), 123)

    def test_sources_prefer_over_legacy(self):
        cfg = AppConfig.model_validate(
            {
                "github": {"repo": "old/repo"},
                "sources": [{"id": "new", "type": "github", "repo": "new/repo"}],
                "embedding": {"provider": "echo"},
            }
        )
        eff = cfg.effective_sources()
        self.assertEqual(len(eff), 1)
        self.assertEqual(eff[0].get("repo"), "new/repo")

    def test_collect_docs_empty_when_import_dir_missing(self):
        """流水线 collect_docs：导入目录不存在（markdown/pdf）应返回空而不中断。"""
        import os
        import tempfile
        from vllm_kb.pipeline import collect_docs

        tmp = tempfile.TemporaryDirectory()
        old_root = os.environ.get("VLLM_KB_DATA_ROOT")
        os.environ["VLLM_KB_DATA_ROOT"] = tmp.name
        try:
            cfg = AppConfig.model_validate(
                {
                    "sources": [{"id": "md", "type": "markdown", "path": "x"}],
                    "embedding": {"provider": "echo"},
                }
            )
            docs = collect_docs(cfg, pull=False, limit=None)
            self.assertEqual(docs, [])
        finally:
            if old_root is None:
                os.environ.pop("VLLM_KB_DATA_ROOT", None)
            else:
                os.environ["VLLM_KB_DATA_ROOT"] = old_root
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
