"""config.json 统一入口的加载与校验。"""
import json
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig, PROJECT_ROOT


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        # 默认 config 为 openai_compatible，运行时校验要求 key：模拟环境变量后应可加载
        import os

        os.environ["EMBEDDING_API_KEY"] = "dummy-for-test"
        try:
            cfg = AppConfig.load(PROJECT_ROOT / "config.json")
            self.assertEqual(cfg.github, None)  # 已迁移为 sources 格式
            sources = cfg.effective_sources()
            by_id = {s.id: s for s in sources}
            self.assertIn("vllm", by_id)
            vllm = by_id["vllm"]
            self.assertEqual(vllm.type, "github")
            self.assertEqual(vllm.get("repo"), "vllm-project/vllm")
            self.assertEqual(vllm.get("issue_state"), "all")
            self.assertTrue(vllm.get("include_prs"))
            self.assertEqual(vllm.get("max_issues"), 0)  # 0 = 全量
            # vllm-ascend 已启用且排在前（优先完成，主知识库）
            asc = by_id["vllm-ascend"]
            self.assertEqual(asc.get("max_issues"), 0)
            self.assertEqual(sources[0].id, "vllm-ascend")
        finally:
            del os.environ["EMBEDDING_API_KEY"]

    def test_example_config_valid(self):
        # 示例配置也应能通过 pydantic 校验（运行时校验会因空 key 拒绝，这里只校验结构）
        data = json.loads((PROJECT_ROOT / "config.example.json").read_text(encoding="utf-8"))
        cfg = AppConfig.model_validate(data)
        self.assertEqual(cfg.embedding.provider, "openai_compatible")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            AppConfig.load(PROJECT_ROOT / "no_such_config.json")

    def test_openai_compatible_requires_key(self):
        data = {
            "embedding": {"provider": "openai_compatible", "base_url": "http://x/v1", "api_key": ""},
        }
        cfg = AppConfig.model_validate(data)
        with self.assertRaises(ValueError):
            cfg.validate_runtime()

    def test_echo_provider_no_key_needed(self):
        data = {
            "embedding": {"provider": "echo", "base_url": "", "api_key": ""},
        }
        cfg = AppConfig.model_validate(data)
        cfg.validate_runtime()  # 不应抛错

    def test_bad_backend_rejected(self):
        data = {"storage": {"vector_backend": "nope"}}
        cfg = AppConfig.model_validate(data)
        with self.assertRaises(ValueError):
            cfg.validate_runtime()

    def test_env_var_fallback(self):
        import os

        os.environ["VLLM_KB_TEST_TOKEN"] = "abc"
        try:
            data = {"github": {"token": "", "token_env": "VLLM_KB_TEST_TOKEN"}}
            cfg = AppConfig.model_validate(data)
            self.assertEqual(cfg.github.effective_token, "abc")
        finally:
            del os.environ["VLLM_KB_TEST_TOKEN"]


if __name__ == "__main__":
    unittest.main()
