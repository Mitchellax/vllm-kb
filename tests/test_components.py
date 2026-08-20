"""组件版本提取与仓库组件推断测试。"""
import unittest

from vllm_kb.components import default_component_for_repo, extract_component_versions


class TestExtractComponentVersions(unittest.TestCase):
    def test_vllm_template(self):
        out = extract_component_versions("- **vLLM version**: 0.26.0")
        self.assertEqual(out.get("vllm"), "0.26.0")

    def test_vllm_ascend_and_companions(self):
        body = (
            "vllm-ascend version: 0.18.0\n"
            "vllm 0.12.1, CANN 8.1.RC2, torch 2.6.0, torch_npu 2.6.0.post1"
        )
        out = extract_component_versions(body)
        self.assertEqual(out.get("vllm-ascend"), "0.18.0")
        self.assertEqual(out.get("vllm"), "0.12.1")
        self.assertEqual(out.get("cann"), "8.1.RC2")
        self.assertEqual(out.get("pytorch"), "2.6.0")
        self.assertEqual(out.get("pytorch-ascend"), "2.6.0.post1")

    def test_vllm_ascend_not_matched_by_vllm(self):
        """'vllm-ascend' 不应被 'vllm' 别名误匹配。"""
        out = extract_component_versions("vllm-ascend version: 0.18.0")
        self.assertNotIn("vllm", out)
        self.assertEqual(out.get("vllm-ascend"), "0.18.0")

    def test_torch_inside_pytorch_not_matched(self):
        out = extract_component_versions("pytorch 2.6.0")
        self.assertEqual(out.get("pytorch"), "2.6.0")
        self.assertNotIn("pytorch-ascend", out)

    def test_incomplete_version_not_matched(self):
        out = extract_component_versions("known issue with vllm 0.1 and vLLM 0.23")
        self.assertNotIn("vllm", out)

    def test_npu_driver(self):
        out = extract_component_versions("npu-driver 24.1.rc1")
        self.assertEqual(out.get("npu-driver"), "24.1.rc1")

    def test_empty(self):
        self.assertEqual(extract_component_versions(None), {})
        self.assertEqual(extract_component_versions(""), {})


class TestDefaultComponent(unittest.TestCase):
    def test_repos(self):
        self.assertEqual(default_component_for_repo("vllm-project/vllm"), "vllm")
        self.assertEqual(default_component_for_repo("vllm-project/vllm-ascend"), "vllm-ascend")
        self.assertEqual(default_component_for_repo("pytorch/pytorch"), "pytorch")
        self.assertEqual(default_component_for_repo("Ascend/pytorch-ascend"), "pytorch-ascend")
        self.assertEqual(default_component_for_repo("some/cann-extra"), "cann")


if __name__ == "__main__":
    unittest.main()
