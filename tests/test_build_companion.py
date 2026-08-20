"""build_companion_matrix.py 纯逻辑测试（不触网）：去重、env 提取、release 提取、合并、缺口报告。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_companion_matrix as bm  # noqa: E402


class TestStripSuffix(unittest.TestCase):
    def test_platform_suffixes(self):
        cases = {
            "v0.18.0": "v0.18.0",
            "v0.18.0-a3": "v0.18.0",
            "v0.18.0-a3-openeuler": "v0.18.0",
            "v0.18.0-310p-openeuler": "v0.18.0",
            "v0.13.0rc2-a3-openeuler": "v0.13.0rc2",
            "v0.7.3.post1-openeuler": "v0.7.3.post1",
            "glm5.2-a3-openeuler": "glm5.2",
            "kimi-k3-a3": "kimi-k3",
            "DeepSeekV4-flash-0731-a3-openeuler": "DeepSeekV4-flash-0731",
            "bailing-flash-arm-a3-openeuler": "bailing-flash-arm",
        }
        for tag, expected in cases.items():
            self.assertEqual(bm.strip_platform_suffix(tag), expected, tag)


class TestExtractFromEnv(unittest.TestCase):
    def test_cann_soc_python(self):
        env = [
            "ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.1",
            "PATH=/usr/local/Ascend/cann-8.5.1/bin:/usr/local/python3.11.14/bin:/usr/bin",
            "SOC_VERSION=ascend910b1",
        ]
        out = bm.extract_from_env(env)
        self.assertEqual(out["cann"], "8.5.1")
        self.assertEqual(out["soc"], "ascend910b1")
        self.assertEqual(out["python"], "3.11.14")

    def test_empty_env(self):
        out = bm.extract_from_env([])
        self.assertEqual(out, {"cann": "", "soc": "", "python": ""})


class TestExtractVllmFromRelease(unittest.TestCase):
    def test_upstream_statement(self):
        body = "This release aligns the plugin with upstream vLLM v0.23.0 and expands model support."
        ver, src = bm.extract_vllm_from_release("v0.23.0rc1", body)
        self.assertEqual(ver, "0.23.0")
        self.assertIn("release", src)

    def test_based_on_statement(self):
        ver, src = bm.extract_vllm_from_release("v0.19.1rc1", "This is based on vLLM v0.19.1.")
        self.assertEqual(ver, "0.19.1")

    def test_heuristic_when_no_release(self):
        ver, src = bm.extract_vllm_from_release("v0.19.1rc1", "")
        self.assertEqual(ver, "0.19.1")
        self.assertIn("启发式", src)

    def test_model_tag_no_vllm(self):
        ver, src = bm.extract_vllm_from_release("glm5", "")
        self.assertEqual(ver, "")
        self.assertEqual(src, "")


class TestMergeWithManual(unittest.TestCase):
    def test_manual_wins_for_nonempty(self):
        auto = [
            {"vllm-ascend": "v0.18.0", "vllm": "0.18.0", "cann": "8.5.1",
             "pytorch": "", "pytorch-ascend": "", "npu-driver": "", "notes": "", "source": "自动(x)"}
        ]
        manual = [
            {"vllm-ascend": "v0.18.0", "vllm": "0.12.1", "cann": "", "pytorch": "2.6.0",
             "pytorch-ascend": "2.6.0.post1", "npu-driver": "", "notes": "人工核对", "source": "人工"}
        ]
        merged = bm.merge_with_manual(auto, manual)
        row = merged[0]
        self.assertEqual(row["vllm"], "0.12.1")  # 人工 vllm 优先
        self.assertEqual(row["cann"], "8.5.1")  # 自动补空
        self.assertEqual(row["pytorch"], "2.6.0")
        self.assertEqual(row["source"], "人工")

    def test_manual_rows_outside_quay_preserved(self):
        auto = [{"vllm-ascend": "v0.18.0", "vllm": "0.18.0", "cann": "8.5.1",
                 "pytorch": "", "pytorch-ascend": "", "npu-driver": "", "notes": "", "source": ""}]
        manual = [{"vllm-ascend": "internal-fix", "vllm": "0.10.1", "cann": "8.0", "notes": "内部版本"}]
        merged = bm.merge_with_manual(auto, manual)
        self.assertEqual([r["vllm-ascend"] for r in merged], ["internal-fix", "v0.18.0"])


class TestGapReport(unittest.TestCase):
    def test_gap_count(self):
        rows = [
            {"vllm-ascend": "v0.18.0", "vllm": "0.18.0", "cann": "8.5.1",
             "pytorch": "2.6.0", "pytorch-ascend": "2.6.0.post1", "npu-driver": ""},
            {"vllm-ascend": "glm5", "vllm": "", "cann": "8.5.0",
             "pytorch": "", "pytorch-ascend": "", "npu-driver": ""},
            {"vllm-ascend": "v0.11.0", "vllm": "0.11.0", "cann": "",
             "pytorch": "", "pytorch-ascend": "", "npu-driver": ""},
        ]
        n = bm.report_gaps(rows)
        # glm5（缺 vllm/pytorch/pytorch-ascend）与 v0.11.0（缺 cann/pytorch/pytorch-ascend）计入；
        # 仅缺 npu-driver 的不计（HDK 与镜像解耦，属预期）
        self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
