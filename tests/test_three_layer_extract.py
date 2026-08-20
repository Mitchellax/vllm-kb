"""三层签名提取测试：源码符号表 + 结构解析 + 信号词（agent 判断）。"""
import tempfile
import unittest
from pathlib import Path

from vllm_kb.signature import Signature, _dedupe, extract_signatures
from vllm_kb.symbol_table import SymbolTable


def make_table() -> SymbolTable:
    t = SymbolTable()
    t.add("dispatch_ffn_combine", "kernel", "v0.23.0rc1", 2.5)
    t.add("mega_moe", "kernel", "v0.24.0rc1", 2.5)
    t.add("npu_moe_distribute_dispatch", "kernel", "v0.23.0rc1", 3.0)
    t.add("enable_fused_mc2", "feature", "v0.23.0rc1", 1.8)
    t.add("vllm_ascend_enable_fused_mc2", "feature", "v0.23.0rc1", 2.0)
    t.add("dsa_cp", "feature", "v0.23.0rc1", 2.0)
    t.add("GLM-5.1", "model", "kb", 1.0)
    return t


class TestThreeLayerExtract(unittest.TestCase):
    def setUp(self):
        self.table = make_table()

    def test_symbol_table_layer_kernel(self):
        text = "kernel_name=DispatchFFNCombine, errorStr: timeout or trap error"
        sigs = extract_signatures(text, symbol_table=self.table)
        kinds = {s.kind for s in sigs}
        self.assertIn("kernel", kinds)
        # 变体归一化：驼峰 DispatchFFNCombine 命中下划线 dispatch_ffn_combine 符号
        self.assertTrue(
            any("dispatch_ffn_combine" in s.text or "dispatchffncombine" in s.text for s in sigs)
        )

    def test_symbol_table_layer_feature(self):
        text = "VLLM_ASCEND_ENABLE_FUSED_MC2=1 GLM-5.1"
        sigs = extract_signatures(text, symbol_table=self.table)
        self.assertTrue(any("vllm_ascend_enable_fused_mc2" in s.text for s in sigs))
        self.assertTrue(any("glm-5.1" in s.text for s in sigs))

    def test_structural_layer_stack(self):
        text = (
            'File "/usr/lib/python3.11/site-packages/vllm_ascend/worker/model_runner_v1.py", '
            "line 463, in execute_model\nRuntimeError: aclnnMoeDistributeDispatchV4 failed"
        )
        sigs = extract_signatures(text, symbol_table=self.table)
        self.assertTrue(any(s.kind == "stack_func" and s.text == "execute_model" for s in sigs))
        self.assertTrue(any("model_runner_v1" in s.text for s in sigs))

    def test_structural_layer_kv_and_errcode(self):
        text = "drvRetCode=6 kernel_name=DispatchFFNCombine error code is 561000"
        sigs = extract_signatures(text, symbol_table=self.table)
        self.assertTrue(any(s.kind == "errcode" and s.text == "561000" for s in sigs))

    def test_signal_words_passthrough(self):
        """信号词只透传标注（agent 判断），不参与过滤。"""
        text = "MTP speculative decoding crash"
        sigs = extract_signatures(text, symbol_table=self.table, signal_words=["MTP"])
        self.assertTrue(any(s.kind == "signal" and s.text == "mtp" for s in sigs))

    def test_fallback_regex_without_table(self):
        """无符号表时保留基础正则兜底（API 兼容）。"""
        text = "kernel_name=DispatchFFNCombine aclnnMoeDistributeDispatchV4 error code is 561000"
        sigs = extract_signatures(text)
        self.assertTrue(any("dispatchffncombine" in s.text for s in sigs))
        self.assertTrue(any(s.kind == "errcode" for s in sigs))

    def test_dedupe_keeps_longer(self):
        sigs = [
            Signature(text="trap error", kind="phrase"),
            Signature(text="timeout or trap error", kind="phrase"),
            Signature(text="trap error", kind="phrase"),
        ]
        out = _dedupe(sigs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "timeout or trap error")


if __name__ == "__main__":
    unittest.main()
