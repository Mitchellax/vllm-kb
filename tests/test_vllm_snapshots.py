"""build_vllm_snapshots.py 纯逻辑测试（不触网）：companion 矩阵的 vllm 版本提取与排序。

回归背景（实测）：矩阵里的 vllm 值可能带预发布/后缀（镜像 VLLM_TAG 就是 `0.11.0rc3` 这种），
旧实现 `sorted(versions, key=lambda s: [int(x) for x in s.split(".")])` 直接
`ValueError: invalid literal for int() with base 10: '0rc3'`，把
"代码快照（vllm 主仓，companion 对应）"整步打挂（非致命 → 静默降级，快照没生成）。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_vllm_snapshots as bvs  # noqa: E402

from vllm_kb.config import AppConfig  # noqa: E402


def _cfg(tmp: Path, rows: list[dict]) -> AppConfig:
    """构造只带 companion_file 的配置（绝对路径 → resolve 原样返回）。"""
    matrix = tmp / "vllm-ascend.json"
    matrix.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    cfg = AppConfig.model_validate({
        "embedding": {"provider": "echo", "dimensions": 8},
        "sources": [],
        "storage": {"companion_file": str(matrix)},
    })
    return cfg


class TestVersionSortKey(unittest.TestCase):
    def test_rc_and_plain_and_post_are_ordered(self):
        versions = ["0.23.0.post1", "0.11.0rc3", "0.23.0", "0.11.0", "0.9.2rc1"]
        self.assertEqual(
            sorted(versions, key=bvs.version_sort_key),
            ["0.9.2rc1", "0.11.0rc3", "0.11.0", "0.23.0", "0.23.0.post1"],
        )

    def test_rc_tail_number_is_numeric(self):
        """rc3 < rc10（按数值比，不是字符串：'rc10' < 'rc3' 是错的）。"""
        self.assertEqual(sorted(["0.11.0rc10", "0.11.0rc3"], key=bvs.version_sort_key),
                         ["0.11.0rc3", "0.11.0rc10"])

    def test_unknown_value_sorts_last_without_crash(self):
        out = sorted(["latest", "0.11.0rc3", "", "0.9.0"], key=bvs.version_sort_key)
        self.assertEqual(out[-1], "latest")
        self.assertEqual(out[:2], ["0.9.0", "0.11.0rc3"])

    def test_leading_v_is_tolerated(self):
        self.assertEqual(sorted(["v0.11.0rc3", "v0.9.0"], key=bvs.version_sort_key),
                         ["v0.9.0", "v0.11.0rc3"])


class TestCompanionVllmVersions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_suffix_versions_do_not_crash(self):
        """核心回归：矩阵里有 rc 版本时不再 ValueError。"""
        cfg = _cfg(self.root, [
            {"vllm-ascend": "v0.11.0rc0", "vllm": "0.11.0rc3"},
            {"vllm-ascend": "v0.23.0", "vllm": "0.23.0"},
            {"vllm-ascend": "v0.23.0.post1", "vllm": "0.23.0.post1"},
        ])
        self.assertEqual(bvs.companion_vllm_versions(cfg),
                         ["0.11.0rc3", "0.23.0", "0.23.0.post1"])

    def test_dedupes_and_skips_incomplete_rows(self):
        cfg = _cfg(self.root, [
            {"vllm-ascend": "v0.23.0", "vllm": "0.23.0"},
            {"vllm-ascend": "v0.23.0rc1", "vllm": "0.23.0"},          # 去重
            {"vllm-ascend": "glm5.2", "vllm": "0.23.0"},              # 非 vX.Y 主键 → 跳过
            {"vllm-ascend": "v0.26.0rc1", "vllm": ""},                # 无 vllm 值 → 跳过
            {"vllm-ascend": "", "vllm": "0.27.1"},                    # 无主键 → 跳过
        ])
        self.assertEqual(bvs.companion_vllm_versions(cfg), ["0.23.0"])

    def test_missing_file_returns_empty(self):
        cfg = AppConfig.model_validate({
            "embedding": {"provider": "echo", "dimensions": 8},
            "sources": [],
            "storage": {"companion_file": str(self.root / "nope.json")},
        })
        self.assertEqual(bvs.companion_vllm_versions(cfg), [])


if __name__ == "__main__":
    unittest.main()
