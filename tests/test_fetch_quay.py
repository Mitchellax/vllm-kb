"""fetch_quay_tags.py 的纯逻辑测试（不触网）：看护/排除策略与分类。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_quay_tags as fq  # noqa: E402


class TestManagedFilter(unittest.TestCase):
    def test_formal_versions_managed(self):
        for name in ("v0.18.0", "0.18.0", "v0.11.0", "v0.7.3", "v0.9.1"):
            self.assertTrue(fq.is_managed(name), name)

    def test_rc_versions_managed(self):
        for name in ("v0.18.0-rc1", "0.18.0.rc1", "v0.18.0rc2", "v0.23.0rc1"):
            self.assertTrue(fq.is_managed(name), name)

    def test_platform_variants_managed(self):
        """版本型 tag 的平台变体（-a3/-310p/-openeuler）同样看护。"""
        for name in ("v0.18.0-a3", "v0.18.0-310p-openeuler", "v0.13.0rc2-a3-openeuler"):
            self.assertTrue(fq.is_managed(name), name)

    def test_model_specific_images_managed(self):
        """模型专属镜像（0day 适配）需要看护。"""
        for name in (
            "glm5", "glm5.2-a3-openeuler", "deepseekv4", "deepseekv4-a3",
            "kimi-k3", "kimi-k3-a3-openeuler", "bailing-flash-arm-a3-openeuler",
            "DeepSeekV4-flash-0731", "v0.11.0rc0-a3-deepseek-v3.2-exp",
        ):
            self.assertTrue(fq.is_managed(name), name)

    def test_nightly_and_branch_excluded(self):
        """日构建与仓库分支不看护（内容可能随构建变化，版本匹配会误导）。"""
        for name in (
            "nightly-main", "nightly-main-a3-openeuler", "ntightly",
            "nightly-releases-v0.18.0", "nightly-releases-v0.23.0-a5-openeuler",
            "releases-v0.13.0", "releases-v0.13.0-a3-openeuler",
        ):
            self.assertFalse(fq.is_managed(name), name)

    def test_trunk_and_dev_excluded(self):
        for name in ("latest", "main", "main-310p", "main-a3-openeuler", "develop",
                     "v0.11.0-dev", "v0.11.0-dev-a3-openeuler"):
            self.assertFalse(fq.is_managed(name), name)


class TestExcludeCategory(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(fq.categorize_excluded("nightly-main-a3"), "nightly(日构建)")
        self.assertEqual(fq.categorize_excluded("ntightly"), "nightly(日构建)")
        self.assertEqual(fq.categorize_excluded("releases-v0.13.0"), "branch(releases-*)")
        self.assertEqual(fq.categorize_excluded("nightly-releases-v0.18.0"), "nightly(日构建)")
        self.assertEqual(fq.categorize_excluded("latest"), "主干/最新")
        self.assertEqual(fq.categorize_excluded("main-310p"), "主干/最新")
        self.assertEqual(fq.categorize_excluded("v0.11.0-dev"), "dev构建")


if __name__ == "__main__":
    unittest.main()
