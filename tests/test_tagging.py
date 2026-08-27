"""文档级标签（两级分类）提取核心测试：词典、标题提取、tier 判定、合并公式。"""
import json
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig
from vllm_kb.tagging import (
    TIER_DOMAIN,
    TIER_PURPOSE,
    TagEntry,
    TagRegistry,
    extract_tags,
    headings_from_markdown,
    headings_from_pdf,
    merge_final,
    normalize_tag,
    save_registry_to_config,
    tier_for,
)


class TestNormalize(unittest.TestCase):
    def test_strip_wrappers(self):
        self.assertEqual(normalize_tag("  HCCL "), "HCCL")
        self.assertEqual(normalize_tag('"超时排查"'), "超时排查")
        self.assertEqual(normalize_tag("`命令参考`"), "命令参考")
        self.assertEqual(normalize_tag("  a   b "), "a b")

    def test_empty(self):
        self.assertEqual(normalize_tag(""), "")
        self.assertEqual(normalize_tag("   "), "")


class TestTierHeuristic(unittest.TestCase):
    def test_purpose_signal_words(self):
        for name in ("超时排查", "命令参考", "错误码表", "安装部署", "性能调优", "故障诊断", "升级指南"):
            self.assertEqual(tier_for(name), TIER_PURPOSE, name)

    def test_domain_default(self):
        for name in ("HCCL", "NPU", "网络", "算子", "CANN"):
            self.assertEqual(tier_for(name), TIER_DOMAIN, name)


class TestHeadings(unittest.TestCase):
    def test_pdf_numbered_headings(self):
        text = (
            "目录\n1 用户指南....5\n2.34 获取network版本号信息\n2.35 命令格式\n"
            "正文段落\n错误码 107020 memory allocation failed"
        )
        hs = headings_from_pdf(text)
        self.assertIn("获取network版本号信息", hs)
        self.assertIn("命令格式", hs)
        # 目录点线填充行不提取
        self.assertNotIn("用户指南", hs)

    def test_markdown_headings(self):
        text = "# GLM5.1 崩溃三板斧\n\n## 排查超时问题\n### [命令参考](cmd.md)\n正文\n## `错误码表`\n"
        hs = headings_from_markdown(text)
        self.assertEqual(hs, ["GLM5.1 崩溃三板斧", "排查超时问题", "命令参考", "错误码表"])


class TestRegistry(unittest.TestCase):
    def test_load_accepts_dict_and_str(self):
        cfg = AppConfig.model_validate({"tags": {"registry": [
            {"name": "HCCL", "tier": "domain"},
            {"name": "超时排查", "tier": "purpose"},
            "裸字符串",
        ]}})
        r = TagRegistry.load(cfg)
        self.assertEqual(r.tier("HCCL"), TIER_DOMAIN)
        self.assertEqual(r.tier("超时排查"), TIER_PURPOSE)
        # 裸字符串 tier 走启发式（无信号词 → domain）
        self.assertEqual(r.tier("裸字符串"), TIER_DOMAIN)
        self.assertTrue(r.contains("HCCL"))

    def test_load_missing_tags_ok(self):
        cfg = AppConfig.model_validate({})
        r = TagRegistry.load(cfg)
        self.assertEqual(r.entries, [])

    def test_match_substring_case_insensitive(self):
        r = TagRegistry(entries=[TagEntry("HCCL", TIER_DOMAIN), TagEntry("超时排查", TIER_PURPOSE)])
        hits = r.match("HCCL超时排查指南")
        self.assertEqual({h.name for h in hits}, {"HCCL", "超时排查"})

    def test_add_rename_set_tier_remove(self):
        r = TagRegistry()
        r.add("HCCL")
        self.assertTrue(r.contains("HCCL"))
        self.assertEqual(r.tier("HCCL"), TIER_DOMAIN)  # 无信号词 → domain
        r.add("超时排查", TIER_PURPOSE)
        self.assertEqual(r.tier("超时排查"), TIER_PURPOSE)
        r.rename("超时排查", "超时定位")
        self.assertFalse(r.contains("超时排查"))
        self.assertTrue(r.contains("超时定位"))
        r.set_tier("HCCL", TIER_PURPOSE)
        self.assertEqual(r.tier("HCCL"), TIER_PURPOSE)
        self.assertTrue(r.remove("HCCL"))
        self.assertFalse(r.remove("HCCL"))

    def test_save_registry_to_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.json"
            p.write_text(json.dumps({"project": {"name": "x"}}), encoding="utf-8")
            r = TagRegistry(entries=[TagEntry("HCCL", TIER_DOMAIN)])
            save_registry_to_config(AppConfig.model_validate({}), r, config_path=p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["tags"]["registry"], [{"name": "HCCL", "tier": "domain"}])


class TestExtractTags(unittest.TestCase):
    def test_registry_hit_and_latin_tokens(self):
        r = TagRegistry(entries=[TagEntry("HCCL", TIER_DOMAIN), TagEntry("超时排查", TIER_PURPOSE)])
        tags, cands = extract_tags(
            "HCCL超时排查指南",
            ["2.34 排查超时问题", "获取network版本号信息"],
            registry=r,
            stopwords=["guide", "manual"],
        )
        names = [t.name for t in tags]
        self.assertIn("HCCL", names)
        self.assertIn("超时排查", names)
        cand_names = [c.name for c in cands]
        # 拉丁 token（network）进候选；guide/manual 停用词过滤
        self.assertIn("network", cand_names)
        self.assertNotIn("guide", cand_names)
        self.assertNotIn("manual", cand_names)

    def test_version_tokens_dropped(self):
        _, cands = extract_tags("guide_v1.0", ["0.23.0rc1 说明"])
        self.assertNotIn("v1.0", [c.name for c in cands])
        self.assertNotIn("0.23.0rc1", [c.name for c in cands])

    def test_short_heading_candidate(self):
        _, cands = extract_tags("doc", ["错误码表"], heading_max_chars=12)
        self.assertIn("错误码表", [c.name for c in cands])
        # 超长标题不直接作候选
        _, cands2 = extract_tags("doc", ["获取network版本号信息的完整流程说明"], heading_max_chars=12)
        self.assertNotIn("获取network版本号信息的完整流程说明", [c.name for c in cands2])

    def test_min_len(self):
        _, cands = extract_tags("a b cd", [], min_len=2)
        names = [c.name for c in cands]
        self.assertNotIn("a", names)
        self.assertNotIn("b", names)
        self.assertIn("cd", names)


class TestMergeFinal(unittest.TestCase):
    def test_union_and_exclude(self):
        self.assertEqual(
            merge_final(["HCCL", "网络", "超时排查"], ["超时排查"], ["命令参考"]),
            ["HCCL", "网络", "命令参考"],
        )

    def test_manual_priority_and_dedupe(self):
        # 人工添加的标签即使被排除也保留（人工优先）；同名不重复
        self.assertEqual(
            merge_final(["HCCL", "超时排查"], ["超时排查"], ["超时排查", "命令参考"]),
            ["HCCL", "超时排查", "命令参考"],
        )

    def test_empty(self):
        self.assertEqual(merge_final([], [], []), [])
        self.assertEqual(merge_final(None, None, None), [])


if __name__ == "__main__":
    unittest.main()
