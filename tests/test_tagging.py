"""文档级标签（两级分类）提取核心测试：词典、标题提取、tier 判定、合并公式。"""
import json
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig
from vllm_kb.tagging import (
    TIER_DOMAIN,
    TIER_PURPOSE,
    DeviceGlossary,
    TagEntry,
    TagRegistry,
    extract_tags,
    headings_from_markdown,
    headings_from_pdf,
    merge_final,
    normalize_tag,
    save_registry_to_config,
    tag_key,
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

    def test_cjk_latin_boundary_space_stripped(self):
        """PDF/Word 标题里 CJK↔拉丁边界空格不稳定——剥离后变体收敛为同一候选词。"""
        self.assertEqual(normalize_tag("配置pip 源"), "配置pip源")
        self.assertEqual(normalize_tag("配置 pip源"), "配置pip源")
        self.assertEqual(normalize_tag("升级NPU 驱动"), "升级NPU驱动")
        self.assertEqual(normalize_tag("通过apt-get 升级"), "通过apt-get升级")
        self.assertEqual(normalize_tag("查询NPU 的详细信息"), "查询NPU的详细信息")

    def test_cjk_cjk_space_kept(self):
        """纯 CJK 间/纯拉丁间的空格不属于不稳定边界，保留。"""
        self.assertEqual(normalize_tag("方式二 构建容器镜像"), "方式二 构建容器镜像")
        self.assertEqual(normalize_tag("foo bar"), "foo bar")


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

    def test_space_variant_headings_collapse(self):
        """同一标题的空格变体（不同手册/版本的 PDF 抽取差异）收敛为一个候选词——
        审核队列不再出现 '配置pip源'/'配置pip 源' 两条。"""
        _, cands = extract_tags("doc", ["配置pip源", "配置pip 源", "升级NPU 驱动"])
        names = [c.name for c in cands]
        self.assertEqual(names.count("配置pip源"), 1)
        self.assertNotIn("配置pip 源", names)
        self.assertIn("升级NPU驱动", names)


class TestTagKeyVariants(unittest.TestCase):
    """聚合键 tag_key（2026-10 业务积压：候选大量同义词=同一词的排版变体）——
    大小写/分隔符（空格-连字符-下划线）/CJK 间空格在 key 层收敛；显示与词典 name 保留原形。"""

    def test_variant_groups_converge(self):
        from vllm_kb.tagging import tag_key

        self.assertEqual(tag_key("Atlas A3"), tag_key("atlas_a3"))
        self.assertEqual(tag_key("Atlas A3"), tag_key("ATLAS A3"))
        self.assertEqual(tag_key("Atlas A3"), tag_key("Atlas-A3"))
        self.assertEqual(tag_key("超时 排查"), tag_key("超时排查"))
        self.assertEqual(tag_key("配置pip 源"), tag_key("配置pip源"))
        # 首尾句读剥离（2026-10 业务实测："复位芯片。" 与 "复位芯片" 两个 tag）
        self.assertEqual(tag_key("复位芯片。"), tag_key("复位芯片"))
        self.assertEqual(tag_key("复位芯片."), tag_key("复位芯片"))
        self.assertEqual(tag_key("，网卡故障；"), tag_key("网卡故障"))
        # 版本号的点保留（中间标点不伤）；数字串不合并
        self.assertEqual(tag_key("v0.11.0"), "v0.11.0")
        self.assertEqual(normalize_tag("v0.11.0"), "v0.11.0")
        self.assertNotEqual(tag_key("v0.11.0"), tag_key("v0.12.0"))

    def test_normalize_tag_strips_edge_punct(self):
        self.assertEqual(normalize_tag("复位芯片。"), "复位芯片")
        self.assertEqual(normalize_tag("复位芯片。 "), "复位芯片")
        self.assertEqual(normalize_tag("Atlas A3."), "Atlas A3")
        self.assertEqual(normalize_tag("v0.11.0"), "v0.11.0")     # 中间点保留
        self.assertEqual(normalize_tag("方式二 构建容器镜像"), "方式二 构建容器镜像")

    def test_registry_contains_and_add_by_key(self):
        r = TagRegistry(entries=[TagEntry("Atlas A3", TIER_DOMAIN)])
        self.assertTrue(r.contains("atlas_a3"))       # 变体查重命中
        self.assertTrue(r.contains("ATLAS-A3"))
        # add 同 key → 保留已有原形词条，不产生第二个
        e = r.add("atlas_a3")
        self.assertEqual(e.name, "Atlas A3")
        self.assertEqual(len(r.entries), 1)

    def test_load_converges_variant_entries(self):
        """存量 config 词典里同 key 变体词条（都曾被人工认证入典）——加载即收敛为首条。"""
        cfg = AppConfig.model_validate({"tags": {"registry": [
            {"name": "CANN版本说明", "tier": "purpose"},
            {"name": "CANN 版本说明", "tier": "domain"},   # 变体（后被认证）
        ]}})
        r = TagRegistry.load(cfg)
        self.assertEqual(len(r.entries), 1)
        self.assertEqual(r.entries[0].name, "CANN版本说明")   # 首见原形
        self.assertEqual(r.entries[0].tier, TIER_PURPOSE)     # 首条 tier
        self.assertTrue(r.contains("CANN 版本说明"))

    def test_match_separator_insensitive(self):
        r = TagRegistry(entries=[TagEntry("hccn-tool", TIER_DOMAIN)])
        hits = r.match("执行 hccn_tool 检查")
        self.assertEqual([h.name for h in hits], ["hccn-tool"])

    def test_extract_tags_variant_single_candidate(self):
        # 同一词的两种排版经 heading 路径产出不同 name、同 key——seed 聚合时收敛为一条
        # （latin token 层形态差异是预期：下划线连写 "Atlas_A3" 是一个 token、空格拆两个）
        _, c1 = extract_tags("doc", ["Atlas_A3 部署"])
        _, c2 = extract_tags("doc", ["Atlas A3 部署"])
        self.assertIn(tag_key("Atlas_A3 部署"), {tag_key(c.name) for c in c1})
        self.assertIn(tag_key("Atlas A3 部署"), {tag_key(c.name) for c in c2})
        self.assertEqual(tag_key("Atlas_A3 部署"), tag_key("Atlas A3 部署"))

    def test_cjk_stopword_whole_word_only(self):
        from vllm_kb.tagging import _DEFAULT_CJK_STOPWORDS

        self.assertIn(tag_key("排查"), {tag_key(s) for s in _DEFAULT_CJK_STOPWORDS})
        # 整词等于（heading 短标题）→ 滤；含停用词的短语 → 保留（有价值标签不受影响）
        _, c_alone = extract_tags("doc", ["排查"])
        _, c_phrase = extract_tags("doc", ["HCCL 超时排查"])
        self.assertEqual([c.name for c in c_alone], [])
        self.assertTrue(any("超时排查" in c.name for c in c_phrase))


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


class TestDeviceGlossary(unittest.TestCase):
    """设备词表：机型/芯片别名归一 + 机器→芯片单向（芯片绝不反推机型）。

    领域事实（2026-10 确认）：A2/A3 是 Atlas 机型不是版本号，硬件构造差距很大需独立
    打标；800I A2/800T A2 本质同机归一为 A2；300I A2 与 800I A2 同用 910B 芯片但拓扑
    差异巨大——所以 910B 文本绝不能自动打 A2 标。
    """

    DEVICES = [
        {"name": "A2", "tier": "domain", "kind": "machine",
         "aliases": ["800I A2", "800T A2", "Atlas 800I A2"], "uses_chips": ["910B"],
         "note": "800I/800T 规格细微差异本质同机；300I A2 同芯片但拓扑差异巨大，勿混"},
        {"name": "A3", "tier": "domain", "kind": "machine", "aliases": ["800I A3"]},
        {"name": "300I A2", "tier": "domain", "kind": "machine", "uses_chips": ["910B"]},
        {"name": "910B", "tier": "domain", "kind": "chip", "aliases": ["910B4_1"]},
    ]

    def _registry(self, devices=None):
        import types

        from vllm_kb.config import TagCfg

        return TagRegistry.load(types.SimpleNamespace(
            tags=TagCfg(registry=[], devices=self.DEVICES if devices is None else devices)))

    def test_alias_canonicalization_and_auto_registry(self):
        reg = self._registry()
        names = [e.name for e in reg.match("Atlas 800I A2 组网配置指南")]
        self.assertIn("A2", names)              # 别名 → canonical 设备标签
        self.assertNotIn("800I A2", names)      # 别名本身不是标签
        self.assertTrue(reg.contains("A2"))     # 设备 name 自动入词典（registry 未显式声明）
        self.assertTrue(reg.contains("910B"))

    def test_chip_does_not_infer_machine(self):
        """方向性铁律：910B4_1 芯片文本 → 只打 910B，绝不反推 A2/300I A2。"""
        names = [e.name for e in self._registry().match("910B4_1 芯片故障排查")]
        self.assertIn("910B", names)
        self.assertNotIn("A2", names)
        self.assertNotIn("300I A2", names)

    def test_long_alias_hit_no_duplicate_canonical(self):
        names = [e.name for e in self._registry().match("800I A2")]
        self.assertEqual(names.count("A2"), 1)

    def test_validation_warnings(self):
        raw = [{"name": "A2", "aliases": ["x"]},
               {"name": "B", "aliases": ["x"]},           # 别名冲突（先入先得）
               {"name": "C", "uses_chips": ["未定义芯片"]},  # 引用未定义
               {"name": "D", "kind": "router"},           # kind 非法 → machine
               "junk",                                     # 非 dict
               {"aliases": ["no-name"]}]                   # 缺 name
        g = DeviceGlossary._from_raw(raw)
        self.assertEqual(len(g.warnings), 5)
        self.assertEqual([e.name for e in g.entries], ["A2", "B", "C", "D"])
        self.assertEqual(g.entries[3].kind, "machine")

    def test_device_words_exempt_from_noise_filter(self):
        """A3 是机型词（词表命中即标签），不被候选降噪过滤；别名组成 token 也不进候选。"""
        reg = self._registry()
        matched, cands = extract_tags("Atlas 800I A3 故障案例", [], registry=reg)
        self.assertIn("A3", [e.name for e in matched])
        cand_names = [c.name for c in cands]
        self.assertNotIn("A3", cand_names)
        self.assertNotIn("800I", cand_names)   # 别名 "800I A3" 的组成 token 已归一


class TestCandidateNoiseFilter(unittest.TestCase):
    """候选降噪（2026-10）：真实积压里 for/纯数字类无检索价值——发端过滤而非采纳端。"""

    def test_default_stopwords_pure_numbers_filtered(self):
        _, cands = extract_tags("for of and 123 Config 456", [], registry=TagRegistry())
        names = {c.name for c in cands}
        self.assertNotIn("for", names)
        self.assertNotIn("of", names)
        self.assertNotIn("and", names)
        self.assertNotIn("123", names)      # 纯数字
        self.assertNotIn("456", names)
        self.assertIn("Config", names)      # 正常词保留

    def test_cfg_stopwords_still_mergeable(self):
        _, cands = extract_tags("foo config bar", [], registry=TagRegistry(),
                                stopwords=["config"])
        self.assertNotIn("config", {c.name for c in cands})


if __name__ == "__main__":
    unittest.main()
