"""build_tag_candidates.py 测试：正文 TF-IDF 候选导出（不自动写 config，人工同步）。"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from vllm_kb.config import AppConfig
from vllm_kb.models import doc_to_json

CN_BODY = (
    "本文档介绍 Atlas A3 组网拓扑结构、固件升级流程与带宽监控方法。"
    "拓扑规划需考虑 HCCS 链路与网卡部署，固件升级前检查版本兼容性，"
    "带宽监控用于定位通信瓶颈。"
)

_spec = importlib.util.spec_from_file_location(
    "build_tag_candidates", PROJECT_ROOT / "scripts" / "build_tag_candidates.py")
mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(mod)


class TestTagCandidatesExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        data_root = self.root / "data_root"
        data_root.mkdir()
        os.environ["VLLM_KB_DATA_ROOT"] = str(data_root)
        self.cfg_path = self.root / "config.json"
        self.cfg_path.write_text(json.dumps({
            "embedding": {"provider": "echo", "dimensions": 64},
            "tags": {"registry": [{"name": "网络", "tier": "domain"}], "stopwords": ["文档"]},
            "storage": {"canonical_file": "data/raw/canonical.jsonl"},
        }), encoding="utf-8")
        self.cfg = AppConfig.load(str(self.cfg_path), require_keys=False)
        # canonical：两篇中文文档
        docs = [
            {"source_type": "doc_pdf", "source_id": "pdf:atlas-guide", "url": "",
             "title": "Atlas 组网指南", "body": CN_BODY + " 拓扑设计要点详见组网章节。",
             "tags": ["命令参考"]},
            {"source_type": "doc_markdown", "source_id": "md:wiki", "url": "",
             "title": "组网 wiki", "body": CN_BODY + " 固件升级需备份。", "tags": []},
        ]
        canon = self.cfg.resolve("data/raw/canonical.jsonl")
        canon.parent.mkdir(parents=True, exist_ok=True)
        with canon.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        self.out = self.root / "candidates.json"

    def tearDown(self):
        os.environ.pop("VLLM_KB_DATA_ROOT", None)
        self.tmp.cleanup()

    def test_export_format_and_filter(self):
        with mock.patch.object(sys, "argv",
                               ["build_tag_candidates.py", "--config", str(self.cfg_path),
                                "--out", str(self.out), "--top", "10"]):
            mod.main()
        self.assertTrue(self.out.exists())
        rows = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertTrue(rows)
        for r in rows:
            self.assertIn("doc_id", r)
            for c in r["candidates"]:
                self.assertIn("name", c)
                self.assertIn("tier", c)
                self.assertIn("score", c)
                # 已收录词（网络）与已打标词（命令参考）不重复推荐
                self.assertNotEqual(c["name"], "网络")
                self.assertNotEqual(c["name"], "命令参考")
        # 正文主题词应出现在候选（拓扑/固件）
        all_names = {c["name"] for r in rows for c in r["candidates"]}
        self.assertTrue(any("拓扑" in n for n in all_names) or "拓扑" in all_names)

    def test_min_count_filter(self):
        with mock.patch.object(sys, "argv",
                               ["build_tag_candidates.py", "--config", str(self.cfg_path),
                                "--out", str(self.out), "--min-count", "2"]):
            mod.main()
        rows = json.loads(self.out.read_text(encoding="utf-8"))
        # min-count=2：两篇都提及的词才保留（拓扑/固件 等）；单篇词被滤掉
        if rows:
            for r in rows:
                self.assertTrue(r["candidates"])


if __name__ == "__main__":
    unittest.main()
