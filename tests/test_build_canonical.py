"""build_canonical.py 测试：只再生 canonical（不入库、不拉取），从临时 raw 生成 canonical.jsonl。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vllm_kb.models import KbDocument

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build_canonical.py"


def _item(number, title="t"):
    return {
        "number": number, "title": title, "body": "hello",
        "labels": [{"name": "bug"}], "state": "closed",
        "created_at": "2026-07-01T00:00:00Z", "closed_at": "2026-08-01T00:00:00Z",
        "html_url": f"https://github.com/vllm-project/vllm/issues/{number}",
        "user": {"login": "alice"},
    }


class TestBuildCanonical(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # 临时 raw（github 源）
        raw = self.root / "raw"
        (raw / "issues").mkdir(parents=True)
        (raw / "issues" / "1.json").write_text(json.dumps(_item(1)), encoding="utf-8")
        (raw / "issues" / "2.json").write_text(json.dumps(_item(2, title="second")), encoding="utf-8")
        self.canon = self.root / "canonical.jsonl"
        cfg = {
            "project": {"name": "test", "data_root": "data"},
            "sources": [{
                "id": "vllm-test", "type": "github", "repo": "vllm-project/vllm",
                "token": "", "token_env": "GITHUB_TOKEN",
                "raw_dir": str(raw), "checkpoint_file": str(self.root / "cp.json"),
            }],
            "embedding": {"provider": "echo", "dimensions": 1024, "batch_size": 4},
            "chunking": {"max_chunk_chars": 3000, "overlap_chars": 100},
            "storage": {
                "vector_backend": "python",
                "lancedb_path": str(self.root / "vec.json"),
                "sqlite_path": str(self.root / "kb.sqlite3"),
                "canonical_file": str(self.canon),
            },
        }
        self.cfg_file = self.root / "config.json"
        self.cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--config", str(self.cfg_file)],
            capture_output=True, text=True, encoding="utf-8",
        )

    def test_regenerates_canonical_from_raw(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("新增 2", r.stdout)
        lines = [l for l in self.canon.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        docs = [KbDocument.model_validate_json(l) for l in lines]
        by_id = {d.source_id: d for d in docs}
        self.assertIn("github:vllm-project-vllm:issue:1", by_id)
        self.assertIn("github:vllm-project-vllm:issue:2", by_id)
        d1 = by_id["github:vllm-project-vllm:issue:1"]
        self.assertEqual(d1.status, "closed")
        self.assertEqual(d1.body, "hello")
        self.assertEqual(d1.labels, ["bug"])
        self.assertEqual(d1.component, "vllm")  # 仓库推断主组件
        self.assertEqual(d1.extra["repo"], "vllm-project/vllm")

    def test_idempotent_rerun(self):
        self.assertEqual(self._run().returncode, 0)
        r2 = self._run()
        self.assertEqual(r2.returncode, 0, r2.stderr)
        lines = [l for l in self.canon.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)  # upsert 幂等，不重复追加


if __name__ == "__main__":
    unittest.main()
