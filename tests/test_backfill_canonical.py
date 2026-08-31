"""backfill_canonical.py 测试：kb 有、canonical 缺失 → 回填（body 拼回 / extra 保留 / tags 快照）。

不依赖外部 API 与网络；用 ingest._SCHEMA 建临时 kb.sqlite3。
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vllm_kb.config import AppConfig
from vllm_kb.ingest import _SCHEMA
from vllm_kb.models import KbDocument

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "backfill_canonical.py"

A_ID = "github:vllm-project-test:pr:1"
B_ID = "github:vllm-project-test:pr:2"


def _cfg_dict(tmp: Path) -> dict:
    return {
        "project": {"name": "test", "data_root": "data"},
        "github": {"token": "", "token_env": "GITHUB_TOKEN"},
        "embedding": {"provider": "echo", "dimensions": 1024, "batch_size": 4},
        "chunking": {"max_chunk_chars": 3000, "overlap_chars": 100},
        "storage": {
            "vector_backend": "python",
            "lancedb_path": str(tmp / "vec.json"),
            "sqlite_path": str(tmp / "kb.sqlite3"),
            "canonical_file": str(tmp / "canonical.jsonl"),
        },
        "retrieval": {"final_top_k": 5, "vector_top_k": 20, "fts_top_k": 20},
    }


class TestBackfillCanonical(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.kb = self.root / "kb.sqlite3"
        self.canon = self.root / "canonical.jsonl"
        self.cfg_file = self.root / "config.json"
        self.cfg_file.write_text(json.dumps(_cfg_dict(self.root)), encoding="utf-8")
        self._build_kb()

    def tearDown(self):
        self.tmp.cleanup()

    def _build_kb(self):
        import sqlite3

        conn = sqlite3.connect(str(self.kb))
        conn.executescript(_SCHEMA)
        # 文档 A：canonical 已有；文档 B：canonical 缺失（待回填）
        for sid, title, status in ((A_ID, "A title", "closed"), (B_ID, "B title", "merged")):
            conn.execute(
                "INSERT INTO docs (source_id, source_type, url, title, created_at, resolved_at, "
                "status, labels, version_span_min, version_span_max, reliability, component, "
                "content_hash, embed_hash, extra, tags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, "github_pr", f"https://github.com/test/{sid}", title,
                 "2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z", status,
                 json.dumps(["bug"]), "0.20.0", "0.20.2", 0.9, "vllm-ascend",
                 "h1", "h2",
                 json.dumps({"repo": "vllm-project/vllm-ascend", "github_number": 1,
                             "merged": True, "merged_at": "2026-06-01T00:00:00Z"}),
                 "[]"),
            )
        # chunks：B 两条（body 拼回验证）
        chunks = {
            A_ID: ["A part1"],
            B_ID: ["B part1", "B part2"],
        }
        for sid, parts in chunks.items():
            for seq, text in enumerate(parts):
                conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, doc_id, indexed_text, text) VALUES (?,?,?,?)",
                    (f"{sid}#{seq}", sid, text, text),
                )
                conn.execute(
                    "INSERT INTO chunks_meta (chunk_id, doc_id, seq, section) VALUES (?,?,?,?)",
                    (f"{sid}#{seq}", sid, seq, ""),
                )
        # doc_tags：B 有 auto_snapshot（canonical.tags 应取快照而非 docs.tags）
        conn.execute(
            "INSERT INTO doc_tags (source_id, auto_snapshot, excluded, manual, updated_at, reviewer) "
            "VALUES (?,?,?,?,?,?)",
            (B_ID, json.dumps(["auto-b"]), "[]", "[]", "", "tester"),
        )
        conn.commit()
        conn.close()

        # canonical 只含 A
        doc_a = KbDocument(
            source_type="github_pr", source_id=A_ID, url="", title="A title", body="A part1",
            status="closed", component="vllm-ascend",
            extra={"repo": "vllm-project/vllm-ascend", "github_number": 1},
        )
        self.canon.write_text(doc_a.model_dump_json() + "\n", encoding="utf-8")

    def _run(self, *extra) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--config", str(self.cfg_file), *extra],
            capture_output=True, text=True, encoding="utf-8",
        )

    def test_dry_run_lists_missing_without_writing(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(B_ID, r.stdout)
        # 未 --write：canonical 不变（仍只有 A）
        lines = [l for l in self.canon.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)

    def test_write_backfills_missing_doc(self):
        r = self._run("--write")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("已回填 1 条", r.stdout)
        lines = [l for l in self.canon.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        back = next(json.loads(l) for l in lines if json.loads(l)["source_id"] == B_ID)
        # body 由 chunks 按 seq 拼回（"\n\n".join）
        self.assertEqual(back["body"], "B part1\n\nB part2")
        # 关键字段还原
        self.assertEqual(back["source_type"], "github_pr")
        self.assertEqual(back["status"], "merged")
        self.assertEqual(back["resolved_at"], "2026-06-01T00:00:00Z")
        self.assertEqual(back["component"], "vllm-ascend")
        self.assertEqual(back["version_span"], {"min": "0.20.0", "max": "0.20.2"})
        # extra 原样带出（图构建依赖 repo/github_number/merged_at）
        self.assertEqual(back["extra"]["repo"], "vllm-project/vllm-ascend")
        self.assertEqual(back["extra"]["merged_at"], "2026-06-01T00:00:00Z")
        # tags 取 doc_tags.auto_snapshot（canonical 语义 = 自动标签）
        self.assertEqual(back["tags"], ["auto-b"])
        # 可被 KbDocument 解析（与 load_canonical 兼容）
        KbDocument.model_validate(back)

    def test_idempotent_rerun(self):
        self.assertEqual(self._run("--write").returncode, 0)
        r2 = self._run("--write")
        self.assertEqual(r2.returncode, 0, r2.stderr)
        lines = [l for l in self.canon.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)  # 不重复追加

    def test_doc_filter_only_backfills_target(self):
        r = self._run("--doc", B_ID, "--write")
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = [l for l in self.canon.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
