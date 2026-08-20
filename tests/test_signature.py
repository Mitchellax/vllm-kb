"""签名提取与精确检索测试（不触网）：规则提取 + FTS 精确命中聚合。"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from vllm_kb.signature import extract_signatures, format_hits, format_signatures, signature_search


class TestSignatureExtract(unittest.TestCase):
    def test_kernel_and_errcode(self):
        text = (
            "halMemCreate failed drvRetCode=6, driver error: out of memory, "
            "kernel_name=DispatchFFNCombine, extend info errorStr: timeout or trap error"
        )
        sigs = extract_signatures(text)
        kinds = {s.kind: {s.text for s in sigs} for s in sigs}
        self.assertIn("dispatchffncombine", kinds.get("kernel", set()))
        self.assertIn("6", kinds.get("errcode", set()))
        self.assertTrue(any("timeout or trap error" in s.text for s in sigs))
        self.assertTrue(any("out of memory" in s.text for s in sigs))

    def test_acl_errno(self):
        text = "RuntimeError: aclnnMoeDistributeDispatchV4 failed, error code is 561000"
        sigs = extract_signatures(text)
        self.assertTrue(any(s.kind == "errcode" and s.text == "561000" for s in sigs))
        self.assertTrue(any("aclnnmoe" in s.text or "aclnnMoeDistributeDispatchV4".lower() in s.text for s in sigs))

    def test_env_and_model(self):
        text = "VLLM_ASCEND_ENABLE_FUSED_MC2=1 GLM-5.1 w8a8 PD分离 部署"
        sigs = extract_signatures(text)
        texts = {s.text for s in sigs}
        self.assertIn("vllm_ascend_enable_fused_mc2", texts)
        self.assertTrue(any("glm-5.1" in t for t in texts))

    def test_dedupe(self):
        text = "kernel_name=DispatchFFNCombine kernel_name=DispatchFFNCombine"
        sigs = extract_signatures(text)
        self.assertEqual(sum(1 for s in sigs if s.text == "dispatchffncombine"), 1)

    def test_empty(self):
        self.assertEqual(extract_signatures(""), [])


def make_db(tmp: Path) -> sqlite3.Connection:
    """建最小 chunks_fts + docs 用于精确检索测试。"""
    conn = sqlite3.connect(str(tmp / "t.sqlite3"))
    conn.executescript(
        """
        CREATE TABLE docs (source_id TEXT PRIMARY KEY, title TEXT, url TEXT, component TEXT);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, doc_id UNINDEXED, text);
        """
    )
    docs = [
        ("github:vllm-project-vllm-ascend:issue:1001", "DispatchFFNCombine kernel crash",
         "https://github.com/vllm-project/vllm-ascend/issues/1001", "vllm-ascend"),
        ("github:vllm-project-vllm-ascend:issue:1002", "halMemCreate OOM in workspace",
         "https://github.com/vllm-project/vllm-ascend/issues/1002", "vllm-ascend"),
        ("github:vllm-project-vllm:issue:3001", "CUDA illegal memory access",
         "https://github.com/vllm-project/vllm/issues/3001", "vllm"),
    ]
    conn.executemany(
        "INSERT INTO docs VALUES (?,?,?,?)", docs
    )
    conn.executemany(
        "INSERT INTO chunks_fts VALUES (?,?,?)",
        [
            ("c1", docs[0][0], "kernel_name=DispatchFFNCombine timeout or trap error"),
            ("c2", docs[1][0], "halMemCreate failed drvRetCode=6 out of memory workspace"),
            ("c3", docs[2][0], "CUDA error illegal memory access"),
        ],
    )
    conn.commit()
    return conn


class TestSignatureSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = make_db(Path(self.tmp.name))

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_exact_hit_aggregates(self):
        text = "kernel_name=DispatchFFNCombine errorStr: timeout or trap error"
        sigs = extract_signatures(text)
        hits = signature_search(self.conn, sigs, top_k=5)
        self.assertTrue(hits)
        top = hits[0]
        self.assertIn("1001", top.doc_id)
        self.assertIn("dispatchffncombine", top.hit_signatures)

    def test_component_filter(self):
        text = "halMemCreate failed drvRetCode=6"
        sigs = extract_signatures(text)
        hits = signature_search(self.conn, sigs, component="vllm-ascend")
        self.assertTrue(all("vllm-ascend" in h.doc_id for h in hits))

    def test_no_sigs_no_hits(self):
        self.assertEqual(signature_search(self.conn, []), [])


if __name__ == "__main__":
    unittest.main()
