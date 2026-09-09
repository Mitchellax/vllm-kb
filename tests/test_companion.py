"""组件配套矩阵与组件查询解析测试。"""
import json
import tempfile
import unittest
from pathlib import Path

from vllm_kb.companion import CompanionMatrix, parse_component_query

MATRIX = {
    "rows": [
        {
            "vllm-ascend": "0.18.0",
            "vllm": "0.12.1",
            "cann": "8.1.RC2",
            "pytorch": "2.6.0",
            "pytorch-ascend": "2.6.0.post1",
            "npu-driver": "",
            "notes": "",
        },
        {
            "vllm-ascend": "0.17.0",
            "vllm": "0.11.3",
            "cann": "8.0.RC3",
            "pytorch": "2.5.1",
            "pytorch-ascend": "2.5.1.post1",
            "npu-driver": "",
            "notes": "",
        },
        {
            "vllm-ascend": "0.16.0",
            "vllm": "0.10.1",
            "cann": "8.0.RC2",
            "pytorch": "2.5.1",
            "pytorch-ascend": "2.5.1.post1",
            "npu-driver": "",
            "notes": "",
        },
    ]
}


def make_matrix(tmp: Path) -> CompanionMatrix:
    p = tmp / "matrix.json"
    p.write_text(json.dumps(MATRIX), encoding="utf-8")
    return CompanionMatrix.load(p)


class TestParseComponentQuery(unittest.TestCase):
    def test_parse_component_prefix(self):
        comp, ver, rest = parse_component_query("vllm-ascend:0.18.0 GLM5.1 PD分离P节点挂死")
        self.assertEqual(comp, "vllm-ascend")
        self.assertEqual(ver, "0.18.0")
        self.assertEqual(rest, "GLM5.1 PD分离P节点挂死")

    def test_parse_plain_query(self):
        comp, ver, rest = parse_component_query("CUDA illegal memory access")
        self.assertIsNone(comp)
        self.assertIsNone(ver)
        self.assertEqual(rest, "CUDA illegal memory access")

    def test_parse_vllm_component(self):
        comp, ver, rest = parse_component_query("vllm:0.26.0 openai server 500")
        self.assertEqual(comp, "vllm")
        self.assertEqual(ver, "0.26.0")

    def test_parse_model_image_as_version(self):
        """模型专属镜像名也可作为版本（0day 适配镜像的查询）。"""
        comp, ver, rest = parse_component_query("vllm-ascend:glm5 GLM5 长稳压测挂死")
        self.assertEqual(comp, "vllm-ascend")
        self.assertEqual(ver, "glm5")
        self.assertEqual(rest, "GLM5 长稳压测挂死")

    def test_parse_plain_query_with_colon_unaffected(self):
        """普通查询里含冒号（如错误码）不应误解析为组件查询。"""
        comp, ver, rest = parse_component_query("PD分离: 挂死 如何处理")
        self.assertIsNone(comp)
        self.assertIsNone(ver)


class TestCompanionMatrix(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.matrix = make_matrix(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_get_exact(self):
        rows = self.matrix.get("vllm-ascend", "0.18.0")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].vllm, "0.12.1")

    def test_get_v_prefix_tolerant(self):
        self.assertEqual(len(self.matrix.get("vllm-ascend", "v0.18.0")), 1)

    def test_nearest(self):
        rows = self.matrix.nearest("vllm-ascend", "0.17.5", k=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].vllm_ascend, "0.17.0")  # 距离最近

    def test_expand_reverse(self):
        out = self.matrix.expand("vllm-ascend", "0.18.0")
        self.assertEqual(out.get("vllm"), ["0.12.1"])
        self.assertEqual(out.get("cann"), ["8.1.RC2"])
        self.assertEqual(out.get("pytorch"), ["2.6.0"])
        self.assertNotIn("vllm-ascend", out)  # 自身不出现在配套里

    def test_expand_nearest_when_no_exact(self):
        out = self.matrix.expand("vllm-ascend", "0.17.5")
        self.assertIn("vllm", out)
        self.assertIn("0.11.3", out["vllm"])  # 最近版本 0.17.0 的配套

    def test_expand_by_other_component(self):
        """反向：从 vllm 版本出发也能查到（多组件多版本存储支持）。"""
        out = self.matrix.expand("vllm", "0.12.1")
        self.assertEqual(out.get("vllm-ascend"), ["0.18.0"])

    def test_missing_file_returns_none(self):
        self.assertIsNone(CompanionMatrix.load(Path("no/such/file.json")))
        self.assertIsNone(CompanionMatrix.load(""))


class TestCommitTraceFields(unittest.TestCase):
    """commit 溯源字段（image_created / vllm_commit / fork 字段）随 /matrix 暴露给 agent。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rows, generated_at="2026-09-09T00:00:00+00:00"):
        p = self.dir / "m.json"
        p.write_text(json.dumps({"generated_at": generated_at, "rows": rows}), encoding="utf-8")
        return CompanionMatrix.load(p)

    def test_official_row_commit_fields_survive(self):
        m = self._write([{
            "vllm-ascend": "kimi-k3", "vllm": "0.23.0", "cann": "9.0.1",
            "pytorch": "2.10.0", "pytorch-ascend": "2.10.0.post4", "npu-driver": "",
            "image_created": "2026-07-27T15:39:42Z",
            "vllm_commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
            "vllm_commit_date": "2026-06-15T03:35:17Z",
            "notes": "", "source": "自动",
        }])
        row = m.rows[0]
        self.assertEqual(row.image_created, "2026-07-27T15:39:42Z")
        self.assertEqual(row.vllm_commit[:12], "0fc695fc6d1d")
        self.assertEqual(row.vllm_commit_date, "2026-06-15T03:35:17Z")
        dump = row.model_dump(by_alias=True)   # /matrix 端点用的就是它
        for k in ("image_created", "vllm_commit", "vllm_commit_date"):
            self.assertIn(k, dump)
            self.assertTrue(dump[k])

    def test_fork_row_fields_survive(self):
        m = self._write([{
            "vllm-ascend": "glm5.2", "vllm": "0.23.0", "cann": "9.0.0",
            "pytorch": "", "pytorch-ascend": "", "npu-driver": "",
            "vllm_repo": "ZYang6263/vllm", "vllm_ref": "glm52", "vllm_base": "0.23.0",
            "vllm_sha": "1" * 40, "image_digest": "sha256:" + "a" * 64,
            "image_created": "2026-07-27T15:39:42Z",
            "vllm_commit_date": "2026-07-20T01:02:03Z",
            "notes": "", "source": "自动",
        }])
        row = m.rows[0]
        self.assertEqual(row.vllm_repo, "ZYang6263/vllm")
        self.assertEqual(row.vllm_sha, "1" * 40)
        self.assertEqual(row.image_digest, "sha256:" + "a" * 64)
        self.assertEqual(row.vllm_commit_date, "2026-07-20T01:02:03Z")

    def test_generated_at_exposed(self):
        m = self._write([{"vllm-ascend": "0.18.0", "vllm": "0.12.1", "cann": "8.1.RC2",
                          "pytorch": "2.6.0", "pytorch-ascend": "2.6.0.post1"}],
                        generated_at="2026-09-09T12:34:56+00:00")
        self.assertEqual(m.generated_at, "2026-09-09T12:34:56+00:00")

    def test_old_matrix_without_new_fields_ok(self):
        m = self._write([{"vllm-ascend": "0.18.0", "vllm": "0.12.1", "cann": "8.1.RC2",
                          "pytorch": "2.6.0", "pytorch-ascend": "2.6.0.post1"}])
        row = m.rows[0]
        self.assertEqual(row.image_created, "")
        self.assertEqual(row.vllm_commit, "")
        self.assertEqual(row.vllm_commit_date, "")


if __name__ == "__main__":
    unittest.main()
