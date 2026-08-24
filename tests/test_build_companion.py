"""build_companion_matrix.py 纯逻辑测试（不触网）：去重、env 提取、release 提取、合并、缺口报告。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_companion_matrix as bm  # noqa: E402


class TestStripSuffix(unittest.TestCase):
    def test_platform_suffixes(self):
        cases = {
            "v0.18.0": "v0.18.0",
            "v0.18.0-a3": "v0.18.0",
            "v0.18.0-a3-openeuler": "v0.18.0",
            "v0.18.0-310p-openeuler": "v0.18.0",
            "v0.13.0rc2-a3-openeuler": "v0.13.0rc2",
            "v0.7.3.post1-openeuler": "v0.7.3.post1",
            "glm5.2-a3-openeuler": "glm5.2",
            "kimi-k3-a3": "kimi-k3",
            "DeepSeekV4-flash-0731-a3-openeuler": "DeepSeekV4-flash-0731",
            "bailing-flash-arm-a3-openeuler": "bailing-flash-arm",
        }
        for tag, expected in cases.items():
            self.assertEqual(bm.strip_platform_suffix(tag), expected, tag)


class TestExtractFromEnv(unittest.TestCase):
    def test_cann_soc_python(self):
        env = [
            "ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.1",
            "PATH=/usr/local/Ascend/cann-8.5.1/bin:/usr/local/python3.11.14/bin:/usr/bin",
            "SOC_VERSION=ascend910b1",
        ]
        out = bm.extract_from_env(env)
        self.assertEqual(out["cann"], "8.5.1")
        self.assertEqual(out["soc"], "ascend910b1")
        self.assertEqual(out["python"], "3.11.14")
        self.assertEqual(out["vllm_tag"], "")

    def test_empty_env(self):
        out = bm.extract_from_env([])
        self.assertEqual(out, {"cann": "", "soc": "", "python": "", "vllm_tag": ""})

    def test_vllm_tag_extraction(self):
        # 带 v 前缀的 VLLM_TAG（镜像构建锁定的 vllm 配套版本）
        out = bm.extract_from_env(["VLLM_TAG=v0.26.0", "ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.1"])
        self.assertEqual(out["vllm_tag"], "0.26.0")
        self.assertEqual(out["cann"], "8.5.1")
        # 无 v 前缀
        out2 = bm.extract_from_env(["VLLM_TAG=0.23.0"])
        self.assertEqual(out2["vllm_tag"], "0.23.0")

    def test_vllm_tag_priority_over_release(self):
        """VLLM_TAG（镜像 env）优先于 release 说明作为 vllm 配套版本。"""
        groups = {"deepseekv4-flash-0731": [{"name": "deepseekv4-flash-0731", "manifest_digest": "d"}]}
        releases = {"deepseekv4-flash-0731": "aligns with upstream vLLM v0.22.1"}
        from unittest import mock

        with mock.patch("build_companion_matrix.fetch_image_env", return_value=["VLLM_TAG=v0.26.0"]):
            rows = bm.build_rows(groups, releases, "T")
        self.assertEqual(rows[0]["vllm"], "0.26.0")
        self.assertIn("VLLM_TAG", rows[0]["source"])
        # 无 VLLM_TAG 时回退 release 说明
        with mock.patch("build_companion_matrix.fetch_image_env", return_value=["cann-8.5.1"]):
            rows2 = bm.build_rows(groups, releases, "T")
        self.assertEqual(rows2[0]["vllm"], "0.22.1")

    def test_base_version_key(self):
        self.assertEqual(bm.base_version_key("v0.13.0rc1"), "0.13.0")
        self.assertEqual(bm.base_version_key("v0.13.0"), "0.13.0")
        self.assertEqual(bm.base_version_key("v0.13.0rc3"), "0.13.0")
        self.assertEqual(bm.base_version_key("glm5"), "")
        self.assertEqual(bm.base_version_key("deepseekv4-flash-0731"), "")

    def test_cann_fallback_same_base_series(self):
        """Env 无 cann 的版本（如 v0.13.0rc1）按基础版本号回退同系列其他形态。"""
        from unittest import mock

        groups = {
            "v0.13.0rc1": [{"name": "v0.13.0rc1", "manifest_digest": "d1"}],
            "v0.13.0": [{"name": "v0.13.0", "manifest_digest": "d2"}],
            "v0.13.0rc2": [{"name": "v0.13.0rc2", "manifest_digest": "d3"}],
        }
        envs = {
            "v0.13.0rc1": ["SOC_VERSION=ascend910b1"],  # 无 cann
            "v0.13.0": ["ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.0"],
            "v0.13.0rc2": ["ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.0"],
        }

        def fake_env(tag_info, token, **kw):
            return envs[tag_info["name"]]

        with mock.patch("build_companion_matrix.fetch_image_env", side_effect=fake_env):
            rows = bm.build_rows(groups, {}, "T")
        by_key = {r["vllm-ascend"]: r for r in rows}
        # rc1 从同系列回退得到 cann
        self.assertEqual(by_key["v0.13.0rc1"]["cann"], "8.5.0")
        self.assertIn("同系列回退(0.13.0)", by_key["v0.13.0rc1"]["source"])
        # 有 Env 的版本仍用自己镜像的 cann
        self.assertEqual(by_key["v0.13.0"]["cann"], "8.5.0")
        self.assertIn("镜像env", by_key["v0.13.0"]["source"])
        self.assertEqual(by_key["v0.13.0rc2"]["cann"], "8.5.0")

    def test_cann_missing_all_kept_empty_for_manual(self):
        """cann 同系列也未找到：存空 + source 标注缺失，交人工看护。"""
        from unittest import mock

        groups = {
            "glm5": [{"name": "glm5", "manifest_digest": "d1"}],
            "v0.20.0": [{"name": "v0.20.0", "manifest_digest": "d2"}],
        }

        def fake_env(tag_info, token, **kw):
            return ["SOC_VERSION=ascend910b1"]  # 全部无 cann

        with mock.patch("build_companion_matrix.fetch_image_env", side_effect=fake_env):
            rows = bm.build_rows(groups, {}, "T")
        for r in rows:
            self.assertEqual(r["cann"], "")
            self.assertIn("cann=缺失(待人工)", r["source"])

    def test_validate_version_fields(self):
        valid = [
            {"vllm-ascend": "v0.26.0", "vllm": "0.26.0", "cann": "8.5.1",
             "pytorch": "2.6.0", "pytorch-ascend": "2.6.0.post1", "npu-driver": "25.1.0"},
        ]
        bad = [
            {"vllm-ascend": "v0.26.0", "vllm": "0.26.0", "cann": "cann-8.5.1-恶意",  # 非法
             "pytorch": "latest", "pytorch-ascend": "", "npu-driver": ""},
        ]
        self.assertEqual(bm.validate_version_fields([dict(x) for x in valid]), 0)
        n = bm.validate_version_fields(bad)
        self.assertGreaterEqual(n, 2)
        self.assertEqual(bad[0]["cann"], "")   # 非法置空
        self.assertEqual(bad[0]["pytorch"], "")
        self.assertEqual(bad[0]["vllm"], "0.26.0")  # 合法保留
        # rc 后缀合法
        self.assertTrue(bm._VERSION_VALID_RE.match("0.13.0rc1"))
        self.assertTrue(bm._VERSION_VALID_RE.match("2.6.0.post1"))
        self.assertFalse(bm._VERSION_VALID_RE.match("latest"))
        self.assertFalse(bm._VERSION_VALID_RE.match("cann-8.5.1"))


class TestExtractVllmFromRelease(unittest.TestCase):
    def test_upstream_statement(self):
        body = "This release aligns the plugin with upstream vLLM v0.23.0 and expands model support."
        ver, src = bm.extract_vllm_from_release("v0.23.0rc1", body)
        self.assertEqual(ver, "0.23.0")
        self.assertIn("release", src)

    def test_based_on_statement(self):
        ver, src = bm.extract_vllm_from_release("v0.19.1rc1", "This is based on vLLM v0.19.1.")
        self.assertEqual(ver, "0.19.1")

    def test_heuristic_when_no_release(self):
        ver, src = bm.extract_vllm_from_release("v0.19.1rc1", "")
        self.assertEqual(ver, "0.19.1")
        self.assertIn("启发式", src)

    def test_model_tag_no_vllm(self):
        ver, src = bm.extract_vllm_from_release("glm5", "")
        self.assertEqual(ver, "")
        self.assertEqual(src, "")


class TestMergeWithManual(unittest.TestCase):
    def test_manual_wins_for_nonempty(self):
        auto = [
            {"vllm-ascend": "v0.18.0", "vllm": "0.18.0", "cann": "8.5.1",
             "pytorch": "", "pytorch-ascend": "", "npu-driver": "", "notes": "", "source": "自动(x)"}
        ]
        manual = [
            {"vllm-ascend": "v0.18.0", "vllm": "0.12.1", "cann": "", "pytorch": "2.6.0",
             "pytorch-ascend": "2.6.0.post1", "npu-driver": "", "notes": "人工核对", "source": "人工"}
        ]
        merged = bm.merge_with_manual(auto, manual)
        row = merged[0]
        self.assertEqual(row["vllm"], "0.12.1")  # 人工 vllm 优先
        self.assertEqual(row["cann"], "8.5.1")  # 自动补空
        self.assertEqual(row["pytorch"], "2.6.0")
        self.assertEqual(row["source"], "人工")

    def test_manual_rows_outside_quay_preserved(self):
        auto = [{"vllm-ascend": "v0.18.0", "vllm": "0.18.0", "cann": "8.5.1",
                 "pytorch": "", "pytorch-ascend": "", "npu-driver": "", "notes": "", "source": ""}]
        manual = [{"vllm-ascend": "internal-fix", "vllm": "0.10.1", "cann": "8.0", "notes": "内部版本"}]
        merged = bm.merge_with_manual(auto, manual)
        self.assertEqual([r["vllm-ascend"] for r in merged], ["internal-fix", "v0.18.0"])


class TestGapReport(unittest.TestCase):
    def test_gap_count(self):
        rows = [
            {"vllm-ascend": "v0.18.0", "vllm": "0.18.0", "cann": "8.5.1",
             "pytorch": "2.6.0", "pytorch-ascend": "2.6.0.post1", "npu-driver": ""},
            {"vllm-ascend": "glm5", "vllm": "", "cann": "8.5.0",
             "pytorch": "", "pytorch-ascend": "", "npu-driver": ""},
            {"vllm-ascend": "v0.11.0", "vllm": "0.11.0", "cann": "",
             "pytorch": "", "pytorch-ascend": "", "npu-driver": ""},
        ]
        n = bm.report_gaps(rows)
        # glm5（缺 vllm/pytorch/pytorch-ascend）与 v0.11.0（缺 cann/pytorch/pytorch-ascend）计入；
        # 仅缺 npu-driver 的不计（HDK 与镜像解耦，属预期）
        self.assertEqual(n, 2)


class TestFetchReleases(unittest.TestCase):
    """fetch_releases：分页全量、重试、失败不阻塞。"""

    def _resp(self, rows):
        r = unittest.mock.Mock()
        r.raise_for_status.return_value = None
        r.json.return_value = rows
        return r

    def test_single_page(self):
        from unittest import mock

        with mock.patch("vllm_kb.net.get_session") as ms, mock.patch("socket.setdefaulttimeout"):
            ms.return_value.get.return_value = self._resp(
                [{"tag_name": "v0.23.0", "body": "aligns with upstream vLLM v0.23.0"},
                 {"tag_name": "v0.22.1", "body": ""}])
            rel = bm.fetch_releases()
        self.assertEqual(rel, {"v0.23.0": "aligns with upstream vLLM v0.23.0", "v0.22.1": ""})

    def test_pagination_all_pages(self):
        from unittest import mock

        def fake_get(url, **kw):
            page = kw["params"]["page"]
            if page == 1:
                return self._resp([{"tag_name": f"v0.20.{i}", "body": ""} for i in range(100)])
            return self._resp([{"tag_name": "v0.19.1", "body": "x"}, {}])

        with mock.patch("vllm_kb.net.get_session") as ms, mock.patch("socket.setdefaulttimeout"):
            ms.return_value.get.side_effect = fake_get
            rel = bm.fetch_releases()
        self.assertEqual(len(rel), 101)  # 100 + 1（空 dict 不计）
        self.assertIn("v0.19.1", rel)
        self.assertIn("v0.20.0", rel)

    def test_retry_then_success(self):
        from unittest import mock

        attempts = {"n": 0}

        def flaky_get(url, **kw):
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise RuntimeError("boom")
            return self._resp([{"tag_name": "v0.23.0", "body": ""}])

        with mock.patch("vllm_kb.net.get_session") as ms, mock.patch("socket.setdefaulttimeout"), \
                mock.patch("time.sleep"):
            ms.return_value.get.side_effect = flaky_get
            rel = bm.fetch_releases()
        self.assertEqual(attempts["n"], 3)
        self.assertEqual(rel, {"v0.23.0": ""})

    def test_all_fail_returns_empty(self):
        from unittest import mock

        with mock.patch("vllm_kb.net.get_session") as ms, mock.patch("socket.setdefaulttimeout"), \
                mock.patch("time.sleep"):
            ms.return_value.get.side_effect = RuntimeError("total down")
            rel = bm.fetch_releases()
        self.assertEqual(rel, {})  # 不抛异常，矩阵生成不阻塞


if __name__ == "__main__":
    unittest.main()
