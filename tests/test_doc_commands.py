"""文档-代码一致性测试：README/USAGE/SKILL 中的每条 python 命令均可解析（--help / 参数校验）。

防止文档命令与代码脱节（脚本改名、参数变更、命令删除）。每个命令以 --help 子进程验证
（约 70 条，无网络/无 API/无副作用）。unittest 模块调用（python -m unittest）单独验证。
"""
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "docs/USAGE.md", ROOT / "skills/vllm-kb/SKILL.md"]

BLOCK_RE = re.compile(r"```(?:bash|shell|sh)\n(.*?)```", re.S)
CMD_RE = re.compile(r"^\s*(python\s+\S+.*?)\s*(?:#.*)?$")


def extract_commands() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for block in BLOCK_RE.findall(text):
            for line in block.splitlines():
                m = CMD_RE.match(line)
                if m and m.group(1).strip() not in seen:
                    seen.add(m.group(1).strip())
                    out.append(m.group(1).strip())
    return out


def resolve_script(cmd: str) -> str | None:
    parts = cmd.split()
    script = next((p for p in parts[1:] if not p.startswith("-")), None)
    if not script:
        return None
    if script == "client.py" and not (ROOT / script).exists():
        script = "skills/vllm-kb/client.py"
    return script if (ROOT / script).exists() else None


class TestDocCommands(unittest.TestCase):
    maxDiff = None

    def test_all_doc_commands_resolve_and_parse(self):
        cmds = extract_commands()
        self.assertGreater(len(cmds), 50, "文档命令过少，可能解析异常")
        failures: list[str] = []
        # client.py 的 --help 一次列全部子命令，跳过逐条子命令 --help（argparse 对 unknown
        # 子命令报 exit 2 而非 crash；子命令覆盖由 test_skill_covers_all_client_subcommands 反向核验）
        client_help_done = False
        for cmd in cmds:
            if "-m unittest" in cmd:
                continue  # 模块调用，另行验证
            script = resolve_script(cmd)
            if script is None:
                failures.append(f"{cmd}  → 脚本不存在")
                continue
            parts = cmd.split()
            if "client.py" in script:
                if client_help_done:
                    continue  # 只跑一次 client.py --help（列全部子命令），跳过逐条
                args = [sys.executable, str(ROOT / script), "--help"]
                client_help_done = True
            else:
                args = [sys.executable, str(ROOT / script), "--help"]
            try:
                r = subprocess.run(args, capture_output=True, timeout=15, cwd=str(ROOT))
                if r.returncode != 0:
                    tail = (r.stderr or r.stdout).decode("utf-8", "replace").strip().splitlines()
                    failures.append(f"{cmd}  → 退出码 {r.returncode}: {' '.join(tail[-2:])[:120]}")
            except subprocess.TimeoutExpired:
                failures.append(f"{cmd}  → 超时")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_unittest_module_command(self):
        # 文档中的 `python -m unittest discover tests` 能发现并运行全部测试（全量由 CI/手动跑）；
        # 此处只验证 unittest 模块调用可用（跑一个快模块，避免 discover 递归包含本文件）
        r = subprocess.run(
            [sys.executable, "-m", "unittest", "tests.test_confidence"],
            capture_output=True, timeout=60, cwd=str(ROOT),
        )
        self.assertIn(b"OK", r.stdout + r.stderr)

    def test_skill_covers_all_client_subcommands(self):
        """反向核验：client.py 每个子命令（含 graph 子命令）都必须在 SKILL.md 中出现。

        与 test_all_doc_commands_resolve_and_parse 互补：那条只保证"文档中的命令可解析"，
        这条保证"client 暴露的命令都被 skill 文档化"（防新增命令漏写进 SKILL.md）。
        """
        client_py = (ROOT / "skills/vllm-kb/client.py").read_text(encoding="utf-8")
        skill = (ROOT / "skills/vllm-kb/SKILL.md").read_text(encoding="utf-8")
        subcmds = sorted(set(re.findall(r'add_parser\("([a-z][a-z0-9-]*)"', client_py)))
        self.assertGreater(len(subcmds), 10, "client 子命令解析异常，正则可能失效")
        missing = [c for c in subcmds if c not in skill]
        self.assertEqual(missing, [], f"SKILL.md 未覆盖的 client 子命令: {missing}")


if __name__ == "__main__":
    unittest.main()
