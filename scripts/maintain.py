"""日常维护入口：单指令全量部署 / 单指令增量更新。

继承环境变量 VLLM_KB_INSECURE / VLLM_KB_GITHUB_BASE 等，自动降级（非核心步骤
失败不会中断整个流程）。

用法（在项目根）：
    python scripts/maintain.py deploy              # 全量部署：拉取+入库+建图+建FTS+辅助数据
    python scripts/maintain.py deploy --help       # 查看 deploy 选项
    python scripts/maintain.py deploy --skip-code-snapshots  # 跳过代码快照（省时）

    python scripts/maintain.py update              # 增量更新：增量拉取+入库+重建图
    python scripts/maintain.py update --help
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 控制台输出容错（同 client.py：进程内把 stdout/stderr 重设为 UTF-8，errors=replace——
# 否则 GBK 控制台打印 emoji/生僻字会抛 UnicodeEncodeError）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# 子步骤列表（(脚本相对路径, 附加参数, 是否致命, 标签名称)）
# 非致命步骤失败时只打印告警，不中断流程（自动降级）
_DEPLOY_STEPS = [
    ("scripts/build_kb.py", [], True, "数据拉取与入库"),
    ("scripts/build_graph.py", [], False, "图构建（Kùzu）"),
    ("scripts/build_fts.py", [], False, "全文索引重建（FTS5）"),
    ("scripts/build_release_calendar.py", ["--all-repos"], False, "版本日历"),
    ("scripts/build_companion_matrix.py", [], False, "配套矩阵"),
]

_UPDATE_STEPS = [
    ("scripts/build_kb.py", ["--incremental"], True, "增量拉取与入库"),
    ("scripts/build_graph.py", [], False, "图重建（Kùzu）"),
]


def _resolve(script: str) -> str:
    return str(REPO_ROOT / script)


def _run_step(script: str, extra_args: list[str], fatal: bool, tag: str,
              insecure_env: dict[str, str], config: str | None) -> bool:
    """执行一个维护步骤。

    参数：
        script:      脚本路径（相对 REPO_ROOT）
        extra_args:  附加参数列表
        fatal:       失败时是否中断整个流程
        tag:         步骤标签（日志/告警用）
        insecure_env: 环境变量覆盖（VLLM_KB_INSECURE 等）
        config:      config.json 路径（None 则自动发现）

    返回 True=成功/降级继续，False=致命失败应中断。
    """
    cmd = [sys.executable, _resolve(script)]
    if config:
        cmd += ["--config", config]
    cmd += extra_args

    label = tag or script
    env = {**os.environ, **insecure_env}

    try:
        print(f"\n{'='*60}")
        print(f"[maintain] 步骤：{label}")
        print(f"[maintain] 命令：{' '.join(cmd)}")
        print(f"{'='*60}", flush=True)

        r = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT),
                           timeout=86400)  # 24h 超时（全量拉取几天）
        if r.returncode == 0:
            print(f"[maintain] ✅ {label} 完成\n")
            return True

        msg = f"[maintain] ❌ {label} 退出码 {r.returncode}"
        if fatal:
            print(msg)
            return False
        print(f"{msg}（非致命，自动降级继续）\n")
        return True

    except subprocess.TimeoutExpired:
        msg = f"[maintain] ⏰ {label} 超时"
        if fatal:
            print(msg)
            return False
        print(f"{msg}（非致命，自动降级继续）\n")
        return True

    except FileNotFoundError:
        print(f"[maintain] ❌ {label} 脚本不存在: {_resolve(script)}（跳过）")
        return not fatal

    except Exception as e:
        msg = f"[maintain] ❌ {label} 异常: {e}"
        if fatal:
            print(msg)
            return False
        print(f"{msg}（非致命，自动降级继续）\n")
        return True


def _insecure_env_from_args(args) -> dict[str, str]:
    """从 argparse 结果构建 insecurity 环境变量字典。"""
    env = {}
    if args.insecure:
        env["VLLM_KB_INSECURE"] = "1"
    if args.github_base:
        env["VLLM_KB_GITHUB_BASE"] = args.github_base
    if args.quay_base:
        env["VLLM_KB_QUAY_BASE"] = args.quay_base
    return env


def cmd_deploy(args) -> int:
    """全量部署。"""
    insecure_env = _insecure_env_from_args(args)
    steps = list(_DEPLOY_STEPS)

    # 代码快照默认包含（非致命，降级）：vllm-ascend 用 config.code.versions
    # （不传 --all 避免下载全部 tag），vllm 主仓用 companion 矩阵对应版本
    if not args.skip_code_snapshots:
        steps.append(
            ("scripts/build_code_snapshots.py", [], False, "代码快照（vllm-ascend，config.code.versions）")
        )
        steps.append(
            ("scripts/build_vllm_snapshots.py", [], False, "代码快照（vllm 主仓，companion 对应）")
        )

    start = time.time()
    print(f"[maintain] 🚀 全量部署开始 ...")
    print(f"[maintain] 配置: {'默认 config.json' if args.config is None else args.config}")
    if insecure_env:
        print(f"[maintain] 不安全模式: {insecure_env}")
    print()

    success_count = 0
    for script, extra, fatal, tag in steps:
        ok = _run_step(script, extra, fatal, tag, insecure_env, args.config)
        if not ok:
            print(f"\n[maintain] 💥 致命步骤失败，部署中止。")
            return 1
        success_count += 1

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n{'='*60}")
    print(f"[maintain] ✅ 全量部署完成：{success_count}/{len(steps)} 步骤成功")
    print(f"[maintain] 耗时：{mins} 分 {secs} 秒")
    return 0


def cmd_update(args) -> int:
    """增量更新（日常维护）。"""
    insecure_env = _insecure_env_from_args(args)
    start = time.time()
    print(f"[maintain] 🔄 增量更新开始 ...")
    if args.config:
        print(f"[maintain] 配置: {args.config}")
    if insecure_env:
        print(f"[maintain] 不安全模式: {insecure_env}")
    print()

    success_count = 0
    for script, extra, fatal, tag in _UPDATE_STEPS:
        ok = _run_step(script, extra, fatal, tag, insecure_env, args.config)
        if not ok:
            print(f"\n[maintain] 💥 增量更新中止。")
            return 1
        success_count += 1

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n{'='*60}")
    print(f"[maintain] ✅ 增量更新完成：{success_count}/{len(_UPDATE_STEPS)} 步骤成功")
    print(f"[maintain] 耗时：{mins} 分 {secs} 秒")
    print(f"[maintain] 提示：增量更新后图已重建，检索 API 需重启才能加载新图。")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="vllm-kb 日常维护入口：单指令全量部署 / 单指令增量更新",
    )
    ap.add_argument("--config", default=None,
                    help="config.json 路径（默认项目根 config.json，自动发现）")
    ap.add_argument("--insecure", action="store_true",
                    help="跳过 SSL 证书校验（真实业务环境自签证书/SSL 被禁；"
                         "亦可用环境变量 VLLM_KB_INSECURE=1，子步骤自动继承）")
    ap.add_argument("--github-base", default=None,
                    help="GitHub API 镜像前缀（默认 https://api.github.com；"
                         "亦可用环境变量 VLLM_KB_GITHUB_BASE，子步骤自动继承）")
    ap.add_argument("--quay-base", default=None,
                    help="quay 镜像前缀（默认 https://quay.io；"
                         "亦可用环境变量 VLLM_KB_QUAY_BASE，子步骤自动继承）")

    sub = ap.add_subparsers(dest="command", required=True)

    # --- deploy ---
    p_deploy = sub.add_parser("deploy", help="全量部署：拉取+入库+建图+建FTS+辅助数据")
    p_deploy.add_argument("--skip-code-snapshots", action="store_true",
                          help="跳过代码快照下载（省时省网络）")
    p_deploy.set_defaults(func=cmd_deploy)

    # --- update ---
    p_update = sub.add_parser("update", help="增量更新：增量拉取+入库+重建图")
    p_update.set_defaults(func=cmd_update)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()