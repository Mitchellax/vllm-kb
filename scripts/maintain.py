"""日常维护入口：单指令全量部署 / 单指令增量更新。

继承环境变量 VLLM_KB_INSECURE / VLLM_KB_GITHUB_BASE 等，自动降级（非核心步骤
失败不会中断整个流程）。

用法（在项目根）：
    python scripts/maintain.py deploy              # 全量部署：拉取+入库+建图+建FTS+辅助数据
    python scripts/maintain.py deploy --help       # 查看 deploy 选项
    python scripts/maintain.py deploy --skip-graph # 跳过图构建（省时 / Kùzu 未装）
    python scripts/maintain.py deploy --skip-code-snapshots  # 跳过代码快照（省时）
    python scripts/maintain.py deploy --all-code   # vllm-ascend 代码拉**全部 tag**（默认仅 config.code.versions）

    python scripts/maintain.py update              # 增量更新：增量拉取+入库+重建图
    python scripts/maintain.py update --skip-graph # 仅增量入库，不改图（API 不便停时）

前置要求：
    - config.json 已配置（默认项目根 config.json，也可 --config 指定）
    - GITHUB_TOKEN / EMBEDDING_API_KEY 等密钥已设置（环境变量）
    - 首次部署需网络（后续增量更新离线可用）
    - **更新前请停止 serve_api**（build_graph 需要 Kùzu 单写者）

自动降级（非核心步骤失败不中断，只告警）：
    - kuzu 未装 → build_graph 跳过（告警继续，不影响 KB）
    - jieba 未装 → FTS 降级为原文（无中文分词，索引不变）
    - 网络不可达 → 版本日历/配套矩阵/代码快照跳过
    - 建图/FTS/日历/矩阵/快照任一步失败 → 告警继续，已完成的入库不受影响
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

_STEP_KEY = {  # 步骤名 → key（供 --skip-* 过滤）
    "图构建（Kùzu）": "graph",
    "图重建（Kùzu）": "graph",
    "全文索引重建（FTS5）": "fts",
    "版本日历": "calendar",
    "配套矩阵": "matrix",
    "代码快照（vllm-ascend，config.code.versions）": "code_snapshots",
    "代码快照（vllm-ascend，全部 tag --all）": "code_snapshots",
    "代码快照（vllm 主仓，companion 对应）": "code_snapshots",
}


def _remind_stop_api(skip_keys: set[str]):
    """在图相关步骤将执行时打印停服务提醒。"""
    if "graph" not in skip_keys:
        print("\n[maintain] ⚠️  注意：build_graph 要求停 serve_api（Kùzu 单写者）。")
        print("[maintain]   如果 serve_api 仍在运行，请先 Ctrl-C 停止，更新完成后重启。\n")


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


def _filter_steps(steps, skip_keys: set[str]):
    """按 --skip-* 过滤步骤（保留无对应 key 的步骤）。"""
    return [s for s in steps if _STEP_KEY.get(s[3], "") not in skip_keys]


def cmd_deploy(args) -> int:
    """全量部署。"""
    insecure_env = _insecure_env_from_args(args)
    steps = list(_DEPLOY_STEPS)

    # 代码快照默认包含（非致命，降级）：vllm-ascend 默认按 config.code.versions 拉取
    # （精选版本，省时省磁盘）；--all-code 时传 --all 拉取全部 tag（数 GB、耗时，按需开启）；
    # vllm 主仓用 companion 矩阵对应版本
    if not args.skip_code_snapshots:
        code_extra = ["--all"] if args.all_code else []
        steps.append(
            ("scripts/build_code_snapshots.py", code_extra, False,
             "代码快照（vllm-ascend" + ("，全部 tag --all" if args.all_code else "，config.code.versions") + "）")
        )
        steps.append(
            ("scripts/build_vllm_snapshots.py", [], False, "代码快照（vllm 主仓，companion 对应）")
        )
    steps = _filter_steps(steps, args.skip)

    start = time.time()
    print(f"[maintain] 🚀 全量部署开始 ...")
    print(f"[maintain] 配置: {'默认 config.json' if args.config is None else args.config}")
    if insecure_env:
        print(f"[maintain] 不安全模式: {insecure_env}")
    if args.skip:
        print(f"[maintain] 跳过步骤: {', '.join(sorted(args.skip))}")
    _remind_stop_api(args.skip)
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
    steps = _filter_steps(_UPDATE_STEPS, args.skip)
    start = time.time()
    print(f"[maintain] 🔄 增量更新开始 ...")
    if args.config:
        print(f"[maintain] 配置: {args.config}")
    if insecure_env:
        print(f"[maintain] 不安全模式: {insecure_env}")
    if args.skip:
        print(f"[maintain] 跳过步骤: {', '.join(sorted(args.skip))}")
    _remind_stop_api(args.skip)
    print()

    success_count = 0
    for script, extra, fatal, tag in steps:
        ok = _run_step(script, extra, fatal, tag, insecure_env, args.config)
        if not ok:
            print(f"\n[maintain] 💥 增量更新中止。")
            return 1
        success_count += 1

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n{'='*60}")
    print(f"[maintain] ✅ 增量更新完成：{success_count}/{len(steps)} 步骤成功")
    print(f"[maintain] 耗时：{mins} 分 {secs} 秒")
    print(f"[maintain] 提示：增量更新后图已重建，检索 API 需重启才能加载新图。")
    return 0


def _add_common_args(ap, *, suppress_defaults=False) -> None:
    """公共参数（主命令与子命令都接受，位置灵活：`maintain.py --insecure deploy` 或
    `maintain.py deploy --insecure` 均可）。

    suppress_defaults：子命令用（用 argparse.SUPPRESS 避免子命令默认值覆盖父命令已解析值）。
    """
    cfg_def = argparse.SUPPRESS if suppress_defaults else None
    ins_def = argparse.SUPPRESS if suppress_defaults else False
    ap.add_argument("--config", default=cfg_def,
                    help="config.json 路径（默认项目根 config.json，自动发现）")
    ap.add_argument("--insecure", action="store_true", default=ins_def,
                    help="跳过 SSL 证书校验（真实业务环境自签证书/SSL 被禁；"
                         "亦可用环境变量 VLLM_KB_INSECURE=1，子步骤自动继承）")
    ap.add_argument("--github-base", default=cfg_def,
                    help="GitHub API 镜像前缀（默认 https://api.github.com；"
                         "亦可用环境变量 VLLM_KB_GITHUB_BASE，子步骤自动继承）")
    ap.add_argument("--quay-base", default=cfg_def,
                    help="quay 镜像前缀（默认 https://quay.io；"
                         "亦可用环境变量 VLLM_KB_QUAY_BASE，子步骤自动继承）")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="vllm-kb 日常维护入口：单指令全量部署 / 单指令增量更新",
    )
    _add_common_args(ap)

    sub = ap.add_subparsers(dest="command", required=True)

    # --- deploy ---
    p_deploy = sub.add_parser("deploy", help="全量部署：拉取+入库+建图+建FTS+辅助数据")
    _add_common_args(p_deploy, suppress_defaults=True)
    p_deploy.add_argument("--skip-graph", action="store_true",
                          help="跳过图构建（Kùzu 未装或不想重建时）")
    p_deploy.add_argument("--skip-fts", action="store_true",
                          help="跳过全文索引重建")
    p_deploy.add_argument("--skip-calendar", action="store_true",
                          help="跳过版本日历拉取")
    p_deploy.add_argument("--skip-matrix", action="store_true",
                          help="跳过配套矩阵拉取")
    p_deploy.add_argument("--skip-code-snapshots", action="store_true",
                          help="跳过代码快照下载（省时省网络）")
    p_deploy.add_argument("--all-code", action="store_true",
                          help="vllm-ascend 代码快照拉取**全部 tag**（默认只拉 "
                               "config.code.versions 精选版本；全量数 GB、耗时长，按需开启）")
    p_deploy.set_defaults(func=cmd_deploy)

    # --- update ---
    p_update = sub.add_parser("update", help="增量更新：增量拉取+入库+重建图")
    _add_common_args(p_update, suppress_defaults=True)
    p_update.add_argument("--skip-graph", action="store_true",
                          help="跳过图重建（API 不便停时，仅增量入库）")
    p_update.set_defaults(func=cmd_update)

    args = ap.parse_args()
    # 汇总 --skip-* 为步骤 key 集合，供步骤过滤
    skip = set()
    if getattr(args, "skip_graph", False):
        skip.add("graph")
    if getattr(args, "skip_fts", False):
        skip.add("fts")
    if getattr(args, "skip_calendar", False):
        skip.add("calendar")
    if getattr(args, "skip_matrix", False):
        skip.add("matrix")
    if getattr(args, "skip_code_snapshots", False):
        skip.add("code_snapshots")
    args.skip = skip
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()