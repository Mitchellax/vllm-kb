"""对比两个预存版本快照中指定文件的源码差异——定位"哪个版本引入/修改了某代码"。

典型场景：在 main/新版代码里看到一段修复逻辑（如 slot_mapping.fill_(-1)），
想知道它从哪个 release 开始存在：对比相邻版本该文件的 diff，新增行出现的版本即引入版本，
再配合 GitHub commits 溯源 PR。

用法：
    python scripts/diff_code_versions.py vllm_ascend/worker/model_runner_v1.py v0.22.1rc1 v0.23.0rc1
    python scripts/diff_code_versions.py vllm_ascend/worker/model_runner_v1.py v0.23.0rc1 v0.23.0
    python scripts/diff_code_versions.py --repo vllm vllm/v1/executor/multiproc_executor.py 0.22.1 0.23.0
    python scripts/diff_code_versions.py ... --keyword "fill_(-1)"   # 只看含关键词的差异行
"""
import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig
from vllm_kb.code_index import VersionedCode


def main() -> None:
    ap = argparse.ArgumentParser(description="对比两个预存版本快照中指定文件的差异")
    ap.add_argument("path", help="文件路径（相对仓库根，如 vllm_ascend/worker/model_runner_v1.py）")
    ap.add_argument("v1", help="旧版本（如 v0.22.1rc1）")
    ap.add_argument("v2", help="新版本（如 v0.23.0rc1）")
    ap.add_argument("--repo", default="vllm-ascend", help="仓库：vllm-ascend（默认）| vllm")
    ap.add_argument("--keyword", default=None, help="只显示包含该关键词的差异行（定位修复代码）")
    ap.add_argument("--context", type=int, default=3, help="diff 上下文行数")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = AppConfig.load(args.config, require_keys=False)
    ci = VersionedCode(cfg, repo=args.repo)

    p1 = ci.find_file(args.v1, args.path)
    p2 = ci.find_file(args.v2, args.path)
    if p1 is None or p2 is None:
        missing = [v for v, p in ((args.v1, p1), (args.v2, p2)) if p is None]
        print(f"[diff] 版本 {missing} 未预存文件 {args.path}（先运行 scripts/build_code_snapshots.py 或 build_vllm_snapshots.py）")
        sys.exit(1)

    t1 = p1.read_text(encoding="utf-8", errors="replace").splitlines()
    t2 = p2.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"===== {args.path}  {args.v1} ({len(t1)} 行) → {args.v2} ({len(t2)} 行) =====")

    diff = list(difflib.unified_diff(t1, t2, fromfile=f"{args.v1}:{args.path}",
                                     tofile=f"{args.v2}:{args.path}", n=args.context, lineterm=""))
    shown = 0
    for line in diff:
        if line.startswith(("+++", "---", "@@")):
            print(line)
        elif args.keyword and args.keyword.lower() not in line.lower():
            continue
        else:
            print(line)
            shown += 1
    if shown == 0 and args.keyword:
        print(f"[diff] 无包含关键词 '{args.keyword}' 的差异行（v1/v2 该文件可能无差异）")


if __name__ == "__main__":
    main()
