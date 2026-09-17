"""诊断代码仓符号索引（只读，无副作用）：排查 code search 503「缺 kind 列」等 schema 问题。

用法（项目根）：
    python scripts/diagnose_code_index.py [--config config.json]
    python scripts/diagnose_code_index.py --all        # 所有命名空间（vllm-ascend/vllm/fork:/img:）

只读：不创建/修改任何文件；index.sqlite3 以 mode=ro 打开。
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402
from vllm_kb.code_index import VersionedCode  # noqa: E402


def _check(code: VersionedCode) -> None:
    print(f"--- {code.repo} ---")
    print(f"code_root   = {code.root}")
    print(f"index_path  = {code.index_path}  存在={code.index_path.exists()}")
    print(f"zips_dir    = {code.zips_dir}    存在={code.zips_dir.exists()}")
    print(f"snapshots   = {code.snapshots_dir} 存在={code.snapshots_dir.exists()}")
    print(f"available_versions = {code.available_versions or '(空)'}")
    if code.index_path.exists():
        print(f"index 大小 = {code.index_path.stat().st_size} bytes")
        try:
            conn = sqlite3.connect(f"file:{code.index_path}?mode=ro", uri=True)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(symbols)")]
            print(f"symbols 列 = {cols}")
            if "kind" in cols:
                nulls = conn.execute("SELECT COUNT(*) FROM symbols WHERE kind IS NULL").fetchone()[0]
                print(f"kind IS NULL 行 = {nulls}")
            conn.close()
        except sqlite3.Error as e:
            print(f"sqlite 打开失败: {e}")
    else:
        print("index.sqlite3 不存在（API 会报「索引不存在」而非「缺 kind 列」）")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="诊断代码仓符号索引（只读）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--all", action="store_true", help="检查所有命名空间索引")
    args = ap.parse_args()

    print("环境: VLLM_KB_DATA_ROOT =", repr(os.environ.get("VLLM_KB_DATA_ROOT")))
    print("cwd:", os.getcwd())
    cfg = AppConfig.load(args.config)
    print()

    _check(VersionedCode(cfg))
    if args.all:
        # vllm 主仓 + 已存在的 fork:/img: 命名空间
        for ns in ("vllm",):
            _check(VersionedCode(cfg, repo=ns))
        for base, prefix in ((cfg.resolve("data/code/forks"), "fork:"),
                             (cfg.resolve("data/code/images"), "img:")):
            if base.is_dir():
                for d in sorted(base.iterdir()):
                    if d.is_dir():
                        _check(VersionedCode(cfg, repo=prefix + d.name))


if __name__ == "__main__":
    main()
