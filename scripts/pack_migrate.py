"""迁移打包：业务环境重新嵌入（不传向量库）。

场景：本地已有完整数据（含 41GB LanceDB 向量库），业务环境要重新嵌入
（向量库太大不传输）。本脚本打包"重嵌入最小集"：

必需（重嵌入输入 + 业务数据）：
  - data/raw/canonical.jsonl        统一 canonical（66K 文档全文，重嵌入唯一输入）
  - data/compatibility/             release 日历 + 组件矩阵（置信度/图构建用）
  - data/imports/ + data/assets/    业务原始数据（PDF 手册、图片资产等）
  - data/parsed/                    PDF 表格解析产物（图构建用）
  - data/secrets.local.json         本地密钥（EMBEDDING_API_KEY 等）
  - config.json                     配置（业务环境需按需改 embedding.base_url）

可选（传 vs 重建）：
  - data/graph/   66MB，直接传省去业务环境跑 build_graph.py（推荐传）
  - data/code/    1.7GB，业务环境能访问 GitHub 则可重建（build_code_snapshots.py），
                  否则必须传（/code/search 依赖）
  - data/review.sqlite3  审核队列状态，全新环境可不传（<1KB）

不传（业务环境重建）：
  - data/lancedb/   41GB 向量库 —— 业务环境跑 build_kb.py --rebuild 重新嵌入
  - data/kb.sqlite3 645MB —— --rebuild 会重建（docs + FTS5）
  - data/raw/github/、data/raw/vllm-ascend/  818MB 原始 JSON，仅增量拉取需要
  - data/checkpoints/、data/graph_debug*/     调试/断点产物

用法：
    python scripts/pack_migrate.py                     # 默认最小集（不含 graph/code）
    python scripts/pack_migrate.py --with-graph        # 含 data/graph/（推荐）
    python scripts/pack_migrate.py --with-code         # 含 data/code/（无外网时）
    python scripts/pack_migrate.py --with-review       # 含审核状态
    python scripts/pack_migrate.py --out D:/out/migrate.tar.gz
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402


def _add(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    if path.exists():
        tar.add(path, arcname=arcname, recursive=True)
        print(f"  + {arcname} ({path.stat().st_size / 1e6:.1f} MB)")


def pack(cfg: AppConfig, out: Path, with_graph: bool, with_code: bool, with_review: bool) -> None:
    root = cfg.resolve("data")
    if not root.exists():
        print(f"[pack] 数据目录不存在: {root}")
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        print("[pack] 必需项：")
        _add(tar, root / "raw" / "canonical.jsonl", "data/raw/canonical.jsonl")
        _add(tar, root / "compatibility", "data/compatibility")
        _add(tar, root / "imports", "data/imports")
        _add(tar, root / "assets", "data/assets")
        _add(tar, root / "parsed", "data/parsed")
        _add(tar, root / "secrets.local.json", "data/secrets.local.json")
        if Path("config.json").exists():
            tar.add("config.json", arcname="config.json")
            print("  + config.json")
        print("[pack] 可选项：")
        if with_graph:
            _add(tar, root / "graph", "data/graph")
        if with_code:
            _add(tar, root / "code", "data/code")
        if with_review:
            _add(tar, root / "review.sqlite3", "data/review.sqlite3")
    size = out.stat().st_size / 1e6
    print(f"[pack] -> {out}（{size:.1f} MB）")


def print_steps() -> None:
    print("""
===== 业务环境重建步骤 =====

【1. 部署代码 + 依赖】
  git clone <你的仓库> && cd vllm-kb
  pip install -r requirements.txt          # 含 kuzu；业务来源解析另装 pymupdf 等

【2. 解压数据包】解到项目根（包内已是 data/... 结构）：
  tar -xzf migrate.tar.gz -C vllm-kb/

【3. 检查 config.json】（本地 config.json 已在包内）
  - embedding.base_url 改为业务环境的 embedding 服务（其他服务器 vLLM 端点）
  - 密钥经 data/secrets.local.json 或环境变量 EMBEDDING_API_KEY 提供
  - 如需拉取新 issue，配 GITHUB_TOKEN 并保留 data/raw/github*（本包未含）

【4. 重新嵌入（重建向量库 + kb.sqlite3）】：
  python scripts/build_kb.py --rebuild
  # 从 data/raw/canonical.jsonl 重新分块+嵌入，进度见 [ingest] 行
  # 66K 文档 / ~122K chunks，耗时取决于 embedding 服务吞吐

【5. 图（可选）】：
  - 已传 data/graph/：跳过
  - 未传：python scripts/build_graph.py   # 需 canonical + parsed（已打包）

【6. 代码仓（可选，/code/search 用）】：
  - 已传 data/code/：跳过
  - 未传且有外网：python scripts/build_code_snapshots.py

【7. 启动】：
  VLLM_KB_DATA_ROOT=<数据目录> python scripts/serve_api.py --host 0.0.0.0 --port 8000
""")


def main() -> None:
    ap = argparse.ArgumentParser(description="迁移打包：业务环境重新嵌入（不传向量库）")
    ap.add_argument("--with-graph", action="store_true", help="含 data/graph/（66MB，推荐）")
    ap.add_argument("--with-code", action="store_true", help="含 data/code/（1.7GB，无外网时）")
    ap.add_argument("--with-review", action="store_true", help="含审核状态 review.sqlite3")
    ap.add_argument("--out", default="deploy/migrate.tar.gz", help="输出路径（默认 deploy/migrate.tar.gz）")
    ap.add_argument("--steps", action="store_true", help="只打印业务环境步骤，不打包")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    if args.steps:
        print_steps()
        return
    cfg = AppConfig.load(args.config)
    pack(cfg, Path(args.out), args.with_graph, args.with_code, args.with_review)
    print("[pack] 业务环境步骤见: python scripts/pack_migrate.py --steps")


if __name__ == "__main__":
    main()
