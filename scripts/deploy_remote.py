"""远程部署辅助：把"存算分离"落地到远程服务器。

本地 skill（算）只保留 client.py + SKILL.md（~16KB）；
向量库/SQLite/代码仓等数据（存）部署到远程服务器，API 在远程跑。

用法（在项目根）：
    python scripts/deploy_remote.py --gen-config      # 生成远程 server 的最小 config（data 指向远程数据根）
    python scripts/deploy_remote.py --pack-data       # 打包数据（LanceDB/SQLite/code/compatibility）成 tar
    python scripts/deploy_remote.py --print-steps     # 打印远程部署步骤

远程服务器（数据 + API）：
    # 1. 上传 config.json（--gen-config 生成）+ 数据包（--pack-data）+ 代码
    # 2. pip install fastapi uvicorn lancedb
    # 3. VLLM_KB_DATA_ROOT=/data/vllm-kb python scripts/serve_api.py --host 0.0.0.0 --port 8000

本地（skill，只算不发数据）：
    export VLLM_KB_BASE=http://<remote>:8000
    python skills/vllm-kb/client.py search "..."   # 全部命令走远程
"""
import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402


def gen_remote_config(cfg: AppConfig, out: Path) -> None:
    """生成远程 server 的最小 config：data 路径全部相对（由 VLLM_KB_DATA_ROOT 重定向）。"""
    p = Path("config.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    # 清理 token（远程不需要写 API，且避免密钥外泄）
    for src in data.get("sources", []):
        src.pop("token", None)
    data.get("embedding", {}).pop("api_key", None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[deploy] 远程 config -> {out}（已移除 token/api_key）")
    print(f"[deploy] 远程启动: VLLM_KB_DATA_ROOT=<数据目录> python scripts/serve_api.py --host 0.0.0.0")


def pack_data(cfg: AppConfig, out: Path, include_raw: bool = False) -> None:
    """打包数据（LanceDB/SQLite/code/compatibility，可选 raw）成 tar.gz。

    体积：LanceDB ~41GB → 打包可能仍大；建议远程直接用数据目录拷贝（rsync/scp -r），
    此脚本适用于中小数据或目录拷贝不便时。
    """
    root = cfg.resolve("data")
    if not root.exists():
        print(f"[deploy] 数据目录不存在: {root}")
        return
    dirs = ["lancedb", "code", "compatibility", "graph"]
    files = ["kb.sqlite3"]
    if include_raw:
        dirs.append("raw")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for d in dirs:
            p = root / d
            if p.exists():
                tar.add(p, arcname=f"data/{d}")
        for f in files:
            p = root / f
            if p.exists():
                tar.add(p, arcname=f"data/{f}")
    size = out.stat().st_size / 1e9
    print(f"[deploy] 数据包 -> {out}（{size:.2f} GB）")
    print(f"[deploy] 提示: 41GB LanceDB 打包/上传较慢，推荐 rsync/scp -r 整目录拷贝到远程")


def print_steps() -> None:
    print("""
===== 远程部署步骤（存算分离）=====

【远程服务器】（存数据 + 跑 API）
  1. 拷贝代码:  scp -r vllm-kb root@<remote>:/opt/
  2. 拷贝数据:  rsync -av vllm-kb/data/ root@<remote>:/data/vllm-kb/     # 或 tar 包
  3. 依赖:      pip install fastapi uvicorn lancedb kuzu
     （Phase 2 图检索需 kuzu；业务来源解析/OCR 另装 pymupdf python-docx trafilatura paddleocr）
  4. 启动:      cd /opt/vllm-kb && VLLM_KB_DATA_ROOT=/data/vllm-kb python scripts/serve_api.py --host 0.0.0.0 --port 8000
     （VLLM_KB_DATA_ROOT 让 data/* 路径全部重定向到 /data/vllm-kb）

【本地】（只算，skill 轻量）
  export VLLM_KB_BASE=http://<remote>:8000
  python skills/vllm-kb/client.py health          # 验证连通
  python skills/vllm-kb/client.py search "..."    # 语义检索
  python skills/vllm-kb/client.py signature "..." # 签名精确检索
  python skills/vllm-kb/client.py code <符号> --repo vllm --version 0.22.1   # 代码仓检索

【安全】
  - API 无写端点 + SQLite mode=ro（结构只读）；
  - 如需鉴权，可在远程加反向代理（nginx basic auth / token）。

【数据更新】
  数据更新仍在"拥有数据"的一端跑流水线（build_kb.py / build_*），
  更新后同步数据目录到远程即可（增量 rsync）。
""")


def main() -> None:
    ap = argparse.ArgumentParser(description="远程部署辅助（存算分离）")
    ap.add_argument("--gen-config", action="store_true", help="生成远程 server 最小 config")
    ap.add_argument("--pack-data", action="store_true", help="打包数据成 tar.gz")
    ap.add_argument("--include-raw", action="store_true", help="打包时包含 raw/ 原始数据")
    ap.add_argument("--out", default="deploy/", help="输出目录（默认 deploy/）")
    ap.add_argument("--print-steps", action="store_true", help="打印部署步骤")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    out_dir = Path(args.out)
    if args.gen_config:
        gen_remote_config(cfg, out_dir / "config.remote.json")
    if args.pack_data:
        pack_data(cfg, out_dir / "vllm-kb-data.tar.gz", include_raw=args.include_raw)
    if args.print_steps:
        print_steps()
    if not (args.gen_config or args.pack_data or args.print_steps):
        print_steps()


if __name__ == "__main__":
    main()
