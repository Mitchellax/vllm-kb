"""启动只读检索 API：python scripts/serve_api.py [--config config.json] [--host ...] [--port ...]

结构只读：SQLite mode=ro + 向量库写操作抛错 + 无写端点（见 vllm_kb/api.py）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="启动 vllm-kb 只读检索 API")
    ap.add_argument("--config", default=None, help="config.json 路径（默认项目根）")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    # 检索服务只读启动：不要求密钥（embedding 不可用时检索降级为全文，见 search.py）
    # 注：AppConfig.load 已自动加载 data/secrets.local.json（审核工作台写入的密钥）
    cfg = AppConfig.load(args.config, require_keys=False)
    host = args.host or cfg.api.host
    port = args.port or cfg.api.port

    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("[api] 缺少依赖：pip install fastapi uvicorn")
        sys.exit(1)

    from vllm_kb.api import create_app
    from vllm_kb.logging_setup import setup_logging

    # 总日志：打屏 + 可选分卷落盘（config.json logging 段，默认不落盘）
    setup_logging(cfg, log_name="serve_api")

    config_path = str(Path(args.config or "config.json").resolve())
    print(f"[api] 启动只读检索服务 http://{host}:{port}（结构只读：SQLite mode=ro + 向量库写操作抛错 + 无写端点）", flush=True)
    uvicorn.run(create_app(config_path), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
