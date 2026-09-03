"""网络访问统一入口：真实业务环境（SSL 被禁 / 镜像源）支持。

供脚本（build_companion_matrix / fetch_quay_tags / build_release_calendar /
build_code_snapshots 等）共用，避免每个脚本重复实现。

- **insecure**：跳过 SSL 证书校验（自签证书/SSL 被禁），requests 传 verify=False，
  urllib 用 CERT_NONE opener；
- **base 覆盖**：把公网域名前缀换成业务侧镜像（http/https 均可）。

配置来源（优先级：命令行参数 > 环境变量 > 默认）：
    --insecure / VLLM_KB_INSECURE=1
    --github-base / VLLM_KB_GITHUB_BASE   （GitHub API 镜像，默认 https://api.github.com）
    --quay-base   / VLLM_KB_QUAY_BASE     （quay 镜像，默认 https://quay.io）
"""
from __future__ import annotations

import os
import ssl
from typing import Optional

DEFAULT_GITHUB_BASE = "https://api.github.com"
DEFAULT_QUAY_BASE = "https://quay.io"


def insecure_from_env() -> bool:
    return os.environ.get("VLLM_KB_INSECURE", "").strip().lower() in ("1", "true", "yes", "on")


def get_session(insecure: bool) -> object:
    """返回 requests.Session（insecure 时跳过 SSL 证书校验 + 抑制告警）。

    insecure 实现为 **请求级 verify=False hook**（包装 session.request 缺省注入），
    而非仅设 session.verify=False：requests 的 merge_environment_settings 只在
    请求级 verify 为 True/None 时读 REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE 覆盖，
    而 merge_setting 请求级优先——session.verify=False 在不传请求级 verify 时
    是 None，会被这两个环境变量**覆盖成 CA bundle 路径**，校验"复活"
    （requests 2.34 实测复现；真实业务环境常设这两个变量让其他工具过 MITM）。
    请求级显式 False 不进覆盖分支，merge_setting 直接取 False——与
    requests.get(verify=False) 行为严格一致。trust_env 保持 True（代理不受影响）。
    """
    import requests

    s = requests.Session()
    if insecure:
        import urllib3

        orig_request = s.request

        def _request(method, url, **kwargs):
            kwargs.setdefault("verify", False)
            return orig_request(method, url, **kwargs)

        s.request = _request  # 实例属性遮蔽；get/post/redirect 均经此入口
        s.verify = False      # 双保险（请求级缺省时 merge_setting 的 session 级值）
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return s


def get_opener(insecure: bool):
    """返回 urllib opener（insecure 时 CERT_NONE，不校验证书）。"""
    if not insecure:
        return __import__("urllib.request", fromlist=["build_opener"]).build_opener()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return __import__("urllib.request", fromlist=["build_opener"]).build_opener(
        __import__("urllib.request", fromlist=["HTTPSHandler"]).HTTPSHandler(context=ctx)
    )


def github_api_base(args_base: Optional[str]) -> str:
    """解析 GitHub API 前缀：命令行 > 环境变量 > 默认。"""
    return (args_base or os.environ.get("VLLM_KB_GITHUB_BASE") or DEFAULT_GITHUB_BASE).rstrip("/")


def quay_base(args_base: Optional[str]) -> str:
    """解析 quay 前缀：命令行 > 环境变量 > 默认。"""
    return (args_base or os.environ.get("VLLM_KB_QUAY_BASE") or DEFAULT_QUAY_BASE).rstrip("/")


def add_insecure_args(ap) -> None:
    """给 argparse 统一加 --insecure / --github-base / --quay-base 参数。"""
    ap.add_argument("--insecure", action="store_true",
                    help="跳过 SSL 证书校验（真实业务环境自签证书/SSL 被禁；亦可用环境变量 VLLM_KB_INSECURE=1）")
    ap.add_argument("--github-base", default=None,
                    help=f"GitHub API 镜像前缀（默认 {DEFAULT_GITHUB_BASE}；亦可用环境变量 VLLM_KB_GITHUB_BASE）")
    ap.add_argument("--quay-base", default=None,
                    help=f"quay 镜像前缀（默认 {DEFAULT_QUAY_BASE}；亦可用环境变量 VLLM_KB_QUAY_BASE）")
