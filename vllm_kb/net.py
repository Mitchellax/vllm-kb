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

# certifi 叠加提示只打一次（多个 opener 共用一个进程时避免刷屏）
_CERTIFI_ANNOUNCED = False


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


def ssl_context(insecure: bool = False) -> ssl.SSLContext:
    """HTTPS 客户端上下文：默认校验，并叠加 certifi 根证书（**修 urllib 全线 SSL 失败**）。

    背景（本仓库真实踩到，Windows 实测）：这台机器的 venv 是 OpenSSL 1.1.1t，
    `ssl.create_default_context()` 只加载到 **26 张 CA**（缺 GitHub/quay 需要的根证书），
    于是所有 urllib HTTPS 都报
    `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate (_ssl.c:1007)`——
    表现为版本日历/代码快照/vllm 快照"全部 SSL 失败"。叠加 certifi 后 138 张，直连正常。
    `requests` 自带 certifi 所以一直没事：这就是同一台机器上 requests 能跑、urllib 全废的根因。

    叠加 certifi 是**只增不减**（公开根证书，requests 对每个请求都是这么做的），
    不会挤掉业务侧自签根证书；用户若显式设了 `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`，
    则完全尊重其选择（不叠加）。insecure 时退回 CERT_NONE（自签/SSL 被禁环境）。
    """
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE"):
        return ctx
    global _CERTIFI_ANNOUNCED
    try:
        import certifi

        before = ctx.cert_store_stats().get("x509_ca", 0)
        ctx.load_verify_locations(certifi.where())
        if not _CERTIFI_ANNOUNCED:
            _CERTIFI_ANNOUNCED = True
            after = ctx.cert_store_stats().get("x509_ca", 0)
            print(f"[net] 已叠加 certifi 根证书（{before} → {after} 张）：{certifi.where()}",
                  flush=True)
    except Exception:
        pass
    return ctx


def get_opener(insecure: bool):
    """返回 urllib opener（insecure 时 CERT_NONE，不校验证书）。

    证书上下文统一走 ssl_context（系统 CA 为空时自动兜底 certifi，见其文档）。
    """
    import urllib.request

    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl_context(insecure))
    )


def parse_json(r, what: str):
    """解析 HTTP 响应 JSON，带防护：200 空响应 / HTML 错误页 / 网关页等非 JSON 体
    统一转成带上下文（状态码 + body 前缀）的 RuntimeError，便于定位与降级处理。

    requests 的 r.json() 对非 JSON 抛 requests.exceptions.JSONDecodeError
    （ValueError 子类），裸调用会让业务方（增量更新等）直接崩在 json 解析上——
    统一在这里捕获并给上下文。urllib 场景传入已读出的 bytes/str 亦可。
    """
    import json as _json

    try:
        if isinstance(r, (bytes, str)):
            raw = r.decode("utf-8") if isinstance(r, bytes) else r
            return _json.loads(raw)
        return r.json()
    except ValueError as e:
        try:
            text = r.text if not isinstance(r, (bytes, str)) else (r if isinstance(r, str) else r.decode("utf-8"))
        except Exception:
            text = "<unreadable>"
        snippet = (text or "")[:200].replace("\n", " ")
        status = getattr(r, "status_code", "?")
        raise RuntimeError(
            f"{what}：HTTP {status} 响应非 JSON（查询本身合法但被网关/代理/限流"
            f"页或空体替代）；body 前缀: {snippet!r}"
        ) from e


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
