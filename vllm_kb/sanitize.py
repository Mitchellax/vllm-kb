"""内部数据脱敏：IP 与路径（Excel 等业务来源的 cell 值、正文中的敏感形态）。

**原则**：
- **IP**：IPv4 统一脱敏为 `<IP>`（端口保留，有诊断价值）；`keep_ips` 白名单内保留
  （默认回环/通配 127.0.0.1、0.0.0.0、255.255.255.255、::1，无泄露风险）；
- **路径**：绝对路径（Windows 盘符 `C:\` / 以 `/` 开头）按 `keep_paths` 白名单判定——
  白名单内（昇腾默认安装/系统日志/配置目录，诊断价值高且不含内部信息）**保留**，
  白名单外（`/home/<user>`、`/data/...`、`/root/...` 等内部路径）→ `<PATH>`；
- **相对路径**（`./x`、`a/b/c`，无盘符非 `/` 开头）低泄露风险，保留。

**参数语义（与 config.sanitize 对齐）**：
- `keep_paths` / `keep_ips` 为 `None` → 使用默认（本模块常量）；
  显式空列表 `[]` → 全部脱敏（不保留任何路径/IP）。
  业务侧可在 config.json 的 `sanitize` 段配置覆盖。
"""
from __future__ import annotations

import re
from typing import Optional

# 默认保留路径前缀（昇腾/系统默认，诊断价值高、不含内部机器信息）
DEFAULT_KEEP_PATHS: tuple[str, ...] = (
    "/usr/local/Ascend",   # 昇腾 CANN/驱动默认安装目录
    "/var/log/",           # 系统/服务日志（含 npu 日志默认位置）
    "/etc/ascend",         # 昇腾配置
    "/etc/npu",            # NPU 配置
    "/tmp/",               # 临时目录
    "/usr/bin/",           # 系统可执行
    "/bin/",
    "/sbin/",
    "/usr/lib/",           # 系统库（部分库路径在报错堆栈中常见）
    "/lib/",
)

# 默认保留的 IP（无泄露风险：回环/通配/广播）
DEFAULT_KEEP_IPS: tuple[str, ...] = ("127.0.0.1", "0.0.0.0", "255.255.255.255", "::1")

# IPv4（含端口分离——端口保留）
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# 路径形态：Windows 盘符 `C:\` 或以 `/` 开头（前一个字符不能是字母/数字/下划线/点/斜杠，
# 避免把 `a/b/c` 相对路径或 URL path 误判为绝对路径的起始）——匹配到空白/引号/括号为止
_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:[A-Za-z]:[\\/]|/)[^\s\"'()<>]+"
)


def sanitize_text(text: str,
                  keep_paths: Optional[list[str]] = None,
                  keep_ips: Optional[list[str]] = None) -> str:
    """对文本中的 IP 与路径脱敏。

    - IP：`10.0.0.5` → `<IP>`；`10.0.0.5:8000` → `<IP>:8000`（端口保留）；
    - 路径：白名单外绝对路径 → `<PATH>`（盘符或 `/` 开头）；白名单内保留；
    - keep_paths/keep_ips 为 None → 默认白名单；显式 [] → 全部脱敏。
    """
    if not text:
        return text
    keep = DEFAULT_KEEP_PATHS if keep_paths is None else tuple(keep_paths)
    keep_ip = DEFAULT_KEEP_IPS if keep_ips is None else tuple(keep_ips)

    def _ip(m: re.Match) -> str:
        ip = m.group(0)
        return ip if ip in keep_ip else "<IP>"

    def _path(m: re.Match) -> str:
        path = m.group(0)
        for p in keep:
            if path.startswith(p):
                return path  # 默认路径/日志路径保留（诊断价值）
        return "<PATH>"

    return _PATH_RE.sub(_path, _IPV4_RE.sub(_ip, text))
