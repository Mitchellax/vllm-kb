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

**维护日志**：`collector`（可选）收集本次被脱敏命中的原始 IP/路径，调用方可
`save_sanitize_log` 合并写入 `data/sanitize_log.json`（本地维护文件，**不进库、
不返回给 agent**）——方便维护者查看"哪些内部值被脱敏"，据此调整白名单。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
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
                  keep_ips: Optional[list[str]] = None,
                  collector: Optional[dict] = None) -> str:
    """对文本中的 IP 与路径脱敏。

    - IP：`10.0.0.5` → `<IP>`；`10.0.0.5:8000` → `<IP>:8000`（端口保留）；
    - 路径：白名单外绝对路径 → `<PATH>`（盘符或 `/` 开头）；白名单内保留；
    - keep_paths/keep_ips 为 None → 默认白名单；显式 [] → 全部脱敏；
    - collector（可选 dict）：就地收集被脱敏的原始值
      （collector["ips"]/["paths"] 为 set），供 save_sanitize_log 落盘维护。
    """
    if not text:
        return text
    keep = DEFAULT_KEEP_PATHS if keep_paths is None else tuple(keep_paths)
    keep_ip = DEFAULT_KEEP_IPS if keep_ips is None else tuple(keep_ips)

    def _ip(m: re.Match) -> str:
        ip = m.group(0)
        if ip in keep_ip:
            return ip
        if collector is not None:
            collector.setdefault("ips", set()).add(ip)
        return "<IP>"

    def _path(m: re.Match) -> str:
        path = m.group(0)
        for p in keep:
            if path.startswith(p):
                return path  # 默认路径/日志路径保留（诊断价值）
        if collector is not None:
            collector.setdefault("paths", set()).add(path)
        return "<PATH>"

    return _PATH_RE.sub(_path, _IPV4_RE.sub(_ip, text))


def save_sanitize_log(cfg, collector: Optional[dict] = None):
    """把被脱敏命中的原始 IP/路径合并写入 data/sanitize_log.json（本地维护文件）。

    - 幂等合并（累积历史被脱敏值，便于维护白名单）；文件在 data/ 下（gitignore，不进库、
      不返回给 agent）；
    - 返回写入的路径。collector 为空时也刷新 updated_at。
    """
    path = cfg.resolve("data/sanitize_log.json")
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    ips = set(data.get("ips", [])) | set((collector or {}).get("ips", []))
    paths = set(data.get("paths", [])) | set((collector or {}).get("paths", []))
    out = {
        "ips": sorted(ips),
        "paths": sorted(paths),
        "count": {"ips": len(ips), "paths": len(paths)},
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "被脱敏命中的原始 IP/路径（本地维护用，不进库、不返回给 agent）；"
                "据此调整 config.json sanitize.keep_ips / keep_paths 白名单",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
