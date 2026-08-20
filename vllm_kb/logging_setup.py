"""总日志接口：打屏 + 可选落盘（分卷）。

- **打屏（默认）**：所有服务日志输出到控制台（现状保持）；
- **落盘（config 开启）**：`RotatingFileHandler` 按大小分卷（max_bytes / backup_count），
  避免单文件无限增长；
- 覆盖范围：root logger（服务入口的 uvicorn 访问/错误日志）+ 业务模块的 logging 调用；
  现有 print 输出仍打屏（不受影响）。

配置（config.json logging 段）：
    {"console": true, "file": false, "file_path": "logs/vllm-kb.log",
     "max_bytes": 10485760, "backup_count": 5}
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import AppConfig

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_CONSOLE_FORMAT = "%(levelname)s [%(name)s] %(message)s"


def setup_logging(cfg: Optional["AppConfig"] = None,
                  log_name: str = "vllm-kb") -> None:
    """配置 root logger：控制台（打屏）+ 可选分卷文件（config 开启）。

    幂等：重复调用只补文件 handler，不重复添加。
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 控制台 handler（打屏）
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        root.addHandler(console)

    # 文件 handler（可选，分卷）
    if cfg is not None:
        lc = cfg.logging
        if lc.file and lc.file_path:
            path = cfg.resolve(lc.file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not any(getattr(h, "baseFilename", "") == str(path) for h in root.handlers):
                fh = logging.handlers.RotatingFileHandler(
                    str(path), maxBytes=lc.max_bytes, backupCount=lc.backup_count,
                    encoding="utf-8",
                )
                fh.setFormatter(logging.Formatter(_LOG_FORMAT))
                root.addHandler(fh)

    # uvicorn 日志接入 root（访问/错误日志随总日志落盘）
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True  # 传播到 root（控制台 + 文件）

    logging.getLogger(log_name).info("日志初始化完成（console=%s, file=%s）",
                                     True, (cfg.logging.file if cfg else False))
