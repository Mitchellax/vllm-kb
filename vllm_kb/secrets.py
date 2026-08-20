"""本地密钥存储：data/secrets.local.json（不入 config.json，遵守"密钥走环境变量"原则）。

- 审核工作台 API 配置中心填写 API key 时写入本文件；
- serve_api / review_ui 启动时 load_secrets()：文件中的 key 在环境变量**未设置**时注入
  os.environ，之后现有 effective_api_key / OCR 读取逻辑自动生效；
- 文件含敏感信息：.gitignore 已覆盖（data/ 不入库）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import AppConfig

SECRET_KEYS = ("EMBEDDING_API_KEY", "OCR_API_KEY", "GITHUB_TOKEN")


def secrets_path(cfg: Optional["AppConfig"] = None) -> Path:
    if cfg is not None:
        return cfg.resolve("data/secrets.local.json")
    return Path("data/secrets.local.json")


def load_secrets(cfg: Optional["AppConfig"] = None) -> dict[str, str]:
    """读取密钥文件；已设置的环境变量优先（不覆盖）。返回 {KEY: value}。"""
    p = secrets_path(cfg)
    out: dict[str, str] = {}
    if not p.exists():
        return out
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    for k, v in (data or {}).items():
        if k in SECRET_KEYS and isinstance(v, str) and v:
            out[k] = v
            if not os.environ.get(k):
                os.environ[k] = v
    return out


def save_secret(cfg: Optional["AppConfig"], key: str, value: str) -> None:
    """写入/更新单个密钥（merge 到 secrets 文件；空值表示删除）。"""
    if key not in SECRET_KEYS:
        raise ValueError(f"不支持的密钥: {key}（支持 {SECRET_KEYS}）")
    p = secrets_path(cfg)
    data: dict = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    value = (value or "").strip()
    if value:
        data[key] = value
    else:
        data.pop(key, None)
        os.environ.pop(key, None)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
