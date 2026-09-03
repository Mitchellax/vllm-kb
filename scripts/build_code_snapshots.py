"""预存版本化代码仓快照：下载主要镜像版本源码 zip -> 解压 -> 构建符号索引。

用法（在项目根）：
    python scripts/build_code_snapshots.py                 # 预存 config.code.versions 或全部 tag
    python scripts/build_code_snapshots.py --version v0.23.0rc1   # 只预存指定版本
    python scripts/build_code_snapshots.py --index-only    # 已下载的版本只重建符号索引
    python scripts/build_code_snapshots.py --all           # 预存仓库全部 tag（41 个，~570MB）
    python scripts/build_code_snapshots.py --insecure      # 真实业务环境：跳过 SSL 证书校验
    python scripts/build_code_snapshots.py --base-url http://mirror:8080   # 业务侧 http 镜像源

真实业务环境（SSL 被禁/证书不受信）：
    - --insecure：urllib 用不校验证书的 context（自签证书/代理拦截时）；
    - --base-url：把下载源换成业务侧镜像（http 或 https 均可），默认 https://codeload.github.com；
    - 两者也可经环境变量 VLLM_KB_INSECURE=1 / VLLM_KB_CODE_BASE=<url> 指定（脚本外统一配置）。

下载后每个版本 zip 存 data/code/zips/{version}.zip（~14MB），
首次访问时按需解压到 data/code/snapshots/{version}/，符号索引建在 data/code/index.sqlite3。
幂等：已下载的版本跳过，重跑只补缺失。
"""
import argparse
import json
import os
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.code_index import VersionedCode  # noqa: E402
from vllm_kb.config import AppConfig  # noqa: E402

DEFAULT_DOWNLOAD_BASE = "https://codeload.github.com"
DEFAULT_API_BASE = "https://api.github.com"


def _opener(insecure: bool) -> urllib.request.OpenerDirector:
    """按需构造跳过证书校验的 opener（真实业务环境自签证书/SSL 被禁时用）。"""
    if not insecure:
        return urllib.request.build_opener()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def _insecure_from_env() -> bool:
    return os.environ.get("VLLM_KB_INSECURE", "").strip().lower() in ("1", "true", "yes", "on")


def list_tags(repo: str, insecure: bool = False, api_base: str = DEFAULT_API_BASE) -> list[str]:
    """GitHub REST 列出全部 tag（无 git 依赖）。api_base 可换业务侧 API 镜像。"""
    tags: list[str] = []
    page = 1
    opener = _opener(insecure)
    while True:
        url = f"{api_base.rstrip('/')}/repos/{repo}/tags?per_page=100&page={page}"
        req = urllib.request.Request(url, headers={"User-Agent": "vllm-kb"})
        try:
            with opener.open(req, timeout=30) as r:
                batch = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"[warn] 列 tag 失败（page {page}）: {e}")
            break
        if not batch:
            break
        tags.extend(t["name"] for t in batch)
        if len(batch) < 100:
            break
        page += 1
    return tags


def download_zip(repo: str, version: str, dest: Path, insecure: bool = False,
                 base_url: str = DEFAULT_DOWNLOAD_BASE) -> bool:
    """下载版本源码 zip 到 dest（幂等：已存在跳过）。base_url 可换业务侧镜像源。"""
    if dest.exists() and dest.stat().st_size > 1000:
        return False
    url = f"{base_url.rstrip('/')}/{repo}/zip/refs/tags/{version}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[code] 下载 {version} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-kb"})
    try:
        with _opener(insecure).open(req, timeout=300) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        print(f"[warn] {version} 下载失败: {e}")
        dest.unlink(missing_ok=True)
        return False
    print(f"[code] {version} 下载完成 ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def index_version(code: VersionedCode, version: str) -> None:
    """确保版本解压并重建符号索引。"""
    try:
        snap = code.ensure_snapshot(version)
    except Exception as e:
        print(f"[warn] {version} 解压失败: {e}")
        return
    n = code.build_index_for_version(version)
    print(f"[code] {version} 符号索引 {n} 个符号（快照 {snap}）")


def build_symbol_table(cfg) -> None:
    """从全部已解压快照生成跨版本符号表（symbols.json），并统计社区高频信号词。"""
    from vllm_kb.symbol_table import (
        build_symbol_table_from_snapshots,
        save_symbol_table,
        _extract_models_and_versions_from_issues,
    )
    from vllm_kb.code_index import VersionedCode

    code = VersionedCode(cfg)
    table = build_symbol_table_from_snapshots(code.snapshots_dir)
    issue_dir = cfg.resolve("data/raw/vllm-ascend/issues")
    _extract_models_and_versions_from_issues(issue_dir, table)
    out = cfg.resolve(cfg.storage.code_root) / "symbols.json"
    save_symbol_table(table, out)
    print(f"[code] 符号表 {len(table.entries)} 个 -> {out}")

    # 社区高频信号词（供 agent 判断）
    from build_signal_words import collect_signal_words  # noqa: E402

    words = collect_signal_words(issue_dir)
    sig_out = cfg.resolve(cfg.storage.code_root) / "signal_words.json"
    sig_out.parent.mkdir(parents=True, exist_ok=True)
    sig_out.write_text(json.dumps({"note": "社区高频信号词（供 agent 判断，非自动过滤）",
                                   "words": words}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[code] 信号词 {len(words)} 个 -> {sig_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="预存版本化代码仓快照")
    ap.add_argument("--version", action="append", default=None, help="指定版本（可多次）")
    ap.add_argument("--all", action="store_true", help="预存全部 tag")
    ap.add_argument("--list", action="store_true",
                    help="只列出可用版本对比（GitHub 全部 tag / 已预存 / 缺失），不下载——"
                         "方便人工更新 config.code.versions")
    ap.add_argument("--index-only", action="store_true", help="只重建索引，不下载")
    ap.add_argument("--insecure", action="store_true",
                    help="跳过 SSL 证书校验（真实业务环境自签证书/SSL 被禁；亦可用环境变量 VLLM_KB_INSECURE=1）")
    ap.add_argument("--base-url", default=None,
                    help=f"下载源前缀（业务侧镜像，http/https 均可；默认 {DEFAULT_DOWNLOAD_BASE}；"
                         f"亦可用环境变量 VLLM_KB_CODE_BASE）")
    ap.add_argument("--api-base", default=None,
                    help=f"列 tag 的 API 地址（默认 {DEFAULT_API_BASE}；亦可用环境变量 VLLM_KB_CODE_API）")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    insecure = args.insecure or _insecure_from_env()
    base_url = args.base_url or os.environ.get("VLLM_KB_CODE_BASE", DEFAULT_DOWNLOAD_BASE)
    api_base = args.api_base or os.environ.get("VLLM_KB_CODE_API", DEFAULT_API_BASE)
    if insecure:
        print(f"[code] --insecure：跳过 SSL 证书校验（真实业务环境）")
    if base_url != DEFAULT_DOWNLOAD_BASE or api_base != DEFAULT_API_BASE:
        print(f"[code] 下载源 {base_url} / API {api_base}")

    cfg = AppConfig.load(args.config)
    code = VersionedCode(cfg)
    repo = cfg.code.repo

    if args.index_only:
        for v in code.available_versions:
            index_version(code, v)
        build_symbol_table(cfg)
        print(f"[code] 索引重建完成，可用版本: {code.available_versions}")
        return

    if args.list:
        all_tags = list_tags(repo, insecure=insecure, api_base=api_base)
        stored = set(code.available_versions)
        cfg_list = set(cfg.code.versions or [])
        missing = [t for t in all_tags if t not in stored]
        print(f"[code] 仓库 {repo} 版本对比（GitHub 全部 tag {len(all_tags)} 个）：")
        print(f"  已预存 {len(stored)} 个:")
        for v in sorted(stored):
            tag = "[config]" if v in cfg_list else ""
            print(f"    {v} {tag}")
        print(f"  缺失 {len(missing)} 个（未预存，可加到 config.code.versions 或 --version/--all 预存）:")
        for v in missing:
            print(f"    {v}")
        print(f"[code] config.code.versions 当前 {len(cfg_list)} 个: {sorted(cfg_list)}")
        return

    if args.version:
        versions = args.version
    elif args.all:
        versions = list_tags(repo, insecure=insecure, api_base=api_base)
        print(f"[code] 全部 tag: {len(versions)} 个")
    elif cfg.code.versions:
        versions = list(cfg.code.versions)
    else:
        versions = list_tags(repo, insecure=insecure, api_base=api_base)
        print(f"[code] config 未指定，取全部 tag: {len(versions)} 个")

    # 1) 下载 zip（幂等）
    downloaded = 0
    for v in versions:
        if download_zip(repo, v, code.zips_dir / f"{v}.zip", insecure=insecure, base_url=base_url):
            downloaded += 1
    if downloaded:
        print(f"[code] 本轮新下载 {downloaded} 个版本")

    # 2) 全部解压 + 建索引
    for v in versions:
        if (code.zips_dir / f"{v}.zip").exists():
            index_version(code, v)

    # 3) 生成跨版本符号表 + 社区高频信号词
    build_symbol_table(cfg)

    print(f"[code] 完成。可用版本: {code.available_versions}")


if __name__ == "__main__":
    main()
