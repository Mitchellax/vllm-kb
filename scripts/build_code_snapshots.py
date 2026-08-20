"""预存版本化代码仓快照：下载主要镜像版本源码 zip -> 解压 -> 构建符号索引。

用法（在项目根）：
    python scripts/build_code_snapshots.py                 # 预存 config.code.versions 或全部 tag
    python scripts/build_code_snapshots.py --version v0.23.0rc1   # 只预存指定版本
    python scripts/build_code_snapshots.py --index-only    # 已下载的版本只重建符号索引
    python scripts/build_code_snapshots.py --all           # 预存仓库全部 tag（41 个，~570MB）

下载后每个版本 zip 存 data/code/zips/{version}.zip（~14MB），
首次访问时按需解压到 data/code/snapshots/{version}/，符号索引建在 data/code/index.sqlite3。
幂等：已下载的版本跳过，重跑只补缺失。
"""
import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.code_index import VersionedCode  # noqa: E402
from vllm_kb.config import AppConfig  # noqa: E402


def list_tags(repo: str) -> list[str]:
    """GitHub REST 列出全部 tag（无 git 依赖）。"""
    tags: list[str] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/tags?per_page=100&page={page}"
        req = urllib.request.Request(url, headers={"User-Agent": "vllm-kb"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
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


def download_zip(repo: str, version: str, dest: Path) -> bool:
    """下载版本源码 zip 到 dest（幂等：已存在跳过）。"""
    if dest.exists() and dest.stat().st_size > 1000:
        return False
    url = f"https://codeload.github.com/{repo}/zip/refs/tags/{version}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[code] 下载 {version} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-kb"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
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
    ap.add_argument("--index-only", action="store_true", help="只重建索引，不下载")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    code = VersionedCode(cfg)
    repo = cfg.code.repo

    if args.index_only:
        for v in code.available_versions:
            index_version(code, v)
        build_symbol_table(cfg)
        print(f"[code] 索引重建完成，可用版本: {code.available_versions}")
        return

    if args.version:
        versions = args.version
    elif args.all:
        versions = list_tags(repo)
        print(f"[code] 全部 tag: {len(versions)} 个")
    elif cfg.code.versions:
        versions = list(cfg.code.versions)
    else:
        versions = list_tags(repo)
        print(f"[code] config 未指定，取全部 tag: {len(versions)} 个")

    # 1) 下载 zip（幂等）
    downloaded = 0
    for v in versions:
        if download_zip(repo, v, code.zips_dir / f"{v}.zip"):
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
