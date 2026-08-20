"""生成版本日历：GitHub Releases API -> data/compatibility/release_calendar.json。

版本日历内容：
    {
      "generated_at": "...",
      "repo": "vllm-project/vllm-ascend",
      "releases": [
        {"tag": "v0.23.0", "date": "2026-08-16T22:18:14Z", "prerelease": false, "kind": "release"},
        {"tag": "v0.23.0rc1", "date": "2026-07-19T13:55:17Z", "prerelease": true, "kind": "rc"}
      ]
    }

用途：
1. **置信度版本上界**（confidence.py version_at_date）：resolved_at -> 该日期前最近发布的版本，
   让 w_ver 从"只有 min 下界"升级为"[min, max] 区间"（修复落地版本上界）；
2. **版本形态判断**（version_kind）：正式 release vs rc/pre——用于故障分析时判断
   "0.18.0 是正式版"（用户部署形态影响修复 backport 判断）；
3. **组件配套**：与 companion 矩阵（vllm-ascend -> vllm/cann 版本）联动。

用法：
    python scripts/build_release_calendar.py                     # 默认 vllm-ascend（config.code.repo）
    python scripts/build_release_calendar.py --repo vllm-project/vllm
    python scripts/build_release_calendar.py --all-repos
"""
import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402

_UA = {"User-Agent": "vllm-kb"}


def fetch_releases(repo: str) -> list[dict]:
    """GitHub Releases API 全量拉取（分页）。返回 [{tag, date, prerelease}...] 按日期降序。"""
    releases: list[dict] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                batch = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"[warn] {repo} releases 拉取失败（page {page}）: {e}")
            break
        if not batch:
            break
        for rel in batch:
            if rel.get("tag_name") and rel.get("published_at"):
                releases.append({
                    "tag": rel["tag_name"],
                    "date": rel["published_at"],
                    "prerelease": bool(rel.get("prerelease")),
                })
        if len(batch) < 100:
            break
        page += 1
    # 按日期降序
    releases.sort(key=lambda r: r["date"], reverse=True)
    return releases


def classify_version(tag: str) -> str:
    """版本形态分类：release（正式）| rc（预发布）| pre（早期预发布）。

    规则（vllm/vllm-ascend 惯例：vX.Y.Z 正式，vX.Y.ZrcN 预发布）：
    - 含 'rc' -> rc
    - 含 'alpha'/'beta'/'dev'/'post' -> pre
    - 其余纯 semver -> release
    """
    t = tag.lower()
    if "rc" in t:
        return "rc"
    if any(k in t for k in ("alpha", "beta", "dev", "pre", "post")):
        return "pre"
    return "release"


def build_calendar(repo: str) -> dict:
    releases = fetch_releases(repo)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": repo,
        "releases": [
            {**r, "kind": classify_version(r["tag"])}
            for r in releases
        ],
    }


def write_calendar(cfg: AppConfig, calendar: dict, path: Path | None = None) -> Path:
    """写日历：--all-repos 时每仓库独立文件 release_calendar.{repo}.json，
    单仓库默认 data/compatibility/release_calendar.json。"""
    if path is None:
        repo_slug = calendar.get("repo", "").replace("/", "-")
        if repo_slug:
            out = cfg.resolve(f"data/compatibility/release_calendar.{repo_slug}.json")
        else:
            out = cfg.resolve("data/compatibility/release_calendar.json")
    else:
        out = path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calendar, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="生成版本日历（GitHub Releases API）")
    ap.add_argument("--repo", default=None, help="仓库（默认 config.code.repo）")
    ap.add_argument("--all-repos", action="store_true", help="vllm-ascend + vllm 两个仓库")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    repos = []
    if args.all_repos:
        repos = ["vllm-project/vllm-ascend", "vllm-project/vllm"]
    else:
        repos = [args.repo or cfg.code.repo]

    for repo in repos:
        print(f"[calendar] 拉取 {repo} releases ...", flush=True)
        cal = build_calendar(repo)
        out = write_calendar(cfg, cal)
        kinds = {}
        for r in cal["releases"]:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        n_rel = len(cal["releases"])
        print(f"[calendar] {repo}: {n_rel} 个 release（{kinds}）-> {out}")
        if n_rel:
            print(f"[calendar] 最近 5 个:")
            for r in cal["releases"][:5]:
                print(f"  {r['tag']:14s} {r['date'][:10]}  {r['kind']}")


if __name__ == "__main__":
    main()
