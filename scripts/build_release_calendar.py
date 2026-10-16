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
    python scripts/build_release_calendar.py --insecure          # 真实业务环境：跳过 SSL 证书校验
    python scripts/build_release_calendar.py --github-base http://mirror/api/v3
        # GitHub API 镜像（http/https 均可）；亦可用环境变量 VLLM_KB_GITHUB_BASE / VLLM_KB_INSECURE
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402

_UA = {"User-Agent": "vllm-kb"}


class CalendarFetchError(RuntimeError):
    """releases 拉取失败（网络/证书/限流/镜像不可用）。

    **绝不能**用它的结果覆盖已有日历：抓不到 ≠ 没有 release。实测（本地代理掐 TLS）
    一次失败就把带 v0.23.0 的日历写成 `"releases": []`，下游 w_ver 与图里
    MERGED_IN→release 关联整体消失，而且**静默**（只有一行 warn）。
    """

    def __init__(self, repo: str, page: int, cause: Exception, partial: list[dict]):
        super().__init__(f"{repo} releases 拉取失败（page {page}）: {cause}")
        self.repo = repo
        self.page = page
        self.cause = cause
        self.partial = partial


class CalendarWriteRefused(RuntimeError):
    """写盘会丢数据（新结果为空而现有日历非空）→ 拒绝覆盖。"""


def _github_token(cfg: AppConfig) -> str:
    """GitHub token（未认证 60 次/小时/IP，**共享代理出口 IP 极易被限流**；带 token 5000/h）。

    取 github 来源配置的 token_env 指向的环境变量（config 加载时已自动注入）。
    """
    import os

    for sc in cfg.effective_sources():
        if sc.type == "github":
            return os.environ.get(sc.get("token_env", "GITHUB_TOKEN"), "") or ""
    return os.environ.get("GITHUB_TOKEN", "")


def fetch_releases(repo: str, insecure: bool = False, gbase: str = "https://api.github.com",
                   token: str = "") -> list[dict]:
    """GitHub Releases API 全量拉取（分页）。返回 [{tag, date, prerelease}...] 按日期降序。

    走 **requests**（get_session：代理掐 TLS 自动直连兜底）——**不用 urllib**：实测
    （2026-10-16 deploy）同一台机器上 urllib/OpenSSL 的 TLS ClientHello 被环境按指纹
    掐断（api.github.com 的 vllm-ascend releases 路径 7/7 `EOF occurred in violation
    of protocol`，代理与直连全灭、加重试也无解；requests 同 URL 稳定 200）。

    **任何非 200 / 非 release 列表形状的响应都算拉取失败**（实测第二坑：共享代理出口
    IP 撞 GitHub 限流，403 + JSON dict 体被当成正常批次，遍历 dict 得到字符串 key、
    `rel.get` 裸 AttributeError 炸穿——状态码/形状必须显式校验）。
    token 仅在走官方 api.github.com 时附加（**不向镜像前缀泄凭证**）。

    任何一页失败都抛 CalendarFetchError（带已抓到的部分），**不返回残缺列表**：
    部分结果若落盘，会让日历比原来更小（旧版本文档的 tag->日期映射丢失），
    且与"真失败"无法区分。宁可保留上一版完整日历，下次再拉。
    """
    from vllm_kb.net import get_session, parse_json

    headers = dict(_UA)
    if token and gbase.rstrip("/") == "https://api.github.com":
        headers["Authorization"] = f"Bearer {token}"
    s = get_session(insecure)
    releases: list[dict] = []
    page = 1
    while True:
        url = f"{gbase.rstrip('/')}/repos/{repo}/releases?per_page=100&page={page}"
        try:
            r = s.get(url, timeout=30, headers=headers)
            if r.status_code != 200:
                snippet = (r.text or "").strip().replace("\n", " ")[:160]
                raise RuntimeError(f"HTTP {r.status_code}: {snippet or '<空响应体>'}")
            batch = parse_json(r, f"{repo} releases page {page}")
            if not isinstance(batch, list) or any(not isinstance(x, dict) for x in batch):
                raise RuntimeError(f"响应不是 release 列表（限流/错误体?）：{str(batch)[:160]}")
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
        except Exception as e:
            print(f"[warn] {repo} releases 拉取失败（page {page}）: {e}")
            raise CalendarFetchError(repo, page, e, releases) from e
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


def build_calendar(repo: str, insecure: bool = False, gbase: str = "https://api.github.com",
                   token: str = "") -> dict:
    releases = fetch_releases(repo, insecure=insecure, gbase=gbase, token=token)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": repo,
        "releases": [
            {**r, "kind": classify_version(r["tag"])}
            for r in releases
        ],
    }


def calendar_output_path(cfg: AppConfig, calendar: dict, path: Path | None = None) -> Path:
    """日历输出路径：--all-repos 时每仓库独立文件 release_calendar.{repo}.json，
    单仓库默认 data/compatibility/release_calendar.json。"""
    if path is not None:
        return path
    repo_slug = calendar.get("repo", "").replace("/", "-")
    if repo_slug:
        return cfg.resolve(f"data/compatibility/release_calendar.{repo_slug}.json")
    return cfg.resolve("data/compatibility/release_calendar.json")


def would_drop_releases(out: Path, calendar: dict) -> str:
    """写盘前的丢数据检查。返回拒绝原因，空串 = 可以写。

    只拦"新结果为空、现有非空"这一种**确定**的丢数据：抓取失败（见
    CalendarFetchError）与"该仓库真的没有 release"在结果上无从区分，所以
    宁可保留旧日历并告警。现有文件缺失/为空/损坏时不拦（首次生成照常）。
    """
    new_n = len(calendar.get("releases") or [])
    if new_n:
        return ""
    try:
        old = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    old_n = len((old or {}).get("releases") or [])
    if old_n:
        return (f"新结果 0 个 release，而现有 {out.name} 有 {old_n} 个——"
                f"拒绝覆盖（抓取失败 ≠ 没有 release）")
    return ""


def write_calendar(cfg: AppConfig, calendar: dict, path: Path | None = None,
                   force: bool = False) -> Path:
    """写日历。默认受保护：会用空结果清掉现有 release 时抛 CalendarWriteRefused。"""
    out = calendar_output_path(cfg, calendar, path)
    if not force:
        reason = would_drop_releases(out, calendar)
        if reason:
            raise CalendarWriteRefused(reason)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calendar, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    from vllm_kb.net import add_insecure_args, github_api_base, insecure_from_env

    ap = argparse.ArgumentParser(description="生成版本日历（GitHub Releases API）")
    ap.add_argument("--repo", default=None, help="仓库（默认 config.code.repo）")
    ap.add_argument("--all-repos", action="store_true", help="vllm-ascend + vllm 两个仓库")
    ap.add_argument("--config", default=None)
    add_insecure_args(ap)
    args = ap.parse_args()

    insecure = args.insecure or insecure_from_env()
    gbase = github_api_base(args.github_base)
    if insecure:
        print("[calendar] --insecure：跳过 SSL 证书校验（真实业务环境）")
    if gbase != "https://api.github.com":
        print(f"[calendar] GitHub API 镜像 {gbase}")

    cfg = AppConfig.load(args.config)
    repos = []
    if args.all_repos:
        repos = ["vllm-project/vllm-ascend", "vllm-project/vllm"]
    else:
        repos = [args.repo or cfg.code.repo]

    failed: list[str] = []
    token = _github_token(cfg)
    if token:
        print("[calendar] 已附加 GitHub token（限流 5000/h）")
    for repo in repos:
        print(f"[calendar] 拉取 {repo} releases ...", flush=True)
        try:
            cal = build_calendar(repo, insecure=insecure, gbase=gbase, token=token)
        except CalendarFetchError as e:
            print(f"[warn] {repo}: {e}；**保留上一版日历不覆盖**", flush=True)
            failed.append(repo)
            continue
        try:
            out = write_calendar(cfg, cal)
        except CalendarWriteRefused as e:
            print(f"[warn] {repo}: {e}；**保留上一版日历不覆盖**", flush=True)
            failed.append(repo)
            continue
        kinds = {}
        for r in cal["releases"]:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        n_rel = len(cal["releases"])
        print(f"[calendar] {repo}: {n_rel} 个 release（{kinds}）-> {out}")
        if n_rel:
            print(f"[calendar] 最近 5 个:")
            for r in cal["releases"][:5]:
                print(f"  {r['tag']:14s} {r['date'][:10]}  {r['kind']}")
    if failed:
        print(f"[calendar] 失败 {len(failed)}/{len(repos)}：{', '.join(failed)}"
              f"（旧日历已保留；检查代理/证书，必要时 --insecure 或 VLLM_KB_GITHUB_BASE）")
        sys.exit(1)


if __name__ == "__main__":
    main()
