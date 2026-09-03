"""从 quay.io 拉取 ascend/vllm-ascend 镜像 tag，辅助维护组件配套矩阵。

看护策略：
- **排除**：日构建（nightly/ntightly/daily）、仓库分支（releases-v0.13.0 等）、
  主干/最新（main/latest/develop/master）、dev 构建 —— 这些 tag 内容可能随构建变化，
  用它们做版本匹配会误导，不看护；
- **看护**：其余全部 —— 正式版、rc 版、以及**模型专属镜像**（0day 适配，如 glm5、
  deepseekv4、kimi-k3、bailing-flash-*、DeepSeekV4-flash-0731，虽不长期推荐但仍需跟踪）；
- quay 的 tag 列表不含组件版本信息（需看镜像 build history / 镜像内环境），
  因此本脚本只负责给出候选 tag 清单，具体配套版本仍需人工核对后填入
  data/compatibility/vllm-ascend.json（脚本无法解析时可纯人工填写）；
- 全程打印运行状态：分页进度 / 重试 / 过滤统计 / 耗时。

用法：
    python scripts/fetch_quay_tags.py
    python scripts/fetch_quay_tags.py --json-out data/compatibility/quay_tags.json
    python scripts/fetch_quay_tags.py --timeout 60 --max-retries 5
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API = "https://quay.io/api/v1/repository/ascend/vllm-ascend/tag/"

# 排除：日构建 / 仓库分支 / 主干 / dev 构建（内容可能随构建变化，版本匹配会误导）
_EXCLUDE_RE = re.compile(
    r"(^|[-_])(nightly|ntightly|daily)([-_]|$)"  # 日构建：nightly-main、nightly-releases-v0.18.0
    r"|(^|[-_])releases([-_]|$)"  # 仓库分支：releases-v0.13.0
    r"|(^|[-_])(main|latest|develop|master|dev)([-_]|$)",  # 主干/最新/dev 构建
    re.IGNORECASE,
)

# 版本型 tag（正式版/rc 及其平台变体）：v0.18.0、v0.18.0rc1、v0.18.0-a3-openeuler ...
VERSION_RE = re.compile(r"^v?\d+\.\d+\.\d+")


def _tags_url(base: str) -> str:
    """quay tag 列表 URL（base 可换业务侧镜像前缀，默认 https://quay.io）。"""
    return f"{base.rstrip('/')}/api/v1/repository/ascend/vllm-ascend/tag/"


def fetch_tags(timeout: int = 30, max_retries: int = 3, insecure: bool = False,
               base: str = "https://quay.io") -> list[dict]:
    """分页拉取全部活跃 tag；带重试与逐页进度日志。"""
    from vllm_kb.net import get_session

    url = _tags_url(base)
    session = get_session(insecure)
    tags: list[dict] = []
    page = 1
    start = time.time()
    print(
        f"[quay] 开始拉取 {url}（onlyActiveTags=true, timeout={timeout}s, 最大重试 {max_retries} 次"
        + (", 跳过SSL校验" if insecure else "") + "）",
        flush=True,
    )
    while True:
        params = {"limit": 100, "onlyActiveTags": True, "page": page}
        r = None
        for attempt in range(max_retries + 1):
            try:
                r = session.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:
                if attempt == max_retries:
                    raise RuntimeError(f"page {page} 请求异常（已重试 {max_retries} 次）: {e}") from e
                wait = 2 ** attempt
                print(f"[quay] page {page} 请求异常: {e}，{wait}s 后重试（{attempt + 1}/{max_retries}）", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 502, 503):
                if attempt == max_retries:
                    raise RuntimeError(f"page {page} HTTP {r.status_code}（已重试 {max_retries} 次）")
                wait = 2 ** attempt
                print(
                    f"[quay] page {page} HTTP {r.status_code}，{wait}s 后重试（{attempt + 1}/{max_retries}）",
                    flush=True,
                )
                time.sleep(wait)
                continue
            r.raise_for_status()  # 其他 4xx：立即失败，不重试

        data = r.json()
        batch = data.get("tags", []) or []
        tags.extend(batch)
        has_more = bool(data.get("has_additional", False))
        print(f"[quay] page {page} 完成：+{len(batch)} 个 tag（累计 {len(tags)}，还有更多: {has_more}）", flush=True)
        if not has_more:
            break
        page += 1

    elapsed = time.time() - start
    print(f"[quay] 拉取完成：共 {len(tags)} 个 tag，耗时 {elapsed:.1f}s", flush=True)
    return tags


def is_managed(tag_name: str) -> bool:
    """看护规则：排除日构建/仓库分支/主干/dev 构建；其余（正式版、rc、模型专属镜像）都看护。"""
    return not _EXCLUDE_RE.search(tag_name)


def categorize_excluded(tag_name: str) -> str:
    """对不维护的 tag 分类，便于核对过滤规则是否符合预期。"""
    low = tag_name.lower()
    if any(k in low for k in ("nightly", "ntightly", "daily")):
        return "nightly(日构建)"
    if low.startswith("releases") or "-releases" in low:
        return "branch(releases-*)"
    if low in ("latest", "develop", "master") or low.startswith("main") or low.startswith("latest"):
        return "主干/最新"
    if low.startswith("dev") or "-dev" in low or "_dev" in low:
        return "dev构建"
    return "其他"


def fmt_ts(ts) -> str:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return "-"


def main() -> None:
    from vllm_kb.net import add_insecure_args, insecure_from_env, quay_base

    ap = argparse.ArgumentParser(description="拉取 quay.io ascend/vllm-ascend tag 清单")
    ap.add_argument("--json-out", default=None, help="把候选 tag 写入 JSON（骨架，供人工填写配套）")
    ap.add_argument("--all", action="store_true", help="列出全部 tag（含 nightly/release 等）")
    ap.add_argument("--timeout", type=int, default=30, help="单请求超时秒数")
    ap.add_argument("--max-retries", type=int, default=3, help="每页最大重试次数")
    add_insecure_args(ap)
    args = ap.parse_args()

    insecure = args.insecure or insecure_from_env()
    base = quay_base(args.quay_base)
    start = time.time()
    try:
        tags = fetch_tags(timeout=args.timeout, max_retries=args.max_retries,
                          insecure=insecure, base=base)
    except Exception as e:
        print(f"[quay] 拉取失败: {e}")
        print("[quay] 可改为人工核对：访问 https://quay.io/repository/ascend/vllm-ascend?tab=tags")
        sys.exit(1)

    managed = [t for t in tags if is_managed(t["name"])]
    excluded = [t for t in tags if not is_managed(t["name"])]
    buckets: dict[str, int] = {}
    for t in excluded:
        cat = categorize_excluded(t["name"])
        buckets[cat] = buckets.get(cat, 0) + 1
    bucket_detail = ", ".join(f"{k}={v}" for k, v in sorted(buckets.items()))
    version_tags = sorted([t for t in managed if VERSION_RE.match(t["name"])], key=lambda t: t["name"])
    model_tags = sorted([t for t in managed if not VERSION_RE.match(t["name"])], key=lambda t: t["name"])
    print(
        f"[quay] 过滤：看护 {len(managed)} 个（版本型 {len(version_tags)} + 模型专属 {len(model_tags)}）；"
        f"排除 {len(excluded)} 个（{bucket_detail}）",
        flush=True,
    )

    def _print_tags(items: list[dict]) -> None:
        for t in items:
            digest = str(t.get("manifest_digest") or "")[:12]
            print(f"  {t['name']:<40} created={fmt_ts(t.get('start_ts'))} digest={digest}")

    print(f"\n[quay] 版本型候选（正式版/rc，{len(version_tags)} 个）：")
    _print_tags(version_tags)
    print(f"\n[quay] 模型专属候选（0day 适配，{len(model_tags)} 个）：")
    _print_tags(model_tags)
    if args.all:
        print(f"\n[quay] 全部 tag（共 {len(tags)} 个，含不维护的）：")
        for t in sorted(tags, key=lambda t: t["name"]):
            print(f"  {t['name']:<40} created={fmt_ts(t.get('start_ts'))}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_comment": "quay tag 候选清单（看护：正式版/rc + 模型专属 0day 适配；排除 nightly/分支/主干/dev）。"
                        "配套组件版本需人工核对后填入 vllm-ascend.json 的 rows",
            "source": "https://quay.io/repository/ascend/vllm-ascend?tab=tags",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "version_tags": [{"name": t["name"], "start_ts": t.get("start_ts"), "manifest_digest": t.get("manifest_digest")} for t in version_tags],
            "model_tags": [{"name": t["name"], "start_ts": t.get("start_ts"), "manifest_digest": t.get("manifest_digest")} for t in model_tags],
            "rows": [],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[quay] 已写入 {out}（版本型 {len(version_tags)} + 模型专属 {len(model_tags)}），"
            "请核对后把 rows 补进 data/compatibility/vllm-ascend.json",
            flush=True,
        )

    print(f"[quay] 完成，总耗时 {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
