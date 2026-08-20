"""自动生成/更新 vllm-ascend 组件配套矩阵（quay 镜像 + GitHub release 自动匹配）。

数据源与自动匹配规则：
1. quay tag 清单（复用 fetch_quay_tags）→ 看护 tag → 按基础版本去重（去平台后缀 -a3/-310p/-openeuler）；
2. 镜像 config Env → **cann** 版本（路径里 cann-X.Y.Z）、SOC 型号、python 版本（进 notes）；
3. vllm-ascend GitHub release 说明 → **vllm** 配套版本
   （"aligns ... with upstream vLLM v0.23.0" / "based on vLLM v0.19.1"；
    无说明时启发式：vllm-ascend 版本号跟踪上游 vllm，剥 rc 后缀）。

自动无法确定的（pytorch / pytorch-ascend / npu-driver 等）→ 告警并留空，人工修复：
人工已填写的字段优先保留（merge 时自动只填空字段）。

用法：
    python scripts/build_companion_matrix.py                 # 下载+匹配+写回+缺口报告
    python scripts/build_companion_matrix.py --strict        # 存在缺口时退出码 1（CI 用）
    python scripts/build_companion_matrix.py --no-write      # 只报告不写回
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_quay_tags as fq  # noqa: E402

API = "https://quay.io/api/v1/repository/ascend/vllm-ascend"
V2 = "https://quay.io/v2/ascend/vllm-ascend"
GITHUB_RELEASES = "https://api.github.com/repos/vllm-project/vllm-ascend/releases"

COMPANION_FIELDS = ["vllm", "cann", "pytorch", "pytorch-ascend", "npu-driver"]
# npu-driver(HDK) 与镜像版本不耦合（特定 HDK 有特定问题），缺失符合预期，不计入缺口告警
REQUIRED_FIELDS = ["vllm", "cann", "pytorch", "pytorch-ascend"]

PLATFORM_SUFFIXES = ("-openeuler", "-310p", "-a3", "-a5", "-a2")
_CANN_RE = re.compile(r"cann-(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
_SOC_RE = re.compile(r"^SOC_VERSION=(.+)$")
_PYTHON_RE = re.compile(r"python(\d+\.\d+(?:\.\d+)?)")
_UPSTREAM_VLLM_RE = re.compile(
    r"(?i)(?:upstream\s+vllm|based\s+on\s+vllm|align(?:ing|s)?\s+the\s+plugin\s+with\s+upstream\s+vllm)"
    r"[^\d]*v?(\d+\.\d+(?:\.\d+)?)"
)
_VLLM_ANY_RE = re.compile(r"(?i)\bvllm\s+v?(\d+\.\d+(?:\.\d+)?)")
_BASE_VERSION_RE = re.compile(r"^v?(\d+\.\d+(?:\.\d+)?)")


# ---------------- tag 处理 ----------------

def strip_platform_suffix(tag: str) -> str:
    """去掉平台/OS 后缀，得到基础版本（v0.18.0-a3-openeuler -> v0.18.0；kimi-k3-a3 -> kimi-k3）。"""
    base = tag
    changed = True
    while changed:
        changed = False
        for suf in PLATFORM_SUFFIXES:
            if base.endswith(suf):
                base = base[: -len(suf)]
                changed = True
    return base


def group_base_versions(tags: list[dict]) -> dict[str, list[dict]]:
    """看护 tag 按基础版本分组（同一版本的平台变体归一组）。"""
    groups: dict[str, list[dict]] = {}
    for t in tags:
        if not fq.is_managed(t["name"]):
            continue
        groups.setdefault(strip_platform_suffix(t["name"]), []).append(t)
    return groups


def pick_representative(group: list[dict]) -> dict:
    """优先取无平台后缀的原名 tag（如 v0.18.0），否则取排序后第一个。"""
    base = strip_platform_suffix(group[0]["name"])
    for t in group:
        if t["name"] == base:
            return t
    return sorted(group, key=lambda t: t["name"])[0]


# ---------------- quay 镜像 env ----------------

def get_quay_token() -> str:
    import requests

    r = requests.get(
        "https://quay.io/v2/auth",
        params={"service": "quay.io", "scope": "repository:ascend/vllm-ascend:pull"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["token"]


def fetch_image_env(tag_info: dict, token: str, timeout: int = 30, max_retries: int = 3) -> list[str]:
    """拉取镜像 config Env（manifest list -> amd64 子清单 -> config blob）。失败返回 []。"""
    import requests

    digest = tag_info["manifest_digest"]
    for attempt in range(max_retries + 1):
        try:
            m = requests.get(f"{API}/manifest/{digest}", timeout=timeout)
            m.raise_for_status()
            md = json.loads(m.json()["manifest_data"])
            if md.get("manifests"):
                arch = next(
                    (x for x in md["manifests"] if x["platform"].get("architecture") == "amd64"),
                    md["manifests"][0],
                )
                cm = requests.get(f"{API}/manifest/{arch['digest']}", timeout=timeout)
                cm.raise_for_status()
                md = json.loads(cm.json()["manifest_data"])
            cfg_digest = md["config"]["digest"]
            blob = requests.get(
                f"{V2}/blobs/{cfg_digest}",
                headers={"Authorization": "Bearer " + token},
                timeout=timeout,
            )
            blob.raise_for_status()
            env = blob.json().get("config", {}).get("Env", []) or []
            return [str(e) for e in env]
        except Exception as e:
            if attempt == max_retries:
                print(f"[matrix] 拉取 {tag_info['name']} 镜像 env 失败: {e}")
                return []
            wait = 2 ** attempt
            print(f"[matrix] {tag_info['name']} 拉取失败({e})，{wait}s 后重试", flush=True)
            time.sleep(wait)
    return []


def extract_from_env(env: list[str]) -> dict[str, str]:
    """从镜像 Env 提取 cann 版本 / SOC 型号 / python 版本。"""
    out = {"cann": "", "soc": "", "python": ""}
    for e in env:
        if not out["cann"]:
            m = _CANN_RE.search(e)
            if m:
                out["cann"] = m.group(1)
        if not out["soc"]:
            m = _SOC_RE.match(e)
            if m:
                out["soc"] = m.group(1)
        if not out["python"]:
            m = _PYTHON_RE.search(e)
            if m and "PATH" in e:
                out["python"] = m.group(1)
    return out


# ---------------- GitHub release 说明 ----------------

def fetch_releases() -> dict[str, str]:
    """拉取 vllm-ascend release 说明，返回 {tag_name: body}。失败返回 {}。"""
    import requests

    try:
        r = requests.get(GITHUB_RELEASES, params={"per_page": 100}, headers={"User-Agent": "vllm-kb"}, timeout=30)
        r.raise_for_status()
        return {rel.get("tag_name", ""): (rel.get("body") or "") for rel in r.json()}
    except Exception as e:
        print(f"[matrix] 拉取 GitHub release 说明失败: {e}")
        return {}


def extract_vllm_from_release(tag: str, body: str) -> tuple[str, str]:
    """从 release 说明提取配套 vllm 版本，返回 (version, 来源说明)。

    优先显式声明（"upstream vLLM v0.23.0"），无说明时启发式：
    vllm-ascend 版本号跟踪上游 vllm（v0.19.1rc1 -> vllm 0.19.1）。
    """
    m = _UPSTREAM_VLLM_RE.search(body)
    if m:
        return m.group(1), "release说明(upstream)"
    m = _VLLM_ANY_RE.search(body)
    if m:
        return m.group(1), "release说明"
    m = _BASE_VERSION_RE.match(tag)
    if m:
        return m.group(1), "启发式(版本号跟踪上游)"
    return "", ""


# ---------------- 构建与合并 ----------------

def build_rows(groups: dict, releases: dict[str, str], token: str) -> list[dict]:
    rows = []
    total = len(groups)
    for i, base in enumerate(sorted(groups), 1):
        rep = pick_representative(groups[base])
        env = fetch_image_env(rep, token)
        info = extract_from_env(env)
        rel_body = releases.get(base, "")
        vllm_ver, vllm_src = extract_vllm_from_release(base, rel_body) if rel_body else extract_vllm_from_release(base, "")
        notes = []
        if info["soc"]:
            notes.append(f"SOC={info['soc']}")
        if info["python"]:
            notes.append(f"python={info['python']}")
        provenance = []
        if info["cann"]:
            provenance.append("cann=镜像env")
        if vllm_src:
            provenance.append(f"vllm={vllm_src}")
        rows.append(
            {
                "vllm-ascend": base,
                "vllm": vllm_ver,
                "cann": info["cann"],
                "pytorch": "",
                "pytorch-ascend": "",
                "npu-driver": "",
                "notes": "; ".join(notes),
                "source": "自动(" + "+".join(provenance) + ")" if provenance else "待人工",
            }
        )
        flag = "ok" if (info["cann"] or vllm_ver) else "gap"
        print(
            f"[matrix] ({i}/{total}) {base:<28} cann={info['cann'] or '-':<8} vllm={vllm_ver or '-':<8} "
            f"{vllm_src or ''} [{flag}]",
            flush=True,
        )
    return rows


def merge_with_manual(auto_rows: list[dict], manual_rows: list[dict]) -> list[dict]:
    """合并：人工行优先（非空字段保留），自动只填空字段；quay 列表外的人工行保留。"""
    manual_by_key = {r.get("vllm-ascend", ""): r for r in manual_rows}
    merged: list[dict] = []
    seen: set[str] = set()
    for a in auto_rows:
        key = a["vllm-ascend"]
        seen.add(key)
        m = manual_by_key.get(key)
        if m:
            merged.append(
                {
                    "vllm-ascend": key,
                    "vllm": m.get("vllm") or a["vllm"],
                    "cann": m.get("cann") or a["cann"],
                    "pytorch": m.get("pytorch") or "",
                    "pytorch-ascend": m.get("pytorch-ascend") or "",
                    "npu-driver": m.get("npu-driver") or "",
                    "notes": m.get("notes") or a["notes"],
                    "source": m.get("source") or a["source"],
                }
            )
        else:
            merged.append(a)
    for key, m in manual_by_key.items():
        if key and key not in seen:
            merged.append(m)
    merged.sort(key=lambda r: r["vllm-ascend"])
    return merged


def report_gaps(rows: list[dict]) -> int:
    """缺口报告：每行缺失的必需配套组件（npu-driver/HDK 可选，不计缺口）；返回缺口行数。"""
    filled = {f: sum(1 for r in rows if r.get(f)) for f in COMPANION_FIELDS}
    print("[matrix] 自动匹配统计: " + ", ".join(f"{k}={v}" for k, v in filled.items()))
    gaps: list[tuple[str, list[str]]] = []
    for r in rows:
        missing = [f for f in REQUIRED_FIELDS if not r.get(f)]
        if missing:
            gaps.append((r["vllm-ascend"], missing))
    if gaps:
        print(f"[matrix] [!] 缺口 {len(gaps)}/{len(rows)} 行，需人工修复（对照 quay build history 或内部版本记录）：")
        for key, missing in gaps:
            print(f"    {key:<28} 缺: {', '.join(missing)}")
    else:
        print("[matrix] 必需配套完整（npu-driver/HDK 与镜像解耦，缺失属预期）")
    return len(gaps)


def default_matrix_path() -> str:
    try:
        cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
        return cfg.get("storage", {}).get("companion_file", "data/compatibility/vllm-ascend.json")
    except Exception:
        return "data/compatibility/vllm-ascend.json"


def suggest_from_issues(canonical_path: str | Path, min_issues: int = 1) -> None:
    """从已入库的 vllm-ascend issue 数据统计"某版本上真实部署的配套组合"（众数）。

    仅作人工核对的参考：issue 里的版本是真实部署环境（pip freeze 段），
    不是官方配套声明。镜像 build history 为空（buildkit 构建不留记录），
    这是当前唯一能自动获取 torch/torch_npu 配套的途径。
    """
    from collections import Counter

    from vllm_kb.github_pull import load_canonical

    docs = load_canonical(canonical_path)
    groups: dict[str, list] = {}
    for d in docs:
        if d.component != "vllm-ascend" or not d.version_span.min:
            continue
        groups.setdefault(d.version_span.min, []).append(d)
    print("[matrix] issue 数据参考（真实部署众数，非官方配套，仅供人工核对）：")
    shown = 0
    for ver in sorted(groups):
        docs_v = groups[ver]
        if len(docs_v) < min_issues:
            continue
        parts = [f"n={len(docs_v)}"]
        for field in ("vllm", "cann", "pytorch", "pytorch-ascend", "npu-driver"):
            vals = [d.component_versions.get(field) for d in docs_v if d.component_versions.get(field)]
            if vals:
                mode, cnt = Counter(vals).most_common(1)[0]
                parts.append(f"{field}={mode}({cnt})")
        print(f"    {ver:<24} " + " | ".join(parts))
        shown += 1
    if not shown:
        print("    (无足够带版本的 vllm-ascend issue 数据)")


def main() -> None:
    ap = argparse.ArgumentParser(description="自动生成/更新 vllm-ascend 组件配套矩阵")
    ap.add_argument("--matrix", default=None, help="矩阵文件路径（默认取 config.json 的 storage.companion_file）")
    ap.add_argument("--strict", action="store_true", help="存在缺口时以退出码 1 结束（CI 用）")
    ap.add_argument("--no-write", action="store_true", help="只报告不写回")
    ap.add_argument("--suggest-from-issues", nargs="?", const="data/raw/canonical.jsonl", metavar="CANONICAL",
                    help="从 canonical 的 issue 数据统计配套参考（真实部署众数，不写回矩阵）")
    args = ap.parse_args()

    if args.suggest_from_issues:
        suggest_from_issues(args.suggest_from_issues)
        if args.no_write:
            return

    matrix_path = Path(args.matrix or default_matrix_path())
    start = time.time()

    # 现有手工行（人工填写的优先保留）
    manual_rows: list[dict] = []
    if matrix_path.exists():
        manual_rows = json.loads(matrix_path.read_text(encoding="utf-8")).get("rows", []) or []
    print(f"[matrix] 现有手工行 {len(manual_rows)} 条（非空字段将优先保留）", flush=True)

    # 1) quay tag
    tags = fq.fetch_tags()
    groups = group_base_versions(tags)
    print(f"[matrix] 看护 tag 分组为 {len(groups)} 个基础版本（含模型专属镜像）", flush=True)

    # 2) 镜像 env + 3) release 说明
    token = get_quay_token()
    releases = fetch_releases()
    print(f"[matrix] 获取 GitHub release 说明 {len(releases)} 条", flush=True)

    auto_rows = build_rows(groups, releases, token)
    merged = merge_with_manual(auto_rows, manual_rows)
    n_gaps = report_gaps(merged)

    if not args.no_write:
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_comment": "vllm-ascend 组件配套矩阵（自动生成 + 人工修复）。cann 来自镜像 env，vllm 来自 release 说明/启发式；"
                        "pytorch/pytorch-ascend/npu-driver 需人工核对 quay build history 后填入。",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": merged,
        }
        matrix_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[matrix] 已写回 {matrix_path}（{len(merged)} 行），总耗时 {time.time() - start:.1f}s", flush=True)

    if args.strict and n_gaps:
        sys.exit(1)


if __name__ == "__main__":
    main()
