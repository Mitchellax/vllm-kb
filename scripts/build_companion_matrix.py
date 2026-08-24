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
    python scripts/build_companion_matrix.py --insecure      # 内网：跳过 SSL 证书校验
    python scripts/build_companion_matrix.py --quay-base http://mirror:8080 --github-base http://mirror/api/v3
        # 内网 http 镜像（quay / GitHub API）；亦可用环境变量 VLLM_KB_QUAY_BASE / VLLM_KB_GITHUB_BASE / VLLM_KB_INSECURE
"""
import argparse
import json
import re
import socket
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
# 镜像 Env 里的 vllm tag（如 VLLM_TAG=v0.26.0）——比 release 说明更直接的配套证据
_VLLM_TAG_RE = re.compile(r"(?:^|[\s;])VLLM_TAG\s*=\s*v?(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
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

def _quay_api(qbase: str) -> str:
    """quay API 前缀（内网镜像时替换域名）。"""
    return f"{qbase.rstrip('/')}/api/v1/repository/ascend/vllm-ascend"


def _quay_v2(qbase: str) -> str:
    return f"{qbase.rstrip('/')}/v2/ascend/vllm-ascend"


def get_quay_token(insecure: bool = False, qbase: str = "https://quay.io") -> str:
    from vllm_kb.net import get_session

    r = get_session(insecure).get(
        f"{qbase.rstrip('/')}/v2/auth",
        params={"service": "quay.io", "scope": "repository:ascend/vllm-ascend:pull"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["token"]


def fetch_image_env(tag_info: dict, token: str, timeout: int = 30, max_retries: int = 3,
                    insecure: bool = False, qbase: str = "https://quay.io") -> list[str]:
    """拉取镜像 config Env（manifest list -> amd64 子清单 -> config blob）。失败返回 []。"""
    from vllm_kb.net import get_session

    digest = tag_info["manifest_digest"]
    session = get_session(insecure)
    api = _quay_api(qbase)
    v2 = _quay_v2(qbase)
    for attempt in range(max_retries + 1):
        try:
            m = session.get(f"{api}/manifest/{digest}", timeout=timeout)
            m.raise_for_status()
            md = json.loads(m.json()["manifest_data"])
            if md.get("manifests"):
                arch = next(
                    (x for x in md["manifests"] if x["platform"].get("architecture") == "amd64"),
                    md["manifests"][0],
                )
                cm = session.get(f"{api}/manifest/{arch['digest']}", timeout=timeout)
                cm.raise_for_status()
                md = json.loads(cm.json()["manifest_data"])
            cfg_digest = md["config"]["digest"]
            blob = session.get(
                f"{v2}/blobs/{cfg_digest}",
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
    """从镜像 Env 提取 cann 版本 / SOC 型号 / python 版本 / vllm tag。

    vllm_tag：镜像 Env 的 VLLM_TAG（如 VLLM_TAG=v0.26.0）——镜像构建时锁定的
    vllm 配套版本，比 GitHub release 说明/版本号启发式更可靠。
    """
    out = {"cann": "", "soc": "", "python": "", "vllm_tag": ""}
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
        if not out["vllm_tag"]:
            m = _VLLM_TAG_RE.search(e)
            if m:
                out["vllm_tag"] = m.group(1)
    return out


# ---------------- GitHub release 说明 ----------------

def fetch_releases(insecure: bool = False, gbase: str = "https://api.github.com") -> dict[str, str]:
    """拉取 vllm-ascend 全部 release 说明，返回 {tag_name: body}。失败返回 {}。

    - 分页全量拉取（per_page=100，翻页直到取完，vllm-ascend release 已超 100 条）；
    - socket 级默认超时兜底：requests timeout 不覆盖 DNS 解析/代理握手，
      内网环境可能挂远超 timeout，先设 socket 超时保证任何阶段都限时；
    - 带重试（5xx/限流/网络抖动）；单页重试耗尽返回已拿到的部分，不阻塞矩阵。
    """
    import time as _t

    from vllm_kb.net import get_session

    print(f"[matrix] 拉取 GitHub release 说明（{gbase}/repos/vllm-project/vllm-ascend/releases）...",
          flush=True)
    # requests timeout 只覆盖连接+读取，DNS/代理握手无超时——socket 层兜底
    socket.setdefaulttimeout(30)
    session = get_session(insecure)
    releases: dict[str, str] = {}
    page = 1
    while True:
        url = f"{gbase.rstrip('/')}/repos/vllm-project/vllm-ascend/releases"
        ok = False
        for attempt in range(3):
            try:
                r = session.get(url, params={"per_page": 100, "page": page},
                                headers={"User-Agent": "vllm-kb"}, timeout=30)
                r.raise_for_status()
                batch = r.json()
                for rel in batch:
                    if rel.get("tag_name"):
                        releases[rel["tag_name"]] = rel.get("body") or ""
                ok = True
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"[matrix] release 拉取失败（page {page}）: {e}，{wait}s 后重试", flush=True)
                    _t.sleep(wait)
                else:
                    print(f"[matrix] 拉取 GitHub release 说明失败（page {page}）: {e}，"
                          f"已返回部分数据 {len(releases)} 条", flush=True)
                    return releases
        if not ok:
            return releases
        print(f"[matrix] release 说明 {len(releases)} 条（page {page} 完成）", flush=True)
        if len(batch) < 100:
            return releases
        page += 1


# torch-npu 依赖行（requirements.txt）：torch-npu==2.10.0.post2 / torch_npu==2.6.0.post1
_TORCH_NPU_RE = re.compile(r"^torch[-_]npu\s*==\s*(\d+\.\d+(?:\.\d+)?(?:\.post\d+)?)", re.IGNORECASE)


def fetch_pta_from_requirements(tag: str, insecure: bool = False,
                                gbase: str = "https://api.github.com") -> str:
    """从 vllm-ascend 指定 tag 的 requirements.txt 提取 pytorch-ascend（torch-npu）版本。

    返回 ""（无法获取）。GitHub API contents 端点（走内网镜像 gbase）。
    """
    from vllm_kb.net import get_session

    url = (f"{gbase.rstrip('/')}/repos/vllm-project/vllm-ascend/"
           f"contents/requirements.txt?ref={tag}")
    try:
        r = get_session(insecure).get(url, headers={"User-Agent": "vllm-kb", "Accept": "application/vnd.github.raw"},
                                      timeout=30)
        if r.status_code == 404:
            print(f"[matrix] {tag} 无 requirements.txt（跳过 PTA 提取）", flush=True)
            return ""
        r.raise_for_status()
        for line in r.text.splitlines():
            m = _TORCH_NPU_RE.match(line.strip())
            if m:
                return m.group(1)
        return ""
    except Exception as e:
        print(f"[matrix] {tag} requirements.txt 拉取失败: {e}", flush=True)
        return ""


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

def base_version_key(tag: str) -> str:
    """基础版本号（跨 rc/pre 回退用）：v0.13.0rc1 / v0.13.0 / v0.13.0rc3 -> 0.13.0。

    部分早期镜像（如 v0.13.0rc1）的 Env 不含 cann 版本，但同基础版本的其他
    形态（rc2/rc3/正式版）有——按基础版本号回退推断，避免 cann 缺成 '-'
    （cann 是随大版本演进，同 0.13.0 系列的 cann 基本一致）。
    """
    m = re.match(r"^v?(\d+\.\d+(?:\.\d+)?)", tag)
    return m.group(1) if m else ""


def build_rows(groups: dict, releases: dict[str, str], token: str,
               insecure: bool = False, qbase: str = "https://quay.io",
               gbase: str = "https://api.github.com") -> list[dict]:
    # 第一遍：收齐所有组的 env 信息（同基础版本的组共享 cann 用于回退）
    infos: dict[str, dict] = {}  # base(group key) -> extract_from_env 结果
    for base in groups:
        rep = pick_representative(groups[base])
        env = fetch_image_env(rep, token, insecure=insecure, qbase=qbase)
        infos[base] = extract_from_env(env)
    # 基础版本号 -> 该系列任一非空 cann（首个遇到即用，确定性依赖排序）
    cann_by_base: dict[str, str] = {}
    for base in sorted(groups):
        c = infos[base]["cann"]
        if c:
            cann_by_base.setdefault(base_version_key(base), c)

    # 版本型 tag：从各自 requirements.txt 提取 PTA（pytorch-ascend）
    pta_by_tag: dict[str, str] = {}
    for base in sorted(groups):
        if base_version_key(base):
            p = fetch_pta_from_requirements(base, insecure=insecure, gbase=gbase)
            if p:
                pta_by_tag[base] = p
    # 基础版本号 -> PTA（同系列回退用）
    pta_by_base: dict[str, str] = {}
    for base in sorted(pta_by_tag):
        pta_by_base.setdefault(base_version_key(base), pta_by_tag[base])

    rows = []
    total = len(groups)
    for i, base in enumerate(sorted(groups), 1):
        info = infos[base]
        cann = info["cann"]
        cann_src = "镜像env"
        if not cann:
            # Env 无 cann：按基础版本号回退同系列其他形态（rc2/rc3/正式版）
            fb = cann_by_base.get(base_version_key(base), "")
            if fb:
                cann, cann_src = fb, f"同系列回退({base_version_key(base)})"
        rel_body = releases.get(base, "")
        # vllm 配套版本优先级：镜像 VLLM_TAG（构建时锁定）> release 说明 > 版本号启发式
        vllm_ver, vllm_src = info["vllm_tag"], "镜像env(VLLM_TAG)"
        if not vllm_ver:
            vllm_ver, vllm_src = extract_vllm_from_release(
                base, rel_body) if rel_body else extract_vllm_from_release(base, "")
        # pytorch-ascend（PTA）：
        #   版本型 tag -> 自身 requirements.txt（torch-npu==x.y.z.postN）；
        #   0day 模型（无基础版本号）-> 参考其 vllm 版本对应 tag 的 PTA；
        #   torch（pytorch 字段）不必须，留空。
        pta = ""
        pta_src = ""
        bv = base_version_key(base)
        if bv:
            pta = pta_by_tag.get(base) or pta_by_base.get(bv, "")
            pta_src = f"requirements({bv})"
        elif vllm_ver:
            # 0day：vllm 版本 -> 找 vllm-ascend tag（v{vllm_ver} 或 v{vllm_ver}rc*）的 PTA
            cand = f"v{vllm_ver}"
            pta = pta_by_tag.get(cand) or pta_by_base.get(vllm_ver, "")
            if pta:
                pta_src = f"0day→v{vllm_ver}的requirements"
        notes = []
        if info["soc"]:
            notes.append(f"SOC={info['soc']}")
        if info["python"]:
            notes.append(f"python={info['python']}")
        provenance = []
        if cann:
            provenance.append(f"cann={cann_src}")
        if vllm_src:
            provenance.append(f"vllm={vllm_src}")
        if pta:
            provenance.append(f"pytorch-ascend={pta_src}")
        if not cann:
            # Env 无 cann 且同系列也没有：存空人工看护（缺口由 report_gaps 列出）
            provenance.append("cann=缺失(待人工)")
        rows.append(
            {
                "vllm-ascend": base,
                "vllm": vllm_ver,
                "cann": cann,
                "pytorch": "",          # torch 不必须，留空
                "pytorch-ascend": pta,
                "npu-driver": "",
                "notes": "; ".join(notes),
                "source": "自动(" + "+".join(provenance) + ")" if provenance else "待人工",
            }
        )
        flag = "ok" if (cann or vllm_ver or pta) else "gap"
        print(
            f"[matrix] ({i}/{total}) {base:<28} cann={cann or '-':<8} vllm={vllm_ver or '-':<8} "
            f"pta={pta or '-':<10} {vllm_src or ''} [{flag}]",
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


# 合法版本号：x.y[.z][rcN/.postN]（vllm/cann/pytorch/pytorch-ascend 数字版本，含 rc/post 后缀）
_VERSION_VALID_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?(?:rc\d+|\.post\d+)*$", re.IGNORECASE)


def validate_version_fields(rows: list[dict]) -> int:
    """写回前校验所有版本字段合法性：非法值置空 + 告警，避免非法版本存入矩阵。

    返回非法字段数。合法格式示例：0.26.0 / 8.5.1 / 2.6.0 / 0.13.0rc1。
    """
    bad = 0
    for r in rows:
        for f in COMPANION_FIELDS:
            v = r.get(f) or ""
            if v and not _VERSION_VALID_RE.match(v):
                print(f"[matrix] [!] {r['vllm-ascend']} 的 {f}={v!r} 非法版本，置空待人工", flush=True)
                r[f] = ""
                bad += 1
    return bad


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
    from vllm_kb.net import add_insecure_args, github_api_base, insecure_from_env, quay_base

    ap = argparse.ArgumentParser(description="自动生成/更新 vllm-ascend 组件配套矩阵")
    ap.add_argument("--matrix", default=None, help="矩阵文件路径（默认取 config.json 的 storage.companion_file）")
    ap.add_argument("--strict", action="store_true", help="存在缺口时以退出码 1 结束（CI 用）")
    ap.add_argument("--no-write", action="store_true", help="只报告不写回")
    ap.add_argument("--suggest-from-issues", nargs="?", const="data/raw/canonical.jsonl", metavar="CANONICAL",
                    help="从 canonical 的 issue 数据统计配套参考（真实部署众数，不写回矩阵）")
    add_insecure_args(ap)
    args = ap.parse_args()

    insecure = args.insecure or insecure_from_env()
    qbase = quay_base(args.quay_base)
    gbase = github_api_base(args.github_base)
    if insecure:
        print("[matrix] --insecure：跳过 SSL 证书校验（内网模式）")
    if qbase != "https://quay.io" or gbase != "https://api.github.com":
        print(f"[matrix] 镜像源 quay={qbase} github={gbase}")

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
    tags = fq.fetch_tags(insecure=insecure, base=qbase)
    groups = group_base_versions(tags)
    print(f"[matrix] 看护 tag 分组为 {len(groups)} 个基础版本（含模型专属镜像）", flush=True)

    # 2) 镜像 env + 3) release 说明
    token = get_quay_token(insecure=insecure, qbase=qbase)
    releases = fetch_releases(insecure=insecure, gbase=gbase)
    print(f"[matrix] 获取 GitHub release 说明 {len(releases)} 条", flush=True)

    auto_rows = build_rows(groups, releases, token, insecure=insecure, qbase=qbase, gbase=gbase)
    merged = merge_with_manual(auto_rows, manual_rows)
    # 写回前版本号合法性校验：非法版本置空（不污染矩阵，缺口报告会列出）
    n_bad = validate_version_fields(merged)
    if n_bad:
        print(f"[matrix] [!] {n_bad} 个非法版本字段已置空（详见上方）", flush=True)
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
