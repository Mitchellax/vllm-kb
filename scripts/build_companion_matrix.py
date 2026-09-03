"""自动生成/更新 vllm-ascend 组件配套矩阵（quay 镜像 + GitHub release 自动匹配）。

数据源与自动匹配规则：
1. quay tag 清单（复用 fetch_quay_tags）→ 看护 tag → 按基础版本去重（去平台后缀 -a3/-310p/-openeuler）；
2. 镜像 config Env → **cann** 版本（路径里 cann-X.Y.Z）、SOC 型号、python 版本（进 notes）；
2.5 镜像 build history（buildkit created_by）→ **vllm** 配套版本（官方仓 VLLM_TAG，构建锁定）；
    fork 仓（0day 开发分支镜像）→ vllm_repo/vllm_ref/vllm_base（基线版本），
    锁定 commit 由 clone 层 .git 扫描固化（见 extract_fork_sha）；
3. vllm-ascend GitHub release 说明 → **vllm** 配套版本
   （"aligns ... with upstream vLLM v0.23.0" / "based on vLLM v0.19.1"；
    无说明时启发式：vllm-ascend 版本号跟踪上游 vllm，剥 rc 后缀）。

自动无法确定的（npu-driver 等）→ 告警并留空，人工修复：
人工已填写的字段优先保留（merge 时手工非空替换，自动只填空字段）。

torch / pytorch-ascend：requirements.txt 提取（本地快照 zip 优先——54 个版本逐
tag 走 GitHub API 会撞未认证限流 60 次/小时；快照由 build_code_snapshots 预拉，
同源同内容），快照未预存的 tag 走 GitHub API 兜底。

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

OFFICIAL_VLLM_REPO = "vllm-project/vllm"  # 官方 vllm 仓（非此 owner/name 即视为开发 fork）

COMPANION_FIELDS = ["vllm", "cann", "pytorch", "pytorch-ascend", "npu-driver", "vllm_base"]
# npu-driver(HDK) 与镜像版本不耦合（特定 HDK 有特定问题），缺失符合预期，不计入缺口告警
REQUIRED_FIELDS = ["vllm", "cann", "pytorch", "pytorch-ascend"]

PLATFORM_SUFFIXES = ("-openeuler", "-310p", "-a3", "-a5", "-a2")
_CANN_RE = re.compile(r"cann-(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
_SOC_RE = re.compile(r"^SOC_VERSION=(.+)$")
_PYTHON_RE = re.compile(r"python(\d+\.\d+(?:\.\d+)?)")
# 镜像 Env 里的 vllm tag（如 VLLM_TAG=v0.26.0）——比 release 说明更直接的配套证据
_VLLM_TAG_RE = re.compile(r"(?:^|[\s;])VLLM_TAG\s*=\s*v?(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
# ---- buildkit history（created_by）提取：vllm 构建参数三形态 ----
# 形态 1/2：ARG 声明（ARG VLLM_TAG=v0.23.0）或 RUN 内联参数（RUN |5 K=V ... /bin/bash -c ...）
# 值不跨空白：VLLM_COMMIT= 后跟空格即空值（不能吞掉后续的 /bin/bash）
_BK_ARG_RE = re.compile(r"(?:^|[\s;])VLLM_(REPO|TAG|COMMIT)=([^\s]*)", re.IGNORECASE)
# 形态 3：旧式 Dockerfile 把 clone 硬编码在 RUN 命令行（git clone --depth 1 --branch glm52 <url>）
_GIT_CLONE_RE = re.compile(r"git\s+clone\s.*?(?:--branch|-b)\s+(\S+)\s+(\S+)", re.IGNORECASE)
# github URL -> owner/name（容忍 .git 后缀与 / 路径）
_REPO_SLUG_RE = re.compile(
    r"github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?=[\s/]|$)",
    re.IGNORECASE,
)
# fork 基线版本：Env VLLM_VERSION / history SETUPTOOLS_SCM_PRETEND_VERSION / 分支名内版本号
_ENV_VLLM_VERSION_RE = re.compile(r"^VLLM_VERSION=v?(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
_PRETEND_VER_RE = re.compile(
    r'SETUPTOOLS_SCM_PRETEND_VERSION="?(v?\d+\.\d+(?:\.\d+)?)', re.IGNORECASE)
# 官方仓 buildkit ref 的版本号形态（v0.23.0 / 0.23.0rc1）；分支名（dev_hy4/glm52）不匹配
_REF_VERSION_RE = re.compile(r"^v?(\d+\.\d+(?:\.\d+)?(?:rc\d+)?)")
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


def fetch_image_config(tag_info: dict, token: str, timeout: int = 30, max_retries: int = 3,
                       insecure: bool = False, qbase: str = "https://quay.io") -> dict:
    """拉取镜像 config（manifest list -> amd64 子清单 -> config blob）。

    返回 {"env": [...], "history": [{"created_by": str, "empty_layer": bool}, ...],
    "layers": [digest, ...]}；失败返回全空。history 的 created_by 含 buildkit 构建参数
    （ARG 值 / RUN 内联参数 / git clone 命令行），是 vllm 配套版本与 fork 仓信息
    的最直接证据（Env 里通常没有）；layers 为子 manifest 层 digest 列表（按序对应
    history 非空条目，供 clone 层定位）。
    """
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
            cfg = blob.json()
            env = cfg.get("config", {}).get("Env", []) or []
            history = [
                {
                    "created_by": str(h.get("created_by") or ""),
                    "empty_layer": bool(h.get("empty_layer")),
                }
                for h in cfg.get("history", []) or []
            ]
            layers = [str(l.get("digest") or "") for l in md.get("layers", []) or []]
            return {"env": [str(e) for e in env], "history": history, "layers": layers}
        except Exception as e:
            if attempt == max_retries:
                print(f"[matrix] 拉取 {tag_info['name']} 镜像 config 失败: {e}")
                return {"env": [], "history": [], "layers": []}
            wait = 2 ** attempt
            print(f"[matrix] {tag_info['name']} 拉取失败({e})，{wait}s 后重试", flush=True)
            time.sleep(wait)
    return {"env": [], "history": [], "layers": []}


def _repo_slug(url: str) -> str:
    """github 仓库 URL -> owner/name（https://github.com/a/b.git -> a/b）。非 github 域返回 ''。"""
    m = _REPO_SLUG_RE.search(url or "")
    return f"{m.group(1)}/{m.group(2)}" if m else ""


def extract_buildkit_info(history: list) -> dict:
    """从镜像 build history 的 created_by 提取 vllm 构建参数（三形态）。

    1. ARG 声明：ARG VLLM_REPO=... / ARG VLLM_TAG=... / ARG VLLM_COMMIT=...
    2. RUN 内联参数：RUN |N K=V ... /bin/bash -c ...（buildkit 携带的 ARG 值，
       同一参数会在多条 history 重复出现，取首个非空）
    3. 旧式硬编码：git clone --depth 1 --branch <ref> <url>（ref/url 写死在命令行）

    返回 {repo, ref, commit, is_fork}：repo 为 owner/name（空=未识别）；
    is_fork = repo 非空且非官方仓（0day 开发分支镜像）。
    """
    repo_url = ref = commit = ""
    clone_ref = clone_url = ""
    for h in history:
        cb = h.get("created_by", "") if isinstance(h, dict) else str(h or "")
        if not cb:
            continue
        m = _GIT_CLONE_RE.search(cb)
        if m and not clone_ref:
            r, u = m.group(1), m.group(2)
            # `-b $VLLM_TAG $VLLM_REPO` 形态是 shell 变量，非真实值（真实值由 ARG 提供）
            if not r.startswith("$") and not u.startswith("$"):
                clone_ref, clone_url = r, u
        for km in _BK_ARG_RE.finditer(cb):
            key, val = km.group(1).upper(), km.group(2)
            if key == "REPO" and val and not repo_url:
                repo_url = val
            elif key == "TAG" and val and not ref:
                ref = val
            elif key == "COMMIT" and val and not commit:
                commit = val
    if not ref and clone_ref:
        ref = clone_ref
    if not repo_url and clone_url:
        repo_url = clone_url
    slug = _repo_slug(repo_url)
    return {
        "repo": slug,
        "ref": ref,
        "commit": commit,
        "is_fork": bool(slug) and slug != OFFICIAL_VLLM_REPO,
    }


def _fork_base_version(env: list[str], history: list, ref: str) -> str:
    """fork 镜像的 vllm 基线版本（fork 分支基于的官方版本），三重回退：

    Env VLLM_VERSION（如 hy4 镜像）→ history SETUPTOOLS_SCM_PRETEND_VERSION（pip 安装层）
    → 分支名内嵌版本号。均无证据时返回 ''（vllm 字段留空待人工）。
    """
    for e in env:
        m = _ENV_VLLM_VERSION_RE.match(e)
        if m:
            return m.group(1)
    for h in history:
        cb = h.get("created_by", "") if isinstance(h, dict) else str(h or "")
        m = _PRETEND_VER_RE.search(cb)
        if m:
            return m.group(1).lstrip("v")
    m = _REF_VERSION_RE.match(ref or "")
    return m.group(1) if m else ""


def extract_from_image(env: list[str], history: list) -> dict[str, str]:
    """从镜像 config（Env + build history）提取配套信息。

    - Env：cann 版本 / SOC 型号 / python 版本 / vllm_tag（少见）
    - history created_by（buildkit 参数）：
      - 官方仓 + 版本号 ref → vllm_tag（构建锁定，最高优先级证据）
      - fork 仓 → vllm_repo / vllm_ref / vllm_base（基线版本）/ vllm_commit
    """
    out = extract_from_env(env)
    out["vllm_tag_src"] = "镜像env(VLLM_TAG)" if out["vllm_tag"] else ""
    bk = extract_buildkit_info(history)
    out["vllm_repo"] = bk["repo"]
    out["vllm_ref"] = bk["ref"]
    out["vllm_commit"] = bk["commit"]
    out["is_fork"] = bk["is_fork"]
    if not bk["is_fork"] and bk["ref"] and not bk["commit"]:
        # 官方仓 + 版本号 ref：buildkit 锁定的配套版本（v0.23.0 -> 0.23.0）
        m = _REF_VERSION_RE.match(bk["ref"])
        if m and not out["vllm_tag"]:
            out["vllm_tag"] = m.group(1)
            out["vllm_tag_src"] = "镜像buildkit(VLLM_TAG)"
    out["vllm_base"] = _fork_base_version(env, history, bk["ref"]) if bk["is_fork"] else ""
    return out


# ---------------- fork 锁定 SHA（clone 层扫描） ----------------
# 背景：0day fork 镜像按分支构建（git clone --branch X），分支会推进，但镜像层内
# 的 .git（depth 1 浅克隆）固化了构建时的 commit——这是"镜像内实际代码"的唯一
# 可靠锚点（GitHub 分支 HEAD 会漂移，buildkit 参数里也只有分支名）。

# 包管理器前缀（出现在 git clone 之前的复合命令层，非 clone 专属层）
_PKG_MGRS = ("apt-get", "yum ", "dnf ")


def _is_dedicated_clone_cmd(cb: str) -> bool:
    """created_by 是否为 git clone 专属命令（clone 前无包管理器安装段）。

    hy4 形态：`RUN |4 ... /bin/bash -c git clone --depth 1 ... # buildkit`
    glm5.2 形态：`RUN |2 ... /bin/bash -c git clone --depth 1 --branch glm52 <url> ...`
    反例（排除）：apt-get 复合层 `... /bin/bash -c apt-get update && ... && git clone ...`
    """
    idx = cb.find("git clone")
    if idx < 0:
        return False
    head = cb[:idx]
    return not any(p in head for p in _PKG_MGRS)


def _locate_clone_layer(history: list, layers: list) -> int:
    """定位 git clone 专属层的索引（history 非空条目与 layers 按序一一对应）。

    找不到返回 -1。OCI 规范：带 empty_layer=True 的 history 条目不占层，
    其余按顺序对应 layers[]（实测 hy4-a3 15/15、glm5.2-a3 16/16 对应）。
    """
    j = -1
    for h in history:
        if h.get("empty_layer"):
            continue
        j += 1
        if _is_dedicated_clone_cmd(h.get("created_by", "")):
            return j
    return -1


def _scan_tar_for_git_sha(fileobj) -> str:
    """流式扫 tar 层，读 .git 的锁定 commit。

    证据优先级：shallow（浅克隆根，最可靠）> packed-refs > refs/heads/* > FETCH_HEAD。
    """
    import tarfile

    sha = refs_sha = packed = fetch_head = ""
    tf = tarfile.open(fileobj=fileobj, mode="r|")
    for m in tf:
        if not m.isfile() or m.size > 65536:
            continue
        low = m.name.lower()
        if low.endswith("/.git/shallow") or low == ".git/shallow":
            data = tf.extractfile(m).read().decode("utf-8", "replace").strip()
            if _SHA_VALID_RE.match(data):
                sha = data.lower()
                break
        elif low.endswith("/.git/packed-refs") or low == ".git/packed-refs":
            packed = tf.extractfile(m).read().decode("utf-8", "replace")
        elif "/.git/refs/heads/" in low and not refs_sha:
            data = tf.extractfile(m).read().decode("utf-8", "replace").strip()
            if _SHA_VALID_RE.match(data):
                refs_sha = data.lower()
        elif low.endswith("/.git/fetch_head") or low == ".git/fetch_head":
            fetch_head = tf.extractfile(m).read().decode("utf-8", "replace")
    if sha:
        return sha
    if packed:
        m = re.search(r"^([0-9a-f]{40})\s+refs/", packed, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).lower()
    if refs_sha:
        return refs_sha
    if fetch_head:
        m = re.match(r"\s*([0-9a-f]{40})\b", fetch_head, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return ""


def _scan_layer_for_git_sha(session, v2: str, layer_digest: str, token: str,
                            timeout: int = 300) -> str:
    """下载 clone 层 blob（~75MB）并扫描 .git，返回锁定 commit（失败 ''）。

    先按 gzip 流解（docker 层标准存储格式），失败再按未压缩 tar 兜底。
    """
    import gzip
    import io
    import tarfile

    r = session.get(f"{v2}/blobs/{layer_digest}",
                    headers={"Authorization": "Bearer " + token}, timeout=timeout)
    r.raise_for_status()
    data = r.content
    for open_ in (lambda: gzip.GzipFile(fileobj=io.BytesIO(data)),
                  lambda: io.BytesIO(data)):
        try:
            sha = _scan_tar_for_git_sha(open_())
        except (OSError, EOFError, tarfile.ReadError):
            sha = ""
        if sha:
            return sha
    return ""


def extract_fork_sha(tag_info: dict, token: str, timeout: int = 30, max_retries: int = 3,
                     insecure: bool = False, qbase: str = "https://quay.io") -> dict:
    """扫描 fork 镜像的 git clone 层，读出镜像内实际代码的锁定 commit。

    返回 {"sha": 40-hex 或 "", "layer": 层 digest, "error": 失败原因}。
    层内的 .git 是构建时 git clone 留下的（depth 1），不随 fork 分支推进漂移。
    """
    from vllm_kb.net import get_session

    out = {"sha": "", "layer": "", "error": ""}
    cfg = fetch_image_config(tag_info, token, timeout=timeout, max_retries=max_retries,
                             insecure=insecure, qbase=qbase)
    layers = cfg.get("layers", [])
    j = _locate_clone_layer(cfg.get("history", []), layers)
    if j < 0 or j >= len(layers):
        out["error"] = "未定位到 git clone 层（非 fork 构建形态）"
        return out
    layer_digest = layers[j]
    out["layer"] = layer_digest
    session = get_session(insecure)
    v2 = _quay_v2(qbase)
    for attempt in range(2):
        try:
            sha = _scan_layer_for_git_sha(session, v2, layer_digest, token, timeout=max(timeout, 300))
            if sha:
                out["sha"] = sha
            else:
                out["error"] = "clone 层内未找到 .git commit（浅克隆元数据被清理？）"
            return out
        except Exception as e:
            if attempt == 1:
                out["error"] = str(e)
                return out
            time.sleep(2)


def enrich_fork_sha(rows: list[dict], groups: dict, token: str,
                    insecure: bool = False, qbase: str = "https://quay.io") -> None:
    """就地回填 fork 行的 vllm_sha（镜像 clone 层内的锁定 commit）。

    image_digest 锚定：tag 未重推（digest 不变）= 层内容不变 = SHA 不变，直接沿用
    已有值（跳过 ~75MB 层下载）；digest 变化（镜像重推）才重新扫描。扫描失败
    保留旧值并告警（不阻塞矩阵写回）。
    """
    fork_rows = [r for r in rows if r.get("vllm_repo")]
    if not fork_rows:
        return
    print(f"[matrix] fork 行 {len(fork_rows)} 条：扫描 clone 层固化锁定 commit ...", flush=True)
    for r in fork_rows:
        g = groups.get(r["vllm-ascend"])
        if not g:
            print(f"[matrix]    {r['vllm-ascend']}: quay 无对应 tag 组，跳过", flush=True)
            continue
        rep = pick_representative(g)
        cur = rep["manifest_digest"]
        if r.get("vllm_sha") and r.get("image_digest") == cur:
            print(f"[matrix]    {r['vllm-ascend']}: 镜像未重推（digest 锚命中），"
                  f"SHA 沿用 {r['vllm_sha'][:12]}", flush=True)
            continue
        out = extract_fork_sha(rep, token, insecure=insecure, qbase=qbase)
        if out["sha"]:
            r["vllm_sha"] = out["sha"]
            r["image_digest"] = cur
            print(f"[matrix]    {r['vllm-ascend']}: 锁定 commit {out['sha'][:12]}"
                  f"（{r['vllm_repo']}@{r.get('vllm_ref') or '?'}）", flush=True)
        else:
            print(f"[matrix]    [!] {r['vllm-ascend']}: SHA 扫描失败"
                  f"（{out['error'] or '未知原因'}），保留旧值", flush=True)

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
# torch 依赖行（requirements.txt）：torch==2.10.0（torch 与 torch-npu 基础版本一致）
_TORCH_RE = re.compile(r"^torch\s*==\s*(\d+\.\d+(?:\.\d+)?(?:\.post\d+)?)", re.IGNORECASE)


def default_code_root() -> str:
    """本地快照根（config.json storage.code_root，默认 data/code）。"""
    try:
        cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
        return cfg.get("storage", {}).get("code_root", "data/code")
    except Exception:
        return "data/code"


def _requirements_from_snapshot(tag: str) -> str:
    """从本地 vllm-ascend 快照 zip（data/code/zips/{tag}.zip）读 requirements.txt。

    返回文本（无 zip / 无文件返回 ""）。零网络：54 个版本逐 tag 走 GitHub API
    contents 端点会撞未认证限流（60 次/小时），快照由 build_code_snapshots
    预先拉取，同源同内容。
    """
    import zipfile

    zpath = Path(default_code_root()) / "zips" / f"{tag}.zip"
    if not zpath.exists():
        return ""
    try:
        with zipfile.ZipFile(zpath) as zf:
            # codeload zip 顶层目录：vllm-ascend-{tag}/requirements.txt
            for name in zf.namelist():
                if name.count("/") == 1 and name.endswith("/requirements.txt"):
                    return zf.read(name).decode("utf-8", "replace")
    except Exception as e:
        print(f"[matrix] {tag} 快照 zip 读取失败: {e}", flush=True)
    return ""


def _requirements_from_github(tag: str, insecure: bool = False,
                              gbase: str = "https://api.github.com") -> str:
    """GitHub API contents 端点兜底（快照未预存的 tag）。带 2 次重试。"""
    import time as _t

    from vllm_kb.net import get_session

    url = (f"{gbase.rstrip('/')}/repos/vllm-project/vllm-ascend/"
           f"contents/requirements.txt?ref={tag}")
    session = get_session(insecure)
    for attempt in range(3):
        try:
            r = session.get(url, headers={"User-Agent": "vllm-kb", "Accept": "application/vnd.github.raw"},
                            timeout=15)
            if r.status_code == 404:
                print(f"[matrix] {tag} 无 requirements.txt（跳过 torch/PTA 提取）", flush=True)
                return ""
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt == 2:
                print(f"[matrix] {tag} requirements.txt 拉取失败: {e}", flush=True)
                return ""
            wait = 2 ** attempt
            print(f"[matrix] {tag} requirements.txt 拉取失败({e})，{wait}s 后重试", flush=True)
            _t.sleep(wait)
    return ""


def extract_torch_pair(text: str) -> tuple[str, str]:
    """从 requirements.txt 文本提取 (torch, torch-npu) 版本。无对应行返回空。"""
    torch_v = pta = ""
    for line in text.splitlines():
        s = line.strip()
        if not torch_v:
            m = _TORCH_RE.match(s)
            if m:
                torch_v = m.group(1)
        if not pta:
            m = _TORCH_NPU_RE.match(s)
            if m:
                pta = m.group(1)
    return torch_v, pta


def fetch_pta_from_requirements(tag: str, insecure: bool = False,
                                gbase: str = "https://api.github.com") -> dict:
    """提取指定 tag 的 torch / pytorch-ascend(torch-npu) 版本，返回 {torch, torch_npu, src}。

    优先本地快照 zip（零网络、无限流），快照未预存时 GitHub API 兜底。
    """
    text = _requirements_from_snapshot(tag)
    if text:
        torch_v, pta = extract_torch_pair(text)
        return {"torch": torch_v, "torch_npu": pta, "src": "requirements(本地快照)"}
    text = _requirements_from_github(tag, insecure=insecure, gbase=gbase)
    torch_v, pta = extract_torch_pair(text) if text else ("", "")
    return {"torch": torch_v, "torch_npu": pta, "src": "requirements(github)" if text else ""}


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
    total = len(groups)
    for i, base in enumerate(sorted(groups), 1):
        rep = pick_representative(groups[base])
        print(f"[matrix] 镜像 config {i}/{total}：{base}（代表 tag {rep['name']}）...", flush=True)
        cfg = fetch_image_config(rep, token, insecure=insecure, qbase=qbase)
        infos[base] = extract_from_image(cfg["env"], cfg["history"])
        got = infos[base]
        fork_mark = f" fork={got['vllm_repo']}@{got['vllm_ref']}" if got.get("is_fork") else ""
        print(f"[matrix]    cann={got['cann'] or '(空)'} vllm_tag={got['vllm_tag'] or '(空)'} "
              f"soc={got['soc'] or '(空)'} python={got['python'] or '(空)'}{fork_mark}", flush=True)
    # 基础版本号 -> 该系列任一非空 cann（首个遇到即用，确定性依赖排序）
    cann_by_base: dict[str, str] = {}
    for base in sorted(groups):
        c = infos[base]["cann"]
        if c:
            cann_by_base.setdefault(base_version_key(base), c)

    # 版本型 tag：从 requirements.txt 提取 torch / PTA（本地快照优先，GitHub 兜底）
    req_by_tag: dict[str, dict] = {}
    versioned = [b for b in sorted(groups) if base_version_key(b)]
    for i, base in enumerate(versioned, 1):
        print(f"[matrix] torch/PTA 提取 {i}/{len(versioned)}：{base} ...", flush=True)
        r = fetch_pta_from_requirements(base, insecure=insecure, gbase=gbase)
        if r["torch_npu"] or r["torch"]:
            req_by_tag[base] = r
            print(f"[matrix]    {base} -> torch {r['torch'] or '-'} "
                  f"pytorch-ascend {r['torch_npu'] or '-'}（{r['src']}）", flush=True)
    # 基础版本号 -> torch / PTA（同系列回退用）
    torch_by_base: dict[str, str] = {}
    pta_by_base: dict[str, str] = {}
    for base in sorted(req_by_tag):
        bv = base_version_key(base)
        torch_by_base.setdefault(bv, req_by_tag[base]["torch"])
        pta_by_base.setdefault(bv, req_by_tag[base]["torch_npu"])
    pta_by_tag = {b: r["torch_npu"] for b, r in req_by_tag.items()}

    rows = []
    for i, base in enumerate(sorted(groups), 1):
        print(f"[matrix] 生成行 {i}/{total}：{base}", flush=True)
        info = infos[base]
        rep = pick_representative(groups[base])
        cann = info["cann"]
        cann_src = "镜像env"
        if not cann:
            # Env 无 cann：按基础版本号回退同系列其他形态（rc2/rc3/正式版）
            fb = cann_by_base.get(base_version_key(base), "")
            if fb:
                cann, cann_src = fb, f"同系列回退({base_version_key(base)})"
        rel_body = releases.get(base, "")
        # vllm 配套版本优先级：官方仓 buildkit VLLM_TAG（构建锁定）> Env VLLM_TAG
        #   > release 说明 > 版本号启发式；fork 仓（0day 开发分支）→ 基线版本 vllm_base。
        if info.get("is_fork"):
            vllm_ver = info.get("vllm_base", "")
            vllm_src = f"fork基线({info['vllm_repo']}@{info['vllm_ref']})" if vllm_ver else ""
        else:
            vllm_ver, vllm_src = info["vllm_tag"], info["vllm_tag_src"]
        if not vllm_ver:
            vllm_ver, vllm_src = extract_vllm_from_release(
                base, rel_body) if rel_body else extract_vllm_from_release(base, "")
        # torch / pytorch-ascend（PTA）：
        #   版本型 tag -> 自身 requirements.txt（torch==x / torch-npu==y）；
        #   0day 模型（无基础版本号）-> 参考其 vllm 版本对应 tag 的组合；
        #   requirements 无显式 torch 行时，由 torch-npu 基础版本推导（两者配套发布）。
        pta = torch_v = ""
        pta_src = ""
        bv = base_version_key(base)
        if bv:
            r = req_by_tag.get(base) or {}
            pta = r.get("torch_npu") or pta_by_base.get(bv, "")
            torch_v = r.get("torch") or torch_by_base.get(bv, "")
            pta_src = (r.get("src") or f"requirements({bv})") if (pta or torch_v) else ""
        elif vllm_ver:
            # 0day：vllm 版本 -> 找 vllm-ascend tag（v{vllm_ver} 或 v{vllm_ver}rc*）的组合
            cand = f"v{vllm_ver}"
            pta = pta_by_tag.get(cand) or pta_by_base.get(vllm_ver, "")
            torch_v = torch_by_base.get(vllm_ver, "")
            if pta:
                pta_src = f"0day→v{vllm_ver}的requirements"
        if pta and not torch_v:
            # torch-npu==2.10.0.post4 -> torch 2.10.0（配套基础版本）
            torch_v = re.match(r"^(\d+\.\d+(?:\.\d+)?)", pta).group(1) \
                if re.match(r"^(\d+\.\d+(?:\.\d+)?)", pta) else ""
        notes = []
        if info["soc"]:
            notes.append(f"SOC={info['soc']}")
        if info["python"]:
            notes.append(f"python={info['python']}")
        if info.get("is_fork"):
            notes.append(f"fork={info['vllm_repo']}@{info['vllm_ref']}")
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
        row = {
            "vllm-ascend": base,
            "vllm": vllm_ver,
            "cann": cann,
            "pytorch": torch_v,    # requirements torch== 行，或由 torch-npu 基础版本推导
            "pytorch-ascend": pta,
            "npu-driver": "",
            "notes": "; ".join(notes),
            "source": "自动(" + "+".join(provenance) + ")" if provenance else "待人工",
        }
        if info.get("is_fork"):
            # fork 行（0day 开发分支镜像）：记录 fork 仓/分支/基线 + 镜像 digest
            # （digest 是锁定 SHA 扫描的不可变锚：未重推即可跳过 75MB 层下载）
            row["vllm_repo"] = info["vllm_repo"]
            row["vllm_ref"] = info["vllm_ref"]
            row["vllm_base"] = info["vllm_base"]
            row["image_digest"] = rep["manifest_digest"]
        rows.append(row)
        flag = "ok" if (cann or vllm_ver or pta) else "gap"
        print(
            f"[matrix] ({i}/{total}) {base:<28} cann={cann or '-':<8} vllm={vllm_ver or '-':<8} "
            f"pta={pta or '-':<14} torch={torch_v or '-':<9} {vllm_src or ''} [{flag}]",
            flush=True,
        )
    return rows


# merge 输出列序：已知字段在前，未知透传字段按字母序追加（新增字段不影响旧文件）
_FIELD_ORDER = [
    "vllm-ascend", "vllm", "cann", "pytorch", "pytorch-ascend", "npu-driver",
    "vllm_repo", "vllm_ref", "vllm_base", "vllm_sha", "image_digest",
    "notes", "source",
]


def merge_with_manual(auto_rows: list[dict], manual_rows: list[dict]) -> list[dict]:
    """合并：手工行优先（非空字段**替换**自动值），自动只填空字段；quay 列表外的人工行保留。

    字段取两边并集（未知字段透传，不再固定白名单），统一规则"手工非空 > 自动非空 > 空串"——
    旧版把 pytorch/pytorch-ascend/npu-driver 写死 `m.get(...) or ""`，自动提取的
    pytorch-ascend 会被直接丢弃，本版修正。
    """
    manual_by_key = {r.get("vllm-ascend", ""): r for r in manual_rows}
    merged: list[dict] = []
    seen: set[str] = set()

    def _merge_row(a: dict, m: dict) -> dict:
        union = set(a) | set(m)
        ordered = [f for f in _FIELD_ORDER if f in union]
        ordered += sorted(union - set(_FIELD_ORDER))
        return {k: (m.get(k) or a.get(k) or "") for k in ordered}

    for a in auto_rows:
        key = a["vllm-ascend"]
        seen.add(key)
        m = manual_by_key.get(key)
        if m:
            merged.append(_merge_row(a, m))
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


# 合法版本号：x.y[.z][rcN/.postN]（vllm/cann/pytorch/pytorch-ascend/vllm_base 数字版本，含 rc/post 后缀）
_VERSION_VALID_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?(?:rc\d+|\.post\d+)*$", re.IGNORECASE)
# fork 行附加字段格式（非版本号，单独规则）
_SHA_VALID_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)                 # vllm_sha：git commit
_DIGEST_VALID_RE = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)       # image_digest：镜像 manifest
_REPO_SLUG_VALID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")      # vllm_repo：owner/name


def validate_version_fields(rows: list[dict]) -> int:
    """写回前校验所有版本/格式字段合法性：非法值置空 + 告警，避免非法值存入矩阵。

    返回非法字段数。合法格式示例：0.26.0 / 8.5.1 / 2.6.0 / 0.13.0rc1；
    fork 附加字段：vllm_base 同版本规则，vllm_sha 40 位 hex，
    image_digest 形如 sha256:<64hex>，vllm_repo 形如 owner/name。
    """
    bad = 0

    def _check(r: dict, field: str, valid_re) -> None:
        nonlocal bad
        v = r.get(field) or ""
        if v and not valid_re.match(v):
            print(f"[matrix] [!] {r['vllm-ascend']} 的 {field}={v!r} 非法，置空待人工", flush=True)
            r[field] = ""
            bad += 1

    for r in rows:
        for f in COMPANION_FIELDS:
            _check(r, f, _VERSION_VALID_RE)
        _check(r, "vllm_sha", _SHA_VALID_RE)
        _check(r, "image_digest", _DIGEST_VALID_RE)
        _check(r, "vllm_repo", _REPO_SLUG_VALID_RE)
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
    不是官方配套声明。镜像 build history 的 buildkit 参数只含 vllm 仓/ref
    （torch/torch_npu 版本不在其中），issue 众数仍是 torch/torch_npu 配套的
    唯一自动获取途径。
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
    # fork 行：clone 层扫描固化锁定 commit（digest 锚定，镜像未重推则跳过层下载）
    enrich_fork_sha(merged, groups, token, insecure=insecure, qbase=qbase)
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
