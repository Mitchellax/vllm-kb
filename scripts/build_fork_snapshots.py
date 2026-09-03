"""拉取 0day fork 仓代码快照：按镜像锁定的 commit SHA（矩阵 vllm_sha 字段）。

数据来源：data/compatibility/vllm-ascend.json 的 fork 行（vllm_repo + vllm_sha 非空）。

fork 分支会推进，但镜像构建时的 commit 已由 build_companion_matrix 的 clone 层
扫描固化在矩阵（vllm_sha）——快照按 SHA 拉取，与镜像内实际代码严格一致
（GitHub 分支 HEAD 会漂移，不能用）。

拉取通道评估（当前 fork 均为公开仓）：
- codeload.github.com/{repo}/zip/{sha} 是普通归档下载（非 REST API），公开仓
  无认证、无限流配额，按 SHA 下载内容确定性锁定——公开仓场景此通道已足够；
- 未来若出现私有 fork，需改走 api.github.com zipball + Authorization 头
  （当前不实现，出现时再加）；
- SHA 可能不可达：fork 删除/转私有，或分支 force-push 后旧 commit 被 GC。
  下载失败时明确报错并保留矩阵内的 SHA（镜像层仍是事实源，可从镜像导出）。

用法（在项目根）：
    python scripts/build_fork_snapshots.py                 # 拉取全部 fork 行
    python scripts/build_fork_snapshots.py --model hy4     # 只拉指定模型（可多次）
    python scripts/build_fork_snapshots.py --list          # 只列出 fork 行状态
    python scripts/build_fork_snapshots.py --index-only    # 已下载的只重建索引
    python scripts/build_fork_snapshots.py --insecure      # 内网：跳过 SSL 证书校验
    python scripts/build_fork_snapshots.py --base-url http://mirror:8080   # 内网镜像源

内网场景（SSL 被禁/证书不受信）：--insecure 跳过证书校验；--base-url 换内网镜像
（默认 https://codeload.github.com）；也可用环境变量 VLLM_KB_INSECURE=1 / VLLM_KB_CODE_BASE。

存储（与官方 vllm / vllm-ascend 物理隔离，检索需显式 repo=fork:{model}）：
    data/code/forks/{model}/zips/{sha12}.zip     # fork 源码 zip（sha12 = SHA 前 12 位）
    data/code/forks/{model}/snapshots/{sha12}/   # 解压缓存
    data/code/forks/{model}/index.sqlite3        # fork 符号索引（独立，不污染官方索引）
    data/code/forks/{model}/meta.json            # repo/ref/base/全量 sha/镜像 digest
"""
import argparse
import json
import os
import re
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402

DEFAULT_DOWNLOAD_BASE = "https://codeload.github.com"
# 模型目录名（防路径穿越）：字母/数字/点/下划线/连字符
_SAFE_MODEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _opener(insecure: bool) -> urllib.request.OpenerDirector:
    """按需构造跳过证书校验的 opener（内网自签证书/SSL 被禁时用）。"""
    if not insecure:
        return urllib.request.build_opener()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def fork_rows(cfg: AppConfig) -> list[dict]:
    """从 companion 矩阵提取 fork 行（vllm_repo + vllm_sha 均非空）。"""
    path = cfg.resolve(cfg.storage.companion_file)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in data.get("rows", []):
        repo = (r.get("vllm_repo") or "").strip()
        sha = (r.get("vllm_sha") or "").strip()
        if repo and sha:
            rows.append(r)
    return rows


def model_name(row_key: str) -> str:
    """fork 模型目录名（= 矩阵行键去平台后缀，如 hy4 / glm5.2）；不安全字符返回 ''。"""
    name = (row_key or "").strip()
    return name if _SAFE_MODEL_RE.match(name) else ""


def sha12(sha: str) -> str:
    return sha[:12]


def forks_root(cfg: AppConfig) -> Path:
    return cfg.resolve(cfg.storage.code_root) / "forks"


def model_root(cfg: AppConfig, model: str) -> Path:
    return forks_root(cfg) / model


def download(repo: str, sha: str, dest: Path, insecure: bool = False,
             base_url: str = DEFAULT_DOWNLOAD_BASE, max_retries: int = 3) -> bool:
    """按锁定 SHA 下载 fork zip（幂等：已存在且非空跳过）。

    鲁棒性：
    - 重试 + 指数退避（网络抖动/代理抖动，实测内网偶发 SSL EOF）；
    - 404/410 明确报错（fork 仓删除/转私有，或分支 force-push 后旧 commit 被回收
      ——SHA 锁定在镜像层内，仓库侧不保证永久可取，需尽早暴露）；
    - zip 魔数校验（PK\\x03\\x04）：404 的 HTML/JSON 错误页不落盘为 .zip，
      避免污染快照与符号索引。
    """
    if dest.exists() and dest.stat().st_size > 1000:
        return False
    url = f"{base_url.rstrip('/')}/{repo}/zip/{sha}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fork] 下载 {repo}@{sha12(sha)} ...", flush=True)

    import time as _t
    import urllib.error

    last_err = ""
    for attempt in range(max_retries):
        tmp = dest.with_suffix(".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vllm-kb"})
            with _opener(insecure).open(req, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            # zip 魔数校验（codeload 正常响应必为 zip 流）
            with open(tmp, "rb") as f:
                if f.read(4) != b"PK\x03\x04":
                    raise ValueError("响应不是 zip（魔数校验失败，可能是错误页）")
            tmp.replace(dest)
            print(f"[fork] {repo}@{sha12(sha)} 下载完成 ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                print(f"[fork] [!] {repo}@{sha12(sha)} HTTP {e.code}：fork 仓已删除/转私有，"
                      f"或锁定 commit 因 force-push 被回收（镜像层内的 SHA 仍有效，"
                      f"代码需从镜像本身导出）", flush=True)
                tmp.unlink(missing_ok=True)
                return False
            last_err = f"HTTP {e.code}"
        except Exception as e:
            last_err = str(e)
        tmp.unlink(missing_ok=True)
        if attempt < max_retries - 1:
            wait = 2 ** attempt
            print(f"[fork] {repo}@{sha12(sha)} 下载失败({last_err})，{wait}s 后重试", flush=True)
            _t.sleep(wait)
    print(f"[warn] {repo}@{sha12(sha)} 下载失败（重试 {max_retries} 次）: {last_err}", flush=True)
    return False


def ensure_snapshot(root: Path, version: str) -> Path:
    snap = root / "snapshots" / version
    if snap.is_dir() and any(snap.iterdir()):
        return snap
    zpath = root / "zips" / f"{version}.zip"
    if not zpath.exists():
        raise FileNotFoundError(f"{version} zip 不存在")
    snap.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(snap)
    return snap


def build_index(root: Path, version: str) -> int:
    """为 fork 版本构建符号索引（复用 vllm_kb.code_index 的提取器，独立库）。"""
    import sqlite3

    snap = ensure_snapshot(root, version)
    index_path = root / "index.sqlite3"
    conn = sqlite3.connect(str(index_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS symbols (
          version TEXT NOT NULL, symbol TEXT NOT NULL, file TEXT NOT NULL,
          line INTEGER, snippet TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sym_ver ON symbols(symbol, version);
        """
    )
    # 顶层目录兼容（codeload zip 解压出 {repo}-{sha}/ 单目录）
    repo_root = snap
    subdirs = [d for d in snap.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and not (snap / "vllm").exists():
        repo_root = subdirs[0]
    from vllm_kb import code_index as _CI

    count = 0
    conn.execute("DELETE FROM symbols WHERE version = ?", (version,))
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_root).as_posix()
        if not rel.startswith("vllm/"):
            continue
        if p.suffix not in (".py", ".cpp", ".hpp", ".h", ".cc"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = text.splitlines()
        for m in _CI._PY_DEF_RE.finditer(text):
            name = m.group(1) or m.group(2)
            if name:
                ln = text[: m.start()].count("\n") + 1
                sn = lines[ln - 1].strip()[:120] if ln <= len(lines) else ""
                conn.execute(
                    "INSERT OR IGNORE INTO symbols VALUES (?,?,?,?,?)",
                    (version, name.lower(), rel, ln, sn),
                )
                count += 1
    conn.commit()
    conn.close()
    return count


def write_meta(root: Path, row: dict) -> None:
    """记录 fork 元信息（检索端展示用：SHA ↔ 镜像/分支/基线的对应关系）。"""
    meta = {
        "model": row.get("vllm-ascend", ""),
        "repo": row.get("vllm_repo", ""),
        "ref": row.get("vllm_ref", ""),
        "base": row.get("vllm_base", ""),
        "sha": row.get("vllm_sha", ""),
        "image_digest": row.get("image_digest", ""),
    }
    (root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="按镜像锁定 SHA 拉取 0day fork 仓快照")
    ap.add_argument("--model", action="append", default=None,
                    help="指定 fork 模型（矩阵行键，如 hy4 / glm5.2，可多次）")
    ap.add_argument("--list", action="store_true", help="只列出 fork 行状态")
    ap.add_argument("--index-only", action="store_true", help="已下载的只重建索引")
    ap.add_argument("--insecure", action="store_true",
                    help="跳过 SSL 证书校验（内网自签证书/SSL 被禁；亦可用环境变量 VLLM_KB_INSECURE=1）")
    ap.add_argument("--base-url", default=None,
                    help=f"下载源前缀（内网镜像，http/https 均可；默认 {DEFAULT_DOWNLOAD_BASE}；"
                         f"亦可用环境变量 VLLM_KB_CODE_BASE）")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    insecure = args.insecure or os.environ.get(
        "VLLM_KB_INSECURE", "").strip().lower() in ("1", "true", "yes", "on")
    base_url = args.base_url or os.environ.get("VLLM_KB_CODE_BASE", DEFAULT_DOWNLOAD_BASE)
    if insecure:
        print("[fork] --insecure：跳过 SSL 证书校验（内网模式）")
    if base_url != DEFAULT_DOWNLOAD_BASE:
        print(f"[fork] 下载源 {base_url}")

    cfg = AppConfig.load(args.config)
    rows = fork_rows(cfg)
    if args.model:
        wanted = set(args.model)
        rows = [r for r in rows if r.get("vllm-ascend") in wanted]
    if not rows:
        print("[fork] 矩阵无 fork 行（vllm_repo + vllm_sha 均非空）。"
              "先跑 scripts/build_companion_matrix.py 生成/回填 fork 行。")
        return

    # 模型目录名校验（防路径穿越）
    for r in rows:
        m = model_name(r.get("vllm-ascend", ""))
        if not m:
            print(f"[fork] [!] 行键 {r.get('vllm-ascend')!r} 含不安全字符，跳过")
    rows = [r for r in rows if model_name(r.get("vllm-ascend", ""))]

    if args.list:
        print(f"[fork] fork 行 {len(rows)} 条:")
        for r in rows:
            m = model_name(r["vllm-ascend"])
            root = model_root(cfg, m)
            v = sha12(r["vllm_sha"])
            stored = (root / "zips" / f"{v}.zip").exists()
            print(f"    {m:<24} {r['vllm_repo']}@{r.get('vllm_ref') or '?'} "
                  f"base={r.get('vllm_base') or '-':<8} sha={v} "
                  f"{'已预存' if stored else '缺失'}")
        return

    if args.index_only:
        for r in rows:
            m = model_name(r["vllm-ascend"])
            root = model_root(cfg, m)
            v = sha12(r["vllm_sha"])
            if (root / "zips" / f"{v}.zip").exists():
                n = build_index(root, v)
                print(f"[fork] {m}@{v} 符号索引 {n} 个符号")
        return

    # 1) 下载（按锁定 SHA，与镜像内代码一致）
    downloaded = 0
    for r in rows:
        m = model_name(r["vllm-ascend"])
        root = model_root(cfg, m)
        v = sha12(r["vllm_sha"])
        if download(r["vllm_repo"], r["vllm_sha"], root / "zips" / f"{v}.zip",
                    insecure=insecure, base_url=base_url):
            downloaded += 1
        write_meta(root, r)
    if downloaded:
        print(f"[fork] 本轮新下载 {downloaded} 个 fork 快照")

    # 2) 解压 + 索引
    for r in rows:
        m = model_name(r["vllm-ascend"])
        root = model_root(cfg, m)
        v = sha12(r["vllm_sha"])
        if (root / "zips" / f"{v}.zip").exists():
            n = build_index(root, v)
            print(f"[fork] {m}@{v} 符号索引 {n} 个符号")

    print("[fork] 完成。检索需显式 repo=fork:{model}（与官方版本隔离）:")
    for r in rows:
        m = model_name(r["vllm-ascend"])
        print(f"    /code/versions?repo=fork:{m}  ->  [{sha12(r['vllm_sha'])}]")


if __name__ == "__main__":
    main()
