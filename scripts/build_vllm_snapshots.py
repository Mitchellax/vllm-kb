"""拉取 vllm 主仓代码快照：覆盖所有 vllm-ascend 版本对应的 vllm 版本。

数据来源：data/compatibility/vllm-ascend.json 的 companion 矩阵
（vllm-ascend 版本 -> vllm 版本 的校准映射）。

用法（在项目根）：
    python scripts/build_vllm_snapshots.py                    # 拉取全部对应 vllm 版本
    python scripts/build_vllm_snapshots.py --version 0.22.1   # 只拉指定 vllm 版本
    python scripts/build_vllm_snapshots.py --list             # 只列出需要拉取的版本
    python scripts/build_vllm_snapshots.py --index-only       # 已下载的只重建索引

存储（与 vllm-ascend 分开，避免符号表混淆）：
    data/code/vllm/zips/{v}.zip          # vllm 主仓源码 zip
    data/code/vllm/snapshots/{v}/        # 解压缓存
    data/code/vllm/index.sqlite3         # vllm 符号索引
"""
import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402

VLLM_REPO = "vllm-project/vllm"


def companion_vllm_versions(cfg: AppConfig) -> list[str]:
    """从 companion 矩阵提取 vllm-ascend -> vllm 的唯一 vllm 版本列表。"""
    path = cfg.resolve(cfg.storage.companion_file)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    versions = set()
    for r in data.get("rows", []):
        va = (r.get("vllm-ascend") or "").strip()
        vv = (r.get("vllm") or "").strip()
        if va and vv and va.startswith("v") and "." in va:
            versions.add(vv)
    return sorted(versions, key=lambda s: [int(x) for x in s.split(".")])


def _vllm_root(cfg: AppConfig) -> Path:
    return cfg.resolve(cfg.storage.code_root) / "vllm"


def download(repo: str, version: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return False
    url = f"https://codeload.github.com/{repo}/zip/refs/tags/v{version}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[vllm] 下载 v{version} ...", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vllm-kb"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        print(f"[warn] v{version} 下载失败: {e}")
        dest.unlink(missing_ok=True)
        return False
    print(f"[vllm] v{version} 下载完成 ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def ensure_snapshot(root: Path, version: str) -> Path:
    snap = root / "snapshots" / version
    if snap.is_dir() and any(snap.iterdir()):
        return snap
    zpath = root / "zips" / f"{version}.zip"
    if not zpath.exists():
        raise FileNotFoundError(f"v{version} zip 不存在")
    snap.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(snap)
    return snap


def build_index(root: Path, version: str) -> int:
    """为 vllm 主仓版本构建符号索引（复用 vllm_kb.code_index 的提取器）。"""
    from vllm_kb.code_index import VersionedCode  # noqa: E402

    snap = ensure_snapshot(root, version)
    index_path = root / "index.sqlite3"
    import sqlite3

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
    # 顶层目录兼容
    repo_root = snap
    subdirs = [d for d in snap.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and not (snap / "vllm").exists():
        repo_root = subdirs[0]
    # 复用 vllm_kb.code_index 的正则与提取逻辑
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


def main() -> None:
    ap = argparse.ArgumentParser(description="拉取 vllm 主仓对应版本快照")
    ap.add_argument("--version", action="append", default=None, help="指定 vllm 版本（可多次）")
    ap.add_argument("--list", action="store_true", help="只列出需要拉取的版本")
    ap.add_argument("--index-only", action="store_true", help="已下载的只重建索引")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    root = _vllm_root(cfg)
    versions = args.version or companion_vllm_versions(cfg)

    if args.list:
        print(f"需要拉取的 vllm 版本（{len(versions)} 个）:")
        for v in versions:
            print(f"  {v}")
        return

    if args.index_only:
        for v in versions:
            if (root / "zips" / f"{v}.zip").exists():
                n = build_index(root, v)
                print(f"[vllm] v{v} 索引 {n} 符号")
        return

    # 1) 下载
    downloaded = 0
    for v in versions:
        if download(VLLM_REPO, v, root / "zips" / f"{v}.zip"):
            downloaded += 1
    if downloaded:
        print(f"[vllm] 本轮新下载 {downloaded} 个")

    # 2) 解压 + 索引
    for v in versions:
        if (root / "zips" / f"{v}.zip").exists():
            n = build_index(root, v)
            print(f"[vllm] v{v} 索引 {n} 符号")

    print(f"[vllm] 完成。vllm 主仓可用版本: "
          f"{sorted(d.name for d in (root / 'zips').glob('*.zip')) if (root / 'zips').exists() else '(无)'}")


if __name__ == "__main__":
    main()
