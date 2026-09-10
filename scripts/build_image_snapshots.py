"""提取 0day 镜像内的 vllm-ascend 插件源码（只拉插件层，不拉整镜像）。

背景：0day 模型镜像的插件代码是 `COPY . /vllm-workspace/vllm-ascend/` 拷进镜像的，
镜像内没有版本/commit 痕迹——要审查"这个镜像实际跑的插件代码"只能从镜像层里取。
本脚本只下载**插件源码层**（实测 23~101MB；整镜像 6GB+，其中 4GB 是 CANN，属二进制层，
审查无用途，故不拉），解到 data/code/images/{tag}/snapshots/{tag}/ 并复用
vllm_kb.code_index 建符号/报错字面量索引，供 `code --repo img:{tag}` 检索。

vllm 主仓代码不提取：镜像里那份由 buildkit 的 VLLM_TAG/VLLM_REPO 锁定，
检索侧用矩阵的 vllm_commit（tag→commit）对照官方/fork 快照即可。

用法（在项目根）：
    python scripts/build_image_snapshots.py --list             # 列出候选镜像与提取状态
    python scripts/build_image_snapshots.py                    # 提取全部未提取/已变更的 0day 镜像
    python scripts/build_image_snapshots.py --tag glm5.2       # 只提取指定镜像（可多次）
    python scripts/build_image_snapshots.py --index-only       # 只重建索引（不联网）
    python scripts/build_image_snapshots.py --refresh          # 忽略 digest 锚，强制重取
    python scripts/build_image_snapshots.py --insecure         # 真实业务环境：跳过 SSL 校验

目录布局（与 forks/ 同构，检索端零改动复用 VersionedCode）：
    data/code/images/{tag}/snapshots/{tag}/vllm_ascend/... csrc/...   # 插件源码
    data/code/images/{tag}/index.sqlite3                             # 符号索引
    data/code/images/{tag}/meta.json                                 # 镜像锚与 commit 溯源
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_companion_matrix as bcm  # noqa: E402  （复用 quay/矩阵/tag→commit 工具）
import fetch_quay_tags as fq  # noqa: E402

# 插件源码在镜像里的位置（Dockerfile: COPY . /vllm-workspace/vllm-ascend/）
_PLUGIN_PREFIX = "vllm-workspace/vllm-ascend/"
_COPY_MARK = "COPY . /vllm-workspace/vllm-ascend/"
# 0day 模型镜像 = 非版本型 tag 组（版本型镜像的插件源码已由 GitHub tag 快照覆盖）
_VERSION_TAG_RE = re.compile(r"^v?\d+\.\d+\.\d+")


def images_root(code_root: Path | str) -> Path:
    return Path(code_root) / "images"


def image_dir(row_tag: str, code_root: Path | str) -> Path:
    return images_root(code_root) / row_tag


# ---------------- 层定位 ----------------

def _non_empty_history(history: list) -> list:
    return [h for h in history if not (isinstance(h, dict) and h.get("empty_layer"))]


def locate_plugin_layer(history: list, layers: list) -> int:
    """定位插件源码层在 layers 中的下标（-1 未找到）。

    Docker 层与"非空 history 条目"一一对应，据此把 created_by 里的 COPY 指令映射回层。
    """
    for i, h in enumerate(_non_empty_history(history)):
        cb = h.get("created_by", "") if isinstance(h, dict) else str(h or "")
        if cb.startswith(_COPY_MARK) or ("/vllm-workspace/vllm-ascend/" in cb and cb.startswith("COPY")):
            return i if i < len(layers) else -1
    return -1


def fetch_image_layers(tag_info: dict, token: str, insecure: bool = False,
                       qbase: str = "https://quay.io", retries: int = 4) -> dict:
    """取镜像（多架构取 amd64 子清单）的 {layers, history, digest, env}；失败抛异常。"""
    from vllm_kb.net import get_session

    session = get_session(insecure)
    api = bcm._quay_api(qbase)
    v2 = bcm._quay_v2(qbase)
    digest = tag_info["manifest_digest"]

    def _get(url, **kw):
        last = None
        for i in range(retries):
            try:
                kw.setdefault("timeout", 90)
                return session.get(url, **kw)
            except Exception as e:  # noqa: BLE001  （quay CDN 偶发握手/读超时）
                last = e
                time.sleep(2 ** i)
        raise last

    md = json.loads(_get(f"{api}/manifest/{digest}").json()["manifest_data"])
    if md.get("manifests"):
        arch = next((x for x in md["manifests"]
                     if x["platform"].get("architecture") == "amd64"), md["manifests"][0])
        md = json.loads(_get(f"{api}/manifest/{arch['digest']}").json()["manifest_data"])
    cfg = _get(f"{v2}/blobs/{md['config']['digest']}",
               headers={"Authorization": "Bearer " + token}).json()
    return {"layers": [str(l.get("digest") or "") for l in md.get("layers", []) or []],
            "layer_sizes": [int(l.get("size") or 0) for l in md.get("layers", []) or []],
            "history": cfg.get("history", []) or [],
            "session": session, "v2": v2, "token": token}


# ---------------- 层解包（tar 安全 + 前缀剥离） ----------------

def _safe_rel(name: str, prefix: str) -> str | None:
    """tar 条目名 → 相对路径（剥前缀）；不安全/不在前缀下返回 None。"""
    n = name[2:] if name.startswith("./") else name
    if not n.startswith(prefix):
        return None
    rel = n[len(prefix):].lstrip("/")
    if not rel:
        return None
    pure = Path(rel)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    return rel


def extract_plugin_tree(blob_stream, dest: Path, prefix: str = _PLUGIN_PREFIX,
                        progress=None) -> dict:
    """流式解压插件层到 dest；返回 {files, bytes, has_git, plugin_commit}。

    - 只保留 `{prefix}` 下的条目（其它路径不属于插件源码）；
    - 拒绝绝对路径与 `..`（层内容虽来自官方镜像，仍按不可信输入处理）；
    - symlink/hardlink 一律跳过（审查用源码，不需要链接）；
    - 若构建上下文里带了 .git（构建方未剥离时），顺带读出插件 commit。
    """
    dest.mkdir(parents=True, exist_ok=True)
    files = 0
    total = 0
    has_git = False
    plugin_commit = ""
    tf = tarfile.open(fileobj=blob_stream, mode="r|gz")
    for m in tf:
        if progress:
            if m.size:
                total += m.size
                progress(total)
        if not m.isfile():
            continue
        rel = _safe_rel(m.name, prefix)
        if rel is None:
            continue
        if rel.startswith(".git/") or rel == ".git":
            has_git = True
            if rel in (".git/shallow", ".git/HEAD", ".git/packed-refs") and m.size <= 65536:
                try:
                    data = tf.extractfile(m).read().decode("utf-8", "replace").strip()
                except Exception:
                    data = ""
                mm = re.search(r"\b([0-9a-f]{40})\b", data, re.IGNORECASE)
                if mm and rel == ".git/shallow":
                    plugin_commit = mm.group(1).lower()
            continue  # .git 不入快照（体积大且非源码）
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        src = tf.extractfile(m)
        if src is None:
            continue
        with open(out, "wb") as f:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        files += 1
    try:
        tf.close()
    except Exception:
        pass
    return {"files": files, "bytes": total, "has_git": has_git, "plugin_commit": plugin_commit}


def download_layer(meta: dict, layer_digest: str, layer_size: int,
                   progress_every: int = 10 << 20, retries: int = 3):
    """流式下载层 blob，返回可解压的字节流（带进度行 + 重试）。

    quay CDN 偶发握手/读超时（实测同一 tag 的 config 请求正常、层体却读超时），
    这里对**整个层体**重试；每次重试前打印一行，避免长时间静默被误判卡死。
    """
    session = meta["session"]
    last: Exception | None = None
    for attempt in range(retries):
        buf = io.BytesIO()
        got = 0
        mark = progress_every
        try:
            r = session.get(f"{meta['v2']}/blobs/{layer_digest}",
                            headers={"Authorization": "Bearer " + meta["token"]},
                            timeout=600, stream=True)
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                if not chunk:
                    continue
                buf.write(chunk)
                got += len(chunk)
                if got >= mark:
                    print(f"[img]     下载 {got / 1e6:.0f}MB / {layer_size / 1e6:.0f}MB", flush=True)
                    mark += progress_every
            buf.seek(0)
            return buf
        except Exception as e:  # noqa: BLE001  （网络/超时：整层重试）
            last = e
            print(f"[img]     下载失败（已 {got / 1e6:.0f}MB，第 {attempt + 1}/{retries} 次）："
                  f"{type(e).__name__}: {str(e)[:80]}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"层下载失败（重试 {retries} 次）：{last}")


# ---------------- 候选镜像 ----------------

def candidate_images(tags: list[dict], only: list[str] | None = None) -> list[dict]:
    """候选 = 看护 tag 里的**非版本型**组（0day 模型镜像），每组取代表 tag。

    版本型镜像（v0.23.0 等）的插件源码已由 GitHub tag 快照覆盖，无需从镜像层取。
    only 可给组名、代表 tag 或组内任一平台变体 tag（如 glm5.2 / glm5.2-a3）。
    """
    groups = bcm.group_base_versions(tags)
    wanted = set(only or [])
    out: list[dict] = []
    for base in sorted(groups):
        if base_version_key(base):
            continue
        rep = bcm.pick_representative(groups[base])
        variants = [t["name"] for t in groups[base]]
        if wanted and not ({base, rep["name"]} & wanted) and not (set(variants) & wanted):
            continue
        out.append({"group": base, "tag": rep["name"], "tag_info": rep, "variants": variants})
    return out


def base_version_key(tag: str) -> str:
    return bcm.base_version_key(tag)


# ---------------- 主流程 ----------------

def _matrix_rows(cfg) -> dict:
    """矩阵行（key=vllm-ascend）——用于把 vllm_commit/vllm_base/image_created 带进 meta。"""
    path = cfg.resolve(cfg.storage.companion_file)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {r.get("vllm-ascend", ""): r for r in (data.get("rows") or [])}


def extract_one(cand: dict, cfg, token: str, matrix: dict, insecure: bool = False,
                qbase: str = "https://quay.io", refresh: bool = False,
                index_only: bool = False) -> dict:
    """提取单个镜像的插件源码 + 建索引 + 写 meta。返回状态 dict。"""
    from vllm_kb.code_index import VersionedCode

    tag = cand["tag"]
    root = image_dir(tag, cfg.resolve(cfg.storage.code_root))
    snap = root / "snapshots" / tag
    meta_path = root / "meta.json"
    cur_digest = cand["tag_info"].get("manifest_digest", "")
    old_meta: dict = {}
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            old_meta = {}

    need_extract = refresh or not (snap.is_dir() and any(snap.iterdir()))
    if not need_extract and old_meta.get("image_digest") == cur_digest:
        print(f"[img] {tag}: 镜像未变更（digest 锚命中），跳过提取", flush=True)
    elif index_only:
        if not (snap.is_dir() and any(snap.iterdir())):
            print(f"[img] {tag}: 无快照，--index-only 跳过", flush=True)
            return {"tag": tag, "status": "skip"}
    else:
        meta = fetch_image_layers(cand["tag_info"], token, insecure=insecure, qbase=qbase)
        idx = locate_plugin_layer(meta["history"], meta["layers"])
        if idx < 0:
            print(f"[img] {tag}: 未找到插件层（{_COPY_MARK}），跳过", flush=True)
            return {"tag": tag, "status": "no-layer"}
        layer_digest = meta["layers"][idx]
        layer_size = meta["layer_sizes"][idx] if idx < len(meta["layer_sizes"]) else 0
        print(f"[img] {tag}: 插件层 layer[{idx}] {layer_size / 1e6:.1f}MB，下载并解包 ...", flush=True)
        if snap.exists():
            # 清掉旧快照（镜像可能重推，文件集可能变化）
            import shutil

            shutil.rmtree(snap)
        blob = download_layer(meta, layer_digest, layer_size)
        info = extract_plugin_tree(blob, snap)
        print(f"[img] {tag}: 解出 {info['files']} 个文件（{info['bytes'] / 1e6:.1f}MB 原始）"
              f"{'，含 .git' if info['has_git'] else ''}", flush=True)
        row = matrix.get(cand["group"], {}) or {}
        vllm_commit = row.get("vllm_commit", "")
        vllm_commit_date = row.get("vllm_commit_date", "")
        if not vllm_commit and row.get("vllm") and bcm._github_token():
            # 矩阵没解析过 commit：现场按 tag 解析（未配 token 时跳过，留空待补）
            res = bcm.fetch_tag_commit(bcm.OFFICIAL_VLLM_REPO, f"v{row.get('vllm')}",
                                       insecure=insecure, gbase=os.environ.get(
                                           "VLLM_KB_GITHUB_BASE", "https://api.github.com"))
            vllm_commit = res.get("sha", "") or ""
            vllm_commit_date = vllm_commit_date or res.get("date", "")
        meta_out = {
            "tag": tag,
            "group": cand["group"],
            "variants": cand["variants"],
            "image_digest": cur_digest,
            "image_created": row.get("image_created", "") or bcm._image_created(cand["tag_info"]),
            "layer_digest": layer_digest,
            "layer_size": layer_size,
            "plugin_layer_index": idx,
            "plugin_commit": info["plugin_commit"],
            "files": info["files"],
            "vllm_baseline": row.get("vllm", "") or row.get("vllm_base", ""),
            "vllm_commit": vllm_commit,
            "vllm_commit_date": vllm_commit_date,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        root.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 建索引（复用 VersionedCode：img:{tag} → data/code/images/{tag}/）
    code = VersionedCode(cfg, repo=f"img:{tag}")
    n = code.build_index_for_version(tag)
    print(f"[img] {tag}: 索引 {n} 个文件", flush=True)
    return {"tag": tag, "status": "ok", "indexed": n,
            "meta": json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}}


def main() -> None:
    from vllm_kb.config import AppConfig
    from vllm_kb.net import add_insecure_args, insecure_from_env, quay_base

    ap = argparse.ArgumentParser(description="提取 0day 镜像内的 vllm-ascend 插件源码（只拉插件层）")
    ap.add_argument("--tag", action="append", default=None,
                    help="只处理指定镜像（quay tag 或矩阵行键，可多次）")
    ap.add_argument("--list", action="store_true", help="只列出候选镜像与提取状态")
    ap.add_argument("--index-only", action="store_true", help="只重建索引（不联网）")
    ap.add_argument("--refresh", action="store_true", help="忽略 digest 锚，强制重取")
    ap.add_argument("--config", default=None)
    add_insecure_args(ap)
    args = ap.parse_args()

    insecure = args.insecure or insecure_from_env()
    qbase = quay_base(args.quay_base)
    cfg = AppConfig.load(args.config, require_keys=False)
    matrix = _matrix_rows(cfg)
    dest_root = images_root(cfg.resolve(cfg.storage.code_root))

    if args.index_only:
        cands = []
        if dest_root.is_dir():
            cands = [{"group": d.name, "tag": d.name, "tag_info": {}, "variants": []}
                     for d in sorted(p for p in dest_root.iterdir() if p.is_dir())]
        if args.tag:
            wanted = set(args.tag)
            cands = [c for c in cands if c["tag"] in wanted or c["group"] in wanted]
        for c in cands:
            extract_one(c, cfg, "", matrix, insecure=insecure, qbase=qbase, index_only=True)
        print(f"[img] 索引重建完成：{len(cands)} 个镜像", flush=True)
        return

    tags = fq.fetch_tags(insecure=insecure, base=qbase)
    cands = candidate_images(tags, only=args.tag)
    print(f"[img] 候选 0day 镜像 {len(cands)} 个（非版本型 tag 组，取代表 tag）", flush=True)

    if args.list:
        from vllm_kb.code_index import list_image_snapshots

        extracted = {e["tag"]: e for e in list_image_snapshots(cfg.resolve(cfg.storage.code_root))}
        for c in cands:
            e = extracted.get(c["tag"], {})
            row = matrix.get(c["group"], {}) or {}
            state = "已提取" if e.get("extracted") else "未提取"
            indexed = f"，索引={'有' if e.get('indexed') else '无'}"
            print(f"  {c['tag']:<34} {state}{indexed}  vllm={row.get('vllm') or '-':<8} "
                  f"commit={(row.get('vllm_commit') or '-')[:12]}  "
                  f"组={c['group']}  变体={len(c['variants'])}", flush=True)
        print(f"[img] 目标目录：{dest_root}", flush=True)
        print("[img] 检索前缀：repo=img:{tag}（tag = 上表第一列；组名/变体仅用于对应"
              "用户口中的镜像名，如「hy4 镜像」实际 tag 是 hy4-a3）", flush=True)
        return

    token = bcm.get_quay_token(insecure=insecure, qbase=qbase)
    ok = skipped = failed = 0
    start = time.time()
    for c in cands:
        try:
            r = extract_one(c, cfg, token, matrix, insecure=insecure, qbase=qbase,
                            refresh=args.refresh)
            if r["status"] == "ok" and r.get("indexed"):
                ok += 1
            else:
                skipped += 1
        except Exception as e:  # noqa: BLE001  （单镜像失败不阻塞其余；quay CDN 抖动常见）
            failed += 1
            print(f"[img] [!] {c['tag']} 提取失败：{type(e).__name__}: {e}", flush=True)
    print(f"[img] 完成：成功 {ok}，跳过 {skipped}，失败 {failed}，耗时 {time.time() - start:.0f}s",
          flush=True)
    print("[img] 检索：client.py code-versions --repo img   /   code <符号> --repo img:<tag>",
          flush=True)


if __name__ == "__main__":
    main()
