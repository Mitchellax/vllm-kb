"""版本化代码仓：预存主要镜像版本的源码快照 + 符号索引 + 按需读取。

设计：
- 快照以 zip 存储（每版本 ~14MB，41 版本 ~570MB 原始下载 / 磁盘占用小），
  解压目录为缓存（data/code/snapshots/{version}/），首次访问按需解压；
- 符号索引（SQLite）：算子名/内核名/函数名 -> {version, file, line, snippet}，
  构建时一次性提取（不依赖网络，离线可用）；
- 检索流程：签名/关键词 + 版本 -> 符号索引定位 -> 解压读取代码片段；
- 全部只读：本模块不提供任何写入口给 API（构建由 scripts/build_code_snapshots.py 触发）。

目录布局：
    data/code/
        index.sqlite3                  # 符号索引（只读连接）
        zips/{version}.zip             # 源码快照（zip）
        snapshots/{version}/...        # 解压缓存（可删，zip 是事实源）
"""
from __future__ import annotations

import json
import re
import sqlite3
import zipfile
from pathlib import Path
from typing import Optional

from .config import AppConfig

# 符号提取规则（Python / C++ / 配置）
_PY_DEF_RE = re.compile(
    r"^(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(|"
    r"^class\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.M,
)
_CPP_DEF_RE = re.compile(
    r"\b(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)|"
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*(?:const)?\s*\{",
)
_OP_ATTR_RE = re.compile(r"\b(aclnn[A-Z][A-Za-z0-9_]*|npu[A-Z][A-Za-z0-9_]*)")
_KERNEL_NAME_RE = re.compile(r"\b(dispatch_ffn_combine|mega_moe|[a-z0-9_]+_combine|[a-z0-9_]+_dispatch)\b")
_ENV_NAME_RE = re.compile(r"\b(VLLM_ASCEND_[A-Z0-9_]+|HCCL_[A-Z0-9_]+)")

# 重要文件白名单（提取符号时只扫这些目录，控制索引体积）
_INDEXED_SUBDIRS = ("csrc", "vllm_ascend")
_EXCLUDE_DIRS = ("tests", "third_party", "docs", "examples", "benchmarks")


class CodeIndexError(RuntimeError):
    pass


class VersionedCode:
    """版本化代码仓访问器（只读）。

    cfg: AppConfig（读 storage.code_root / code.versions）
    repo: 仓库子目录名——"vllm-ascend"（默认，data/code/）| "vllm"（data/code/vllm/）。
    """

    def __init__(self, cfg: AppConfig, repo: str = "vllm-ascend"):
        self.cfg = cfg
        self.repo = repo
        self.root = cfg.resolve(cfg.storage.code_root)
        if repo and repo != "vllm-ascend":
            self.root = self.root / repo
        self.index_path = self.root / "index.sqlite3"
        self.zips_dir = self.root / "zips"
        self.snapshots_dir = self.root / "snapshots"

    # ---------------- 版本与快照 ----------------

    @property
    def available_versions(self) -> list[str]:
        """已预存（有 zip 或已解压）的版本列表。"""
        versions: set[str] = set()
        if self.zips_dir.exists():
            for p in self.zips_dir.glob("*.zip"):
                versions.add(p.stem)
        if self.snapshots_dir.exists():
            for p in self.snapshots_dir.iterdir():
                if p.is_dir():
                    versions.add(p.name)
        return sorted(versions)

    def has_version(self, version: str) -> bool:
        return (self.zips_dir / f"{version}.zip").exists() or (self.snapshots_dir / version).is_dir()

    def _snapshot_dir(self, version: str) -> Path:
        return self.snapshots_dir / version

    def snapshot_exists(self, version: str) -> bool:
        return (self.snapshots_dir / version / "vllm_ascend").exists() or \
               any((self.snapshots_dir / version).glob("*")) and (self.snapshots_dir / version).is_dir()

    def ensure_snapshot(self, version: str) -> Path:
        """确保版本已解压，返回快照根目录（zip 存在时按需解压）。"""
        snap = self._snapshot_dir(version)
        if snap.is_dir() and any(snap.iterdir()):
            return snap
        zpath = self.zips_dir / f"{version}.zip"
        if not zpath.exists():
            raise CodeIndexError(
                f"版本 {version} 未预存：可用版本 {self.available_versions or '(无)'}。"
                "运行 scripts/build_code_snapshots.py 预存。"
            )
        snap.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(snap)
        return snap

    def find_file(self, version: str, path: str) -> Optional[Path]:
        """按仓库相对路径（如 csrc/mc2/.../dispatch_ffn_combine.cpp）找文件。"""
        snap = self._repo_root(self.ensure_snapshot(version))
        rel = path.lstrip("/")
        cand = snap / rel
        return cand if cand.is_file() else None

    def read_file(self, version: str, path: str, max_chars: int = 6000,
                  mark_truncation: bool = True) -> Optional[str]:
        """读取指定版本源码文件（截断保护：默认 max_chars 字符）。

        mark_truncation=True（默认）：截断时在末尾追加明确标记
        "\n... (已截断，共 N 字符，仅前 max_chars)"，调用方（agent）能识别
        内容不完整，而不是拿到一段被静默切掉的代码。
        """
        f = self.find_file(version, path)
        if f is None:
            return None
        text = f.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            if mark_truncation:
                return text[:max_chars] + f"\n... (已截断：全文 {len(text)} 字符，仅前 {max_chars})"
            return text[:max_chars]
        return text

    # ---------------- 符号索引 ----------------

    def _connect_index(self, write: bool = False) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        if write:
            conn = sqlite3.connect(str(self.index_path))
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS symbols (
                  version TEXT NOT NULL,
                  symbol  TEXT NOT NULL,
                  file    TEXT NOT NULL,
                  line    INTEGER,
                  snippet TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sym_ver ON symbols(symbol, version);
                """
            )
            return conn
        if not self.index_path.exists():
            raise CodeIndexError("符号索引不存在：运行 scripts/build_code_snapshots.py 构建")
        return sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True)

    def search_symbols(
        self,
        symbol: str,
        version: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """符号精确检索：符号名 + 可选版本过滤。"""
        conn = self._connect_index(write=False)
        try:
            if version:
                rows = conn.execute(
                    "SELECT version, file, line, snippet FROM symbols "
                    "WHERE symbol = ? AND version = ? LIMIT ?",
                    (symbol.lower(), version, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT version, file, line, snippet FROM symbols "
                    "WHERE symbol = ? LIMIT ?",
                    (symbol.lower(), limit),
                ).fetchall()
            return [
                {"version": r[0], "file": r[1], "line": r[2], "snippet": r[3]}
                for r in rows
            ]
        finally:
            conn.close()

    def grep(
        self,
        keyword: str,
        version: Optional[str] = None,
        limit: int = 20,
        path_sub: Optional[str] = None,
        per_version: bool = False,
    ) -> list[dict]:
        """关键词全文检索（在已解压快照里 grep）。返回文件:行号 + 行内容。

        - path_sub：限定文件路径子串（如 "worker/model_runner_v1.py"）——避免全仓命中
          把目标文件的行挤出结果（教训：dummy graph capture 修复曾被其他命中淹没）；
        - per_version=True：每个版本各自收集（不因全局 limit 提前截断），
          返回各版本命中的行号，便于对比"哪个版本引入/移动了该代码"。
        """
        versions = [version] if version else self.available_versions
        hits: list[dict] = []
        kw_lower = keyword.lower()
        for v in versions:
            try:
                snap = self._repo_root(self.ensure_snapshot(v))
            except CodeIndexError:
                if version:
                    raise  # 显式指定版本未预存：明确报错（API 转 404），不静默返回空
                continue  # 全量遍历：跳过缺失版本
            v_hits: list[dict] = []
            for ext in ("*.py", "*.cpp"):
                for p in snap.rglob(ext):
                    if any(seg in p.parts for seg in _EXCLUDE_DIRS):
                        continue
                    rel = p.relative_to(snap).as_posix()
                    if path_sub and path_sub.lower() not in rel.lower():
                        continue
                    try:
                        for lineno, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                            if kw_lower in line.lower():
                                v_hits.append({
                                    "version": v, "file": rel, "line": lineno,
                                    "snippet": line.strip()[:160],
                                })
                                if not per_version and len(hits) >= limit:
                                    return hits
                    except Exception:
                        continue
            if per_version:
                hits.extend(v_hits[:limit])
            else:
                hits.extend(v_hits)
                if len(hits) >= limit:
                    return hits
        return hits

    # ---------------- 构建（由脚本调用） ----------------

    def build_index_for_version(self, version: str) -> int:
        """为单个已解压版本构建符号索引。返回索引符号数。"""
        snap = self.ensure_snapshot(version)
        conn = self._connect_index(write=True)
        count = 0
        indexed_dirs = ("vllm",) if self.repo == "vllm" else _INDEXED_SUBDIRS
        try:
            conn.execute("DELETE FROM symbols WHERE version = ?", (version,))
            # 兼容 zip 解压带顶层目录（vllm-ascend-0.23.0rc1/）：统一到真实根
            root = self._repo_root(snap)
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(root).as_posix()
                if not any(rel.startswith(d + "/") for d in indexed_dirs):
                    continue
                if any(f"/{d}/" in f"/{rel}" for d in _EXCLUDE_DIRS):
                    continue
                if p.suffix in (".py", ".cpp", ".hpp", ".h", ".cc", ".cxx"):
                    self._index_file(conn, version, rel, p)
                    count += 1
            conn.commit()
        finally:
            conn.close()
        return count

    @staticmethod
    def _repo_root(snap: Path) -> Path:
        """快照目录下真实仓库根：若只有一个子目录（zip 顶层目录），下钻。"""
        if snap.is_dir():
            subdirs = [d for d in snap.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                # zip 顶层目录（vllm-ascend-0.23.0rc1/ 或 vllm-0.22.1/）
                return subdirs[0]
        return snap

    @staticmethod
    def _index_file(conn: sqlite3.Connection, version: str, rel: str, p: Path) -> None:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        lines = text.splitlines()
        syms: set[tuple[str, int, str]] = set()
        for m in _PY_DEF_RE.finditer(text):
            name = m.group(1) or m.group(2)
            if name:
                line_no = text[: m.start()].count("\n") + 1
                snippet = lines[line_no - 1].strip()[:120] if line_no <= len(lines) else ""
                syms.add((name.lower(), line_no, snippet))
        for m in _CPP_DEF_RE.finditer(text):
            name = m.group(2) or m.group(3)
            if name and len(name) >= 4:
                line_no = text[: m.start()].count("\n") + 1
                snippet = lines[line_no - 1].strip()[:120] if line_no <= len(lines) else ""
                syms.add((name.lower(), line_no, snippet))
        for m in _OP_ATTR_RE.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            snippet = lines[line_no - 1].strip()[:120] if line_no <= len(lines) else ""
            syms.add((m.group(1).lower(), line_no, snippet))
        for m in _KERNEL_NAME_RE.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            snippet = lines[line_no - 1].strip()[:120] if line_no <= len(lines) else ""
            syms.add((m.group(1).lower(), line_no, snippet))
        for m in _ENV_NAME_RE.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            snippet = lines[line_no - 1].strip()[:120] if line_no <= len(lines) else ""
            syms.add((m.group(1).lower(), line_no, snippet))
        for sym, ln, sn in syms:
            conn.execute(
                "INSERT OR IGNORE INTO symbols (version, symbol, file, line, snippet) VALUES (?,?,?,?,?)",
                (version, sym, rel, ln, sn),
            )


def load_versioned_code(cfg: Optional[AppConfig] = None) -> VersionedCode:
    cfg = cfg or AppConfig.load()
    return VersionedCode(cfg)
