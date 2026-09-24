"""文件型来源的共用骨架：发现 → 身份 → 资产落盘 → 解析缓存。

## 为什么单独一个模块

四种文件型来源（pdf / markdown / word / excel）原先各自实现"扫哪个目录、怎么收敛历史版本、
怎么造 source_id、怎么消歧同名"，于是同一类 bug 要修四遍（"更新一篇文档产生重复文档"
就是这么来的）。这里把与**格式无关**的那部分收成一个可复用的契约。

## 契约（未来重构的接口）

- **输入**：`DiscoverSpec`——glob 形态、导入路径（**文件或目录**）、资产层目录，
  外加纯展示用的前缀/标签；
- **输出**：`Discovery`——有序的 `SourceFile` 列表 + 是否回退 + 诊断（资产层多余族）；
- **不变量**：
  1. `SourceFile.stem` 是**族名**，在内容改动下稳定（不含 `_copy_asset` 写的 `.<sha12>`）
     ——它就是文档身份，漂移等于换了一篇文档（标签/补标/遥测全丢）；
  2. **导入路径优先**：它有匹配文件时，资产层不参与文档集（只参与诊断）；
  3. `SourceFile.duplicate` 为真时必须用路径指纹消歧（`sid()` 负责），否则同族名会
     互相覆盖（`docs` 是 `INSERT OR REPLACE`，后者胜）。

## 依赖

本模块只依赖标准库与**路径**语义：不 import 任何解析库，也不知道 `source_type` 字符串，
更不碰数据库。格式相关的一切由调用方注入。
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

# 资产层"同名异内容"版本后缀（`_copy_asset` 写的 `stem.<sha12>.suffix`）
_VER_SUFFIX_RE = re.compile(r"^(?P<base>.+)\.(?P<sha>[0-9a-f]{12})$")


def path_tag(rel: str) -> str:
    """相对路径指纹（8 hex），用于同名文件消歧。

    用**路径**而非内容的 sha：文件每次编辑内容都变，内容寻址会让 `source_id` 漂移
    （旧文档成孤儿、审核状态/遥测丢失、检索重复）；路径指纹在内容改动下稳定，
    且**不可读**——`source_id` 会出现在 /search 的 doc_id、/doc/{id} 与遥测库里，
    不能暴露目录名（可能含客户/项目名）。
    """
    return hashlib.sha256(rel.replace("\\", "/").encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class SourceFile:
    """一个待处理文件：路径 + 来源层 + **逻辑族名**（= 文档身份）+ 是否需消歧。"""

    path: Path
    from_imports: bool
    stem: str
    duplicate: bool = False

    def rel_key(self, base_dir: Path) -> str:
        """相对 base_dir 的 posix 路径（消歧指纹的输入；取不到则退化为文件名）。"""
        try:
            return self.path.relative_to(base_dir).as_posix()
        except ValueError:
            return self.path.name

    def sid(self, prefix: str, import_path: Path, assets_dir: Path) -> str:
        """文档身份：`<prefix>:<族名>`；同族名多份时加相对路径指纹 `--<sha8>`。

        默认（不重名）不加后缀 —— 存量 `source_id` 不漂移。
        """
        if not self.duplicate:
            return f"{prefix}:{self.stem}"
        base = import_path if self.from_imports else assets_dir
        return f"{prefix}:{self.stem}--{path_tag(self.rel_key(base))}"


@dataclass(frozen=True)
class DiscoverSpec:
    """文件发现的输入。字段都是**纯数据**，便于未来换成 registry/plugin 形态。"""

    patterns: tuple[str, ...]
    import_path: Path
    assets_dir: Path
    sid_prefix: str = ""
    label: str = "文件"
    log_prefix: str = "[sources]"
    # 回退时附加的说明（各来源特有，例如 md 的"图片相对路径已失锚"）
    fallback_note: str = ""


@dataclass(frozen=True)
class Discovery:
    """文件发现的输出。`stray` 只用于诊断，不参与文档集。"""

    files: tuple[SourceFile, ...]
    fallback: bool
    stray: tuple[str, ...] = ()


def _family_key(p: Path) -> tuple[str, str]:
    """版本族键 `(族名, 后缀)`——`x.md` 与 `x.<sha12>.md` 同族。"""
    m = _VER_SUFFIX_RE.match(p.stem)
    return (m.group("base") if m else p.stem, p.suffix.lower())


def _families(paths: list[Path]) -> dict[tuple[str, str], tuple[float, str, Path, str]]:
    """按版本族分组，保留 `(mtime, 文件名, 路径, 族名)`；mtime 取不到的文件跳过。"""
    fams: dict[tuple[str, str], tuple[float, str, Path, str]] = {}
    for p in paths:
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        key = _family_key(p)
        cand = (mtime, p.name, p, key[0])
        cur = fams.get(key)
        if cur is None or (cand[0], cand[1]) > (cur[0], cur[1]):
            fams[key] = cand
    return fams


def latest_versions(paths: list[Path]) -> list[tuple[Path, str]]:
    """资产层扁平副本按**版本族**收敛，返回 `[(最新版本的路径, 族名)]`（按路径排序）。

    `case.md` 与 `case.<sha12>.md` 是同一篇的不同版本，每族只保留最新（mtime 最大；
    并列时取文件名字典序较大者，保证确定性）的一个。

    `_copy_asset` 在"同名异内容"时写成 `stem.<sha12>.suffix`，所以资产层会累积历史版本
    （原始 `case.md` 永久保留 + 每个不同内容一份）。回退模式若把它们都当文档，会把旧版本
    **复活成独立文档**（源文件删掉后尤其明显，一篇变多篇）。

    返回的**族名**（而非带 `.<sha12>` 的文件名）才是文档身份：否则收敛到 sha 副本时
    `source_id` 会从 `md:case` 漂移成 `md:case.<sha12>`，等于换了篇文档（审核状态/标签丢失）。

    后缀无关（按 `(族名, 后缀)` 分组），各扁平副本层共用。
    """
    fams = _families(paths)
    return sorted(((v[2], v[3]) for v in fams.values()), key=lambda t: t[0])


def _scan_imports(spec: DiscoverSpec) -> list[Path]:
    """导入路径下的候选文件（**文件或目录**都能吃）。"""
    p = spec.import_path
    if p.is_file():
        return [p] if any(p.match(pat) for pat in spec.patterns) else []
    if not p.is_dir():
        return []
    out: list[Path] = []
    for pat in spec.patterns:
        out.extend(sorted(p.rglob(pat)))
    return out


def discover(spec: DiscoverSpec) -> Discovery:
    """发现待处理文件：**导入路径优先**，为空则回退资产层扁平副本（按版本族收敛）。

    - 导入路径是**文件**：后缀不匹配时明确提示（配置写错不该静默改读别处）；
    - 导入路径是**目录**：`rglob` 保留操作者的目录树 → 同名不同目录可按键路径指纹消歧，
      且编辑源文件不会因资产层累积副本而变成多篇；
    - 导入路径为空/不存在 → 回退资产层，族名作 stem（见 `latest_versions`）；
    - 导入路径非空时额外做一次**诊断**：资产层有、导入路径没有的族 → 这些文档已不再纳入
      canonical。**必须打出来**：`ingest` 不删文档、`upsert_unified_canonical` 也不删行，
      于是"文件不再被刷新"是**静默偏离**，比直接丢掉更难发现。
    """
    pref = spec.log_prefix
    imports = _scan_imports(spec)
    if spec.import_path.is_file() and not imports:
        print(f"{pref} ⚠ 配置的导入路径是文件但后缀不匹配（{spec.import_path}）："
              f"期望 {', '.join(spec.patterns)}；已跳过该来源", flush=True)
        return Discovery((), False)

    if imports:
        stray: list[str] = []
        if spec.assets_dir.exists():
            covered = {(p.stem, p.suffix.lower()) for p in imports}
            names: list[str] = []
            for key, (_mtime, name, _p, base) in _families(_all_asset_files(spec)).items():
                if (base, key[1]) not in covered:
                    names.append(name)
            stray = sorted(names)
        if stray:
            shown = ", ".join(stray[:5]) + (" …" if len(stray) > 5 else "")
            print(f"{pref} ⚠ 资产层有 {len(stray)} 个族不在导入路径中，**已不再纳入 canonical**"
                  f"（如需继续纳管，请复制到 {spec.import_path}）：{shown}", flush=True)
        return Discovery(_with_duplicates(imports, True, spec), False, tuple(stray))

    if not spec.assets_dir.exists():
        # 导入路径与资产层都没有：不喊"回退"（没有副本可退），由调用方给出自己的提示
        return Discovery((), True)
    cand: list[Path] = []
    for pat in spec.patterns:
        cand.extend(sorted(spec.assets_dir.glob(pat)))
    fams = _families(cand)
    files = tuple(
        SourceFile(path=v[2], from_imports=False, stem=v[3])
        for v in sorted(fams.values(), key=lambda t: t[2])
    )
    note = f"；{spec.fallback_note}" if spec.fallback_note else ""
    print(f"{pref} ⚠ 导入路径无 {spec.label} 文件（{spec.import_path}），已回退到资产层副本 "
          f"{spec.assets_dir}（按版本族收敛，每族只取最新一份）{note}", flush=True)
    return Discovery(files, True)


def _all_asset_files(spec: DiscoverSpec) -> list[Path]:
    out: list[Path] = []
    for pat in spec.patterns:
        out.extend(sorted(spec.assets_dir.glob(pat)))
    return out


def _with_duplicates(paths: list[Path], from_imports: bool,
                     spec: DiscoverSpec) -> tuple[SourceFile, ...]:
    """给导入路径的文件标记"同族名多份"（需路径指纹消歧），并打一次告警。"""
    counts: dict[str, int] = {}
    for p in paths:
        counts[p.stem] = counts.get(p.stem, 0) + 1
    dups = {s for s, n in counts.items() if n > 1}
    if dups:
        shown = ", ".join(sorted(dups)[:5]) + (" …" if len(dups) > 5 else "")
        print(f"{spec.log_prefix} ⚠ 检测到 {len(dups)} 组同名 {spec.label}（{shown}）："
              f"已用相对路径指纹消歧为 {spec.sid_prefix}:<stem>--<sha8>，避免互相覆盖",
              flush=True)
    return tuple(SourceFile(path=p, from_imports=from_imports, stem=p.stem,
                            duplicate=p.stem in dups)
                 for p in paths)
