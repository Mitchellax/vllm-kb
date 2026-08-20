"""组件配套关系矩阵 + 组件查询解析。

配套矩阵（默认 data/compatibility/vllm-ascend.json）：以 vllm-ascend 版本为主键，
记录配套的 vllm / cann / pytorch / pytorch-ascend / npu-driver 版本。
- 来源：quay.io ascend/vllm-ascend 镜像 build history（scripts/fetch_quay_tags.py 辅助），
  或人工核对后手工填写（脚本无法解析时）；
- 只管理正式版与 rc 版；nightly 与 "-release"（branch 镜像）不维护；
- 查询时按 "组件:版本" 反向查配套（expand），把其他组件文档的版本权重关联进来。

查询格式："vllm-ascend:0.18.0 GLM5.1 PD分离P节点挂死"
    -> component="vllm-ascend", version="0.18.0", rest="GLM5.1 PD分离P节点挂死"（语义检索词）
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .components import COMPONENTS
from .confidence import parse_version, version_distance


class CompanionRow(BaseModel):
    """一行配套：vllm-ascend 版本 -> 其余组件配套版本。

    JSON 键使用组件名（含连字符，如 "vllm-ascend"），通过 alias 映射到字段。
    source 记录来源（自动匹配 / 人工），便于核对。
    """

    model_config = ConfigDict(populate_by_name=True)

    vllm_ascend: str = Field(default="", alias="vllm-ascend")
    vllm: str = ""
    cann: str = ""
    pytorch: str = ""
    pytorch_ascend: str = Field(default="", alias="pytorch-ascend")
    npu_driver: str = Field(default="", alias="npu-driver")
    notes: str = ""  # 已知问题/注意事项（可含 NPU 驱动已知问题、SOC/python 环境）
    source: str = ""  # 来源：自动(镜像env+release) / 人工 / 待人工


class CompanionMatrix:
    """配套矩阵：精确匹配 + 最近版本匹配 + 反向配套展开。"""

    def __init__(self, rows: list[CompanionRow]):
        self.rows = rows
        self._by: dict[tuple[str, str], list[CompanionRow]] = {}
        for r in rows:
            for comp, key in self._iter_versions(r):
                if key:
                    self._by.setdefault((comp, key), []).append(r)

    @staticmethod
    def _iter_versions(r: CompanionRow):
        yield "vllm-ascend", r.vllm_ascend
        yield "vllm", r.vllm
        yield "cann", r.cann
        yield "pytorch", r.pytorch
        yield "pytorch-ascend", r.pytorch_ascend
        yield "npu-driver", r.npu_driver

    @classmethod
    def load(cls, path: str | Path | None) -> Optional["CompanionMatrix"]:
        """加载矩阵；文件缺失/为空返回 None（配套功能静默降级）。加载后对不完整行告警。"""
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data.get("rows", []) if isinstance(data, dict) else data
        if not rows:
            return None
        matrix = cls([CompanionRow.model_validate(r) for r in rows])
        matrix._warn_gaps()
        return matrix

    def _warn_gaps(self) -> None:
        """配套行不完整（缺 vllm/cann/pytorch/pytorch-ascend）时告警（每进程一次）。

        npu-driver(HDK) 不与镜像版本耦合（特定 HDK 有特定问题），缺失符合预期，不告警。
        """
        global _GAP_WARNED
        if _GAP_WARNED["done"]:
            return
        required = ("vllm", "cann", "pytorch", "pytorch-ascend")
        gaps: list[tuple[str, list[str]]] = []
        for r in self.rows:
            missing = [f for f in required if not getattr(r, f.replace("-", "_"), "")]
            if missing:
                gaps.append((r.vllm_ascend or "?", missing))
        if gaps:
            sample = gaps[0]
            print(
                f"[warn] 配套矩阵 {len(gaps)}/{len(self.rows)} 行不完整"
                f"（如 {sample[0]} 缺 {', '.join(sample[1])}）："
                "运行 python scripts/build_companion_matrix.py 自动匹配，或人工补齐",
                flush=True,
            )
        _GAP_WARNED["done"] = True

    def get(self, component: str, version: str) -> list[CompanionRow]:
        """精确匹配（字符串相等，容忍 v 前缀差异）。"""
        cands = []
        for row in self._by.get((component, version), []):
            cands.append(row)
        if cands:
            return cands
        # 容忍 "0.18.0" vs "v0.18.0" 之类
        norm = version.lstrip("vV")
        for (comp, key), rows in self._by.items():
            if comp == component and key.lstrip("vV") == norm:
                cands.extend(rows)
        return cands

    def nearest(self, component: str, version: str, k: int = 3) -> list[CompanionRow]:
        """按版本距离取最近的 k 行（无精确匹配时的模糊配套）。"""
        tv = parse_version(version)
        if tv is None:
            return []
        scored = []
        for row in self.rows:
            rv = parse_version(getattr(row, _field_for(component), ""))
            if rv is None:
                continue
            scored.append((version_distance(tv, rv), row))
        scored.sort(key=lambda t: t[0])
        return [r for _, r in scored[:k]]

    def expand(self, component: str, version: str) -> dict[str, list[str]]:
        """反向配套：查询 (component, version) -> 其他组件 -> 版本列表（去重、保序）。

        例如 query("vllm-ascend", "0.18.0") -> {"vllm": ["0.12.1"], "cann": ["8.1.RC2"], ...}
        """
        rows = self.get(component, version) or self.nearest(component, version, k=3)
        out: dict[str, list[str]] = {}
        for row in rows:
            for comp, key in self._iter_versions(row):
                if comp == component or not key:
                    continue
                if key not in out.setdefault(comp, []):
                    out[comp].append(key)
        return out

    def resolve(self, component: str, version: str) -> dict[str, list[str]]:
        """查询组件自身的候选版本（精确优先，其次最近版本）。"""
        exact = self.get(component, version)
        if exact:
            return {component: [version]}
        near = self.nearest(component, version, k=3)
        versions = []
        for row in near:
            v = getattr(row, _field_for(component), "")
            if v and v not in versions:
                versions.append(v)
        return {component: versions}


def _field_for(component: str) -> str:
    return component.replace("-", "_")


# ---------------- 组件查询解析 ----------------

# "vllm-ascend:0.18.0 GLM5.1 PD分离P节点挂死" / "vllm-ascend:glm5 GLM5 挂死"（模型专属镜像名也可作版本）
_COMPONENT_QUERY_RE = re.compile(
    r"^\s*([A-Za-z0-9_\-]+)\s*:\s*([A-Za-z0-9][A-Za-z0-9._\-]*)\s+(.+)$", re.S
)

# 缺口告警每进程只打一次，避免反复加载矩阵时刷屏
_GAP_WARNED = {"done": False}


def parse_component_query(query: str) -> tuple[Optional[str], Optional[str], str]:
    """解析 "组件:版本 其余内容" 前缀；无前缀则视为普通查询。

    返回 (component, version, semantic_rest)。
    """
    m = _COMPONENT_QUERY_RE.match(query)
    if m:
        return m.group(1), m.group(2), m.group(3).strip()
    return None, None, query.strip()
