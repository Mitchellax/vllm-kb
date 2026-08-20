"""组件定义与版本提取：vllm / vllm-ascend / cann / pytorch / pytorch-ascend / npu-driver。

知识库主要服务于 vllm-ascend 故障排查，需要在这几个组件之间建立配套关系：
- 每个知识文档（KbDocument）标记主组件 component 与主组件版本（version_span）；
- 正文/环境段里提到的其他组件版本提取到 component_versions（如 vllm-ascend issue 里
  同时出现 "vllm version: 0.12.1"、"CANN 8.1.RC2"，用于跨组件关联）。
"""
from __future__ import annotations

import re

# 规范组件名（配套矩阵 JSON 与查询组件名统一使用这些）
COMPONENTS = ["vllm", "vllm-ascend", "cann", "pytorch", "pytorch-ascend", "npu-driver"]

# 每个组件：别名列表（处理顺序决定歧义）+ 版本正则。
# vllm 系要求 0.x.y 完整 semver（vLLM/vllm-ascend 版本线永远是 0.x），
# 避免 "vllm 0.1"、"26.1.0" 这类残缺/噪音版本；cann 等允许 x.y / x.y.z / 带 rc/post 后缀。
_COMPONENT_SPECS: dict[str, tuple[list[str], str]] = {
    "vllm": (["vllm"], r"0\.\d+\.\d+"),
    "vllm-ascend": (["vllm-ascend", "vllm_ascend", "vllm ascend"], r"0\.\d+\.\d+"),
    "cann": (["cann"], r"\d+\.\d+(?:\.\d+)?(?:\.?rc\d+)?"),
    "pytorch-ascend": (
        ["pytorch-ascend", "pytorch_ascend", "torch-npu", "torch_npu", "torchnpu"],
        r"\d+\.\d+(?:\.\d+)?(?:\.?post\d+)?(?:\.?rc\d+)?",
    ),
    "pytorch": (["pytorch", "torch"], r"\d+\.\d+(?:\.\d+)?"),
    "npu-driver": (["npu-driver", "npu driver", "driver"], r"\d+\.\d+(?:\.\d+)?(?:\.?rc\d+)?"),
}

# 预编译：每个组件取最长别名优先（如 "pytorch-ascend" 先于 "pytorch"/"torch"）
_COMPILED: dict[str, list[re.Pattern]] = {
    name: [
        re.compile(
            r"(?<![A-Za-z0-9_\-])"  # 避免匹配到别的词内部（如 pytorch 里的 torch）
            + re.escape(alias)
            + r"(?:\s+version\s*)?[\s:=_\-*`]*v?"  # 允许 "version:"、": "、"-"、"**:**" 等分隔
            + "(" + ver_pat + ")",  # 版本捕获组
            re.IGNORECASE,
        )
        for alias in sorted(aliases, key=len, reverse=True)
    ]
    for name, (aliases, ver_pat) in _COMPONENT_SPECS.items()
}


def extract_component_versions(text: str | None) -> dict[str, str]:
    """从正文/环境段提取各组件版本。返回 {规范组件名: 版本}，只取每个组件第一个命中。

    例：
      "**vLLM version**: 0.26.0"                    -> {"vllm": "0.26.0"}
      "vllm-ascend version: 0.18.0, vllm 0.12.1"    -> {"vllm-ascend": "0.18.0", "vllm": "0.12.1"}
      "CANN 8.1.RC2, torch 2.6.0, torch_npu 2.6.0.post1" -> {"cann": "8.1.RC2", "pytorch": "2.6.0", "pytorch-ascend": "2.6.0.post1"}
    """
    out: dict[str, str] = {}
    for name, patterns in _COMPILED.items():
        for pat in patterns:
            m = pat.search(text or "")
            if m:
                out[name] = m.group(1)
                break
    return out


def default_component_for_repo(repo: str) -> str:
    """根据 GitHub 仓库名推断组件（SourceCfg 未显式指定 component 时用）。"""
    slug = (repo or "").lower()
    if "vllm-ascend" in slug:
        return "vllm-ascend"
    if "pytorch-ascend" in slug or "torch-npu" in slug or "torch_npu" in slug:
        return "pytorch-ascend"
    if "cann" in slug:
        return "cann"
    if "pytorch" in slug:
        return "pytorch"
    if "vllm" in slug:
        return "vllm"
    return slug.replace("/", "-") or "doc"
