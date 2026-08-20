"""报错文本结构化解析：解析**结构**而非匹配内容，天然兼容报错形态变化。

覆盖：
1. Python 堆栈帧：`File "...", line N, in func_name` —— 提取函数名（结构化）；
2. 键值对：`kernel_name=xxx`、`errorStr: xxx`、`detail: xxx`、`drvRetCode=6` ——
   键固定、值动态，用键解析 + 值查符号表/信号词表；
3. 包/模块路径：`vllm_ascend/worker/xxx.py`、`vllm/v1/...` —— 提取模块路径段；
4. 错误类型行：`RuntimeError: ...`、`ValueError: ...`、`AssertionError: ...` —— 提取异常类型。

与 symbol_table 配合：解析出的"值"（函数名/模块段/键值）去符号表查，而非硬编码。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------- 结构化规则（不是内容正则，是结构解析） ----------------

# Python 堆栈帧
_STACK_FRAME_RE = re.compile(r'File\s+"([^"]+)",\s*line\s+(\d+)(?:,\s*in\s+([A-Za-z_][A-Za-z0-9_]*))?')
# 异常类型行
_EXC_TYPE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Failure))\s*:", re.M)
# 键值对（键固定：kernel_name/kName/errorStr/errStr/detail/retCode/drvRetCode/errCode/error code）
_KV_RE = re.compile(
    r"\b(kernel_name|kName|kname|errorStr|errStr|detail|retCode|drvRetCode|errCode|err_code|error_code|"
    r"error code|errno|error)\s*[=:]\s*[\"']?([^\"',;\n]{2,80})"
)
# 模块路径段（vllm_ascend/...、vllm/...、csrc/...）
_MODULE_PATH_RE = re.compile(r"\b((?:vllm_ascend|vllm|csrc|torch_npu|torch)/[A-Za-z0-9_./\-]+)")
# ACL 错误码：E 开头 + 数字，或 5/6 位裸数字
_ACL_CODE_RE = re.compile(r"\b(E\d{4,6}|10[7-9]\d{3}|50\d{4}|56\d{4}|0x[0-9a-fA-F]{5,8})\b")
# HDK/CANN 版本形态
_VER_RE = re.compile(
    r"\b(CANN\s*\d+\.\d+(?:\.\d+)?|HDK\s*\d+\.\d+(?:\.\d+)?|torch-?npu\s*\d+\.\d+(?:\.\d+)?|"
    r"vllm[-_]ascend\s*v?\d+\.\d+(?:\.\d+)?(?:rc\d+)?|vLLM\s*v?\d+\.\d+(?:\.\d+)?)\b",
    re.I,
)


@dataclass
class ParsedToken:
    text: str        # 解析出的值（归一）
    kind: str        # stack_func | stack_file | exc_type | kv | module | acl_code | version
    weight: float = 1.0
    origin: str = ""  # 解析来源（如 "stack frame"、"kernel_name="）


def parse_error_text(text: str) -> list[ParsedToken]:
    """结构化解析报错文本，返回解析出的 token 列表。"""
    tokens: list[ParsedToken] = []
    if not text:
        return tokens

    # 1) Python 堆栈帧：函数名优先（优先级高），文件路径次之
    for m in _STACK_FRAME_RE.finditer(text):
        fname = m.group(3)
        if fname:
            tokens.append(ParsedToken(text=fname, kind="stack_func", weight=2.5, origin="stack"))
        fpath = m.group(1)
        # 只取 vllm_ascend/... 的模块路径段作为候选
        mm = re.search(r"(vllm_ascend|vllm)/[A-Za-z0-9_./\-]+\.py", fpath)
        if mm:
            tokens.append(ParsedToken(text=mm.group(0), kind="stack_file", weight=1.5, origin="stack"))

    # 2) 异常类型行
    for m in _EXC_TYPE_RE.finditer(text):
        tokens.append(ParsedToken(text=m.group(1), kind="exc_type", weight=1.2, origin="exc"))

    # 3) 键值对
    for m in _KV_RE.finditer(text):
        key = m.group(1).lower()
        val = m.group(2).strip()
        # 错误码键 → 高权；名称键 → 中权
        if key in ("drvretcode", "retcode", "errcode", "err_code", "error_code", "error code", "errno"):
            tokens.append(ParsedToken(text=val, kind="acl_code", weight=3.0, origin=f"{key}="))
        elif key in ("kernel_name", "kname", "errstr", "detail", "error"):
            tokens.append(ParsedToken(text=val, kind="kv", weight=2.0, origin=f"{key}="))

    # 4) 模块路径段（不在堆栈帧里的裸路径）
    for m in _MODULE_PATH_RE.finditer(text):
        tokens.append(ParsedToken(text=m.group(1), kind="module", weight=1.0, origin="module"))

    # 5) ACL 错误码
    for m in _ACL_CODE_RE.finditer(text):
        tokens.append(ParsedToken(text=m.group(1), kind="acl_code", weight=3.0, origin="acl"))

    # 6) HDK/CANN 版本
    for m in _VER_RE.finditer(text):
        tokens.append(ParsedToken(text=m.group(1), kind="version", weight=0.8, origin="version"))

    return tokens


def format_tokens(tokens: list[ParsedToken]) -> str:
    if not tokens:
        return "(结构解析无 token)"
    return "\n".join(
        f"  [{t.kind}] {t.text}  (w={t.weight:.1f}{f', {t.origin}' if t.origin else ''})"
        for t in tokens
    )
