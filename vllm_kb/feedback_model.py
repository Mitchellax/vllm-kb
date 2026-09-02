"""后验置信度模型：行为遥测反馈 → Beta 后验统计量 → w_hist 历史可靠度。

数学模型（时间维度指数遗忘，非事件次数）：
    回流时按上次更新时间差衰减后加新事件：
        a <- a × 2^(-Δt / HL) + w × hit
        b <- b × 2^(-Δt / HL) + w × miss
        n_unknown <- n_unknown × 2^(-Δt / HL) + 1   （unknown 不进 a/b，单独计数）
        last_update_ts <- now
    - Δt = now - last_update_ts（天）
    - HL = config.confidence.feedback_half_life_days（默认 365，复用半衰期语义）
    - w = 反馈权重（弱 0.3 / 中 0.5，上限 0.5——自证循环阻尼）

后验统计量：
    hit_w = a, miss_w = b
    n_eff = hit_w + miss_w         （加权观察，非次数；unknown 不进 n_eff）
    mean = hit_w / (hit_w + miss_w)  （n_eff=0 时不存在）
    sd = sqrt(hit_w × miss_w / ((hit_w+miss_w)^2 × (hit_w+miss_w+1)))
    lb = mean - z × sd             （z=1.0，单侧置信下界；冷启动保护强度，n_eff 大时 sd→0）

w_hist 三段式（当期值不缓存，每次查询重算）：
    n_eff = 0     → w_hist = 1.0（中性，不用 seed 套 lb 公式——避免误杀新文档）
    0 < n_eff < n_min → w_hist = lb（正常算，但 flag=accumulating）
    n_eff ≥ n_min    → w_hist = lb（flag=supported/used_but_unconfirmed/evidence_thin/failing）

seed 初始值 a=1.0, b=1.0（弱先验）：随遗忘一起衰减，HL 决定多久失效。seed 是初期担保
非永久信用；带遗忘的稳态 n_eff 约为 HL 窗口内加权事件数，有天然上限不会无限增长。
seed 强度 1+1=2，数据部分需 ≥3 超先验 1.5 倍才主导后验。隐含前提：平均确认频率约
每季度 1 条；冷门 domain 检索频率过低时 supported 状态不会出现（符合设计——冷门且无
反馈证据的文档保持中性，不被误杀）。

与现有置信度的关系（正交，不乘进 w_rel——保护审计链）：
    conf = w_time × (α·w_ver + β·w_rel)     # 不变，确定性可审计
    final = sim^γ × conf^(1-γ) × lb^σ       # 新增 lb^σ 乘性
    - 乘性保证：sim=0 时 final=0（无匹配不被历史翻盘）；lb=0 时 final=0（历史全负不被相似度翻盘）
    - z（检索侧排序）vs p_min（消费侧决策）分工：vllm-kb 只输出 lb，不参与消费侧决断
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import ConfidenceCfg


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


@dataclass
class DocFeedback:
    """单个 doc 的后验反馈状态（持久化在 confidence_feedback.json）。"""
    a: float  # 加权命中累计（含 seed 衰减）
    b: float  # 加权未命中累计
    n_unknown: float  # unknown 单独计数（不进 a/b/n_eff）
    last_update_ts: str  # ISO，上次更新时间（用于指数遗忘）


@dataclass
class PosteriorStats:
    """后验统计量 + w_hist + flag + sigma（当期值，每次查询重算，不缓存）。"""
    n_eff: float
    mean: Optional[float]  # n_eff=0 时 None
    sd: Optional[float]
    lb: float  # 后验下界 mean - z·sd（n_eff=0 时 1.0）
    w_hist: float  # 历史可靠度（= lb，n_eff=0 时 1.0）
    n_unknown: float
    history_flag: str  # new/accumulating/supported/used_but_unconfirmed/evidence_thin/failing
    sigma: float  # lb 在 final 中的指数权重（n_eff 线性插值 sigma_min→history_sigma，避免 n_min 边界跳变）


def compute_posterior(fb: Optional[DocFeedback], cfg: ConfidenceCfg) -> PosteriorStats:
    """从 DocFeedback 算后验统计量 + w_hist + flag + sigma（当期值不缓存）。

    fb=None（无反馈记录的新文档）→ n_eff=0, w_hist=1.0, flag=new, sigma 不参与。
    sigma 线性插值：n_eff 从 0→n_min 时 sigma 从 sigma_min→history_sigma，
    n_eff≥n_min 后固定 history_sigma。避免 n_min 边界 lb^σ 跳变。
    """
    if fb is None or (fb.a + fb.b) <= 0:
        return PosteriorStats(
            n_eff=0.0, mean=None, sd=None, lb=1.0, w_hist=1.0,
            n_unknown=fb.n_unknown if fb else 0.0, history_flag="new",
            sigma=cfg.history_sigma,  # n_eff=0 不参与（w_hist=1.0），值不影响
        )

    hit_w = fb.a
    miss_w = fb.b
    n_eff = hit_w + miss_w
    mean = hit_w / n_eff if n_eff > 0 else None
    # sd = sqrt(ab / ((a+b)^2 * (a+b+1)))
    sd = math.sqrt(
        (hit_w * miss_w) / ((n_eff ** 2) * (n_eff + 1))
    ) if n_eff > 0 else None

    z = cfg.feedback_z
    lb = (mean - z * sd) if (mean is not None and sd is not None) else 1.0
    lb = max(0.0, min(1.0, lb))  # 裁剪到 [0,1]

    # sigma 线性插值（n_eff 0→n_min: sigma_min→history_sigma；≥n_min: history_sigma）
    n_min = cfg.feedback_n_min
    if n_eff >= n_min:
        sigma = cfg.history_sigma
    elif n_min > 0:
        t = n_eff / n_min  # 0→1
        sigma = cfg.history_sigma_min + t * (cfg.history_sigma - cfg.history_sigma_min)
    else:
        sigma = cfg.history_sigma

    # flag 判定（当期值，按 n_eff/mean/n_unknown 综合）
    flag = _determine_flag(n_eff, mean, fb.n_unknown, cfg)

    return PosteriorStats(
        n_eff=n_eff, mean=mean, sd=sd, lb=lb, w_hist=lb,
        n_unknown=fb.n_unknown, history_flag=flag, sigma=sigma,
    )


def _determine_flag(n_eff: float, mean: Optional[float],
                    n_unknown: float, cfg: ConfidenceCfg) -> str:
    """history_flag 判定（当期值不缓存）。"""
    n_min = cfg.feedback_n_min
    if n_eff == 0:
        return "new"
    if n_eff < n_min:
        # 证据积累中：unknown 占比高时标 evidence_thin
        total = n_eff + n_unknown
        if total > 0 and n_unknown / total > 0.5:
            return "evidence_thin"
        return "accumulating"
    # n_eff >= n_min
    if mean is not None:
        if mean < 0.3:
            # 失败过多：可能过时/错误
            return "failing"
        # 高频使用但从未确认（全是 unknown 或偏负，mean 低但不到 failing）
        total = n_eff + n_unknown
        if total > 0 and n_unknown / total > 0.5 and mean < 0.5:
            return "used_but_unconfirmed"
    return "supported"


def apply_feedback_event(fb: Optional[DocFeedback], state: str, weight: float,
                         now: datetime, cfg: ConfidenceCfg) -> DocFeedback:
    """应用一条反馈事件到 DocFeedback（指数遗忘后加新事件）。

    state: hit / miss / unknown（unknown 不进 a/b，单独计 n_unknown）
    weight: 反馈权重（弱 0.3 / 中 0.5）
    now: 当前时间（用于算 Δt）
    """
    hl = cfg.feedback_half_life_days
    if fb is None:
        # 新文档：用 seed 初始化
        fb = DocFeedback(
            a=cfg.feedback_seed_success,
            b=cfg.feedback_seed_fail,
            n_unknown=0.0,
            last_update_ts=now.isoformat(),
        )

    # 时间衰减（按上次更新时间差）
    last = _parse_iso(fb.last_update_ts)
    if last is not None:
        dt_days = max(0.0, (now - last).total_seconds() / 86400.0)
        decay = 2.0 ** (-dt_days / hl)
    else:
        decay = 0.0  # 无上次时间：全部衰减掉（从 seed 重新开始语义）

    a = fb.a * decay
    b = fb.b * decay
    n_unknown = fb.n_unknown * decay

    # 加新事件
    if state == "hit":
        a += weight
    elif state == "miss":
        b += weight
    elif state == "unknown":
        n_unknown += 1.0  # unknown 不进 a/b，单独计；权重不影响 n_unknown（计数 1）

    return DocFeedback(a=a, b=b, n_unknown=n_unknown, last_update_ts=now.isoformat())


# ---------------- 反馈表加载/保存 ----------------

def load_feedback_table(path: Optional[Path]) -> dict[str, DocFeedback]:
    """加载 confidence_feedback.json → {doc_id: DocFeedback}。

    文件不存在/损坏时返回空 dict（新文档无反馈记录，w_hist=1.0 中性）。
    """
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, DocFeedback] = {}
    for doc_id, fb in data.items():
        if not isinstance(fb, dict):
            continue
        try:
            out[doc_id] = DocFeedback(
                a=float(fb.get("a", 1.0)),
                b=float(fb.get("b", 1.0)),
                n_unknown=float(fb.get("n_unknown", 0.0)),
                last_update_ts=fb.get("last_update_ts", ""),
            )
        except (TypeError, ValueError):
            continue
    return out


def save_feedback_table(path: Path, table: dict[str, DocFeedback]) -> None:
    """保存反馈表到 confidence_feedback.json。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        doc_id: {
            "a": fb.a, "b": fb.b, "n_unknown": fb.n_unknown,
            "last_update_ts": fb.last_update_ts,
        }
        for doc_id, fb in table.items()
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
