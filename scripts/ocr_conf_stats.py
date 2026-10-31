"""OCR 置信度分布汇总——logprobs 置信度的阈值校准依据。

**背景**（2026-10 业务实测）：paddleocr-vl 走 logprobs 置信度后全部图被判低置信——
`exp(mean(logprob))` 逐 token 均值与模型自报置信度（整体准确度主观分，普遍 ~0.9）是
**不同分布的度量**：OCR 逐字生成时清晰文本的单字概率也就 0.7~0.95，长文本一平均分数
整体下移。`ocr_min_confidence` 默认 0.6 是按自报校准的，对 logprobs 均值过高。

**用法**（阈值校准三步）：
  python scripts/ocr_conf_stats.py          # 看分布：分位数/直方图/长度交叉
  # 阈值定在"清晰图分数下界"附近（通常 p10~p25 之间，看直方图的清晰/模糊分界谷）
  # config 里调 image 源 ocr_min_confidence（不进引擎指纹，改阈值不重跑 OCR，只重判定）

排序性前提：均值聚合的"清晰图分数 > 模糊图分数"排序通常成立——若分布出现清晰/模糊
**不**分层（清晰图与模糊图分数区间重叠严重），说明聚合算法有缺陷，回报分布形状再议。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUCKETS = [(i / 20, (i + 1) / 20) for i in range(20)]   # 0.05 步长直方图


def percentile(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数（q∈[0,1]；空列表返回 nan 由调用方避免）。"""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac


def summarize(records: list[dict]) -> dict:
    """records: [{confidence, source, anomaly, text_len}] → 汇总（可单测的纯函数）。"""
    valid = [r for r in records if r.get("confidence") is not None]
    out: dict = {"n": len(records), "no_confidence": len(records) - len(valid)}
    if not valid:
        return out
    vals = sorted(float(r["confidence"]) for r in valid)
    out["min"] = round(vals[0], 4)
    out["max"] = round(vals[-1], 4)
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        out[f"p{int(q * 100)}"] = round(percentile(vals, q), 4)
    out["mean"] = round(statistics.fmean(vals), 4)
    hist = {f"{lo:.2f}-{hi:.2f}": 0 for lo, hi in BUCKETS}
    for v in vals:
        idx = min(int(v * 20), 19)
        hist[f"{BUCKETS[idx][0]:.2f}-{BUCKETS[idx][1]:.2f}"] += 1
    out["histogram"] = hist
    for th in (0.4, 0.5, 0.6, 0.7, 0.8):
        out[f"share_below_{th}"] = round(sum(1 for v in vals if v < th) / len(vals), 4)
    # 文本长度交叉：短文本（<30 字符，多为标题/单行）vs 长文本的中位分——长度效应直观化
    short = sorted(float(r["confidence"]) for r in valid if (r.get("text_len") or 0) < 30)
    long = sorted(float(r["confidence"]) for r in valid if (r.get("text_len") or 0) >= 30)
    if short:
        out["median_len_lt30"] = round(percentile(short, 0.5), 4)
    if long:
        out["median_len_ge30"] = round(percentile(long, 0.5), 4)
    return out


def main() -> int:
    from vllm_kb.config import AppConfig

    cfg = AppConfig.load("config.json", require_keys=False)
    images = cfg.resolve("data/parsed/images")
    records = []
    if images.exists():
        for art in sorted(images.glob("*.ocr.json")):
            try:
                d = json.loads(art.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            records.append({
                "confidence": d.get("confidence"),
                "source": d.get("confidence_source", "?"),
                "anomaly": d.get("anomaly", ""),
                "text_len": len(str(d.get("text", "") or "")),
            })
    if not records:
        print(f"[stats] {images} 下没有 ocr.json（先跑一轮图片 OCR）")
        return 1
    by_source: dict[str, list[dict]] = {}
    for r in records:
        by_source.setdefault(str(r.get("source")), []).append(r)
    print(f"[stats] 共 {len(records)} 张（{images}）\n")
    for src, grp in sorted(by_source.items()):
        s = summarize(grp)
        print(f"== confidence_source={src}（{s['n']} 张，无置信度 {s.get('no_confidence', 0)}）==")
        for k in ("min", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "max", "mean",
                  "median_len_lt30", "median_len_ge30"):
            if k in s:
                print(f"  {k:16} {s[k]}")
        for th in (0.4, 0.5, 0.6, 0.7, 0.8):
            if f"share_below_{th}" in s:
                print(f"  <{th} 占比        {s[f'share_below_{th}']:.1%}")
        print("  直方图（0.05 步长）：")
        for k, v in s.get("histogram", {}).items():
            if v:
                bar = "#" * max(1, round(v / max(s["n"] - s.get("no_confidence", 0), 1) * 50))
                print(f"    {k}  {v:5}  {bar}")
        print()
    print("[stats] 阈值校准：把 image 源 ocr_min_confidence 定在清晰图分数下界附近"
          "（通常 p10~p25 之间；改阈值不进指纹，不重跑 OCR 只重判定 text_included/"
          "review_reason）。若清晰图与模糊图分数区间重叠严重（排序失效），回报分布形状。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
