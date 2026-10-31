"""OCR 基准集评估工具（模糊小字/歪斜/透视等难场景优化的量化依据）。

**为什么需要**（2026-10 业务实测教训）：预处理手段（x3 放大 / CLAHE）"感觉有效"
的判断不可靠——实测无提升且引入乱码。任何 OCR 优化（调参/矫正/换模型）先过基准集
量化收益，再上生产。

基准集结构（data_root/ocr_bench/，不进 git）：

    ocr_bench/
      images/            # 基准图片（normal_/blurry_/tilt_/persp_ 前缀=场景）
      labels.json        # 标注：期望文本 + 场景 + （派生图）变换参数
      report.json        # 评估报告（脚本产出；--baseline 对比历史）

选图与标注（30 张配比，标注成本约 1 小时）：
  - normal ×8 / blurry ×8：从**业务库真实截图**挑——`--pick-from-assets` 按 ocr.json
    缓存置信度分桶自动拷贝 + **labels 草稿预填 OCR 文本**（人工只需校对，不用手打）；
  - tilt ×7 / persp ×7：**程序变换生成**（--gen-transforms 对清晰图 rotate/透视）——
    期望文本=原图，零标注成本、角度参数可控（3/7/12°）；
  - blurry 的期望文本同样预填自高置信 OCR + 人工校对。

用法：
  python scripts/eval_ocr_bench.py --pick-from-assets                # ① 挑图+草稿（默认资产目录）
  python scripts/eval_ocr_bench.py --gen-transforms                  # ② 生成歪斜/透视
  # ③ 人工校对 labels.json 的 expect（重点：blurry 桶）
  python scripts/eval_ocr_bench.py --run                             # ④ 跑评估出报告
  python scripts/eval_ocr_bench.py --run --baseline report.json      # ⑤ 改参后对比
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_kb.ocr import norm_text, score_pair  # noqa: E402  去空白后 SequenceMatcher ratio

PASS_SCORE = 0.95          # 字符准确率通过线
SCENES = ("normal", "blurry", "tilt", "persp")


# ---------------- 纯评估函数（可单测；公共实现已上移 vllm_kb.ocr，此处别名兼容） ----------------

from vllm_kb.ocr import norm_text, score_pair  # noqa: E402  去空白后 SequenceMatcher ratio


def summarize(rows: list[dict]) -> dict:
    """分场景汇总：n / avg_score / pass_rate / avg_confidence（无置信度计入 None）。"""
    out: dict = {}
    for scene in (*SCENES, "overall"):
        grp = rows if scene == "overall" else [r for r in rows if r.get("scene") == scene]
        if not grp:
            continue
        scores = [r["score"] for r in grp]
        confs = [r["confidence"] for r in grp if r.get("confidence") is not None]
        out[scene] = {
            "n": len(grp),
            "avg_score": round(sum(scores) / len(scores), 4),
            "pass_rate": round(sum(1 for s in scores if s >= PASS_SCORE) / len(grp), 4),
            "avg_confidence": (round(sum(confs) / len(confs), 4) if confs else None),
            "anomaly_n": sum(1 for r in grp if r.get("anomaly")),
        }
    return out


# ---------------- 子命令实现 ----------------

def _load_labels(bench: Path) -> dict:
    p = bench / "labels.json"
    if not p.exists():
        raise SystemExit(f"[bench] 标注文件不存在：{p}（先 --pick-from-assets 生成草稿）")
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_run(cfg, bench: Path, scene: str | None, baseline: Path | None) -> dict:
    from vllm_kb.ocr import _OCR_PROMPT_VERSION, ocr_config_from_cfg, ocr_image_detail

    labels = _load_labels(bench)
    items = [it for it in labels.get("items", [])
             if isinstance(it, dict) and it.get("image") and (scene is None or it.get("scene") == scene)]
    if not items:
        raise SystemExit("[bench] labels.json 无可评估项（scene 过滤太严或文件为空）")
    oc = ocr_config_from_cfg(cfg)
    if oc is None or oc.provider != "api" or oc.mode != "openai":
        raise SystemExit("[bench] 需要 image 源 openai 模式 OCR 配置（ocr_provider: api + ocr_api_mode: openai）")
    rows = []
    for it in items:
        img = bench / "images" / it["image"]
        if not img.exists():
            print(f"[bench] 跳过（图片缺失）：{it['image']}")
            continue
        r = ocr_image_detail(img, "api", api_base=oc.api_base, api_key=oc.api_key,
                             model=oc.model, mode="openai", timeout=oc.timeout,
                             max_tokens=oc.max_tokens, temperature=oc.temperature,
                             prompt=oc.prompt)
        score = score_pair(str(it.get("expect", "")), r.text or "")
        rows.append({
            "image": it["image"], "scene": it.get("scene", ""),
            "score": round(score, 4),
            "pass": bool(score >= PASS_SCORE and not r.anomaly),
            "confidence": (round(r.confidence, 4) if r.confidence is not None else None),
            "confidence_source": r.confidence_source,
            "anomaly": r.anomaly,
            "main_text": r.text or "",   # 主引擎全文（--cross 交叉比对 + 人工核对用）
        })
        flag = "PASS" if rows[-1]["pass"] else "FAIL"
        print(f"[bench] {it['image']:24} [{flag}] score={score:.3f} "
              f"conf={rows[-1]['confidence']} anomaly={r.anomaly or '-'}")
        if score < PASS_SCORE and it.get("expect"):
            print(f"         expect: {norm_text(str(it['expect']))[:80]}")
            print(f"         actual: {norm_text(r.text or '')[:80]}")
    summary = summarize(rows)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": {"provider": "api", "mode": "openai", "model": oc.model,
                   "prompt": oc.prompt, "temperature": oc.temperature,
                   "max_tokens": oc.max_tokens, "prompt_version": _OCR_PROMPT_VERSION},
        "summary": summary, "items": rows,
    }
    out = bench / "report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ov = summary.get("overall", {})
    print(f"\n[bench] 总体：{ov.get('n', 0)} 张 · avg_score={ov.get('avg_score')} · "
          f"pass_rate={ov.get('pass_rate')} · avg_conf={ov.get('avg_confidence')}")
    for s in SCENES:
        if s in summary:
            print(f"  {s:8} n={summary[s]['n']:2} avg={summary[s]['avg_score']} "
                  f"pass={summary[s]['pass_rate']} conf={summary[s]['avg_confidence']}")
    print(f"[bench] 报告已写 {out}")
    if baseline is not None:
        old = json.loads(Path(baseline).read_text(encoding="utf-8"))
        print("\n[bench] 与基线对比（avg_score 变化）：")
        for s, cur in summary.items():
            prev = (old.get("summary") or {}).get(s) or {}
            if prev.get("avg_score") is not None:
                delta = round(cur["avg_score"] - prev["avg_score"], 4)
                print(f"  {s:8} {prev['avg_score']} -> {cur['avg_score']}（{delta:+}）")
    return report


def cmd_gen_transforms(bench: Path) -> int:
    """对 labels 里 scene=tilt/persp 且带 derived_from 的项，程序生成变体图（Pillow）。

    期望文本 = 原图（零标注成本）；角度/位移参数取 item.param（缺省 3/7/12°）。
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit(f"[bench] 生成变换需要 Pillow：pip install pillow（{e}）")
    labels = _load_labels(bench)
    n = 0
    for it in labels.get("items", []):
        if it.get("scene") not in ("tilt", "persp") or not it.get("derived_from"):
            continue
        out = bench / "images" / it["image"]
        if out.exists():
            continue
        src = bench / "images" / it["derived_from"]
        if not src.exists():
            print(f"[bench] 源图缺失，跳过：{it['derived_from']}")
            continue
        img = Image.open(src).convert("RGB")
        param = it.get("param") or {}
        if it["scene"] == "tilt":
            angle = float(param.get("rotate", 7))
            # 旋转后放大画布避免裁切，白底填充（模拟拍照边界）
            big = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
            big.save(out)
        else:  # persp：四角向内位移（模拟斜拍）；coefficients = 目标四角
            w, h = img.size
            kx, ky = float(param.get("kx", 0.12)), float(param.get("ky", 0.08))
            coeffs = _find_coeffs(
                [(w * kx, h * ky), (w * (1 - kx * 0.6), h * ky * 0.5),
                 (w * (1 - kx), h * (1 - ky * 0.4)), (w * kx * 0.4, h * (1 - ky))],
                [(0, 0), (w, 0), (w, h), (0, h)])
            img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC).save(out)
        n += 1
        print(f"[bench] 生成 {it['image']}（{it['scene']}，param={param}）")
    print(f"[bench] 共生成 {n} 张变体")
    return n


def _find_coeffs(target: list[tuple], source: list[tuple]) -> tuple:
    """PIL PERSPECTIVE 的 8 系数求解（source 四角 → target 四角）。"""
    import numpy as np

    A = []
    for (x, y), (X, Y) in zip(source, target):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    b = [c for pt in target for c in pt]
    h = np.linalg.lstsq(np.array(A, dtype=float), np.array(b, dtype=float), rcond=None)[0]
    return tuple(h)


def cmd_pick_from_assets(assets_dir: Path, bench: Path, count: int) -> int:
    """从库内资产挑基准图：按 ocr.json 缓存置信度**分桶**（高/低置信各 count/2 张），
    拷贝进 images/ 并生成 labels 草稿——expect 预填 OCR 文本，人工只需校对。"""
    assets = Path(assets_dir)
    if not assets.exists():
        raise SystemExit(f"[bench] 资产目录不存在：{assets}")
    cached: list[tuple[float, Path, dict]] = []
    for art in assets.glob("*.ocr.json"):
        try:
            d = json.loads(art.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        conf = d.get("confidence")
        img = assets / (art.stem[: -len(".ocr")] if art.stem.endswith(".ocr") else art.stem)
        img = img.with_suffix(".png") if not img.exists() else img
        if conf is None or not img.exists():
            continue
        cached.append((float(conf), img, d))
    if not cached:
        raise SystemExit("[bench] 资产目录没有 ocr.json 缓存（先跑过一轮图片 OCR）")
    cached.sort(key=lambda t: -t[0])
    half = max(count // 2, 1)
    picked = [(c, p, d, "normal") for c, p, d in cached[:half]] + \
             [(c, p, d, "blurry") for c, p, d in cached[-half:]]
    imgs = bench / "images"
    imgs.mkdir(parents=True, exist_ok=True)
    items = []
    for i, (conf, img, d, scene) in enumerate(picked, 1):
        name = f"{scene}_{i:02d}.png"
        shutil.copy2(img, imgs / name)
        items.append({"image": name, "scene": scene,
                      "expect": str(d.get("text", "") or ""),   # 预填，人工校对
                      "source_conf": round(conf, 4), "note": "expect=OCR 预填，需人工校对"})
    labels = {"version": 1, "items": items}
    (bench / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"[bench] 挑选 {len(items)} 张（normal=高置信 / blurry=低置信，conf 见 source_conf），"
          f"labels 草稿已写 {bench / 'labels.json'}——**人工校对 expect 后再 --run**")
    return len(items)


def cmd_cross(bench: Path, threshold: float) -> int:
    """双引擎交叉验证：主引擎（API VL，--run 的 report.json）vs 本地 PP-OCRv5。

    **为什么**（2026-10 业务实测教训）：VL 端到端模型对小字会"脑补"、对干扰内容
    （导航栏/侧栏）选择性输出——本地 PP-OCRv5（det+rec 两段式）是独立第二意见。
    一致度 = score_pair(pp_text, main_text)；低于 threshold 的样本列出并进人工核对。
    依赖（可选）：pip install paddleocr paddlepaddle（CPU 可跑，首次自动拉模型权重）。
    """
    report_path = bench / "report.json"
    if not report_path.exists():
        raise SystemExit(f"[bench] 先 --run 生成主引擎报告：{report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        hint = str(e)
        extra = ""
        if "DLL load failed" in hint:
            extra = ("（Windows：缺 VC++ 运行库——装 Microsoft Visual C++ Redistributable "
                     "2015-2022 x64 后重试；业务环境/服务器一般已装）")
        raise SystemExit(f"[bench] 交叉验证需要本地 PP-OCRv5：pip install paddleocr "
                         f"paddlepaddle{extra}（{e}）")
    print("[bench] 初始化 PP-OCRv5（首次运行会自动下载 det/rec 模型权重）…", flush=True)
    # enable_mkldnn=False：paddle 3.3.1 CPU 的 oneDNN+PIR 对 PP-OCR 模型有
    # ConvertPirAttribute2RuntimeAttribute 兼容 bug（NotImplementedError），纯 CPU 后端可跑
    try:  # 3.x 新 API
        engine = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                           use_textline_orientation=True, lang="ch", enable_mkldnn=False)
    except TypeError:  # 2.x 兼容
        engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

    def pp_text(img: Path) -> str:
        if hasattr(engine, "predict"):  # 3.x
            lines = []
            for res in engine.predict(str(img)):
                d = res.json.get("res", res.json) if hasattr(res, "json") else {}
                lines.extend(d.get("rec_texts") or [])
            return "\n".join(lines)
        out = engine.ocr(str(img), cls=True)  # 2.x
        lines = []
        for page in (out or []):
            for line in (page or []):
                if line and len(line) >= 2 and isinstance(line[1], (list, tuple)):
                    lines.append(str(line[1][0]))
        return "\n".join(lines)

    rows = report.get("items") or []
    if not rows:
        raise SystemExit("[bench] report.json 无 items")
    n_agree = 0
    for it in rows:
        img = bench / "images" / str(it.get("image", ""))
        if not img.exists():
            continue
        txt = pp_text(img)
        agree = score_pair(str(it.get("main_text", "")), txt)
        it["cross_ppocr_score"] = round(agree, 4)
        it["cross_ppocr_text_len"] = len(norm_text(txt))
        flag = "AGREE" if agree >= threshold else "DIVERGE"
        n_agree += agree >= threshold
        print(f"[bench] {it.get('image'):24} [{flag}] 一致度={agree:.3f} "
              f"ppocr字数={len(norm_text(txt))} 主引擎字数={len(norm_text(str(it.get('main_text', ''))))}")
        if agree < threshold:
            print(f"         ppocr头: {norm_text(txt)[:100]}")
    report["cross_check"] = {"engine": "PP-OCRv5(local)", "threshold": threshold,
                             "agree_rate": round(n_agree / len(rows), 4) if rows else None}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[bench] 交叉一致率 {n_agree}/{len(rows)}，已写回 {report_path}——"
          f"DIVERGE 样本进人工核对（双引擎分歧=VL'脑补'或 PP 漏识的高发信号）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true", help="跑评估，写 report.json")
    g.add_argument("--gen-transforms", action="store_true", help="生成 tilt/persp 变体图")
    g.add_argument("--pick-from-assets", metavar="DIR",
                   help="从资产目录按置信度分桶挑图 + labels 草稿（默认 data/assets/images）")
    g.add_argument("--cross", action="store_true",
                   help="PP-OCRv5 本地交叉验证（依赖 paddleocr，先 --run）")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="交叉一致度阈值（低于=DIVERGE 进人工，默认 0.85）")
    ap.add_argument("--bench", default=None, help="基准集目录（默认 <data_root>/ocr_bench）")
    ap.add_argument("--scene", choices=SCENES, help="只评估指定场景")
    ap.add_argument("--baseline", metavar="REPORT", help="对比基线报告（--run 时）")
    ap.add_argument("--count", type=int, default=16, help="pick-from-assets 挑选张数（默认 16）")
    args = ap.parse_args()

    from vllm_kb.config import AppConfig

    if args.cross:      # 交叉验证只用 report.json + 本地 PP-OCRv5，不需要业务 config
        return cmd_cross(Path(args.bench) if args.bench else Path("data/ocr_bench"),
                         args.threshold)
    cfg = AppConfig.load(args.config, require_keys=False)
    bench = Path(args.bench) if args.bench else cfg.resolve("data/ocr_bench")
    if args.pick_from_assets:
        assets = Path(args.pick_from_assets) if args.pick_from_assets.strip() != "-" \
            else cfg.resolve("data/assets/images")
        return cmd_pick_from_assets(assets, bench, args.count)
    if args.gen_transforms:
        cmd_gen_transforms(bench)
        return 0
    cmd_run(cfg, bench, args.scene,
            Path(args.baseline) if args.baseline else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
