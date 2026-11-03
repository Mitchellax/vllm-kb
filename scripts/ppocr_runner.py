"""PP-OCR 第二意见 runner（**独立 venv 隔离运行**）。

背景：业务环境主 venv 是 vllm-ascend 共享环境（opencv 被其依赖链引用，装/换 opencv
会破坏 vllm-ascend）——paddle 全家桶放**独立 venv**，主进程经子进程长驻管道调用：
    ppvenv/bin/python -X utf8 scripts/ppocr_runner.py --serve
主进程（vllm_kb.ocr.PpocrSubprocess）逐图写路径、读一行 JSON——paddle 只在 PP venv
初始化一次，主 venv **零 paddle 依赖**。

协议（--serve）：
  stdin  每行一个图片路径；空行/EOF 退出；
  stdout 每行一个 JSON：{"text": "..."} 或 {"text": "", "error": "..."}。

单图模式（调试）：python ppocr_runner.py <image> → stdout 纯文本。
"""
import json
import sys


def _new_engine():
    from paddleocr import PaddleOCR

    try:  # 3.x 新 API
        return PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                         use_textline_orientation=True, lang="ch", enable_mkldnn=False)
    except TypeError:  # 2.x 兼容
        return PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)


def _ocr_text(engine, img_path: str) -> str:
    if hasattr(engine, "predict"):  # 3.x
        lines = []
        for res in engine.predict(img_path):
            d = res.json.get("res", res.json) if hasattr(res, "json") else {}
            lines.extend(d.get("rec_texts") or [])
        return "\n".join(lines)
    out = engine.ocr(img_path, cls=True)  # 2.x
    lines = []
    for page in (out or []):
        for line in (page or []):
            if line and len(line) >= 2 and isinstance(line[1], (list, tuple)):
                lines.append(str(line[1][0]))
    return "\n".join(lines)


def main() -> int:
    args = [a for a in sys.argv[1:] if a and not a.startswith("-")]
    if "--serve" in sys.argv:
        engine = None
        for line in sys.stdin:
            path = line.strip()
            if not path:
                break
            try:
                if engine is None:  # 首图才初始化（paddle 冷启动 ~10s 只付一次）
                    real = sys.stdout
                    sys.stdout = sys.stderr  # paddle 初始化噪音（Fetching/Creating）不让它污染协议管道
                    try:
                        engine = _new_engine()
                    finally:
                        sys.stdout = real
                print(json.dumps({"text": _ocr_text(engine, path)}, ensure_ascii=False),
                      flush=True)
            except Exception as e:  # noqa: BLE001 —— 单图失败不杀服务
                print(json.dumps({"text": "", "error": f"{type(e).__name__}: {e}"},
                                 ensure_ascii=False), flush=True)
        return 0
    if args:
        try:
            sys.stdout.write(_ocr_text(_new_engine(), args[0]))
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"{type(e).__name__}: {e}", file=sys.stderr)
            return 1
    print("usage: ppocr_runner.py --serve | <image>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
