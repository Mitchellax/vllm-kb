"""离线演示：写入一批模拟 GitHub API 原始响应（issue/PR + 评论），无需网络。

用法：
    python scripts/seed_demo.py [--config config.offline.json]
    python scripts/build_kb.py --config config.offline.json --skip-pull
    python scripts/verify.py --config config.offline.json --version 0.6.1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig  # noqa: E402

# 模拟 GitHub issues 接口返回的原始 JSON（含 issue 模板里的 vLLM version，演示版本提取）
DEMO_ITEMS = [
    {
        "number": 10001,
        "title": "CUDA illegal memory access during inference",
        "body": (
            "When running vLLM inference with tensor parallel, I get "
            "CUDA error: an illegal memory access was encountered. "
            "It happens after a few steps of generation with llama 70b on 2 GPUs.\n\n"
            "### Your current environment\n- **vLLM version**: 0.5.4"
        ),
        "labels": [{"name": "bug"}],
        "state": "closed",
        "created_at": "2024-02-20T00:00:00Z",
        "closed_at": "2024-03-10T00:00:00Z",
        "html_url": "https://github.com/vllm-project/vllm/issues/10001",
        "user": {"login": "reporter1"},
    },
    {
        "number": 10002,
        "title": "Not enough memory to allocate paged attention blocks",
        "body": (
            "ValueError: Not enough memory to allocate paged attention blocks. "
            "The model requires 12GB but only 8GB is available. "
            "This happens when max_model_len is too large for the GPU.\n\n"
            "### Your current environment\n- **vLLM version**: 0.6.1"
        ),
        "labels": [{"name": "bug"}, {"name": "oom"}],
        "state": "closed",
        "created_at": "2024-04-01T00:00:00Z",
        "closed_at": "2024-04-10T00:00:00Z",
        "html_url": "https://github.com/vllm-project/vllm/issues/10002",
        "user": {"login": "reporter2"},
    },
    {
        "number": 10003,
        "title": "openai api server returns 500 when calling chat completions",
        "body": (
            "The OpenAI-compatible server returns 500 Internal Server Error "
            "intermittently when calling /v1/chat/completions with a long prompt. "
            "Log shows an exception in the scheduler loop.\n\n"
            "### Your current environment\n- **vLLM version**: 0.6.0"
        ),
        "labels": [{"name": "bug"}, {"name": "openai-server"}],
        "state": "open",
        "created_at": "2024-06-01T00:00:00Z",
        "closed_at": None,
        "html_url": "https://github.com/vllm-project/vllm/issues/10003",
        "user": {"login": "reporter3"},
    },
    {
        "number": 10004,
        "title": "ValueError: model max model len exceeds maximum number of tokens",
        "body": (
            "ValueError: The model's max model len (8192) is larger than the "
            "maximum number of tokens (4096). Set --max-model-len accordingly.\n\n"
            "### Your current environment\n- vLLM: v0.6.3"
        ),
        "labels": [{"name": "bug"}, {"name": "usage"}],
        "state": "closed",
        "created_at": "2024-07-15T00:00:00Z",
        "closed_at": "2024-07-20T00:00:00Z",
        "html_url": "https://github.com/vllm-project/vllm/issues/10004",
        "user": {"login": "reporter4"},
    },
    {
        "number": 10005,
        "title": "Feature request: support fp8 quantization for mixtral",
        "body": "It would be nice to support fp8 quantization for mixtral models.",
        "labels": [{"name": "feature-request"}],
        "state": "open",
        "created_at": "2024-05-01T00:00:00Z",
        "closed_at": None,
        "html_url": "https://github.com/vllm-project/vllm/issues/10005",
        "user": {"login": "reporter5"},
    },
    {
        "number": 20001,
        "title": "[Fix] Release paged attention blocks on OOM to allow retry",
        "body": "Fixes the OOM failure by releasing paged attention blocks before retrying allocation.",
        "labels": [{"name": "bugfix"}],
        "state": "closed",
        "created_at": "2024-04-11T00:00:00Z",
        "closed_at": "2024-04-12T00:00:00Z",
        "html_url": "https://github.com/vllm-project/vllm/pull/20001",
        "user": {"login": "dev1"},
        "pull_request": {
            "merged": True,
            "merged_at": "2024-04-12T00:00:00Z",
            "merge_commit_sha": "9f8e7d6c5b4a3210",
        },
    },
]

DEMO_COMMENTS = {
    10001: [
        {
            "user": {"login": "commenter"},
            "created_at": "2024-03-01T00:00:00Z",
            "body": "Try reducing max_num_seqs or disabling chunked prefill as a workaround.",
        }
    ],
    10002: [
        {
            "user": {"login": "maintainer"},
            "created_at": "2024-04-05T00:00:00Z",
            "body": "Reduce max_model_len or use gpu_memory_utilization=0.9.",
        }
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser(description="写入离线演示原始数据")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = AppConfig.load(args.config)
    raw = cfg.resolve(cfg.github.raw_dir)
    (raw / "issues").mkdir(parents=True, exist_ok=True)
    (raw / "prs").mkdir(parents=True, exist_ok=True)
    (raw / "comments").mkdir(parents=True, exist_ok=True)
    n_issues = n_prs = 0
    for item in DEMO_ITEMS:
        number = item["number"]
        kind = "prs" if "pull_request" in item else "issues"
        (raw / kind / f"{number}.json").write_text(
            json.dumps(item, ensure_ascii=False), encoding="utf-8"
        )
        if kind == "prs":
            n_prs += 1
        else:
            n_issues += 1
    for number, comments in DEMO_COMMENTS.items():
        (raw / "comments" / f"{number}.json").write_text(
            json.dumps(comments, ensure_ascii=False), encoding="utf-8"
        )
    print(f"[demo] 写入 {n_issues} 条 issue + {n_prs} 条 PR + 评论 -> {raw}")


if __name__ == "__main__":
    main()
