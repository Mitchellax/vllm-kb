"""Embedding 客户端：统一接口，支持多 provider。

- openai_compatible：任何 OpenAI 兼容 /embeddings 端点（OpenAI、SiliconFlow、Jina、Voyage、DeepSeek 等）；
- echo：确定性 n-gram 哈希向量（离线开发/自测用，无需网络和 key；语义相似度粗糙但可跑通全链路）。
"""
from __future__ import annotations

import hashlib
import math
import time
from typing import Optional

import requests

from .config import EmbeddingCfg


class EmbeddingError(RuntimeError):
    pass


class EmbeddingClient:
    def __init__(self, cfg: EmbeddingCfg):
        self.cfg = cfg
        if cfg.provider not in ("openai_compatible", "echo"):
            raise ValueError(f"不支持的 embedding provider: {cfg.provider}")
        if cfg.provider == "openai_compatible" and not cfg.base_url:
            raise ValueError("embedding.base_url 为空")

    def embed(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        bs = max(1, self.cfg.batch_size)
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            if self.cfg.provider == "echo":
                out.extend(self._embed_echo_batch(batch))
            else:
                out.extend(self._embed_api_batch(batch))
        return out

    # ---------------- OpenAI 兼容 ----------------

    def _embed_api_batch(self, batch: list[str]) -> list[list[float]]:
        inputs = [t[: self.cfg.max_input_chars] for t in batch]
        url = self.cfg.base_url.rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {self.cfg.effective_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.cfg.model, "input": inputs}
        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.cfg.timeout_seconds)
            except requests.RequestException as e:
                last_exc = e
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                data = r.json().get("data") or []
                data.sort(key=lambda d: d.get("index", 0))
                return [d["embedding"] for d in data]
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt * 3)
                continue
            raise EmbeddingError(f"embedding API {r.status_code}: {r.text[:300]}")
        raise EmbeddingError(f"embedding API 重试失败: {last_exc}")

    # ---------------- echo（离线自测） ----------------

    def _embed_echo_batch(self, batch: list[str]) -> list[list[float]]:
        dim = max(8, self.cfg.dimensions)
        return [self._echo_vec(t, dim) for t in batch]

    @staticmethod
    def _echo_vec(text: str, dim: int) -> list[float]:
        """字符 n-gram + 词级 1/2-gram 哈希 -> 稀疏近似语义向量，L2 归一化。

        用 hashlib（而非内置 hash()）保证跨进程/跨运行确定性，持久化后可复用。
        词级 2-gram 让短语级重合（如 "illegal memory access"）显著拉高相似度，
        提升离线演示/自测的区分度。
        """
        import re as _re

        vec = [0.0] * dim
        norm = text.lower()
        for n in (2, 3, 4):
            for i in range(max(0, len(norm) - n + 1)):
                gram = norm[i : i + n]
                h = int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:8], "big")
                vec[h % dim] += 1.0
        words = _re.findall(r"[a-z0-9]+", norm)
        for w in words:  # 词 1-gram（权重 2）
            h = int.from_bytes(hashlib.md5(("w:" + w).encode("utf-8")).digest()[:8], "big")
            vec[h % dim] += 2.0
        for i in range(len(words) - 1):  # 词 2-gram（权重 3，短语信号）
            gram = words[i] + " " + words[i + 1]
            h = int.from_bytes(hashlib.md5(("b:" + gram).encode("utf-8")).digest()[:8], "big")
            vec[h % dim] += 3.0
        mag = math.sqrt(sum(v * v for v in vec))
        if mag == 0:
            return vec
        return [v / mag for v in vec]
