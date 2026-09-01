"""行为遥测采集：FastAPI logging middleware + 独立 telemetry.sqlite3。

全量记录查询行为（不推断），供离线 scripts/build_feedback.py 从会话序列推断三态反馈。
设计三段分离（审计可逆）：原始行为（本模块）→ 离线推断证据 → 查询期 w_hist 打分。

会话归属：X-Session-Id header（agent 侧管，经 VLLM_KB_SESSION 环境变量透传）；
缺失时回退 client_ip + 30min 时间窗聚类（粗粒度兜底，多 agent 同机/NAT 后可能冲突）。

只读姿态不受影响：只写 telemetry.sqlite3（mode=rw，独立库），不碰 kb.sqlite3。
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import AppConfig


# 记录行为的端点（辅助端点 /health /stats 不记）
_TRACKED_PREFIXES = ("/search", "/signature-search", "/title", "/doc/", "/code/search", "/code/diff")
# 会话回退：ip + 时间窗聚类窗口（秒）
_SESSION_FALLBACK_WINDOW = 1800  # 30min


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _query_hash(text: str) -> str:
    """查询文本 sha256（截断后），用于改述重查检测。脱敏前原始文本。"""
    if not text:
        return ""
    return hashlib.sha256(text[:2000].encode("utf-8")).hexdigest()[:16]


def _query_normalized(text: str) -> str:
    """查询文本小写+去空白+去标点，用于同查询重查检测。"""
    if not text:
        return ""
    return re.sub(r"[\s\W]+", "", text.lower())[:200]


def _extract_request_info(request, body: Optional[bytes]) -> dict:
    """从请求中提取遥测字段（端点/查询文本/签名/结果等）。

    body 是中间件读到的原始请求体（POST 的 JSON）；GET 的查询参数从 URL 取。
    """
    from urllib.parse import parse_qs, urlparse

    path = request.url.path
    method = request.method
    info: dict = {"endpoint": path, "method": method}

    # POST 请求体（search/signature/code-search）
    if body and method == "POST":
        try:
            payload = json.loads(body)
        except (ValueError, json.JSONDecodeError):
            payload = {}
        if path == "/search":
            info["query_text"] = payload.get("query", "")
            info["component"] = payload.get("component", "")
            info["target_version"] = payload.get("target_version") or payload.get("version") or ""
        elif path == "/signature-search":
            info["query_text"] = payload.get("text", "")
            info["component"] = payload.get("component", "")
        elif path == "/code/search":
            info["query_text"] = payload.get("keyword", "")
            info["component"] = "code"
            info["repo"] = payload.get("repo", "vllm-ascend")
        elif path == "/tags/match":
            info["query_text"] = payload.get("text", "")
    # GET 请求参数（title/doc/code-diff）
    elif method == "GET":
        qs = parse_qs(urlparse(str(request.url)).query)
        if path == "/title":
            info["query_text"] = (qs.get("keyword", [""])[0])
            info["component"] = qs.get("component", [""])[0]
        elif path.startswith("/doc/"):
            info["query_text"] = ""
            info["doc_id"] = path[len("/doc/"):]
        elif path == "/code/diff":
            info["query_text"] = qs.get("keyword", [""])[0]
            info["repo"] = qs.get("repo", ["vllm-ascend"])[0]

    info["query_hash"] = _query_hash(info.get("query_text", ""))
    info["query_normalized"] = _query_normalized(info.get("query_text", ""))
    return info


def _extract_response_info(response_body: bytes, endpoint: str) -> dict:
    """从响应体提取命中 doc_id 列表 + 结果数。"""
    info: dict = {"result_doc_ids": [], "result_count": 0}
    if not response_body:
        return info
    try:
        data = json.loads(response_body)
    except (ValueError, json.JSONDecodeError):
        return info
    results = data.get("results") if isinstance(data, dict) else None
    if isinstance(results, list):
        info["result_count"] = len(results)
        info["result_doc_ids"] = [
            r.get("doc_id", "") for r in results if isinstance(r, dict) and r.get("doc_id")
        ][:50]  # 最多记 50 条
    # signature-search 的 signatures_text（规范实体投影用）
    if endpoint == "/signature-search":
        sig_text = data.get("signatures_text", "") if isinstance(data, dict) else ""
        info["signature_text"] = sig_text
        sigs = data.get("signatures", []) if isinstance(data, dict) else []
        info["signature_entities"] = [
            {"text": s.get("text", ""), "kind": s.get("kind", ""), "weight": s.get("weight", 0)}
            for s in sigs if isinstance(s, dict)
        ][:20]
    return info


def _resolve_session(request) -> str:
    """会话归属：X-Session-Id header → 回退 client_ip + 时间窗。"""
    sid = request.headers.get("x-session-id", "").strip()
    if sid:
        return sid
    # 回退：ip + 30min 时间窗（同一 ip 30min 内归为一个会话）
    client_ip = request.client.host if request.client else "unknown"
    window = int(time.time()) // _SESSION_FALLBACK_WINDOW
    return f"ip:{client_ip}:{window}"


class TelemetryStore:
    """遥测库访问器（mode=rw，独立于只读 kb.sqlite3）。"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS query_events (
                    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    client_ip   TEXT,
                    ts          TEXT NOT NULL,
                    endpoint    TEXT NOT NULL,
                    method      TEXT,
                    query_hash  TEXT,
                    query_normalized TEXT,
                    signature_hash TEXT,
                    signature_entities TEXT,
                    signature_text TEXT,
                    result_doc_ids TEXT,
                    result_count INTEGER,
                    component   TEXT,
                    target_version TEXT,
                    repo        TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_session_ts ON query_events(session_id, ts);
                CREATE INDEX IF NOT EXISTS idx_signature_hash ON query_events(signature_hash);
                CREATE INDEX IF NOT EXISTS idx_query_normalized ON query_events(query_normalized);
            """)

    def record(self, **fields) -> None:
        """写一条行为事件（中间件调用）。"""
        cols = ("session_id", "client_ip", "ts", "endpoint", "method",
                "query_hash", "query_normalized", "signature_hash",
                "signature_entities", "signature_text",
                "result_doc_ids", "result_count", "component",
                "target_version", "repo")
        vals = [fields.get(c) for c in cols]
        placeholders = ",".join("?" * len(cols))
        with self._conn() as c:
            c.execute(f"INSERT INTO query_events ({','.join(cols)}) VALUES ({placeholders})", vals)

    @staticmethod
    def should_track(path: str) -> bool:
        """是否记录该端点的行为。"""
        return any(path.startswith(p) or path == p for p in _TRACKED_PREFIXES)


def make_middleware(app, cfg: "AppConfig"):
    """挂 logging middleware 到 FastAPI app（仅 feedback_enabled 时生效）。

    在路由注册后调用；记请求 + 响应摘要到 telemetry.sqlite3。
    """
    if not cfg.confidence.feedback_enabled:
        return app  # 未启用：不挂中间件，零开销

    store = TelemetryStore(cfg.resolve(cfg.confidence.telemetry_path))

    @app.middleware("http")
    async def telemetry_middleware(request, call_next):
        path = request.url.path
        if not TelemetryStore.should_track(path):
            return await call_next(request)

        # 读请求体（POST）——中间件读后需重新包装供路由消费
        body = await request.body() if request.method == "POST" else b""

        # 执行请求拿到响应
        response = await call_next(request)

        # 读响应体（记命中 doc_id + 结果数）
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        # 组装遥测字段
        req_info = _extract_request_info(request, body if body else None)
        resp_info = _extract_response_info(response_body, path)
        session_id = _resolve_session(request)

        # signature_hash：用 signature_text 的规范实体投影 hash（缺口检测主键）
        sig_hash = ""
        sig_entities_json = ""
        sig_text = resp_info.get("signature_text", "")
        if sig_text and path == "/signature-search":
            entities = resp_info.get("signature_entities", [])
            # 规范实体集合：kind+text 排序后 hash（消除提取噪声，同故障不同措辞聚合）
            entity_keys = sorted(f"{e['kind']}:{e['text']}" for e in entities)
            sig_hash = hashlib.sha256("|".join(entity_keys).encode("utf-8")).hexdigest()[:16]
            sig_entities_json = json.dumps(entities, ensure_ascii=False)

        try:
            store.record(
                session_id=session_id,
                client_ip=request.client.host if request.client else "",
                ts=_now_iso(),
                endpoint=path,
                method=request.method,
                query_hash=req_info.get("query_hash", ""),
                query_normalized=req_info.get("query_normalized", ""),
                signature_hash=sig_hash,
                signature_entities=sig_entities_json,
                signature_text=sig_text[:500] if sig_text else "",
                result_doc_ids=json.dumps(resp_info.get("result_doc_ids", []), ensure_ascii=False),
                result_count=resp_info.get("result_count", 0),
                component=req_info.get("component", ""),
                target_version=req_info.get("target_version", ""),
                repo=req_info.get("repo", ""),
            )
        except Exception as e:
            # 遥测失败不影响检索（宁可丢遥测不丢查询）
            print(f"[telemetry] 记录失败（不影响查询）: {e}", flush=True)

        # 重新构造响应（body 已被消费，需重建）
        from starlette.responses import Response
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    return app
