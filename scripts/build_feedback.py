"""离线推断管道：从遥测行为序列 → 三态反馈证据 → 后验更新 + 知识缺口判决。

三段分离的中间段（原始行为在 telemetry 库，后验产出给查询期 w_hist）：
1. 扫 telemetry.sqlite3 的 query_events，按 session_id 分组重建会话行为序列
2. 行为推断规则：从序列模式推断 hit/miss/unknown（权重≤0.5，自证循环阻尼）
3. 后验更新：apply_feedback_event（时间维度指数遗忘）→ confidence_feedback.json
4. 知识缺口复合判决：强签名零命中/命中无结论/低分重复 → knowledge_gaps 表

推断规则无偏（正负对称）+ 可重算（改规则重跑，原始数据不丢）。
unknown 不进 n_eff（单独计数），统计正确。

用法：
    python scripts/build_feedback.py [--config config.json]
"""
import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.config import AppConfig
from vllm_kb.feedback_model import DocFeedback, apply_feedback_event, save_feedback_table


# 行为推断权重（弱 0.3 / 中 0.5，上限 0.5——自证循环阻尼）
_W_HIT_WEAK = 0.3  # 弱正：拉 doc 后不重查 / signature 命中后直接结束
_W_HIT_MED = 0.5   # 中正：有 doc 命中后调 code/diff
_W_MISS_WEAK = 0.3  # 弱负：60s 内改述重查
_W_MISS_MED = 0.5   # 中负：无 doc 命中后调 code/diff
# 会话超时窗口（判定会话结束：无动作超过此时间视为会话结束）
_SESSION_TIMEOUT = 300  # 5min
# 改述重查窗口（短时间内不同 query 重查）
_REPHRASE_WINDOW = 60  # 60s


def _load_events(telemetry_path: Path) -> list[dict]:
    """加载遥测事件（排除探索/测试打标 probe≠0），按 (session_id, ts) 排序。

    probe 语义见 vllm_kb/telemetry.py（0 真实 / 1 显式声明 / 2 启发式识别）；
    探索行为打标不删除——排除发生在推断层，规则调整后重跑即恢复（原始数据不丢）。
    旧库无 probe 列时视为 0（全量进推断，与打标机制引入前行为一致）。
    """
    if not telemetry_path.exists():
        print(f"[feedback] 遥测库不存在: {telemetry_path}（先启用 feedback_enabled 跑 serve_api 积累数据）")
        return []
    conn = sqlite3.connect(str(telemetry_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM query_events ORDER BY session_id, ts"
        ).fetchall()
    finally:
        conn.close()
    events = [dict(r) for r in rows]
    kept = [e for e in events if not e.get("probe", 0)]
    excluded = len(events) - len(kept)
    if excluded:
        by_reason = {}
        for e in events:
            p = e.get("probe", 0)
            if p:
                by_reason[p] = by_reason.get(p, 0) + 1
        detail = ", ".join(f"probe={p}({'显式声明' if p == 1 else '启发式识别'})×{n}"
                           for p, n in sorted(by_reason.items()))
        print(f"[feedback] 排除探索/测试事件 {excluded} 条（{detail}），不进反馈推断")
    return kept


def _group_sessions(events: list[dict]) -> dict[str, list[dict]]:
    """按 session_id 分组，每个会话内按时间排序 + 超时切分。"""
    by_session: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_session[e["session_id"]].append(e)
    # 会话内超时切分（无动作超过 _SESSION_TIMEOUT 视为新会话段）
    sessions: dict[str, list[dict]] = {}
    for sid, evts in by_session.items():
        evts.sort(key=lambda e: e["ts"])
        segments = []
        cur = [evts[0]]
        for prev, e in zip(evts, evts[1:]):
            dt = _parse_ts(e["ts"]) - _parse_ts(prev["ts"])
            if dt.total_seconds() > _SESSION_TIMEOUT:
                segments.append(cur)
                cur = [e]
            else:
                cur.append(e)
        segments.append(cur)
        for i, seg in enumerate(segments):
            sessions[f"{sid}#{i}"] = seg
    return sessions


def _parse_ts(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _result_doc_ids(e: dict) -> set[str]:
    try:
        return set(json.loads(e.get("result_doc_ids") or "[]"))
    except (json.JSONDecodeError, TypeError):
        return set()


def _count_resolved_docs(doc_ids: set[str], cfg: AppConfig) -> int:
    """查 kb.sqlite3（只读）：命中 doc 中有多少已 resolved（status IN closed/merged AND resolved_at）。

    用于 soft_gap 判定——命中但全无 resolved 结论才算缺口。
    """
    if not doc_ids:
        return 0
    kb_path = cfg.resolve(cfg.storage.sqlite_path)
    if not kb_path.exists():
        return 0  # 库不存在，保守返回 0（视为无结论）
    import sqlite3 as _sqlite3
    try:
        conn = _sqlite3.connect(f"file:{kb_path}?mode=ro", uri=True)
        try:
            placeholders = ",".join("?" * len(doc_ids))
            row = conn.execute(
                f"SELECT count(*) FROM docs WHERE source_id IN ({placeholders}) "
                f"AND status IN ('closed', 'merged') AND resolved_at IS NOT NULL",
                list(doc_ids),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
    except Exception:
        return 0  # 查询失败保守返回 0


def _is_strong_signature(entities_json: str, cfg: AppConfig) -> bool:
    """强签名判定：kind 白名单 + weight≥阈值（过滤短码如 drvRetCode=6 weight=1.0 根因太泛）。

    weight 阈值 strong_sig_weight_min（默认 2.5，后继从反馈数据学）。
    """
    if not entities_json:
        return False
    try:
        entities = json.loads(entities_json)
    except (json.JSONDecodeError, TypeError):
        return False
    wmin = cfg.confidence.strong_sig_weight_min
    kinds = set(cfg.confidence.strong_sig_kinds)
    return any(
        e.get("kind") in kinds and e.get("weight", 0) >= wmin
        for e in entities if isinstance(e, dict)
    )


def infer_feedback(events: list[dict], cfg: AppConfig) -> list[dict]:
    """从行为序列推断三态反馈证据。

    返回 [{doc_id, state, weight, session_id, ts}, ...]。
    unknown 不进 n_eff（feedback_model.apply_feedback_event 处理）。
    """
    sessions = _group_sessions(events)
    evidence: list[dict] = []

    for sid, evts in sessions.items():
        # 会话内行为序列分析
        has_doc_pull = any(e["endpoint"].startswith("/doc/") for e in evts)
        has_code = any(e["endpoint"] in ("/code/search", "/code/diff") for e in evts)
        search_events = [e for e in evts if e["endpoint"] == "/search"]
        sig_events = [e for e in evts if e["endpoint"] == "/signature-search"]

        # 1. signature 命中后会话直接结束（无后续 doc/code/重查）→ 弱正
        for se in sig_events:
            doc_ids = _result_doc_ids(se)
            if doc_ids:
                # 检查会话内是否在 sig 之后有动作
                ts_sig = _parse_ts(se["ts"])
                later_actions = [
                    e for e in evts
                    if _parse_ts(e["ts"]) > ts_sig
                    and e["endpoint"] not in ("/health",)
                ]
                if not later_actions:
                    # 直接结束 → 弱正（大概率找到答案）
                    for did in doc_ids:
                        evidence.append({"doc_id": did, "state": "hit", "weight": _W_HIT_WEAK,
                                         "session_id": sid, "ts": se["ts"]})

        # 2. search/signature 命中后拉 doc 且不重查同 signature → 弱正
        for se in (search_events + sig_events):
            doc_ids = _result_doc_ids(se)
            if not doc_ids:
                continue
            ts_se = _parse_ts(se["ts"])
            # 是否拉了 doc
            pulled_docs = {
                e["endpoint"][len("/doc/"):] for e in evts
                if e["endpoint"].startswith("/doc/") and _parse_ts(e["ts"]) > ts_se
            }
            if pulled_docs & doc_ids:
                # 是否重查同 signature（同 query_normalized）
                later_same = [
                    e for e in evts
                    if _parse_ts(e["ts"]) > ts_se
                    and e.get("query_normalized") == se.get("query_normalized")
                ]
                if not later_same:
                    for did in (pulled_docs & doc_ids):
                        evidence.append({"doc_id": did, "state": "hit", "weight": _W_HIT_WEAK,
                                         "session_id": sid, "ts": se["ts"]})

        # 3. code/diff 在有 doc 命中后 → 中正；无 doc 命中时不产生 doc 级证据
        # （无命中的查询进缺口检测，不进后验——没有目标 doc 可标）
        if has_code:
            any_doc_hit = any(_result_doc_ids(e) for e in (search_events + sig_events))
            if any_doc_hit:
                # 中正：深入探索核对修复
                for e in (search_events + sig_events):
                    for did in _result_doc_ids(e):
                        evidence.append({"doc_id": did, "state": "hit", "weight": _W_HIT_MED,
                                         "session_id": sid, "ts": e["ts"]})

        # 4. 60s 内改述重查（不同 query_normalized）→ 弱负
        for i, se in enumerate(search_events):
            ts_se = _parse_ts(se["ts"])
            for se2 in search_events[i + 1:]:
                dt = _parse_ts(se2["ts"]) - ts_se
                if dt.total_seconds() > _REPHRASE_WINDOW:
                    break
                if se2.get("query_normalized") != se.get("query_normalized"):
                    # 改述重查 → 弱负（前一次命中的 doc）
                    for did in _result_doc_ids(se):
                        evidence.append({"doc_id": did, "state": "miss", "weight": _W_MISS_WEAK,
                                         "session_id": sid, "ts": se["ts"]})

        # 5. 零后续动作（search 命中后无 doc/code/重查）→ unknown（不进 n_eff）
        for se in search_events:
            doc_ids = _result_doc_ids(se)
            if not doc_ids:
                continue
            ts_se = _parse_ts(se["ts"])
            later_actions = [
                e for e in evts
                if _parse_ts(e["ts"]) > ts_se and e["endpoint"] != "/health"
            ]
            if not later_actions:
                for did in doc_ids:
                    evidence.append({"doc_id": did, "state": "unknown", "weight": 0.0,
                                     "session_id": sid, "ts": se["ts"]})

    return evidence


def update_posteriors(evidence: list[dict], existing: dict[str, DocFeedback],
                      cfg: AppConfig) -> dict[str, DocFeedback]:
    """把证据应用到后验表（时间维度指数遗忘 + 加新事件）。"""
    table = dict(existing)
    for ev in evidence:
        did = ev["doc_id"]
        now = _parse_ts(ev["ts"])
        fb = table.get(did)
        table[did] = apply_feedback_event(fb, ev["state"], ev["weight"], now, cfg.confidence)
    return table


def detect_gaps(events: list[dict], cfg: AppConfig) -> list[dict]:
    """知识缺口复合判决：强签名零命中 / 命中无结论 / 低分重复。

    每种事件不同权重打分，复合判决。条件可配（阈值后继从反馈数据学）。
    返回 [{gap_type, signature_hash, signature_entities, component, session_count, ...}]。
    """
    # 按 signature_hash 聚合（跨会话）
    by_sig: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        sh = e.get("signature_hash", "")
        if not sh:
            continue
        by_sig[sh].append(e)

    gaps: list[dict] = []
    for sh, evts in by_sig.items():
        # 跨会话去重
        sessions = {e["session_id"].split("#")[0] for e in evts}
        if len(sessions) < 3:
            continue  # 至少 3 个不同会话

        # 强签名判定
        entities_json = evts[0].get("signature_entities", "")
        is_strong = _is_strong_signature(entities_json, cfg)

        # 统计
        zero_hits = [e for e in evts if e.get("result_count", 0) == 0]
        has_hits = [e for e in evts if e.get("result_count", 0) > 0]
        component = evts[0].get("component", "")

        gap_type = None
        if is_strong and zero_hits:
            gap_type = "hard_gap"  # 强签名零命中
        elif has_hits:
            # 命中但无结论：查命中 doc 是否有 resolved（status IN closed/merged AND resolved_at）
            all_doc_ids = set()
            for e in has_hits:
                all_doc_ids.update(_result_doc_ids(e))
            resolved_count = _count_resolved_docs(all_doc_ids, cfg)
            if resolved_count == 0:
                gap_type = "soft_gap"  # 命中但全无 resolved 结论
        elif zero_hits:
            gap_type = "quality_gap"  # 零命中但弱签名

        if gap_type:
            gaps.append({
                "gap_type": gap_type,
                "signature_hash": sh,
                "signature_entities": entities_json,
                "signature_text": evts[0].get("signature_text", "")[:200],
                "component": component,
                "session_count": len(sessions),
                "first_seen": min(e["ts"] for e in evts),
                "last_seen": max(e["ts"] for e in evts),
                "sample_queries": list({e.get("query_normalized", "") for e in evts})[:5],
                "avg_result_count": sum(e.get("result_count", 0) for e in evts) / len(evts),
            })
    return gaps


def save_gaps(gaps: list[dict], telemetry_path: Path) -> None:
    """缺口写到 telemetry 库的 knowledge_gaps 表（与 query_events 同库，独立表）。"""
    conn = sqlite3.connect(str(telemetry_path))
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                gap_id INTEGER PRIMARY KEY AUTOINCREMENT,
                gap_type TEXT,
                signature_hash TEXT,
                signature_entities TEXT,
                signature_text TEXT,
                component TEXT,
                session_count INTEGER,
                first_seen TEXT, last_seen TEXT,
                sample_queries TEXT, avg_result_count REAL,
                detected_at TEXT
            )
        """)
        conn.execute("DELETE FROM knowledge_gaps")  # 全量重算时清空旧
        for g in gaps:
            conn.execute(
                "INSERT INTO knowledge_gaps (gap_type, signature_hash, signature_entities, "
                "signature_text, component, session_count, first_seen, last_seen, "
                "sample_queries, avg_result_count, detected_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (g["gap_type"], g["signature_hash"], g["signature_entities"],
                 g["signature_text"], g["component"], g["session_count"],
                 g["first_seen"], g["last_seen"], json.dumps(g["sample_queries"], ensure_ascii=False),
                 g["avg_result_count"], datetime.now(timezone.utc).isoformat()),
            )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="离线推断：行为序列 → 三态反馈 → 后验更新 + 缺口判决")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = AppConfig.load(args.config, require_keys=False)
    if not cfg.confidence.feedback_enabled:
        print("[feedback] confidence.feedback_enabled=False，无遥测数据。改配置后重启 serve_api 积累数据。")
        return

    telemetry_path = cfg.resolve(cfg.confidence.telemetry_path)
    feedback_path = cfg.resolve(cfg.confidence.feedback_path)

    # 1. 加载遥测事件
    events = _load_events(telemetry_path)
    if not events:
        print("[feedback] 无遥测事件（serve_api 需启用 feedback_enabled 积累数据）")
        return
    print(f"[feedback] 加载 {len(events)} 条遥测事件")

    # 2. 行为推断
    evidence = infer_feedback(events, cfg)
    print(f"[feedback] 推断出 {len(evidence)} 条三态证据")
    by_state = defaultdict(int)
    for ev in evidence:
        by_state[ev["state"]] += 1
    for state, n in by_state.items():
        print(f"  {state}: {n}")

    # 3. 后验更新
    from vllm_kb.feedback_model import load_feedback_table
    existing = load_feedback_table(feedback_path)
    table = update_posteriors(evidence, existing, cfg)
    save_feedback_table(feedback_path, table)
    print(f"[feedback] 后验表更新：{len(table)} 个 doc → {feedback_path}")

    # 4. 缺口判决
    gaps = detect_gaps(events, cfg)
    save_gaps(gaps, telemetry_path)
    print(f"[feedback] 知识缺口：{len(gaps)} 个 → {telemetry_path}/knowledge_gaps")
    for g in gaps[:10]:
        print(f"  [{g['gap_type']}] {g['signature_text'][:60]}  (sessions={g['session_count']})")

    print("[feedback] 完成。重启 serve_api 后 w_hist 生效。")


if __name__ == "__main__":
    main()
