"""GitHub 数据采集（REST + GraphQL）：全量历史数据拉取。数据源之一（type=github）。

设计（三段分离，支持全量/增量/断点续传）：
  拉取阶段只把原始 JSON 落盘（issues/{n}.json、prs/{n}.json、comments/{n}.json）+ checkpoint，
  不产生任何派生数据；canonical 由 recanonicalize() 从原始数据可再生（统一单文件），
  入库由 ingest 按内容哈希增量跳过 —— 换代码/换提取逻辑/补拉评论都不需要重拉 GitHub。

- 同时拉 issue 与 PR（issues 接口含 PR，按 pull_request 字段分流）；
- issue_state 支持 open/closed/all（all = 历史全量）；max_issues=0 表示全量；
- source_id 带 repo 命名空间（如 github:vllm-project-vllm:issue:123），多仓库不冲突；
- checkpoint 记录已拉编号 + 评论是否已拉 + 最后页 + 当时的查询参数，
  查询参数（state/sort/direction）变化时自动重置分页起点，避免新旧排序错位漏拉；
- 内网模式：VLLM_KB_INSECURE=1（或来源 insecure: true）跳过 SSL 校验，
  api_base / VLLM_KB_GITHUB_BASE 换镜像（REST 与 GraphQL 端点均跟随）。
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from .components import default_component_for_repo, extract_component_versions
from .config import PROJECT_ROOT, SourceCfg
from .models import KbDocument, VersionSpan
from .net import github_api_base, get_session, insecure_from_env

_LABEL_VERSION_RE = re.compile(r"v?(0\.\d+(?:\.\d+)?)")  # vLLM/vllm-ascend 均为 0.x.y 系，排除 3.5/26.1.0 等噪音
_BODY_VERSION_RE = re.compile(r"(?i)vllm\s+version[^0-9]*(\d+\.\d+(?:\.\d+)?)")
# 宽松回退要求完整 semver（x.y.z）：避免把 "vllm 0.1"、"vLLM 0.23" 这类残缺/噪音版本当真
_BODY_VERSION_LOOSE_RE = re.compile(r"(?i)\bvllm[\s:=_\-]*v?(\d+\.\d+\.\d+)")

# 环境转储段标记：issue 模板把环境信息放在这里，对语义检索是噪音，剥离到 extra
_ENV_MARKERS = [
    "### your current environment",
    "## your current environment",
    "<summary>environment</summary>",
]


def _split_environment(body: str) -> tuple[str, str]:
    """把 'Your current environment' 环境转储段从正文剥离（pip list/GPU/网卡等噪音），
    放入 extra['environment']（截断）备用。返回 (语义正文, 环境段)。"""
    lowered = body.lower()
    idx = -1
    for marker in _ENV_MARKERS:
        i = lowered.find(marker)
        if i != -1 and (idx == -1 or i < idx):
            idx = i
    if idx == -1:
        return body.strip(), ""
    return body[:idx].strip(), body[idx:].strip()[:4000]


class GithubPuller:
    # endCursor 重复（活跃仓库分页窗口滑动，GitHub GraphQL 已知行为）时的恢复参数：
    # 重试同一游标 STALL_RETRIES 次、间隔 STALL_RETRY_SECONDS；仍重复则跳过该 kind（不中断）。
    STALL_RETRIES = 2
    STALL_RETRY_SECONDS = 1.0

    def __init__(self, source: SourceCfg, project_root: Path = PROJECT_ROOT):
        self.source = source
        self.id = source.id
        self.repo = source.get("repo", "vllm-project/vllm")
        self.repo_slug = self.repo.replace("/", "-")  # source_id 命名空间
        # 主组件：来源显式指定优先，否则按仓库推断（vllm / vllm-ascend / ...）
        self.component = source.get("component", "") or default_component_for_repo(self.repo)
        # 内网模式（SSL 被禁/自签证书）：VLLM_KB_INSECURE=1 或来源配置 insecure: true。
        # 与 build_companion_matrix / build_release_calendar 等共用 net 统一入口。
        self.insecure = insecure_from_env() or bool(source.get("insecure", False))
        self.api_base = github_api_base(source.get("api_base") or None)
        self.per_page = source.get("per_page", 100)
        self.max_issues = source.get("max_issues", 0)  # 0 = 全量
        self.issue_state = source.get("issue_state", "all")
        self.include_prs = source.get("include_prs", True)
        self.sort = source.get("sort", "created")
        self.direction = source.get("direction", "desc")
        self.fetch_comments = source.get("fetch_comments", True)
        self.request_timeout_seconds = source.get("request_timeout_seconds", 30)
        self.max_retries = source.get("max_retries", 3)
        self.retry_backoff_seconds = source.get("retry_backoff_seconds", 5)
        self.raw_dir = project_root / source.get("raw_dir", f"data/raw/{source.id}")
        self.checkpoint_path = project_root / source.get(
            "checkpoint_file", f"data/checkpoints/{source.id}.json"
        )

        self.session = get_session(self.insecure)
        self.session.headers.update(
            {"Accept": "application/vnd.github+json", "User-Agent": "vllm-kb/0.1"}
        )
        token = source.get("token", "") or os.environ.get(
            source.get("token_env", "GITHUB_TOKEN"), ""
        )
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.base = self.api_base.rstrip("/")
        # GraphQL 端点跟随 api_base（GitHub Enterprise / 内网镜像同样走 GraphQL）
        self.graphql_url = f"{self.base}/graphql"

    # ---------------- HTTP 与限流 ----------------

    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.request_timeout_seconds)
            except requests.RequestException as e:
                last_exc = e
                time.sleep(self.retry_backoff_seconds * (attempt + 1))
                continue

            remaining = r.headers.get("X-RateLimit-Remaining")
            if remaining is not None and int(remaining) < 100:
                print(f"[github:{self.id}] 剩余配额 {remaining}，接近上限，注意限流")

            if r.status_code == 200:
                return r
            if r.status_code in (403, 429):
                if r.status_code == 403 and "rate limit" not in r.text.lower():
                    r.raise_for_status()  # 非限流的 403（权限等），直接报错
                self._sleep_for_rate_limit(r, tag="REST", attempt=attempt)
                continue
            if r.status_code >= 500:
                time.sleep(self.retry_backoff_seconds * (attempt + 1))
                continue
            r.raise_for_status()
        raise RuntimeError(f"GET 重试失败: {url} ({last_exc})")

    # ---------------- checkpoint ----------------

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_path.exists():
            return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        return {"issues": {}, "last_page": 0}

    def _save_checkpoint(self, cp: dict) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_text(json.dumps(cp, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 采集（GraphQL 游标分页，无 10k 上限） ----------------

    _ISSUES_QUERY = """
    query($owner: String!, $repo: String!, $states: [IssueState!], $after: String,
          $withComments: Boolean!, $filter: IssueFilters, $order: IssueOrder!) {
      repository(owner: $owner, name: $repo) {
        issues(first: 100, after: $after, states: $states, filterBy: $filter, orderBy: $order) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            number title body state createdAt updatedAt closedAt url
            author { login }
            labels(first: 20) { nodes { name } }
            comments(first: 100) @include(if: $withComments) {
              pageInfo { hasNextPage }
              nodes { author { login } createdAt body }
            }
          }
        }
      }
    }
    """

    _PRS_QUERY = """
    query($owner: String!, $repo: String!, $states: [PullRequestState!], $after: String,
          $withComments: Boolean!, $order: IssueOrder!) {
      repository(owner: $owner, name: $repo) {
        pullRequests(first: 100, after: $after, states: $states, orderBy: $order) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            number title body state createdAt updatedAt closedAt url merged mergedAt
            mergeCommit { oid }
            author { login }
            labels(first: 20) { nodes { name } }
            comments(first: 100) @include(if: $withComments) {
              pageInfo { hasNextPage }
              nodes { author { login } createdAt body }
            }
          }
        }
      }
    }
    """

    def _graphql_request(self, query: str, variables: dict) -> dict:
        """GraphQL 查询（带重试）。返回 repository 对象。

        注意 GraphQL 限流的两个形态：
        - 主限流超限时 HTTP 状态码**仍是 200**，body 报错且 x-ratelimit-remaining=0
          （必须识别并等到 X-RateLimit-Reset，而不是当普通错误抛异常中止）；
        - 次级限流返回 403/429 + Retry-After，按 header 精确等待。
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.post(
                    self.graphql_url,
                    json={"query": query, "variables": variables},
                    timeout=self.request_timeout_seconds,
                )
            except requests.RequestException as e:
                last_exc = e
                time.sleep(self.retry_backoff_seconds * (attempt + 1))
                continue
            if r.status_code == 200:
                data = r.json()
                if data.get("errors"):
                    msgs = [e.get("message", "") for e in data["errors"]]
                    joined = " ".join(msgs).lower()
                    if "rate limit" in joined or "rate_limit" in joined:
                        self._sleep_for_rate_limit(r, tag="GraphQL", attempt=attempt)
                        continue
                    raise RuntimeError(f"GraphQL 错误: {data['errors']}")
                return data["data"]["repository"]
            if r.status_code in (403, 429):
                self._sleep_for_rate_limit(r, tag="GraphQL", attempt=attempt)
                continue
            if r.status_code >= 500:
                time.sleep(self.retry_backoff_seconds * (attempt + 1))
                continue
            r.raise_for_status()
        raise RuntimeError(f"GraphQL 重试失败: {last_exc}")

    def _sleep_for_rate_limit(self, r: requests.Response, tag: str, attempt: int = 0) -> None:
        """按限流响应头精确等待：Retry-After > X-RateLimit-Reset > 指数退避。"""
        retry_after = r.headers.get("Retry-After")
        wait: int | None = None
        if retry_after:
            try:
                wait = int(retry_after)
            except ValueError:
                wait = None
        if wait is None:
            reset = r.headers.get("X-RateLimit-Reset")
            if reset:
                try:
                    wait = max(0, int(reset) - int(time.time()) + 1)
                except ValueError:
                    wait = None
        if wait is None:
            wait = self.retry_backoff_seconds * (attempt + 1)
        if wait > 60:
            when = time.strftime("%H:%M:%S", time.localtime(int(time.time()) + wait))
            print(f"[github:{self.id}] {tag} 限流，等待 {wait}s（预计 {when} 恢复）...")
        else:
            print(f"[github:{self.id}] {tag} 限流，等待 {wait}s ...")
        time.sleep(wait)

    @staticmethod
    def _graphql_states(issue_state: str, is_pr: bool) -> list[str]:
        """把 issue_state(open/closed/all) 映射为 GraphQL 枚举。"""
        if is_pr:
            mapping = {
                "open": ["OPEN"],
                "closed": ["CLOSED", "MERGED"],
                "all": ["OPEN", "CLOSED", "MERGED"],
            }
        else:
            mapping = {"open": ["OPEN"], "closed": ["CLOSED"], "all": ["OPEN", "CLOSED"]}
        return mapping.get(issue_state, mapping["all"])

    @staticmethod
    def _item_from_node(node: dict, is_pr: bool) -> dict:
        """GraphQL 节点 -> REST 形状的 item（复用 _to_doc / 落盘格式）。"""
        labels = [
            {"name": l["name"]}
            for l in (node.get("labels") or {}).get("nodes") or []
            if l.get("name")
        ]
        item = {
            "number": node["number"],
            "title": node["title"] or "",
            "body": node["body"] or "",
            "state": "open" if node["state"] == "OPEN" else "closed",
            "created_at": node["createdAt"],
            "closed_at": node["closedAt"],
            "html_url": node["url"],
            "labels": labels,
            "user": {"login": (node.get("author") or {}).get("login") or "unknown"},
        }
        if is_pr:
            item["pull_request"] = {
                "merged": bool(node.get("merged")),
                "merged_at": node.get("mergedAt"),
                "merge_commit_sha": (node.get("mergeCommit") or {}).get("oid"),
            }
        return item

    @staticmethod
    def _comments_from_node(node: dict) -> list[dict]:
        """GraphQL 评论节点 -> REST 形状评论（与 _get_comments 的输出一致，_to_doc 复用）。"""
        cnode = node.get("comments") or {}
        return [
            {
                "user": {"login": (c.get("author") or {}).get("login") or "unknown"},
                "created_at": c.get("createdAt"),
                "body": c.get("body") or "",
            }
            for c in (cnode.get("nodes") or [])
            if c
        ]

    def _collect_graphql(self, cp: dict, g: dict, kind: str, incremental: bool = False,
                         missing: bool = False) -> tuple[int, int]:
        """按游标枚举一类集合（issues/prs），落原始 JSON + checkpoint。

        返回 (本轮新增, 本轮跳过)。进度行带 totalCount 完成度与新增/跳过计数。

        **增量模式**（incremental=True，由 --incremental 触发）：
        - 窗口起点 = checkpoint 的 `{kind}_since`（上次增量看到的 max createdAt，
          updatedAt ≥ since 的新条目必在窗口内——新增的 updatedAt=createdAt 天然满足；
          被更新过的旧条目也会进入窗口，靠已拉编号跳过，无副作用）；
        - 无 since（首次增量 / 旧 checkpoint）→ 从头枚举（created desc），兼容旧行为；
        - 排序 UPDATED_AT DESC：新更新在前，翻过有更新的区域后**连续 3 页无新增
          即提前停止**——增量翻页数与"自上次以来有更新的条目数"成正比，
          不再全量扫描（issues 同时带 filterBy.since 服务端过滤）。
        默认（incremental=False）：从 checkpoint 游标续拉（断点续传，created desc 不变），
        done 后由 pull() 跳过（不重复拉取）。

        **补差模式**（missing=True，由 --pull-missing 触发）：从头枚举（created desc，
        清游标），跳过基准 = **raw 目录已有编号 ∪ checkpoint 编号**——只拉缺失条目
        （补历史旧条目），翻到最新（hasNextPage=False）后置 done；不推进增量时间窗口。

        游标异常（endCursor 与上一页相同，活跃仓库分页窗口滑动所致）先自动重试
        STALL_RETRIES 次；仍重复则保留已拉数据 + checkpoint 并跳过本 kind（不中断全流程），
        下次运行从 checkpoint 续传。
        """
        is_pr = kind == "prs"
        query = self._PRS_QUERY if is_pr else self._ISSUES_QUERY
        cursor = g.get(f"{kind}_cursor")
        states = self._graphql_states(self.issue_state, is_pr)
        owner, repo = self.repo.split("/", 1)
        new_count = 0
        skip_count = 0
        pages = 0
        total = 0
        # 增量窗口跟踪：本轮看到的 max createdAt（窗口只前进不后退——
        # 若窗口内全是被更新的旧条目，其 createdAt 可能比 since 更早，不能回退窗口）
        max_created: str | None = g.get(f"{kind}_since")
        since: str | None = None
        # 增量/补差均从头枚举：游标序列与上次断点续传无关，清掉残留 last_cursor，
        # 否则第一页 endCursor 若恰好等于上次中断时的游标会被误判"卡住"整轮放弃。
        if incremental or missing:
            cursor = None
            g.pop(f"{kind}_last_cursor", None)
        if incremental:
            since = g.get(f"{kind}_since") or None
            if since:
                print(f"[github:{self.id}] {kind} 增量拉取（--incremental：时间窗口 "
                      f"since={since}，UPDATED_AT DESC，连续 3 页无新增停止）", flush=True)
            else:
                print(f"[github:{self.id}] {kind} 增量拉取（--incremental：无历史窗口，"
                      f"从头枚举，连续 3 页无新增停止）", flush=True)
        elif missing:
            print(f"[github:{self.id}] {kind} 补差拉取（--pull-missing：从头枚举，"
                  f"跳过 raw/checkpoint 已有编号，只拉缺失）", flush=True)
        stale_pages = 0
        # 本类已收集数 = 原始目录里的条目数（跨 REST/GraphQL 累计）
        raw_kind_dir = self.raw_dir / kind
        collected_kind = len(list(raw_kind_dir.glob("*.json"))) if raw_kind_dir.exists() else 0
        # 补差模式：已拉集合 = raw 目录已有编号 ∪ checkpoint 编号（跳过基准以 raw 为准）
        known: set[str] | None = None
        if missing:
            known = {p.stem for p in raw_kind_dir.glob("*.json")} | set(cp["issues"])
        while True:
            if self.max_issues and len(cp["issues"]) >= self.max_issues:
                break
            variables = {
                "owner": owner, "repo": repo, "states": states, "after": cursor,
                "withComments": self.fetch_comments,
                # 增量按更新时间排序（新更新在前，提前停止）；断点续传保持创建时间
                # 排序不变（checkpoint 游标兼容）。
                "order": {"field": "UPDATED_AT" if incremental else "CREATED_AT",
                          "direction": "DESC"},
            }
            if not is_pr:
                # issues 支持服务端时间过滤（filterBy.since 语义 = updatedAt ≥ T）
                variables["filter"] = {"since": since} if since else None
            data = self._graphql_request(query, variables)
            collection = data["pullRequests" if is_pr else "issues"]
            total = collection.get("totalCount") or 0
            pinfo = collection["pageInfo"]
            # 游标前进检查：endCursor 与上一页相同。活跃仓库分页期间新条目持续创建
            # （created desc 排序窗口滑动），GitHub GraphQL 可能返回与上一页相同的
            # endCursor —— 属已知行为，不是死循环。先按相同 after 重试恢复；
            # 重试仍重复则保留已拉数据、保存 checkpoint，本轮跳过该 kind（不中断其他
            # 采集），下次运行自动从 checkpoint 续传。
            if pinfo.get("endCursor") and pinfo["endCursor"] == g.get(f"{kind}_last_cursor"):
                recovered = False
                for attempt in range(self.STALL_RETRIES):
                    print(f"[github:{self.id}] {kind} 游标未前进（endCursor 重复），"
                          f"第 {attempt + 1}/{self.STALL_RETRIES} 次重试同一游标 ...", flush=True)
                    time.sleep(self.STALL_RETRY_SECONDS)
                    data = self._graphql_request(query, variables)
                    collection = data["pullRequests" if is_pr else "issues"]
                    pinfo = collection["pageInfo"]
                    if not pinfo.get("endCursor") or pinfo["endCursor"] != g.get(f"{kind}_last_cursor"):
                        recovered = True
                        break
                if not recovered:
                    if incremental and max_created:
                        g[f"{kind}_since"] = max_created  # 窗口推进不丢失
                    print(f"[github:{self.id}] {kind} 游标持续重复（分页窗口滑动未恢复），"
                          f"保留已拉数据并保存 checkpoint，本轮跳过 {kind} 继续其他采集；"
                          f"下次运行自动从 checkpoint 续传。若持续卡住，可删除 checkpoint 全量重拉。",
                          flush=True)
                    self._save_checkpoint(cp)
                    break
            nodes = collection["nodes"] or []
            if not nodes:
                g[f"{kind}_done"] = True
                break
            page_new = 0
            for node in nodes:
                if self.max_issues and len(cp["issues"]) >= self.max_issues:
                    break
                # 增量窗口跟踪：记录本轮看到的 max createdAt（作为下次 since）。
                # ISO-8601 UTC 等宽（YYYY-MM-DDTHH:MM:SSZ），字典序即时间序，字符串比较安全。
                created = node.get("createdAt") or ""
                if created and (max_created is None or created > max_created):
                    max_created = created
                number = node["number"]
                rec = cp["issues"].get(str(number))
                if missing:
                    # 补差：raw 目录或 checkpoint 已有且评论齐 → 跳过（跳过基准以 raw 为准）
                    comments_done = bool(rec and rec.get("comments")) or (
                        self.raw_dir / "comments" / f"{number}.json").exists()
                    if str(number) in known and (not self.fetch_comments or comments_done):
                        skip_count += 1
                        continue
                else:
                    comments_done = bool(rec and rec.get("comments"))
                    if rec and (not self.fetch_comments or comments_done):
                        skip_count += 1
                        continue  # 已拉且评论已齐（断点续传/增量均跳过）
                item = self._item_from_node(node, is_pr)
                comments = (
                    self._comments_from_node(node)
                    if self.fetch_comments and not comments_done
                    else []
                )
                # 评论超过 GraphQL 单次 first:100 上限：REST 兜底补全（极少见的热门帖）
                if comments and (node.get("comments") or {}).get("pageInfo", {}).get("hasNextPage"):
                    comments = self._get_comments(number) or comments
                self._save_raw(kind, number, item)
                if comments:
                    self._save_raw("comments", number, comments)
                cp["issues"][str(number)] = {
                    "fetched_at": _now_iso(),
                    "comments": comments_done or self.fetch_comments,
                }
                new_count += 1
                page_new += 1
                collected_kind += 1
            pages += 1
            # 游标推进（已确认与上一页不同或首页）
            g[f"{kind}_last_cursor"] = pinfo["endCursor"]
            cursor = pinfo["endCursor"]
            g[f"{kind}_cursor"] = cursor
            if not pinfo.get("hasNextPage"):
                g[f"{kind}_done"] = True
            # 每页落 checkpoint：可断点续传
            self._save_checkpoint(cp)
            pct = f"{collected_kind}/{total}" if total else f"{collected_kind}"
            if incremental:
                # 增量带 filterBy.since 后 totalCount 是**窗口内**条目数（非全量），
                # collected_kind 是历史累计——两者不可比，分开显示避免倒挂误导
                pct = f"累计 {collected_kind} / 窗口 {total}" if total else f"累计 {collected_kind}"
            print(
                f"[github:{self.id}] {kind} 第 {pages} 页完成 | "
                f"本轮 新增 {new_count} / 跳过 {skip_count} | "
                f"{kind} 已收 {pct} | 全部已收 {len(cp['issues'])} | 游标 {pinfo['endCursor'][:12]}...",
                flush=True,
            )
            if incremental:
                # 连续 3 页无新增 → 新数据区已拉完，提前停止（避免每次全扫）
                if page_new == 0:
                    stale_pages += 1
                    if stale_pages >= 3:
                        if max_created:
                            g[f"{kind}_since"] = max_created  # 窗口推进（单调不后退）
                        print(f"[github:{self.id}] {kind} 增量完成"
                              f"（连续 {stale_pages} 页无新增）", flush=True)
                        break
                else:
                    stale_pages = 0
            if g.get(f"{kind}_done"):
                break
        if incremental and max_created:
            g[f"{kind}_since"] = max_created  # 正常翻完（hasNextPage=false）也推进窗口
        if not g.get(f"{kind}_done"):
            self._save_checkpoint(cp)
        return new_count, skip_count

    def pull(self, incremental: bool = False, missing: bool = False,
             numbers: Optional[list[int]] = None) -> int:
        """拉取 GitHub 原始数据（issue + PR + 评论），返回新增条数。

        - **默认（断点续传）**：从 checkpoint 游标续拉；一次拉完后置 done——
          之后默认**不再拉取**（日志打印"done 已设置，跳过；如需增量用 --incremental"）；
          拉取中断后重跑同一命令自动续传；
        - **incremental=True（--incremental）**：done 后仍增量拉取——时间窗口：
          从 checkpoint 的 `{kind}_since`（上次增量 max createdAt）起，issues 走
          filterBy.since 服务端过滤、PR 走 UPDATED_AT DESC 排序，跳过已有编号，
          连续 3 页无新增停止，把社区新增条目刷入并推进窗口；
        - **missing=True（--pull-missing）**：补差拉取——从头枚举（created desc），
          跳过 **raw 目录与 checkpoint 中已有的编号**，只拉缺失条目（补历史旧条目），
          翻到最新后置 done；与时间窗增量互补（增量只覆盖近期，补差不限新旧）；
        - **numbers=[...]（--numbers）**：REST 单条补拉——对指定编号先试
          `/pulls/{n}`（404 则 `/issues/{n}`）+ 评论落 raw（隐含 missing 语义，
          走 REST 不需要 GraphQL token，未认证限流 60 次/小时够单条场景）；
        - **全量重拉**：删除 data/raw/{source_id}/ 与 checkpoint 后重跑（见 USAGE）。
        """
        if numbers is not None:
            return self._pull_numbers(list(numbers))
        if not self.session.headers.get("Authorization"):
            raise RuntimeError(
                "拉取（GraphQL）需要 GitHub token：config.github.token 或 GITHUB_TOKEN"
            )
        cp = self._load_checkpoint()
        g = cp.setdefault("graphql", {})
        # 查询参数变化（closed->all、direction 等）时重置游标（已拉编号保留，不重复）
        if (
            cp.get("state") != self.issue_state
            or cp.get("sort") != self.sort
            or cp.get("direction") != self.direction
        ):
            print(f"[github:{self.id}] 查询参数变化，重置 GraphQL 游标（已拉编号保留）")
            # 增量时间窗口与查询参数无关，保留（否则下次增量退化回从头全扫）
            since_issues, since_prs = g.get("issues_since"), g.get("prs_since")
            g.clear()
            g.update(
                {"issues_cursor": None, "prs_cursor": None,
                 "issues_done": False, "prs_done": False}
            )
            if since_issues:
                g["issues_since"] = since_issues
            if since_prs:
                g["prs_since"] = since_prs

        new_count = 0
        skip_count = 0
        # 1) issues（先）
        if g.get("issues_done") and not incremental and not missing:
            print(f"[github:{self.id}] issues 已拉取完成（done），默认跳过——"
                  f"如需拉取社区增量请用 --incremental；补历史缺失用 --pull-missing；"
                  f"全量重拉请删除 raw 与 checkpoint")
        else:
            n, s = self._collect_graphql(cp, g, kind="issues", incremental=incremental,
                                         missing=missing)
            new_count += n
            skip_count += s
        # 2) PRs（后）
        if self.include_prs:
            if g.get("prs_done") and not incremental and not missing:
                print(f"[github:{self.id}] prs 已拉取完成（done），默认跳过——"
                      f"如需拉取社区增量请用 --incremental；补历史缺失用 --pull-missing")
            else:
                n, s = self._collect_graphql(cp, g, kind="prs", incremental=incremental,
                                             missing=missing)
                new_count += n
                skip_count += s

        cp["state"] = self.issue_state
        cp["sort"] = self.sort
        cp["direction"] = self.direction
        self._save_checkpoint(cp)
        print(
            f"[github:{self.id}] 完成：本轮新增 {new_count} / 跳过 {skip_count}，"
            f"已收集 {len(cp['issues'])} 条"
        )
        return new_count

    def _get_comments(self, number: int) -> list[dict]:
        url = f"{self.base}/repos/{self.repo}/issues/{number}/comments"
        params = {"per_page": 100}
        all_comments: list[dict] = []
        while True:
            r = self._get(url, params)
            batch = r.json()
            if not batch:
                break
            all_comments.extend(batch)
            if "next" not in r.links:
                break
            url = r.links["next"]["url"]
            params = None
        return all_comments

    # ---------------- 规范化 ----------------

    def _to_doc(self, item: dict, comments: list[dict]) -> KbDocument:
        number = item["number"]
        labels = [l.get("name", "") for l in item.get("labels", []) if l.get("name")]
        raw_body = item.get("body") or ""
        is_pr = "pull_request" in item
        pr_info = item.get("pull_request") or {}
        kind = self._kind_from_title(item.get("title") or "")

        # 环境转储段剥离：语义正文（用于分块/嵌入/检索）+ 环境段（extra 备用）
        semantic_body, env_section = _split_environment(raw_body)
        body = semantic_body
        if comments:
            parts = [
                f"### {c.get('user', {}).get('login', 'unknown')} ({c.get('created_at', '')}):\n{(c.get('body') or '')}"
                for c in comments
            ]
            body = body + "\n\n---\n\n" + "\n\n".join(parts)

        if is_pr:
            merged = bool(pr_info.get("merged"))
            source_type = "github_pr"
            status = "merged" if merged else ("closed" if item.get("state") == "closed" else "open")
            resolved_at = pr_info.get("merged_at") or item.get("closed_at")
            extra: dict[str, Any] = {
                "comment_count": len(comments),
                "github_number": number,
                "repo": self.repo,
                "kind": kind,
                "merged": merged,
                "merged_at": pr_info.get("merged_at"),
                "merge_commit_sha": pr_info.get("merge_commit_sha"),
            }
            # PR 正文提到版本不可靠（常见于兼容性/变更说明），版本信号只信标签
            span_min = self._version_from_labels(labels)
            component_versions: dict[str, str] = {}
        else:
            source_type = "github_issue"
            status = "closed" if item.get("state") == "closed" else "open"
            resolved_at = item.get("closed_at")
            extra = {
                "comment_count": len(comments),
                "github_number": number,
                "repo": self.repo,
                "kind": kind,
            }
            # 正文/环境段提取所有组件版本：主组件进 version_span，其余进 component_versions
            comp_versions = extract_component_versions(raw_body)
            span_min = self._version_from_labels(labels) or comp_versions.get(self.component)
            component_versions = {k: v for k, v in comp_versions.items() if k != self.component}
            if component_versions:
                extra["component_versions"] = component_versions
        if env_section:
            extra["environment"] = env_section

        # source_id 带 repo 命名空间：不同仓库的 issue/PR 编号不冲突
        kind = "pr" if is_pr else "issue"
        return KbDocument(
            source_type=source_type,
            source_id=f"github:{self.repo_slug}:{kind}:{number}",
            url=item.get("html_url") or f"https://github.com/{self.repo}/issues/{number}",
            title=item.get("title") or "",
            body=body,
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
            resolved_at=resolved_at,
            status=status,
            labels=labels,
            version_span=VersionSpan(min=span_min),
            component=self.component,
            component_versions=component_versions,
            extra=extra,
        )

    @staticmethod
    def _version_from_labels(labels: list[str]) -> str | None:
        """从标签（如 'bug: v0.6.x'）提取版本（弱信号）。"""
        for label in labels:
            m = _LABEL_VERSION_RE.search(label)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _kind_from_title(title: str) -> str:
        """按标题前缀判断 issue 类型（故障知识库中 bug/fix 权威，doc/feature/rfc 反馈类降权）。"""
        t = (title or "").strip().lower()
        for prefix, kind in (
            ("[bug]", "bug"),
            ("[fix]", "fix"),
            ("[doc]", "doc"),
            ("[feature]", "feature"),
            ("[rfc]", "rfc"),
        ):
            if t.startswith(prefix):
                return kind
        return "other"

    @staticmethod
    def _version_from_body(body: str) -> str | None:
        """从 issue 模板提取版本，两种格式：
        - '**vLLM version**: 0.26.0'（bug 模板强制字段，优先）
        - '- vLLM: v0.24.0' / 'vllm-0.26.0' / 'pip install vllm==0.5.4'（宽松回退）
        刻意不匹配 'vllm/vllm-openai:v0.26.0' 这类镜像 tag（vllm 后跟 '/' 不在允许分隔符内）。
        """
        m = _BODY_VERSION_RE.search(body)
        if m:
            return m.group(1)
        m = _BODY_VERSION_LOOSE_RE.search(body)
        return m.group(1) if m else None

    # ---------------- 落盘 ----------------

    def _save_raw(self, kind: str, number: int, data: Any) -> None:
        d = self.raw_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{number}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # ---------------- REST 单条补拉（--numbers） ----------------

    @staticmethod
    def _item_from_rest(data: dict, is_pr: bool) -> dict:
        """REST 响应（/pulls/{n} 或 /issues/{n}）→ 与 _item_from_node 输出一致的 item 格式。"""
        item = {
            "number": data["number"],
            "title": data.get("title") or "",
            "body": data.get("body") or "",
            "state": "open" if data.get("state") == "open" else "closed",
            "created_at": data.get("created_at"),
            "closed_at": data.get("closed_at"),
            "html_url": data.get("html_url"),
            "labels": [{"name": l["name"]} for l in data.get("labels") or [] if l.get("name")],
            "user": {"login": (data.get("user") or {}).get("login") or "unknown"},
        }
        if is_pr:
            item["pull_request"] = {
                "merged": bool(data.get("merged")),
                "merged_at": data.get("merged_at"),
                "merge_commit_sha": data.get("merge_commit_sha"),
            }
        return item

    def _pull_numbers(self, numbers: list[int]) -> int:
        """REST 单条补拉（--numbers，隐含 missing 语义）：对每个编号先试 `/pulls/{n}`
        （404 则 `/issues/{n}`）+ 评论落 raw，登记 checkpoint。已存在于 raw/checkpoint
        的编号跳过；不要求 GraphQL token（未认证限流 60 次/小时够单条场景）。"""
        import requests

        cp = self._load_checkpoint()
        new = 0
        for n in numbers:
            n = int(n)
            sn = str(n)
            if (self.raw_dir / "prs" / f"{n}.json").exists() or \
               (self.raw_dir / "issues" / f"{n}.json").exists():
                print(f"[github:{self.id}] #{n} raw 已有，跳过")
                continue
            if sn in cp["issues"]:
                print(f"[github:{self.id}] #{n} checkpoint 已有，跳过")
                continue
            # 先试 pulls（含 merged/merged_at/merge_commit_sha），404 再试 issues
            item: dict = {}
            kind = ""
            try:
                data = self._get(f"{self.base}/repos/{self.repo}/pulls/{n}").json()
                item = self._item_from_rest(data, is_pr=True)
                kind = "prs"
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code != 404:
                    raise
                try:
                    data = self._get(f"{self.base}/repos/{self.repo}/issues/{n}").json()
                except requests.HTTPError as e2:
                    if e2.response is not None and e2.response.status_code == 404:
                        print(f"[github:{self.id}] #{n} 不存在（pulls/issues 均 404），跳过")
                        continue
                    raise
                # issues 端点对 PR 也会返回（带 pull_request 字段）——补拉 pulls 拿 merged 信息
                if data.get("pull_request"):
                    try:
                        data = self._get(f"{self.base}/repos/{self.repo}/pulls/{n}").json()
                        item = self._item_from_rest(data, is_pr=True)
                        kind = "prs"
                    except requests.HTTPError:
                        item = self._item_from_rest(data, is_pr=True)  # merged 信息缺失，降级
                        kind = "prs"
                else:
                    item = self._item_from_rest(data, is_pr=False)
                    kind = "issues"
            comments = self._get_comments(n) if self.fetch_comments else []
            self._save_raw(kind, n, item)
            if comments:
                self._save_raw("comments", n, comments)
            cp["issues"][sn] = {"fetched_at": _now_iso(), "comments": self.fetch_comments}
            new += 1
            print(f"[github:{self.id}] #{n} 补拉完成（{kind}）")
        self._save_checkpoint(cp)
        return new


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_canonical(path: str | Path) -> list[KbDocument]:
    """读取统一 canonical 单文件（--rebuild 用）。单条损坏时告警跳过，不阻塞整体。"""
    p = Path(path)
    if not p.exists():
        return []
    docs = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            docs.append(KbDocument.model_validate_json(line))
        except Exception as e:
            print(f"[warn] canonical 第 {lineno} 行解析失败，跳过: {e}")
    return docs


def recanonicalize(source: SourceCfg, project_root: Path = PROJECT_ROOT) -> list[KbDocument]:
    """从该来源的原始 JSON（issues/、prs/、comments/）再生 canonical 文档。

    代码升级（版本提取逻辑、PR 映射等）后无需重新调用 GitHub API：
    原始快照是事实源，canonical 是可再生的派生物。
    """
    puller = GithubPuller(source, project_root)
    raw = puller.raw_dir
    docs: list[KbDocument] = []
    for kind in ("issues", "prs"):
        d = raw / kind
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json"), key=lambda p: int(p.stem)):
            item = json.loads(p.read_text(encoding="utf-8"))
            comments: list[dict] = []
            cp = raw / "comments" / f"{item['number']}.json"
            if cp.exists():
                comments = json.loads(cp.read_text(encoding="utf-8"))
            docs.append(puller._to_doc(item, comments))
    print(f"[github:{source.id}] recanonicalize 完成：{len(docs)} 条")
    return docs
