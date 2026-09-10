# vllm-kb 数据流说明（入库 / 查询）

本文档回答两个常规操作的核心问题：

1. **入库**：数据从哪里来？经过什么处理？最终存在什么地方？
2. **查询**：agent 的请求从哪个 API 接口进来？后端查询哪些数据库？

配套代码：`vllm_kb/pipeline.py`（入库入口）、`vllm_kb/ingest.py`（落库）、`vllm_kb/api.py`（查询服务组装）、
`skills/vllm-kb/client.py`（agent 侧客户端）。查询端点按检索域拆分在
`vllm_kb/api_meta.py`（辅助）/ `api_community.py`（社区+文档）/ `api_code.py`（本地代码仓）/
`api_code_graph.py`（gh-puller 代码图谱，可选启用），`api.py` 只负责组装与出口脱敏。

## 1. 总览

```
┌────────────────────────── 入库（写） ──────────────────────────┐
│  GitHub 社区 ──pull──▶ data/raw/{source_id}/ ─canonicalize─┐    │
│  业务文件   ──资产+解析─▶ data/assets/ + data/parsed/ ──────┼──▶ canonical.jsonl
│                                                           │    │
│  canonical.jsonl ──chunk──▶ embed ──▶ LanceDB 向量库            │
│                └──────────▶ SQLite（docs / chunks_fts /         │
│                             chunks_meta / doc_tags）            │
│  代码快照 ──▶ data/code/{zips,snapshots}/ + index.sqlite3       │
│  版本日历/配套矩阵 ──▶ data/compatibility/*.json                │
│  canonical+parsed+kb.sqlite3 ──build_graph──▶ data/graph（Kùzu）│
└──────────────────────────────────────────────────────────────┘

┌────────────────────────── 查询（只读） ────────────────────────┐
│  Agent ──▶ skills/vllm-kb/client.py ──HTTP──▶ serve_api.py      │
│              （只读 FastAPI，127.0.0.1:8000 / VLLM_KB_BASE）    │
│  ├─ /search            ──▶ LanceDB + kb.sqlite3(FTS5) + 日历    │
│  ├─ /signature-search  ──▶ 符号表/信号词 + kb.sqlite3(FTS)      │
│  ├─ /code/*            ──▶ data/code（index.sqlite3 + 快照）    │
│  ├─ /graph/*           ──▶ data/graph（Kùzu）                   │
│  ├─ /tags/*            ──▶ kb.sqlite3(docs.tags) + 词典         │
│  ├─ /code-graph/*      ──▶ gh-puller 代码图谱（可选，外部服务） │
│  └─ 出口统一脱敏（sanitize）→ 返回 agent                        │
│                                                                │
│  行为遥测（feedback_enabled 时）：中间件记查询行为              │
│    ──▶ data/telemetry.sqlite3 ──离线推断──▶ confidence_feedback │
│    ──▶ 查询期 w_hist 因子（见 3.6）                             │
└──────────────────────────────────────────────────────────────┘
```

## 2. 入库数据流

### 2.1 GitHub 社区来源（issue / PR / comment）

触发命令：`python scripts/build_kb.py`（可加 `--limit N` / `--incremental` / `--pull-missing` / `--numbers`）。
拉取模式（互斥）：**断点续传**（默认，done 后跳过）/ **`--incremental` 时间窗增量**（近期新增）/
**`--pull-missing` 补差**（从头枚举，跳过 raw/checkpoint 已有，只拉缺失——补历史旧条目）/
**`--numbers` REST 单条**（指定编号走 `/pulls/{n}`→`/issues/{n}`，无需 GraphQL token）。

| 阶段 | 处理 | 产物 |
|---|---|---|
| 1. 拉取 | `GithubSource.pull()`：REST/GraphQL（issues 含 PR，PR 带 head 三元组 `head_repo/head_branch/head_sha`——fork PR 的来源仓/分支/锁定 commit，与 fork 快照对齐）+ 评论；限流、重试、`data/checkpoints/{source_id}.json` 断点续传 | `data/raw/{source_id}/`（如 `data/raw/github`、`data/raw/vllm-ascend`，子目录 `issues/` `prs/` `comments/`）原始 JSON 快照（**事实源**，可重放） |
| 2. 规范化 | `src.canonicalize()`：原始 JSON → 统一 `KbDocument`（source_id / title / body / 组件 / 版本区间 / status / extra） | 追加/upsert 到统一 `data/raw/canonical.jsonl`（按 source_id 幂等） |
| 3. 入库 | `ingest_docs()`（见 2.6） | LanceDB 向量 + kb.sqlite3 |

> **只再生 canonical（不入库）**：`scripts/build_canonical.py` 遍历来源 canonicalize → upsert
> canonical.jsonl，不 ingest——提取逻辑（版本/kind/组件/标签规则）升级后先跑它再 `build_graph.py`
> 建图，无需重嵌向量；`build_kb.py` 内部复用同一 canonical 处理（`pipeline.upsert_unified_canonical`）。

### 2.2 业务来源（PDF 手册 / Markdown / Excel 登记表 / 截图 OCR）

文件放 `data/imports/{pdf,md,xlsx}/`（截图走 images source），config 启用对应 source 后跑
`python scripts/build_kb.py`（注意：本地文件导入**不要**用 `--skip-pull`，会跳过资产复制，见使用指南 §2.3）。

| 阶段 | 处理 | 产物 |
|---|---|---|
| 1. 资产复制 | `BaseSource.pull()` 把导入文件复制进资产层 | `data/assets/{pdf,md,images}/`，sha256 命名不可变（**资产路径不进检索库**，只存 asset_id） |
| 2. 解析 | PDF 文字层 + 表格提取；Markdown 正文 + 图片收集；Excel schema-free 任意 sheet/列拼接入库；截图 OCR（provider 可插拔：`api`（含 `mode=custom` 自研协议 / `openai` 兼容）/ `paddle` / `none` 默认关闭，未知值报错） | `data/parsed/`（PDF 表格 JSON `*.tables.json` 与解析缓存 `*.extract.json`、OCR 结果 `*.ocr.json`，可重跑） |
| 3. 规范化 | `canonicalize()`：正文拼装 + 文档级**两级标签**（tagging：词典 `config.tags.registry` 子串命中 + 文件名/标题 token） | 同 2.1 步骤 2 → canonical.jsonl |
| 4. 入库 | 同 2.6 | LanceDB + kb.sqlite3 |

### 2.3 版本化代码仓（code 检索的数据源）

| 命令 | 产物 |
|---|---|
| `python scripts/build_code_snapshots.py` | vllm-ascend 各版本：`data/code/zips/{version}.zip` + 解压 `data/code/snapshots/{version}/` |
| `python scripts/build_vllm_snapshots.py` | 对应 vllm 主仓快照（版本由配套矩阵映射，自动跟随） |
| `python scripts/build_fork_snapshots.py` | 0day fork 仓快照（hy4/glm5.2 等模型开发分支）：`data/code/forks/{model}/`，版本=镜像锁定 commit（SHA 前 12 位）——检索走 `repo=fork:{model}` 命名空间，与官方版本物理隔离 |
| `python scripts/build_image_snapshots.py` | **0day 镜像内的 vllm-ascend 插件源码**：只拉镜像的 `COPY . /vllm-workspace/vllm-ascend/` 层（实测 23~101MB；整镜像 6GB+ 中的 CANN/编译层属二进制，不拉），解到 `data/code/images/{tag}/snapshots/{tag}/` + `index.sqlite3` + `meta.json`（image_digest/镜像时间/vllm commit/插件层 digest/层内 `.git` 若有则记 plugin_commit）——检索走 `repo=img:{tag}`，版本键=镜像 tag；`repo=img` 为列举入口；镜像 digest 未变则跳过重取 |
| `python scripts/build_code_snapshots.py --index-only` | 派生数据重建：`data/code/index.sqlite3`（符号索引 + 报错字面量索引）、`symbols.json`（三层签名符号表）、`signal_words.json`（社区高频信号词，统计实现 `scripts/build_signal_words.py`，可单独运行） |

### 2.4 辅助数据

| 命令 | 产物 | 用途 |
|---|---|---|
| `python scripts/build_release_calendar.py --all-repos` | `data/compatibility/release_calendar.{repo}.json`（分仓，slug 如 `vllm-project-vllm-ascend`；单仓模式写无后缀的 `release_calendar.json`） | 版本形态判断（release/rc/pre）+ 置信度版本上界（查询期现算） |
| `python scripts/build_companion_matrix.py`（`fetch_quay_tags.py` 辅助） | `data/compatibility/vllm-ascend.json`（每行含 `image_created`（镜像最后推送时间）、`vllm_commit`/`vllm_commit_date`（vllm 代码 commit 与提交日期）；fork 行另有 `vllm_repo/ref/base/sha/image_digest`）+ 跨运行缓存 `data/cache/`（fork 层 SHA 永久 / GitHub releases TTL 7 天 / requirements 永久 / tag→commit 永久，`--refresh-cache` 强制刷新） | 组件配套反向展开（vllm-ascend:0.18.0 → vllm/cann/pytorch-ascend）；commit 溯源回答"这个镜像对应哪个 commit"（官方行 tag→commit，fork 行 `vllm_sha`）；fork 行锁定 commit 供 `build_fork_snapshots.py` 按 SHA 拉快照 |

### 2.5 图存储（Kùzu）

命令：`python scripts/build_graph.py`（**必须先停检索服务**，Kùzu 单写者）。

输入：统一 `canonical.jsonl` + `data/parsed/`（手册表格→ErrorCode 节点）+ `kb.sqlite3`
（`doc_tags` 人工标签覆盖层）+ `config.tags.registry`（词典 Tag 节点）。
输出：`data/graph` —— Issue/PR/Release/Doc/Interface/Tag 节点 + FIXES / MERGED_IN / MENTIONS /
DOCUMENTS / CORROBORATES / TAGGED_WITH 边。

### 2.6 落库细节（`ingest_docs`，幂等双哈希增量）

```
预扫描：比较 docs 表存的 content_hash / embed_hash 与当前文档
  ├─ 两哈希均未变      → 跳过（不重嵌，崩溃续传按此粒度恢复）
  ├─ 仅元数据变化      → 刷新 docs 行 + 向量 meta（不重嵌）
  └─ 内容变化/新文档    → 全量路径：
       chunking（按段切块，max_chunk_chars=4000 / overlap=200；PDF/MD 带章节结构，
                 标题注入 chunk 文本并记 section）→ embed（OpenAI 兼容 /embeddings，
                 攒批 64 chunk/批）→ 写 LanceDB（攒批 200 条 flush）+ kb.sqlite3
```

> 空正文文档（chunking 结果为空）仍写 `docs` 行 + 两个哈希并计入 `skipped_empty`，
> 下次预扫描直接跳过、不重扫；旧库缺 `indexed_text` 列时 FTS 表自动 DROP 重建
> （只影响全文索引），需跑 `scripts/build_fts.py` 重建（见 2.7）。

写入位置：

| 存储 | 内容 |
|---|---|
| `data/lancedb` | 每个 chunk 一条向量，meta 含 doc_id / title / 组件 / 版本区间 / tags（最终标签）/ section / reliability；原文存 text |
| `data/kb.sqlite3` 的 `docs` | 每篇文档一行：元数据 + `content_hash` / `embed_hash`（增量判断依据）+ `tags`（最终标签 JSON） |
| `data/kb.sqlite3` 的 `chunks_fts` | FTS5：`indexed_text` 存 jieba 分词结果（中文可独立命中）、`text` 存原文（snippet 展示用） |
| `data/kb.sqlite3` 的 `chunks_meta` | chunk_id → doc_id / seq / section |
| `data/kb.sqlite3` 的 `doc_tags` | 人工标签覆盖层（auto_snapshot / excluded / manual），最终标签 = (auto − excluded) ∪ manual |

> FTS 分词只影响索引列，向量库（原文嵌入）不受影响——`build_fts.py` 重建全文索引**无需重嵌向量**。

### 2.7 索引重建与审核侧存储（不参与检索）

| 命令 / 组件 | 产物 | 说明 |
|---|---|---|
| `python scripts/build_fts.py` | 重建 `kb.sqlite3` 的 `chunks_fts`（读现有 chunk 原文重新 jieba 分词，chunk_id 与向量库严格一致） | 升级分词规则/旧库升级后使用；**不重新分块、不重嵌向量**，普通增量入库自动分词无需运行 |
| `python scripts/review_ui.py` | `data/review.sqlite3`（`review_items` 审核队列 / `asset_registry` 资产路径注册 / `doc_tags` 标签覆盖层） | 审核工作台独立端口，**只读检索 API 全程不碰该库**；资产路径只存 `asset_id → rel_path`，不进 canonical/检索库 |

> 审核队列的 7 类人工确认项、API 配置中心、知识缺口展示见
> [使用指南 §3.2](USAGE.md#32-审核工作台人工确认统一入口--api-配置中心)。

## 3. 查询数据流

### 3.1 请求入口

- Agent 只调用 skill：`python skills/vllm-kb/client.py <命令>`（标准库实现，零依赖，输出强制 UTF-8）；
- client 发 HTTP 到只读 FastAPI（`scripts/serve_api.py`）：`--base` > 环境变量 `VLLM_KB_BASE` >
  默认 `http://127.0.0.1:8000`；`code-graph` 子命令另走 `VLLM_KB_CODE_GRAPH_BASE`（缺省回落同一 base）；
- 服务端结构性只读：SQLite URI `mode=ro`、向量库经只读包装（写操作抛错）、无写端点、不导入可写模块；
- 路由按检索域拆分注册（`api.py` 只做组装）：`api_meta` / `api_community` / `api_code` 恒注册，
  `api_code_graph` 仅 `config.code_graph.enabled=true` 时注册（未启用则端点不存在）；
  `feedback_enabled=true` 时额外挂遥测中间件（见 3.6）。

### 3.2 端点 → 存储映射

| client 命令 | HTTP 端点（方法） | 后端查询的存储/文件 | 备注 |
|---|---|---|---|
| `search "组件:版本 问题"` | `POST /search` | LanceDB 向量召回（top 50）+ `kb.sqlite3` FTS5 BM25（top 50，jieba 分词）+ 配套矩阵/分仓日历 | 混合去重 → 置信度重排（时间衰退/版本区间/可靠度/验证状态/历史可靠度）→ 未解决兜底 → 按文档去重 |
| `signature "原始报错"` | `POST /signature-search` | 现场三层提取签名（`data/code/symbols.json` 符号表 + `signal_words.json` 信号词 + 结构化正则）→ `kb.sqlite3` FTS 短语 + 标题命中 | 返回提取签名 + 精确命中 + 标题命中 |
| `title "关键词"` | `GET /title` | `kb.sqlite3` `docs` 表（title / source_id SQL LIKE） | 已知现象找 issue 最快路径 |
| `version 0.18.0` | `GET /version` | `data/compatibility/release_calendar.{repo}.json` | 版本形态判断 |
| `code <关键词>` | `POST /code/search` | `data/code/index.sqlite3` 符号索引命中；未命中退 grep 版本快照（`snapshots/`，按需解压 zip）| `--in-file` 限文件、`--per-version` 分版本；`--kind msg` 走报错字面量索引 |
| `code --repo vllm` / `--repo fork:{model}` | 同上（repo 参数路由） | vllm 主仓快照（`build_vllm_snapshots.py`）/ 0day fork 仓快照（`data/code/forks/{model}/`，版本=锁定 commit）| fork 命名空间与官方版本物理隔离，必须显式传 `repo=fork:` 才命中 |
| `code --repo img:{tag}` | 同上（repo 参数路由） | **0day 镜像内插件源码**（`data/code/images/{tag}/`，版本键=镜像 tag；`build_image_snapshots.py` 提取） | 与官方/fork 三方隔离；`code --repo img`（无 tag）只用于列举已提取镜像 |
| `code --file <路径>` | `GET /code/file` | `data/code` 指定版本快照文件（截断带标记） | |
| `diff <v1> <v2> <路径>` | `GET /code/diff` | 两个快照同一文件的 unified diff（difflib），**两侧可属不同命名空间** | 版本参数带前缀：`img:{tag}` / `fork:{model}@{sha12}` / `vllm-ascend:{版本}` / `vllm:{版本}`；`--keyword` 只留相关差异行 |
| `code-versions` | `GET /code/versions` | `data/code` 可用预存版本清单；`repo=img` 返回 `images[]`（tag/digest/镜像时间/vllm commit/是否有索引） | 管理员调试；`repo=img` 是 agent 发现 `img:` 前缀的入口 |
| `doc <source_id>` | `GET /doc/{source_id}` | `kb.sqlite3`：docs 行 + chunks_meta 排序 + chunks_fts 原文拼装 | extra 出口白名单清理（不返回服务器路径） |
| `components` / `stats` / `health` | `GET` | `kb.sqlite3` 聚合 / 向量库 count | `/health` 含 embedding 状态 |
| `companion` / `matrix` | `GET /companion` `/matrix` | `data/compatibility/vllm-ascend.json` | 配套反向展开 / 全量矩阵 |
| `graph chain/fixes/sig/doc/tags/evidence/stats` | `GET /graph/*` | Kùzu `data/graph`（只读查询） | 图未构建时返回引导提示（503→client 展示） |
| `tags list` / `tags docs <标签>` / `context "问题"` | `GET /tags` `/tags/{tag}/docs` `POST /tags/match` | `kb.sqlite3` `docs.tags`（最终标签）+ `config.tags.registry` 词典 | 能力发现：先知道知识库有哪些文档类别 |
| `code-graph search/code-search/trace/query/architecture/changes` | `POST /code-graph/*` | **外部 gh-puller 代码图谱服务**（`config.code_graph.base_url` + `path`），本库不落数据 | 可选能力：`enabled=false` 时端点不注册；不可达 → 503 + 引导用 `code` 查本地索引（**不回退**） |
| `code-graph health` | `GET /code-graph/health` | gh-puller 可达性探测 | 不触发熔断计数，仅展示 |

### 3.3 关键路径举例（search）

```
POST /search {query:"vllm-ascend:0.23.0rc1 GLM5.1 PD分离P节点挂死"}
  │
  ├─ parse_component_query：拆出 component=vllm-ascend, version=0.23.0rc1, 语义词
  ├─ companion.expand(vllm-ascend, 0.23.0rc1) → 配套版本（vllm 0.23.0rc1、cann …）
  │     —— 其他组件文档按其配套版本参与打分（vllm 文档记 vllm 自己的版本）
  ├─ embed(语义词) → LanceDB 向量召回 50 条（embedding 不可用 → 熔断降级，只走 FTS）
  ├─ FTS5 BM25 召回 50 条（查询串 jieba 分词构造 MATCH；中文词独立命中）
  ├─ 合并去重（vector / fts / both）→ filters（--tag 等）→ 每篇按生效版本参考
  │     现算置信度 conf = w_time×(α·w_ver + β·w_rel)（α=0.6 / β=0.4，含 verification
  │     下限提升；w_rel 不信任库值、查询期按 kind 重算）
  │     → final = sim^γ·conf^(1−γ)·w_hist^σ（γ=0.6；w_hist 历史可靠度，见 3.6，
  │       feedback_enabled=false 时 =1.0 中性、不影响排序）
  ├─ 未解决兜底 + 按文档去重 → top 10
  └─ 出口脱敏（内部 IP/路径 → 占位）→ 返回 agent
```

### 3.4 只读与安全

- 无写端点；SQLite `mode=ro`（写操作在连接层必然失败）；向量库写操作抛 `ReadOnlyError`；
  `scripts/check_readonly.py` 可在运行前验证；
- 出口统一脱敏：正文/标题/图结果递归脱敏（`config.sanitize` 白名单，改配置即时生效、无需重嵌）；
  被脱敏原始值落 `data/sanitize_log.json`；
- `extra`/`evidence` 走字段白名单清理，`source_ref` 仅保留 http(s) URL——检索响应不含服务器路径。

### 3.5 代码图谱检索（可选，gh-puller 接入）

与 `/code/*`（本地版本化符号索引）**并列、不替换、不回退**——能力互补不重叠：

```
client.py code-graph <子命令> ──HTTP──▶ /code-graph/*（本服务只做协议转换）
                                        │  repo 简名 → gh-puller project 映射
                                        │  （config.code_graph.repo_project_map）
                                        ▼
                              gh-puller 代码图谱服务（MCP Streamable HTTP）
                              base_url + path（默认 /gh-puller/graph，端口 8787）
```

| 子命令 | 端点（方法） | 能力 |
|---|---|---|
| `search` | `POST /code-graph/search` | BM25/正则/语义三模搜函数/类/路由（优先于 grep 找定义） |
| `code-search` | `POST /code-graph/code-search` | grep + 图增强（去重到函数、按结构重要性排序） |
| `trace` | `POST /code-graph/trace` | 调用链/数据流/跨服务路径（`direction`/`depth`/`mode`/翻页 cursor） |
| `query` | `POST /code-graph/query` | Cypher 多跳/聚合/跨服务分析 |
| `architecture` | `POST /code-graph/architecture` | 架构总览（聚类/边界/热点/层次/依赖） |
| `changes` | `POST /code-graph/changes` | git diff → 变更影响面（blast radius） |
| `health` | `GET /code-graph/health` | gh-puller 可达性探测 |

- 启用条件：`config.code_graph.enabled=true` + `base_url`；未启用时端点**不存在**（比 503 更干净）；
- 失败语义：工具级错误 → 400（带行动建议），服务不可达 → 503 + 引导改用 `code` 查本地索引；
- 本库不存图谱数据，因此该链路**不涉及任何落库/重建**。

### 3.6 行为遥测 → 置信度反馈闭环（`feedback_enabled`）

三段分离，审计可逆：原始行为只记不删，推断可重算，后验只影响查询期排序。

```
serve_api（只读）                            离线周期
┌──────────────────────────┐              ┌──────────────────────────────┐
│ 中间件记查询行为          │              │ scripts/build_feedback.py    │
│ → data/telemetry.sqlite3 │ ───────────▶ │ 会话重建 + 行为推断三态      │
│ （独立库，不碰 kb.sqlite3）│              │ → data/confidence_feedback.json
│                          │              │ → knowledge_gaps（知识缺口） │
│ compute_confidence       │              └──────────────────────────────┘
│ + w_hist（读反馈表）      │ ◀────────────── 重启 serve_api 生效
│ final = sim^γ·conf^(1−γ)·lb^σ
└──────────────────────────┘
```

- 采集范围（`_TRACKED_PREFIXES`）：`/search`、`/signature-search`、`/title`、`/doc/`、`/code/search`、`/code/diff`；
  会话归属经 `VLLM_KB_SESSION` → `X-Session-Id` header（缺失回退 ip+时间窗）；
- 探索/测试行为打标 `probe`（`client.py --probe` / `VLLM_KB_PROBE=1` 显式 + 占位词启发式兜底），
  推断层排除 `probe≠0` 事件但保留原始行；
- `w_hist` 是**与 w_rel 正交**的独立因子（后验下界 lb），`n_eff=0` 时取 1.0 中性——
  与 §4 的 docs/chunks 表无关，不改库中任何数据；不启用时零开销、不挂中间件。

## 4. kb.sqlite3 表结构

| 表 | 列 | 说明 |
|---|---|---|
| `docs` | source_id（PK）/ source_type / url / title / created_at / resolved_at / status / labels / version_span_min / version_span_max / reliability / component / content_hash / embed_hash / extra / tags | 文档元数据 + 增量哈希 + 最终标签 |
| `chunks_fts` | chunk_id（UNINDEXED）/ doc_id（UNINDEXED）/ indexed_text / text（UNINDEXED） | FTS5 虚拟表；indexed_text 存 jieba 分词、text 存原文 |
| `chunks_meta` | chunk_id（PK）/ doc_id / seq / section | 分块序号与章节（PDF/MD 手册） |
| `doc_tags` | source_id（PK）/ auto_snapshot / excluded / manual / updated_at / reviewer | 人工标签覆盖层（审核工作台维护） |

### 4.1 旁路存储（不参与检索）

| 库 | 表 | 说明 |
|---|---|---|
| `data/review.sqlite3` | `review_items`（id/category/item_ref/payload/status/created_at/reviewed_at/reviewer/result）、`asset_registry`（asset_id/rel_path/sha256/size/source_type）、`doc_tags` | 审核队列 + 资产路径注册 + 标签覆盖层；只读检索 API 全程不碰；资产路径不进 canonical/检索库 |
| `data/telemetry.sqlite3` | `query_events`（session_id/client_ip/ts/endpoint/method/query_hash/query_normalized/signature_hash/signature_entities/signature_text/result_doc_ids/result_count/component/target_version/repo/probe） | 行为遥测原始事件（`feedback_enabled` 时写入）；离线推断产出 `data/confidence_feedback.json`（见 3.6） |

## 5. 关键设计点

- **幂等双哈希**：`embed_hash`（source_id+title+body）决定是否重嵌；`meta_hash`（整篇）决定是否刷新元数据——
  元数据变化不触发重嵌，崩溃续传按文档粒度恢复（`meta_hash` 存在 `docs.content_hash` 列，旧库兼容沿用列名）；
- **断点续传**：拉取断点存 `data/checkpoints/`；canonical 按 source_id upsert；重跑同一命令即续传；
- **canonical 是唯一事实源**：`--rebuild` 从 `canonical.jsonl` 全量重建（清空向量库 + 删 kb.sqlite3，需 TTY 确认或 `--yes`）；
  修改过的业务文件必须回写 canonical（upsert 覆盖），否则全量重建会回退旧内容；
  若 kb 与 canonical 漂移（kb 有、canonical 无 → 图/rebuild 丢文档），用
  `scripts/backfill_canonical.py` 从 kb.sqlite3 回填缺失文档（默认 dry-run，`--write` 回填）；
- **Kùzu 单写者**：更新图前必须先停检索服务，更新完重启；
- **标签候选治理**：文档打标时的未收录候选（`docs.extra.tag_candidates`，文件名/标题提取）→ 审核队列
  tag_candidate **按词聚合** → 采纳 = 入词典（config.json）+ 对全部提及文档写 `doc_tags.manual` +
  同步 `docs.tags`（检索侧立即生效；图侧重建后入图；向量 chunk meta 是入库快照，需重入库才一致）；
  正文 TF-IDF 候选（`build_tag_candidates.py` → `data/tag_candidates_manual.json`）是独立手动路径，
  不自动打标。详见 [使用指南 §3.2](USAGE.md#32-审核工作台人工确认统一入口--api-配置中心)；
- **后置脱敏**：库中存原文（原文检索），只在 serve_api 出口统一脱敏——改白名单即时生效、无需重嵌；
- **查询期现算**：修复落地版本上界（version_span_max 历史派生值有跨仓库错配风险）不落库，
  查询期按文档仓库的分仓日历实时计算，仅参与打分；
- **历史可靠度正交**：`w_hist`（行为遥测后验）作为独立因子乘在 final 上，不乘进 `w_rel`——
  同一文档元数据不变时 conf 部分保持确定性可审计；反馈数据全在旁路库/文件，不写检索库；
- **代码图谱是可选外挂**：`/code-graph/*` 由配置开关注册，数据在外部 gh-puller 服务，
  本库不落库、不参与重建；不可达时明确 503 + 引导，不做静默降级；
- **代码快照的三套命名空间互不混**：官方版本（`data/code/`）、fork 仓 vllm 代码（`data/code/forks/`）、
  镜像插件源码（`data/code/images/`）——默认检索只命中官方版本，必须显式 `repo=fork:…` / `repo=img:…`
  才检索 0day 代码；镜像插件代码无 git 元数据（`COPY` 拷入），只能靠跨命名空间 diff 定位定制点。
