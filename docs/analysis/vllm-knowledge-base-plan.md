# vLLM 故障知识库（图 + 向量）建设方案

> 目标：全量拉取 vLLM 社区数据（代码、Issues、PR、Comments、Discussions、Releases、文档），
> 在本地构建一个**图 + 向量**知识库，供处理业务故障/问题的 Agent 检索，
> 置信度按 **release 版本 + 讨论日期** 双维度动态计算（查询时算，不预存）。
> 不要求动态更新、不要求高可用；要求能平滑导入其他 wiki / 文档。

---

## 1. 需求拆解与关键决策

| 需求 | 含义 | 方案影响 |
|---|---|---|
| 拉取全部代码/issue/pr/讨论 | 全量快照，一次性 | 采集层要分页、限流、断点续传；采集与入库分离，可重跑 |
| 图向量库 | 既要语义检索（向量）又要关系追溯（图） | 双存储：向量库 + 属性图库，元数据/全文另存 |
| 置信度随版本 + 日期衰退 | 旧讨论、旧版本的结论对新环境参考价值递减 | 所有时间/版本信息作为**一等字段**入库；置信度公式查询时计算 |
| 查询动态计算置信度 | 不预存、不后台衰减 | 置信度 = 纯函数(查询参数, 元数据)，无定时任务 |
| 不动态更新 / 不高可用 | 快照式，单机 | 嵌入式存储即可（SQLite/LanceDB/Kùzu），不引入服务端集群 |
| 兼容导入其他 wiki/文档 | 未来任意来源可接入 | 定义**统一 Canonical Schema** + 适配器（adapter）模式 |

**核心设计原则：采集 → 规范化 → 入库 三段分离，任何新数据源只写一个 adapter。**

---

## 2. 总体架构

```
┌───────────────────── 数据源 ─────────────────────┐
│ GitHub REST API   │  GitHub GraphQL   │ git clone │
│ (issues/PRs/comments/reviews/timeline/releases)  │ (Discussions) │ (代码/tags/docs/) │
└─────────┬──────────────────────┬─────────────────┘
          ▼                      ▼
   ┌─────────────┐        ┌──────────────┐
   │ 采集适配器   │        │ 文档/代码适配器│   ← 未来 wiki/文档 也只加 adapter
   └─────────────┘        └──────────────┘
          ▼
   ┌─────────────────────────────────────────┐
   │ Canonical 中间格式（JSONL / Parquet 快照） │   ← 唯一事实源，可重放、可重嵌
   └──────────────┬──────────────────────────┘
                  ▼
   ┌──────────────┴───────────────┐
   │ 入库流水线（幂等，可重跑）      │
   │ ① chunk + embedding（向量）   │
   │ ② 实体/关系抽取 → 图（时间窗边）│
   │ ③ 元数据 + FTS5 全文索引       │
   └──────────────┬───────────────┘
                  ▼
   ┌─────────────────────────────────────────┐
   │ 检索服务（本地 FastAPI / MCP Server）      │
   │ 混合检索：向量 top-k + 全文 + 图扩展       │
   │ 置信度动态计算 + 重排 + 证据输出           │
   └──────────────┬──────────────────────────┘
                  ▼
           故障处理 Agent（LangGraph / 自研 / MCP 客户端）
```

---

## 3. 数据采集层（GitHub）

### 3.1 要拉什么
| 数据 | 接口 | 说明 |
|---|---|---|
| Issues | REST `GET /repos/vllm-project/vllm/issues?state=all&per_page=100` | PR 也混在这个接口（有 `pull_request` 字段则分流） |
| PR 详情 | REST `GET /repos/.../pulls/{n}` | merged_at、merge_commit_sha、base/head 分支 |
| Issue/PR 评论 | REST `GET /repos/.../issues/{n}/comments` | 正文+评论作为一个"讨论线"处理 |
| PR review 评论 | REST `GET /repos/.../pulls/{n}/comments` + `/reviews` | 含评审意见 |
| Timeline/事件 | REST `GET /repos/.../issues/{n}/timeline`（需 media type） | closed/linked PR/"Fixes #x" 等事件 |
| Releases | REST `GET /repos/.../releases?per_page=100` | **版本日历**：tag → 发布日期（置信度模型的地基） |
| 里程碑/标签 | REST | 版本信息的重要结构化来源 |
| Discussions | **GraphQL 专用**（REST 不支持） | vLLM 讨论区；需要 GraphQL 分页（cursor） |
| 代码/文档 | `git clone --filter=blob:none`（blobless） | docs/、README、release notes、examples 值得索引；全量代码按需后补 |
| Git tags | `git ls-remote --tags` 或本地 tag | 补全版本日历（含被删除的 release） |

### 3.2 工程要点
- 认证 token：5000 req/h（未认证只有 60/h，必须带 token）；全程 `per_page=100` 分页。
- 数据量估算（以调研到的规模，Phase 0 用实际 API 校准）：
  - PR 编号已到 **4.6 万+**，issues+PRs 总数约 **5~7 万**；
  - 评论/评审约 **30~50 万**条；
  - 全量拉取 REST 约 1~2 万次请求，token 配额下**几个小时**可完成，做好 backoff。
- **断点续传**：按 page/资源 id 记录 checkpoint；失败重试指数退避；注意 GitHub secondary rate limit。
- 原始响应**原样落盘**（JSONL），解析/清洗放后面 —— 保证可重跑、可重嵌（换 embedding 模型时不用重新拉取）。
- 拉取脚本只做"搬运"，不做任何业务解析，保持幂等。

---

## 4. Canonical Schema（兼容一切数据源的关键）

所有来源最终归一为统一记录，入库流水线只认这个格式：

```jsonc
{
  "source_type": "github_issue | github_pr | discussion | wiki | doc | code_chunk",
  "source_id": "github:issue:12345",
  "url": "https://github.com/vllm-project/vllm/issues/12345",
  "title": "...",
  "body": "...",                 // 正文；issue 类含完整讨论线（按时间序拼正文+评论）
  "created_at": "2024-03-01T10:00:00Z",
  "updated_at": "2024-03-10T00:00:00Z",
  "resolved_at": "2024-03-08T00:00:00Z",   // closed_at / 修复合并时间，null 表示未解决
  "status": "open | closed | merged | archived",
  "labels": ["bug", "bug: v0.6.x"],
  "version_span": {"min": "0.5.0", "max": "0.6.4"},   // 见 §5，尽量用结构化信号填充
  "reliability": 0.9,            // 来源可靠度（§5），也可入库后规则计算
  "extra": {}                    // 来源特有字段，如 PR 的 merge_commit_sha、讨论的 category
}
```

- 分块（chunking）：issue 类按"讨论线"整体或按评论边界切块（保留线程上下文元数据）；文档/代码按标题/函数切块，块与父文档保持 `PART_OF` 边。
- 一条 canonical 记录 → 一个或多个 chunk → 一个 embedding；元数据在 chunk 级冗余一份，方便过滤。

---

## 5. 存储层选型（本地嵌入式，推荐组合）

| 用途 | 推荐 | 备选 | 理由 |
|---|---|---|---|
| 向量 | **LanceDB**（嵌入式） | Qdrant local / Chroma | 零服务、单机、本地文件；百万级向量轻松 |
| 图 | **Kùzu**（嵌入式属性图，Cypher） | Neo4j Community / NetworkX | 支持时间属性边、Cypher 查询、批量导入快；NetworkX 全内存，数据量大后吃力 |
| 元数据 + 全文 | **SQLite + FTS5** | DuckDB | FTS5 提供 BM25 关键词检索 + 版本/标签过滤，零依赖 |
| 原始快照 | Parquet/JSONL 文件 | — | 唯一事实源，可重放 |

> 备选方案 B：直接用 **Graphiti + Neo4j**（见 §8 对比）。核心推荐仍是上面这组：全部嵌入式、
> 无服务端、置信度完全自控，且因为 Canonical 中间格式的存在，以后想换 Neo4j 成本很低。

### 图的 Schema（时间窗是一等公民）

节点：`Issue` `PR` `Discussion` `Comment` `Commit` `Release` `Doc` `Chunk` `Version` `Module` `ErrorSignature` `Label`

边（每条边带 `created_at` / `valid_at` / `invalid_at`，对应 Graphiti 的 bi-temporal 模型）：

- `COMMENTS_ON`：评论 → Issue/PR
- `FIXES`：PR → Issue（解析"Fixes #n"、关联事件）
- `ADDRESSES`：Commit → Issue（commit message 里的 "fixes #n"）
- `MERGED_IN`：PR → Release（merge 时间落在两个 tag 日期之间）
- `PART_OF`：Chunk → Doc / Issue
- `MENTIONS`：Issue/PR → Version / Module / ErrorSignature（结构化抽取 + 规则）
- `RELATED`：Issue ↔ Issue（duplicate / 互相引用）

典型故障查询的图路径：`ErrorSignature →(MENTIONS) Issue →(FIXES) PR →(MERGED_IN) Release` —— 这正好回答"这报错在哪个版本修好的"。

---

## 6. 置信度模型（核心：查询时动态计算）

三个独立因子，全部由存储的元数据在查询瞬间算出，**没有任何后台衰减任务**：

### 6.1 版本相关性 `w_ver`（针对目标部署版本 V）
用版本日历（Release tag → 日期）把每条知识映射到版本区间 `[v_min, v_max]`：
- `v_min`：issue 创建时间对应的最近已发布版本（或 issue 里声明的版本）；
- `v_max`：修复合并时间对应的发布版本（无修复则 +∞，或取 issue 最后活跃时间）。
- 打分：`V` 落在区间内 → 高分；区间外 → 按**小版本距离**衰减；无法确定 → 默认 0.5。
  ```
  w_ver = 1.0                         若 v_min ≤ V ≤ v_max
  w_ver = exp(-dist(V, 区间) / σ)      否则（dist 按小版本数计，σ 默认 1~2）
  w_ver = 0.5                         版本信息缺失
  ```
- 语义：修复类知识只对"修复落地之前的版本"最相关；`v_max < V` 表示"你已在新版本，旧 bug 可能不存在了"。

### 6.2 时间衰退 `w_time`（针对讨论日期）
半衰期模型：
```
w_time = floor + (1 - floor) · 2^( -(T - t_ref) / HL )
HL 默认 365 天，floor 默认 0.15
t_ref = resolved_at（修复类知识，从"结论确定日"开始衰退）
      = created_at（文档/长期有效知识）
```
- floor 保证经典知识（如老 issue 里的标准解法）不会随时间归零；
- 半衰期、floor、是否用 resolved_at，全部做成配置项。

### 6.3 来源可靠性 `w_rel`
| 类型 | 分值 |
|---|---|
| 官方文档 / 该版本的 release notes | 0.85 |
| 已关闭且有合并 PR（有修复） | 0.90 |
| 已关闭（无修复，如 duplicate/answer） | 0.60 |
| 仍 open 的讨论 | 0.40 |
| 社区 wiki / 第三方文档 | 0.70 |

### 6.4 综合与排序
```
confidence = w_time · (α·w_ver + β·w_rel)         α+β=1，默认 α=0.6, β=0.4
最终排序分 = sim^γ · confidence^(1-γ)              γ 默认 0.6（语义相似度与置信度融合）
```
- 输出时**给出置信度分解**（时间/版本/可靠度各项），Agent 可据此判断是否采信；
- 参数集中在 `confidence_config.py`，跑一批真实故障案例后调参（Phase 4 有评估环节）。

---

## 7. 检索与 Agent 接入

1. **召回**：向量 top-k（bge-m3 等多语言模型，支持 dense+sparse）+ FTS5 关键词 top-k，合并去重；
2. **图扩展**：对召回节点沿 `FIXES / ADDRESSES / MERGED_IN / RELATED` 扩展一跳，补齐"相关 issue → 修复 PR → 落地版本"上下文；
3. **过滤 + 置信度重排**：按目标版本 V、日期 T 过滤，动态算 confidence，重排；
4. **证据输出**：每条结果附 URL、日期、版本区间、置信度分解 —— Agent 回答故障时带引用，可溯源。

对外形态：
- 本地 **FastAPI** 服务：`POST /search {query, target_version, now, filters}` → 排序结果；
- 或 **MCP Server**（如 `vllm-kb`），让任意 MCP 客户端（Claude Desktop、自研 agent、LangGraph tool）直接调用；
- 故障处理典型流程：错误签名/堆栈摘要 → 检索 → 图扩展 → 置信度排序 → 给出"原因 + 修复 PR + 落地版本 + 规避方案"。

---

## 8. Graphiti 评估（你说的线索）

**Graphiti 是什么**：[Zep 开源的 Python 时间知识图谱库](https://www.getzep.com/platform/graphiti/)，基于 Neo4j，
面向非结构化文本（对话、文档）自动构建**带 valid_at/invalid_at 时间窗的实体关系图**，内置混合检索（BM25+向量）与增量更新，已被 [Thoughtworks 技术雷达收录](https://www.thoughtworks.com/en-us/radar/platforms/graphiti)。

**契合点**：
- bi-temporal 边模型（valid_at/invalid_at）与"知识随版本/时间失效"的需求高度吻合；
- 混合检索、实体解析开箱即用；
- 用它做 wiki/文档/对话类非结构化来源的入库很合适。

**不契合点**：
- 其流水线面向**非结构化 episode**（对话/文档流），而 GitHub 的 issues/PRs/comments/releases 是**结构化数据**，时间、状态、版本本来就明确，套用它的 episode 流水线反而绕路；
- 实体/关系抽取依赖 **LLM 调用**，几十万条评论级别的规模成本可观、且确定性差；
- 依赖 **Neo4j 服务端**（与"本地、零运维"有出入）；
- 它的时间模型管"知识何时有效"，**不含置信度衰减**，衰减公式仍需自研。

**结论**：**采用 Graphiti 的时间窗 + 混合检索思想，自研嵌入式实现**（推荐，与你的约束最匹配）；
若你更想少写代码、接受 Neo4j + LLM 抽取成本，则直接上 Graphiti，仅需自研置信度层。
两条路都走 Canonical 中间格式，切换成本可控。

---

## 9. 嵌入式自研：代码量与前置知识

按推荐组合（LanceDB + Kùzu + SQLite FTS5 + 外部 embedding API）自研，**核心代码总量约 2000~3000 行 Python**：

| 模块 | 文件 | 预估行数 | 难度与要点 |
|---|---|---|---|
| GitHub 采集（REST） | pull_github.py | 300~500 | 分页、限流、断点续传、原始落盘 |
| Discussions 采集（GraphQL） | pull_discussions.py | 100~200 | GraphQL cursor 分页 |
| Canonical Schema + 各来源 adapter | canonical.py / adapters/ | 150~250 | Pydantic 模型，未来 wiki 只加 adapter |
| 分块 + 去重 | chunking.py | 150~250 | 按评论边界切、相似 stack trace 去重 |
| Embedding（接外部 API） | embed.py | 80~150 | batch 调用、限流、失败重试、checkpoint |
| 入库（LanceDB + SQLite FTS5） | ingest.py | 200~300 | 幂等重建 |
| 图构建（Kùzu） | graph_build.py | 300~500 | **最难点**：Fixes/MERGED_IN 关系推断、版本区间映射 |
| 置信度模块 | confidence.py | 100~200 | 纯函数，无状态 |
| 检索服务（FastAPI） | search.py / api.py | 200~350 | 混合检索 + 图扩展 + 重排 |
| MCP Server（可选） | mcp_server.py | 100~200 | 让任意 agent 可调用 |
| 测试 + 评估 | tests/ eval/ | 200~400 | 真实故障案例评估集 |

难点高度集中：**采集层的 GitHub API 细节**（限流/续传）和**图构建的关系推断**（哪个 PR 修了哪个 issue、合并进哪个 release）约占一半工作量；置信度、检索、入库本身都很薄。

### 需要的前置知识/技术
1. **Python 熟练**：Pydantic、requests/httpx、pandas/polars 处理 Parquet/JSONL；
2. **GitHub API**：REST 分页与 rate limit、timeline media type、GraphQL（Discussions）—— 硬性要求；
3. **信息检索基础**：embedding/向量相似度、BM25（SQLite FTS5）、chunking、hybrid search 权重调优 —— 概念层面即可，不需要深度学习背景；
4. **图模型基础**：节点/边/属性、类 Cypher 查询（Kùzu）、用时间窗交集做关系推断（PR merge 日期落在两个 tag 日期之间 → MERGED_IN）；
5. **运维几乎为零**：三个存储都是文件（复制目录即备份/迁移），唯一的"服务"是本地 FastAPI，可随时启停。

### 外部 embedding API 的注意点（按你的选择）
- 选**多语言模型**（vLLM issue 以英文为主，但将来 wiki/文档可能是中文）：如 bge-m3（部分 API 提供 dense+sparse）、jina-embeddings-v3、voyage-3、text-embedding-3-large；
- 外部 API 一般只返回 dense 向量：**稀疏召回由 SQLite FTS5 兜底**，混合检索不受影响；若 API 支持 sparse（如 bge-m3 类），存双向量进一步提升；
- 按 batch 调用 + 限流重试 + checkpoint；30~50 万条 chunk 的嵌入成本（费用/时间）要在 Phase 0 用 500 条实测后放大估算；
- **换 embedding 模型 = 全量重嵌**：这正是"原始快照与入库分离"的意义，重嵌只是重跑流水线。

> 对比：若改用 Graphiti+Neo4j，图构建与混合检索约省 400~700 行，但换来 Neo4j 服务端运维 + LLM 实体抽取成本 + 置信度仍需自研约 100~200 行。对"单机、无运维"的约束，省这 500 行并不划算。

---

## 10. 分阶段路线（从哪里开始）

### Phase 0 —— 最小链路验证（1~2 天，**先做这个**）
目的：在写大量代码前，证明"拉取 → 规范化 → 嵌入 → 检索 → 置信度"整条链路可行。
1. 建项目骨架（Python + venv + uv/poetry）；`gh auth login` 或 PAT；
2. 写 `pull_github.py`：只拉**最近 500 条 closed issue**（含正文、评论、closed 时间）到 JSONL，分页 + 限流 + checkpoint；
3. 定义 Canonical Schema v0（Pydantic 模型）；
4. LanceDB + SQLite FTS5 建库，用你选定的外部 embedding API embed 这 500 条（batch + 限流 + checkpoint）；
5. 写 50 行检索函数：向量 top-10 + §6 置信度公式 → 打印排序结果。

**验收标准**：拿 3 个真实 vLLM 故障报错（如 OOM / CUDA illegal memory access / 某个 API 报错），能搜到相关 issue 并给出合理置信度分解。跑通即可，不用优化。

### Phase 1 —— 全量采集（2~4 天）
- issues/PRs/评论/评审/timeline/releases/labels/milestones 全量拉到 Parquet 快照；
- Discussions（GraphQL）拉取；
- blobless clone + docs/、release notes 解析；
- 构建**版本日历**（tag→日期，含历史 tag）；断点续传 + 失败重试跑完。

### Phase 2 —— 图与向量入库（2~3 天）
- chunk + embedding（外部 embedding API，batch + 限流 + checkpoint）；
- 结构化实体/关系抽取（版本、模块、错误签名、Fixes 关联、MERGED_IN）→ Kùzu 建图（时间窗边）；
- FTS5 索引；幂等重建脚本。

### Phase 3 —— 检索服务与 Agent 接入（2~3 天）
- 混合检索 + 图扩展 + 置信度重排的 `/search` 服务；
- MCP Server 封装；接入你的故障处理 Agent（LangGraph / 自研）。

### Phase 4 —— 扩展与评估（1~2 天）
- wiki/文档通用 adapter（markdown/HTML wiki dump → canonical 格式），验证兼容性；
- 用你们历史真实故障案例建评估集（50 条），调置信度参数（HL、floor、α/β/γ）；
- 输出使用文档。

---

## 11. 风险与注意事项

- **版本信息提取噪声大**：优先用结构化信号（标签 `bug: v0.6.x`、修复 PR 的 merge 日期、milestone），正文里"我用的 0.6.1"只做辅助；文本提取做不进置信度公式。
- **issue 正文超长/重复 stack trace**：分块时按评论边界切、全局去重相似错误文本，避免污染向量索引。
- **API 限流**：token 配额 + secondary rate limit；原始快照落盘保证失败可续。
- **换 embedding 模型需全量重嵌**：所以原始快照与入库分离（§3.2），重嵌只是重跑流水线。
- **代码索引范围**：先只嵌 docs/、README、release notes、examples；全量代码嵌入量大且对"故障知识"边际价值低，作为可选扩展（按 issue 中引用的文件增量补）。
- **时间语义陷阱**：时区统一 UTC；`updated_at` 不等于"讨论结论时间"，置信度用 created/resolved 而非 updated。

---

## 12. 第一天就能动手的清单

```
git init vllm-kb && cd vllm-kb
python -m venv .venv && pip install lancedb kuzu fastapi pydantic requests
# 1. 拉 500 条 closed issues → data/raw/issues.jsonl（分页 + token + checkpoint）
# 2. canonical.py：Pydantic schema v0
# 3. ingest.py：LanceDB 建表 + embed + SQLite FTS5
# 4. search.py：向量 top-10 + confidence(query_version, now) 公式 + 打印
# 5. 验收：3 个真实报错能搜到答案
```

之后每一阶段（§9）都有明确交付物和验收标准，可以在任意阶段停下来先验证效果再继续。
