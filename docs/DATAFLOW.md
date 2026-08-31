# vllm-kb 数据流说明（入库 / 查询）

本文档回答两个常规操作的核心问题：

1. **入库**：数据从哪里来？经过什么处理？最终存在什么地方？
2. **查询**：agent 的请求从哪个 API 接口进来？后端查询哪些数据库？

配套代码：`vllm_kb/pipeline.py`（入库入口）、`vllm_kb/ingest.py`（落库）、`vllm_kb/api.py`（查询端点）、
`skills/vllm-kb/client.py`（agent 侧客户端）。

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
│  └─ 出口统一脱敏（sanitize）→ 返回 agent                        │
└──────────────────────────────────────────────────────────────┘
```

## 2. 入库数据流

### 2.1 GitHub 社区来源（issue / PR / comment）

触发命令：`python scripts/build_kb.py`（可加 `--limit N` / `--incremental`）。

| 阶段 | 处理 | 产物 |
|---|---|---|
| 1. 拉取 | `GithubSource.pull()`：REST issues（含 PR）+ GraphQL 内联评论；限流、重试、`data/checkpoints/{source_id}.json` 断点续传 | `data/raw/{source_id}/`（如 `data/raw/github`、`data/raw/vllm-ascend`）原始 JSON 快照（**事实源**，可重放） |
| 2. 规范化 | `src.canonicalize()`：原始 JSON → 统一 `KbDocument`（source_id / title / body / 组件 / 版本区间 / status / extra） | 追加/upsert 到统一 `data/raw/canonical.jsonl`（按 source_id 幂等） |
| 3. 入库 | `ingest_docs()`（见 2.6） | LanceDB 向量 + kb.sqlite3 |

### 2.2 业务来源（PDF 手册 / Markdown / Excel 登记表 / 截图 OCR）

文件放 `data/imports/{pdf,md,xlsx}/`（截图走 images source），config 启用对应 source 后跑
`python scripts/build_kb.py`（注意：本地文件导入**不要**用 `--skip-pull`，会跳过资产复制，见使用指南 §2.3）。

| 阶段 | 处理 | 产物 |
|---|---|---|
| 1. 资产复制 | `BaseSource.pull()` 把导入文件复制进资产层 | `data/assets/{pdf,md,images}/`，sha256 命名不可变（**资产路径不进检索库**，只存 asset_id） |
| 2. 解析 | PDF 文字层 + 表格提取；Markdown 正文 + 图片收集；Excel schema-free 任意 sheet/列拼接入库；截图 OCR（provider 可插拔：api/openai 兼容/paddle/ask） | `data/parsed/`（PDF 表格 JSON、OCR 结果 `*.ocr.json`，可重跑） |
| 3. 规范化 | `canonicalize()`：正文拼装 + 文档级**两级标签**（tagging：词典 `config.tags.registry` 子串命中 + 文件名/标题 token） | 同 2.1 步骤 2 → canonical.jsonl |
| 4. 入库 | 同 2.6 | LanceDB + kb.sqlite3 |

### 2.3 版本化代码仓（code 检索的数据源）

| 命令 | 产物 |
|---|---|
| `python scripts/build_code_snapshots.py` | vllm-ascend 各版本：`data/code/zips/{version}.zip` + 解压 `data/code/snapshots/{version}/` |
| `python scripts/build_vllm_snapshots.py` | 对应 vllm 主仓快照（版本由配套矩阵映射，自动跟随） |
| `python scripts/build_code_snapshots.py --index-only` | 派生数据重建：`data/code/index.sqlite3`（符号索引 + 报错字面量索引）、`symbols.json`（三层签名符号表）、`signal_words.json`（社区高频信号词） |

### 2.4 辅助数据

| 命令 | 产物 | 用途 |
|---|---|---|
| `python scripts/build_release_calendar.py --all-repos` | `data/compatibility/release_calendar.{repo}.json`（分仓） | 版本形态判断（release/rc/pre）+ 置信度版本上界（查询期现算） |
| `python scripts/build_companion_matrix.py`（`fetch_quay_tags.py` 辅助） | `data/compatibility/vllm-ascend.json` | 组件配套反向展开（vllm-ascend:0.18.0 → vllm/cann/pytorch-ascend） |

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

写入位置：

| 存储 | 内容 |
|---|---|
| `data/lancedb` | 每个 chunk 一条向量，meta 含 doc_id / title / 组件 / 版本区间 / tags（最终标签）/ section / reliability；原文存 text |
| `data/kb.sqlite3` 的 `docs` | 每篇文档一行：元数据 + `content_hash` / `embed_hash`（增量判断依据）+ `tags`（最终标签 JSON） |
| `data/kb.sqlite3` 的 `chunks_fts` | FTS5：`indexed_text` 存 jieba 分词结果（中文可独立命中）、`text` 存原文（snippet 展示用） |
| `data/kb.sqlite3` 的 `chunks_meta` | chunk_id → doc_id / seq / section |
| `data/kb.sqlite3` 的 `doc_tags` | 人工标签覆盖层（auto_snapshot / excluded / manual），最终标签 = (auto − excluded) ∪ manual |

> FTS 分词只影响索引列，向量库（原文嵌入）不受影响——`build_fts.py` 重建全文索引**无需重嵌向量**。

## 3. 查询数据流

### 3.1 请求入口

- Agent 只调用 skill：`python skills/vllm-kb/client.py <命令>`（标准库实现，零依赖，输出强制 UTF-8）；
- client 发 HTTP 到只读 FastAPI（`scripts/serve_api.py`）：`--base` > 环境变量 `VLLM_KB_BASE` >
  默认 `http://127.0.0.1:8000`；
- 服务端结构性只读：SQLite URI `mode=ro`、向量库经只读包装（写操作抛错）、无写端点、不导入可写模块。

### 3.2 端点 → 存储映射

| client 命令 | HTTP 端点（方法） | 后端查询的存储/文件 | 备注 |
|---|---|---|---|
| `search "组件:版本 问题"` | `POST /search` | LanceDB 向量召回（top 50）+ `kb.sqlite3` FTS5 BM25（top 50，jieba 分词）| 混合去重 → 置信度重排（时间衰退/版本区间/可靠度/验证状态）→ 未解决兜底 → 按文档去重 |
| `signature "原始报错"` | `POST /signature-search` | 现场三层提取签名（`data/code/symbols.json` 符号表 + `signal_words.json` 信号词 + 结构化正则）→ `kb.sqlite3` FTS 短语 + 标题命中 | 返回提取签名 + 精确命中 + 标题命中 |
| `title "关键词"` | `GET /title` | `kb.sqlite3` `docs` 表（title / source_id SQL LIKE） | 已知现象找 issue 最快路径 |
| `version 0.18.0` | `GET /version` | `data/compatibility/release_calendar.{repo}.json` | 版本形态判断 |
| `code <关键词>` | `POST /code/search` | `data/code/index.sqlite3` 符号索引命中；未命中退 grep 版本快照（`snapshots/`，按需解压 zip）| `--in-file` 限文件、`--per-version` 分版本；`--kind msg` 走报错字面量索引 |
| `code --file <路径>` | `GET /code/file` | `data/code` 指定版本快照文件（截断带标记） | |
| `diff <v1> <v2> <路径>` | `GET /code/diff` | 两个版本快照同一文件的 unified diff（difflib） | `--keyword` 只留相关差异行 |
| `code-versions` | `GET /code/versions` | `data/code` 可用预存版本清单 | 管理员调试 |
| `doc <source_id>` | `GET /doc/{source_id}` | `kb.sqlite3`：docs 行 + chunks_meta 排序 + chunks_fts 原文拼装 | extra 出口白名单清理（不返回服务器路径） |
| `components` / `stats` / `health` | `GET` | `kb.sqlite3` 聚合 / 向量库 count | `/health` 含 embedding 状态 |
| `companion` / `matrix` | `GET /companion` `/matrix` | `data/compatibility/vllm-ascend.json` | 配套反向展开 / 全量矩阵 |
| `graph chain/fixes/sig/doc/tags/evidence/stats` | `GET /graph/*` | Kùzu `data/graph`（只读查询） | 图未构建时返回引导提示（503→client 展示） |
| `tags list` / `tags docs <标签>` / `context "问题"` | `GET /tags` `/tags/{tag}/docs` `POST /tags/match` | `kb.sqlite3` `docs.tags`（最终标签）+ `config.tags.registry` 词典 | 能力发现：先知道知识库有哪些文档类别 |

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
  │     现算置信度（w_time×w_ver×w_rel，含 verification 因子）→ final = sim^γ·conf^(1−γ)
  ├─ 未解决兜底 + 按文档去重 → top 10
  └─ 出口脱敏（内部 IP/路径 → 占位）→ 返回 agent
```

### 3.4 只读与安全

- 无写端点；SQLite `mode=ro`（写操作在连接层必然失败）；向量库写操作抛 `ReadOnlyError`；
  `scripts/check_readonly.py` 可在运行前验证；
- 出口统一脱敏：正文/标题/图结果递归脱敏（`config.sanitize` 白名单，改配置即时生效、无需重嵌）；
  被脱敏原始值落 `data/sanitize_log.json`；
- `extra`/`evidence` 走字段白名单清理，`source_ref` 仅保留 http(s) URL——检索响应不含服务器路径。

## 4. kb.sqlite3 表结构

| 表 | 列 | 说明 |
|---|---|---|
| `docs` | source_id（PK）/ source_type / url / title / created_at / resolved_at / status / labels / version_span_min / version_span_max / reliability / component / content_hash / embed_hash / extra / tags | 文档元数据 + 增量哈希 + 最终标签 |
| `chunks_fts` | chunk_id（UNINDEXED）/ doc_id（UNINDEXED）/ indexed_text / text（UNINDEXED） | FTS5 虚拟表；indexed_text 存 jieba 分词、text 存原文 |
| `chunks_meta` | chunk_id（PK）/ doc_id / seq / section | 分块序号与章节（PDF/MD 手册） |
| `doc_tags` | source_id（PK）/ auto_snapshot / excluded / manual / updated_at / reviewer | 人工标签覆盖层（审核工作台维护） |

## 5. 关键设计点

- **幂等双哈希**：`embed_hash`（source_id+title+body）决定是否重嵌；`meta_hash`（整篇）决定是否刷新元数据——
  元数据变化不触发重嵌，崩溃续传按文档粒度恢复；
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
  不自动打标。详见 [使用指南 §3.5](USAGE.md#35-审核工作台人工确认统一入口--api-配置中心)；
- **后置脱敏**：库中存原文（原文检索），只在 serve_api 出口统一脱敏——改白名单即时生效、无需重嵌；
- **查询期现算**：修复落地版本上界（version_span_max 历史派生值有跨仓库错配风险）不落库，
  查询期按文档仓库的分仓日历实时计算，仅参与打分。
