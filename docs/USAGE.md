# vllm-kb 使用指南

本指南覆盖：环境准备、数据采集与更新、查询命令、审核工作台、故障处理推荐流程、远程部署（存算分离）。
数据如何入库、查询请求打到哪个接口 / 查哪些存储，见 [数据流说明](DATAFLOW.md)。

**目录**

- [1. 环境准备](#1-环境准备) —— [1.1 依赖](#11-依赖) / [1.2 配置](#12-配置密钥走环境变量)
- [2. 数据采集与构建](#2-数据采集与构建) —— [2.0 快速运维（单指令）](#20-快速运维单指令) /
  [2.1 增量与重建](#21-增量与重建) /
  [2.2 辅助数据构建](#22-辅助数据构建) / [2.3 业务来源导入（PDF / Markdown）](#23-业务来源导入pdf-手册--markdown-文档--完整实操) /
  [2.4 Excel 登记表导入](#24-excel-登记表导入schema-free--完整实操) /
  [2.5 Word 文档导入](#25-word-文档导入docxdocm--完整实操)
- [3. 启动检索服务](#3-启动检索服务) —— [3.1 日志与降级](#31-总日志接口打屏--可选落盘分卷) /
  [3.2 审核工作台](#32-审核工作台人工确认统一入口--api-配置中心) /
  [3.3 图更新流程](#33-图更新流程kùzu-单写者约束)
- [4. 查询命令（skill）](#4-查询命令skill) —— [4.1 search](#41-search--语义检索) /
  [4.2 signature](#42-signature--签名精确检索) / [4.3 title](#43-title--标题精确检索) /
  [4.4 version](#44-version--版本形态判断) / [4.5 code](#45-code--对应版本源码定位) /
  [4.6 其他](#46-其他) / [4.7 tags / context](#47-文档标签能力发现--tags--context) /
  [4.8 graph](#48-graph--phase-2-图检索关系追溯) /
  [4.9 code-graph](#49-code-graph--代码图谱检索gh-puller-接入) /
  [4.10 行为遥测与反馈](#410-行为遥测与置信度反馈feedbackenabled)
- [5. 故障处理推荐流程](#5-故障处理推荐流程)
- [6. 远程部署（存算分离）](#6-远程部署存算分离)
- [7. 常见问题](#7-常见问题)

## 1. 环境准备

### 1.1 依赖

```bash
pip install -r requirements.txt        # 核心：requests / pydantic / lancedb / kuzu / fastapi / uvicorn
```

> **Windows PowerShell**：设 `PYTHONUTF8=1` 避免 GBK 输出乱码——会话内 `$env:PYTHONUTF8=1`，
> 或持久化 `[Environment]::SetEnvironmentVariable("PYTHONUTF8","1","User")`。
> client.py 已在进程内把 stdout 重设为 UTF-8，但子进程（如 `--help`）与管道场景仍需该变量。

### 1.2 配置（密钥走环境变量）

```bash
cp config.example.json config.json

# GitHub 采集 token（必须，未认证限流 60 次/小时）
export GITHUB_TOKEN=ghp_xxx

# Embedding API key（可选；不设则把 config.json 的 embedding.provider 改为 "echo" 离线运行）
export EMBEDDING_API_KEY=sk-xxx
```

`config.json` 主要配置段：

| 段 | 作用 |
|---|---|
| `sources` | 数据源（默认 vllm-ascend + vllm 两个 GitHub 仓库；可加 `type: pdf/markdown/word/excel/image` 业务来源，见 §2.3 / §2.4 / §2.5） |
| `embedding` | 嵌入端点（OpenAI 兼容 `/embeddings`；`echo` 仅离线演示） |
| `storage` | 数据目录（含 `code_root`：版本化代码仓根） |
| `code` | 代码仓快照来源与预存版本列表 |
| `tags` | 标签词典（两级分类，见 §4.7） |
| `sanitize` | 内部数据脱敏白名单（后置，见 §2.4） |

所有路径相对 `data/`，可整体迁移。

> **URL 可写裸地址**：`embedding.base_url` / OCR `ocr_api_base` 可写 `10.0.0.5:8000/v1` 这种
> 裸 `ip:port`（业务侧 vLLM/OCR 服务），配置加载时自动补 `http://` 前缀并告警；https 需显式写全。

> **离线体验**：不想配 token / embedding key 时，用仓库自带的 `config.offline.json`
> （`echo` 嵌入 + 纯 Python 向量后端）+ 模拟数据跑通全链路：
>
> ```bash
> python scripts/seed_demo.py                                          # 写入模拟 GitHub 原始数据
> python scripts/build_kb.py --config config.offline.json --skip-pull    # 离线构建知识库
> python scripts/verify.py --config config.offline.json --version 0.6.1  # 验收查询
> ```
>
> 效果粗糙（echo 嵌入无语义），仅用于无网络环境验证流程；正式使用请用 `config.example.json`。

## 2. 数据采集与构建

### 2.0 快速运维（单指令）

**不需要了解项目内部结构**——日常维护命令：首次部署用 `deploy`，日常 `update`（数据段）
+ `graph`（凌晨停机窗口）。命令自动编排多步并**自动降级**（某一步缺依赖/断网只告警不中断），
`--insecure` / `--github-base` / `--quay-base` 自动透传给所有子步骤。

```bash
# 🟢 首次部署 / 全量重建（拉取+入库 → 建图 → 建FTS → 版本日历 → 配套矩阵 → 代码快照）
python scripts/maintain.py deploy

# 🔄 日常增量更新（拉取+入库 → 新镜像矩阵 → 三仓代码快照；**不停检索服务**）
python scripts/maintain.py update

# 🌙 凌晨停机窗口：单独重建关系图（Kùzu 单写者，需停 serve_api，跑完重启 API）
python scripts/maintain.py graph

# 真实业务环境（SSL 被禁 / 自签证书）：--insecure 放子命令前或后均可
python scripts/maintain.py deploy --insecure --all-code
python scripts/maintain.py --insecure update
```

> **建图与拉取已拆分**（2026-10 业务维护反馈）：`update` = 拉取入库 + **追新镜像全链路**
> （配套矩阵采新 tag → vllm-ascend / vllm 主仓 / 0day fork 三仓代码快照），**不停服务
> 随时可跑**——0day 特殊镜像的代码在官方 GitHub 仓拿不到，update 按矩阵锁定的 fork SHA
> 补齐；关系图（Kùzu 单写者）由 `maintain.py graph` 在凌晨停机窗口单独同步，图更新后
> 重启 API。`deploy` 仍是一次性全量（含建图）。
> **代码快照幂等**：镜像 tag 与锁定 SHA 的内容发布后不变——已存版本的下载/符号索引/
> 跨版本符号表全部跳过（这是"已存在仓检查太长"的历史问题根因，已修复）；提取规则升级
> 后需要强制重建时加 `--reindex`（三个 build_*_snapshots.py 均支持）。

**细分指令**（按需裁剪步骤，省时 / 适配环境）：

| 指令 | 说明 |
|---|---|
| `deploy --skip-graph` | 跳过建图（Kùzu 未装 / 暂时不需要图） |
| `deploy --skip-fts` | 跳过全文索引重建 |
| `deploy --skip-calendar` | 跳过版本日历拉取（省一次 GitHub API 调用） |
| `deploy --skip-matrix` | 跳过配套矩阵拉取（省 quay+GitHub 调用） |
| `deploy --skip-code-snapshots` | 跳过代码快照下载（省时间 / 省网络） |
| `deploy --all-code` | vllm-ascend 代码拉**全部 tag**（默认仅 `config.code.versions` 精选版本；全量数 GB、耗时长，按需开启） |
| `graph` | **单独建图**（凌晨停机窗口：Kùzu 单写者，先停 serve_api，跑完重启加载新图） |
| `deploy/update/graph --insecure` | 跳过 SSL 证书校验（真实业务环境自签证书；子步骤自动继承） |
| `deploy/update/graph --config path` | 指定 config.json（默认项目根自动发现） |
| `build_kb.py --strict` | 严格模式：任一来源失败即退出码 1（CI/严格流水线用） |

> update 的 `--skip` 拆分：`--skip-matrix` 只跳矩阵（fork/vllm 快照依赖矩阵产物会随之
> 拿不到新行）；`--skip-code-snapshots` 一次跳过三仓代码快照；单独补某仓直接跑对应
> `scripts/build_{code,vllm,fork}_snapshots.py`（幂等，只补缺）。

**自动降级**（子步骤失败不中断，只打印告警）：

- **kuzu 未装** → 建图步骤失败告警跳过（KB 入库不受影响，图查询暂不可用）；
- **jieba 未装** → FTS 分词降级为原文（中文词无法独立命中，索引结构不变）；
- **lancedb 加载失败** → 向量后端降级为纯 Python（暴力余弦检索）。**回退会换库**：
  正式库 `data/lancedb/`、回退库 `data/lancedb_py.json`，回退库看不到既有向量；
- **网络不可达** → 版本日历 / 配套矩阵 / 代码快照步骤失败告警跳过；
- **数据拉取与入库按来源隔离**（`build_kb.py`）：某个来源失败只告警并继续处理其余来源。
  本地文件来源（PDF/Markdown/Word/Excel）不依赖网络，GitHub 掉线时它们照常更新；
  已成功入库的来源成果也不会因后面的来源失败而白跑。
  - **退出码**：全部成功 = `0`；部分失败（降级完成）= `0` + 醒目告警（失败来源的既有数据
    不删，下次重跑从 checkpoint 续传）；**所有来源都失败 = `1`**（真正致命，需人工处理）。
  - `--strict`：把"部分失败"也变成退出码 1。

> **弱网 / 代理环境调优**（`config.json` 对应 github source 的字段，都可选）：
> `transport_retries`（默认 6）——连接/超时/TLS 隧道被掐这类**传输错误**的独立重试预算，
> 每次重试都换新连接（丢弃连接池）；`transport_backoff_max_seconds`（默认 60）退避上限；
> `graphql_page_size` / `comments_page_size`（默认 100/100）——GraphQL 单页体积 =
> 页大小 × 每条最多评论数，100×100 时单页可达数十 MB，弱网下调小能显著降低"大传输被掐"
> 的概率（游标与页大小无关，**中途调小不会让断点失效**）。
>
> ⚠️ `--insecure` 只跳过**证书校验**，治不了 `SSLEOFError`（那是隧道被对端/代理掐断，
> 发生在校验之前）。这类故障靠上面的传输重试 + 来源级降级兜住；根治要么换稳定链路，
> 要么用 `--github-base` / `VLLM_KB_GITHUB_BASE` 指到内网镜像。

```bash
# 全量拉取 + 入库（issues/PRs/comments/releases，数小时~数十小时，可 Ctrl-C 中断后重跑续传）
python scripts/build_kb.py

# 小批量试跑
python scripts/build_kb.py --limit 100
```

### 2.1 增量与重建

**常用命令**

| 场景 | 命令 |
|---|---|
| 日常更新（增量入库；GitHub 默认不重拉） | `python scripts/build_kb.py` |
| 拉取 GitHub 社区增量（新增 + 状态同步） | `python scripts/build_kb.py --incremental` |
| 增量模式下已拉条目状态自动同步（open→closed/文本变化） | 同上（`--incremental`，重拉记录日志标「重拉」） |
| **GitHub 补差拉取**（补历史缺失条目，跳过已有） | `python scripts/build_kb.py --pull-missing` |
| **REST 单条补拉**（指定编号，无需 GraphQL token） | `python scripts/build_kb.py --numbers 9749,9750` |
| **强制重拉指定编号**（忽略 raw/checkpoint 已有记录） | `python scripts/build_kb.py --numbers 9749,9750 --force-numbers` |
| 只重入库不拉取 | `python scripts/build_kb.py --skip-pull` |
| 只再生 canonical（不入库，供建图） | `python scripts/build_canonical.py` |
| 换 embedding 模型全量重建 | `python scripts/build_kb.py --rebuild` |

> 日常维护（全量部署 / 增量更新）优先用 §2.0 的 `maintain.py deploy` / `update` 单指令；
> 本表是细分脚本，供需要精确控制时单独调用。

**拉取策略（GitHub）**

首次拉取完成后置 `done`（checkpoint），此后默认**不再拉取**（日志打印
"已拉取完成（done），默认跳过——如需增量请用 `--incremental`"）；中断后重跑同一命令自动
**断点续传**。三种拉取模式互斥：

| 模式 | 行为 | 补得到历史旧条目吗 |
|---|---|---|
| `--incremental`（时间窗增量） | 从 checkpoint 记录的上次增量 `max createdAt` 起；issues 走 GraphQL `filterBy.since` 服务端过滤、PR 走 `UPDATED_AT DESC` 排序，连续 3 页无新增停止，并把窗口推进到本次所见 `max createdAt`（首次增量无历史窗口时从头枚举一次）。**状态同步**：已拉条目若远端 `updatedAt` 更新（open→closed / 重新打开 / 正文评论变化）自动**重拉覆盖** raw 与评论，canonical/入库随 `meta_hash` 刷新（状态变化不重嵌，只刷元数据） | **不能**——只覆盖近期新增/更新；历史单条用 `--numbers --force-numbers` 或 `--pull-missing` |
| `--pull-missing`（补差拉取） | 从头枚举（created desc），**跳过 raw 目录与 checkpoint 中已有的编号，只拉缺失条目**；翻到最新后置 done | **能**——补历史旧条目（如业务数据缺失的单条 PR） |
| `--numbers N1,N2,...`（REST 单条） | 对指定编号先试 `/pulls/{n}`（404 则 `/issues/{n}`）+ 评论落 raw（隐含 missing 语义，**走 REST 不需要 GraphQL token**），**默认跳过已有**编号 | 能——已知缺失编号时最精准 |
| `--numbers … --force-numbers`（REST 强制重拉） | 同上但**忽略 raw/checkpoint 已有记录强制重拉覆盖**（含评论），刷新 checkpoint 时间戳 + `updated_at` 锚点——用于外部已知该编号状态已变（如 open→closed）需立即同步 | 能——已知状态/内容变化时立即同步 |

**全量重拉**（数据刷新 / 补拉旧条目评论）：删除 `data/raw/{source_id}/` 与
`data/checkpoints/{source_id}.json` 后重跑 `build_kb.py`。

**只再生 canonical（`build_canonical.py`）**

提取逻辑（版本 / kind / 组件 / 标签规则）升级后，只需重新生成 `canonical.jsonl` 再跑
`build_graph.py` 建图——**不重嵌向量、不碰 kb.sqlite3**；如需连带重入库（重嵌向量）再跑
`build_kb.py --skip-pull`（与旧参数 `--recanonicalize` 等价）。

**`--rebuild` 高危确认**

会清空向量库 + 删除 kb.sqlite3 后全量重嵌（66K 文档约数小时）。执行前强制确认——
TTY 交互输入 `y/yes`；**非交互环境（agent/CI）必须加 `--yes`**，否则拒绝执行。
canonical / raw / 图 / 审核库不受影响；中断后重跑仍会先清空再重建。

**日常维护节奏（推荐）**

知识库是"离线数据 + 定期刷新"，不需要实时：

```bash
# 白天随时：增量拉取 + 入库（数据段，不停检索服务，FTS/向量即时生效）
python scripts/maintain.py update

# 凌晨停机窗口：单独重建关系图（Kùzu 单写者，停 serve_api → 重建 → 重启 API）
python scripts/maintain.py graph

# 或分步执行（了解细节时）：
# 1. 拉取社区增量 + 增量入库（新增 + 状态同步：已拉条目 open→closed/内容变化
#    自动重拉覆盖；时间窗口见上；中断后重跑同一命令续传）
python scripts/build_kb.py --incremental

# 2. （可选）业务来源有新增文件（data/imports/）时再跑一次不带参数的 build_kb.py
#    （本地来源 pull 资产 + 入库；GitHub 源已 done 会打印跳过）

# 3. 重建图（增量入库后图不含新文档——graph 查询要覆盖新增条目必须重建）：
#    停 serve_api（Kùzu 单写者）→ 重建 → 重启
python scripts/build_graph.py

# 4. 抽查：python skills/vllm-kb/client.py stats   /   graph stats
```

注意：

- **日常维护分两段**（2026-10 业务维护反馈）：`update`=数据段（拉取+入库，不停服务），
  `graph`=建图（凌晨停机窗口，完成提示重启 API）；
- **全量部署用 `python scripts/maintain.py deploy`**（见 §2.0）：内部按序执行
  `build_kb.py`（全量拉取+入库）→ `build_graph.py` → `build_fts.py` → 版本日历 →
  配套矩阵 → 代码快照（vllm-ascend + vllm 主仓）；**自动降级**：建图/FTS/日历/矩阵/
  快照任一步失败只告警不中断（不影响已完成的入库结果）；`--insecure` / `--github-base` /
  `--quay-base` 传给全部子步骤（也可用 `VLLM_KB_INSECURE` 等环境变量）；
  细分指令（`--skip-graph` / `--skip-fts` / `--skip-code-snapshots` 等）见 §2.0 表格；
- **更新前先停检索 API 的只有建图**：`deploy` / `graph`（含建图）要先停 `serve_api`
  （Kùzu 单写者），完成后再启动；`update`（数据段）**不停服务随时可跑**；
- **增量入库后必须重建图**：`build_graph.py` 从 canonical 全量重建，新增文档不会自动进图
  （凌晨停机窗口跑 `maintain.py graph`）；
- **FTS 全文索引不需要日常重建**：增量入库时新文档已实时写入 `chunks_fts`（含 jieba 分词）；
  仅当升级 jieba / 分词规则 / 标签词典（`tags.registry`）后，才跑 `build_fts.py` 让存量文档
  也用新分词（不重嵌向量；旧库无分词列升级时 ingest 会打印提示）；
- **`--incremental` 补不到历史单条**（窗口从上次 `max createdAt` 起、PR 按更新时间排序、
  连续 3 页无新增即停）——缺旧条目时用 §7 的 `backfill_canonical.py`（canonical 层、无需网络）
  或全量重拉（raw 层也要补时）；
- **状态同步是自动的**：`--incremental` 会对已拉但远端更新过的条目（含 open→closed）重拉覆盖，
  无需干预；若已知某个编号状态/内容已变、想立即同步（不等下一轮增量窗口），用
  `--numbers 编号 --force-numbers` 强制重拉单条；

### 2.2 辅助数据构建

```bash
# 版本日历（tag→日期 + 正式/rc 形态）——"修复落地版本"上界在查询期按文档仓库现算
python scripts/build_release_calendar.py --all-repos   # 生成分仓文件 release_calendar.{repo_slug}.json

# 版本化代码仓：vllm-ascend 各版本（config.code.versions 手动维护；--list 先看对比）
python scripts/build_code_snapshots.py --list       # 可用(GitHub tag)/已预存/缺失 对比
python scripts/build_code_snapshots.py             # 按 config.code.versions 增量下载
python scripts/build_code_snapshots.py --all       # 全量预存（所有 tag）

# 版本化代码仓：对应 vllm 主仓版本（companion 矩阵映射，自动跟随）
python scripts/build_vllm_snapshots.py --list      # 先看要拉哪些（已预存/缺失分组）
python scripts/build_vllm_snapshots.py             # 全部下载 + 建索引

# 版本化代码仓：0day fork 仓（hy4/glm5.2 等模型开发分支，独立 forks 命名空间）
python scripts/build_fork_snapshots.py --list      # fork 行状态（模型/锁定 SHA/已预存）
python scripts/build_fork_snapshots.py             # 全部模型快照 + 建索引
python scripts/build_fork_snapshots.py --model hy4 # 只拉指定模型

# 0day 镜像内的 vllm-ascend 插件源码（只拉插件层，~23~101MB；不拉 6GB 整镜像）
python scripts/build_image_snapshots.py --list         # 候选镜像与提取状态
python scripts/build_image_snapshots.py               # 提取全部（镜像 digest 未变则跳过）
python scripts/build_image_snapshots.py --tag glm5.2   # 只提指定镜像（可多次）
python scripts/build_image_snapshots.py --index-only   # 只重建索引（不联网）
python scripts/build_image_snapshots.py --refresh      # 忽略 digest 锚，强制重取

# 同上，离线归档导入（quay 直连困难时：内网预存镜像压缩包直接解析，不触网、
# 不需要 docker——docker load 能导入的 docker save/OCI 归档即可用）
python scripts/build_image_snapshots.py --archive D:\imgs\glm5.2.tar.gz      # 单个（tag=文件名主干）
python scripts/build_image_snapshots.py --archive glm5.2=D:\imgs\x.tar.gz   # 显式指定 img: 键
python scripts/build_image_snapshots.py --archive-dir D:\imgs\               # 批量（*.tar/.tar.gz/.tgz）
python scripts/build_image_snapshots.py --list --archive-dir D:\imgs\        # 只看导入计划

# 组件配套矩阵（vllm-ascend → vllm/cann/pytorch-ascend 自动匹配）
python scripts/build_companion_matrix.py
python scripts/build_companion_matrix.py --refresh-cache   # 强制刷新跨运行缓存（默认按 TTL/不可变语义命中）

# 社区高频信号词（issue 标题 TF-IDF → data/code/signal_words.json，供 agent 判断用，不参与过滤）
python scripts/build_signal_words.py

# FTS5 全文索引重建（jieba 中文分词——可选，不重嵌向量；装 jieba 或升级分词规则后重跑）
python scripts/build_fts.py                     # 从向量库取权威原文重新分词，全量重建 chunks_fts（FTS 缺行回填同此命令）
python scripts/build_fts.py --limit 1000        # 试跑前 N 个 chunk

# 正文 TF-IDF 标签候选导出（jieba——输出文件，人工审阅后手动写入 config.tags.registry）
# 注意：与审核队列的 tag_candidate 是两条独立路径——本脚本只产文件、不自动打标；
# 审核队列候选来自文件名/标题提取（自动、可一键采纳打标），见 §3.2。
python scripts/build_tag_candidates.py          # 业务文档（doc_*）正文候选 → data/tag_candidates_manual.json
python scripts/build_tag_candidates.py --include-github   # 也处理 github issue/PR（默认仅业务文档）
```

**FTS5 中文分词说明（`jieba` 可选依赖）**：SQLite FTS5 默认把连续中文整段当一个 token
（"超时"无法命中"超时排查"）。安装 `jieba`（`pip install jieba`，离线 wheel 可装）后，
入库自动对 chunk 文本分词写入 FTS 索引、查询侧同步分词——中文词可独立命中；
**向量库不受影响**（原文嵌入，无需重嵌）。升级 jieba / 分词规则后跑一次
`scripts/build_fts.py` 重建索引即可。未装 jieba 时 FTS 行为与旧版一致（无需重建）。

**真实业务环境（SSL 被禁 / 镜像源）**：以上联网脚本均支持 `--insecure`（跳过 SSL 校验）与镜像源参数，
也可用环境变量统一配置（多脚本共享）：

```bash
export VLLM_KB_INSECURE=1                        # 跳过 SSL 证书校验
export VLLM_KB_GITHUB_BASE=http://<镜像>/api/v3  # GitHub API 镜像
export VLLM_KB_QUAY_BASE=http://<镜像>           # quay 镜像
export VLLM_KB_CODE_BASE=http://<镜像>           # codeload 源码 zip 镜像
python scripts/build_companion_matrix.py         # 全部脚本自动走业务环境配置
```

**配套矩阵自动匹配规则**（`build_companion_matrix.py`）：

| 字段 | 来源与回退顺序 |
|---|---|
| `vllm` | 镜像 Env 的 `VLLM_TAG`（构建时锁定，最可靠）> 镜像 buildkit history（fork 仓 `VLLM_REPO/VLLM_REF/VLLM_BASE`）> GitHub release 说明 > 版本号启发式 |
| `image_created` | quay tag 最后推送时间（`last_modified` → ISO-8601 UTC，如 `2026-07-27T15:39:42Z`）；与 `image_digest` 同源（同取代表 tag） |
| `vllm_commit` / `vllm_commit_date` | 镜像锁定的 vllm 版本 tag → GitHub 解析出的 commit SHA 与提交日期（`GET /repos/{repo}/commits/tags/{tag}`）；**未配置 `GITHUB_TOKEN` 且缓存未命中时跳过**（45+ 个 tag 会打满未认证限流），留空待带 token 重跑 |
| fork 行（0day 模型镜像，如 `hy4`/`glm5.2`） | 除上表字段外自动带 `vllm_repo`（fork 仓）、`vllm_ref`（分支）、`vllm_base`（基线版本）、`vllm_sha`（**clone 层扫描固化的锁定 commit**）、`image_digest`（digest 锚定：镜像未重推则跳过重扫）、`vllm_commit_date`（该 fork SHA 的提交日期）；配套代码快照用 `build_fork_snapshots.py` 按锁定 SHA 拉取 |
| `cann` | 镜像 Env 的 `cann-X.Y.Z` 路径；缺失时按**基础版本号**回退同系列其他形态（如 `v0.13.0rc1` 用 `0.13.0` 系列的 cann），同系列也没有则留空人工看护 |
| `pytorch-ascend`（PTA）/ `pytorch`（torch） | 对应 tag 的 `requirements.txt`（`torch-npu==X.Y.Z.postN` / `torch==X.Y.Z`）——**本地快照 zip 优先**（零网络），快照未预存的 tag 走 GitHub API 兜底；0day 模型（非版本 tag）无 requirements 时回退**同 minor 系列已发布 tag**（bailing 0.19.0 → 0.19.x 系列），系列内无 release 或未发布版本留空人工 |

> **SOC 型号已不进矩阵**（2026-10 业务维护反馈）：镜像只与**机器型号**（`-a2`/`-a3`/`-a5`
> 平台后缀，见 `PLATFORM_SUFFIXES`）相关，SOC 对同一机器型号不构成区分维度；且
> `SOC_VERSION` env 只出现在部分镜像——单点值写进 notes 反而误导（看似"该镜像仅支持
> 此 SOC"），故不再采集。

> **"这个镜像对应哪个 commit"**：官方仓镜像（如 `kimi-k3`）看 `vllm_commit` + `vllm_commit_date`；
> fork 镜像（`hy4`/`glm5.2`）看 `vllm_sha` + `vllm_commit_date`（锁的是 **fork 仓**的代码）。
> **注意**：镜像里的 **vllm-ascend 插件代码是 `COPY .` 拷进去的**，镜像内没有版本/commit 痕迹，
> 任何脚本都无法锁定它——需要插件代码审查时，只能按 `vllm` 基线版本对应的官方 tag 快照近似
> （`data/code/snapshots/{version}/`），或由镜像构建方在构建时写入版本信息。
> `matrix` 命令可查看全部字段；`companion` 只返回版本配套关系。

写回前**版本号正则校验**：非法值（`latest`、带前缀等）置空 + 告警，不污染矩阵；
`vllm_commit`/`vllm_sha` 校验 40 位 hex，`image_created`/`vllm_commit_date` 校验 ISO-8601。

**跨运行缓存**（`data/cache/`，防限流与重复下载；`--refresh-cache` 强制刷新）：

| 文件 | 内容 | 失效策略 |
|---|---|---|
| `fork_sha.json` | clone 层 digest → 锁定 SHA | 层不可变，**永久有效**（首次扫描后不再重下 ~75MB 层） |
| `github_releases.json` | release 说明（按 API 前缀键控） | TTL 7 天（兜底新 release） |
| `github_requirements.json` | requirements 兜底结果（含 404） | tag 内容不可变，**永久有效** |
| `github_tag_commits.json` | tag → commit SHA + 提交日期（key `{repo}@{tag}`） | tag 指向的 commit 不可变，**永久有效** |

只有完整/确定性结果才落盘（翻页中途失败、网络/限流失败不缓存，下次自动重试）。
缓存全命中时矩阵生成对 GitHub API 的请求为 **0**（未认证限流 60 次/小时不再是约束）。

### 2.3 业务来源导入（PDF 手册 / Markdown 文档）—— 完整实操

**步骤 1：放置文件（图片随文档一起本地化）**

```
data/imports/pdf/          ← 放 PDF 手册（操作手册/接口指南：硬件排查命令、错误码参考）
data/imports/md/           ← 放 Markdown（案例/架构说明/经验总结）
├── HCCL超时排查.md
├── GLM5.1-PD分离复盘.md
└── images/                ← 建议图片与文档同放 imports（子目录按案例/主题组织）
    └── hccl-timeout/
        ├── 01-报错截图.png
        └── 02-npu-smi输出.png
```

**图片本地化原则**（数据源接入时完成）：文档引用的截图在接入时就存进导入目录
（png/jpg/webp/gif），md 里用**相对路径**引用（`![报错](images/hccl-timeout/01-报错截图.png)`，
以 md 所在目录为基准，子目录任意）——入库时自动资产化进 `data/assets/images/`
（内容寻址 + OCR 注入）。**远程图链不下载**：引用 `http(s)://…` 的图入库时只留
`[图片]` 占位并记 remote evidence，图不进库——所以"把图片放进导入目录"是接入动作，
不是构建动作。

**步骤 2：启用对应 source**（`config.json` 的 `sources`，改 `enabled: true`；`path` 可指向其他目录）

```jsonc
{"id": "manuals", "type": "pdf", "path": "data/imports/pdf", "enabled": true},
{"id": "wiki",    "type": "markdown", "path": "data/imports/md",
 "title_pattern": "^#\\s+(.+)", "enabled": true}
```

**步骤 3：配置 API key**（审核工作台 `http://127.0.0.1:8010` → API 配置 → 编辑保存；
或环境变量 `EMBEDDING_API_KEY` / `GITHUB_TOKEN`；key 存 `data/secrets.local.json`，
任何入口 `AppConfig.load` 自动注入——**写入路径（build_kb）也能读到**）

**步骤 4：导入**

```bash
# 直接跑（pull 幂等：pdf/md 复制到资产层 + 解析 + 入库 + 嵌入）
python scripts/build_kb.py

# --skip-pull 也可用（仅重跑入库）：canonicalize 自己也会把原件落资产层并注册
# （见"文件型来源的统一约定"），所以不会出现"资产层空 → canonicalize 无文件"。
```

- 无 `GITHUB_TOKEN` 时 GitHub source 拉取会报错——**临时把 config 里 github source 的
  `enabled` 改为 `false`** 跑导入（只处理本地来源），完事恢复 `true`（保留标记注意改回）；
- 断点续传：中途 Ctrl-C 后重跑同一命令即可。

> **PDF 的文档集由 `imports/pdf` 决定**（与 md/word 同构）：`imports/pdf` 里扫到的 PDF 就是
> 全部文档，资产层只作回退与诊断。若资产层还有"导入路径里没有"的 PDF，构建日志会打印
> `⚠ 资产层有 N 个族不在导入路径中，已不再纳入 canonical`——**必须处理**（复制回
> `imports/pdf`，或到审核台「文档管理」删除），否则那几篇会**静默停在旧内容**
> （`ingest` 不删文档）。历史版本从"只 pull 不建 imports"升级过来时尤其容易碰到：
>
> ```powershell
> Get-ChildItem data\assets\pdf\*.pdf |
>   Where-Object { -not (Test-Path "data\imports\pdf\$($_.Name)") } |
>   Copy-Item -Destination data\imports\pdf\
> ```
>
> 文件名不变 ⇒ `source_id`（`pdf:<文件名去扩展名>`）不变 ⇒ 原地更新，**不会产生重复文档、
> 不丢标签与补标**。

**步骤 5：验证**

```bash
python skills/vllm-kb/client.py search "hccn_tool 错误码 排查"   # 语义检索命中手册（sim 高）
python skills/vllm-kb/client.py doc pdf:<文件名去扩展名>          # 整篇读取（如 pdf:xxx 接口参考 04）
python skills/vllm-kb/client.py stats                           # chunks 数应增长
```

**步骤 6：产物与质量规则**

```
data/assets/pdf/<name>.pdf               # 原始文件（不可变层，sha256；路径不进库，以 asset_id 标识）
data/parsed/pdf/<asset_id>.tables.json   # 结构化表格（错误码表/命令表，asset_id 命名）
data/raw/canonical.jsonl                 # canonical 追加（verification/tags 等元数据）
```

- PDF 表格转 Markdown 表格拼入正文（FTS 可检索）+ 另存结构化 JSON；
- **自动标签（两级分类）**：入库时从文件名 + 内部标题确定性提取——**主题/领域类**（domain，
  如 `npu-smi`/`Atlas`，= 这是什么领域的知识）与**具体作用类**（purpose，如 `命令参考`/`错误码表`，
  = 文档能帮我做什么），与 `config.json` 的 `tags.registry` 词典子串命中为准；
  未收录强候选进审核队列 `tag_candidate`，采纳后入词典并即时打标；
- 验证状态默认：**PDF 手册 = `expert`**、**Markdown = `unverified`**（审核工作台补标，
  认证后立即生效——见 §3.2 补标回写）；
- 检索结果显示 `验证=expert/unverified`；embedding key 有效时语义检索（向量）生效，
  无效时自动降级全文检索（`search` 仍可用）；
- **路径不进库（安全约束）**：canonical / 检索库不含服务器路径（资产以 asset_id 标识），
  `/doc` 等 API 返回的 extra 经白名单清理；管理员侧路径仅存审核库（asset_registry）。

**PDF 解析缓存（性能，增量入库 / recanonicalize 通用）**：PyMuPDF 逐页提取耗时较长
（241 页手册约 7s/篇），解析中间产物（文字层 + 表格 + 页数）按资产 sha256 缓存到
`data/parsed/pdf/<asset_id>.extract.json`——**资产未变时直接复用缓存**（进度行标注
"缓存命中"，毫秒级），仅标签/元数据提取每次重算（词典/提取规则升级**无需清缓存**即生效）。
缓存里带 **schema 位**（`sources._PDF_EXTRACT_SCHEMA`）：`_extract_pdf` 的**提取逻辑**升级
（例如新增表格提取）时把这个常量 +1，旧缓存即整体失效并自动重解析——否则旧缓存的 sha256
仍然匹配，会**静默沿用旧解析结果**。
**强制重新解析**（如 PyMuPDF 升级后想重新提取文字层）：删除 `data/parsed/pdf/` 目录即可，
资产层（`data/assets/`）与 kb 数据不受影响。

**步骤 7：重启检索服务（改过 key / config 后）**

```bash
# 先停掉旧 serve_api（8000 端口），再：
python scripts/serve_api.py          # http://127.0.0.1:8000
python skills/vllm-kb/client.py health   # chunks 数与预期一致
```

**文件型来源的统一约定（PDF / Markdown / Word / Excel）**

四种来源都是"一个文件一篇文档"，共用同一套发现/身份/分块规则（实现在
`vllm_kb/doc_kinds.py` + `vllm_kb/source_files.py`，说明见数据流 §2.2a）：

| 规则 | 说明 |
|---|---|
| **放文件的位置** | config 里该 source 的 `path`（`data/imports/<你的目录>`），**目录树会被保留**——同名文件放不同子目录不会互相覆盖；`path` 也可以直接指向**单个文件**（Excel 登记表常用） |
| **导入路径优先** | 只要 `path` 下扫到匹配文件，就**不再看**资产层副本。想下线一篇文档 → 从 `path` 删掉并清掉 `assets/<种类>/` 同族副本，再到审核台「文档管理」删记录（完整影响面见下方「删除一篇文档：三条路径 × 数据影响面」） |
| **回退资产层** | `path` 不存在/扫不到文件时，回退 `data/assets/<种类>/` 扁平副本，并按**版本族**收敛（`case.md` 与 `case.<sha12>.md` 是同篇不同版本，只取 mtime 最新一份） |
| **文档身份** | `source_id = <前缀>:<文件名去后缀>`（`pdf:` / `md:` / `word:` / `excel:`）。**改内容不改身份**（标签/补标/遥测都跟着走）；只有同名冲突时才追加**相对路径**指纹 `--<sha8>`，唯一文件名**不会**被加后缀 → 存量库不需要重灌 |
| **文档标题** | **原始文件名 stem 优先**（2026 定版）——文件名由工程师/官方下载命名，信息量高于解析提取的标题（华为手册 PDF 封面只有产品名，多篇手册互相重名）。stem 为无意义名（纯数字/纯 hex hash/纯符号/过短）时回退提取标题（爬虫件的 `<title>` 等）。改标题规则后受影响文档会在下次构建**自动重嵌**（title 参与嵌入输入，一次性成本）。HTML 来源同规则（`html:` 身份），Excel 文件级本就是文件名 |
| **章节切分** | PDF 按编号标题（`2.34 xxx`）、Markdown/Word/Excel 按 `#` 标题切分，chunk 带 `section` 字段；新增来源时在声明表里填策略名即可，`chunking.py` 不用改 |
| **偏离告警** | 导入路径非空、但资产层还有"导入路径里没有"的族时，构建日志会打印 `⚠ 资产层有 N 个族不在导入路径中，已不再纳入 canonical`。**看到这条就要处理**：这些文档不会消失也不会报错，只是**永远停在旧内容**（`ingest` 不删文档，属于静默偏离） |
| **删除策略 deletion_policy** | 来源条目可选 `keep`（**默认**，现有增量语义：发现层只增改不删）/ `mirror`（**本地原始数据优先**，审计驱动可信路径）：快照对账（`data/source_state/<来源id>.json`），**连续 2 次构建缺席 → 自动回撤删除**（备份 doc_row → 四层删 → 墓碑 → "待实际删除"审核项，可撤回）；文件重新出现 → 自动解除墓碑恢复入库资格；**首轮全量对账**：库里该 kind 前缀的存量文档若缺席也计入宽限——开启 mirror 后 2~3 次构建内清掉历史滞留（含源文件早已缺失的）。前提：**同 kind 只能有这一个 mirror 来源**（多来源共享前缀会告警）。设计见 `docs/DESIGN-audited-git-source.md`；痕迹见文档管理页底部"已回撤"区 |
| **新增一种文件型来源** | ① 在 `doc_kinds.KINDS` 加一条 `DocKind`；② 来源类挂 `kind = BY_SOURCE["..."]`；③ 文件发现改走 `_discover_for(self, self.kind, self.import_dir)`。分块章节切分与审核台「文档管理」收录范围会自动跟上（护栏测试：`tests/test_doc_kinds.py`） |

**删除一篇文档：三条路径 × 数据影响面（2026-10 逐层核实；同日引入删除墓碑，见下）**

先记住两条铁律与一个例外，影响面就好推演：
1. **canonical.jsonl 永不自动删行**——增量构建只追加/覆盖（`upsert_unified_canonical`），`--rebuild` 只读不写；canonical 是 `--rebuild` 与 `build_graph` 的**唯一事实源**；
2. **`ingest` 永不删文档**——kb 里的记录只会"静默偏离"（不再刷新），人工删除是唯一删除方式；
3. **例外（删除墓碑，2026-10）**：审核台删除（单/批）与队列"标记删除"会登记 `deleted_docs` 墓碑（review.sqlite3）——`ingest` / canonical upsert / `build_graph` 三处**跳过墓碑 sid**，任何构建路径（增量、资产层回退、GitHub 采集缓存、`--rebuild`）都**不得复活**已删文档；**撤回删除解除墓碑**，之后可重新入库。墓碑使"审核台删除"成为一等删除——下面矩阵中所有"复活"路径对墓碑 sid 一律失效。

**路径 ①：只删磁盘源文件（`data/imports/<kind>/…`）——影响面≈零**

| 数据层 | 影响 |
|---|---|
| imports 文件 | 消失 |
| canonical 行 | **保留**（铁律 1）→ `--rebuild` 本会复活，**墓碑拦截**（审核台删过记录的 sid） |
| kb docs / chunks_fts / chunks_meta / 向量 | **保留**（铁律 2），内容停留在最后一次入库（静默偏离，唯一提示是 stray 告警） |
| Kùzu 图 | 重建后节点仍在（canonical 有行） |
| doc_tags / doc_verification 覆盖层 | 保留 |
| tag_candidate 队列 | 保留且引用有效 |
| 资产层副本 / asset_registry | 保留（副本会成为 stray 告警对象） |
| 词典 / 设备词表 | 无关 |

**路径 ②：删统一 canonical 行（手工编辑 `data/raw/canonical.jsonl`）——`--rebuild` 后真正消失**

| 数据层 | 影响 |
|---|---|
| canonical 行 | 消失 |
| kb 四层（docs/chunks_fts/chunks_meta/向量） | 增量构建不动（静默偏离）；**`--rebuild` 后真正删除** |
| Kùzu 图 | `build_graph.py` 重建后节点消失 |
| doc_tags / doc_verification 覆盖层 | `--rebuild` **先备份后全量恢复**——指向已消失文档的**孤儿行残留**（无害：该 sid 不存在则不被读取；同 sid 重新入库时人工标签/认证**自动恢复生效**——这是特性：删库重灌不丢人工成果） |
| tag_candidate 队列 | **不受影响但会在下次 seed（审核台启动）时自动校正**（2026-10）：全悬空候选自动忽略（result 记原因）、部分悬空刷新引用、词典已收录的忽略——任何删库路径后队列自愈 |
| 资产层副本 / asset_registry | 保留（孤儿映射，无害；重新入库时幂等复用） |
| 词典 / 设备词表 | 无关（config 级事实源） |

**路径 ③：审核台「文档管理」删除（数据库四层，推荐的人工路径）**

| 数据层 | 影响 |
|---|---|
| kb 四层 | **立即删除** + 登记**删除墓碑**（`deleted_docs`，一等删除） |
| 磁盘源文件 | **不动**——imports 文件即使还在，**墓碑 sid 也不会被增量入库复活**（撤回删除解除墓碑后才可重新进来；人工标签/认证覆盖层残留同 sid 重进时自动恢复） |
| canonical 行 | **保留** → `--rebuild` 本会复活，**墓碑拦截**（ingest / canonical upsert / build_graph 三处跳过） |
| tag_candidate 队列 | **联动修剪**（2026-10 新增）：pending 项引用移除该 sid；全部提及文档删光的候选自动忽略（result 记原因）；即便走了旧版本/其他路径漏清，下次 seed（审核台启动）也会自动校正 |
| doc_tags / doc_verification 覆盖层 | 残留（孤儿，无害；同 sid 重进时恢复生效） |
| 资产层副本 / asset_registry / 词典 | 保留 |

**彻底下线一篇文档**（2026-10 起大幅简化，墓碑兜底所有"复活"路径）：
1. **必要**：审核台「文档管理」删记录（kb 四层立即删 + 候选联动清理 + 登记墓碑）——这一步做完，任何构建路径都不会再复活它；
2. 建议：`data/imports/<kind>/` 删源文件、清 `data/assets/<kind>/` 同族副本——防 stray 告警与磁盘占用（不影响复活与否）；
3. canonical 行**无需手工删**（墓碑在 ingest/upsert/建图三处过滤，旧行成为惰性数据）；
4. （可选）`build_graph.py` 重建图，节点消失（重建时同样跳过墓碑 sid）。

反向说明：**"重置单篇文档"**（删库记录让它重新干净入库）需要先**撤回**墓碑——
审核队列「待实际删除」列表里**撤回删除**（恢复行 + 解除墓碑），或走正常入库流重新覆盖。
只删库记录不撤回墓碑的话，文档不会被重新入库（一等删除语义）。

**Markdown 图片处理（随 md 一起入库）**

- md 正文里的图片引用自动收集到 `data/assets/images/`，**正文引用改为不透明占位**（不暴露路径），
  `extra.evidence` 记录 asset_id/sha256（管理员侧经 asset_registry 找回原图）。
  解析规则在 `vllm_kb/md_images.py`，**支持的形态**：

  | 形态 | 示例 | 结果 |
  |---|---|---|
  | 相对路径（以 md 所在目录为基准） | `![a](shot.png)`、`![a](imgs/sub.png)`、`![a](../up/x.png)` | 收集（`local`） |
  | 绝对路径 / `file://` | `![a](D:/x/shot.png)`、`![a](file:///D:/x/shot.png)` | 收集（`local`） |
  | 目标含空格（非标准但常见） | `![a](my file.png)` | 收集（`local`） |
  | 尖括号包裹 | `![a](<my file.png>)` | 收集（`local`） |
  | 目标含括号 | `![a](img(1).png)` | 收集（`local`） |
  | 带 title / 跨行目标 | `![a](shot.png "标题")`、`![a](\n shot.png\n)` | 收集（`local`） |
  | 引用式 | `![a][r1]` + `[r1]: shot.png`；`![a][]`；`![a]`（有同名定义时） | 收集（`local`），定义行目标同时去路径 |
  | HTML | `<img src="shot.png" alt="x" width="200">`（单/双/无引号） | 收集（`local`） |
  | base64 内嵌 | `![a](data:image/png;base64,…)` | 落盘后收集（`base64`） |
  | 网络 URL | `![a](https://…)` | 标记 `remote`，**不下载**，只占位 |
  | 文件不存在 / 目录 / 坏 base64 / 未闭合 | — | 标记 `unresolved`，只占位 |

- **硬约束（对任意输入成立）**：正文与 canonical **绝不留存图片引用里的原始目标**——
  上述未识别或未闭合的写法也一律占位，避免服务器目录结构随正文进检索库
  （回归护栏：`tests/test_md_images.py::TestNoPathLeak`）；
- **代码区不处理**：``` / ~~~ 围栏块与行内 `` `code` `` 内的图片语法**原样保留**
  （否则会破坏代码示例并产生虚假 `unresolved` 记录）；未闭合的围栏块按 CommonMark 吞到文末；
  4 空格缩进代码块与列表续行无法可靠区分，故不识别（其中图片语法按正文处理）；
- 图片的 OCR 由 image source 完成（见下）；**Markdown 正文引用的图片在 `canonicalize` 时同步 OCR**，
  高置信文本注入占位符之后（见"低置信度图片不进正文"）。
  OCR 缓存按图片 sha256 幂等：多处引用同一张图只识别一次，但每处引用各自注入文本；
- **内嵌 base64 图按内容寻址命名** `img_<sha16>.<ext>`：跨文档内嵌同一张图只落一份（自动去重），
  且命名不含 md 文件名 → 同名 md 也不会互相覆盖；
- **同名 md 自动消歧**：`md:<stem>` 会互相覆盖（入库用 `INSERT OR REPLACE`），所以检测到 stem 冲突时
  冲突集合**整体**改用 `md:<stem>--<sha8>`——`sha8` 是**相对路径**的指纹（不是内容 sha：
  内容每次编辑都变会让 id 漂移、旧文档变孤儿），且不可读（`source_id` 会出现在 `/search` 的
  `doc_id`、`/doc/{id}` 与遥测库里，不能暴露目录名）。**唯一 stem 的文档 id 保持 `md:<stem>` 不变**，
  所以存量库不需要重灌。冲突时构建日志会打印告警；
- **回退模式**：`imports/md` 不存在或扫不到 md 时，回退到 `assets/md` 副本（`pull()` 的产物）。
  此时相对路径**已失锚**（assets 是扁平副本），只能按**文件名**到 `assets/images` 尽力反查，
  未命中的照常占位——所以**回退模式同样满足"正文不含路径"**。构建日志会打印显式告警，
  且该篇 `extra.quality.source_mode = "assets_fallback"`（正常为 `"imports"`），
  `extra.quality.images_unresolved` 记录未解析的图片数，可据此筛出需要人工看的文档；
  * **版本族收敛**：编辑过的 md 在资产层会留下多个版本（`case.md` 原始副本永久保留 +
    每次内容变化一份 `case.<sha12>.md`）。回退模式按"版本族"收敛，**每族只取最新一份**
    （mtime 最大），且 `source_id` 取**族名**（`md:case`）而非带 sha 的文件名——
    否则收敛到副本时 id 会漂移成 `md:case.<sha12>`，等于换了篇文档（审核状态/标签丢失）；
  * 因此**删掉 `imports/md/case.md` 后重跑，文档会按最新副本"复活"**（这是回退模式的语义）。
    要彻底下线一篇文档，须同时清掉 `assets/md/` 里的同族副本，再到审核台"文档管理"删除记录。
- **图片资产注册**：每张落盘的图片都会注册到 `asset_registry`（`asset_id → rel_path`，
  `source_type=image`）——`MarkdownSource` 注册它收集到的图，`ImageSource` 注册
  `assets/images/` 下的**全部**图（含手工投放、含 `ocr_provider: none`）。审核工作台据此
  反查路径并**直接预览原图**（`/assets` 静态挂载）。
  注册是**内容寻址**的（`asset_id = sha256 前 16 位`），所以内容相同的多个副本只占一行，
  反查到的是"一份同内容的副本"；未解析（`unresolved`）的引用不注册。

> **实践建议**：图片与 md 放同目录或相对子目录，文件名避免空格与括号（虽然已兼容，但最稳）；
> alt 写有意义的内容（`[图片:alt]` 会进正文，对检索有帮助）；引用式定义行会被去掉目标，
> 若同一标签还被普通链接 `[文字][label]` 使用，该链接会失去目标（正文文字保留）；
> **不要删除 `imports/`**——回退模式只能尽力而为，完整图片/OCR 需要 imports 在场。

**图片 OCR（签名导向，provider 可插拔）**

```bash
# config 的 images source 选择 OCR 方式（ocr_provider）：
#   "ask"（默认）: 配了 ocr_api_base → 走 API；没配 → 询问"是否本地运行（paddle）"，
#                  否定（或非交互终端）→ 跳过 OCR（导入不受阻）
#   "api":        HTTP OCR 服务（强制 API 场景推荐）——失败时同样询问本地/跳过
#   "paddle":     本地 PaddleOCR（明确选择，不询问；未安装 → 提示并跳过）
#   "none":       明确跳过
# ocr_min_confidence: 进正文阈值（默认 0.6，可在审核工作台 API 配置中心改）
python scripts/build_kb.py --skip-pull        # 触发 canonicalize（md 图片 + image source：OCR + 签名提取）

# 产物：data/parsed/images/<name>.ocr.json
#   {image, sha256, provider, mode, model, engine_fingerprint, text, confidence,
#    confidence_source, anomaly, min_confidence, text_included, signatures}
```

- **OCR API 两种调用模式（`ocr_api_mode`，默认 custom）**：
  - `custom`：`POST {ocr_api_base}/ocr`，body `{"image": "<base64>", "filename": "x.png", "model": "<可选>"}`，
    响应 `{"text": "...", "confidence": 0.93}`——适用于自研 FastAPI 包装 paddleocr 等；
  - `openai`：OpenAI 兼容接口（vllm-ascend 部署的 paddleocr-vl / siliconflow 的 DeepSeek-OCR 等）：
    `POST {base}/chat/completions`，`model=ocr_api_model`（**必填**），图片以 data URI 内联；
- **openai 模式调用参数（2026-10 paddleocr-vl-1.6 业务实测定版）**：
  - `ocr_temperature` **默认 0**：vLLM 默认温度 1 会带崩 OCR 专用模型输出（大量 LOC 布局标签/乱码）；
  - `ocr_api_prompt` 默认**模型原生极简指令 "OCR"**：长提示词对 OCR 专用模型是负担；
    通用 VL 模型可换回自报提示词（要求末行输出 `CONFIDENCE: <0~1 小数>`，参考 `vllm_kb/ocr.py` 的 `_CONFIDENCE_PROMPT`）；
  - `ocr_max_tokens` 默认 4096（必须显式——vLLM 服务端默认截断会切掉输出尾部）；
  - **置信度走 logprobs**（`confidence_source=logprobs`，= exp(非特殊 token mean logprob)）：
    paddleocr-vl 类 OCR 专用模型**不自报置信度**，logprobs 是唯一模型侧来源；特殊 token
    （`<|...|>`）不计入均值（会把分数拉低到无意义）；无 logprobs 且无自报行 →
    `anomaly=no_confidence` 进人工核对；输出被截断（finish_reason=length）→ `anomaly=truncated`；
- key 可选（`ocr_api_key` 或环境变量 `OCR_API_KEY`）；`ocr_api_model` 可选透传（openai 模式必填）；
- **置信度来源（单一来源原则）**：`custom` = 服务端返回（`confidence_source=service`）、
  `paddle` = 引擎逐行平均（`engine`）、`openai` = **logprobs 优先，自报回退**（`logprobs`/`model`）——
  不做多源融合，也不做启发式二次打分（混合来源会让回溯时无法判断分数出处；中文 OCR 主要错误是形近字，
  产出仍是合法可打印字符，纯文本启发式识别不了）；
  自报异常（缺失 / 不可解析 / 越界 / 不在末行 / 正文为空却高置信）→ `confidence=null` + `anomaly` 标记，
  交人工审核，**不回退**到启发式评分；
- **预处理不做**（业务实测）：x3 放大 / CLAHE 对模糊小字无提升且引入乱码——插值放大不带
  新信息；优化走源头（截图分辨率规范）与几何矫正（见下）；
- **OCR 基准集与评估**（`scripts/eval_ocr_bench.py`）：任何 OCR 优化先过基准集量化，
  不要靠"感觉有效"。四步：
  ```bash
  python scripts/eval_ocr_bench.py --pick-from-assets   # ① 从资产层按置信度分桶挑图+labels 草稿（expect 预填 OCR 文本，人工校对）
  python scripts/eval_ocr_bench.py --gen-transforms     # ② 程序生成歪斜(3/7/12°)/透视变体（期望文本=原图，零标注成本）
  # ③ 人工校对 <data_root>/ocr_bench/labels.json 的 expect（重点 blurry 桶）
  python scripts/eval_ocr_bench.py --run                # ④ 跑评估 → report.json（分场景 avg_score/pass_rate/avg_confidence）
  python scripts/eval_ocr_bench.py --run --baseline <旧 report.json>   # ⑤ 改参后对比（avg_score delta）
  ```
  场景配比建议 30 张：normal ×8 / blurry ×8 / tilt ×7 / persp ×7（tilt/persp 由清晰图程序变换生成）。
  难场景优化路径（按性价比）：①采集规范（文字像素高 ≥16px、PNG 免 JPEG 伪影）；
  ②几何矫正前置（方向分类 → det 框估角旋转 → 四点透视变换）；③双引擎交叉
  （logprobs <0.85 的图用 PP-OCRv5 det+rec 复跑，编辑距离一致则采纳）；
  ④救不回的走低置信人工审核兜底。
- **embedding 强制 API**（本地 embedding 不做，部署复杂）：`embedding.base_url` 指向 OpenAI 兼容端点
  （可指向其他服务器的 vLLM 部署）；`echo` 仅离线演示（效果粗糙）；
- 连通性测试：审核工作台 API 配置中心对 embedding / OCR 均提供"测试连通"（OCR 用内置测试图走真实识别链路）；
- **幂等键 = 图片 sha256 + 引擎指纹**（`provider|mode|model|提示词版本`）：图片没变就跳过；
  换 OCR 模型 / 换调用模式 → 指纹变化 → 自动重算（不会"图片没变、引擎变了，结果还是旧的"）；
  **阈值 `ocr_min_confidence` 不进指纹**——调阈值只按新阈值重判定 `text_included`，不重跑 OCR；
- 只提取错误签名（算子/错误码/模型/版本）写入 `signatures`；
- 与 md 文档的 `evidence` 联动：`extra.evidence[].ocr` 带 `confidence/confidence_source/anomaly/
  text_included/signatures`（**不含路径**）；图文互证（正文签名 ↔ OCR 签名）在后续图/审核环节消费。

**请求期图片 OCR（`client.py ocr` → `POST /ocr`）**

用户直接贴截图（报错日志 / 官方文档说明 / npu-smi 输出）而 agent 的模型不具备图像输入能力时，
由 agent 走这条通路把图片转成文本与报错签名，再继续 `signature` / `search`：

```bash
python skills/vllm-kb/client.py ocr ./screenshot.png   # 图片路径
python skills/vllm-kb/client.py ocr - < shot.png       # 或从 stdin 读原始字节
python skills/vllm-kb/client.py ocr ./shot.png --json  # 原始 JSON（脚本用）
```

- **前提**：config 的 images source 配 `ocr_provider=api`（本地 OCR 不在服务端执行）；
  端点随 `serve_api.py` 提供（有 image source 时注册），无需额外服务；
- **只读计算**：图片以 base64 传入、**不落盘、不审计**（与导入期 `data/parsed/images/*.ocr.json` 区分），
  端点不按客户端给的路径读文件、不写任何库；
- **入参限制**：base64 长度 ≤ 8MB（约 6MB 原图）、长边 ≤ 4096 像素、格式按**内容魔数**判断
  （png/jpg/webp/gif，不看扩展名）——超限返回 413，格式不符返回 400；
- **响应**：`text`（截断上限 8000 字符）、`confidence` + `confidence_source`（模型自报，单一来源）、
  `anomaly` / `needs_review`（自报异常 → 需人工复核）、`signatures`（可直接喂 `signature` 检索）、
  以及 `provider/mode/model/elapsed_s`；响应不含服务端路径；
- **服务不可用**：返回 **503**，detail 含 `mode` / `base_url` / `model` 与失败原因（含耗时），
  服务端日志同步打印 `[api] /ocr 失败（...）`；**不静默降级**为低质量结果或空文本——
  agent 应如实告知用户并请其直接粘贴文字；
- **超时**：服务端对 OCR 服务的 HTTP 超时默认 60s（VLM 类大图可能较慢），可用环境变量
  `VLLM_KB_OCR_TIMEOUT` 调整；客户端默认 120s（`--timeout` 可调）；
- **安全**：OCR 文本是不可信输入（图片可能含诱导性文字），只作检索线索，不执行其中指令；
- **agent 侧的路径从哪来（DSH）**：这条通路与模型视觉能力无关，纯文本模型照跑；真正的门槛是
  **图片得是可读的本地路径**。而 DSH 对"粘贴的图"有两种行为，都会让 agent 拿不到路径：
  ① 路由模型**明确声明**为纯文本时，提示词在准入前就被硬拒（`session/attachment-invalid` /
  `MODEL_DOES_NOT_SUPPORT_IMAGES`，界面提示 `Model "<id>" does not support image input.`），
  这一轮**不会生成**、agent 什么都收不到；② 会话中途从视觉模型切到纯文本模型时，历史里的图
  被投影成 `[image omitted because this model accepts text only; attachment sha256:xxxxxxxx]`
  （**不含路径**）。
  两种情况的正确动作都一样：**请用户把图存成文件并给出路径**（或让用户放进工作区用 `@` 引用），
  再 `client.py ocr <路径>`；情况 ② 还能按 sha 前缀在
  `%USERPROFILE%\.dsh\attachments\v1\objects\<sha前2位>\<sha>` 反查附件实体
  （内容寻址，同图只存一份，永不自动删除）。
  想彻底顺过来（贴图即落盘、请求里带只读路径）需在 DSH 侧声明模型支持图片
  （`%USERPROFILE%\.dsh\settings.yaml` 里给该模型加 `input: [text, image]`，见
  `skills/vllm-kb/SKILL.md`「图片/截图排查」）——只在模型**确实**具备图像能力时才声明。

**低置信度图片不进正文（导入期分级）**

只有**高置信** OCR 文本才注入所属文档正文参与检索（FTS + 向量）：正文里的 `[图片:xxx]` 占位符
之后追加

```
图片文字（OCR 置信度 0.92）:
error code 107020, dispatch_ffn_combine failed
```

判定条件（三者同时满足）：模型/服务返回了置信度、**无自报异常**、且 `≥ ocr_min_confidence`
（默认 0.6，审核工作台 API 配置中心可改）。低置信 / 自报异常的结果**不进正文**（不污染语义检索），
只保留在 `data/parsed/images/*.ocr.json` 与 `extra.evidence[].ocr`（含签名线索），
并由审核队列的 **`low_confidence_ocr`** 类别按图片 sha256 聚合生成待办
（`payload.reason` 区分 `low` / `anomaly` / `no_confidence`），人工核对后再决定是否采用。

> 正文注入的位置与图片引用一致：`MarkdownSource` 在 `canonicalize` 时**即时 OCR 并按需写缓存**，
> 不依赖 image source 的执行顺序（config 里 markdown 排在 image 之前也能注入）。
> 例外：`ocr_provider=ask` 且未配 `ocr_api_base` 时，Markdown 阶段不做交互询问（询问由 image source
> 统一做一次）——首次构建只产出缓存，**下一次构建**才注入正文。

### 2.4 Excel 登记表导入（schema-free）—— 完整实操

工程师问题定位记录 / 已知问题登记表等（**表头/列不固定**）：

```jsonc
// config.json sources 启用（path 可以是单个文件或目录）
{"id": "engineer", "type": "excel", "path": "data/imports/engineer/问题定位记录.xlsx", "enabled": true}
```

```bash
pip install openpyxl          # 依赖（可选组，离线 wheel 可装）
python scripts/build_kb.py    # 导入：一个文件一篇文档入库 + 嵌入
python scripts/build_graph.py # 建图：错误码/算子等实体自动入图（先停 serve_api）
python skills/vllm-kb/client.py graph sig <错误码>   # 验证实体命中
```

- **schema-free**：不写死任何列名 / sheet 名 / 行号——遍历所有 sheet，每个非空行进正文，
  正文体例是 `## <sheet 名>` + 若干行；
- **结构页过滤**（2026-10 业务实测新增）：封面/扉页/前言/序言/目录/版权/声明/修订记录类
  sheet 名（含 cover/preface/toc/copyright/revision 等英文形态）**不进正文与标签**——
  业务多页 Excel 的首个 sheet 常是封面（"版权所有 ©…"），不滤则文档 body 开头是版权样板、
  审核页标题观感受污染；业务特有结构页用 config `skip_sheets`（正则数组）追加
  （如 `"skip_sheets": ["^运维级别"]`）；滤了哪些看 `quality.sheets_skipped`；全部 sheet
  被滤/为空 → 不产生文档；
- **表头键值对（auto 判定）**：sheet 非空行 ≥3 且首个非空行 ≥2 列、全为短标签（≤30 字符）
  → 视为表头，数据行变 `列名=值 | 列名=值`（表头行保留为 `[表头] …`）——**行 chunk
  自包含列语义**（"故障名称=xxx" 可直接搜）；说明页/无表头结构（行数 <3 或首行长文本）
  原样按列序裸拼，语义无损优先；
- **一个文件一篇文档**：`source_id = excel:<文件名>`（与 PDF/Markdown/Word 同构），
  标签与验证状态都是**整体评估**（登记表整体入库、审核台整体补标），不做逐行判断；
  * **为什么不再逐行一篇**：逐行的身份只能靠**位置**（`文件:sheet:行`），在中间插一行就会让
    其后所有行的身份整体平移——人工治理（标签/补标）会**串到别的行**上。按文件聚合后
    插行/改行都不影响身份；
  * **检索粒度不损失**：分块仍以**行**为单位（每个非空行是独立 chunk），且 `## sheet 名`
    让每个 chunk 带 `section`——命中时能看出属于哪张表，表头键值对让 chunk 自带列语义
    （文档身份 → 表（section）→ 列（键值对）三层语义完整）；
- 错误码/算子/模型/版本由 signature 三层提取**自动入图**（建图只依赖 canonical，无需图侧适配）；
- 验证状态 `unverified`（登记表低优先级），审核台补标认证后按 `tested` 参与 `w_rel`。

**旧行级格式迁移（excel:文件:Sheet:行 → excel:文件名）**

旧版构建的登记表是**逐行一篇**（打标/认证/删除只能逐行做）——`git pull` 新代码后用
迁移工具一次性转换（幂等，可重复跑）：

```bash
python scripts/migrate_excel_rowdocs.py --plan        # ① 干跑：列行级文档+人工标签存档
python scripts/migrate_excel_rowdocs.py --delete      # ② 删行级文档+墓碑（构建不复活）
python scripts/build_kb.py                            # ③ 重跑构建 → excel:<文件名> 一篇/文件
python scripts/migrate_excel_rowdocs.py --apply-tags  # ④ 人工标签写回文件级文档
```

- 行级文档的人工标签/排除按文件**并集**保留（人工意图不丢）；词典命中的自动标签构建时
  自然重现，不迁移；
- 审核队列自动自愈：行级补标审核项由 seed 自动清理（文件级项取代）；标签候选里对旧行
  sid 的引用由 seed 校正（悬空忽略/刷新引用）。

**已认证文档的回撤**（两条路径）

- **队列侧**：审核台勾选"显示已处理"→ 找到该文档的补标审核项 → 认证级别选 `unverified`
  → 批量应用/单条认证 = 清覆盖层回原始值 + 审核项回 pending 重审；
- **文档侧**（2026-10 新增）：文档管理页 docCard 认证 badge（✓ tested/expert · 认证人）
  旁点 **↺ 重置认证**——清覆盖层 + 关联审核项重开（表格文件 = 整表重审）。

**内部数据脱敏（后置，Excel/Markdown 源生效）**

- **库中存原文、出口统一脱敏**：serve_api 返回给 agent 的正文/标题（`/doc` 全文、`/search` snippet、
  `/title`、`/tags`、`/graph` 等）按 `config.sanitize` 白名单脱敏（内部 IP → `<IP>`、
  内部路径 → `<PATH>`，默认路径如 `/var/log/npu/` 保留）——**内部检索用原文**（可按原 IP 检索），
  **改脱敏配置即时生效、无需重嵌**；
- `config.sanitize` 三个字段：
  - `keep_paths`：保留的默认路径前缀；
  - `keep_ips`：保留的 IP（默认回环/通配）；
  - `sources`：入库时扫描维护日志的源（默认 `["excel","markdown","word"]`）。
  `None` = 用默认，显式 `[]` = 全部脱敏 / 全部关闭；
- **被脱敏的原始 IP/路径落盘 `data/sanitize_log.json`**（维护文件，不进库、不返回给 agent）——
  据此调整白名单；审核页（管理员）显示原文。

### 2.5 Word 文档导入（.docx/.docm）—— 完整实操

```jsonc
{"id": "cases", "type": "word", "path": "data/imports/word", "enabled": true}
```

`path` 可为目录（递归）或单个文件。流程与 §2.3 同构：

- **pull**：`*.docx`/`*.docm` → `data/assets/word/`（sha256 不可变层），并注册 `asset_registry`
  （审核台「文档管理」据此反查路径 / 取回原件）；
- **标题样式即章节**：`Heading 1-6` / 中文「标题 1」/ `Title` → Markdown `#`~`######`。
  这是相对 PDF 的实质增益——PDF 只能靠正则猜编号标题，Word 的结构是**显式声明**的。
  分块因此直接复用 markdown 的章节切分：每个 chunk 带 `section`，检索命中就能看到所属章节；
- **列表**：`List Bullet` / `List Number`（或直接编号格式）→ `- ` / `1. `，避免条目粘成一段
  （粘成一段会让 FTS 命中粒度变差）；
- **表格**：转 Markdown 表格**拼入正文**（表内错误码/命令可被 FTS 检索），同时落
  `data/parsed/word/{asset_id}.tables.json` 供结构化消费（建图时提取表格 → 错误码/命令的
  `DOCUMENTS` 边）；
- **内嵌图片**：正文按文档顺序插入**不透明占位符** `[图片]` / `[图片:alt]`（alt 取 `wp:docPr/@descr`），
  图片本身落 `data/assets/images/img_{sha256前16位}.{ext}` 并注册 `asset_registry`；
  正文出现过的图片走 OCR（`data/parsed/images/<asset>.ocr.json` 幂等缓存）——
  **高置信文本注入占位符之后**随正文进 FTS + 向量，低置信/自报异常只留签名线索并进
  「低置信 OCR」审核队列（与 markdown 图片、`ImageSource` **同一套 OCR 缓存与通路**，同图不重复调用）；
- **超链接**：渲染成 `[文字](url)`（`Paragraph.text` 会把链接文字并进来但**丢掉地址**，
  业务文档里「见 XX 手册」的链接是重要线索）。**只认 http/https**：内部书签（`w:anchor`）
  只留文字；`file://`/UNC 等目标是内部路径，**一律不写进正文**（只留链接文字）；
- **解析缓存**：`data/parsed/word/{asset_id}.extract.json`（内容寻址）——文件未变时复用提取结果，
  标签/元数据/**OCR 注入**每轮重算（升级提取规则或调 `ocr_min_confidence` 都**无需清缓存**；
  缓存带 schema 位，提取逻辑升级会自动失效旧缓存）；删该目录即强制重解析；
- **身份**：`word:<文件名去后缀>`（同名不同目录用相对路径指纹消歧为 `word:<stem>--<sha8>`）。
  与 markdown 同构：**优先读 `imports/word`**（保留目录树 → 可消歧、编辑源文件不会因资产层累积
  副本而变成多篇），扫不到时回退 `assets/word` 扁平副本并按版本族收敛（每族只取最新一份、
  `source_id` 取族名）；
- **verification=unverified**：与 markdown/excel **统一路径**——先入库，审核工作台「补标」队列
  人工确认（`verification_pending`）；
- **脱敏**：默认启用（`config.sanitize.sources` 默认含 `word`），正文原文入库、出口统一脱敏，
  被脱敏的 IP/路径落 `data/sanitize_log.json`；
- **标题识别三重信号**：① 样式名/`style_id` 里的 `heading|标题|標題` + 数字（`Heading 1`、
  「标题 1」、繁体「標題 1」）；② 名字表（`Title`→1、`Subtitle`→2）；③ `w:outlineLvl`
  （**自定义标题样式名**如「我的章节」只能靠它——Word 导航窗格/自动目录用的就是这个）。
  块级内容控件（`w:sdt`）会下潜，否则整块内容丢失。

**本版边界**（因此正文不含任何路径）：

- 页眉/页脚/脚注/尾注是**独立部件**（`/word/header1.xml` 等），不在正文遍历范围内 →
  其文字与图片都不提取；
- 文本框：其**图片**在正文 XML 内（`w:txbxContent`）会被遍历到并产占位，但其**文字**
  不在遍历范围内（只下潜"透明"容器：`w:ins`/`w:sdt`/`w:smartTag`，不进 `w:drawing`/
  `w:pict`）→ 文字不提取（两者不对称，已知）；
- 表格内图片的占位符统一排在表格之后（塞进单元格会破 Markdown 表格）；
- 嵌套表格只取外层单元格文本；合并单元格重复文本；
- **矢量/多页图片**（emf/wmf/svg/tiff）注册资产但**不送 OCR**（引擎不收，白花一次调用）；
- 孤儿图片（包内有关系、正文未引用）注册资产但不插占位、不 OCR（没有正文落点）；
- 外链图片（`r:link`，内容不在包里）只留占位，不登记资产（也**不记链接**，避免路径进库）；
- 超链接只渲染 http/https；修订**删除**的文本（`w:del`）不提取（不是可见内容）；
- `.doc`（旧二进制）/加密 docx / RTF / WPS / ODT 不支持 → 跳过并给出**可操作**原因
  （CFB 容器会说明"旧版 .doc 或加密 docx，请另存为未加密 .docx"；目录里只有旧格式时
  会列出文件名并提示转换）。

**验证**：`python -m unittest tests.test_word_source -v`

### 2.6 HTML 网页导入（.html/.htm）—— 完整实操

**适用**：浏览器"另存为"保存的网页（对外案例页 / 内部 wiki 文章页 / 故障复盘帖）。
**只认本地文件，不联网抓取**（远程图片只留占位不下载）。

```json
{"id": "webpages", "type": "html", "path": "data/imports/html", "enabled": true}
```

**建议目录结构（图片在数据源接入时本地化）**：

```
data/imports/html/
├── GLM5.1-PD分离案例.html        ← 浏览器"另存为"，建议选「网页，全部」
├── GLM5.1-PD分离案例_files/      ← 「网页，全部」会顺手生成的图片目录
│   ├── 01-报错截图.png
│   └── 02-npu-smi输出.png
└── （也可自整理：assets/<案例名>/…，src 相对路径任意组织）
```

页面里的 `<img src="./GLM5.1-PD分离案例_files/01-报错截图.png">` 原样保留，
canonicalize 以**网页所在目录**为基准解析 → 资产化 + OCR（`data:URI` 内嵌图同样落
资产）。**远程图链不下载**（与 markdown 同一原则：图片本地化是数据源接入时的事）：
引用 `http(s)://…` 的图入库时安全降级为 `[图片]` 占位 + remote evidence，图不进库。
实操建议：浏览器另存选「网页，全部」（图随页面落本地、src 自动变相对路径）；
「仅 HTML」会把图全部留在远程（只剩占位）——需要图的页面请把截图补进导入目录。

**引擎分工**（`vllm_kb/html_convert.py`，混合：lxml 收割 + trafilatura 正文）：

- **表格收割先于正文抽取**：内容表（≥2×2 或有 th/caption）→ 结构化 rows 落
  `data/parsed/html/{asset_id}.tables.json`（建图抽错误码消费），正文以 pipe table
  **原位还原**（占位 token 随正文引擎带上，表文位置=原文位置）；布局表（<2×2 且无
  th/caption）丢弃，整表文字 ≥80 字符视为"套壳正文"（单格代码框）展开成段落保留；
  丢弃/展开计数进 `quality`；
- **正文**：trafilatura（`output_format="markdown"`，自带去噪，适配文章页）；抽取
  失败/过短 → `baseline` 全文兜底；未安装 → lxml 全文兜底（表格收割不受影响）。
  nav/aside/footer 与 script/style 先于引擎整棵移除——小页面会触发 trafilatura 的
  召回兜底把导航噪音留进正文，script 文本否则会被 baseline 当正文捞回来；
- **图片**：收割后**原位还原**（收割占位 `[[KBIMG:i]]` 是纯文本，trafilatura 不会丢弃，
  图文位置=原文位置；正文引擎丢掉占位段落时退化为文末补回），src 原样
  保留 → 与 markdown 来源同一通路：本地相对路径以网页所在目录为基准、`data:URI`
  落资产（内容寻址 + OCR）、远程只留占位；**装饰图过滤双层**：显式
  `width/height ≤2px` 收割层直接丢弃；文件内容 1x1（GIF/PNG/JPEG 头解析，零依赖）
  资产层拦截——埋点 gif/logo 不再变正文噪声（evidence 记 `decorative`，
  `quality.images_decorative_skipped` 计数）；
- **审计元数据**：`<meta name="x-*">`（x-source-url/x-created/x-updated/x-status/
  x-author/x-categories/x-expiry 等）整体收割进 `extra["meta"]`，不再丢弃；
  `docs.url` 优先取 x-source-url，其次浏览器"另存为"残留的
  `<!-- saved from url=(NNNN)… -->` 注释（无则留空）；
  **消费**：x-categories → 打标管道（词典命中 → 自动标签；未收录 → tag_candidate
  候选队列，采纳后全局进词典——审计侧人工分类是 page.html 布局的主要打标信号）；
  x-created → `docs.created_at`（ISO 日期前缀校验，脏值留空；文档管理排序/遥测时序）；
  x-updated/x-status/x-author/x-expiry 暂存 meta 待后续消费（status 驱动撤回、
  expiry 过期治理是未来扩展项）；
- **统一 page.html 布局**（审计来源每篇文章一个目录、文件名固定 page.html）：同名靠
  相对路径指纹消歧（`html:page--<sha8>`，内容改动下稳定，目录改名=删旧增新）；
  泛名 stem（page/index/default/…）不顶替 `<title>` 提取的真实标题；
- **链接**：只渲染 http/https/mailto 目标，相对路径 / `file://` 剥掉只留文字
  （与 Word 超链接同一安全规则：正文与 canonical 不含任何服务器路径）；
- **编码**：BOM → meta charset → utf-8 → gb18030 → replace（内网 GBK 保存页免转码）；
- **verification=unverified**：与 markdown/word 统一路径（审核台补标）；脱敏默认启用
  （`config.sanitize.sources` 默认含 `html`）；解析中间产物缓存
  `data/parsed/html/{asset_id}.extract.json`（schema 位，二次构建不重解析）。

**已知边界**：

- trafilatura 正文不含 markdown 标题（h1-h6 抽成普通行）→ chunk 无 section（chunking
  有段落级兜底切块，检索粒度略粗）；
- `.mht/.mhtml/.url` 不支持 → 跳过并提示在浏览器里另存为「网页，仅 HTML」。
  **是否值得做 .mht 转换取决于存量**：MHTML 是 MIME 容器（stdlib `email` 可解析——
  取 text/html part 喂现有引擎、图片 part 落地后替换 CID 引用，纯本地、工程量中等）；
  若业务环境盘点出大量 .mht 且**原页面已下线**（无法重新另存）再立项，存量少或页面
  还在，提示另存即可。

**验证**：`python -m unittest tests.test_html_source tests.test_html_convert -v`

## 3. 启动检索服务

```bash
python scripts/serve_api.py                    # http://127.0.0.1:8000
python scripts/serve_api.py --host 0.0.0.0     # 远程访问（配合存算分离）
```

服务结构只读：SQLite `mode=ro`、向量库写操作抛错、无写端点。

### 3.1 总日志接口（打屏 + 可选落盘分卷）

所有服务（serve_api / review_ui / 构建脚本）统一日志：**打屏（默认）**；
需要落盘时在 `config.json` 的 `logging` 段开启：

```jsonc
{"console": true, "file": true, "file_path": "logs/vllm-kb.log",
 "max_bytes": 10485760, "backup_count": 5}
```

- `file=true` 后，uvicorn 访问/错误日志与业务 logging 输出写入 `file_path`，
  **按 `max_bytes` 分卷**（RotatingFileHandler，保留 `backup_count` 个历史卷）；
- `file=false`（默认）不落盘，仅打屏；
- 服务状态监控：`serve_api` 的 `GET /health` 返回 `{status, chunks, embedding, embedding_note}`——
  `status` 恒为 ok（检索自动降级），`embedding` 反映嵌入服务健康（ok / degraded / degraded-retrying）；
  `review_ui` 的 `/api/stats`（审核队列）；业务日志打屏观察即可。

**embedding 不可用时平稳降级**：查询自动降级为全文检索（向量召回跳过，结果照常返回）。
查询用快速失败客户端（5s 超时 × 1 重试），连续失败 3 次熔断 60s（期间跳过 embed 调用零等待降级），
到期自动探测恢复；降级期间 `/search` 响应带 `degraded` 提示，agent 可见。

### 3.2 审核工作台（人工确认统一入口 + API 配置中心）

所有需要人工确认的位置（未验证文档补标、案例标题待审核/待修改、OCR 图文互证不一致、
低置信度 OCR 签名、跨来源合并候选等）共用一个轻量级 Web UI；同时集中展示所有 API 配置。

```bash
pip install fastapi uvicorn
python scripts/review_ui.py                    # http://127.0.0.1:8010（默认自动补单，幂等）
python scripts/review_ui.py --seed-only        # 只补单不启动服务
python scripts/review_ui.py --no-seed          # 启动但不自动补单
```

**功能**

- **概览**：7 类审核项的待办/存疑数（verification_pending / case_title_flag / ocr_mismatch /
  low_confidence_ocr / equivalence_candidate / table_join_candidate / **tag_candidate**）
  与标签词典统计（领域/作用类个数、已打标文档数）；
  `low_confidence_ocr` 由 seed 从 `extra.evidence[].ocr` 中 `text_included=false` 的图片生成，
  **按图片 sha256 聚合**（同一张图被多篇文档引用只生成一条），`payload.reason` 为
  `low` / `anomaly` / `no_confidence`，详情页展示图片 sha 前缀、原因、置信度与 OCR 签名线索；
  其 `item_ref` 是图片而非文档，故**不提供 🗑 标记删除按钮**（不想处理就用 ✓ 认证忽略）；
- **审核队列**（未审核在前、存疑在后）：按类别筛选；详情页可预览原图（assets 静态服务）与
  **正文预览**（`/api/doc-preview`，chunks 按 seq 拼装——审核人先看到内容再判定）。
  **表格（Excel）按文件粒度审核**：同一登记表的待补标行聚成**一条**审核项
  （`excelfile:<文件>:<sha8>`，队列带「N 行」徽标）——认证一次应用到文件内全部行，
  **不逐行审**；表格内容变更后**自动重新进入队列**（认证时的内容指纹记录在覆盖层
  `content_sha`，ingest 合并时指纹不符按未认证处理，seed 清过期覆盖层并重新补单；
  同内容重跑不重复打扰）。
  **审核动作（只做判定，不修改原始内容）**：
  - **✓ 认证**：文档有效，不再提示（补标项可选认证级别 **tested / expert**；表格项一次认证
    **整表生效**）；
  - **？ 存疑**：重新进入队列，排在未审核之后（表格项=整表回退到未认证）；
  - **🗑 标记删除**：只删除 `kb.sqlite3` 数据库记录（表格项=**整表全部行**），**原始资产文件保留**
    ——进入队列底部"**待实际删除**"列表（含资产路径），由人员**手动本地删除**原始文件；
  - **↩ 撤回**：在待删除列表恢复数据库记录（用删除前的备份，表格项整表恢复），重新进入队列；
  - **tag_candidate 采纳**：未收录的自动标签候选 → "✓ 采纳为标签"（入词典 + 对该候选**全部提及文档**打标，
    检索侧立即生效）；不采纳则 **✓ 认证** 即忽略（不改任何数据，候选不再自动出现）。
    候选标签的来源、聚合与生效边界见下方"候选标签（tag_candidate）说明"。
  审核记录存 `data/review.sqlite3`（独立于只读检索库），带审核人/时间戳可审计；
- **文档管理（含两层标签编辑 + 目录分组 + 批量操作）**：列出外源文档（导入的 PDF/Markdown 等，GitHub 不在此列），
  每条显示**自动标签**（领域/作用分区，点 ✕ 排除）、**已排除**（点 ↺ 恢复）、**人工标签**
  （添加/删除）与**最终标签预览**（`(自动 − 排除) ∪ 人工`，与入库/建图一致）；
  人工添加的新标签自动同步词典；**同 stem 重名告警**（人工处理，不自动消歧）；
  支持**彻底删除**——同时清除 docs 行 + chunks_fts + chunks_meta + 向量四层，
  **本地资产文件不动**；下次增量入库时文件仍在本地会**自动重新入库**，文档废弃由管理员手动删本地文件；
  - **目录分组**：组名 = **`来源id\子目录`**（根级 = 来源 id，如 `pdf`；子目录 = `pdf\support官方文档`；
    资产回退件 = `来源id（资产回退）`；网络来源 = 未分组）——不同导入来源即使子目录同名也各自成组。
    分组信息在入库时记入 `docs.extra.rel_src`（来源 id）与 `docs.extra.rel_dir`（第一层子目录）
    （管理员侧专用——检索出口 `_sanitize_extra` 白名单**不透出**，与 source_id 不可读指纹同一安全约束）；
    存量文档下次构建自动补齐（meta_hash 变化走 meta_refresh，**不重嵌入**）；
  - **批量操作**：勾选多篇（或 📁 组头**全选本组**）→ 顶部批量栏：**批量标签**
    （添加/删除人工、排除/恢复自动，逐篇独立生效并记审计）与**批量删除**（数据库四层，
    本地文件保留；向量 store 只建一次）。部分失败逐条返回明细，成功数不含不存在的 id；
- **审核队列 · 批量操作**：`verification_pending` 项勾选 → **✓ 批量认证**（tested/expert 整批应用）；
  `tag_candidate` 候选勾选 → **批量采纳**（入词典 + 全部提及文档打标，tier 可整批覆盖）/
  **忽略所选**（不再出现，不改任何数据）——消化积压是分钟级操作。逐项独立生效记审计，
  部分失败返回明细；
- **设备词表（config.tags.devices）**：机型/芯片的受控小词表——`{name, tier, kind: machine|chip,
  aliases: [同义变体], uses_chips: [所用芯片], note}`。**别名归一**：文本/文件名命中别名
  （如 800I A2）→ canonical 设备标签（A2）自动打标，设备 name 自动补入词典；
  **关系单向**：机器 --uses_chips--> 芯片（A2 → 910B），检索同向扩展安全；**芯片绝不反推机型**
  （300I A2 与 800I A2 同用 910B 但拓扑差异巨大，必须独立处理——2026-10 领域确认）。
  词表词在候选降噪中**永久豁免**（A2/A3 是机型不是版本号）。校验只告警不阻塞（别名冲突/
  引用未定义/kind 非法）；
- **候选降噪（提取端）**：拉丁停用词默认集（for/of/and…）+ **中文整词停用词默认集**
  （排查/说明/常见问题等标题模式词——整词等于才滤，"HCCL 超时排查"不受影响）+ 纯数字
  过滤 + 设备别名组成 token
  抑制（"800I"/"A2" 已归一不再各自成候选）——**在候选生成端降噪，而非采纳端**；
  `tags.stopwords` 可继续追加业务停用词（拉丁与中文同效）；
- **同义变体收敛（聚合端）**：`tag_key` 聚合键（大小写 + 空格/连字符/下划线分隔 + CJK 间
  空格归一）统一候选聚合、词典查重、存量候选合并——排版变体自动合并为一条候选，
  采纳时收敛到词典已有词条；存量 pending 变体由 seed 流程幂等合并（`merged_variant` 审计）；
- **标签管理**：词典（`config.json` `tags.registry`，全局唯一事实源）按领域/作用分组 + 文档数，
  支持**新增 / 改名（全库替换）/ 改 tier / 删除**——均同步 config.json；**不热插图**
  （Kùzu 单写者约束），运行 `build_graph.py` 重建后入图；
  - **减少操作次数（2026-10）**：tier 徽标**点击即切换**（1 击，全局生效）；勾选多个 →
    批量栏**批量移出词典 / 批量改 tier**（逐项独立生效，不存在项进 failed 明细）；
    **审核人填一次全站记忆**（requireReviewer：文档管理排除/恢复、批量删除/认证/采纳等
    所有操作不再重复弹窗——首次填写后自动记住）；
  > **TODO（暂缓）**：① 打标侧仅剩**按主题聚合的打标视图**（目录分组已就绪，
  > 可在组头给"该组高频缺失标签"建议），待积压消化后再评估；② **人工治理数据备份**
  > （2026-10 登记，配合审计来源接入的后置项）：`review.sqlite3`（审核操作/队列/墓碑）+
  > kb 覆盖层（doc_tags/doc_verification——人工成果、canonical 里没有）+ asset_registry
  > 的定期备份与恢复演练——issue 等可重拉，**不可再生的是人工治理数据**；
  > ③ 审计驱动 git 来源接入（设计定稿未实施，见 `docs/DESIGN-audited-git-source.md`）。
- **API 配置中心**：集中查看并**编辑** embedding / OCR / GitHub / code_graph 的配置——
  **非密钥字段**（provider/base_url/model/ocr_provider/ocr_api_mode/ocr_min_confidence 等）保存到 `config.json`；
  **密钥**（embedding key / OCR key / GitHub token）保存到 `data/secrets.local.json`
  （遵守"密钥不入 config.json"，任何入口 `AppConfig.load` 自动加载进环境变量）；
  key 在页面上一律脱敏（只显示已/未配置），**embedding 与 OCR 均支持连通性测试**
  （OCR 用内置测试图走真实识别链路，验证 HTTP/鉴权/模型）；修改对已运行的服务需**重启生效**。
  校验失败（非法字段/未知配置段）返回 **400 + JSON 错误信息**（前端弹窗展示，不再 500）。

**候选标签（tag_candidate）说明** —— 文档打标时的候选与审核队列的候选是同一机制的"产生端/审核端"：

1. **来源（文档打 tag 时）**：入库自动打标（`vllm_kb/tagging.py` 的 `extract_tags`）从**文件名 stem +
   内部标题**确定性提取"未收录强候选"（拉丁词 token + 短标题，tier 启发式），随文档写入
   `kb.sqlite3` 的 `docs.extra.tag_candidates`（PDF/Markdown 来源）。**候选不会打到文档上**——
   文档自动标签只含词典命中项，候选只进审核队列待人工裁决；
2. **审核队列（按词聚合）**：审核工作台自动补单时 `seed_tag_candidates` 扫描全部文档的
   `extra.tag_candidates`，**按候选词聚合**成一条审核项（`item_ref = tag:{tag_key}`，payload 含
   提及文档数 + 文档列表 + 建议 tier；词典已收录的词不生成）——同词多文档只审一次，
   **审一次 = 全部提及文档生效**；候选名聚合前按聚合键 `tag_key` 归一（大小写 /
   空格-连字符-下划线分隔 / CJK 间空格——"Atlas A3"/"atlas_a3"/"超时 排查"/"超时排查"
   等排版变体收敛为一条候选；显示与词典正式名保留首见原形），存量 extras 里的变体
   在补单时即收敛，无需等重建；
3. **采纳（adopt）**：① 候选词写入 `config.json` 的 `tags.registry`（tier 取审核下拉选择或启发式，
   全库词典生效）；**同 key 已有词条（词典 "Atlas A3" vs 候选 "atlas_a3"）→ 收敛到已有
   词条**，manual 写词典正式名，不产生第二个词条；② 对聚合记录的全部提及文档写人工标签
   （`doc_tags.manual`），并按
   `final = (自动 − 排除) ∪ 人工` 同步 `docs.tags`——**检索侧立即生效**；③ 审核项标记 approved，
   重跑补单不再出现；
4. **忽略**：**✓ 认证**即忽略（不采纳、不改任何数据）；因 `item_ref` 已存在（任意状态），
   之后自动补单不再打扰。**🗑 标记删除按钮对 tag_candidate 不适用**（其 `item_ref` 是候选词   `tag:{name}` 而非文档 source_id，点删除会报"文档不存在"）。
5. **已认证 tag 的变体归一（`normalize_tag_variants`，每次 seed 前自动跑，幂等）**：
   存量词典里同 `tag_key` 的多个词条（"CANN版本说明" 与 "CANN 版本说明" 都曾被人工
   认证入典）**收敛为首条原形**，并对全库做标签替换（`docs.tags` / `doc_tags.manual`
   + `excluded`）——**已认证的 tag 自动适用聚合规则，不需要删掉重加**；同时 `TagRegistry`
   加载即按 key 去重（config 文件在下次保存时自然收敛），adopt 同 key 映射到已有正式名。

**生效边界**（"刷新到所有文档"需分清）：

| 目标 | 采纳后何时可见 | 需要做什么 |
|---|---|---|
| 检索 API：`tags` 目录 / `tags docs` / `context` / search 的 **FTS 命中**过滤 | **立即**（读 `docs.tags`，采纳时已同步） | 无 |
| **补标认证的 `w_rel`**（verification_pending → tested/expert） | **立即**（覆盖层 + `docs.extra.verification` 同步刷新） | 无 |
| search 的**向量命中**过滤（chunk 向量 meta.tags 是入库时快照） | 下次入库 | 重跑 `build_kb.py --skip-pull` |
| Kùzu 图（TAGGED_WITH 边 / `graph tags`） | 重建图后 | `build_graph.py`（先停检索服务） |
| 词典对新文档生效（文件名/标题含该词自动打标） | 下次入库 | 无（词典已写入 config.json） |
| 未提及该候选的其他文档 | 仅词典更新，**不自动打标** | 文档管理页人工添加，或走下方正文候选独立路径 |

**与 `build_tag_candidates.py`（正文 TF-IDF）的区别**（易混淆点）：审核队列候选来自**文件名/标题**
提取（自动、可一键采纳打标）；`build_tag_candidates.py` 用 jieba TF-IDF 从**正文**提取候选，
输出 `data/tag_candidates_manual.json` 文件，**人工审阅后手动写入** `config.json` 的 `tags.registry`——
不经过审核队列、**不自动打标**（写入后下次入库/建图对新文档生效）。两条路径独立，勿混淆
（Excel 来源明确走正文路径：不做文件名/标题标签）。

自动补单规则：`verification=unverified` 的文档 → verification_pending；标题含"待审核/待修改"→
case_title_flag；`extra.evidence[].ocr.text_included=false` 的图片 → low_confidence_ocr
（按图片 sha256 聚合）；`extra.tag_candidates`（未收录强候选）→ tag_candidate。

**补标回写（verification_pending）——认证立即生效**：

- 「认证」→ 覆盖层 `doc_verification` 写 `tested`（`result.verification="expert"` 可升为专家认证），
  并**立即**刷新 `docs.extra.verification` 与 `reliability`——**不需要重建**，
  检索算出的 `w_rel` 当场从 0.5 提到 0.85（0.95）；「存疑」→ 清除覆盖层，
  回到**覆盖前**的 canonical 值（不是写死 `unverified`：PDF 手册 canonical 就是 `expert`）；
- **为什么必须有覆盖层**：`docs` 行是 `INSERT OR REPLACE` 写的，直接改 `docs.extra` 会被下次
  构建用 canonical（`unverified`）覆盖回去，人工认证就白做了。所以人工结果存独立表，
  由 `ingest` 每轮合并（与 `doc_tags` 标签覆盖层同构）；
- 认证过的文档不会再进补标队列（`seed` 读的是**生效值**）；覆盖层变化会让 `meta_hash` 改变，
  下次入库走"仅刷新元数据"路径（`embed_hash` 未变 → **不重新嵌入**）；
- `--rebuild` 会删 `kb.sqlite3`，因此**先备份后恢复** `doc_tags` / `doc_verification`——
  人工标签治理与补标认证不会被重建抹掉。

审核结果（标签类）回写 canonical 依赖重跑 `build_canonical.py`（或 `build_kb.py --skip-pull`）。

**与只读检索 API 的关系**：分离端口（检索 8000 / 审核 8010）、分离数据（kb.sqlite3 只读 /
review.sqlite3 可写）；审核库检索 API 不碰。权限（谁能标注专家认证）由部署方加 nginx basic auth 等。

### 3.3 图更新流程（Kùzu 单写者约束）

**更新图（`scripts/build_graph.py`）前必须先停止检索 API**——检索服务持有图库读连接，
Kùzu 单写者会拒绝建图（`Could not set lock on file: data/graph/db`）：

```bash
# 1. 停 serve_api（8000 端口进程）
# 2. 重建图
python scripts/build_graph.py
# 3. 重启 serve_api
python scripts/serve_api.py
```

**路径限制（Kùzu）**：图库路径（`storage.graph_path` = `data/graph`，或存算分离时的
`VLLM_KB_DATA_ROOT`）**不能含非 ASCII 字符**（中文、emoji 等）——Kùzu 打开含非 ASCII
路径的库会报错打不开。若部署根路径含中文（如 `C:\Users\张三\...`），请把数据根移到
纯 ASCII 路径（如 `D:\vllm-kb-data`）后重建图。

> 图的构建内容与查询命令见 [§4.8 graph](#48-graph--phase-2-图检索关系追溯)。

## 4. 查询命令（skill）

所有查询经 `skills/vllm-kb/client.py`，服务地址解析：`--base` > 环境变量 `VLLM_KB_BASE` > 默认 `http://127.0.0.1:8000`。
`--probe` 是全局开关（探索/测试请求打标，见 §4.10），任意子命令均可加。

| 命令 | 一句话用途 | 详见 |
|---|---|---|
| `search` | 语义检索（推荐 `组件:版本 问题`） | §4.1 |
| `signature` | 贴原始报错，签名精确检索 | §4.2 |
| `title` | 标题/文档名精确检索（已知现象找 issue） | §4.3 |
| `version` | 版本形态判断（release / rc / pre） | §4.4 |
| `code` / `diff` | 版本化源码定位、跨版本 diff、报错字面量索引 | §4.5 |
| `doc` / `companion` / `stats` / `health` 等 | 整篇读取、配套展开、规模与健康 | §4.6 |
| `tags` / `context` | 文档能力发现（有哪些文档类别 / 能帮我做什么） | §4.7 |
| `graph` | 关系追溯（issue→修复 PR→落地 release） | §4.8 |
| `code-graph` | 代码图谱（调用链/影响面/架构，需 gh-puller） | §4.9 |

### 4.1 search —— 语义检索

```bash
# 组件:版本 问题描述（推荐，置信度按版本计算）
python skills/vllm-kb/client.py search "vllm-ascend:0.23.0rc1 GLM5.1 PD分离P节点挂死"

# 普通查询 + 目标版本
python skills/vllm-kb/client.py search "CUDA illegal memory access" --version 0.26.0 --top 5

# 按文档标签过滤（可多次，多个标签取交集）
python skills/vllm-kb/client.py search "HCCL 超时" --tag HCCL --tag 超时排查
```

返回：结果标题/URL/是否已解决 + 置信度分解（`w_time` 时间衰退 / `w_ver` 版本匹配 /
`w_rel` 来源可靠度，启用遥测反馈时另有 `w_hist`，见 §4.10）。

### 4.2 signature —— 签名精确检索

```bash
# 贴原始报错，现场提取签名（算子名/ACL 错误码/堆栈函数/特性/模型/版本）并精确匹配
python skills/vllm-kb/client.py signature "halMemCreate failed drvRetCode=6, kernel_name=DispatchFFNCombine, errorStr: timeout or trap error"
```

返回：提取的签名列表 + 命中的社区高频信号词（agent 判断用）+ 精确命中 + 标题命中。
三层提取（源码符号表 → 结构解析 → 通用短语），无需手写正则，随代码仓索引自动更新。

### 4.3 title —— 标题精确检索

```bash
# 已知现象找 issue 的最快路径（SQL LIKE）
python skills/vllm-kb/client.py title "vector core" --component vllm-ascend
python skills/vllm-kb/client.py title "MTE" --component vllm-ascend
```

业务文档（PDF/MD）主题词常在**文件名**里（title 只含首页首行，如手册
"Atlas A3 中心推理和训练硬件"）——`title` 同时匹配文档名（source_id）：
`title "npu-smi"` 能命中 npu-smi 命令参考手册（输出仍显示文档标题，不含文件名）。

### 4.4 version —— 版本形态判断

```bash
python skills/vllm-kb/client.py version 0.18.0           # → release（正式版）
python skills/vllm-kb/client.py version v0.23.0rc1       # → rc（预发布）
python skills/vllm-kb/client.py version 0.26.0 --repo vllm
```

用于判断"部署版本是正式版还是 rc"——影响"该修复是否已 backport 到我的版本"。

### 4.5 code —— 对应版本源码定位

```bash
# vllm-ascend 源码（默认仓库）
python skills/vllm-kb/client.py code DispatchFFNCombine --version v0.23.0rc1
python skills/vllm-kb/client.py code mega_moe_max_tokens --version v0.23.0rc1 --file vllm_ascend/ascend_config.py

# vllm 主仓源码（--repo vllm）
python skills/vllm-kb/client.py code make_zmq_socket --repo vllm --version 0.22.1
python skills/vllm-kb/client.py code worker_busy_loop --repo vllm --version 0.22.1 --file vllm/v1/executor/multiproc_executor.py

# 0day fork 仓源码（--repo fork:{model}，版本=镜像锁定 commit SHA 前 12 位；
# 与官方 rc/release 版本物理隔离，默认检索永不混入 fork 代码）
python skills/vllm-kb/client.py code mega_moe --repo fork:glm5.2 --version 418bd6273c03
python skills/vllm-kb/client.py code-versions --repo fork:glm5.2   # 看该 fork 已预存哪些 SHA

# 0day 镜像内实际部署的插件代码（--repo img:{tag}，版本=镜像 tag）
python skills/vllm-kb/client.py code-versions --repo img            # 有哪些镜像已提取（前缀唯一入口）
python skills/vllm-kb/client.py code DispatchFFNCombine --repo img:glm5.2
python skills/vllm-kb/client.py code "memory leak" --kind msg --repo img:glm5.2

# 列出已预存版本
python skills/vllm-kb/client.py code-versions --repo vllm

# 读取完整源码文件（--file 默认截断 20000 字符，截断带明确"已截断"标记；
# 需要完整函数体时调大 --max-chars）
python skills/vllm-kb/client.py code --file csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp --version v0.23.0rc1 --max-chars 100000
```

返回 `symbol_index`（符号精确命中）或 `grep`（关键词全文命中），含 `version/file/line/snippet`。
**版本未预存**返回 404 并列出可用版本与预存指引（`code-versions` 查看全部已预存）。

**定位"哪个版本引入/移除了某代码"（故障排查高频）**：

```bash
# 限定文件 grep（避免全仓命中淹没目标文件）：
python skills/vllm-kb/client.py code "fill_(-1)" --in-file worker/model_runner_v1.py

# 每个版本各自收集命中——一次对比所有预存版本的行号：
python skills/vllm-kb/client.py code "fill_(-1)" --in-file worker/model_runner_v1.py --per-version
#   → 若 blk_table.slot_mapping.gpu.fill_(-1) 只在 v0.23.0rc1+ 出现，修复引入版本即 v0.23.0rc1

# 跨版本精确 diff（新增行 = 修复引入点；--keyword 只留相关差异行，--context 调上下文行数）：
python skills/vllm-kb/client.py diff v0.22.1rc1 v0.23.0rc1 vllm_ascend/worker/model_runner_v1.py --keyword "fill_(-1)"
#   → 显示该文件两版本 unified diff，`+ blk_table.slot_mapping.gpu.fill_(-1)` 只在 v0.23.0rc1 出现
#   仓库另有 scripts/diff_code_versions.py 等价实现（需完整仓库环境）；client diff 走只读 API 即可

# 报错字面量索引（--kind msg）：报错文本→源码定义处 file:line 的索引命中（无需全文 grep）——
# 检索代码里 raise/assert/logger.error 的错误字符串参数（子串匹配）
python skills/vllm-kb/client.py code "memory leak" --kind msg --version v0.23.0rc1
```

`--kind` 可选值：`def`（函数定义）/ `op`（算子）/ `env`（环境变量）/ `msg`（报错字面量）；
不传则符号索引 + grep 兜底。

**`--repo` 命名空间（四选一 + all 联合，互相物理隔离）**：

| `--repo` | 指向的代码 | 版本键 |
|---|---|---|
| 缺省 `vllm-ascend` | 官方 vllm-ascend 快照（GitHub tag，`build_code_snapshots.py`） | `v0.23.0` 等 tag |
| `vllm` | vllm 主仓快照（`build_vllm_snapshots.py`） | `0.22.1` 等版本号 |
| `all` | **vllm-ascend + vllm 联合检索**（2026-10 业务维护反馈：vllm 与 va 拆开"缺哪边都不好定位"——一次查两仓，结果按 repo 分组；`code-versions --repo all` 联合列举） | 两仓各自键 |
| `fork:{model}` | 0day fork 仓的 **vllm 代码**（`build_fork_snapshots.py`，按镜像锁定 SHA 拉） | SHA 前 12 位 |
| `img:{tag}` | **某个 0day 镜像内实际部署的 vllm-ascend 插件代码**（`build_image_snapshots.py`） | 镜像 tag（如 `glm5.2`） |

> 不确定符号在哪边时用 `--repo all`（vllm 插件在 va 仓、调度/执行器在 vllm 主仓，
> 联合检索一次覆盖）；fork/img 的版本键不同，保持显式前缀。

**跨命名空间 diff**：`diff` 的两个版本参数可带前缀，直接对比不同仓/命名空间的同一文件：

```bash
# 这个 0day 镜像的插件代码相对官方同基线改了什么（0day 定制点审查）
python skills/vllm-kb/client.py diff img:glm5.2 vllm-ascend:v0.23.0 vllm_ascend/platform.py
# 带关键词只看相关差异行
python skills/vllm-kb/client.py diff img:glm5.2 vllm-ascend:v0.23.0 vllm_ascend/worker/model_runner.py --keyword "fill_(-1)"
# fork 仓 vllm 代码 vs 官方版本（fork:{model} 只有一个 SHA 时可省 @{sha12}）
python skills/vllm-kb/client.py diff fork:hy4 vllm:0.23.0 vllm/v1/executor/multiproc_executor.py
```

前缀形态：`vllm-ascend:{版本}` / `vllm:{版本}` / `img:{tag}` / `fork:{model}@{sha12}`；
不带前缀时用 `--repo` 指定的仓（两侧可以是不同命名空间）。

**`img:` 前缀怎么找（agent 的发现路径）**：

```bash
python skills/vllm-kb/client.py code-versions --repo img        # 已提取镜像清单（唯一入口）
#   glm5.2  vllm_commit=0fc695fc6d1d  镜像时间=2026-07-27T15:39:42Z  索引=有
#   hy4-a3  组=hy4  vllm基线=0.23.0   镜像时间=...                   索引=有
```

- 第一列就是 `--repo img:{tag}` 里的 tag；**"组=…"** 对应用户口中的镜像名（例如用户说
  "hy4 镜像"，实际 quay tag 是 `hy4-a3`，检索要用 `--repo img:hy4-a3`）；
- `code-versions --repo img:<tag>` 看该镜像元信息（digest / 镜像时间 / vllm commit / 层内插件 commit）；
- 某镜像不在清单里 → 由管理员跑 `python scripts/build_image_snapshots.py --tag <tag>`；
  quay 直连困难（隔离网/离线）时，内网预存的 docker save/OCI 归档可直接离线导入：
  `--archive <tar.gz>` / `--archive-dir <目录>`（不触网、不需要 docker；meta.source 标注
  `archive:`，其余字段与在线提取一致）
  （agent 无写权限，不要尝试自行提取）。

命中新版本后，用 GitHub commits API 按文件路径过滤找引入 commit → PR 编号 →
再用 `graph fixes/chain` 确认落地 release 与 backport。

> **符号索引升级**：各命名空间的 `index.sqlite3` 是派生数据（zip 快照为事实源）。代码提取规则/schema
> 变更后（如本次 ast 提取 + `--kind msg` 报错字面量索引）重建即可，无需手工迁移：
> 先停检索 API → 按命名空间跑对应 `--index-only`（自动重建 index.sqlite3 + 派生文件；旧索引缺
> `kind` 列时自动补列，但需全量重建才有 kind 数据）→ 重启 API。耗时：7 版本约 2 分钟，41 版本约 11 分钟。
>
> | 命名空间 | 重建命令 |
> |---|---|
> | `vllm-ascend`（默认） | `python scripts/build_code_snapshots.py --index-only` |
> | `vllm` | `python scripts/build_vllm_snapshots.py --index-only` |
> | `fork:{model}` | `python scripts/build_fork_snapshots.py --index-only` |
> | `img:{tag}` | `python scripts/build_image_snapshots.py --index-only` |
>
> 提示命令由服务端 503 详情自动给出（按请求的命名空间对应脚本）。

### 4.6 其他

```bash
python skills/vllm-kb/client.py health                  # 服务健康（含 embedding 状态 ok/degraded）
python skills/vllm-kb/client.py stats                   # 知识库规模
python skills/vllm-kb/client.py components              # 各组件文档数分布
python skills/vllm-kb/client.py doc github:vllm-project-vllm-ascend:issue:10700   # 整篇 issue 全文
python skills/vllm-kb/client.py companion vllm-ascend 0.23.0rc1   # 组件配套版本展开
# matrix / code-versions 为管理员调试命令（列全量配套矩阵 / 预存代码版本），故障流程不使用
```

### 4.7 文档标签（能力发现）—— `tags` / `context`

文档级标签两级分类：**主题/领域类**（domain：HCCL、网络、NPU、CANN…= 这是什么领域的知识）与
**具体作用类**（purpose：超时排查、命令参考、错误码表…= 文档能帮我做什么）。

```bash
# 能力目录：两级分组 + 各标签文档数（agent 先看"知识库有哪些文档类别可提供知识"）
python skills/vllm-kb/client.py tags list

# 按标签检索文档（标题/文档id/验证状态；非文件枚举）
python skills/vllm-kb/client.py tags docs HCCL
python skills/vllm-kb/client.py tags docs 超时排查

# 问题→标签匹配（文档能力发现）：命中领域×作用的文档交集排前
python skills/vllm-kb/client.py context "vllm-ascend:0.23.0 HCCL 超时"
#   → [领域] HCCL（12 篇）→ pdf:xxx HCCL 超时排查指南
#     [作用] 超时排查（8 篇）→ ...
#   → 先读命中文档（doc <id>），再结合 issue/代码下结论

# 图侧标签查询：标签 → 打标文档（Doc/Issue/PR）
python skills/vllm-kb/client.py graph tags HCCL
```

标签来源与治理：入库时从文件名 + 内部标题确定性提取（词典 `config.json` `tags.registry`
子串命中为准）；审核工作台可**排除自动标签 / 添加人工标签 / 管理词典**（新增/改名/改 tier/删除，
均同步 config.json；不热插图——`build_graph.py` 重建后入图）。检索结果（search/doc）携带 `tags`。

### 4.8 graph —— Phase 2 图检索（关系追溯）

先构建图（Kùzu，基于 canonical 与版本日历，确定性零 LLM）：

```bash
python scripts/build_graph.py                # 全量重建（约数分钟）
python scripts/build_graph.py --limit 2000   # 试跑前 N 条
python scripts/build_graph.py --stats        # 只打印现有图统计
```

查询命令（`doc` 参数支持完整 source_id 或 `repo#编号` 简写）：

```bash
# 核心链路：issue → 修复 PR → 落地 release（回答"是否已修复、修复在哪个版本提供"）
python skills/vllm-kb/client.py graph chain vllm-ascend#10700

# PR 视角：该 PR 修复的 issues + 落地 release
python skills/vllm-kb/client.py graph fixes vllm#50241

# 签名实体（算子/错误码/模型/版本）→ 提及它的 issue/PR
python skills/vllm-kb/client.py graph sig dispatch_ffn_combine
python skills/vllm-kb/client.py graph sig 561000

# 文档邻接（调试）与图规模
python skills/vllm-kb/client.py graph doc github:vllm-project-vllm:issue:10700
python skills/vllm-kb/client.py graph stats
# 标签 → 打标文档（Doc/Issue/PR）
python skills/vllm-kb/client.py graph tags HCCL
# 文档互证（Evidence）：与目标文档共享 ≥2 个实体（算子/错误码/模型/版本/接口/标签）的其他文档
python skills/vllm-kb/client.py graph evidence pdf:Atlas A3 中心推理和训练硬件 26.1.x npu-smi 命令参考 02
```

**REFERENCES 关联边**（2026-10 新增）：github_pull 采集 issue timeline 的 `cross-referenced`
事件（页面底部 "Mentioned in" 区）→ `extra.crossrefs` → 建图 REFERENCES 边（Issue/PR 四组合，
双端在库才建）。语义是**提及/引用，非修复**：PR 正文提到 issue 但没写 fixes 关键字（如
"tracked by #8938"）、issue 互引、上游仓引用等都入边。`graph chain` 输出 `related` 段——
**无 fixes 边 ≠ 未解决**（配置规避/使用错误/讨论结案都可能），此时看 related：已合并
（merged）的关联 PR 通常含修复或规避动作的线索。采集开关 `fetch_timeline`（默认开）；
存量回填：跑一轮增量拉取即对旧数据补采 timeline（1 请求/issue，限流内）。

**图内容说明**：Issue/PR/Release（修复链路）+ Operator/ErrorCode/Model/Version（签名实体）+
**Doc**（PDF 手册/Markdown 等非 github 文档）+ **Interface**（手册"命令格式"段提取的工具.子命令，
如 `hccn_tool.bandwidth`）+ **Tag**（文档级标签，两级分类）。文档经两条 `DOCUMENTS` 边入图：

- **错误码表 → ErrorCode**：`graph doc pdf:<手册>` 的 `documents` 含该手册定义的错误码；
- **命令格式段 → Interface**：`documents` 含该手册定义的接口/命令；
- **标签 → Tag**：`TAGGED_WITH` 边（最终标签 = 自动 − 排除 ∪ 人工，与入库一致；
  registry 全量标签也建节点——新增标签重建图即入图）。

错误码/命令与 GitHub issue 的 MENTIONS 共享节点——可回答"这个错误码在哪个手册定义、
社区哪些 issue 提到"、"查带宽用哪个命令、命令在哪本手册"。

图与检索 API 的关系：图由 `scripts/build_graph.py` 构建，`serve_api.py` 的 `/graph/*` 端点只读查询；
**更新图前请停止检索 API**（Kùzu 单写者，构建期间不可并发查询，见 §3.3）。

### 4.9 code-graph —— 代码图谱检索（gh-puller 接入）

代码知识图谱检索，与 `code`（本地版本化符号索引）**并列、能力互补不重叠**：

| 维度 | `code`（本地） | `code-graph`（gh-puller） |
|---|---|---|
| 强项 | 版本化定位、报错字面量索引、离线、跨版本 diff | 调用链/数据流、变更影响面、架构聚类、跨仓边、语义搜索 |
| 速度 | 毫秒级（本地 SQLite） | 秒级（gh-puller + 图谱查询） |
| 依赖 | 离线 | gh-puller HTTP 服务在线 |

**启用**：config `code_graph.enabled = true` + `base_url`（gh-puller Streamable HTTP 地址，
默认端口 8787，路径 `/gh-puller/graph`）。未启用时 `serve_api` 不注册 `/code-graph/*`
端点（404 比 503 干净）。**不可达时端点直接 503 + 引导用 `code` 命令查本地索引**——
本地无等价图谱能力，不走回退（回退无意义）。**工具级错误（未知函数/参数错）返回 400 +
上游错误详情**——服务健康、参数问题，换函数名形态重试即可，勿按服务故障处理。

**命令**（skill，`--graph-base` / `VLLM_KB_CODE_GRAPH_BASE` 独立寻址，缺省沿用 `--base`）：

```bash
# 搜函数/类/路由（BM25/正则/语义三模，优先于 grep 找定义）
client.py code-graph search "update settings" --repo vllm
client.py code-graph code-search "DispatchFFNCombine" --mode full --path-filter "^vllm_ascend"

# 调用链追踪（替代手写 grep 找调用关系）
client.py code-graph trace do_auth --direction outbound --depth 5 --mode data_flow
# trace 函数名双形态：短名（do_auth，bare name 匹配）或 search 返回的完整 qn（原样透传，
# 上游按 project+qn 精确回退——索引前缀是规范 qn 的一部分，勿手动剥离）。
# 裸短名同名多节点 → 返回候选列表（name/qn/file/lines）而非静默取其一；用候选完整 qn 精确重试。
# 属性/descriptor 节点（label 含 property）上游暂不支持直接追踪，400 错误给出宿主类引导
# （gh-puller 侧适配中）。

# Cypher 查知识图谱（多跳/聚合/跨服务分析）
client.py code-graph query "MATCH (n:Function)-[:CALLS]->(m) RETURN n.name, count(m)"

# 架构总览（聚类/边界/热点/层次/依赖）
client.py code-graph architecture --aspects all --path apps/
client.py code-graph architecture --aspects clusters boundaries hotspots

# git diff → 变更影响面（blast radius：变更波及的调用方）
git diff | client.py code-graph changes - --scope impact --direction inbound

# 探测 gh-puller 可达性
client.py code-graph health
```

**何时用 code-graph vs code**：

- 定位某版本某符号在哪定义、报错文本来自哪段代码 → `code`（本地快、版本化）；
- 谁调用了某函数、变更会影响什么、架构怎么分层、跨仓调用关系 → `code-graph`；
- gh-puller 不可达时降级思路：`code-graph` 503 → 改用 `code` 查本地索引（手动）。

### 4.10 行为遥测与置信度反馈（feedback_enabled）

行为遥测自动采集查询行为（不依赖用户主动反馈），离线推断三态反馈（hit/miss/unknown），
用时间维度指数遗忘更新每个 doc 的后验 Beta 分布，查询期新增 **w_hist 历史可靠度**因子
（后验下界 lb = mean - z·sd）并入 final 排序。**与 w_rel 正交，不乘进 w_rel**——
保护审计链（同一文档元数据不变时 conf 部分确定性可审计，w_hist 独立可解释）。

**启用**：config `confidence.feedback_enabled = true`。中间件全量记查询行为到独立
`data/telemetry.sqlite3`（不碰只读 `kb.sqlite3`）。会话归属经 `VLLM_KB_SESSION` 环境变量
→ `X-Session-Id` header 透传（agent 侧管），缺失回退 ip+时间窗。

**探索/测试行为打标（probe，不进反馈推断）**：agent/新同事初接触 skill 时的验证安装、
复现文档示例、占位测试查询会污染 w_hist 统计。双层防线：

- **显式约定（主）**：探索请求带 `--probe`（client.py）或 `VLLM_KB_PROBE=1` →
  `X-VLLM-KB-Probe: 1` header → 遥测 `probe=1`；
- **启发式兜底（辅）**：查询正规化后精确命中 SKILL.md 范例词/占位探测词（test/hello/
  测试 等）→ 自动 `probe=2`，覆盖未读约定的探索行为。

打标不删除（审计可逆）——推断层 `build_feedback.py` 排除 `probe≠0` 事件并输出排除统计，
规则调整后重跑即恢复；原始行保留供易用性分析（哪些命令被探索过）。

**数据流（三段分离，审计可逆）**：

```
serve_api (只读)                      离线周期
┌─────────────────────────┐         ┌───────────────────────────┐
│ middleware 记原始行为    │         │ scripts/build_feedback.py │
│ → telemetry.sqlite3     │ ──────▶ │ 会话重建+行为推断三态      │
│ (独立库,不碰 kb.sqlite3) │         │ → confidence_feedback.json │
│                         │         │ → knowledge_gaps 表        │
│ compute_confidence      │         └───────────────────────────┘
│ + w_hist(读 feedback.json)│ ◀────── 重启 serve_api 生效
│ final=sim^γ·conf^(1-γ)·lb^σ │
└─────────────────────────┘
```

**离线推断**：`python scripts/build_feedback.py` —— 扫遥测库重建会话序列，
按行为模式推断三态（权重 ≤ 0.5，自证循环阻尼）：

- search/signature 命中后拉 doc 不重查 → 弱正 hit（0.3）；
- signature 命中后会话直接结束 → 弱正 hit（0.3）；
- 有 doc 命中后调 code/diff → 中正 hit（0.5）；
- 60s 内改述重查 → 弱负 miss（0.3）；
- 命中后零后续 → unknown（不进 n_eff，单独计数）；
- 无命中的查询进缺口检测，不进后验。

**后验更新（时间维度指数遗忘）**：`a <- a×2^(-Δt/HL) + w×hit`，HL 复用 config
`half_life_days`（非事件次数衰减——克服版本漂移且冷门案例不异常）。seed=1+1 随遗忘
衰减，HL 决定失效。**seed 强度 1+1=2，数据部分需 ≥3 超先验 1.5 倍才主导后验**
（隐含前提：平均确认频率约每季度 1 条；冷门 domain 检索频率过低时 supported 状态
不会出现——符合设计，冷门且无反馈证据的文档保持中性不被误杀）。

**w_hist 三段式**（当期值不缓存，每次查询重算）：

| n_eff | w_hist | history_flag |
|---|---|---|
| `=0` | 1.0（中性，不用 seed 套 lb——避免误杀新文档） | `new` |
| `0 < n_eff < n_min` | `lb`（正常算） | `accumulating` / `evidence_thin` |
| `n_eff ≥ n_min` | `lb` | `supported` / `used_but_unconfirmed` / `failing` |

**关键约束**：

- **z（检索侧排序）vs p_min（消费侧决策）分工**：vllm-kb 只输出 lb，不参与消费侧决断；
- **只标注不拦截**：history_flag 供消费侧决策，vllm-kb 永不过滤候选；
- **n_eff 是加权观察非次数**：w=0.4 的推断只贡献 0.4；
- **unknown 不进 n_eff**：统计正确，单独计数。

**知识缺口**：审核工作台 → "知识缺口" tab，展示同签名跨 ≥3 会话反复查无果的缺口
（hard_gap 强签名零命中 / soft_gap 命中无结论 / quality_gap 弱签名零命中）。
serve_api 不暴露缺口端点（缺口不进检索）。

## 5. 故障处理推荐流程

```
0. context   文档能力发现（硬件/组件/领域名词问题或 signature/search 无强命中时先做）：
             "HCCL 超时" → 命中 HCCL(领域) + 超时排查/命令参考(作用) → 先读对应文档（tags docs / doc）
1. signature  贴原始报错 → 提取签名 + 精确命中（最直接线索）
2. title      拿签名/信号词做标题精确检索 → 找已知 issue
3. search     组件:版本 问题描述 → 语义检索 + 置信度分解，补齐相似问题
4. version    确认部署版本形态（正式/rc）
5. code       按部署版本定位源码 → 判断是否版本相关 bug
6. doc        读 issue 全文 → 结合 resolved 状态与修复 PR 给结论
```

**安全边界（本 skill）**：所有输出不含服务器文件路径、不暴露内部存储结构；知识库对内部文档
只提供检索结果（标题/文档id/片段），无"列出所有文件/文档"指令面（可用工具仅 Bash，
`tags list` 是能力目录、`tags docs <标签>` 是按标签过滤的检索，均非文件枚举）。

**信息缺失与未知名词处理**：关键事实（部署形态、卡数/节点数、部署版本、组件范围）或名词含义
无法确认时，先问用户，不要假设通用拓扑或自行脑补（如日志中 timeout 7 次是否是"8 卡缺 1"，
取决于机器卡数）；用户也不知道时才基于上下文推断，并标注"这是推断，未经确认"。

**未命中反查**：检索不到时用 `code --in-file --per-version` 逐版本对比、`diff` 命令做跨版本
精确 diff（§4.5）定位修复引入版本，仍无果才判定"社区无修复"并说明检索范围。

## 6. 远程部署（存算分离）

数据（向量库、索引、图）放远程服务器，本地 skill 只留 `client.py` + `SKILL.md`（约 68KB）发 HTTP 查询。

```bash
# 远程服务器
rsync -av vllm-kb/ root@<remote>:/opt/vllm-kb
rsync -av vllm-kb/data/ root@<remote>:/data/vllm-kb/      # 一次全量，之后增量
pip install fastapi uvicorn lancedb kuzu
VLLM_KB_DATA_ROOT=/data/vllm-kb python scripts/serve_api.py --host 0.0.0.0 --port 8000
#   VLLM_KB_DATA_ROOT 指向 data/ 目录本身（data/lancedb -> {root}/lancedb，不再包一层 data/）

# 本地 skill
export VLLM_KB_BASE=http://<remote>:8000
python skills/vllm-kb/client.py search "..."    # 全部命令走远程
```

**迁移打包（业务环境重新嵌入，不传向量库）**——`scripts/pack_migrate.py`：

```bash
python scripts/pack_migrate.py --with-graph --out deploy/migrate.tar.gz  # 最小集打包（~90MB）
python scripts/pack_migrate.py --steps        # 业务环境重建步骤（改 embedding.base_url → --rebuild）
python scripts/pack_migrate.py --with-code    # 无外网时连带 1.7GB 代码快照
```

包内含 canonical（重嵌入唯一输入）+ 业务数据 + 图；**不含 lancedb（向量库，业务环境 `--rebuild`
重建，干净库约 0.8GB）与 kb.sqlite3（--rebuild 自动重建）**。详见脚本 docstring。

**辅助脚本**：

```bash
python scripts/deploy_remote.py --gen-config    # 生成远程 config（已去 token/api_key）
python scripts/deploy_remote.py --pack-data     # 打包数据成 tar.gz（大数据推荐 rsync/scp -r）
python scripts/deploy_remote.py --print-steps   # 部署步骤说明
```

`--pack-data` 默认**只打只读 API 需要的部分**（`lancedb` / `code` / `compatibility` / `graph` /
`kb.sqlite3`）——这样的包在远程**无法重建**（`build_kb.py` 跑不出 Markdown 文档），审核台也没有图片预览：

| 需求 | 参数 |
|---|---|
| 只跑只读 API（默认） | `--pack-data` |
| 审核台图片预览 | `--pack-data --include-assets`（含 `assets/` + `review.sqlite3`） |
| 远程重建（改了解析逻辑重跑） | `--pack-data --include-assets --include-imports`（含 `imports/`） |
| 需要 `raw/` 原始快照 | 追加 `--include-raw` |

缺 `imports/` 时 `MarkdownSource` 会回退 `assets/md` **扁平副本**（见"Markdown 图片处理"）。

数据更新在"拥有数据"的一端跑流水线，之后增量 rsync 到远程即可。

## 7. 常见问题

**Q: 查询结果 w_ver 都是默认值？**

A: 依次检查：

1. 语义检索要传目标版本（`--version` 或 `组件:版本` 前缀）；
2. 需先生成**分仓**版本日历（`python scripts/build_release_calendar.py --all-repos`，
   生成 `release_calendar.{repo_slug}.json`）——"修复落地版本"上界在**查询期**按文档
   所属仓库的日历实时计算（resolved_at → 该日期前最近发布版），不落库、不随 API 返回；
3. 文档自身无版本信号（标签/正文未声明版本、未解决无 resolved_at）时无法映射，w_ver 取
   默认值——这属于数据侧信号缺失，不是配置问题。

**Q: 向量库体积异常大（数十 GB）？**

A: 干净向量库约 **0.8GB**（122K chunks × 1024 维）。体积膨胀是 LanceDB **历史版本累积**所致：
每次入库提交新版本 manifest，旧版本从不清理（全量嵌入可累积 4 万+ 份历史快照，占体积 98%）。
数据完整时用 `db.open_table('chunks').cleanup_old_versions()` 清理即可恢复 ~772MB；
业务环境 `--rebuild` 全新库天然无历史版本。也可存算分离把数据放远程，或换更小维度模型。

**Q: 构建日志说"向量后端降级为纯 Python"，但 lancedb 明明装了？**

A: 这两件事不矛盾——**装了 ≠ 能加载**。回退告警现在会如实区分：
`pip list` 里有 lancedb，但 `python -c "import lancedb"` 抛
`ImportError: DLL load failed while importing lib: 应用程序控制策略已阻止此文件。`
（`WinError 4551`）说明原生 DLL 被**安全策略**拦了——Windows 智能应用控制
（Smart App Control）/ WDAC / AppLocker 的典型症状（pyarrow 同样被拦，lancedb 依赖它；
PyMuPDF/Kùzu 通常正常，所以是选择性拦截，不是全局限制）。`pip install` / 重装**无用**。

要紧的不是那句提示，而是**换了库**：正式库 `data/lancedb/`（列式数据集，带 ANN）、
回退库 `data/lancedb_py.json`（单 JSON，暴力余弦）。回退库若为空，既有向量在本次检索里
**全部看不到**（告警会在正式库非空时明确点出这一点）。二选一：

1. **放行原生 DLL**（安全策略里给 pyarrow/lancedb 目录开例外，或换到不受策略约束的环境/容器）
   → 然后正常重跑；
2. **显式接受回退后端**：设 `storage.vector_backend=python`，再
   `python scripts/build_kb.py --rebuild` 把向量全量重建到回退库（万级 chunk 内可用，
   再大就不合适了）。

排查顺序：`python -c "import lancedb"`（看真实异常）→ 确认 `storage.vector_backend`
是不是被显式设成了 python → 看正式库目录大小与 `data/lancedb_py.json` 是否存在。
构建汇总里的 `vector_backends` 会告诉你本轮实际用了哪个后端。

**Q: 真实业务环境 SSL 被禁 / 证书不受信？**

A: 联网脚本（代码快照/版本日历/配套矩阵）加 `--insecure` 跳过证书校验；若域名不可达
需用业务侧 http 镜像（`--github-base/--quay-base/--base-url`，或环境变量 `VLLM_KB_GITHUB_BASE` 等）；
`embedding.base_url` 可写裸 `ip:port` 自动补 `http://`。
注意 `REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE` 环境变量会**覆盖** `--insecure`
（requests 的 `merge_environment_settings` 会用环境 CA 串复活 SSL 校验，
业务环境 MITM 场景典型症状：`CERTIFICATE_VERIFY_FAILED ... HTTPSConnectionPool`）。
本库已内置请求级 `verify=False` hook 修正（`vllm_kb/net.py`，端到端测试固化
`tests/test_net_tls.py`）；若用自己的脚本复用 `get_session(insecure=True)` 即可，
不要裸设 `session.verify = False`（会被环境变量覆盖）。

**Q: 全量部署中途报 `SSLEOFError: EOF occurred in violation of protocol` 然后部署中止？**

A: 这是**传输层**故障（对端/代理把 TLS 隧道掐了），不是限流、不是 token、也**不是
`--insecure` 能治的**（它关的是证书校验，失败发生在校验之前）。现在：

1. `build_kb.py` **按来源隔离**——某个来源失败只告警并继续其余来源，本地文件来源
   （PDF/Markdown/Word/Excel）不依赖网络，照常更新；
2. 退出码：部分失败 = `0`（降级完成，日志有醒目告警），**所有来源都失败才 = 1**。
   所以 `maintain.py deploy` 不会再因一次网络抖动整体中止；
3. 传输类错误按 `transport_retries`（默认 6）**独立预算**重试，每次换新连接、
   逐次打日志；单页体积可用 `graphql_page_size`/`comments_page_size` 调小（弱网推荐
   25/50），游标与页大小无关，**中途调小不会让断点失效**；
4. 中断本身不丢进度：raw 与 checkpoint 每页落盘，重跑同一命令**从游标续传**；
   想先把已拉数据入库再继续拉，用 `build_kb.py --skip-pull`；
5. requests 侧另有一层**直连兜底**：`SSLError`/`ProxyError` 且确有代理生效时，自动改用请求级
   直连重试一次（详见下一条 FAQ 的实测数据）。

**Q: quay / GitHub 报 TLS 超时、`SSLEOFError`、`handshake operation timed out`，怎么办？**

A: 先怀疑**本机系统代理**——它是这类故障最常见的元凶，而且**会悄悄生效**：

```bash
# 看当前实际生效的代理（没设任何 env 变量也可能有值：来自 Windows 注册表）
python -c "import requests.utils; print(requests.utils.getproxies())"
# 例如 {'http': 'http://127.0.0.1:7892', 'https': 'http://127.0.0.1:7892'}
```

实测该本地代理对 `quay.io` 是 **TLS 握手超时**（`_ssl.c:990`，`--insecure` 无效——失败在
证书校验之前），对 GitHub 是**间歇性 `SSLEOFError`**；而 `https://quay.io/` 直连 1~16s 即 200。
**注意方向**：这种情况下"把 `HTTP_PROXY`/`HTTPS_PROXY` 指向同一个地址"是**空操作**
（`requests` 本来就在用它），要的是**绕开**：

```bash
export NO_PROXY=quay.io,api.github.com,github.com,codeload.github.com,objects.githubusercontent.com
# PowerShell: $env:NO_PROXY="quay.io,api.github.com,github.com,codeload.github.com"
```

**requests 侧已自动直连兜底（部署时通常不用再手工设 `NO_PROXY`）**：`vllm_kb.net.get_session()`
在遇到 `SSLError`/`ProxyError` 且**确实有代理生效**时，会用请求级
`proxies={"http":None,"https":None}` **直连重试一次**（只对 GET/HEAD；调用方已显式直连、
或环境里根本没有代理时不重复请求）。实测数据（本机代理 `127.0.0.1:7892`）：

| 请求 | 走系统代理 | 直连 |
|---|---|---|
| `GET https://quay.io/v2/auth`（配套矩阵入口） | **3/3 失败**：`SSLError ... SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1007)')`，每次约 10s | **3/3 成功**，0.8s |

现场表现是 `[maintain] ❌ 配套矩阵 退出码 1`（该步骤非致命，但一次整跑约 85 分钟）。
除这层兜底外，`get_quay_token` 自身也带 3 次重试预算（每次间隔 2s/4s，日志会打出
`[quay] /v2/auth 失败（第 n/3 次）…`）。**超时类**（`handshake operation timed out`、读超时）
不走这条快速兜底（要等满 timeout 才会失败），弱网下仍建议手工设 `NO_PROXY`。

- `NO_PROXY` 对**注册表探测到的代理同样生效**（`should_bypass_proxies`）→ 变成直连；
- `quay.io` 会一并覆盖 `cdn01.quay.io` / `cdn02` / `cdn03`（quay 的 manifest/blob 由 302
  跳到这些 CDN，同一域名后缀）；
- Clash 之类的 TUN + fake-ip 工具即使"直连"也仍走它的隧道（DNS 返回 `100.64/10` 段假 IP），
  所以 `*.quay.io` 的可用性取决于它选中的节点——**换节点/换规则，或直接在 Linux 服务器上跑**更省事。

若仍偶发失败：quay 的 CDN 在抖动链路上实测同一 URL 有 **0.6s 成功**、也有 **90s 读超时**
（连续 3 次全废），所以 `fetch_quay_tags.py` / `build_companion_matrix.py` 的 quay 预算已放宽到
`timeout=60`、`max_retries=4`（指数退避）；单个 tag 拿不到 config 只在日志告警并**降级继续**
（不致命）。`data/cache/` 会累计：fork 层 SHA（永久）、GitHub releases（7 天）、requirements
（永久）、tag→commit（永久）——**重复跑同一命令会逐步收敛**；但**镜像 config 不进缓存**，
那部分缺口只能靠重跑补齐。

而且**重跑是收敛的、不会回退**：矩阵写回时保留行内已有非空字段。实测连续两跑：
`cann` 31→34、`pytorch-ascend` 58、commit 溯源 8→14（含 2 条 fork 的 `vllm_sha`），
某轮 `v0.23.0` 的 config 拉取失败，但该行仍完整保留上一轮的
`cann=9.1.0 / vllm=0.23.0 / pta=2.10.0.post4 / vllm_commit=0fc695fc6d1d`。
所以**弱网下最省事的做法就是隔一阵重跑一次 deploy**，让缺口被逐轮填上
（缺口清单每轮都会打印，剩下的多为老版本镜像 Env 里本就没有 CANN 证据，需人工对照
quay build history 补）。

**Q: 版本日历变成 `{"releases": []}`，置信度的版本上界（w_ver）全退化了？**

A: 这是**旧版本脚本的已知缺陷**（已修）：`build_release_calendar.py` 的 `fetch_releases`
在某一页拉取失败时只是 `[warn]` 并 `break`，返回已经抓到的部分（往往一个都没有），
随后**照样写盘覆盖**——于是"抓取失败"被写成了"这个仓库没有 release"。实测本机代理掐 TLS 时，
一次 `maintain.py deploy` 就把带 `v0.23.0` 的日历写成了 113 字节的空文件，
下游 `w_ver` 版本上界和图里 `MERGED_IN → release` 关联**一起消失**。

现在有两道防护，都**保留上一版日历**：

1. 任一页失败 → 抛 `CalendarFetchError`（附已抓到部分），**不落盘**（残缺结果若落盘，
   日历会比原来更小，旧版本文档的 tag→日期映射会丢）；
2. 写盘前 `would_drop_releases` 检查：新结果为空而现有文件非空 → `CalendarWriteRefused`
   （首次生成、现有文件缺失/损坏时不拦）；
3. 失败时脚本以退出码 1 结束并打印原因（`maintain.py deploy` 里该步骤是非致命的，
   只告警继续）。

恢复办法（先修网络再重跑）：

```bash
export NO_PROXY=api.github.com,github.com            # 绕开会掐 TLS 的本地代理
export SSL_CERT_FILE=".venv/Lib/site-packages/certifi/cacert.pem"   # 可选：现在会自动叠加 certifi
python scripts/build_release_calendar.py --all-repos
```

> **urllib（日历/代码快照/配套矩阵）与 requests（拉取链路）证书来源曾经不同**：requests 自带
> certifi，urllib 走 `ssl.create_default_context()`，而某些 venv 的 OpenSSL 默认 CA 列表**不足以
> 验证 GitHub/quay**（本机实测只有 26 张 CA，缺 GitHub 的根证书）→ 所有 urllib HTTPS 报
> `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate (_ssl.c:1007)`。
> **现已自动修复**：`vllm_kb.net.ssl_context()` 会给默认上下文**叠加 certifi 根证书**
> （26 → 138 张，只增不减），版本日历 / 三个快照脚本 / 一切 `get_opener()` 调用都生效——
> 不需要再设 `SSL_CERT_FILE`，更不需要 `--insecure`。若业务环境要求只用内部根证书，
> 显式设 `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` 即可**关掉**这个叠加（完全尊重你的选择）。

⚠️ 日历有**两处消费者，恢复后的影响不同**：

- **查询期现算**（`confidence.version_at_date`）：`w_ver` 版本上界**立即恢复**，
  不需要重建任何东西——但日历被写空的那段时间里的查询，置信度是退化的；
- **建图时固化**（`MERGED_IN` 的 release_tag）：只有"在日历为空的那段时间里重建过图"，
  图里的 `release_tag` 才会是空的（`/graph/chain` 返回 `release_tag: null`），
  那时要重跑 `build_graph`（需先停 `serve_api`——Kùzu 单写者）。图若没在那段时间重建过，
  里面的 release 关联仍是上一次成功建图时固化的完整数据（实测本机：日历被写空的当天，
  图仍是更早一次成功建图的结果，`MERGED_IN` 与 release_tag 完好）。

**Q: 部署到"代码快照（vllm 主仓，companion 对应）"时崩了：`ValueError: invalid literal for int() with base 10: '0rc3'`？**

A: 已修（旧版本的 `build_vllm_snapshots.companion_vllm_versions` 用
`int(x) for x in version.split(".")` 排序，而配套矩阵里的 `vllm` 值**本来就可能带后缀**——
镜像的 `VLLM_TAG` 就是 `0.11.0rc3` 这种；该步骤是非致命的，所以表现为"静默没有快照"）。
现在排序键用 `confidence.parse_version` 取数值核心，后缀折算成序号
（预发布 `rc/a/beta/dev` < 正式 < `post`；同后缀比尾号 `rc3 < rc10`；无法解析的值排最后），
不丢、不炸。

顺带一提：这一步依赖**配套矩阵文件**，所以它与配套矩阵的产物同生共死——矩阵若没生成
（见上一条），这里就是空的待处理列表。

**Q: deploy 用的是哪个 python？为什么和我在 venv 里跑的单步脚本表现不一致？**

A: `maintain.py` 用 `sys.executable` 启动每个子步骤，也就是**用什么 python 跑
`maintain.py`，整条流水线就用那个解释器**。实测踩过：用系统的
`...\Python310\python.exe` 跑 `deploy`（仓库 venv 在 `.venv`），同一步在两边依赖版本不同、
报错也难归因。现在脚本会检测：仓库里存在 `.venv` 却用了别的解释器时打印醒目提醒
（部署机上没有 venv 则不打扰）。规范写法：

```bash
.venv/Scripts/python.exe scripts/maintain.py deploy      # Windows
.venv/bin/python scripts/maintain.py deploy              # Linux/macOS
```

**Q: 矩阵跑到后半段（约一小时）全是 `401 Client Error: UNAUTHORIZED for url: .../v2/ascend/vllm-ascend/blobs/...`？**

A: 是 **pull token 过期**，不是那些镜像有问题。quay 的 token 是 JWT，实测有效期
`exp - iat = 3600` 秒；而一次完整矩阵实测要跑 5149 秒（重试很费时）→ 60 分钟后所有
blob 请求都 401。症状很好认：**从某个 tag 起集体 401**，且这些行最后只能靠
"启发式(版本号跟踪上游)/release说明" 填 `vllm`，`cann` 大面积为空。

现在收到 401 会**自动换新令牌并当场重发**（不消耗重试预算），一个进程跑完全程没问题；
`clone 层` 扫描同理。若日志里 401 之后没有紧跟"换新令牌重发"，说明是旧版本脚本。

顺带：本步与 `GITHUB_TOKEN` 有关——未配置时脚本会**跳过** tag→commit 解析
（未认证 GitHub 限流 60 次/小时不够），日志会打印"跳过 N 个 tag 的 commit 解析"，
配上 token 重跑即可补齐；已缓存的不受影响。

token 的三级来源（后者覆盖前者）：**`data/secrets.local.json`（审核工作台写入）→ config.json 的
github source（`token` / `token_env`）→ 同名环境变量**。实测踩过：secrets 文件里已配好
GITHUB_TOKEN，矩阵仍打印"未配置…跳过 N 个 tag"——因为该脚本**刻意不依赖 AppConfig**
（只读 config.json 的少数字段），拿不到 `AppConfig.load()` 里那次 secrets 注入。
现已在该脚本内显式补上 `load_secrets()`（只注入未设置的环境变量），行为与其余脚本一致。
注意 `secrets.local.json` 只认三个规范 key（`EMBEDDING_API_KEY` / `OCR_API_KEY` / `GITHUB_TOKEN`）——
config 里把 `token_env` 写成别的名字时，只能靠真实环境变量提供。

**Q: 矩阵生成慢 / 卡在"扫描 clone 层固化锁定 commit"？**

A: fork 行要下载 ~75MB clone 层 blob，业务环境慢链路需数分钟——脚本每 10MB 打一行进度，
静默 ≠ 卡死。扫描结果按层 digest 缓存到 `data/cache/fork_sha.json`（层不可变，
永久有效），**第二次运行起零层下载**；GitHub releases/requirements 同样有跨运行缓存
（全命中时 API 请求为 0）。怀疑缓存脏数据时 `--refresh-cache` 强制重拉。

**Q: 离线能用吗？**

A: 能。采集完成后全部检索离线；嵌入可用 `echo` provider 离线自测（效果粗糙）。

**Q: kb 检索能命中（title/search），但 graph chain/fixes 查不到？**

A: 典型的 **kb↔canonical 不同步**：`canonical.jsonl` 是 `build_graph` 与 `--rebuild` 的
**唯一事实源**——kb 有、canonical 无的文档，图里没有（图从 canonical 建）、全量重建也会丢
（rebuild 从 canonical 重嵌）。常规增量入库不会漂移（pipeline 先 upsert canonical 再 ingest）；
历史旧版流程 / 早期 `--recanonicalize` 参数（raw 快照不全）可能造成。
修复（`scripts/backfill_canonical.py`，从 kb.sqlite3 重建缺失的 canonical 行，无需网络）：

```bash
python scripts/backfill_canonical.py                                    # dry-run：打印缺失清单
python scripts/backfill_canonical.py --write                            # 回填全部缺失（幂等）
python scripts/backfill_canonical.py --doc github:vllm-project-vllm-ascend:pr:9749 --write  # 只补单条
# 然后：
python scripts/build_kb.py --skip-pull     # 回填文档重入库（body 为 chunks 拼回，会触发重嵌，幂等）
# 停 serve_api → python scripts/build_graph.py → 重启 serve_api
```

回填行 `body` 由 chunks 按序拼回（与 `/doc` 端点同法，chunk 重叠有少量重复，不影响建图关系提取）；
`extra` 原样带出（图构建依赖 `repo/github_number/merged_at` 建 FIXES/MERGED_IN 边）；
`tags` 取 `doc_tags.auto_snapshot`（canonical 语义 = 自动标签）。

若 **kb 中也不存在**该文档（从未采集过，如业务数据历史缺失的单条 PR），backfill 无法补——
需重新拉取：`--incremental`（补不到历史单条，见 §2.1）或全量重拉（删 raw + checkpoint 重跑）
或手补该条的 raw 快照（`data/raw/{source_id}/{prs,comments}/{number}.json`，格式同 REST 响应）
后再 `build_kb.py --skip-pull`。

**Q: 图打不开 / build_graph 失败，数据根路径含中文？**

A: Kùzu 图库路径**不能含非 ASCII 字符**（中文、emoji 等）——`data/graph` 或存算分离的
`VLLM_KB_DATA_ROOT` 若在中文路径下（如 `C:\Users\张三\...`），建图/图查询会失败。
把数据根移到纯 ASCII 路径（如 `D:\vllm-kb-data`）后重建图（`scripts/build_graph.py`）。

**Q: PDF 重新入库很慢，怎么跳过已解析的？**

A: 解析中间产物已按资产 sha256 缓存（`data/parsed/pdf/<asset_id>.extract.json`），
资产未变时自动复用（进度行标注"缓存命中"）；想强制重新解析（如 PyMuPDF 升级），
删除 `data/parsed/pdf/` 目录即可，资产层与 kb 数据不受影响。缓存带 schema 位——
**升级提取逻辑**（不只是升 PyMuPDF）时把 `_PDF_EXTRACT_SCHEMA` +1，旧缓存会自动失效重解析。

**Q: 想加自己的故障记录（excel/markdown/word）？**

A: config.json 的 `sources` 加条目即可：`{"id":"engineer-troubleshooting","type":"excel",
"path":"data/imports/...xlsx","enabled":true}`（schema-free 导入，见 §2.4）、
`{"id":"mynotes","type":"markdown","path":"data/mynotes","enabled":true}` 或
`{"id":"cases","type":"word","path":"data/imports/word","enabled":true}`（见 §2.5）；GitHub 源的
增量拉取见 §2.1（`--incremental` / `--pull-missing` / `--numbers` / `--numbers … --force-numbers`）。
