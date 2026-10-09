# vllm-kb 数据流说明（入库 / 查询）

本文档回答两个常规操作的核心问题：

1. **入库**：数据从哪里来？经过什么处理？最终存在什么地方？
2. **查询**：agent 的请求从哪个 API 接口进来？后端查询哪些数据库？

配套代码：`vllm_kb/pipeline.py`（入库入口）、`vllm_kb/ingest.py`（落库）、`vllm_kb/api.py`（查询服务组装）、
`vllm_kb/md_images.py`（Markdown 图片引用扫描/重写）、`skills/vllm-kb/client.py`（agent 侧客户端）。查询端点按检索域拆分在
`vllm_kb/api_meta.py`（辅助）/ `api_community.py`（社区+文档）/ `api_code.py`（本地代码仓）/
`vllm_kb/api_code_graph.py`（gh-puller 代码图谱，可选启用）/ `api_image.py`（请求期图片 OCR，
有 image source 时启用），`api.py` 只负责组装与出口脱敏。

## 1. 总览

```
┌────────────────────────── 入库（写） ──────────────────────────┐
│  GitHub 社区 ──pull──▶ data/raw/{source_id}/ ─canonicalize─┐    │
│  业务文件   ──资产+解析─▶ data/assets/ + data/parsed/ ──────┼──▶ canonical.jsonl
│                                                           │    │
│  canonical.jsonl ──chunk──▶ embed ──▶ LanceDB 向量库            │
│                └──────────▶ SQLite（docs / chunks_fts /         │
│                             chunks_meta / doc_tags /            │
│                             doc_verification）                  │
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
│  ├─ /ocr               ──▶ 外接 OCR 服务（请求期，无本地存储） │
│  └─ 出口统一脱敏（sanitize）→ 返回 agent                        │
│                                                                │
│  行为遥测（feedback_enabled 时）：中间件记查询行为              │
│    ──▶ data/telemetry.sqlite3 ──离线推断──▶ confidence_feedback │
│    ──▶ 查询期 w_hist 因子（见 3.6）                             │
└──────────────────────────────────────────────────────────────┘
```

## 2. 入库数据流

### 2.1 GitHub 社区来源（issue / PR / comment）

触发命令：`python scripts/build_kb.py`（可加 `--limit N` / `--incremental` / `--pull-missing` / `--numbers` / `--numbers … --force-numbers`）。
全量部署 / 增量更新另有 `python scripts/maintain.py deploy | update` 单指令入口（自动编排多步、自动降级、
`--skip-graph`/`--skip-fts`/`--skip-calendar`/`--skip-matrix`/`--skip-code-snapshots`/`--all-code` 细分裁剪，见使用指南 §2.0）。
拉取模式（互斥）：**断点续传**（默认，done 后跳过）/ **`--incremental` 时间窗增量**（近期新增 +
已拉条目远端 updatedAt 更新时重拉覆盖状态/正文/评论，含 open→closed）/
**`--pull-missing` 补差**（从头枚举，跳过 raw/checkpoint 已有，只拉缺失——补历史旧条目）/
**`--numbers` REST 单条**（指定编号走 `/pulls/{n}`→`/issues/{n}`，无需 GraphQL token；
**`--force-numbers`** 强制重拉，忽略已有覆盖 raw+评论）。

| 阶段 | 处理 | 产物 |
|---|---|---|
| 1. 拉取 | `GithubSource.pull()`：REST/GraphQL（issues 含 PR，PR 带 head 三元组 `head_repo/head_branch/head_sha`——fork PR 的来源仓/分支/锁定 commit，与 fork 快照对齐）+ 评论；限流、重试、`data/checkpoints/{source_id}.json` 断点续传；raw 与 checkpoint 均记 `updated_at`，增量窗口内已拉条目远端更新（含 open→closed）自动重拉覆盖；`--numbers --force-numbers` 强制重拉单条 | `data/raw/{source_id}/`（如 `data/raw/github`、`data/raw/vllm-ascend`，子目录 `issues/` `prs/` `comments/`）原始 JSON 快照（**事实源**，可重放） |
| 2. 规范化 | `src.canonicalize()`：原始 JSON → 统一 `KbDocument`（source_id / title / body / 组件 / 版本区间 / status / extra） | 追加/upsert 到统一 `data/raw/canonical.jsonl`（按 source_id 幂等） |
| 3. 入库 | `ingest_docs()`（见 2.6） | LanceDB 向量 + kb.sqlite3 |

#### 拉取的失败分类与降级（部署韧性）

失败按**可恢复性**分三类，处置方式完全不同（对应外网/代理环境实测到的
`SSLEOFError: EOF occurred in violation of protocol` 事故）：

| 类别 | 判定 | 处置 |
|---|---|---|
| **传输类** | 连接被拒/超时/TLS 隧道被掐/读 body 中途断流（`RequestException` 及其子类，含 `SSLEOFError`、`ChunkedEncodingError`） | 按 `transport_retries`（默认 6）**独立预算**重试，指数退避 + 抖动、上限 `transport_backoff_max_seconds`；**每次重试前 `session.close()` 丢弃连接池换新连接**（隧道被掐后池中连接可能已损坏）；每次重试打一行日志；耗尽抛 `GithubUnreachable` |
| **限流 / 5xx / 200 但非 JSON** | 服务端明确响应（主限流 HTTP 200 + body errors、次级限流 403/429 + `Retry-After`、网关页） | 按 `max_retries`（默认 3）重试；限流按 `Retry-After` > `X-RateLimit-Reset` > 指数退避精确等待；耗尽抛 `GithubUnreachable` |
| **永久错误** | 4xx（401 token 失效 / 403 非限流 / 422 查询非法） | **不重试**，立即失败并打印状态码语义（重试无用），抛普通 `RuntimeError` |

- **传输预算与限流预算独立**：否则限流等待会把传输重试额度挤掉（反之亦然）；
- **读 body 也在传输重试单元内**：大页响应读到一半断流是 `ChunkedEncodingError`，
  放在单元外就永远不会被重试（GraphQL 查询只读，整页重发安全）；
- **单页体积可调**：`graphql_page_size` / `comments_page_size`（默认 100/100）——单页
  = 页大小 × 每条最多评论数，弱网下调小可显著降低"大传输被掐"概率；游标与页大小无关，
  **中途调小不会让 checkpoint 失效**；
- **来源级隔离**：`run_build` 逐来源 `try/except`，某来源失败只告警并继续其余来源
  （与 `collect_docs` 既有行为一致）；失败进 `failed`、成功计数进 `sources_ok`，退出码
  由 `_report_and_exit` 决定：全成功/部分失败 = 0（部分失败打印降级告警，`--strict` 时 1），
  **所有来源都失败 = 1**。这样"GitHub 掉线"不会连带牺牲不依赖网络的本地文件来源，
  也不会把已成功入库的成果变成一次失败部署。

> **只再生 canonical（不入库）**：`scripts/build_canonical.py` 遍历来源 canonicalize → upsert
> canonical.jsonl，不 ingest——提取逻辑（版本/kind/组件/标签规则）升级后先跑它再 `build_graph.py`
> 建图，无需重嵌向量；`build_kb.py` 内部复用同一 canonical 处理（`pipeline.upsert_unified_canonical`）。

### 2.2 业务来源（PDF 手册 / Markdown / Word 文档 / Excel 登记表 / 截图 OCR）

文件放 `data/imports/{pdf,md,xlsx}/`（截图走 images source），config 启用对应 source 后跑
`python scripts/build_kb.py`（注意：本地文件导入**不要**用 `--skip-pull`，会跳过资产复制，见使用指南 §2.3）。

| 阶段 | 处理 | 产物 |
|---|---|---|
| 1. 资产复制 | `BaseSource.pull()` 把导入文件复制进资产层；**`canonicalize()` 也会落原件**（§1.6：`--skip-pull` / `build_canonical.py` 只跑 canonicalize，不落盘就断了 `asset_id → asset_registry.rel_path` 预览链） | `data/assets/{pdf,md,word,excel,images}/`，sha256 命名不可变（**资产路径不进检索库**，只存 asset_id） |
| 2. 解析 | PDF 文字层 + 表格提取（**文档集由 `imports/pdf` 决定**，资产层只作回退；历史版本升级后资产层独有的族会打 stray 告警）；Markdown 正文 + 图片收集（`md_images.py`：行内/引用式/HTML 全形态、```/`code` 代码区跳过、**未识别形态也占位**；引用的本地/base64 图片即时 OCR；内嵌图按内容寻址命名；同名 stem 用**相对路径指纹**消歧 `md:<stem>--<sha8>`；imports 缺失时回退 `assets/md`（**按版本族收敛**：`case.md` 与 `case.<sha12>.md` 是同篇不同版本，只取最新一份且 `source_id` 取族名 `md:case` 以防 id 漂移）并按文件名反查图片，`extra.quality.source_mode` 标记）；Excel **一个文件一篇**（body = `## <sheet 名>` + 每个非空行按列序拼接；分块按行，见 §2.2a）；**Word**（`python-docx` 按文档顺序单遍解析：标题样式 `Heading 1-6`/「标题 1」/`Title`/`Subtitle` → Markdown `#` 层级，**自定义标题样式名靠 `w:outlineLvl` 兜底**（分块复用 markdown 章节切分，chunk 带 `section`）、列表样式 → `- `/`1. `、表格 → Markdown 拼入正文、超链接 → `[文字](url)`（**只认 http/https**，`file://`/UNC 只留文字以免内部路径进正文）；**内嵌图片** → `[图片]`/`[图片:alt]` 不透明占位（DrawingML `a:blip` 与 VML `v:imagedata` 都扫、`mc:AlternateContent` 只认 Choice 以免一张图数成两张）+ `assets/images/img_{sha16}.{ext}` 内容寻址资产 + 复用同一套 OCR 幂等缓存（高置信注入占位符之后，低置信进审核队列）；与 markdown 同构：优先读 `imports/word`（保留目录树、可消歧），扫不到时回退 `assets/word` 并按版本族收敛；`.doc`/加密 docx（CFB 容器）/RTF 跳过并给出可操作原因）；截图 OCR（provider 可插拔：`api`（含 `mode=custom` 自研协议 / `openai` 兼容）/ `paddle` / `none` 默认关闭，未知值报错） | `data/parsed/`（PDF/Word 表格 JSON `*.tables.json` 与解析缓存 `*.extract.json`（**带 schema 位**，提取逻辑升级即整体失效重解析）、OCR 产物 `*.ocr.json`，可重跑） |
| 3. 规范化 | `canonicalize()`：正文拼装（高置信 OCR 文本注入 `[图片]` 占位符之后）+ `extra.evidence[].ocr` 摘要 + 文档级**两级标签**（tagging：词典 `config.tags.registry` 子串命中 + 文件名/标题 token） | 同 2.1 步骤 2 → canonical.jsonl |
| 4. 入库 | 同 2.6 | LanceDB + kb.sqlite3 |

### 2.2a 文件型来源的统一契约（发现 / 身份 / 分块）

PDF / Markdown / Word / Excel 都是"一个文件一篇文档"，原先各自实现扫目录、收敛历史版本、
造 `source_id`，于是同一类 bug 要修四遍。现在收成两条接缝（`doc_kinds.py` + `source_files.py`）：

| 维度 | 契约 | 落点 |
|---|---|---|
| **种类声明** | `source_type` / 资产子目录 / `source_id` 前缀 / 后缀 / 章节切分策略 / 默认验证状态，**只声明一次** | `doc_kinds.KINDS`（`BY_SOURCE`/`BY_TYPE` 索引） |
| **文件发现** | `source_files.discover(DiscoverSpec)`：**导入路径优先**（`rglob` 保留目录树），扫不到才回退资产层扁平副本 | `sources._discover_for` 装配，四个来源共用 |
| **身份** | `source_id = <前缀>:<族名>`；族名在**内容改动下稳定**（不含 `_copy_asset` 写的 `.<sha12>`）；同族名多份才追加**相对路径**指纹 `--<sha8>` | `SourceFile.sid()` / `source_files.path_tag` |
| **版本族收敛** | 回退模式按 `(族名, 后缀)` 分组，每族只取 mtime 最大者（并列取文件名字典序较大者），并取**族名**作 stem | `source_files.latest_versions` |
| **偏离诊断** | 导入路径非空时，资产层"有、导入路径没有"的族 → 打告警（这些文档**已不再纳入 canonical**） | `Discovery.stray` |
| **章节切分** | 由 `DocKind.section_splitter` 决定（`pdf`=编号标题 / `markdown`= `#` 标题且同标题正文合成一段 / `lines`= `##` 标题分组但**每行各自成块** / `""`=不切），实现表在 `chunking._SPLITTERS` | `chunking.chunk_doc` |
| **外源文档范围** | 审核台「文档管理」收录哪些 `source_type`，由声明表派生 | `review._EXTERNAL_SOURCE_TYPES = doc_kinds.EXTERNAL_DOC_TYPES` |
| **文档粒度** | 一个文件一篇（PDF/Markdown/Word/Excel 一致）；Excel 的分块仍按**行**（`lines` 策略）——聚合粒度与检索粒度解耦 | `sources.ExcelSource.canonicalize` + `chunking._split_lines_with_headings` |

> **为什么"偏离诊断"必须有**：`ingest` 不删文档、`pipeline.upsert_unified_canonical` 也不删行，
> 所以"某文件不再被刷新"是**静默偏离**（文档还在库里、还能被检索到，只是永远停在旧内容），
> 比直接丢掉更难发现。发现层因此把"资产层有、导入路径没有"的族显式打出来。
> 历史版本升级（原先 PDF 只 pull、不建 `imports/pdf`）时必然触发这条告警，处理办法见使用指南 §2.3。
>
> **为什么 Excel 按文件聚合**：逐行的 `source_id` 只能靠**位置**（`文件:sheet:行`），
> 中间插一行就会让其后的行身份整体平移，人工治理（标签/补标）会**串到别的行**上；
> 按文件聚合后身份 = `excel:<文件名>`，与其余来源同构。**检索粒度不损失**：
> 每个非空行仍是独立 chunk（`lines` 策略不合并行），且 `## sheet 名` 让 chunk 带 `section`。
>
> **刻意不纳入本表**：`confidence.py` 的来源可靠度基线与 `graph.py` 的节点类型映射——
> 那是**评分语义**与**图语义**，不是"文件型来源"的属性，硬塞进来会让声明表退化成万能配置表。

> **OCR 置信度单一来源**：`custom`=服务端返回 / `paddle`=引擎逐行平均 / `openai`=**模型自报**
> （提示词要求末行 `CONFIDENCE: <0~1 小数>`）。自报异常（缺失/不可解析/越界/不在末行/正文空却高置信）
> → `confidence=null` + `anomaly`，交人工审核，**不回退**启发式评分。**只有高置信文本进正文**
> 参与向量/全文检索；低置信与异常结果只留签名线索（原图可回看），不污染检索库。
>
> **进正文判定**：`confidence 非空` ∧ `anomaly 为空` ∧ `confidence ≥ ocr_min_confidence`（默认 0.6）。
> 注入形态：占位符 `[图片:alt]` 之后追加 `图片文字（OCR 置信度 x.xx）:` + 文本（正文仍不含路径）。
> 低置信/异常项由 `review.seed_low_confidence_ocr` 按图片 sha256 聚合进 `low_confidence_ocr` 审核队列。
>
> **幂等**：`*.ocr.json` 的缓存键 = 图片 `sha256` + **引擎指纹**（`provider|mode|model|提示词版本`）。
> 换模型/换模式/改提示词 → 自动重算；**阈值不进指纹** → 调 `ocr_min_confidence` 只重判定
> `text_included`，不重跑 OCR。旧版无指纹的产物视为未命中，升级后首次全量重算一次。
> 由于 `MarkdownSource` 在 `canonicalize` 时**即时 OCR 并按需写缓存**，注入不依赖 source 执行顺序。

### 2.2b 请求期图片 OCR（`POST /ocr`，不经 canonical）

Agent 侧 `client.py ocr <图片路径>`（本地只做 base64 编码）→ `POST /ocr` → `ocr.py`（`provider=api`，
外接 OpenAI 兼容 OCR 服务）→ 文本 + 报错签名，回到 `signature` / `search` 检索链路。

| 特性 | 说明 |
|---|---|
| 存储 | **无**——不落盘、不审计、不写任何库（与 2.2 导入期 `data/parsed/images/*.ocr.json` 区分） |
| 入参 | 图片 base64（≤8MB base64 / 约 6MB 原图、长边 ≤4096、魔数判格式）；**不接受路径**，端点不读客户端文件 |
| 出参 | `text` + `confidence`/`confidence_source`（模型自报单一来源）+ `anomaly`/`needs_review` + `signatures` + `provider/mode/model/elapsed_s`；不含服务端路径 |
| 注册 | 有 image source 时随 `serve_api.py` 注册；provider 非 `api` → 400 说明原因（本地 OCR 不在服务端执行） |
| 不可用 | OCR 服务不可达/超时 → **503** + detail（`mode`/`base_url`/`model`/原因/耗时）+ 服务端日志；**不静默降级** |
| 健康 | `GET /health` 的 `ocr` 字段只报**配置状态**（unconfigured/disabled/configured + endpoint），**不主动探测**服务 |

### 2.3 版本化代码仓（code 检索的数据源）

| 命令 | 产物 |
|---|---|
| `python scripts/build_code_snapshots.py` | vllm-ascend 各版本：`data/code/zips/{version}.zip` + 解压 `data/code/snapshots/{version}/` |
| `python scripts/build_vllm_snapshots.py` | 对应 vllm 主仓快照（版本由配套矩阵映射，自动跟随） |
| `python scripts/build_fork_snapshots.py` | 0day fork 仓快照（hy4/glm5.2 等模型开发分支）：`data/code/forks/{model}/`，版本=镜像锁定 commit（SHA 前 12 位）——检索走 `repo=fork:{model}` 命名空间，与官方版本物理隔离 |
| `python scripts/build_image_snapshots.py` | **0day 镜像内的 vllm-ascend 插件源码**：只拉镜像的 `COPY . /vllm-workspace/vllm-ascend/` 层（实测 23~101MB；整镜像 6GB+ 中的 CANN/编译层属二进制，不拉），解到 `data/code/images/{tag}/snapshots/{tag}/` + `index.sqlite3` + `meta.json`（image_digest/镜像时间/vllm commit/插件层 digest/层内 `.git` 若有则记 plugin_commit）——检索走 `repo=img:{tag}`，版本键=镜像 tag；`repo=img` 为列举入口；镜像 digest 未变则跳过重取。**离线路径**：`--archive <tar.gz>` / `--archive-dir <目录>` 直接解析内网 docker save/OCI 归档（不触网、不需要 docker），meta.source 标注 `archive:`，目录契约与在线路径一致 |
| `python scripts/build_code_snapshots.py --index-only` | vllm-ascend 派生数据重建：`data/code/index.sqlite3`（符号索引 + 报错字面量索引）、`symbols.json`（三层签名符号表）、`signal_words.json`（社区高频信号词，统计实现 `scripts/build_signal_words.py`，可单独运行） |
| `python scripts/build_vllm_snapshots.py --index-only` | vllm 主仓派生数据重建：`data/code/vllm/index.sqlite3`（独立符号索引） |
| `python scripts/build_fork_snapshots.py --index-only` | fork 仓派生数据重建：`data/code/forks/{model}/index.sqlite3`（独立符号索引） |
| `python scripts/build_image_snapshots.py --index-only` | 镜像插件源码派生数据重建：`data/code/images/{tag}/index.sqlite3`（独立符号索引，不联网） |

> 各命名空间索引 schema 同步（含 `kind` 列迁移）在对应 `--index-only` 内部自动完成（旧索引缺 `kind` 列时自动补列）；503 报错详情按命名空间给对应重建命令。
>
> ⚠️ **矩阵里的 `vllm` 值可带后缀**：镜像的 `VLLM_TAG` 本来就可能是 `0.11.0rc3` 这种，
> 所以消费方不能假设纯 `X.Y.Z`——`build_vllm_snapshots.companion_vllm_versions` 旧实现
> 用 `int(x) for x in v.split(".")` 排序，实测直接 `ValueError: ... '0rc3'` 把
> "代码快照（vllm 主仓）"整步打挂（非致命 → 静默没有快照）。现统一用
> `version_sort_key`（`confidence.parse_version` 取数值核心 + 后缀序号：
> 预发布 < 正式 < `post`，同后缀比尾号，无法解析排最后）。

### 2.4 辅助数据

| 命令 | 产物 | 用途 |
|---|---|---|
| `python scripts/build_release_calendar.py --all-repos` | `data/compatibility/release_calendar.{repo}.json`（分仓，slug 如 `vllm-project-vllm-ascend`；单仓模式写无后缀的 `release_calendar.json`） | 版本形态判断（release/rc/pre）+ 置信度版本上界（查询期现算） |
| `python scripts/build_companion_matrix.py`（`fetch_quay_tags.py` 辅助） | `data/compatibility/vllm-ascend.json`（每行含 `image_created`（镜像最后推送时间）、`vllm_commit`/`vllm_commit_date`（vllm 代码 commit 与提交日期）；fork 行另有 `vllm_repo/ref/base/sha/image_digest`）+ 跨运行缓存 `data/cache/`（fork 层 SHA 永久 / GitHub releases TTL 7 天 / requirements 永久 / tag→commit 永久，`--refresh-cache` 强制刷新） | 组件配套反向展开（vllm-ascend:0.18.0 → vllm/cann/pytorch-ascend）；commit 溯源回答"这个镜像对应哪个 commit"（官方行 tag→commit，fork 行 `vllm_sha`）；fork 行锁定 commit 供 `build_fork_snapshots.py` 按 SHA 拉快照 |

> **quay 传输特性**（实测）：quay.io 在国内链路上首连可达 20s+，且 TCP 连接**间歇性**超时
> （同一 URL 有 0.98s 成功、也有 30s/63s 连接超时）。所以 `fetch_quay_tags.fetch_tags` 默认
> `timeout=60` / `max_retries=4`（指数退避）；请求异常按 `type(e).__name__` 记日志并重试，
> 耗尽才抛 `RuntimeError`（**带原始错误**，不得吞掉原因）。分页按 `has_additional` 推进。
>
> ⚠️ **系统代理是常见元凶**：Windows 注册表配置的代理会被 `requests` 自动探测
> （`trust_env=True`，`getproxies()` 直接返回它，**无需任何 env 变量**），实测
> `127.0.0.1:7892` 这类本地代理会把 `quay.io` 的 TLS 握手**卡死**（`_ssl.c:990 handshake
> operation timed out`，`--insecure` 也无效，因为失败在证书校验之前），并间歇性把 GitHub
> 掐成 `SSLEOFError`。此时**显式设同名 `HTTP_PROXY`/`HTTPS_PROXY` 是空操作**（同一个地址），
> 正解是绕过：`NO_PROXY=quay.io,api.github.com,github.com`（`should_bypass_proxies` 对
> 注册表代理同样生效 → 直连）。`fetch_tags` 在检测到生效代理时会把地址和这条绕过提示
> 一并打进失败日志，省一次排查。
>
> **令牌有效期**：quay 的 pull token 是 JWT，实测 `exp - iat = 3600` 秒；而一次完整矩阵
> 实测跑 5149 秒 → **必然**跨过有效期，此后所有 `{v2}/blobs/...` 请求返回 401，日志表现为
> "从某个 tag 起集体 401 UNAUTHORIZED"。这**不是**那些 tag 没有 config——若按缺失处理，
> 该行的配套版本只能退化成启发式推断（实测后半段约 20 个 tag 被这样误判）。
> `fetch_image_config` / `_scan_layer_for_git_sha` 收到 401 即换新令牌并**当场重发**
> （不消耗重试预算：401 说的是令牌过期，不是这个 tag 有问题）。
>
> **重跑收敛**：矩阵写回时**保留行内已有非空字段**，所以网络抖动导致的单轮缺口会被后续轮次
> 填上而不会回退（实测两跑：`cann` 31→34、commit 溯源 8→14；某轮 `v0.23.0` 拉取失败仍完整
> 保留上一轮的值）。弱网环境就让 deploy 多跑几轮，缺口清单每轮都会打印。
>
> **日历的失败语义：抓不到 ≠ 没有 release**。`build_release_calendar.py` 任一页拉取失败
> 抛 `CalendarFetchError`（附已抓到部分），**不落盘**；写盘前还有一道 `would_drop_releases`
> 检查：新结果为空而现有文件非空 → 抛 `CalendarWriteRefused`，**保留上一版**。
> 这道防护是踩出来的——代理掐 TLS 时 `fetch_releases` 静默 `break` 并返回 `[]`，
> 一次 deploy 就把带 v0.23.0 的日历写成 `"releases": []`（113 字节），
> 下游 w_ver 版本上界与图里 `MERGED_IN → release` 关联整体消失。
> 日历有**两处消费者，恢复后的影响不同**：**查询期现算**的 `w_ver` 版本上界立即恢复
> （`confidence.version_at_date` 读当前文件）；**建图时固化**的 `release_tag` 只在
> "日历为空那段时间重建过图"时才为空，那种情况需重跑 `build_graph`（停 `serve_api`，
> Kùzu 单写者）。图未被重建则保留上一次建图时固化的完整关联。
> 注意 urllib（本脚本用）与 requests（拉取链路用）**证书来源不同**：
> urllib 走 Windows 证书库、requests 走 certifi——前者可能报
> `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` 而后者正常，
> 此时设 `SSL_CERT_FILE=<venv>/Lib/site-packages/certifi/cacert.pem` 即可（比 `--insecure` 好）。

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
       chunking（按段切块，max_chunk_chars=4000 / overlap=200；PDF/MD/Word 带章节结构，
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
| `data/kb.sqlite3` 的 `doc_verification` | 人工补标覆盖层（verification / canonical_verification / reviewer），**优先于 canonical 的 verification**——`docs` 行是 INSERT OR REPLACE，人工认证只能存独立表再由 ingest 合并；查询期从 `docs.extra.verification` 取该值 |

> FTS 分词只影响索引列，向量库（原文嵌入）不受影响——`build_fts.py` 重建全文索引**无需重嵌向量**。

#### 向量后端的回退（`build_vector_store`）—— 会换库，必须可见

`storage.vector_backend=lancedb`（默认）时若 lancedb 加载失败，自动回退
`PythonVectorStore`（纯标准库、JSON 持久化、暴力余弦）。两个后端**不是同一个库**：

| 后端 | 位置 | 能力 |
|---|---|---|
| `lancedb` | `data/lancedb/`（列式数据集） | 正式后端，带 ANN，十万级 chunk 无压力 |
| `python`（回退） | `data/lancedb_py.json`（单 JSON） | 语义能力完整，但暴力检索、全量载入内存（万级以内可用） |

回退时**如实**区分两种原因，并说明换库的影响（`_warn_fallback`，每进程只打一次）：

- **未安装** → 提示 `pip install lancedb`；
- **已安装但无法加载** → 打印真实异常。实测现场是 Windows 智能应用控制（Smart App Control）/
  WDAC / AppLocker 拦截 pyarrow、lancedb 的**原生 DLL**
  （`ImportError: DLL load failed while importing lib: 应用程序控制策略已阻止此文件。`；
  `WinError 4551`）。此时 `pip install` 无用，要么放行安全策略，要么显式接受回退后端；
- **正式库非空时额外警告**：回退后端**看不到**既有向量（本次向量检索只会命中新嵌入的文档）。
  处理二选一：放行原生 DLL 后重跑，或设 `storage.vector_backend=python` 并
  `build_kb.py --rebuild` 全量重建到该库。

后端标识（`backend_name` / `location`）进 `process_source` 的 stats 与 `run_build` 汇总
（`vector_backends`），所以**跑完能从日志看出向量到底写进了哪个库**——"静默换库"是这套设计里
最危险的一种降级（库还在、文档还在，只是向量检索悄悄少了一大半）。

### 2.7 索引重建与审核侧存储（不参与检索）

| 命令 / 组件 | 产物 | 说明 |
|---|---|---|
| `python scripts/build_fts.py` | 重建 `kb.sqlite3` 的 `chunks_fts`（读现有 chunk 原文重新 jieba 分词，chunk_id 与向量库严格一致） | 升级分词规则/旧库升级后使用；**不重新分块、不重嵌向量**，普通增量入库自动分词无需运行 |
| `python scripts/review_ui.py` | `data/review.sqlite3`（`review_items` 审核队列 / `asset_registry` 资产路径注册 / `doc_tags` 标签覆盖层） | 审核工作台独立端口，**只读检索 API 全程不碰该库**；资产路径只存 `asset_id → rel_path`，不进 canonical/检索库。图片资产由 `ImageSource`/`MarkdownSource`/`WordSource` 批量注册（内容寻址：同内容多副本共用一行），审核台按 `rel_path` 经 `/assets` 静态挂载预览原图。**补标认证**（verification_pending）写 `kb.sqlite3.doc_verification` 覆盖层并立即刷新 `docs.extra.verification`/`reliability`（w_rel 0.5→0.85 当场生效，无需重建） |

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
  `api_image` 仅存在 image source 时注册（未配置 OCR 则 `/ocr` 不存在）；
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
| `doc <source_id>` | `GET /doc/{source_id}` | `kb.sqlite3`：docs 行 + chunks_meta 排序 + chunks_fts 原文拼装 | extra 出口白名单清理（`_sanitize_extra`：`structure.tables` 只留数量、字符串字段 fail-closed 丢弃，不返回服务器路径） |
| `components` / `stats` / `health` | `GET` | `kb.sqlite3` 聚合 / 向量库 count | `/health` 含 embedding 状态 + `ocr` 配置状态（不主动探测 OCR 服务） |
| `companion` / `matrix` | `GET /companion` `/matrix` | `data/compatibility/vllm-ascend.json` | 配套反向展开 / 全量矩阵 |
| `graph chain/fixes/sig/doc/tags/evidence/stats` | `GET /graph/*` | Kùzu `data/graph`（只读查询） | 图未构建时返回引导提示（503→client 展示） |
| `tags list` / `tags docs <标签>` / `context "问题"` | `GET /tags` `/tags/{tag}/docs` `POST /tags/match` | `kb.sqlite3` `docs.tags`（最终标签）+ `config.tags.registry` 词典 | 能力发现：先知道知识库有哪些文档类别 |
| `code-graph search/code-search/trace/query/architecture/changes` | `POST /code-graph/*` | **外部 gh-puller 代码图谱服务**（`config.code_graph.base_url` + `path`），本库不落数据 | 可选能力：`enabled=false` 时端点不注册；不可达 → 503 + 引导用 `code` 查本地索引（**不回退**） |
| `code-graph health` | `GET /code-graph/health` | gh-puller 可达性探测 | 不触发熔断计数，仅展示 |
| `ocr <图片路径>` | `POST /ocr` | **无存储**（请求期只读计算）：图片 base64 → 外接 OCR 服务（`ocr.py`，provider=api）→ 文本 + 签名 | 不落盘/不审计；provider 非 `api` → 400；服务不可用 → 503 + 原因（不回退） |

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
- `extra`/`evidence` 走字段白名单清理，`source_ref` 仅保留 http(s) URL——检索响应不含服务器路径；
  `evidence[].ocr` 只放行 `confidence/confidence_source/anomaly/text_included/signatures`（纯知识字段）。

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
  **`/ocr` 刻意不在采集范围**（请求期图片 OCR 不审计：图片内容与 OCR 文本不落任何库）；
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
| `chunks_meta` | chunk_id（PK）/ doc_id / seq / section | 分块序号与章节（PDF/MD/Word 手册） |
| `doc_tags` | source_id（PK）/ auto_snapshot / excluded / manual / updated_at / reviewer | 人工标签覆盖层（审核工作台维护） |
| `doc_verification` | source_id（PK）/ verification / canonical_verification / reviewer / updated_at | 人工补标覆盖层（审核工作台认证/存疑维护）；`canonical_verification` 记覆盖前的值，供撤销回退 |

### 4.1 旁路存储（不参与检索）

| 库 | 表 | 说明 |
|---|---|---|
| `data/review.sqlite3` | `review_items`（id/category/item_ref/payload/status/created_at/reviewed_at/reviewer/result）、`asset_registry`（asset_id/rel_path/sha256/size/source_type） | 审核队列 + 资产路径注册；只读检索 API 全程不碰；资产路径不进 canonical/检索库（人工治理覆盖层 `doc_tags`/`doc_verification` 在 **kb.sqlite3**，见 4.0） |
| `data/telemetry.sqlite3` | `query_events`（session_id/client_ip/ts/endpoint/method/query_hash/query_normalized/signature_hash/signature_entities/signature_text/result_doc_ids/result_count/component/target_version/repo/probe） | 行为遥测原始事件（`feedback_enabled` 时写入）；离线推断产出 `data/confidence_feedback.json`（见 3.6） |

## 5. 关键设计点

- **幂等双哈希**：`embed_hash`（source_id+title+body）决定是否重嵌；`meta_hash`（整篇）决定是否刷新元数据——
  元数据变化不触发重嵌，崩溃续传按文档粒度恢复（`meta_hash` 存在 `docs.content_hash` 列，旧库兼容沿用列名）；
- **断点续传**：拉取断点存 `data/checkpoints/`；canonical 按 source_id upsert；重跑同一命令即续传；
- **canonical 是唯一事实源**：`--rebuild` 从 `canonical.jsonl` 全量重建（清空向量库 + 删 kb.sqlite3，需 TTY 确认或 `--yes`）；
  修改过的业务文件必须回写 canonical（upsert 覆盖），否则全量重建会回退旧内容；
  **例外：人工治理覆盖层**（`doc_tags`/`doc_verification`）只存在于 kb.sqlite3、canonical 里没有，
  删库前先备份后恢复，否则标签治理与补标认证会被重建抹掉；
- **人工治理覆盖层优先于 canonical**：`doc_tags`（标签）与 `doc_verification`（验证状态）在 ingest 时
  合并进 `docs` 行；`doc_verification` 参与 `meta_hash`，所以补标后下一轮入库走"仅刷新元数据"
  路径（不重嵌），且覆盖层变化不会被 canonical 冲回；
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
