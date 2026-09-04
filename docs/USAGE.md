# vllm-kb 使用指南

本指南覆盖：环境准备、数据采集与更新、全部查询命令、故障处理推荐流程、远程部署（存算分离）。
数据如何入库、查询时请求打到哪个接口/查哪些存储，见 [数据流说明](DATAFLOW.md)。

## 1. 环境准备

### 1.1 依赖

```bash
pip install -r requirements.txt        # 核心：requests / pydantic / lancedb / kuzu / fastapi / uvicorn
```

> **Windows PowerShell**：设 `PYTHONUTF8=1`（`[Environment]::SetEnvironmentVariable("PYTHONUTF8","1","User")` 或会话内 `$env:PYTHONUTF8=1`）避免 GBK 输出乱码。client.py 已在进程内 reconfigure stdout 为 UTF-8，但子进程（如 test_doc_commands 的 `--help` 子进程）和管道场景仍需此变量。

### 1.2 配置（密钥走环境变量）

```bash
cp config.example.json config.json

# GitHub 采集 token（必须，未认证限流 60 次/小时）
export GITHUB_TOKEN=ghp_xxx

# Embedding API key（可选；不设则把 config.json 的 embedding.provider 改为 "echo" 离线运行）
export EMBEDDING_API_KEY=sk-xxx
```

`config.json` 中 `sources` 定义了数据源（默认 vllm-ascend + vllm 两个 GitHub 仓库；可加
`type: pdf/markdown/excel/image` 业务来源，见 §2.3/§2.4），`embedding` 定义嵌入端点，
`storage` 定义数据目录（含 `code_root`：版本化代码仓根），`code` 定义代码仓快照来源与预存版本列表，
`tags` 定义标签词典（两级分类，见 §4.7），`sanitize` 定义内部数据脱敏白名单（后置，见 §2.4）。
所有路径相对 `data/`，可整体迁移。

> **URL 可写裸地址**：`embedding.base_url` / OCR `ocr_api_base` 可写 `10.0.0.5:8000/v1` 这种
> 裸 ip:port（业务侧 vLLM/OCR 服务），配置加载时自动补 `http://` 前缀并告警；https 需显式写全。

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

```bash
# 全量拉取 + 入库（issues/PRs/comments/releases，数小时~数十小时，可 Ctrl-C 中断后重跑续传）
python scripts/build_kb.py

# 小批量试跑
python scripts/build_kb.py --limit 100
```

### 2.1 增量与重建

| 场景 | 命令 |
|---|---|
| 日常更新（增量入库；GitHub 默认不重拉） | `python scripts/build_kb.py` |
| 拉取 GitHub 社区增量（新增 issue/PR） | `python scripts/build_kb.py --incremental` |
| **GitHub 补差拉取**（补历史缺失条目，跳过已有） | `python scripts/build_kb.py --pull-missing` |
| **REST 单条补拉**（指定编号，无需 GraphQL token） | `python scripts/build_kb.py --numbers 9749,9750` |
| 只重入库不拉取 | `python scripts/build_kb.py --skip-pull` |
| 只再生 canonical（不入库，供建图） | `python scripts/build_canonical.py` |
| 换 embedding 模型全量重建 | `python scripts/build_kb.py --rebuild` |

> **GitHub 拉取策略**：首次拉取完成后置 `done`（checkpoint），此后默认**不再拉取**（日志打印
> "已拉取完成（done），默认跳过——如需增量请用 --incremental"）；中断后重跑同一命令自动
> **断点续传**。三种拉取模式（互斥）：
> - **`--incremental`（时间窗增量）**：从 checkpoint 记录的上次增量 `max createdAt` 起，
>   issues 走 GraphQL `filterBy.since` 服务端过滤、PR 走 `UPDATED_AT DESC` 排序，跳过已有编号、
>   连续 3 页无新增停止，并把窗口推进到本次所见 `max createdAt`（首次增量无历史窗口时从头枚举
>   一次）——**只覆盖近期新增/更新，补不到历史旧条目**；
> - **`--pull-missing`（补差拉取）**：从头枚举（created desc），**跳过 raw 目录与 checkpoint 中
>   已有的编号，只拉缺失条目**——补历史旧条目（如业务数据缺失的单条 PR），翻到最新后置 done；
> - **`--numbers N1,N2,...`（REST 单条补拉）**：对指定编号先试 `/pulls/{n}`（404 则 `/issues/{n}`）
>   + 评论落 raw（隐含 missing 语义，**走 REST 不需要 GraphQL token**）——已知缺失编号时最精准。
> **全量重拉**（数据刷新/补拉旧条目评论）：删除 `data/raw/{source_id}/` 与
> `data/checkpoints/{source_id}.json` 后重跑 `build_kb.py`。

> **只再生 canonical（`build_canonical.py`）**：提取逻辑（版本/kind/组件/标签规则）升级后，
> 只需重新生成 canonical.jsonl 再跑 `build_graph.py` 建图——**不重嵌向量、不碰 kb.sqlite3**；
> 如需连带重入库（重嵌向量）再跑 `build_kb.py --skip-pull`（与旧参数 `--recanonicalize` 等价）。

> **`--rebuild` 高危确认**：会清空向量库 + 删除 kb.sqlite3 后全量重嵌（66K 文档约数小时）。
> 执行前强制确认——TTY 交互输入 `y/yes`；**非交互环境（agent/CI）必须加 `--yes`**，否则拒绝执行。
> canonical/raw/图/审核库不受影响；中断后重跑仍会先清空再重建。

> **日常维护流程（推荐节奏）**——知识库是"离线数据 + 定期刷新"，不需要实时：
>
> ```bash
> # 1. 拉取社区增量 + 增量入库（新增 issue/PR；时间窗口见上；中断后重跑同一命令续传）
> python scripts/build_kb.py --incremental
>
> # 2. （可选）业务来源有新增文件（data/imports/）时再跑一次不带参数的 build_kb.py
> #    （本地来源 pull 资产 + 入库；GitHub 源已 done 会打印跳过）
>
> # 3. 重建图（增量入库后图不含新文档——graph 查询要覆盖新增条目必须重建）：
> #    停 serve_api（Kùzu 单写者）→ 重建 → 重启
> python scripts/build_graph.py
>
> # 4. 抽查：python skills/vllm-kb/client.py stats   /  graph stats
> ```
>
> 注意：
> - **增量入库后必须重建图**：`build_graph.py` 从 canonical 全量重建，新增文档不会自动进图；
> - **FTS 全文索引不需要日常重建**：增量入库时新文档已实时写入 `chunks_fts`（含 jieba 分词）；
>   仅当升级 jieba / 分词规则 / 标签词典（`tags.registry`）后，才跑 `build_fts.py` 让存量文档
>   也用新分词（不重嵌向量；旧库无分词列升级时 ingest 会打印提示）；
> - **`--incremental` 补不到历史单条**（窗口从上次 `max createdAt` 起、PR 按更新时间排序、
>   连续 3 页无新增即停）——缺旧条目时用 §7 的 `backfill_canonical.py`（canonical 层、无需网络）
>   或全量重拉（raw 层也要补时）；
> - 更新前建议停止检索 API，更新完重启（尤其 `--rebuild` / `build_graph.py` 后必须重启）。

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

# 组件配套矩阵（vllm-ascend → vllm/cann/pytorch-ascend 自动匹配）
python scripts/build_companion_matrix.py
python scripts/build_companion_matrix.py --refresh-cache   # 强制刷新跨运行缓存（默认按 TTL/不可变语义命中）

# FTS5 全文索引重建（jieba 中文分词——可选，不重嵌向量；装 jieba 或升级分词规则后重跑）
python scripts/build_fts.py                     # 读现有 chunk 原文重新分词重建 chunks_fts
python scripts/build_fts.py --limit 1000        # 试跑前 N 个 chunk

# 正文 TF-IDF 标签候选导出（jieba——输出文件，人工审阅后手动写入 config.tags.registry）
# 注意：与审核队列的 tag_candidate 是两条独立路径——本脚本只产文件、不自动打标；
# 审核队列候选来自文件名/标题提取（自动、可一键采纳打标），见 §3.5。
python scripts/build_tag_candidates.py          # 业务文档（doc_*）正文候选 → data/tag_candidates_manual.json
python scripts/build_tag_candidates.py --include-github   # 也处理 github issue/PR（默认仅业务文档）
```

**FTS5 中文分词说明（`jieba` 可选依赖）**：SQLite FTS5 默认把连续中文整段当一个 token
（"超时"无法命中"超时排查"）。安装 `jieba`（`pip install jieba`，离线 wheel 可装）后，
入库自动对 chunk 文本分词写入 FTS 索引、查询侧同步分词——中文词可独立命中；
**向量库不受影响**（原文嵌入，无需重嵌）。升级 jieba/分词规则后跑一次
`scripts/build_fts.py` 重建索引即可。未装 jieba 时 FTS 行为与旧版一致（无需重建）。

**真实业务环境（SSL 被禁/镜像源）**：以上联网脚本均支持 `--insecure`（跳过 SSL 校验）与镜像源参数，
也可用环境变量统一配置（多脚本共享）：

```bash
export VLLM_KB_INSECURE=1                        # 跳过 SSL 证书校验
export VLLM_KB_GITHUB_BASE=http://<镜像>/api/v3  # GitHub API 镜像
export VLLM_KB_QUAY_BASE=http://<镜像>           # quay 镜像
export VLLM_KB_CODE_BASE=http://<镜像>           # codeload 源码 zip 镜像
python scripts/build_companion_matrix.py         # 全部脚本自动走业务环境配置
```

**配套矩阵自动匹配规则**（`build_companion_matrix.py`）：

- `vllm`：镜像 Env 的 `VLLM_TAG`（构建时锁定，最可靠）> 镜像 buildkit history（fork 仓
  `VLLM_REPO/VLLM_REF/VLLM_BASE`）> GitHub release 说明 > 版本号启发式；
- **fork 行**（0day 模型镜像，如 `hy4`/`glm5.2`）：除上表字段外自动带 `vllm_repo`（fork 仓）、
  `vllm_ref`（分支）、`vllm_base`（基线版本）、`vllm_sha`（**clone 层扫描固化的锁定 commit**）、
  `image_digest`（digest 锚定：镜像未重推则跳过重扫）。配套代码快照用
  `build_fork_snapshots.py` 按锁定 SHA 拉取（见上）；
- `cann`：镜像 Env 的 `cann-X.Y.Z` 路径；缺失时按**基础版本号**回退同系列其他形态
  （如 `v0.13.0rc1` 用 `0.13.0` 系列的 cann），同系列也没有则留空人工看护；
- `pytorch-ascend`（PTA）与 `pytorch`（torch）：对应 tag 的 `requirements.txt`
  （`torch-npu==X.Y.Z.postN` / `torch==X.Y.Z`）——**本地快照 zip 优先**（零网络），
  快照未预存的 tag 走 GitHub API 兜底；0day 模型（非版本 tag）无 requirements 时回退
  **同 minor 系列已发布 tag**（bailing 0.19.0 → 0.19.x 系列），系列内无 release 或
  未发布版本留空人工；
- 写回前**版本号正则校验**：非法值（`latest`、带前缀等）置空 + 告警，不污染矩阵。

**跨运行缓存**（`data/cache/`，防限流与重复下载；`--refresh-cache` 强制刷新）：

| 文件 | 内容 | 失效策略 |
|---|---|---|
| `fork_sha.json` | clone 层 digest → 锁定 SHA | 层不可变，**永久有效**（首次扫描后不再重下 ~75MB 层） |
| `github_releases.json` | release 说明（按 API 前缀键控） | TTL 7 天（兜底新 release） |
| `github_requirements.json` | requirements 兜底结果（含 404） | tag 内容不可变，**永久有效** |

只有完整/确定性结果才落盘（翻页中途失败、网络/限流失败不缓存，下次自动重试）。
缓存全命中时矩阵生成对 GitHub API 的请求为 **0**（未认证限流 60 次/小时不再是约束）。

### 2.3 业务来源导入（PDF 手册 / Markdown 文档）—— 完整实操

**步骤 1：放置文件**

```
data/imports/pdf/    ← 放 PDF 手册（操作手册/接口指南：硬件排查命令、错误码参考）
data/imports/md/     ← 放 Markdown（案例/架构说明/经验总结）
```

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

# 注意：不要用 --skip-pull 导入本地文件——它同样会跳过 pdf/md 的 pull（资产层复制），
# 导致 canonicalize 无文件。--skip-pull 只用于"仅重跑入库、文件已在资产层"的场景。
```

- 无 `GITHUB_TOKEN` 时 GitHub source 拉取会报错——**临时把 config 里 github source 的
  `enabled` 改为 `false`** 跑导入（只处理本地来源），完事恢复 `true`（保留标记注意改回）；
- 断点续传：中途 Ctrl-C 后重跑同一命令即可。

**步骤 5：验证**

```bash
python skills/vllm-kb/client.py search "hccn_tool 错误码 排查"   # 语义检索命中手册（sim 高）
python skills/vllm-kb/client.py doc pdf:<文件名去扩展名>          # 整篇读取（如 pdf:xxx 接口参考 04）
python skills/vllm-kb/client.py stats                           # chunks 数应增长
```

**步骤 6：产物与质量规则**

```
data/assets/pdf/<name>.pdf           # 原始文件（不可变层，sha256；路径不进库，以 asset_id 标识）
data/parsed/pdf/<asset_id>.tables.json   # 结构化表格（错误码表/命令表，asset_id 命名）
data/raw/canonical.jsonl             # canonical 追加（verification/tags 等元数据）
```

- PDF 表格转 Markdown 表格拼入正文（FTS 可检索）+ 另存结构化 JSON；
- **自动标签（两级分类）**：入库时从文件名 + 内部标题确定性提取——**主题/领域类**（domain，
  如 `npu-smi`/`Atlas`，=这是什么领域的知识）与**具体作用类**（purpose，如 `命令参考`/`错误码表`，
  =文档能帮我做什么），与 `config.json` 的 `tags.registry` 词典子串命中为准；
  未收录强候选进审核队列 `tag_candidate`，采纳后入词典并即时打标；
- 验证状态默认：**PDF 手册 = `expert`**、**Markdown = `unverified`**（审核工作台补标）；
- 检索结果显示 `验证=expert/unverified`；embedding key 有效时语义检索（向量）生效，
  无效时自动降级全文检索（`search` 仍可用）；
- **路径不进库（安全约束）**：canonical/检索库不含服务器路径（资产以 asset_id 标识），
  `/doc` 等 API 返回的 extra 经白名单清理；管理员侧路径仅存审核库（asset_registry）。

**PDF 解析缓存（性能，增量入库 / recanonicalize 通用）**：PyMuPDF 逐页提取耗时较长
（241 页手册约 7s/篇），解析中间产物（文字层 + 表格 + 页数）按资产 sha256 缓存到
`data/parsed/pdf/<asset_id>.extract.json`——**资产未变时直接复用缓存**（进度行标注
"缓存命中"，毫秒级），仅标签/元数据提取每次重算（词典/提取规则升级**无需清缓存**即生效）。
**强制重新解析**（如 PyMuPDF 升级后想重新提取文字层）：删除 `data/parsed/pdf/` 目录即可，
资产层（`data/assets/`）与 kb 数据不受影响。

**步骤 7：重启检索服务（改过 key / config 后）**

```bash
# 先停掉旧 serve_api（8000 端口），再：
python scripts/serve_api.py          # http://127.0.0.1:8000
python skills/vllm-kb/client.py health   # chunks 数与预期一致
```

**Markdown 图片处理（随 md 一起入库）**：

- md 正文里的图片引用自动收集：相对路径（以 md 所在目录为基准）、绝对路径、base64 内嵌 → 复制到
  `data/assets/images/`，**正文引用改为不透明占位 `[图片]`**（不暴露路径），`extra.evidence`
  记录 asset_id/sha256（管理员侧经 asset_registry 找回原图）；
- 网络 URL 图片：标记 `remote` 不阻塞导入（业务环境网络可达时可后续补抓）；
- 引用不存在的本地图片：标记 `unresolved`（不保留路径形态引用）；
- 图片的 OCR 由 image source 完成（见下）。

**图片 OCR（签名导向，provider 可插拔）**：

```bash
# config 的 images source 选择 OCR 方式（ocr_provider）：
#   "ask"（默认）: 配了 ocr_api_base → 走 API；没配 → 询问"是否本地运行（paddle）"，
#                  否定（或非交互终端）→ 跳过 OCR（导入不受阻）
#   "api":        HTTP OCR 服务（强制 API 场景推荐）——失败时同样询问本地/跳过
#   "paddle":     本地 PaddleOCR（明确选择，不询问；未安装 → 提示并跳过）
#   "none":       明确跳过
python scripts/build_kb.py --skip-pull        # 触发 image source 的 canonicalize（OCR + 签名提取）

# 产物：data/parsed/images/<name>.ocr.json（文本 + 置信度 + 错误签名清单）
```

- **OCR API 两种调用模式（`ocr_api_mode`，默认 custom）**：
  - `custom`：`POST {ocr_api_base}/ocr`，body `{"image": "<base64>", "filename": "x.png", "model": "<可选>"}`，
    响应 `{"text": "...", "confidence": 0.93}`——适用于自研 FastAPI 包装 paddleocr 等；
  - `openai`：OpenAI 兼容接口（如 siliconflow 的 DeepSeek-OCR）：`POST {base}/chat/completions`，
    `model=ocr_api_model`（**必填**），图片以 data URI 内联，取 `choices[0].message.content`；
- key 可选（`ocr_api_key` 或环境变量 `OCR_API_KEY`）；`ocr_api_model` 可选透传（openai 模式必填）；
- **embedding 强制 API**（本地 embedding 不做，部署复杂）：`embedding.base_url` 指向 OpenAI 兼容端点
  （可指向其他服务器的 vLLM 部署）；`echo` 仅离线演示（效果粗糙）；
- 连通性测试：审核工作台 API 配置中心对 embedding / OCR 均提供"测试连通"（OCR 用内置测试图走真实识别链路）；
- OCR 结果按图片 sha256 幂等（重跑跳过未变图片）；只提取错误签名（算子/错误码/模型/版本），
  低质量 OCR 不污染向量库——图片靠"签名可达 + 原图可回看"；
- 与 md 文档的 `evidence` 联动：图文互证（正文签名 ↔ OCR 签名）在后续图/审核环节消费。

### 2.4 Excel 登记表导入（schema-free）—— 完整实操

工程师问题定位记录 / 已知问题登记表等（**表头/列不固定**）：

```jsonc
// config.json sources 启用（path 可以是单个文件或目录）
{"id": "engineer", "type": "excel", "path": "data/imports/engineer/问题定位记录.xlsx", "enabled": true}
```

```bash
pip install openpyxl          # 依赖（可选组，离线 wheel 可装）
python scripts/build_kb.py    # 导入：每行一条文档入库 + 嵌入
python scripts/build_graph.py # 建图：错误码/算子等实体自动入图（先停 serve_api）
python skills/vllm-kb/client.py graph sig <错误码>   # 验证实体命中
```

- **schema-free**：不写死任何列名 / sheet 名 / 行号——遍历所有 sheet/行，每行**非空 cell 按列序拼接为自由文本**入库（每行一条文档，`excel:{文件名}:{sheet序号}:{行号}`）；空行跳过；
- 错误码/算子/模型/版本由 signature 三层提取**自动入图**（建图只依赖 canonical，无需图侧适配）；
- 验证状态 `unverified`（登记表低优先级，按未解决 issue 处理）。

**内部数据脱敏（后置，Excel/Markdown 源生效）**：

- **库中存原文、出口统一脱敏**：serve_api 返回给 agent 的正文/标题（/doc 全文、/search snippet、/title、/tags、/graph 等）按 `config.sanitize` 白名单脱敏（内部 IP → `<IP>`、内部路径 → `<PATH>`，默认路径如 `/var/log/npu/` 保留）——**内部检索用原文**（可按原 IP 检索），**改脱敏配置即时生效、无需重嵌**；
- `config.sanitize`：`keep_paths`（保留的默认路径前缀）、`keep_ips`（保留的 IP，默认回环/通配）、`sources`（入库时扫描维护日志的源，默认 `["excel","markdown"]`）；`None`=用默认、显式 `[]`=全部脱敏/全部关闭；
- **被脱敏的原始 IP/路径落盘 `data/sanitize_log.json`**（维护文件，不进库/不返回给 agent）——据此调整白名单；审核页（管理员）显示原文。

> Word/HTML 适配在业务环境阶段开发。

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

### 3.2 图更新流程（Kùzu 单写者约束）

**更新图（scripts/build_graph.py）前必须先停止检索 API**——检索服务持有图库读连接，
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

## 3.5 审核工作台（人工确认统一入口 + API 配置中心）

所有需要人工确认的位置（未验证文档补标、案例标题待审核/待修改、OCR 图文互证不一致、
低置信度 OCR 签名、跨来源合并候选等）共用一个轻量级 Web UI；同时集中展示所有 API 配置。

```bash
pip install fastapi uvicorn
python scripts/review_ui.py                    # http://127.0.0.1:8010（默认自动补单，幂等）
python scripts/review_ui.py --seed-only        # 只补单不启动服务
python scripts/review_ui.py --no-seed          # 启动但不自动补单
```

**功能**：

- **概览**：7 类审核项的待办/存疑数（verification_pending / case_title_flag / ocr_mismatch /
  low_confidence_ocr / equivalence_candidate / table_join_candidate / **tag_candidate**）
  与标签词典统计（领域/作用类个数、已打标文档数）；
- **审核队列**（未审核在前、存疑在后）：按类别筛选；详情页可预览原图（assets 静态服务）。
  **审核动作（只做判定，不修改原始内容）**：
  - **✓ 认证**：文档有效，不再提示；
  - **？ 存疑**：重新进入队列，排在未审核之后；
  - **🗑 标记删除**：只删除 `kb.sqlite3` 数据库记录，**原始资产文件保留**——进入队列底部
    "**待实际删除**"列表（含资产路径），由人员**手动本地删除**原始文件；
  - **↩ 撤回**：在待删除列表恢复数据库记录（用删除前的备份），重新进入队列；
  - **tag_candidate 采纳**：未收录的自动标签候选 → "✓ 采纳为标签"（入词典 + 对该候选**全部提及文档**打标，
    检索侧立即生效）；不采纳则 **✓ 认证** 即忽略（不改任何数据，候选不再自动出现）。
    候选标签的来源、聚合与生效边界见下方"候选标签（tag_candidate）说明"。
  审核记录存 `data/review.sqlite3`（独立于只读检索库），带审核人/时间戳可审计；
- **文档管理（含两层标签编辑）**：列出外源文档（导入的 PDF/Markdown 等，GitHub 不在此列），
  每条显示**自动标签**（领域/作用分区，点 ✕ 排除）、**已排除**（点 ↺ 恢复）、**人工标签**
  （添加/删除）与**最终标签预览**（`(自动 − 排除) ∪ 人工`，与入库/建图一致）；
  人工添加的新标签自动同步词典；**同 stem 重名告警**（人工处理，不自动消歧）；
  支持**彻底删除**——同时清除 docs 行 + chunks_fts + chunks_meta + 向量四层，
  **本地资产文件不动**；下次增量入库时文件仍在本地会**自动重新入库**，文档废弃由管理员手动删本地文件；
- **标签管理**：词典（`config.json` `tags.registry`，全局唯一事实源）按领域/作用分组 + 文档数，
  支持**新增 / 改名（全库替换）/ 改 tier / 删除**——均同步 config.json；**不热插图**
  （Kùzu 单写者约束），运行 `build_graph.py` 重建后入图；
- **API 配置中心**：集中查看并**编辑** embedding / OCR / GitHub / code_graph 的配置——
  **非密钥字段**（provider/base_url/model/ocr_provider/ocr_api_mode 等）保存到 `config.json`；
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
   `extra.tag_candidates`，**按候选词聚合**成一条审核项（`item_ref = tag:{name}`，payload 含
   提及文档数 + 文档列表 + 建议 tier；词典已收录的词不生成）——同词多文档只审一次，
   **审一次 = 全部提及文档生效**；
3. **采纳（adopt）**：① 候选词写入 `config.json` 的 `tags.registry`（tier 取审核下拉选择或启发式，
   全库词典生效）；② 对聚合记录的全部提及文档写人工标签（`doc_tags.manual`），并按
   `final = (自动 − 排除) ∪ 人工` 同步 `docs.tags`——**检索侧立即生效**；③ 审核项标记 approved，
   重跑补单不再出现；
4. **忽略**：**✓ 认证**即忽略（不采纳、不改任何数据）；因 `item_ref` 已存在（任意状态），
   之后自动补单不再打扰。**🗑 标记删除按钮对 tag_candidate 不适用**（其 `item_ref` 是候选词
   `tag:{name}` 而非文档 source_id，点删除会报"文档不存在"）。

**生效边界**（"刷新到所有文档"需分清）：

| 目标 | 采纳后何时可见 | 需要做什么 |
|---|---|---|
| 检索 API：`tags` 目录 / `tags docs` / `context` / search 的 **FTS 命中**过滤 | **立即**（读 `docs.tags`，采纳时已同步） | 无 |
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
case_title_flag；`extra.tag_candidates`（未收录强候选）→ tag_candidate。
审核结果回写 canonical 依赖重跑 `build_canonical.py`（或 `build_kb.py --skip-pull`，后续版本支持热更新）。

**与只读检索 API 的关系**：分离端口（检索 8000 / 审核 8010）、分离数据（kb.sqlite3 只读 /
review.sqlite3 可写）；审核库检索 API 不碰。权限（谁能标注专家认证）由部署方加 nginx basic auth 等。

## 4. 查询命令（skill）

所有查询经 `skills/vllm-kb/client.py`，服务地址解析：`--base` > 环境变量 `VLLM_KB_BASE` > 默认 `http://127.0.0.1:8000`。

### 4.1 search —— 语义检索

```bash
# 组件:版本 问题描述（推荐，置信度按版本计算）
python skills/vllm-kb/client.py search "vllm-ascend:0.23.0rc1 GLM5.1 PD分离P节点挂死"

# 普通查询 + 目标版本
python skills/vllm-kb/client.py search "CUDA illegal memory access" --version 0.26.0 --top 5
```

返回：结果标题/URL/是否已解决 + 置信度分解（w_time 时间衰退 / w_ver 版本匹配 / w_rel 来源可靠度）。

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

命中新版本后，用 GitHub commits API 按文件路径过滤找引入 commit → PR 编号 →
再用 `graph fixes/chain` 确认落地 release 与 backport。

> **符号索引升级**：`index.sqlite3` 是派生数据（zip 快照为事实源）。代码提取规则/schema 变更后
> （如本次 ast 提取 + `--kind msg` 报错字面量索引）重建即可，无需迁移：
> 先停检索 API → `python scripts/build_code_snapshots.py --index-only`（自动重建
> index.sqlite3 + symbols.json + signal_words.json；旧索引缺 `kind` 列时自动补列，但需全量
> 重建才有 kind 数据）→ 重启 API。耗时：7 版本约 2 分钟，41 版本约 11 分钟。

### 4.6 其他

```bash
python skills/vllm-kb/client.py health                  # 服务健康（含 embedding 状态 ok/degraded）
python skills/vllm-kb/client.py stats                   # 知识库规模
python skills/vllm-kb/client.py doc github:vllm-project-vllm-ascend:issue:10700   # 整篇 issue 全文
python skills/vllm-kb/client.py companion vllm-ascend 0.23.0rc1   # 组件配套版本展开
# matrix/code-versions 为管理员调试命令（列全量配套矩阵/预存代码版本），故障流程不使用
```

### 4.7 文档标签（能力发现）—— `tags` / `context`

文档级标签两级分类：**主题/领域类**（domain：HCCL、网络、NPU、CANN…=这是什么领域的知识）与
**具体作用类**（purpose：超时排查、命令参考、错误码表…=文档能帮我做什么）。

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

**图内容说明**：Issue/PR/Release（修复链路）+ Operator/ErrorCode/Model/Version（签名实体）+
**Doc**（PDF 手册/Markdown 等非 github 文档）+ **Interface**（手册"命令格式"段提取的工具.子命令，
如 `hccn_tool.bandwidth`）+ **Tag**（文档级标签，两级分类）。文档经两条 `DOCUMENTS` 边入图：
- **错误码表 → ErrorCode**：`graph doc pdf:<手册>` 的 `documents` 含该手册定义的错误码；
- **命令格式段 → Interface**：`documents` 含该手册定义的接口/命令；
- **标签 → Tag**：`TAGGED_WITH` 边（最终标签 = 自动 − 排除 ∪ 人工，与入库一致；
  registry 全量标签也建节点——新增标签重建图即入图）；
错误码/命令与 GitHub issue 的 MENTIONS 共享节点——可回答"这个错误码在哪个手册定义、
社区哪些 issue 提到"、"查带宽用哪个命令、命令在哪本手册"。

图与检索 API 的关系：图由 `scripts/build_graph.py` 构建，`serve_api.py` 的 `/graph/*` 端点只读查询；
**更新图前请停止检索 API**（Kùzu 单写者，构建期间不可并发查询）。

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
# trace 函数名双形态：短名（do_auth）或 search 返回的完整 qn（自动取末段短名透传）。
# 同名多节点 → 返回候选列表（name/file/lines）而非静默取其一；确认后用短名重试。
# 每次调用先做唯一性预检（翻页 cursor 除外）——多一次 search 往返，换取零错配。

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
- 定位某版本某符号在哪定义、报错文本来自哪段代码 → `code`（本地快、版本化）
- 谁调用了某函数、变更会影响什么、架构怎么分层、跨仓调用关系 → `code-graph`
- gh-puller 不可达时降级思路：`code-graph` 503 → 改用 `code` 查本地索引（手动）

### 4.10 行为遥测与置信度反馈（feedback_enabled）

行为遥测自动采集查询行为（不依赖用户主动反馈），离线推断三态反馈（hit/miss/unknown），
用时间维度指数遗忘更新每个 doc 的后验 Beta 分布，查询期新增 **w_hist 历史可靠度**因子
（后验下界 lb = mean - z·sd）并入 final 排序。**与 w_rel 正交，不乘进 w_rel**——
保护审计链（同一文档元数据不变时 conf 部分确定性可审计，w_hist 独立可解释）。

**启用**：config `confidence.feedback_enabled = true`。中间件全量记查询行为到独立
`data/telemetry.sqlite3`（不碰只读 `kb.sqlite3`）。会话归属经 `VLLM_KB_SESSION` 环境变量
→ `X-Session-Id` header 透传（agent 侧管），缺失回退 ip+时间窗。

**数据流（三段分离，审计可逆）**：
```
serve_api (只读)                    离线周期
┌─────────────────────────┐       ┌───────────────────────────┐
│ middleware 记原始行为    │       │ scripts/build_feedback.py │
│ → telemetry.sqlite3     │ ────▶ │ 会话重建+行为推断三态      │
│ (独立库,不碰 kb.sqlite3) │       │ → confidence_feedback.json │
│                         │       │ → knowledge_gaps 表        │
│ compute_confidence      │       └───────────────────────────┘
│ + w_hist(读 feedback.json)│ ◀──────── 重启 serve_api 生效
│ final=sim^γ·conf^(1-γ)·lb^σ │
└─────────────────────────┘
```

**离线推断**：`python scripts/build_feedback.py` —— 扫遥测库重建会话序列，
按行为模式推断三态（权重≤0.5，自证循环阻尼）：
- search/signature 命中后拉 doc 不重查 → 弱正 hit（0.3）
- signature 命中后会话直接结束 → 弱正 hit（0.3）
- 有 doc 命中后调 code/diff → 中正 hit（0.5）
- 60s 内改述重查 → 弱负 miss（0.3）
- 命中后零后续 → unknown（不进 n_eff，单独计数）
- 无命中的查询进缺口检测，不进后验

**后验更新（时间维度指数遗忘）**：`a <- a×2^(-Δt/HL) + w×hit`，HL 复用 config
`half_life_days`（非事件次数衰减——克服版本漂移且冷门案例不异常）。seed=1+1 随遗忘
衰减，HL 决定失效。**seed 强度 1+1=2，数据部分需≥3 超先验 1.5 倍才主导后验**
（隐含前提：平均确认频率约每季度 1 条；冷门 domain 检索频率过低时 supported 状态
不会出现——符合设计，冷门且无反馈证据的文档保持中性不被误杀）。

**w_hist 三段式**（当期值不缓存，每次查询重算）：
- `n_eff=0` → w_hist=1.0（中性，不用 seed 套 lb——避免误杀新文档），flag=new
- `0<n_eff<n_min` → w_hist=lb（正常算），flag=accumulating/evidence_thin
- `n_eff≥n_min` → w_hist=lb，flag=supported/used_but_unconfirmed/failing

**关键约束**：
- **z（检索侧排序）vs p_min（消费侧决策）分工**：vllm-kb 只输出 lb，不参与消费侧决断
- **只标注不拦截**：history_flag 供消费侧决策，vllm-kb 永不过滤候选
- **n_eff 是加权观察非次数**：w=0.4 的推断只贡献 0.4
- **unknown 不进 n_eff**：统计正确，单独计数

**知识缺口**：审核工作台 → "知识缺口" tab，展示同签名跨≥3会话反复查无果的缺口
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

数据（向量库、索引、图）放远程服务器，本地 skill 只留 ~34KB 发 HTTP 查询。

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

辅助脚本：

```bash
python scripts/deploy_remote.py --gen-config    # 生成远程 config（已去 token/api_key）
python scripts/deploy_remote.py --pack-data     # 打包数据成 tar.gz（大数据推荐 rsync/scp -r）
python scripts/deploy_remote.py --print-steps   # 部署步骤说明
```

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

**Q: 真实业务环境 SSL 被禁 / 证书不受信？**
A: 联网脚本（代码快照/版本日历/配套矩阵）加 `--insecure` 跳过证书校验；若域名不可达
需用业务侧 http 镜像（`--github-base/--quay-base/--base-url`，或环境变量 `VLLM_KB_GITHUB_BASE` 等）；
`embedding.base_url` 可写裸 ip:port 自动补 `http://`。
注意 `REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE` 环境变量会**覆盖** `--insecure`
（requests 的 `merge_environment_settings` 会用环境 CA 串复活 SSL 校验，
业务环境 MITM 场景典型症状：`CERTIFICATE_VERIFY_FAILED ... HTTPSConnectionPool`）。
本库已内置请求级 `verify=False` hook 修正（`vllm_kb/net.py`，端到端测试固化
`tests/test_net_tls.py`）；若用自己的脚本复用 `get_session(insecure=True)` 即可，
不要裸设 `session.verify = False`（会被环境变量覆盖）。

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
删除 `data/parsed/pdf/` 目录即可，资产层与 kb 数据不受影响。

**Q: 想加自己的故障记录（excel/markdown）？**
A: config.json 的 `sources` 加条目即可：`{"id":"engineer-troubleshooting","type":"excel",
"path":"data/imports/...xlsx","enabled":true}`（schema-free 导入，见 §2.4）或
`{"id":"mynotes","type":"markdown","path":"data/mynotes","enabled":true}`；github 源
支持 `--incremental` 增量拉取（见 §2.2）。
