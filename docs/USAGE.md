# vllm-kb 使用指南

本指南覆盖：环境准备、数据采集与更新、全部查询命令、故障处理推荐流程、远程部署（存算分离）。

## 1. 环境准备

### 1.1 依赖

```bash
pip install -r requirements.txt        # 核心：requests / pydantic / lancedb
pip install fastapi uvicorn            # 检索 API（可选，离线查询可不用）
```

### 1.2 配置（密钥走环境变量）

```bash
cp config.example.json config.json

# GitHub 采集 token（必须，未认证限流 60 次/小时）
export GITHUB_TOKEN=ghp_xxx

# Embedding API key（可选；不设则把 config.json 的 embedding.provider 改为 "echo" 离线运行）
export EMBEDDING_API_KEY=sk-xxx
```

`config.json` 中 `sources` 定义了数据源（默认 vllm-ascend + vllm 两个 GitHub 仓库），
`embedding` 定义嵌入端点，`storage` 定义数据目录（含 `code_root`：版本化代码仓根），
`code` 定义代码仓快照来源与预存版本列表。所有路径相对 `data/`，可整体迁移。

> **URL 可写裸地址**：`embedding.base_url` / OCR `ocr_api_base` 可写 `10.0.0.5:8000/v1` 这种
> 裸 ip:port（内网 vLLM/OCR 服务），配置加载时自动补 `http://` 前缀并告警；https 需显式写全。

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
| 日常更新（拉新数据 + 增量入库） | `python scripts/build_kb.py` |
| 只重入库不拉取 | `python scripts/build_kb.py --skip-pull` |
| 提取逻辑升级后重生成 canonical | `python scripts/build_kb.py --recanonicalize` |
| 换 embedding 模型全量重建 | `python scripts/build_kb.py --rebuild` |

> **`--rebuild` 高危确认**：会清空向量库 + 删除 kb.sqlite3 后全量重嵌（66K 文档约数小时）。
> 执行前强制确认——TTY 交互输入 `y/yes`；**非交互环境（agent/CI）必须加 `--yes`**，否则拒绝执行。
> canonical/raw/图/审核库不受影响；中断后重跑仍会先清空再重建。

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

# 组件配套矩阵（vllm-ascend → vllm/cann/pytorch-ascend 自动匹配）
python scripts/build_companion_matrix.py

# FTS5 全文索引重建（jieba 中文分词——可选，不重嵌向量；装 jieba 或升级分词规则后重跑）
python scripts/build_fts.py                     # 读现有 chunk 原文重新分词重建 chunks_fts
python scripts/build_fts.py --limit 1000        # 试跑前 N 个 chunk

# 正文 TF-IDF 标签候选导出（jieba——输出文件，人工审阅后手动写入 config.tags.registry）
python scripts/build_tag_candidates.py          # 业务文档（doc_*）正文候选 → data/tag_candidates_manual.json
python scripts/build_tag_candidates.py --include-github   # 也处理 github issue/PR（默认仅业务文档）
```

**FTS5 中文分词说明（`jieba` 可选依赖）**：SQLite FTS5 默认把连续中文整段当一个 token
（"超时"无法命中"超时排查"）。安装 `jieba`（`pip install jieba`，离线 wheel 可装）后，
入库自动对 chunk 文本分词写入 FTS 索引、查询侧同步分词——中文词可独立命中；
**向量库不受影响**（原文嵌入，无需重嵌）。升级 jieba/分词规则后跑一次
`scripts/build_fts.py` 重建索引即可。未装 jieba 时 FTS 行为与旧版一致（无需重建）。

**内网（SSL 被禁/镜像源）**：以上联网脚本均支持 `--insecure`（跳过 SSL 校验）与镜像源参数，
也可用环境变量统一配置（多脚本共享）：

```bash
export VLLM_KB_INSECURE=1                        # 跳过 SSL 证书校验
export VLLM_KB_GITHUB_BASE=http://<镜像>/api/v3  # GitHub API 镜像
export VLLM_KB_QUAY_BASE=http://<镜像>           # quay 镜像
export VLLM_KB_CODE_BASE=http://<镜像>           # codeload 源码 zip 镜像
python scripts/build_companion_matrix.py         # 全部脚本自动走内网配置
```

**配套矩阵自动匹配规则**（`build_companion_matrix.py`）：

- `vllm`：镜像 Env 的 `VLLM_TAG`（构建时锁定，最可靠）> GitHub release 说明 > 版本号启发式；
- `cann`：镜像 Env 的 `cann-X.Y.Z` 路径；缺失时按**基础版本号**回退同系列其他形态
  （如 `v0.13.0rc1` 用 `0.13.0` 系列的 cann），同系列也没有则留空人工看护；
- `pytorch-ascend`（PTA）：对应 tag 的 `requirements.txt` 中 `torch-npu==X.Y.Z.postN`；
  0day 模型（非版本 tag）参考其 vllm 版本对应 tag 的 PTA，无命中留空；`pytorch`（torch）不必须留空；
- 写回前**版本号正则校验**：非法值（`latest`、带前缀等）置空 + 告警，不污染矩阵。

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
- 网络 URL 图片：标记 `remote` 不阻塞导入（业务环境内网可达时可后续补抓）；
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

> Excel 导入暂不支持（表头不固定）；Word/HTML 适配在业务环境阶段开发。

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
  - **tag_candidate 采纳**：未收录的自动标签候选 → "✓ 采纳为标签"（入词典 + 即时打标），
    忽略不记录（候选可再次出现）。
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
- **API 配置中心**：集中查看并**编辑** embedding / OCR / GitHub 的配置——
  **非密钥字段**（provider/base_url/model/ocr_provider/ocr_api_mode 等）保存到 `config.json`；
  **密钥**（embedding key / OCR key / GitHub token）保存到 `data/secrets.local.json`
  （遵守"密钥不入 config.json"，任何入口 `AppConfig.load` 自动加载进环境变量）；
  key 在页面上一律脱敏（只显示已/未配置），**embedding 与 OCR 均支持连通性测试**
  （OCR 用内置测试图走真实识别链路，验证 HTTP/鉴权/模型）；修改对已运行的服务需**重启生效**。

自动补单规则：`verification=unverified` 的文档 → verification_pending；标题含"待审核/待修改"→
case_title_flag；`extra.tag_candidates`（未收录强候选）→ tag_candidate。
审核结果回写 canonical 依赖重跑 `--recanonicalize`（后续版本支持热更新）。

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

**Q: 内网 SSL 被禁 / 证书不受信？**
A: 联网脚本（代码快照/版本日历/配套矩阵）加 `--insecure` 跳过证书校验；若域名不可达
需用内网 http 镜像（`--github-base/--quay-base/--base-url`，或环境变量 `VLLM_KB_GITHUB_BASE` 等）；
`embedding.base_url` 可写裸 ip:port 自动补 `http://`。

**Q: 离线能用吗？**
A: 能。采集完成后全部检索离线；嵌入可用 `echo` provider 离线自测（效果粗糙）。

**Q: 图打不开 / build_graph 失败，数据根路径含中文？**
A: Kùzu 图库路径**不能含非 ASCII 字符**（中文、emoji 等）——`data/graph` 或存算分离的
`VLLM_KB_DATA_ROOT` 若在中文路径下（如 `C:\Users\张三\...`），建图/图查询会失败。
把数据根移到纯 ASCII 路径（如 `D:\vllm-kb-data`）后重建图（`scripts/build_graph.py`）。

**Q: PDF 重新入库/`--recanonicalize` 很慢，怎么跳过已解析的？**
A: 解析中间产物已按资产 sha256 缓存（`data/parsed/pdf/<asset_id>.extract.json`），
资产未变时自动复用（进度行标注"缓存命中"）；想强制重新解析（如 PyMuPDF 升级），
删除 `data/parsed/pdf/` 目录即可，资产层与 kb 数据不受影响。

**Q: 想加自己的故障记录（excel/markdown）？**
A: config.json 的 `sources` 加 `engineer-troubleshooting` 条目（markdown/excel 接口已预留，
Phase 4 实现；当前可用 github 源或 `--recanonicalize` 流程）。
