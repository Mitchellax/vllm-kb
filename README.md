# vllm-kb

vLLM / vllm-ascend 故障知识库与检索工具链：自动采集 GitHub 社区数据（issues / PRs / comments / releases），构建**向量 + 全文 + 版本化源码 + 图**四位一体的离线知识库，并支持导入业务来源文档（PDF 手册 / Markdown / 截图 OCR），为故障处理工程师提供**签名精确检索、语义检索、对应版本源码定位、修复链路追溯**的完整链路。

```
原始报错 ──▶ signature（签名精确检索）──▶ search（语义检索 + 置信度）
      │                                        │
      └──────────── version（版本形态）──────────┘
                        │
                        ▼
                code（对应版本源码定位）──▶ graph（修复链路：issue→PR→release）
```

## ✨ 特性

- **三层签名提取**：源码符号表（算子/特性自动生成）+ 结构化解析（Python 堆栈/键值对/ACL 错误码）+ 社区信号词，替代手写正则——新增算子只需重新索引代码仓
- **混合检索 + 置信度动态计算**：向量（LanceDB）+ 全文（FTS5）+ 标题精确匹配，置信度按时间衰退 / 版本区间 / 来源可靠度在查询时实时计算；
  **验证状态因子**（expert 官方手册 0.95 / tested 0.85 / unverified 0.5）并入可靠度
- **图存储（Kùzu）**：Issue/PR/Release/Operator/ErrorCode/Model/Version/**Doc/Interface** 节点 +
  FIXES/MERGED_IN/MENTIONS/DOCUMENTS 边——**修复链路追溯**（`graph chain`：issue→修复 PR→落地 release，
  "这个修复是否已进入我的部署版本"）、**手册定义查询**（错误码/命令在哪个手册定义）
- **版本化代码仓**：预存 vllm-ascend 各版本 + 对应 vllm 主仓源码快照，按部署版本定位 `file:line`；
  `--in-file` 限定文件检索、`--per-version` 版本差异对比、`diff` 命令跨版本精确 diff 定位修复引入版本
  （`--keyword` 过滤相关差异行）；`--file --max-chars` 读取完整源码（截断带明确标记）、
  `--list` 对比可用/已存/缺失版本
- **版本日历**：正式 release / rc 形态判断（`version 0.18.0` → release），置信度版本上界映射
- **组件配套矩阵**：vllm-ascend → vllm/cann/pytorch-ascend 自动匹配——vllm 取镜像 `VLLM_TAG`
  （构建锁定，优先于 release 说明）、cann 缺失按同系列回退、PTA 从对应 tag 的 `requirements.txt`
  提取（0day 模型经 vllm 版本关联）；写回前版本号正则校验，非法值置空
- **业务来源导入**：PDF 手册（文字层 + 表格→结构化 JSON/错误码/命令 → 图）、Markdown（图片自动收集、正文不透明占位）、
  Excel 登记表（**schema-free**：任意 sheet/列序拼接入库，每行一条文档）、
  截图**签名导向 OCR**（provider 可插拔：api/custom、openai 兼容如 DeepSeek-OCR、paddle、ask 交互询问）；
  入库自动打**文档级两级标签**（主题/领域类 + 具体作用类，确定性提取自文件名+内部标题，
  词典 `config.tags.registry` 驱动）——经 skill 的 `tags`/`context` 命令做**能力发现**
  （agent 先知道"知识库有哪些文档类别可提供知识"，如 HCCL 超时 → 命中 HCCL 领域 +
  超时排查/命令参考作用类，先读文档再下结论）；**资产路径不进库**（asset_id 标识 + API 出口白名单清理，
  管理员侧路径仅存审核库）
- **内部数据脱敏（后置）**：库中存原文（原文检索）、serve_api 出口统一脱敏（内部 IP → `<IP>`、内部路径 → `<PATH>`，
  默认路径如 `/var/log/npu/` 保留）——改 `config.sanitize`（keep_paths/keep_ips/sources）**即时生效、无需重嵌**；
  被脱敏的原始 IP/路径落盘 `data/sanitize_log.json` 供维护白名单
- **审核工作台**（Web UI）：人工确认统一入口（认证 / 存疑 / 删除+撤回，删除只动数据库记录、原始文件保留）、
  **两层标签治理**（自动标签排除/恢复、人工添加、词典管理——新增/改名/改 tier 同步 config.json，
  重建图后入图；tag_candidate 候选按词聚合、采纳即入词典 + 全部提及文档批量打标（检索侧立即生效，
  图侧重建后入图，向量侧重入库后一致）；同 stem 重名告警）、
  **API 配置中心**（embedding/OCR/GitHub 配置编辑，密钥脱敏存 `data/secrets.local.json`，连通性测试）、
  **文档管理**（外源文档列表 + 彻底删除：docs+chunks+向量四层，本地文件保留可重新入库）——
  启动与操作见 [使用指南 §3.5](docs/USAGE.md#35-审核工作台人工确认统一入口--api-配置中心)
- **平稳降级**：embedding 服务不可用时检索自动降级为全文检索——查询用快速失败客户端（5s）+ 熔断器
  （连续失败 3 次熔断 60s，零等待降级，到期自动探测恢复）；`/health` 暴露 embedding 状态
- **内网部署支持**：所有联网脚本（代码快照/版本日历/配套矩阵）支持 `--insecure` 跳过 SSL 校验 +
  `--github-base/--quay-base/--base-url` 换内网 http 镜像，环境变量统一配置；
  **注意：Kùzu 图库路径（`data/graph` / `VLLM_KB_DATA_ROOT`）不能含非 ASCII 字符**（中文/emoji，
  中文部署根会打不开图库——数据根放纯 ASCII 路径，详见 [使用指南](docs/USAGE.md#32-图更新流程kùzu-单写者约束)）
- **存算分离**：skill 仅两个文件（`SKILL.md` + `client.py`，约 50KB，标准库实现零依赖），
  数据（向量库/索引/图，本仓库样例 ~1GB，全量构建 4~6GB）放远程服务器，本地只发 HTTP 查询；
  `scripts/pack_migrate.py` 打包迁移（业务环境重新嵌入，不传向量库）、`scripts/deploy_remote.py` 远程部署辅助
- **结构只读**：SQLite `mode=ro` + 向量库只读包装 + 无写端点，Agent 提示注入也无法修改知识库
- **高危操作防护**：`build_kb.py --rebuild` 执行前强制确认（TTY 交互 y/yes，非交互需 `--yes`），
  防止误触发全量重嵌
- **总日志接口**：打屏（默认）+ 可选落盘分卷（RotatingFileHandler，config `logging` 段开启）
- **离线可用**：数据采集完成后，全部检索不依赖网络

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 磁盘：全量构建（vllm + vllm-ascend 双仓社区数据 + 代码快照 + 图）约 4~6GB；
  本仓库 `data/` 为已构建样例（~1GB：16k+ 文档 / 31k+ chunks）。若曾用旧版库反复增量入库，
  LanceDB 会累积历史版本致体积膨胀数十 GB，可用 `cleanup_old_versions()` 清理，见 [使用指南 §7](docs/USAGE.md#7-常见问题)
- 外部依赖：GitHub token（采集）、Embedding API key（**强制 API**：OpenAI 兼容端点，可指向其他服务器的 vLLM 部署；`echo` 仅离线演示）

### 安装与配置

```bash
git clone <repo> && cd vllm-kb
pip install -r requirements.txt

# 配置（密钥走环境变量或审核工作台，不入 config.json；模板见 .env.example）
cp config.example.json config.json
export GITHUB_TOKEN=ghp_xxx            # GitHub 采集 token
export EMBEDDING_API_KEY=sk-xxx        # Embedding API key（OpenAI 兼容端点）
```

> **离线体验（无 GitHub token / 无 embedding key）**：仓库自带离线演示配置
> `config.offline.json`（`echo` 嵌入 + 纯 Python 向量后端），配合 `scripts/seed_demo.py`
> 用模拟数据不联网跑通全链路（效果粗糙，仅用于验证流程）：
>
> ```bash
> python scripts/seed_demo.py                                          # 写入模拟 GitHub 原始数据
> python scripts/build_kb.py --config config.offline.json --skip-pull    # 离线构建知识库
> python scripts/verify.py --config config.offline.json --version 0.6.1  # 验收查询
> ```

### 采集数据并构建知识库

```bash
# 全量拉取（issues/PRs/comments）+ 入库（可中断，重跑续传）
python scripts/build_kb.py

# 小批量试跑
python scripts/build_kb.py --limit 100

# 图存储（Kùzu：修复链路/手册定义查询）——需先采集入库
python scripts/build_graph.py
```

### 启动检索服务

```bash
python scripts/serve_api.py            # http://127.0.0.1:8000（fastapi/uvicorn 已含在 requirements.txt）
```

> 更新图（`build_graph.py`）前必须先停止检索服务（Kùzu 单写者，见 [使用指南](docs/USAGE.md#32-图更新流程kùzu-单写者约束)）。

### 启动审核工作台（Web UI）

人工确认统一入口 + API 配置中心（启动后浏览器打开 `http://127.0.0.1:8010`）：

```bash
python scripts/review_ui.py            # http://127.0.0.1:8010（自动补单，幂等）
```

- **审核**：未验证文档补标、案例标题待审核、OCR 图文不一致、低置信度签名、跨来源合并候选等
  6 类待办，逐条 **✓ 认证 / ？存疑 / 🗑 标记删除 / ↩ 撤回**（删除只动数据库记录、原始文件保留）；
- **API 配置中心**：集中编辑 embedding / OCR / GitHub 配置（非密钥进 config.json，
  密钥脱敏存 `data/secrets.local.json`），embedding / OCR 均支持连通性测试；
- 完整操作说明（含审核状态机、待实际删除列表）见 [使用指南 §3.5](docs/USAGE.md#35-审核工作台人工确认统一入口--api-配置中心)。

### 查询

```bash
# 语义检索（组件:版本 问题描述）
python skills/vllm-kb/client.py search "vllm-ascend:0.23.0rc1 GLM5.1 PD分离P节点挂死"

# 签名精确检索（贴原始报错）
python skills/vllm-kb/client.py signature "halMemCreate failed drvRetCode=6, kernel_name=DispatchFFNCombine"

# 标题精确检索（已知现象找 issue）
python skills/vllm-kb/client.py title "vector core" --component vllm-ascend

# 版本形态判断
python skills/vllm-kb/client.py version 0.18.0          # → release（正式版）

# 对应版本源码定位
python skills/vllm-kb/client.py code DispatchFFNCombine --version v0.23.0rc1
python skills/vllm-kb/client.py code make_zmq_socket --repo vllm --version 0.22.1
python skills/vllm-kb/client.py code "fill_(-1)" --in-file worker/model_runner_v1.py --per-version  # 定位修复引入版本
python skills/vllm-kb/client.py diff v0.22.1rc1 v0.23.0rc1 vllm_ascend/worker/model_runner_v1.py --keyword "fill_(-1)"  # 跨版本精确 diff（新增行=修复引入点）

# 图检索：修复链路追溯（issue→修复 PR→落地 release）
python skills/vllm-kb/client.py graph chain vllm-ascend#10700
python skills/vllm-kb/client.py graph fixes vllm-ascend#12885
python skills/vllm-kb/client.py graph sig dispatch_ffn_combine
```

完整用法见 [使用指南](docs/USAGE.md)。

## 📦 数据更新

```bash
# 1. 日常更新（增量入库；GitHub 首次全量后默认不再拉取——日志打印 done 跳过说明）
python scripts/build_kb.py

# 1b. 拉取 GitHub 社区增量（新增 issue/PR；时间窗口：上次增量 max createdAt 起，
#     issues 服务端 filterBy.since 过滤 + PR UPDATED_AT DESC 排序，连续 3 页无新增停止）
python scripts/build_kb.py --incremental
#    全量重拉（数据刷新）：删除 data/raw/{source_id}/ 与 data/checkpoints/{source_id}.json 后重跑

# 1c. 其他拉取模式（与 --incremental 互斥）：
python scripts/build_kb.py --pull-missing         # 补差：从头枚举，跳过已有（raw/checkpoint），只拉缺失（补历史旧条目）
python scripts/build_kb.py --numbers 9749,9750    # REST 单条补拉指定编号（走 REST，无需 GraphQL token）

# 2. 只重新入库，不拉取（改配置/规则后）
python scripts/build_kb.py --skip-pull

# 3. 只再生 canonical（提取逻辑升级后更新 canonical 供建图用，不入库、不重嵌向量）
python scripts/build_canonical.py
#    （build_graph 用新 canonical 建图前先跑本脚本；如需连带重入库再跑 build_kb.py --skip-pull）

# 4. 换 embedding 模型 / 全量重建（高危，需 TTY 确认或 --yes）
python scripts/build_kb.py --rebuild

# 5. 版本日历（GitHub Releases → 版本形态 + 置信度上界；--all-repos 生成 vllm/vllm-ascend 分仓日历）
python scripts/build_release_calendar.py --all-repos

# 6. 版本化代码仓快照（zips/{version}.zip + snapshots/{version}/ + index.sqlite3 + symbols.json）
python scripts/build_code_snapshots.py          # vllm-ascend 版本
python scripts/build_vllm_snapshots.py          # 对应 vllm 主仓版本（配套矩阵映射，自动跟随）
#    符号索引（index.sqlite3）是派生数据：提取规则/schema 升级后重建
#    python scripts/build_code_snapshots.py --index-only   # 索引+符号表+信号词，无需迁移

# 7. 组件配套矩阵（vllm-ascend → vllm/cann/pytorch-ascend 自动匹配；quay tag 辅助获取）
python scripts/build_companion_matrix.py        # 生成/更新 data/compatibility/vllm-ascend.json
python scripts/fetch_quay_tags.py               # 拉 quay.io 镜像 tag（看护策略过滤日构建/分支/主干）

# 8. 图存储重建（修复链路/手册定义；需先停检索服务，Kùzu 单写者）
python scripts/build_graph.py

# 9. FTS 全文索引重建（jieba 中文分词，可选——装 jieba 或升级分词规则后跑；不重嵌向量）
python scripts/build_fts.py

# 10. 社区高频信号词统计（issue 标题 TF-IDF → data/code/signal_words.json，供 agent 判断）
python scripts/build_signal_words.py

# 11. 正文 TF-IDF 标签候选导出（jieba → candidates.json 文件，人工审阅后手动同步 config.tags.registry）
python scripts/build_tag_candidates.py
```

> 更新前建议停止检索 API，更新完重启（尤其 `--rebuild` / `build_graph.py` 后必须重启）。

**维护/验证脚本**：

| 脚本 | 用途 |
|---|---|
| `scripts/verify.py` | 验收：对真实故障报错跑查询，打印排序结果与置信度分解（默认查询见 config `verify.queries`） |
| `scripts/check_readonly.py` | 验证知识库只读姿态（SQLite mode=ro / 向量库只读包装，结构层面，不依赖提示词） |
| `scripts/build_canonical.py` | 只再生统一 canonical（不入库）：提取逻辑升级后更新 canonical 供建图/重建，不重嵌向量 |
| `scripts/backfill_canonical.py` | 修复 kb↔canonical 不同步：从 kb.sqlite3 重建 canonical 缺失文档（默认 dry-run 打印清单，`--write` 回填，`--doc` 可只补指定条）——kb 检索命中但 graph/rebuild 缺文档时用 |
| `scripts/diff_code_versions.py` | 跨版本源码 diff 的仓库侧等价实现（client `diff` 走只读 API 即可） |
| `scripts/deploy_remote.py` | 远程部署辅助：本地 skill（算）+ 远程数据/API（存）落地 |
| `scripts/pack_migrate.py` | 迁移打包：业务环境重新嵌入的最小集（canonical + 业务数据，不传向量库） |
| `scripts/seed_demo.py` | 离线演示：写入模拟 GitHub 原始数据（配 `config.offline.json` 跑通全链路） |

**业务来源导入**（PDF 手册 / Markdown / Excel 登记表 / 截图 OCR）：文件放 `data/imports/{pdf,md,xlsx}/`，
启用 config 对应 source 后跑 `python scripts/build_kb.py`；详见
[使用指南 §2.3](docs/USAGE.md#23-业务来源导入pdf-手册--markdown-文档--完整实操) /
[§2.4 Excel 导入](docs/USAGE.md#24-excel-登记表导入schema-free--完整实操)。

## 🧠 知识库结构

```
data/
├── raw/                    # 采集原始快照（JSON，事实源，可重放，按来源分目录）
│   └── canonical.jsonl     # 统一 Canonical 中间格式（全量构建 66k+ 条；本仓库样例 16k+ 条）
├── lancedb/                # 向量库（bge-m3，全量 122k+ chunks；样例 31k+）
├── kb.sqlite3              # SQLite：docs 元数据 + chunks_meta 分块 + chunks_fts 全文（jieba 分词）+ doc_tags 标签覆盖层
├── code/                   # 版本化代码仓
│   ├── zips/               # vllm-ascend 各版本源码 zip
│   ├── snapshots/          # 解压后的源码快照（按需读取 / diff，版本子目录）
│   ├── index.sqlite3       # 符号索引（算子名→文件:行号 + 报错字面量索引）
│   ├── symbols.json        # 三层签名提取的符号表
│   └── signal_words.json   # 社区高频信号词（build_signal_words.py 生成，供 agent 判断）
├── graph/                  # Kùzu 图（Issue/PR/Release/Doc/Interface/Tag + FIXES/MERGED_IN/MENTIONS/DOCUMENTS/CORROBORATES/TAGGED_WITH）
├── assets/                 # 业务来源原始资产（pdf/md/images，不可变，sha256）
├── parsed/                 # 解析产物（PDF 表格 JSON、OCR 结果，可重跑；建图时提取表格→错误码）
├── imports/                # 业务数据放置目录（pdf/md/xlsx）
├── review.sqlite3          # 审核工作台队列（认证/存疑/删除）
├── tag_candidates_manual.json  # 标签候选人工审阅输出（可手动同步进 config.tags.registry）
├── checkpoints/            # 采集断点（续传）
├── sanitize_log.json       # 脱敏维护日志（被脱敏的原始 IP/路径，供调整白名单）
├── secrets.local.json      # 本地密钥（审核工作台写入；AppConfig 自动加载，不入库）
└── compatibility/          # 组件配套矩阵 + 分仓版本日历
    ├── vllm-ascend.json                        # 配套矩阵（vllm-ascend → vllm/cann/pytorch-ascend）
    └── release_calendar.{repo}.json            # 版本日历（--all-repos 按仓库分文件）
```

## 🔧 架构

```
采集/导入层
   ├─ GitHub REST + GraphQL（issues/PRs/comments/releases）
   ├─ PdfSource / MarkdownSource / ExcelSource（业务来源：资产层 + 解析层；Excel schema-free）
   └─ ImageSource（签名导向 OCR，provider 可插拔）
   ▼
Canonical 规范化（统一中间格式，可重放可重嵌）
   ▼
入库流水线（幂等 + 断点续传）
   ├─ LanceDB 向量库（批量写入，ETA 优化 40×）
   ├─ SQLite（docs + FTS5 全文，jieba 中文分词）
   ├─ 版本化代码仓（zip 快照 + 符号索引）
   ├─ 脱敏日志收集（原始 IP/路径 → data/sanitize_log.json）
   └─ Kùzu 图（Issue/PR/Release/Doc/Interface/Tag + FIXES/MERGED_IN/MENTIONS/DOCUMENTS/CORROBORATES/TAGGED_WITH）
   ▼
检索层（只读 API / skill）
   ├─ signature   签名精确检索（三层提取）
   ├─ search      语义检索 + 置信度动态计算（含验证状态因子）
   ├─ title       标题精确检索
   ├─ version     版本形态判断
   ├─ code        对应版本源码定位（--in-file/--per-version）
   ├─ graph       修复链路追溯 + 手册定义查询
   └─ tags       文档标签能力发现（领域/作用过滤 + 证据互证）
   出口统一脱敏（_out/_sanitize_graph：内部 IP/路径 → 占位，改配置即时生效）
```

**存算分离**：数据可在远程服务器，本地 skill 通过 `VLLM_KB_BASE` 环境变量寻址；服务端数据路径由 `VLLM_KB_DATA_ROOT` 重定向。详见 [使用指南](docs/USAGE.md#远程部署存算分离)。

## 🔀 数据流（常规操作）

两条主链路：**入库**（数据 → 落库）与 **查询**（agent → API → 存储）。详细版见 [docs/DATAFLOW.md](docs/DATAFLOW.md)。

### 入库：数据从哪里来、如何处理、存到哪里

```
GitHub 社区（vllm / vllm-ascend issues/PRs/comments）
   │  build_kb.py ── GithubSource.pull()（REST+GraphQL，限流/断点续传/增量）
   ▼
data/raw/{source_id}/（原始 JSON 快照，事实源，可重放）── canonicalize() ──▶ 统一 canonical.jsonl
                                                                              （中间格式，可重放可重嵌）
业务来源（PDF/MD/Excel/截图，放 data/imports/）
   │  build_kb.py ── 资产层复制（data/assets/，sha256 不可变）＋ 解析（data/parsed/：表格 JSON / OCR 结果）
   ▼
   canonicalize()（文档级标签 tagging）──────────────▶ 同上 canonical.jsonl
   ▼
ingest_docs()（幂等双哈希增量 + 断点续传）
   ├─ chunking 分段切块 ──▶ embed（OpenAI 兼容 /embeddings）──▶ LanceDB 向量库（data/lancedb）
   └─ 写 SQLite（data/kb.sqlite3）：docs 元数据 + chunks_fts 全文（jieba 分词）+ chunks_meta + doc_tags 标签覆盖层

版本化代码仓：build_code_snapshots.py / build_vllm_snapshots.py
   ──▶ data/code/{zips,snapshots}/ 源码快照 + index.sqlite3 符号索引 + symbols.json / signal_words.json

辅助数据：build_release_calendar.py ──▶ data/compatibility/release_calendar.{repo}.json
        build_companion_matrix.py + fetch_quay_tags.py ──▶ data/compatibility/vllm-ascend.json

图存储：build_graph.py（需先停检索服务，Kùzu 单写者）
   ──▶ 读 canonical.jsonl + data/parsed/（表格→错误码）+ kb.sqlite3（人工标签覆盖层）──▶ Kùzu data/graph
```

各存储的写入方与内容：

| 存储 | 写入方 | 内容 |
|---|---|---|
| `data/raw/{source_id}/` | `build_kb.py` 拉取阶段 | GitHub 原始 JSON 快照（按来源分目录；checkpoints 断点续传） |
| `data/raw/canonical.jsonl` | `build_kb.py` canonicalize | 多来源统一中间格式（按 source_id upsert，幂等） |
| `data/lancedb` | `ingest.py`（批量攒批写入） | chunk 向量 + 原文 + meta（title/组件/版本区间/标签/section…） |
| `data/kb.sqlite3` | `ingest.py` | `docs`（文档元数据+哈希）、`chunks_fts`（jieba 分词全文）、`chunks_meta`（分块序号/章节）、`doc_tags`（人工标签覆盖层） |
| `data/code/*` | `build_code_snapshots.py` / `build_vllm_snapshots.py` | 各版本源码 zip + 解压快照 + 符号/报错字面量索引 + 符号表/信号词 |
| `data/compatibility/*` | `build_release_calendar.py` / `build_companion_matrix.py` / `fetch_quay_tags.py` | 分仓版本日历 + 组件配套矩阵 |
| `data/graph` | `build_graph.py` | Kùzu 图（Issue/PR/Release/Doc/Interface/Tag 节点 + 6 类边） |
| `data/review.sqlite3` | `review_ui.py` 审核操作 | 审核队列（认证/存疑/删除），不参与检索 |

### 查询：agent 请求从哪个接口进来、查什么库

Agent 只调用 skill（`skills/vllm-kb/client.py`，标准库零依赖）→ HTTP 打到 `scripts/serve_api.py`
启动的**只读 FastAPI**（默认 `http://127.0.0.1:8000`，远程经 `VLLM_KB_BASE`）→ 各端点按需读
对应存储。检索服务结构只读：SQLite `mode=ro`、向量库写操作抛错、无写端点。

| client 命令 | HTTP 端点（方法） | 查询的存储 |
|---|---|---|
| `search` | `POST /search` | LanceDB 向量召回 + kb.sqlite3 FTS5 全文（BM25，jieba 分词）+ 配套矩阵/分仓版本日历（查询期现算置信度，不落库） |
| `signature` | `POST /signature-search` | 现场提取签名（`data/code/symbols.json` 符号表 + `signal_words.json`）→ kb.sqlite3 FTS 短语匹配 + 标题匹配 |
| `title` | `GET /title` | kb.sqlite3 `docs` 表（title/source_id SQL LIKE） |
| `version` | `GET /version` | `data/compatibility/release_calendar.{repo}.json`（版本形态判断） |
| `code` | `POST /code/search` | `data/code/index.sqlite3` 符号/报错字面量索引命中，未命中退 grep 版本快照（snapshots/ 或 zips/）；`kind=msg` 走报错字面量索引 |
| `code --file` | `GET /code/file` | `data/code` 对应版本快照按需解压读取（截断带标记） |
| `diff` | `GET /code/diff` | 两个版本快照同一文件的 unified diff |
| `code-versions` | `GET /code/versions` | `data/code` 可用预存版本清单 |
| `doc` | `GET /doc/{source_id}` | kb.sqlite3 `docs` + `chunks_meta` + `chunks_fts` 按序拼装全文 |
| `components` / `stats` / `health` | `GET` | kb.sqlite3 聚合 / 向量库 count（`/health` 含 embedding 状态） |
| `companion` / `matrix` | `GET /companion` `/matrix` | `data/compatibility/vllm-ascend.json` 配套矩阵 |
| `graph chain/fixes/sig/doc/tags/evidence/stats` | `GET /graph/*` | Kùzu `data/graph`（只读查询，未构建返回引导提示） |
| `tags list` / `tags docs` / `context` | `GET /tags` `/tags/{tag}/docs` `POST /tags/match` | kb.sqlite3 `docs.tags`（最终标签）+ `config.tags.registry` 词典 |

所有出口统一后置脱敏（`sanitize.py`：内部 IP/路径 → 占位，白名单见 config `sanitize`）；
embedding 服务不可用时 `search`/`signature` 自动降级为全文检索（快速失败客户端 + 熔断器）。

## 模块结构（`vllm_kb/`）

| 模块 | 职责 |
|---|---|
| `sources.py` / `github_pull.py` | 数据源适配器（`BaseSource`：github/markdown/pdf/excel/image）+ GitHub 采集（限流、断点续传、评论 GraphQL 内联、`--incremental` 增量） |
| `models.py` / `config.py` | Canonical 统一中间格式 + 唯一配置入口（旧版单源折叠兼容；secrets 自动加载） |
| `chunking.py` / `embed.py` | 讨论线按段切块 + 批量嵌入（攒批降 API 调用） |
| `ingest.py` / `vectorstore.py` / `pipeline.py` | 幂等入库流水线：SQLite+FTS5、LanceDB 批量写入、`build_kb.py` 入口 |
| `signature.py` / `error_parse.py` / `symbol_table.py` | 三层签名提取（源码符号表 → 结构化解析 → 社区信号词） |
| `tagging.py` | 文档级标签（两级分类）确定性提取：词典 registry 子串命中 + 文件名/标题 token；tier 启发式；合并公式唯一实现 `final=(auto−excluded)∪manual`（ingest 与建图共用） |
| `search.py` / `confidence.py` | 混合检索（向量+FTS+标题）+ 查询时置信度（时间衰退/版本区间/来源可靠度/验证状态） |
| `fts_tokenizer.py` | FTS5 中文分词（jieba 可选，未装降级原文）：入库侧分词写索引、查询侧分词构造 MATCH——中文词独立索引（"超时"可命中"超时排查"），词典词注册防拆分 |
| `graph.py` / `graph_rels.py` | Kùzu 图：建图（FIXES/MERGED_IN/MENTIONS/DOCUMENTS/CORROBORATES/TAGGED_WITH，含手册表格→错误码、命令格式→Interface、文档互证）+ 链路查询（tags/evidence/sig 大小写不敏感） |
| `sanitize.py` | 内部数据脱敏（后置）：出口统一脱敏（IP/路径白名单 keep_paths/keep_ips）+ 入库命中收集（sources）→ `data/sanitize_log.json` |
| `code_index.py` / `companion.py` / `components.py` | 版本化代码仓符号索引（grep path/per-version）、配套矩阵、组件分布 |
| `review.py` / `secrets.py` | 审核队列（认证/存疑/删除+撤回）+ 外源文档管理（四层彻底删除）+ 本地密钥文件 |
| `ocr.py` | 签名导向 OCR：api(custom/openai 兼容)/paddle/none，可插拔 |
| `net.py` | 网络统一入口：内网模式（跳过 SSL 校验 + GitHub/quay 镜像源覆盖，环境变量配置） |
| `logging_setup.py` | 总日志：打屏 + 可选落盘分卷（RotatingFileHandler） |
| `api.py` | 只读 FastAPI 检索服务（SQLite `mode=ro`、向量库只读包装、无写端点、/graph/* 与 /tags/* 端点、/health 含 embedding 状态） |

## 🗺️ 版本计划

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | 最小链路（拉取→规范化→嵌入→检索→置信度） | ✅ 完成 |
| 1 | 全量采集 + 版本日历 | ✅ 完成 |
| 2 | 图 + 向量双存储（Kùzu，修复链路/手册定义） | ✅ 核心完成（Issue/PR/Release/Doc/Interface/Tag + FIXES/MERGED_IN/MENTIONS/DOCUMENTS/CORROBORATES/TAGGED_WITH）；多来源（PDF/MD/Excel/OCR/审核工作台）✅；Evidence 互证 ✅；等价合并 🔲 |
| 3 | MCP Server 封装（任意 MCP 客户端接入） | 🔲 规划 |
| 4 | wiki/文档通用 adapter（word/html 等） | 🔲 规划（excel 已提前完成：schema-free 导入 ✅） |
| 5 | 评估集 + 置信度参数调优（真实故障案例） | 🔲 规划 |

## 🧪 测试

```bash
python -m unittest discover tests    # 453 个测试（含文档-命令一致性核验、API 无路径泄漏审计）
```

## 📄 文档

- [使用指南](docs/USAGE.md) —— 完整命令手册、业务来源导入、审核工作台、日志接口、远程部署
- [数据流说明](docs/DATAFLOW.md) —— 入库 / 查询两条主链路的数据走向、端点↔存储映射、SQLite 表结构
- [故障分析案例](docs/analysis/) —— 社区问题定位的实战记录（comm-bind / mte-repeat / port-65536 / task2 等）
- [Phase 2 数据布局](docs/analysis/phase2-data-model.md) —— 多来源统一存储与质量分级设计
- [业务环境迁移清单](docs/analysis/business-migration-checklist.md) —— 迁移业务环境的差距清单与跟踪

## 🤝 贡献

欢迎提交 PR / issue。开发注意：

- 代码遵循 `vllm_kb/` 模块化结构，新增数据源实现 `BaseSource` 适配器即可；
- 环境变量约定：`GITHUB_TOKEN`（采集）、`EMBEDDING_API_KEY`（嵌入）、`VLLM_KB_BASE`（远程 API 地址）、`VLLM_KB_DATA_ROOT`（远程数据根）；
- 密钥一律走环境变量或审核工作台写入的 `data/secrets.local.json`，config.json 不入库（模板见 `.env.example`）。

## 📜 许可

MIT
