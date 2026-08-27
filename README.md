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
  截图**签名导向 OCR**（provider 可插拔：api/custom、openai 兼容如 DeepSeek-OCR、paddle、ask 交互询问）；
  入库自动打**文档级两级标签**（主题/领域类 + 具体作用类，确定性提取自文件名+内部标题，
  词典 `config.tags.registry` 驱动）——经 skill 的 `tags`/`context` 命令做**能力发现**
  （agent 先知道"知识库有哪些文档类别可提供知识"，如 HCCL 超时 → 命中 HCCL 领域 +
  超时排查/命令参考作用类，先读文档再下结论）；**资产路径不进库**（asset_id 标识 + API 出口白名单清理，
  管理员侧路径仅存审核库）
- **审核工作台**（Web UI）：人工确认统一入口（认证 / 存疑 / 删除+撤回，删除只动数据库记录、原始文件保留）、
  **两层标签治理**（自动标签排除/恢复、人工添加、词典管理——新增/改名/改 tier 同步 config.json，
  重建图后入图；tag_candidate 候选采纳；同 stem 重名告警）、
  **API 配置中心**（embedding/OCR/GitHub 配置编辑，密钥脱敏存 `data/secrets.local.json`，连通性测试）、
  **文档管理**（外源文档列表 + 彻底删除：docs+chunks+向量四层，本地文件保留可重新入库）——
  启动与操作见 [使用指南 §3.5](docs/USAGE.md#35-审核工作台人工确认统一入口--api-配置中心)
- **平稳降级**：embedding 服务不可用时检索自动降级为全文检索——查询用快速失败客户端（5s）+ 熔断器
  （连续失败 3 次熔断 60s，零等待降级，到期自动探测恢复）；`/health` 暴露 embedding 状态
- **内网部署支持**：所有联网脚本（代码快照/版本日历/配套矩阵）支持 `--insecure` 跳过 SSL 校验 +
  `--github-base/--quay-base/--base-url` 换内网 http 镜像，环境变量统一配置
- **存算分离**：skill 仅 ~34KB，数据（向量库/索引/图，约 0.8GB 干净库）放远程服务器，本地只发 HTTP 查询；
  `scripts/pack_migrate.py` 打包迁移（业务环境重新嵌入，不传向量库）
- **结构只读**：SQLite `mode=ro` + 向量库只读包装 + 无写端点，Agent 提示注入也无法修改知识库
- **高危操作防护**：`build_kb.py --rebuild` 执行前强制确认（TTY 交互 y/yes，非交互需 `--yes`），
  防止误触发全量重嵌
- **总日志接口**：打屏（默认）+ 可选落盘分卷（RotatingFileHandler，config `logging` 段开启）
- **离线可用**：数据采集完成后，全部检索不依赖网络

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 磁盘：数据约 4~6GB（canonical/原始快照/代码快照/向量库/图；若曾用旧版库反复增量入库，
  LanceDB 会累积历史版本致体积膨胀数十 GB，可用 `cleanup_old_versions()` 清理，见 [使用指南 §7](docs/USAGE.md#7-常见问题)）
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
pip install fastapi uvicorn
python scripts/serve_api.py            # http://127.0.0.1:8000
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
# 1. 日常更新：拉取新数据 + 增量入库（断点续传）
python scripts/build_kb.py

# 2. 只重新入库，不拉取（改配置/规则后）
python scripts/build_kb.py --skip-pull

# 3. 代码升级后重新生成 canonical 并入库（不重拉 GitHub）
python scripts/build_kb.py --recanonicalize

# 4. 换 embedding 模型 / 全量重建
python scripts/build_kb.py --rebuild

# 5. 版本日历（GitHub Releases → 版本形态 + 置信度上界）
python scripts/build_release_calendar.py --all-repos

# 6. 版本化代码仓快照
python scripts/build_code_snapshots.py          # vllm-ascend 版本
python scripts/build_vllm_snapshots.py          # 对应 vllm 主仓版本
#    符号索引（index.sqlite3）是派生数据：提取规则/schema 升级后重建
#    python scripts/build_code_snapshots.py --index-only   # 索引+符号表+信号词，无需迁移

# 7. 图存储重建（修复链路/手册定义；需先停检索服务）
python scripts/build_graph.py
```

> 更新前建议停止检索 API，更新完重启（尤其 `--rebuild` / `build_graph.py` 后必须重启）。

**业务来源导入**（PDF 手册 / Markdown / 截图 OCR）：文件放 `data/imports/{pdf,md}/`，启用 config 对应 source
后跑 `python scripts/build_kb.py`；详见 [使用指南 §2.3](docs/USAGE.md#23-业务来源导入pdf-手册--markdown-文档--完整实操)。

## 🧠 知识库结构

```
data/
├── raw/                    # GitHub 原始快照（JSON，事实源，可重放）
│   └── canonical.jsonl     # 统一 Canonical 中间格式（66k+ 条）
├── lancedb/                # 向量库（bge-m3，122k+ chunks）
├── kb.sqlite3              # 文档元数据 + FTS5 全文索引
├── code/                   # 版本化代码仓（zips + 符号索引 + 符号表）
│   ├── zips/               # vllm-ascend 各版本源码 zip
│   ├── vllm/               # 对应 vllm 主仓源码（repo 隔离）
│   ├── index.sqlite3       # 符号索引（算子名→文件:行号）
│   └── symbols.json        # 三层签名提取的符号表
├── graph/                  # Kùzu 图（Issue/PR/Release/Doc/Interface + FIXES/MERGED_IN/MENTIONS/DOCUMENTS）
├── assets/                 # 业务来源原始资产（pdf/md/images，不可变，sha256）
├── parsed/                 # 解析产物（PDF 表格 JSON、OCR 结果，可重跑）
├── imports/                # 业务数据放置目录（pdf/md）
├── review.sqlite3          # 审核工作台队列（认证/存疑/删除）
├── checkpoints/            # 采集断点（续传）
└── compatibility/          # 组件配套矩阵 + 版本日历
```

## 🔧 架构

```
采集/导入层
   ├─ GitHub REST + GraphQL（issues/PRs/comments/releases）
   ├─ PdfSource / MarkdownSource（业务来源：资产层 + 解析层）
   └─ ImageSource（签名导向 OCR，provider 可插拔）
   ▼
Canonical 规范化（统一中间格式，可重放可重嵌）
   ▼
入库流水线（幂等 + 断点续传）
   ├─ LanceDB 向量库（批量写入，ETA 优化 40×）
   ├─ SQLite（docs + FTS5 全文）
   ├─ 版本化代码仓（zip 快照 + 符号索引）
   └─ Kùzu 图（Issue/PR/Release/Doc/Interface + FIXES/MERGED_IN/MENTIONS/DOCUMENTS）
   ▼
检索层（只读 API / skill）
   ├─ signature   签名精确检索（三层提取）
   ├─ search      语义检索 + 置信度动态计算（含验证状态因子）
   ├─ title       标题精确检索
   ├─ version     版本形态判断
   ├─ code        对应版本源码定位（--in-file/--per-version）
   └─ graph       修复链路追溯 + 手册定义查询
```

**存算分离**：数据可在远程服务器，本地 skill 通过 `VLLM_KB_BASE` 环境变量寻址；服务端数据路径由 `VLLM_KB_DATA_ROOT` 重定向。详见 [使用指南](docs/USAGE.md#远程部署存算分离)。

### 模块结构（`vllm_kb/`）

| 模块 | 职责 |
|---|---|
| `sources.py` / `github_pull.py` | 数据源适配器（`BaseSource`：github/markdown/pdf/image）+ GitHub 采集（限流、断点续传、评论 GraphQL 内联） |
| `models.py` / `config.py` | Canonical 统一中间格式 + 唯一配置入口（旧版单源折叠兼容；secrets 自动加载） |
| `chunking.py` / `embed.py` | 讨论线按段切块 + 批量嵌入（攒批降 API 调用） |
| `ingest.py` / `vectorstore.py` / `pipeline.py` | 幂等入库流水线：SQLite+FTS5、LanceDB 批量写入、`build_kb.py` 入口 |
| `signature.py` / `error_parse.py` / `symbol_table.py` | 三层签名提取（源码符号表 → 结构化解析 → 社区信号词） |
| `tagging.py` | 文档级标签（两级分类）确定性提取：词典 registry 子串命中 + 文件名/标题 token；tier 启发式；合并公式唯一实现 `final=(auto−excluded)∪manual`（ingest 与建图共用） |
| `search.py` / `confidence.py` | 混合检索（向量+FTS+标题）+ 查询时置信度（时间衰退/版本区间/来源可靠度/验证状态） |
| `graph.py` / `graph_rels.py` | Kùzu 图：建图（FIXES/MERGED_IN/MENTIONS/DOCUMENTS，含手册表格→错误码、命令格式→Interface）+ 链路查询 |
| `code_index.py` / `companion.py` / `components.py` | 版本化代码仓符号索引（grep path/per-version）、配套矩阵、组件分布 |
| `review.py` / `secrets.py` | 审核队列（认证/存疑/删除+撤回）+ 外源文档管理（四层彻底删除）+ 本地密钥文件 |
| `ocr.py` | 签名导向 OCR：api(custom/openai 兼容)/paddle/none，可插拔 |
| `net.py` | 网络统一入口：内网模式（跳过 SSL 校验 + GitHub/quay 镜像源覆盖，环境变量配置） |
| `logging_setup.py` | 总日志：打屏 + 可选落盘分卷（RotatingFileHandler） |
| `api.py` | 只读 FastAPI 检索服务（SQLite `mode=ro`、向量库只读包装、无写端点、/graph 端点、/health 含 embedding 状态） |

## 🗺️ 版本计划

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | 最小链路（拉取→规范化→嵌入→检索→置信度） | ✅ 完成 |
| 1 | 全量采集 + 版本日历 | ✅ 完成 |
| 2 | 图 + 向量双存储（Kùzu，修复链路/手册定义） | ✅ 核心完成（Issue/PR/Release/Doc/Interface + FIXES/MERGED_IN/MENTIONS/DOCUMENTS）；多来源（PDF/MD/OCR/审核工作台）✅；Evidence/等价合并等依赖业务数据 🔲 |
| 3 | MCP Server 封装（任意 MCP 客户端接入） | 🔲 规划 |
| 4 | wiki/文档通用 adapter（word/html/excel 等） | 🔲 规划 |
| 5 | 评估集 + 置信度参数调优（真实故障案例） | 🔲 规划 |

## 🧪 测试

```bash
python -m unittest discover tests    # 321 个测试（含文档-命令一致性核验）
```

## 📄 文档

- [使用指南](docs/USAGE.md) —— 完整命令手册、业务来源导入、审核工作台、日志接口、远程部署
- [故障分析案例](docs/analysis/) —— 社区问题定位的实战记录
- [Phase 2 数据布局](docs/analysis/phase2-data-model.md) —— 多来源统一存储与质量分级设计
- [业务环境迁移清单](docs/analysis/business-migration-checklist.md) —— 迁移业务环境的差距清单与跟踪

## 🤝 贡献

欢迎提交 PR / issue。开发注意：

- 代码遵循 `vllm_kb/` 模块化结构，新增数据源实现 `BaseSource` 适配器即可；
- 环境变量约定：`GITHUB_TOKEN`（采集）、`EMBEDDING_API_KEY`（嵌入）、`VLLM_KB_BASE`（远程 API 地址）、`VLLM_KB_DATA_ROOT`（远程数据根）；
- 密钥一律走环境变量或审核工作台写入的 `data/secrets.local.json`，config.json 不入库（模板见 `.env.example`）。

## 📜 许可

MIT
