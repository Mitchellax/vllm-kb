# 业务环境迁移清单（vllm-kb）

> 状态：**跟踪文档**（勾选式，随迁移推进更新）。
> 目标：把 vllm-kb（Phase 0/1/2 核心 + 业务数据接入）迁移到业务环境，供故障处理工程师日常使用。
>
> **关键约定（2026 定版）**：后续数据导入验证**均在业务环境进行**——业务数据（华为案例
> PDF/Word/网页、工程师中间记录、markdown wiki、已知问题登记表、截图/拍照日志）直接在业务环境
> 采集、解析、入库、验证；开发环境的 GitHub 知识库（66,124 canonical + 41GB LanceDB + 71,389 节点图）
> 作为**初始数据**打包迁移。
>
> 前置设计文档：[Phase 2 数据布局](phase2-data-model.md)（轨道 A 完成 / 轨道 B 待实施）、
> [总体建设方案](vllm-knowledge-base-plan.md)。

---

## 1. 迁移拓扑

```
开发环境（已完成）                    业务环境（迁移目标）
─────────────────────                ─────────────────────────────
GitHub 采集（66k canonical）   ──①──▶  部署：代码 + 初始知识库（含图）
嵌入（bge-m3）                         部署：解析/OCR/审核配套
代码快照 + 版本日历               ──②──▶  业务数据采集 → 解析 → 入库 → 验证（后续全部在此）
Phase 2 图（Kùzu）                    部署：serve_api + review_ui
                                      （业务环境可能无外网：GitHub 数据一次性迁入，
                                        业务数据本地采集，embedding 走内网端点或本地模型）
```

---

## 2. 状态总览

| 批次 | 内容 | 状态 |
|---|---|---|
| 0 | 开发环境核心（检索 + 图） | ✅ 完成 |
| 1 | 数据能进库：解析层 adapter + canonical 扩展 | 🔲 待办 |
| 2 | 质量可治理：审核工作台 + 置信度融合 | 🔲 待办 |
| 3 | 图片类：签名导向 OCR | 🔲 待办 |
| 4 | 跨来源关系：图 Evidence/EQUIVALENT_TO 边 | 🔲 待办 |
| 5 | 部署运维：打包/服务/embedding/更新流程 | 🔲 部分（见 B1） |
| 6 | 标定验证：OCR 阈值 + 置信度参数 + 业务复验 | 🔲 待办 |

---

## 3. 差距清单

### A. 代码功能（轨道 B）

| # | 项 | 说明 | 验证点 | 状态 |
|---|---|---|---|---|
| A1 | 解析层 adapter | `PdfSource`（PyMuPDF 文字层 + 表格→Markdown/JSON）/ `MarkdownSource`（标题提取 + 全文）；**Excel 明确不做**（表头不固定）；Word/HTML 待业务环境 | 1 份接口指南 PDF → md + tables.json；1 份 Word 案例 → 标题层级保留 | 🚧 PDF/Markdown ✅，Word/HTML 🔲 |
| A2 | 图片链路 | `ImageSource` + 签名导向 OCR：**预留 API**（`ocr_provider: api`，POST {base}/ocr 协议）+ 本地 paddle；**无 API 时交互询问本地/跳过**（ask 默认，非 tty 自动跳过）；md 图片自动收集重写资产路径 + evidence；sha256 幂等 | 10 张真实截图签名召回率（**业务环境用 paddle 标定 B/C 阈值**）；OCR API 服务对接 | 🚧 框架 ✅（ask/api/paddle/none 决策 + 收集/重写/产物/幂等），paddle 实测 🔲 |
| A3 | canonical 扩展落地 | `extra.asset/quality/verification/evidence` 字段 + 验证状态默认规则（官方手册=expert / 案例标题解析 待审核·审核通过·待修改 / 表格=低优先级按 open issue / wiki=unverified 待补标） | 混合来源 canonical 行数 + 质量字段完整性 | 🚧 asset/quality/verification/evidence ✅（检索结果透传 verification），案例标题解析 🔲 |
| A4 | 置信度融合 | `verification_factor`（expert 0.95 / tested 0.85 / unverified 0.5，可配置）并入 `w_rel`（max 下限提升）；向量路径 meta 补全 verification | 手册检索 w_rel 0.4→0.95（✅ 实测）；样例置信度分解可见 verification | ✅ 完成 |
| A5 | 审核工作台 | `review.sqlite3` + `review_ui.py`（6 类审核项、原图 assets 预览、标注回写）+ **API 配置中心**（embedding/OCR/GitHub 配置状态，key 脱敏，embedding 连通测试） | 各类别各 1 条走完 pending→confirmed 闭环 | ✅ 框架（UI + store + seed 幂等 + 配置中心）；审核结果回写 canonical 热更新 🔲 |
| A6 | 图扩展 | `Doc` 通用节点 + `DOCUMENTS` 边（手册表格 → ErrorCode；**手册"命令格式"段 → Interface 节点**，工具.子命令级如 hccn_tool.bandwidth）；`Evidence`/OCR 边与 EQUIVALENT_TO 待业务环境 | 手册表格错误码 + 44 个 Interface（Atlas HCCN 实测）；`graph doc pdf:<手册>` 显示 documents | 🚧 手册→ErrorCode + Interface ✅，Evidence/EQUIVALENT_TO 🔲 |
| A7 | 文档级标签 | **两级分类**（主题/领域 domain + 具体作用 purpose）：确定性提取（文件名+内部标题，词典 `config.tags.registry` 驱动）+ 审核页治理（自动排除/恢复、人工添加、词典新增/改名/改 tier 同步 config）+ **图 Tag 节点/TAGGED_WITH 边**（registry 全量节点）+ **skill 能力发现**（`tags` 目录、`context` 问题→标签匹配，agent 先发现"有哪些文档可提供知识"如 HCCL 超时→命中领域×作用）；**资产路径不进库**（asset_id + API 出口白名单） | npu-smi 命令参考 PDF 提取 [npu-smi/Atlas/命令参考] 命中 + `context "HCCL 超时"` 返回文档线索；`graph tags` 可查 | ✅ 框架（tagging/ingest/graph/review/api/skill）；词典与候选质量待业务数据标定 |

### B. 部署/运维

| # | 项 | 说明 | 状态 |
|---|---|---|---|
| B1 | 部署脚本补全 | `deploy_remote.py` 打包缺 `data/graph/`（Kùzu 图）；`print_steps` 依赖清单补 kuzu/paddleocr/解析库；**本次已顺手修复 pack_data（加 graph）** | ✅ 修复 |
| B2 | embedding/OCR 强制 API | **决策（2026）**：本地 embedding/OCR 不做（部署复杂）——embedding 用 OpenAI 兼容端点（**其他服务器 vLLM 部署**，config `embedding.base_url` 指向）；OCR 用 `ocr_provider: api`（`ocr_api_mode: custom|openai` 指向远端 OCR 服务）；`echo` 仅离线演示 | 内网端点连通实测 | ✅ 决策定案（强制 API）；连通性实测待业务环境 |
| B3 | 服务与日志 | **总日志接口**（`vllm_kb/logging_setup.py`：打屏 + 可选落盘分卷，config `logging` 段）；状态经 `/health`、`/api/stats`；**图更新流程文档化**（先停 serve_api 再 build_graph，Kùzu 单写者） | 日志落盘分卷实测（✅ 测试）；业务环境启用 `logging.file=true` | ✅ 完成（不引入 systemd/服务管理器） |
| B4 | 数据迁移 | 初始知识库（canonical/LanceDB/代码快照/图/compatibility）rsync 或 tar 迁入；后续业务数据在业务环境本地采集（见关键约定） | 🔲 |
| B5 | 依赖安装 | `requirements.txt` 分组整理（核心/服务/业务来源/OCR）；离线环境 `pip download -r requirements.txt -d wheels/` 打包内网安装 | ✅ 整理；离线打包待业务环境 |

### C. 安全/合规（需业务侧决策）

| # | 项 | 需要确认 | 状态 |
|---|---|---|---|
| C1 | 脱敏边界 | 业务数据在**业务环境内**明文存储是否可接受（案例/工程师记录/截图含敏感信息）？出业务环境必须脱敏——已约定采集验证在业务环境进行，基本消除外泄面，但需书面确认 | 🔲 |
| C2 | 审核权限 | 谁能标注"专家认证/已测试有效"？review_ui 权限方案：nginx basic auth（V1）或内置登录+角色 | 🔲 |
| C3 | 密钥注入 | GITHUB_TOKEN / EMBEDDING_API_KEY 在业务环境的注入方式（环境变量/密钥管理服务，不入 config.json） | 🔲 |

### D. 验证/标定

| # | 项 | 说明 | 状态 |
|---|---|---|---|
| D1 | OCR 阈值标定 | M2 用 10 张真实截图标定 B/C 档阈值（当前 0.85 为草案值） | 🔲 |
| D2 | 置信度参数标定 | Phase 5：业务真实故障案例评估集 → 标定 verification_factor 与 α/β/γ | 🔲 |
| D3 | 业务复验 | 接入真实业务数据后复验：search/signature/code/graph chain 全链路 + 复杂问题追溯 | 🔲 |

---

## 4. 关键决策点（业务侧，阻塞部分批次）

| # | 决策 | 影响 | 建议 |
|---|---|---|---|
| D0-1 | C1 脱敏边界确认 | 决定 A1/A2 部署位置与存储策略（当前约定：业务环境本地，风险最低） | 先确认 |
| D0-2 | embedding/OCR 端点（B2） | **已决策：强制 API**——embedding 用 OpenAI 兼容端点（其他服务器 vLLM 部署，`embedding.base_url`）；OCR 用 `ocr_provider: api`（`ocr_api_mode: custom|openai` 指向远端服务）；本地 embedding/OCR 不做 | ✅ 定案；端点连通性实测待业务环境 |
| D0-3 | 审核权限方案（C2） | 决定 A5 review_ui 的认证实现 | V1 用 nginx basic auth |

---

## 5. 实施顺序与依赖

```
第一批（数据能进库）：A1 + A3          —— 依赖 D0-1 确认；产出：案例/表格/wiki 可入库
第二批（质量可治理）：A5 + A4          —— 依赖 A3；产出：审核补标闭环、可信度生效
第三批（图片类）   ：A2                —— 依赖 A1 骨架；产出：截图/拍照数据入库
第四批（跨来源）   ：A6                —— 依赖 A2+A5（OCR 边与审核队列）
第五批（部署）     ：B2/B3/B4/B5       —— 可与第一批并行；依赖 D0-2/D0-3
第六批（标定）     ：D1/D2/D3          —— 依赖数据积累（业务数据入库后）
```

预计工作量：**5~9 个工作日**（不含数据迁移/网络等待与标定周期）。

---

## 6. 迁移完成定义（验收标准）

- [ ] 初始知识库迁入业务环境：`client.py health/stats` + `graph stats` 正常，图链路复验通过（D3）
- [ ] A1-A3：案例/表格/wiki 从业务环境导入并检索命中（验证状态默认规则生效）
- [ ] A5：审核工作台可用，至少 1 条补标闭环回写 canonical
- [ ] A2/A6：截图签名入库，图文互证不一致进入审核队列
- [ ] B2-B5：embedding 可用（或明确降级）、服务由管理脚本拉起、更新窗口流程可用
- [ ] D1/D2：OCR 阈值与置信度参数经业务数据标定，写入 config

---
*维护：随迁移推进更新本表状态（🔲→✅），决策点确认后归档到对应批次。*
