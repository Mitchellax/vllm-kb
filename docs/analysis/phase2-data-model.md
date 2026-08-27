# Phase 2 数据布局：多来源统一存储与质量分级设计

> 状态：设计草案（V1），作为 Phase 2 数据工程的起点。
> 关联：[总体建设方案](vllm-knowledge-base-plan.md)（Phase 2 图+向量）、[验收报告](vllm-kb-acceptance-report.md)。
>
> 背景：知识库后期将不只包含 GitHub 仓库记录，还会接入：
> 华为对外开放的问题定位案例（PDF/Word/网页）、工程师解决问题的中间记录、
> 从其他位置拉取的 markdown wiki、已知问题登记表格（Excel）等来源。
> 其中 PDF 只能以 pdf 格式下载（如 npu hccn_tool 接口指南）；案例是 Word 或网页，
> 通常只审核技术链路真实性、不逐字校对文本；大量日志以截图/拍照形式呈现且图片质量较差。

---

## 1. 目标与边界

**目标**：定义一套"任意来源、任意格式、任意质量"都能统一入库的存储布局，
并让**质量**成为一等元数据（而不是事后补救），使低质量数据不污染检索、又不丢失任何原始信息。

**边界（V1 不做）**：

- 不做图片向量/多模态检索（成本高、效果差、非刚需）；图片靠"OCR 签名 + 原图可回看"满足排查需求；
- 不做云端 OCR 为主力（本地 PaddleOCR 优先，云端仅兜底）；
- 不把 PDF/Word 全文塞进向量库做语义检索（以结构化抽取 + 签名为主，正文仍入库但分级见 §6）。

## 2. 设计原则（不变式）

1. **原始资产不可变**：原始文件（PDF/Word/HTML/截图）只增不改，存 `data/assets/`，带 sha256 校验；
2. **解析可重跑**：解析/OCR 产物（Markdown/JSON）存 `data/parsed/`，换解析器/OCR 模型后重新生成，不碰资产；
3. **canonical 仍是唯一事实源**：新来源 = 新 adapter 产出 `KbDocument`，现有"采集→规范化→入库"流水线结构不动；
4. **质量是元数据不是开关**：质量信息随 canonical 一起入库（`extra` 承载），索引/置信度/图在查询时按质量处理；
5. **确定性优先**：Tier 0/1（结构化映射/规则抽取）零 LLM 成本且可重放；LLM（Tier 2）只用于工程师自由文本，结果必须落盘为可重放产物。

## 3. 总体布局

```
data/
├── assets/                    # ① 原始资产层（不可变，sha256，永不修改）
│   ├── pdf/                   #   华为接口指南等（npu_hccn_tool_guide.pdf）
│   ├── docx/                  #   对外案例 Word
│   ├── html/                  #   网页案例（含抓取时间）
│   └── images/                #   日志截图/拍照原图（issue10700_log.png）
├── parsed/                    # ② 解析产物层（可重跑，按资产一一对应）
│   ├── pdf/<asset>.md         #   文字层→Markdown（含表格占位引用）
│   ├── pdf/<asset>.tables.json  # 结构化表格（接口参数表/错误码表）
│   ├── docx/<asset>.md        #   Word→Markdown（保留标题层级）
│   ├── html/<asset>.md        #   网页正文→Markdown（保留 URL/抓取时间）
│   └── images/<asset>.ocr.json  # OCR 结果（文本+置信度+签名命中）
├── raw/                       # ③ 原始数据层（现状不变）
│   ├── github/ vllm-ascend/   #   GitHub 原始 JSON（事实源，可重放）
│   └── canonical.jsonl        #   统一 canonical（66k+ 条，新来源并入）
├── lancedb/ kb.sqlite3        # ④ 索引层（向量 + FTS5，按质量分级进入，见 §6）
├── code/                      # ⑤ 版本化代码仓（现状不变）
└── graph/                     # ⑥ Phase 2：Kùzu 图（节点/边见 §9）
```

**与现有布局的兼容**：`raw/`、`lancedb/`、`code/` 保持现状；新增 `assets/`、`parsed/`、`graph/` 三个目录；
canonical 仍统一单文件，仅扩展字段（§7）。

## 4. 各格式处理方案

| 格式 | 处理 | 工具 | 关键点 |
|---|---|---|---|
| 华为 PDF（接口指南类） | 文字层 → Markdown + **表格转结构化 JSON** | PyMuPDF / pdfplumber | 接口指南的价值大头在**参数表/错误码表/版本对应表**——转 `tables.json` 直接进图（`Interface`/`ErrorCode` 节点），比正文更值钱 |
| PDF 扫描件 | 整页 OCR | PaddleOCR（本地，中文强） | 少见；接口指南一般有文字层，扫描件才走 OCR |
| Word 案例 | 正文 → Markdown + 表格 | python-docx / pandoc | 保留标题层级（案例模板：现象/根因/修复版本/规避方案 → 段落级元数据） |
| 网页案例 | 正文抽取 → Markdown | trafilatura / readability | 保留 URL + 抓取时间；正文引用的截图单独下载进 assets |
| markdown wiki | 直接读取 + 标题结构切分 | 现有 `MarkdownSource` | 接口已预留（Phase 4 实现） |
| Excel 登记表 | 行 → 记录，列 → 字段 | 现有 `ExcelSource` | 接口已预留；列"版本/修复PR/URL"可与 GitHub 节点 join |
| 日志截图/拍照 | 签名导向 OCR（§5） | PaddleOCR + 预处理 | 不追求全文，只追求**错误签名可达** |

## 5. 图片/截图专项策略（签名导向 OCR）

日志截图的价值不在整段文字，而在**错误码 / 算子名 / 报错短语**，因此：

1. **面向签名而非全文**：OCR 后只跑签名提取（复用现有 `signature.py`/`error_parse.py` 三层引擎），
   产出 `{ocr_confidence, 命中的签名列表}`。OCR 对 `drvRetCode=6`、`aclnnXXX failed`、`kernel_name=...`
   这类**短串+数字+固定格式**的识别率远高于长段自然语言——签名导向天然适配低质量图片。
2. **图像预处理管线**：裁剪日志区域 → 灰度/二值化 → 对比度增强 → 等宽字体行分割 → 逐行 OCR。
   日志是固定格式文本，行分割后识别率显著提升。
3. **图文互证**：案例正文（Word/网页文字）提到的错误码 ↔ 截图 OCR 出来的错误码做交叉一致性检查：
   - 一致 → 两者置信度都上调（OCR 被正文印证，正文事实被截图佐证）——对"技术链路已审核、文本不逐字校对"的华为案例尤其有效；
   - 不一致 → 标记"待人工确认"，不自动入图。
4. **原图永远保留**：OCR 只是派生层，任何时候可回看原始截图。

## 6. 质量分级模型（两个正交维度）

质量不是一维的。V1 采用**两个独立维度**，避免"图片质量差 = 内容不可信"的错误耦合：

### 维度 A：信号质量（数据本身的可读性）→ 决定**索引去向**

| 档位 | 判定 | 去向 |
|---|---|---|
| A：原始文字层 | 无 OCR（PDF 文字版 / Word / HTML 正文） | 向量库 + FTS5 + 图 |
| B：高质量 OCR | OCR 置信度 ≥ 阈值（默认 0.85，可配）且行结构完整 | FTS5 + 图；不进向量库（避免污染语义检索） |
| C：低质量 OCR | OCR 置信度 < 阈值（拍照图为主） | 只提取**签名**进符号表/图（`MENTIONS` 低权重边）；原图存 assets |

### 维度 B：内容可信度（结论的验证状态）→ 决定**置信度权重**

| 档位 | 含义 | 建议权重（草案，Phase 5 标定） |
|---|---|---|
| 未验证 | 来源记录未经核对（如 wiki 转载、网友讨论） | 0.5 |
| 已测试有效 | 技术链路经实测验证（华为对外案例"技术链路已审核"落此档；工程师自测记录） | 0.85 |
| 专家认证 | 经专家/权威渠道确认（高级工程师复核、官方发布） | 0.95 |

> 与现有 `reliability` 的关系：现有 `w_rel` 按 status/source_type 规则计算（merged_fix 0.9 / wiki 0.7 ...），
> 是**来源形态**维度；维度 B 是**验证状态**维度。**已实现（A4）**：verification 作为
> `w_rel` 的**下限提升**（`w_rel = max(规则值, verification_factor)`，expert 0.95 / tested 0.85 /
> unverified 0.5，可配置）——官方手册 status=open→0.4 提升到 0.95；已实现"向量路径 meta 补全
> verification"。具体融合公式（乘法/加权）留待 Phase 5 用真实故障案例标定。

#### 各来源验证状态默认规则（先入库，后续审核补标）

默认**所有来源先入库**（标记 `unverified`），按来源套用以下初始规则，人工确认统一走 §10 审核入口：

| 来源 | 初始验证状态 | 说明 |
|---|---|---|
| 官方手册（华为 PDF 接口指南类） | **专家认证**（expert） | 官方发布渠道，直接最高档，不进审核队列（解析异常除外） |
| 对外案例（Word/网页） | **从标题解析** | 标题含"待审核"→ `unverified`（入审核队列）；"审核通过"→ `tested`；"待修改"→ `unverified` + `flagged: 待修改`（入审核队列）；无标记 → 保守 `unverified` |
| 已知问题登记表（Excel） | **低优先级**，按**未解决 issue** 处理 | 本质是 issue 的补充：初始 reliability 按 open issue（≈0.4）计，`status=open` 语义，不主动提权；有"修复PR/URL"列时可与 GitHub 节点 join 后按 merge 状态修正 |
| markdown wiki | `unverified`，**后续审核补标** | 质量参差不齐，全部先入审核队列（verification_pending），审核通过后升级为 tested/expert |
| 工程师中间记录 | `unverified` | 自由文本，Tier 2 抽取 + 入审核队列；专家复核后升级 |
| 截图/拍照（Evidence） | 随所属文档 | 本身不单独评级；图文互证不一致 → 入审核队列（ocr_mismatch） |

> 设计决策：**先入库、后补标**——避免"等审核完才能用"的阻塞；未验证内容以低权重参与检索，
> 审核升级后权重自动提升（重跑 `--recanonicalize` 或审核回写即生效）。

### 组合矩阵（两个维度正交生效）

```
                信号质量 A            B              C
内容可信度 B
未验证          A×未验证=正常入库    B×未验证=FTS仅   C×未验证=仅签名
已测试有效      A×有效=高权重入库    B×有效=中权重    C×有效=签名+低权重边
专家认证        A×认证=最高权重      B×认证=较高权重  C×认证=签名+中权重边（见下）
```

### 图片降权的矛盾与 V1 决策（重要 trade-off）

**矛盾**：通常只能拍屏记录的业务保密等级较高，能接触这类环境的专家业务能力往往更强，
其记录（即便图片质量差）可能比普通可复制日志**价值更高**——按 OCR 质量降权会系统性低估高价值来源。

**V1 决策：仍采用降权方案**（C 档只取签名），理由：

- 图片质量差 → 签名召回率下降是**客观事实**，C 档不进向量库是防污染底线；
- "专家能力更强"无法自动度量，不能作为确定性规则提权；
- 实现简单、行为确定、可重放。

**升级路径（记录在案，后续版本评估）**：

1. **专家认证提权**：当来源带"专家认证"标记（维度 B 最高档）时，其 OCR 证据权重上调——
   "认证"与"图片质量"正交，认证本身就是独立信号（上表 C×专家认证 = 签名+中权重边）；
2. **图像增强**：超分/去噪提升 OCR 召回率（成本可控时启用）；
3. **多模态检索**：Phase 5+ 若评估集显示图片类来源高频命中，再引入图片向量（V1 明确不做）。

## 6.5 文档级标签（两级分类）—— 让"有哪些文档可提供知识"可被发现

**背景**：故障定位时 agent 需要知道知识库有哪些文档类别（如 HCCL 命令参考/超时排查指南），
而不是搜不到 issue 就直接查代码下判断。文档级标签提供**能力目录**（agent 侧 `tags`/`context` 命令）。

**两级分类（标签固有属性，词典全局唯一）**：
- 主题/领域类（domain）：HCCL、网络、NPU、CANN… = 这是什么领域的知识（过滤/圈定范围）；
- 具体作用类（purpose）：超时排查、命令参考、错误码表… = 文档能帮我做什么（能力/动作匹配）；
- 检索语义：domain 过滤 × purpose 匹配，交集文档最相关（`context` 输出即此）。

**数据流**：
1. **自动提取（确定性、零 LLM）**：入库时从文件名 + 内部标题（PDF 编号标题 / Markdown `#` 标题），
   词典 `config.tags.registry` 子串命中为准（未收录强候选 → `extra.tag_candidates` 进审核队列）；
2. **人工覆盖层**：`kb.sqlite3.doc_tags`（auto_snapshot / excluded 可恢复 / manual），
   **最终标签 = (auto − excluded) ∪ manual**（`tagging.merge_final` 单点实现，ingest 与建图共用）；
   审核页修改后 `docs.tags` 立即同步（检索侧即时生效），图侧重建时同公式一致；
3. **图**：Tag(id, tier) 节点 + TAGGED_WITH(Doc/Issue/PR → Tag) 边；registry 全量标签也建节点
   （新增标签重建图即入图，Kùzu 单写者约束下不做热插）；
4. **能力发现（skill）**：`tags list`（目录）、`tags docs <标签>`（按标签检索）、
   `context <问题描述>`（问题→标签自动匹配，返回领域×作用交集文档线索）——agent 先读文档再下结论。

**安全边界**：标签体系与路径脱敏配套——文档路径不进 canonical/检索库（asset_id 标识）、
API 出口白名单清理、agent 无文件枚举指令面（可用工具仅 Bash）。

## 7. Canonical Schema 扩展

`KbDocument` 现有字段不动（`source_type`/`source_id`/`title`/`body`/`version_span`/`reliability`...），
扩展一律放 `extra` 与 `reliability` 语义内：

```jsonc
{
  "source_type": "doc_pdf | doc_word | doc_html | evidence_image | engineer_record | table_row",
  "source_id": "case:huawei:glm5-1-oom",          // 来源命名空间保证全局唯一
  "title": "...", "body": "（Markdown 化正文，图片以不透明占位 [图片] 引用，不暴露路径）",
  "reliability": 0.85,                              // 维度 B 折算后的初始可靠度（或 None 走规则）
  "tags": ["npu-smi", "命令参考"],                  // 文档级自动标签（两级分类，见 §6.5）
  "extra": {
    "asset": {"asset_id": "d4c7ead16c5b59e6", "sha256": "...", "format": "pdf", "pages": 132},
    // 安全约束：asset 只存不透明 asset_id（sha256 前缀），**不存服务器路径**；
    // 管理员侧路径仅存审核库 asset_registry（asset_id → rel_path 映射）
    "quality": {"text_source": "text_layer | ocr", "ocr_confidence": 0.0, "parsed_with": "pymupdf | paddleocr"},
    "verification": "unverified | tested | expert",  // 维度 B 原始标记（未验证/已测试有效/专家认证）
    "structure": {"sections": ["接口概览", "错误码表"], "tables": ["parsed/pdf/xxx.tables.json"]},
    "evidence": [{"path": "assets/images/xxx.png", "ocr": "parsed/images/xxx.ocr.json",
                  "ocr_confidence": 0.72, "matched_signatures": ["halMemCreate", "drvRetCode=6"]}],
    "source_url": "https://...", "captured_at": "2026-05-01T00:00:00Z"
  }
}
```

**分块**：正文（Markdown）仍走现有 `chunking.py`；图片证据的签名不进正文 chunk，
单独在入库时写入符号表/图（§9），避免 OCR 文本污染语义向量。

## 8. Adapter 设计（延续 BaseSource）

```python
class PdfSource(BaseSource):    type = "pdf"    # 华为接口指南等：pull=复制进 assets，canonicalize=解析+表格抽取
class WordSource(BaseSource):   type = "word"   # 对外案例
class HtmlSource(BaseSource):   type = "html"   # 网页案例（trafilatura 正文抽取）
class ImageSource(BaseSource):  type = "image"  # 截图/拍照：pull=复制进 assets，canonicalize=签名导向 OCR
# markdown / excel 已在 Phase 4 预留（现有 MarkdownSource / ExcelSource 骨架）
```

每个 adapter 实现 `pull()`（原始资产落 `data/raw/{source_id}/` 或直接 `data/assets/`）+ `canonicalize()`，
注册进 `_REGISTRY` 即可接入全链路，**不修改任何现有模块**（与 `sources.py` 接口约定完全一致）。

## 9. 对 Phase 2 图的影响（节点/边补充）

在上一轮图设计（内容层 `Doc` + 实体层 `Version/Model/Operator/ErrorCode/...`）之上新增：

- **`Evidence` 节点**：每张截图/拍照一张卡，属性 `{format, ocr_confidence, sha256, quality_tier}`；
- **边**：
  - `Evidence -(OCR_EXTRACTS)-> ErrorCode / Operator`（低权重，`method: ocr`）
  - `Evidence -(BELONGS_TO)-> Doc(case)`（图片归属案例）
  - OCR 签名与正文签名一致 → `CORROBORATES` 边（权重上调，图文互证的图表达）
  - 接口指南表格 → `Interface / ErrorCode / Parameter` 节点 + `DOCUMENTS` 边（指向来源页），
    可回答"这个错误码在哪个接口文档定义、取值范围是什么"
- **置信度**：单边可信度 = 来源可靠度 × 抽取方法可靠度 × 验证状态因子（§6 维度 B）；
  同一结论被多条独立来源边支持 → 三角验证提权（多来源互证成为置信度信号）。

## 10. 人工确认统一入口（审核工作台）

所有需要人工确认的位置**共用一个审核队列**，由一个轻量级 Web UI 统一处理，避免分散在
不同脚本/文件里各标各的。进入审核队列的类别：

| 类别 | 触发条件 | 标注动作 |
|---|---|---|
| `verification_pending` | 默认先入库的未验证来源（wiki、无标记案例、工程师记录） | 升级为 tested / expert，或维持 unverified |
| `case_title_flag` | 案例标题含"待审核"/"待修改" | 确认/修正验证状态 |
| `ocr_mismatch` | 图文互证不一致（正文签名 ↔ OCR 签名） | 确认正文正确 / 确认 OCR 正确 / 修正签名 |
| `low_confidence_ocr` | C 档低质量 OCR 的签名命中 | 人工核对签名是否可信 |
| `equivalence_candidate` | `EQUIVALENT_TO` 候选边（跨来源疑似同一问题） | 确认合并 / 拒绝 |
| `table_join_candidate` | 表格行引用 GitHub 编号/URL 的 join 候选 | 确认连接 / 忽略 |

**审核数据存储**：独立 `data/review.sqlite3`（不放进只读的 `kb.sqlite3`，检索 API 完全不碰）；
每条审核项：`{id, category, item_ref, payload(展示用), status(pending/confirmed/rejected/modified),
created_at, reviewed_at, reviewer, result(回写用)}`。

**轻量级 Web UI（V1 方案）**：`scripts/review_ui.py` 独立进程（FastAPI + 单页 HTML，无构建步骤），
与只读检索 API（`serve_api.py`）**分离部署、分离端口**——检索 API 维持结构只读不变，审核入口可写审核库：

```
GET  /queue?category=&status=        # 待办列表（分页、过滤）
GET  /item/{id}                      # 详情：正文/OCR 结果/签名对照/原图链接（assets 静态服务）
POST /item/{id}/review               # 提交标注：verification/签名修正/确认合并/备注
GET  /stats                          # 各类别待办量（工作台概览）
```

- 原图通过 assets 静态目录直接预览（标注人员看截图本体，不只看 OCR 文本）；
- 审核结果**回写**到 canonical（`--recanonicalize` 或按 source_id 定点更新），
  后续 `--rebuild`/重入库时生效——审核是增量数据，不破坏"canonical 可重放"性质；
- 标注动作带 reviewer + 时间戳，可审计。

## 11. 实施顺序（两条轨道，当前优先轨道 A）

> **开发优先级（2026 定版）**：业务来源（PDF/Word/截图/表格/wiki）大多含业务数据，未脱敏前无法取到
> 开发环境验证——相关适配**延后到业务环境单独开发**。当前阶段**只做轨道 A（图存储）**，
> 且图构建**只依赖 canonical**：任何来源只要产出 canonical 即可入图，业务来源的扩展字段
> （§7 asset/quality/verification/evidence）与布局（assets/parsed/review.sqlite3）已预留，轨道 B 无需改图侧。

### 轨道 A：Phase 2 图存储（开发环境，现在，最高优先）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| G1 图 schema + 确定性建图 | Kùzu 图：Issue/PR/Release/Operator/ErrorCode/Model/Version 节点 + FIXES/MERGED_IN/MENTIONS 边；从 canonical + 版本日历 + 签名/错误码提取（零 LLM） | ✅ 完成（71,389 节点 / 75,633 边：Issue 21,173 / PR 44,951 / Release 142 / 实体 5,123；FIXES 4,757 / MERGED_IN 25,750 / MENTIONS 45,126） |
| G2 图检索接口 | api.py `/graph/*` 端点 + client.py `graph` 命令 + SKILL.md/USAGE 更新 | ✅ 完成（chain / fixes / sig / doc / stats） |
| G3 复杂问题验证 | GLM-5.1 场景（#10700/#12933/#9944、PR #12885、dispatch_ffn_combine/561000） | ✅ 完成（结果与验收报告人工分析一致：#10700/#9944 无修复 PR；#12933 → PR#12885 → v0.23.0） |
| G4 单测 | tests/test_graph_build.py（抽取规则、建图、链路查询） | ✅ 完成（11 个用例，全量回归 197 通过） |

### 轨道 B：多来源适配（业务环境，后期）

| 里程碑 | 内容 | 验证点 |
|---|---|---|
| M1 资产+解析层 | `assets/`/`parsed/` 布局落地；**PdfSource / MarkdownSource 已实现**（表格→Markdown/JSON、验证状态默认规则）；WordSource/HtmlSource 待业务环境 | PDF 手册 → md + tables.json（✅）；Word 案例 → 标题层级（🔲） |
| M2 图片链路 | ImageSource + 签名导向 OCR（provider 可插拔：paddle / none 占位；md 图片自动收集重写资产路径 + evidence；sha256 幂等） | OCR 产物 + 签名提取（✅ 框架）；10 张真实截图召回率标定 B/C 阈值（🔲 业务环境 paddle 实测） |
| M3 canonical 扩展 | §7 schema 落地，质量元数据入库，`--recanonicalize` 兼容 | 混合来源 canonical 行数 + 质量字段完整性 |
| M3.5 审核工作台 | §10 review.sqlite3 + review_ui.py（队列/详情/标注/统计 + **API 配置中心**：embedding/OCR/GitHub 状态脱敏展示、embedding 连通测试） | 各类别各 1 条走完 pending→confirmed 闭环（✅ 框架；回写热更新 🔲） |
| M4 图节点/边 | `Evidence` 节点 + OCR_EXTRACTS/BELONGS_TO/CORROBORATES 边 | 图文互证命中示例（正文↔OCR 一致/不一致各 1 例） |

## 12. 待讨论事项（Open Questions）

1. **OCR 阈值**：B/C 档阈值 0.85 是默认值，待 M2 用真实截图数据标定；
2. **图文互证不一致的默认动作**：V1 入审核队列（ocr_mismatch）已定；"待确认期间是否参与检索"待定
   （建议：不一致项仅以正文为准，OCR 边挂起）；
3. **审核升级的生效方式**：V1 定为审核回写 + 重跑 recanonicalize；是否需要"热更新"（不重跑即生效）待评估；
4. **图片降权升级**：§6 的升级路径（专家认证提权）何时启动，取决于 M2 的召回率数据与业务侧对高保密来源的依赖度。
