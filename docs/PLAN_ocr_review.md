# OCR 双模型审评设计（实施与验收）

> **状态：已实施**（2026-10，全量测试绿）。实施与设计的两处偏差：
> ① 锁定消费在缓存 miss 分支直接以锁定文本构造产物（**引擎不重跑**，比设计『load 后叠加』更省——换模型/提示词后也不重跑）；
> ② 复核一致档数据源扫描 ocr.json 的 confidence_source=cross_confirmed（API /api/ocr/auto-consistent），独立于 review_items（不入队不落数据库）。

> 2026-10 需求：review 界面 OCR 审评升级——双模型 diff 显示（**全量**，含复核一致档可展开）、
> 差异行三选（VL/PP/自编）、三档分流（复核一致/有分歧待审/人工审核）、审评锁定与孤儿数据、
> 删除语义。已确认：双模型 = **VL vs 本地 PP-OCRv5/v6**；PP 复核**全量跑**（高置信也复核）；
> 审评**锁定优先**于引擎指纹。

## 0. 术语与核心语义

| 术语 | 定义 |
|---|---|
| 主引擎（VL） | 现有 openai 模式 OCR（paddleocr-vl，温度0+极简提示词+logprobs） |
| 第二意见（PP） | 本地 PP-OCRv5/v6 det+rec（paddleocr，**可选依赖**），**对所有 OCR 图跑**（幂等缓存，只对新/未复核图推理）；**PP 不审 layout**——纯文本流对 VL 的版面取舍存在系统性"分歧误报"（四图实测：侧栏/目录取舍导致 0.52 一致度） |
| 自动（auto） | **高置信（≥阈值）**→ 现状自动通过；PP 复核结果**仅记录**（ocr.json second_opinion，供抽查/统计），**不影响归档、不产生待办**（PP 对高置信的版面误报不应淹没队列） |
| 复核一致（auto_consistent） | **原低置信/no_confidence 结果 + PP 复核一致（agree≥0.85）**——阈值边缘翻案：自动采纳主引擎，**不进待办**；UI「复核一致」档可见、可展开 diff 抽查（这是人工劳动的替代，需可追溯） |
| 有分歧（diverged） | **低置信/no_confidence + PP 复核分歧（agree<0.85）** → 进待办队列（diff 视图） |
| 人工审核（manual_review） | 低置信/anomaly **且 PP 未复核**（未装/失败）→ 现状待办（无 diff）；**结构类 anomaly**（truncated/contradictory/misplaced）即便 PP 一致也进人工（文本完整性问题 PP 弥补不了；no_confidence 是纯置信度缺失，参与翻案） |
| 锁定 | 审评保存后，OCR 文本以**图片 sha256** 为锚固化；重建时优先采用，不再重跑 |
| 孤儿数据 | `ocr_reviews` 行的 sha256 不再出现在资产层（图片被替换/删除）→ 保留不删，界面可见 |
| 删除审评项 | 删 `ocr_reviews` 行 + 清 ocr.json 的 review 字段；**图片与 ocr.json 文件一律不动** |

### 0.1 三档分流矩阵（每张 OCR 图的唯一归档）

| 置信度 | PP 复核 | 归档 | 进待办 |
|---|---|---|---|
| 高置信 | 任意（一致/分歧/未复核） | **自动**（现状；PP 结果仅记录不参与判定） | 否 |
| 低置信 | 一致 ≥0.85 | **复核一致**（翻案，自动采纳） | 否（UI 档可见可展开 diff） |
| 低置信 | 分歧 <0.85 | **有分歧** | 是（diff 待办） |
| no_confidence | 一致 ≥0.85 | **复核一致**（翻案） | 否 |
| no_confidence | 分歧 <0.85 | **有分歧** | 是 |
| 低置信/no_confidence | 未复核（PP 未装/失败） | **人工审核** | 是（现状，无 diff） |
| 结构类 anomaly（truncated 等） | 任意 | **人工审核**（不参与翻案） | 是 |

\* 阈值 = ocr_min_confidence；复核一致阈值 = 交叉一致度 0.85（可配）。

## 1. 数据模型

### 1.1 新表 `ocr_reviews`（review.db，沿用现有迁移机制）

```sql
CREATE TABLE IF NOT EXISTS ocr_reviews (
  sha256      TEXT PRIMARY KEY,   -- 图片内容寻址 = 锁定锚（文件变动→新sha→旧记录自然孤儿）
  status      TEXT NOT NULL,      -- diverged | manual | approved_auto
  final_text  TEXT NOT NULL,      -- 审评采纳的最终文本
  decision    TEXT NOT NULL,      -- vl | pp | edited
  vl_text     TEXT,               -- 主引擎输出快照（diff 左侧）
  pp_text     TEXT,               -- 第二意见输出快照（diff 右侧）
  agree       REAL,               -- 保存时的一致度（编辑后为 null）
  reviewer    TEXT NOT NULL,
  reviewed_at TEXT NOT NULL
);
```

### 1.2 ocr.json 扩展字段

```json
{
  "...现有字段": "",
  "second_opinion": {              // 阶段2 产出（只有进队列的图才有）
    "engine": "PP-OCRv6(local)", "text": "...", "agree": 0.52
  },
  "review": {                      // 审评保存后写入（阶段3）
    "final_text": "...", "decision": "pp", "reviewer": "admin",
    "reviewed_at": "2026-10-30T18:40:00"
  }
}
```

### 1.3 锁定优先级（build 消费）

`load_ocr_artifact` 命中后叠加：若 `ocr_reviews` 有该 sha 且 `review` 在 →
**final_text 覆盖 text**、`confidence_source="manual"`、`confidence=1.0`、
`anomaly=""`、`text_included` 按高置信注入。引擎指纹变化**不再触发该图重跑**
（人工锁定优先于引擎指纹；除非用户删除审评项）。

## 2. 实施步骤（5 阶段，每阶段独立可测可交付）

### 阶段 1：数据层（review.py）

- [x] `_ensure_ocr_reviews_table(conn)`（沿用 review.db 迁移模式，幂等 CREATE IF NOT EXISTS）；
- [x] `upsert_ocr_review(cfg, sha256, final_text, decision, vl_text, pp_text, agree, reviewer)`；
- [x] `get_ocr_review(cfg, sha256)` / `delete_ocr_review(cfg, sha256)`；
- [x] `list_ocr_review_orphans(cfg)`：LEFT JOIN asset_registry/现资产 sha 集合，返回 sha 不在场的行（只读）；
- [x] `load_ocr_artifact` 消费锁定（见 1.3）；
- [x] **测试**：CRUD 往返；锁定优先（final_text 覆盖 + manual 来源 + 指纹变化不重跑）；孤儿判定（改图片内容模拟 sha 变化）；删除语义（db 行删、文件在）。

### 阶段 2：第二意见生产链路（sources.py / ocr.py）

- [x] `vllm_kb/ocr.py` 新增 `ppocr_second_opinion(img) -> str | None`：懒 import paddleocr（3.x `predict` 优先，2.x `ocr` 兼容），`enable_mkldnn=False`（3.3.1 oneDNN+PIR bug 规避），`PADDLE_PDX_CACHE_HOME`/`USERPROFILE` 重定向支持；未安装/失败 → None（**不阻塞导入**）；
- [x] **全量复核**：ImageSource 与 `_ocr_artifact_for` 在写出 ocr.json 时（**无论置信度**）调用第二意见 → ocr.json 写 `second_opinion` + `agree = score_pair(pp_text, main_text)`；**幂等**：ocr.json 已带 second_opinion 的图不重跑 PP（本地推理 ~1-2s/张，700 张首轮 ~20 分钟一次性成本）；
- [x] 队列 seed 规则升级（review.py `seed_low_confidence_ocr` → 按 0.1 矩阵三档）：
  - **高置信 → 自动（现状）**：PP 结果仅记录，不参与判定、不产生待办（PP 对高置信的版面误报不进队列）；
  - **低置信 + agree≥0.85** → `status=auto_consistent`（**复核一致翻案**：不进待办，OCR 审评页「复核一致」档可见、可展开 diff）；
  - **低置信/no_confidence + agree<0.85** → `review_reason="diverged"` 待办项（extra 带 vl_text/pp_text/agree）；
  - **低置信/no_confidence + PP 未复核**（未装/失败）→ 现状人工审核（无 diff，`pp_status="skipped"`）；
  - **结构类 anomaly**（truncated/contradictory/misplaced/missing）→ 不参与翻案，照旧人工（即便 PP 一致）；
- [x] **测试**：mock paddleocr 模块 → 全量复核写出 second_opinion（高置信也写）；三档矩阵各一条（高置信分歧**不**入待办 / 低置信一致翻案不入待办但进复核一致档 / 低置信分歧入 diverged / truncated 一致仍人工）；幂等（第二次构建不重跑 PP）；PP 未装退化；PP 异常不阻塞主流程。

### 阶段 3：Review API（review_ui.py）

- [x] `GET /api/ocr/review?sha=<前16>`：返回 `{vl_text, pp_text, agree, diff}`——diff 由后端计算：`difflib.SequenceMatcher(None, vl_lines, pp_lines).get_opcodes()` → 结构化 opcodes JSON（`[{tag: equal|replace|delete|insert, vl:[i,j], pp:[i,j], vl_lines:[...], pp_lines:[...]}]`），前端纯渲染不含 diff 算法；
- [x] `POST /api/ocr/review`：body `{sha, final_text, decision, reviewer}` → upsert_ocr_review + ocr.json 写 review 字段 + 对应队列项转 approved；
- [x] `DELETE /api/ocr/review/{sha}`：删 ocr_reviews 行 + 清 ocr.json review 字段（**不碰图片/ocr.json 文件本身**）+ 队列项按新状态重判（回 pending）；
- [x] `GET /api/ocr/orphans`：孤儿列表（sha 前缀、审评人、时间、字数；**只读**）；
- [x] **测试**：API 层沿用现有 review_ui 测试模式（直接调 handler / TestClient）：保存锁定往返、删除语义、孤儿列表、diff opcodes 结构断言（equal/replace 段正确）。

### 阶段 4：前端 diff UI（review_ui.py `_HTML`）

- [x] **OCR 审评页三档分组**（tab 或分节）：「有分歧（待审核）」「人工审核」为待办队列；「复核一致」独立列表（可搜索、默认折叠，**每条可展开 diff**——全量 diff 能力，只是不进待办）；
- [x] `renderDiff(opcodes)`——纯 JS（无外部库）行级渲染：`replace/delete` 段 = 红底行（VL 独有）+ 绿底行（PP 独有）上下排列（类 GitHub unified，`-`/`+` 前缀）；`equal` 段折叠显示上下文 3 行；
- [x] **每个差异块三选**：单选组 [采用 VL] [采用 PP] [手动编辑]——编辑选中时该块切换为 `<textarea>`（预填当前选择侧文本）；未动的差异块默认"采用 VL"；
- [x] 组装 final_text：按 opcodes 逐块应用选择 → 预览 → `POST /api/ocr/review`（decision 自动判断：全 VL=vl / 全 PP=pp / 任一编辑=edited）；
- [x] 删除按钮（每条已审评项 + 队列项）：确认弹窗注明"仅删除审评数据，不动文件"；孤儿数据独立小节（只读列表）；
- [x] **测试**：沿用 JS 护栏（`new Function` 语法 + 无 `\\${` 转义扫描）+ 关键交互静态断言（renderDiff/保存/删除函数存在、三选单选组、三档分组结构）。

### 阶段 5：生命周期闭环 + 文档

- [x] 锁定与重建联动验证（阶段 1 代码的端到端）：审评→重建→不重跑+正文注入审评文本；
- [x] 图片替换 → 新 sha → 正常 OCR 流程 → 旧审评进孤儿列表（不删）；
- [x] USAGE：三档分流、diff 操作、锁定/孤儿/删除语义、PP 可选依赖与缓存重定向；
- [x] 全量回归门（unittest discover tests）。

## 3. 验收清单（业务环境人工执行）

1. **三档分流**：构建后 OCR 审评页——「复核一致」仅含**原低置信/no_confidence 但 PP 认可**的翻案结果（高置信图不在此档）；「有分歧（待审核）」仅含低置信/no_confidence 且 PP 分歧；「人工审核」含结构异常与 PP 未复核项；高置信图不出现在任何待办（PP 对高置信的版面误报被矩阵吸收）；
2. **diff 正确性**：任一档展开 diff——红/绿/上下文段落与两侧文本逐行对应；无外部依赖、页面无 JS 报错（控制台干净）；
3. **三选操作**：选 VL 保存 → final_text=VL 全文；选 PP → final_text=PP 全文；编辑一行 → decision=edited 且 final_text 含编辑内容；
4. **锁定生效**：审评保存后重建 build_kb——日志确认该图未重跑 OCR（跳过计数），文档 evidence/正文使用审评文本（confidence_source=manual）；**换提示词/模型后仍不重跑**（锁定优先），删除审评项后恢复重跑；
5. **孤儿语义**：把某张已审评图片替换为不同内容 → 重建 → 新图正常 OCR 进三档流程，旧审评条目出现在「孤儿数据」（未被删除、未被误关联到新图）；
6. **删除语义**：删除该审评项 → db 行消失、ocr.json 的 review 字段清除、**图片文件与 ocr.json 文件仍在磁盘**；该项按当前状态重新归档（如仍分歧则回待办）；
7. **复核成本**：首轮全量 PP 复核一次性完成（700 张约 20 分钟量级）；第二轮构建 PP 不重跑（second_opinion 缓存幂等）；
8. **退化路径**：临时卸载/改名 paddleocr 包 → 构建 → 低置信图照常进「人工审核」栏（无 diff 数据，界面不崩），复核一致档带"未复核"标注；
9. **回归**：unittest discover tests 全绿；现有低置信 OCR 队列、图文互证等其他功能不受影响。

## 4. 边界与风险

- **PP 不审 layout（四图实证）**：PP 是纯 det+rec 全量文本流，对 VL 的版面取舍（侧栏/目录略读）会系统性报"分歧"（issue 页 0.522 / doc 页 0.545，内容实际重叠）——因此**高置信结果不吃 PP 判定**（仅记录），复核一致翻案只作用于低置信结果（PP 认可 = 文本层面可靠，阈值边缘误伤被纠正）；DIVERGE 样本人工比对时先区分"版面差异"与"真识别错误"；

- **PP 引擎是可选依赖**：业务环境需 `pip install paddleocr paddlepaddle`（+VC 运行库）；未装 → 复核一致/有分歧两档不存在（退化为现状：低置信直进人工），功能不残废；
- **PP 全量跑的成本**：首轮 ~20 分钟（700 张，本地 CPU 1-2s/张）；幂等缓存后增量仅新图；与 build 串行——量大时可后置独立脚本跑（v1 随 build）；
- **锁定 vs 指纹**：人工锁定优先于引擎指纹——用户改提示词/模型后，已审评图**不自动重跑**（预期行为，删除审评项即可重审）；USAGE 必须写明；
- **diff 行对齐质量**：两引擎版面顺序差异大时 diff 块会碎片化——后端先按"行集相似度"决定 unified 还是 side-by-side 阈值？**v1 只做 unified**（碎片化可接受，人工核对本意）；
- **并发写**：review UI 单用户场景（现状），不做乐观锁。
