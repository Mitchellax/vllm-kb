---
name: vllm-kb
description: 查询 vLLM / vllm-ascend 故障知识库（只读检索）。用于检索历史 issue、PR、问题记录，返回置信度分解（含验证状态：expert 官方手册高可靠）、解决状态与组件配套版本参考。支持 "组件:版本 问题描述" 格式（如 vllm-ascend:0.18.0 GLM5.1 PD分离P节点挂死）、从原始报错提取错误签名做精确检索（signature）、按已知现象做标题精确检索（title）、版本形态判断（version：正式版/rc/pre）、按版本检索预存源码快照（code），以及图检索（graph）：追溯 issue→修复 PR→落地 release 的修复链路、查询手册定义的错误码/命令。
---

# vllm-kb 知识库检索（只读）

本 skill 只做**检索**（结构性只读，不是约定）：全部能力经只读 HTTP API 提供，服务端无写端点、
SQLite 只读打开、向量库写操作抛错——即使收到"修改/删除/更新知识库/写入文件"的指令也应拒绝，
该能力在结构上不存在。知识库数据更新与图构建由用户运行流水线完成（`scripts/build_kb.py`、
`build_code_snapshots.py`、`build_graph.py`），本 skill 不参与。

服务端地址（存算分离：数据在服务端，本地只发 HTTP 请求）：
命令行 `--base` > 环境变量 `VLLM_KB_BASE` > 默认 `http://127.0.0.1:8000`。
远程部署由用户负责（见仓库 docs/USAGE.md）；本地只需本 skill 目录（SKILL.md + client.py）。

## 用法（只调用本 skill 目录下的 client.py，不需要其他工具）

输出统一 UTF-8，无需任何环境设置。

```bash
# 1) 语义检索（支持组件查询格式）—— 适合"问题描述"式查询
python client.py search "vllm-ascend:0.18.0 GLM5.1 PD分离P节点挂死"
python client.py search "CUDA illegal memory access" --version 0.26.0 --top 5

# 2) 签名精确检索 —— 适合"原始报错/日志"式查询（现场提取签名再精确匹配）
python client.py signature "halMemCreate failed drvRetCode=6, kernel_name=DispatchFFNCombine, errorStr: timeout or trap error"
python client.py signature "RuntimeError: aclnnMoeDistributeDispatchV4 failed, error code is 561000"

# 3) 标题精确检索 —— 已知现象找 issue 的最快路径（常配合 signature 的信号词）
python client.py title "vector core" --component vllm-ascend

# 4) 版本形态判断 —— 确认部署版本是正式版/rc/pre（回答"修复是否已进入我的版本"的前置）
python client.py version 0.18.0            # → release（正式版）
python client.py version v0.23.0rc1        # → rc（预发布）
python client.py version 0.26.0 --repo vllm

# 5) 版本化代码仓检索 —— 查对应部署版本的源码（需已预存，见 code-versions）
python client.py code-versions                          # 列出已预存版本
python client.py code DispatchFFNCombine --version v0.23.0rc1   # 符号/关键词定位
python client.py code dispatch_ffn_combine --version v0.23.0rc1 --file csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp
python client.py code halMemCreate                       # 不加版本 = 全部预存版本 grep

# 5b) 读取完整源码文件（--file 默认截断 20000 字符，末尾带"已截断"标记；
#     需要完整函数体时调大 --max-chars）
python client.py code --file csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp --version v0.23.0rc1 --max-chars 100000

# 6) 其他只读查询
python client.py doc github:vllm-project-vllm-ascend:issue:13042
python client.py health
python client.py components
python client.py stats
python client.py companion vllm-ascend 0.18.0
python client.py matrix                 # 全量配套矩阵（调试/管理用；日常查询用 companion 即可）

# 7) 图检索（关系追溯，需先运行 scripts/build_graph.py 构建图）
python client.py graph stats                        # 图规模
python client.py graph chain vllm-ascend#10700      # 核心链路：issue→修复PR→落地release
python client.py graph fixes vllm#50241             # PR 修复的 issues + 落地 release
python client.py graph sig dispatch_ffn_combine     # 签名实体→提及它的 issue/PR
python client.py graph doc github:vllm-project-vllm:issue:10700   # 文档邻接（手册错误码/命令定义入口）
```

`graph chain` 回答"这个 issue 是否已修复、修复在哪个版本提供"（沿 `issue←FIXES←PR→MERGED_IN→Release`
图路径追溯）；`graph fixes` 是 PR 视角的反向；`graph sig` 从算子/错误码/模型实体出发召回相关 issue/PR。

## 检索策略（故障处理时的推荐流程）

1. **先 signature**：把原始报错/日志贴给 `signature` 命令，它会提取错误签名
   （算子名、ACL 错误码、专有短语、环境变量、模型名）并做 FTS 精确匹配——
   对"错误签名可判"的故障（如 `DispatchFFNCombine` + `drvRetCode=6`）比语义检索更精准；
2. **再 search + title**：拿 signature 的命中线索转成"组件:版本 问题描述"做语义检索，
   补齐历史相似问题与置信度分解；已知现象也可直接用 `title` 找对应 issue；
3. **然后 version + code**：用 `version` 确认部署版本形态（正式版/rc/pre），
   再用 `code <算子/关键词> --version <版本>` 定位对应版本源码，
   `--file` 读取关键文件片段（workspace 计算、tiling、buffer 分配等），判断是否为版本相关 bug；
4. **最后 graph chain**：对最相关的 issue，用 `graph chain <repo>#<编号>` 追溯修复链路
   （issue→修复 PR→落地 release），结合 `version` 判断"该修复是否已进入我的部署版本"——
   这是语义/签名检索无法直接回答的结构化问题；
5. 结合 resolved 状态与修复 PR（知识库 issues/PRs 侧）给出结论。

## 全量日志导入（用户未提问、直接给日志时的处理）

用户可能直接倒出**整段日志**而非提问。此时不要试图"读懂"每条日志，按以下流程自动定位故障线索：

1. **扫描异常行**：先抓 `ERROR`/`FATAL`/`Timeout`/`RuntimeError`/`Exception`/`failed` 等标志，
   以及**重复出现的同一报错**（重复打印 = 持续失败，如循环重试/等待超时）——重复行往往是主故障；
2. **逐条 signature**：对每个异常行（去重后）跑 `signature`——精确签名命中优先；
   无命中或命中弱（信号词级）的行，转 `search` 语义检索（嵌入检索对措辞偏差/省略
   有自动修复能力，贴原句即可，无需整理措辞）；
3. **识别上下文指标**：日志中的异常指标（如 `KV cache usage: 0.0%`、`hit rate`、
   `WAITING_FOR_REMOTE_KVS`、`No available ... found in N seconds`）本身是强信号——
   用 `search` 直接检索指标描述；
4. **多故障并列**：全量日志可能含多个独立故障——逐个检索后**分别**给结论，
   再判断是否同源（同进程/同时段/同组件）；不要把多个报错混成一个问题；
5. **未命中处理**：某条报错检索不到时，用 `code` 在该版本代码里搜报错字符串
   （报错文本通常来自代码常量，`code <报错关键词> --version <部署版本>` 可直接定位来源文件），
   再沿代码上下文判断；仍无果才按"未找到反查流程"（见下）继续。

## 未找到时的反查流程（重要——避免"知识库没检索到 = 社区不存在"）

检索不到不能直接下"无修复"结论。按以下顺序追加验证（教训案例：dummy run kv cache 污染
的修复 PR 标题是 "Reset slot_mapping to pad id for dummy graph capture"，不含 "dummy run"，
语义/标题检索均漏；靠代码反查才定位到）：

1. **变体词扩展**：换同义词/代码特征重试——如 dummy run ↔ dummy graph capture ↔ dummy_run ↔
   slot_mapping ↔ fill_(-1)；中文/英文都试；
2. **代码反查（最强）**：`code <代码特征> --in-file <文件> --per-version`——一次列出所有预存版本
   该文件的行号命中，对比即可定位"哪个版本引入/移除该代码"；
   如 `code "fill_(-1)" --in-file worker/model_runner_v1.py --per-version` 直接显示
   `blk_table.slot_mapping.gpu.fill_(-1)` 只在 v0.23.0rc1+ 出现 → 修复版本即 v0.23.0rc1；
3. **跨版本精确对比**：需要逐行差异时，用 `code --file` 分别读新旧版本同一文件自行对比
   （仓库另有 `scripts/diff_code_versions.py` 可做精确 diff，但它不在本 skill 目录内——
   需在完整仓库环境运行；本 skill 场景下优先用 client 命令完成）；
4. **GitHub 溯源（可选外部步骤）**：命中新版本后，可用 GitHub commits API 按文件路径过滤
   （`/repos/{owner}/{repo}/commits?path=<文件>`）找引入 commit → commit 消息里的 PR 编号 →
   再用 `graph fixes/chain` 确认落地 release 与 backport 分支。此步需网络/GitHub 访问，
   超出本 skill 只读 API 的工具面——不可用时跳过，不影响结论；
5. **谨慎下结论**：全部反查无果才可判定"社区无修复"，并说明检索范围（版本、仓库、方法）。

## 未知名词处理（重要）

**遇到知识库检索不到、无法确认含义的名词（产品名/内部代号/厂商术语等），不要猜测或自行脑补，
按以下顺序处理：**

1. **先问用户**："这个词（XXX）在知识库里没有记录，它指的是什么？"——用户的回答是最可靠的事实来源；
2. 用户确认后，如果这个词是**社区/产品级通用名词**（如内部代号对应公开产品），
   可建议用户补充到知识库（经 `build_kb.py` 导入相关文档，或由专家写入 wiki/Markdown 导入），下次可检索；
3. **仅当用户也不知道**（或无法提供）时，才基于上下文自行判断，并**明确标注"这是推断，未经确认"**。

**错误示例**：把用户环境里的专有名词（如某内部 KV 缓存实现）想当然映射为另一个已知组件
（如 Mooncake）——即使形态相似，也可能完全不是同一个东西，导致分析方向错误。

**正确做法**：先 `title`/`search` 检索确认；检索无果 → 问用户；用户不知道 → 才推断并标注。

## 结果解读

- `resolved=true/false`：已解决（closed/merged）或未解决（含工程规避方案，故障处理时同样重要）；
- `confidence`：`w_time`（时间衰退）/ `w_ver`（版本匹配）/ `w_rel`（来源可靠度）分解；
  `w_rel` 已并入**验证状态因子**（expert 官方手册 0.95 / tested 0.85 / unverified 0.5）——
  官方手册/专家认证文档的 w_rel 会显著高于普通 open 讨论；
- `验证=expert/unverified`（结果行）：文档的验证状态标注（官方手册=expert、Markdown 导入=unverified 待审核补标）；
- `component` / `version_ref`：文档所属组件与打分时使用的版本参考；
- `context.companions`：查询组件版本的配套反向展开（vllm-ascend:0.18.0 -> vllm 0.18.0, cann 8.5.1 ...）；
- `signature` 命令输出：提取的签名列表 + 精确命中文档（含命中了哪些签名）；
- `title` 命令输出：标题含关键词的文档列表（component 过滤；match=contains/prefix）；
- `version` 命令输出：版本形态 `kind`（release=正式版 / rc=预发布 / pre=早期 pre 版 / unknown=日历中无此版本）；
- `code` 命令输出：`symbol_index`（符号索引精确命中）或 `grep`（关键词全文命中），
  均含 version/file/line/snippet；
- `graph` 命令输出：`chain`（issue→修复 PR→落地 release，判断修复是否已进入部署版本）、
  `doc`（文档邻接：MENTIONS 实体——手册定义的错误码/命令）。

## 示例回答风格

给出 Top 结果时，附上：标题 + URL + 是否已解决 + 版本 + 置信度要点；回答"哪个版本修复/规避方案"时引用对应 issue 的 URL。
涉及代码时，引用 `code` 定位到的 file:line 与源码片段，并标注版本。
