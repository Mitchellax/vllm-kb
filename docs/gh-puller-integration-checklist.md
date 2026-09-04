# gh-puller 对接清单（vllm-kb 侧）

vllm-kb 已接入 gh-puller 代码图谱检索（MCP Streamable HTTP 单端点）。本文件记录对接结论与参数范例，供 gh-puller 侧适配参考。

## 传输与协议（已对接）

- **传输**：MCP Streamable HTTP，stateless 单端点 `POST {base_url}{path}`
  - `base_url`：gh-puller HTTP 服务地址（如 `http://localhost:8787`），经审核工作台配置
  - `path`：端点路径（默认 `/gh-puller/graph`），显式配置
- **协议**：`tools/call` JSON-RPC，`params.name` + `params.arguments`，免 `initialize` 握手
- **响应**：纯 JSON，`result.structuredContent`（优先取）/ `result.content[0].text`（回退）；`isError=true` 视为工具级错误

## 对接结论

| # | 议题 | 结论 |
|---|---|---|
| 1 | project 标识 | **gh-puller server 端承诺适配客户端传入的任意参数**。vllm-kb 侧约定 `project` 传 `vllm-project/vllm-ascend` 形态（见下方范例），gh-puller 侧按此适配 |
| 2 | 预索引触发 | gh-puller server 端适配（客户端传入参数即处理，含首次未索引场景的兜底） |
| 3 | 版本维度 | gh-puller server 端适配（客户端传入版本参数即处理，见下方范例的 `version` 字段） |
| 4 | 鉴权 | **暂不加**（真实业务环境运行） |
| 5 | 端点路径 | **显式配置**：`base_url` + `path` 经审核工作台与 embedding/OCR/GitHub 同级管理（含连通性测试） |

## 工具面总览（6 工具 + 1 协议探测，全部为 vllm-kb 服务能力所必须）

gh-puller 侧需实现下表全部 MCP 面（vllm-kb 的 skill 命令 / REST 端点逐工具绑定，缺任一即对应命令不可用）：

| # | MCP 工具 | vllm-kb 端点 | skill 命令 | 业务环境验证 |
|---|---|---|---|---|
| 1 | `search_graph` | POST /code-graph/search | `code-graph search` | ✅ 已验证 |
| 2 | `search_code` | POST /code-graph/code-search | `code-graph code-search` | ⬜ 待验证 |
| 3 | `trace_path` | POST /code-graph/trace | `code-graph trace` | ✅ 已验证 |
| 4 | `query_graph` | POST /code-graph/query | `code-graph query` | ✅ 已验证 |
| 5 | `get_architecture` | POST /code-graph/architecture | `code-graph architecture` | ⬜ 待验证 |
| 6 | `detect_changes` | POST /code-graph/changes | `code-graph changes` | ✅ 已验证 |
| — | `tools/list`（协议方法） | GET /code-graph/health；审核工作台连通性测试 | `code-graph health` | ⬜ 待验证 |

> `tools/list` 不是业务工具，是**连通性探测的协议依赖**（见下方"连通性探测"）——
> `/code-graph/health` 与审核工作台的"测试连通"都发它，不通则健康检查报 unreachable。

## 参数范例（vllm-kb 侧约定，gh-puller 侧适配）

### search_graph（搜函数/类/路由）
```json
{"jsonrpc":"2.0","id":1,"method":"tools/call",
 "params":{"name":"search_graph","arguments":{
   "project":"vllm-project/vllm-ascend",
   "query":"update settings",
   "limit":10,"offset":0
 }}}
```

### search_code（grep + 图增强）—— ⬜ 待业务环境验证
```json
{"jsonrpc":"2.0","id":5,"method":"tools/call",
 "params":{"name":"search_code","arguments":{
   "project":"vllm-project/vllm-ascend",
   "pattern":"DispatchFFNCombine",
   "mode":"full",
   "path_filter":"^vllm_ascend",
   "limit":10
 }}}
```

`pattern` 必填（grep 文本）；`mode`: compact|full|files（默认 compact）；`path_filter` 可选（结果文件路径正则）。语义：grep 命中去重到函数粒度、按结构重要性排序。

### trace_path（调用链/数据流追踪）
```json
{"jsonrpc":"2.0","id":2,"method":"tools/call",
 "params":{"name":"trace_path","arguments":{
   "project":"vllm-project/vllm-ascend",
   "function_name":"do_auth",
   "direction":"outbound","depth":3,"mode":"calls","limit":100
 }}}
```

`function_name` **双形态约定**（gh-puller 侧已实现，CBM ≥0.10.8）：

| 形态 | 例 | 说明 |
|---|---|---|
| 短名 | `do_auth` | bare name 列匹配（上游先查此列） |
| 完整 qn | `vllm-kb-vllm-0.23.0.tests.utils.do_auth` | **精确标识**：bare name 未命中时按 `project + 完整 qn` 精确回退 |

- `<project>.` 前缀（即 `snapshot.index_name`）是 CBM 规范 qn 的**一部分**——剥前缀产生
  module-relative 名，CBM **不解析**（实测三态：完整 qn ✓ / 剥前缀 qn ✗ / 短名 ✓，见
  gh-puller 侧调查 `apps/vllm-kb-adapter/archive/2026-09-04-qn-investigation.md`）。
  此前"完整 qn 上游不识别"的实测结论是误诊——失败案例均为非函数节点（模块/属性），
  与 qn 形态无关。
- **vllm-kb 侧归一化（已实现）**：**原样透传**——dotted 输入（完整 qn / `module.func`）
  不改写不预检（qn 自身已精确，且保 cursor 翻页参数一致）；裸短名先经 search_graph
  唯一性预检——多命中 → `{"status":"ambiguous","candidates":[...]}` 候选（200，候选含
  qn，agent 拿完整 qn 精确重试）；唯一/零命中/预检不可达 → 原样透传。
  翻页（`cursor`）跳过预检。
- **零命中错误增强（vllm-kb 侧已实现）**：上游 function not found → 400 可读错误——
  补一次轻量预检取节点 label，含 property/field/attribute 时明确指出属性节点 + 引导
  改 trace 宿主类方法（属性节点上游暂不支持直接追踪，见待适配项）。

### query_graph（Cypher 查询）
```json
{"jsonrpc":"2.0","id":4,"method":"tools/call",
 "params":{"name":"query_graph","arguments":{
   "project":"vllm-project/vllm-ascend",
   "query":"MATCH (n:Function)-[:CALLS]->(m) RETURN n.name, count(m) LIMIT 10"
 }}}
```

### get_architecture（架构总览）—— ⬜ 待业务环境验证
```json
{"jsonrpc":"2.0","id":6,"method":"tools/call",
 "params":{"name":"get_architecture","arguments":{
   "project":"vllm-project/vllm-ascend",
   "aspects":["clusters","boundaries","hotspots"],
   "path":"vllm_ascend/"
 }}}
```

`aspects` 可选（list[str]，支持 all/overview/structure/dependencies/routes/clusters/boundaries/hotspots/hierarchy，不传=全量）；`path` 可选（目录前缀限定范围）。

### detect_changes（变更影响面）—— 含版本维度
```json
{"jsonrpc":"2.0","id":3,"method":"tools/call",
 "params":{"name":"detect_changes","arguments":{
   "project":"vllm-project/vllm-ascend",
   "version":"v0.23.0rc1",
   "diff":"diff --git a/x b/x\n+new",
   "scope":"impact","direction":"inbound","depth":2,"limit":20
 }}}
```

### tools/list（连通性探测，协议方法）—— ⬜ 待业务环境验证
```json
{"jsonrpc":"2.0","id":0,"method":"tools/list","params":{}}
```

判定规则（vllm-kb 侧实现）：**HTTP 2xx + JSON 响应体含 `result` 键**即判可达——同样免
`initialize` 握手（stateless 直发）。响应内容不作校验（标准 MCP 形态 `result.tools` 数组
建议返回，便于人工核对工具清单，但 vllm-kb 不依赖其内容）。

> **注**：`version` 字段（detect_changes 等）**gh-puller 侧已实现路由**——adapter 快照
> 注册表按 PEP 440 语义解析 `version` → 版本化索引（显式 version 必须命中快照，可带前缀
> `v`；缺省取语义版本最高）。vllm-kb 侧按原约定传 version 即可。

## 错误语义（vllm-kb 侧已实现，gh-puller 侧无需改动）

| 情形 | vllm-kb 返回 | 语义 |
|---|---|---|
| gh-puller 不可达（连接失败/超时/非 2xx/非 JSON） | **503** + 引导用 `code` 命令 | 服务故障，触发熔断计数 |
| 工具级错误（响应 `isError:true`：未知函数/参数错） | **400** + 上游错误详情 | 服务健康，参数问题——agent 应换函数名形态/参数重试而非放弃 |

gh-puller 侧只需照常返回 MCP 标准信封（`isError` 字段区分工具级错误），vllm-kb 侧据此分流。

## gh-puller 侧适配状态

### 已实现（vllm-kb 侧已核实，2026-09-04 复核）

1. **工具面 6 工具全量放行**（commit `6bee2f2`）：adapter `CHECKLIST_TOOLS` 扩至
   search_code/get_architecture（原 4 工具白名单会拒两者为 `unknown tool`）；
   新增 `_JSON_FORMAT_TOOLS` 结构化格式转发 + 归一化（search_code 的 raw_matches
   表格、get_architecture 各 aspect 表格 → rows）。
2. **trace_path 完整 qn 精确匹配**（CBM ≥0.10.8 原生能力，adapter 原样透传即正确）：
   bare name 未命中时按 `project + 完整 qn` 精确回退。`<project>.` 前缀是 CBM 规范
   qn 的一部分，**勿剥**——剥前缀的 module-relative 名不可解析（实测三态见
   `apps/vllm-kb-adapter/archive/2026-09-04-qn-investigation.md`）。集成不变量：
   `trace_path.project == snapshot.index_name` 且 `function_name` 以
   `snapshot.index_name + "."` 开头时精确命中；完整 qn 被拒 → 查 CBM 版本 /
   project 路由 / 快照版本匹配（环境错配，非形态问题）。
3. **version 字段路由**：adapter 快照注册表 PEP 440 语义解析（显式 version 命中快照，
   缺省取语义版本最高）。

### 待适配（仍开放）

1. **trace_path 节点覆盖缺口——属性/descriptor 节点不可追踪**（实测，主要修复项）：
   search_graph 合法枚举的节点传入 trace_path 报 function not found。实测案例：
   `vllm-kb-vllm-0.23.0.vllm.config.model.ModelConfig.registry`（label=Method/property 类节点）
   ——完整 qn、短名均零命中（上游仅匹配函数/方法节点）。注：原"qn 尾段解析（末两段）"
   子项已被"完整 qn 直接可用"取代（见已实现 #2），勿再实现末两段 partial 形态。需：
   - **属性节点解析到宿主类**：property/descriptor/field 节点无法直接追踪时，
     解析到宿主类（如 `ModelConfig.registry` → `ModelConfig`）以其为锚追踪调用方
     （属性访问即宿主类使用），而非直接拒绝；
   - **search↔trace 闭环契约**：search_graph 枚举的每个节点，以其 qn 传 trace_path
     必须可消费——枚举即承诺可追踪；不可追踪的节点类型应在 search 结果中显式标注
     （vllm-kb 侧据此提前引导，而非等 trace 报错）。

## vllm-kb 侧配置（审核工作台管理）

config.json 的 `code_graph` 段：
```json
{
  "code_graph": {
    "enabled": false,
    "base_url": "",
    "path": "/gh-puller/graph",
    "timeout_seconds": 30,
    "max_retries": 1,
    "repo_project_map": {
      "vllm-ascend": "vllm-project/vllm-ascend",
      "vllm": "vllm-project/vllm"
    }
  }
}
```

- `enabled=false`（默认）：`serve_api` 不注册 `/code-graph/*` 端点（404 比 503 干净）
- `enabled=true` + `base_url`：注册端点；gh-puller 不可达时端点 503 + 引导用 `code` 命令查本地索引（**不走回退**——本地无等价图谱能力）
- 连通性测试：审核工作台 → API 配置中心 → code_graph → 测试连通（调 `tools/list` 探测可达性，判定规则见上方"tools/list（连通性探测）"）
