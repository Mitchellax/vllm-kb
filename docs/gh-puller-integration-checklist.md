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
| 4 | 鉴权 | **暂不加**（内网工作流运行） |
| 5 | 端点路径 | **显式配置**：`base_url` + `path` 经审核工作台与 embedding/OCR/GitHub 同级管理（含连通性测试） |

## 工具面总览（6 工具 + 1 协议探测，全部为 vllm-kb 服务能力所必须）

gh-puller 侧需实现下表全部 MCP 面（vllm-kb 的 skill 命令 / REST 端点逐工具绑定，缺任一即对应命令不可用）：

| # | MCP 工具 | vllm-kb 端点 | skill 命令 | 内网测试 |
|---|---|---|---|---|
| 1 | `search_graph` | POST /code-graph/search | `code-graph search` | ✅ 已测 |
| 2 | `search_code` | POST /code-graph/code-search | `code-graph code-search` | ⬜ 待测 |
| 3 | `trace_path` | POST /code-graph/trace | `code-graph trace` | ✅ 已测 |
| 4 | `query_graph` | POST /code-graph/query | `code-graph query` | ✅ 已测 |
| 5 | `get_architecture` | POST /code-graph/architecture | `code-graph architecture` | ⬜ 待测 |
| 6 | `detect_changes` | POST /code-graph/changes | `code-graph changes` | ✅ 已测 |
| — | `tools/list`（协议方法） | GET /code-graph/health；审核工作台连通性测试 | `code-graph health` | ⬜ 待测 |

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

### search_code（grep + 图增强）—— ⬜ 待内网测试
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

### query_graph（Cypher 查询）
```json
{"jsonrpc":"2.0","id":4,"method":"tools/call",
 "params":{"name":"query_graph","arguments":{
   "project":"vllm-project/vllm-ascend",
   "query":"MATCH (n:Function)-[:CALLS]->(m) RETURN n.name, count(m) LIMIT 10"
 }}}
```

### get_architecture（架构总览）—— ⬜ 待内网测试
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

### tools/list（连通性探测，协议方法）—— ⬜ 待内网测试
```json
{"jsonrpc":"2.0","id":0,"method":"tools/list","params":{}}
```

判定规则（vllm-kb 侧实现）：**HTTP 2xx + JSON 响应体含 `result` 键**即判可达——同样免
`initialize` 握手（stateless 直发）。响应内容不作校验（标准 MCP 形态 `result.tools` 数组
建议返回，便于人工核对工具清单，但 vllm-kb 不依赖其内容）。

> **注**：`version` 字段（detect_changes 等）是 vllm-kb 侧约定的版本限定参数范例。
> gh-puller server 端适配时，若支持版本/tag/commit 切片则按 `version` 过滤图谱范围；
> 若暂不支持则忽略该字段（按当前主干态返回），vllm-kb 侧会把版本信息拼进 `query`/`diff` 文本兜底。

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
