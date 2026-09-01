# gh-puller 对接确认清单

vllm-kb 已接入 gh-puller 代码图谱检索（MCP Streamable HTTP 单端点）。以下开放点需 gh-puller 侧确认，确认后只改 vllm-kb 的 `config.code_graph` 默认值即可，调用方不动。

## 已对接（基于当前 gh-puller 实现）

- **传输**：MCP Streamable HTTP，stateless 单端点 `POST {base_url}{path}`（path 默认 `/gh-puller/graph`），免 `initialize` 握手，纯 JSON 响应
- **协议**：`tools/call` JSON-RPC，`params.name` + `params.arguments`
- **响应**：`result.structuredContent`（优先取）/ `result.content[0].text`（回退）；`isError=true` 视为工具级错误
- **工具**：search_graph / search_code / trace_path / query_graph / get_architecture / detect_changes
- **project 映射**：vllm-kb 内部 `vllm-ascend`/`vllm` → config `repo_project_map` → gh-puller `project` 参数（默认 `vllm-project/vllm-ascend`）

## 待 gh-puller 确认

### 1. project 标识
gh-puller 的 `project` 参数是 `index_repository` 时的 repo_url（如 `github.com/vllm-project/vllm-ascend`），还是自定义短名？vllm-kb 默认映射成 `vllm-project/vllm-ascend` 形态，是否可直接作为 `project` 传入？

### 2. 预索引触发
vllm-kb 首次查某仓时，gh-puller 是否已索引？未索引时（`RepoNotIndexedError` 或 425）vllm-kb 该：
- (a) 自动触发 `index_repository` 后重试？还是
- (b) 返回提示让用户手动 `index_repository`？

（当前 vllm-kb 侧不可达即 503 + 引导用本地 `code` 命令，未自动触发索引）

### 3. 版本维度
gh-puller 图谱是当前主干态。vllm-kb 需要"v0.23.0rc1 版本的调用链"时：
- 能否在 `project` 或 `arguments` 里限定到 tag/commit？
- 还是 vllm-kb 把版本信息拼进 `query` 文本由 LLM/图谱自行理解？

（开发者已说 version 查询后继实现，确认优先级即可）

### 4. 鉴权
当前 `tools/call` 无鉴权。若后续加 token，vllm-kb 该用什么 header 传？（已预留 `config.code_graph` 可加 `token`/`token_env` 字段）

### 5. 端点路径与端口
默认 `path=/gh-puller/graph`、端口 8787 是否会是稳定默认值？还是建议 vllm-kb 部署时显式配 `path`？
