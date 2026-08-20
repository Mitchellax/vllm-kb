# vllm-kb 验收报告

> 场景：vllm-ascend 0.23.0rc1，PD 分离部署（P 节点 = kv_producer），
> 多模型（Qwen3.5 / GLM-5.1 / DeepSeek-V4-Flash）在业务流量接入后长期运行（~1天）P 节点崩溃。
> GLM-5.1 崩溃特征：`halMemCreate failed drvRetCode=6, out of memory`；
> `kernel_name=DispatchFFNCombine`；`Fused_MC2=1`；`errorStr: timeout or trap error`。
>
> 验收方式：仅以本地知识库（vllm-kb，向量 + FTS，66,124 docs / 122,487 chunks）为知识来源，
> 不联网（任务2 允许下载 GitHub 代码用于分析）。

---

## 任务1：社区已知 top10 相关问题 + 是否存在修复

检索策略（本次查询中调整的置信度逻辑）：
- 限定 `component=vllm-ascend`（用户问题属 vllm-ascend，排除 vllm 主仓库噪音）；
- 综合分 = 语义相似度 + **错误签名关键词加分**（halMemCreate/dispatch_ffn_combine/fused_mc2/
  GLM-5.1/运行一段时间/OOM/PD分离 等精确命中加权）——适配"专有错误签名比语义更可判"的实际；
- 多查询面（10 组）合并去重，交叉验证原始 issues/PRs 的 resolved 状态与 Fixes 引用。

### Top10 结果（按综合相关度）

| # | Issue | 相关性 | 修复情况 |
|---|---|---|---|
| 1 | [#10700](https://github.com/vllm-project/vllm-ascend/issues/10700) `GLM5.1 未添加enforce_eager true运行一段时间后崩溃` | **运行一段时间后崩溃，与用户模式一致** | 未解决，无明确修复 PR |
| 2 | [#9186](https://github.com/vllm-project/vllm-ascend/issues/9186) `GLM5.1-w4a8 dispatch_ffn_combine() bias1 Expected Optional[List[Tensor]]` | **dispatch_ffn_combine 算子直接报错** | 已解决(2026-08-10)，未见明确修复 PR |
| 3 | [#12933](https://github.com/vllm-project/vllm-ascend/issues/12933) `GLM5.1-W8A8 P2P hang and Memcache OOM with DRAM_cache path` | **GLM-5.1 + OOM + hang** | 已解决(2026-07-31)，修复 PR[#12885](https://github.com/vllm-project/vllm-ascend/pull/12885)（open） |
| 4 | [#10944](https://github.com/vllm-project/vllm-ascend/issues/10944) `GLM-5.1-W8A8 aclnnMoeDistributeDispatchV4 failed 561000` | **GLM-5.1 MoE 算子报错** | 已解决(2026-08-07)，未见明确修复 PR |
| 5 | [#9748](https://github.com/vllm-project/vllm-ascend/issues/9748) `deepseekv4 开启VLLM_ASCEND_ENABLE_FUSED_MC2报错` | **Fused_MC2 开启报错** | 已解决(2026-08-11)，未见明确修复 PR |
| 6 | [#11531](https://github.com/vllm-project/vllm-ascend/issues/11531) `GLM5.1 aclnnSparseFlashAttention算子报错` | GLM-5.1 算子崩溃 | 已解决(2026-08-02)，未见明确修复 PR |
| 7 | [#10370](https://github.com/vllm-project/vllm-ascend/issues/10370) `GLM5.1-w4a8 图模式算子通信问题，只有开启单算子才能规避` | **GLM-5.1 算子通信问题** | 已解决(2026-08-10)，未见明确修复 PR |
| 8 | [#14050](https://github.com/vllm-project/vllm-ascend/issues/14050) `GLM5.2-w4a8 A3 开FUSED_MC2=1 mega_moe weight1 must be 2-dimension` | **Fused_MC2 + mega_moe 维度报错** | 未解决，无明确修复 PR |
| 9 | [#10391](https://github.com/vllm-project/vllm-ascend/issues/10391) `glm5.1-w8a8 A2 双机混部 中英文混杂` | GLM-5.1 PD 混部输出异常 | 已解决(2026-07-10)，未见明确修复 PR |
| 10 | [#9944](https://github.com/vllm-project/vllm-ascend/issues/9944) `Qwen3.5-397B PD分离 压测并发60 D节点挂死shutdown` | **PD 分离压测挂死** | 未解决，无明确修复 PR |

### 补充（与用户场景直接相关的社区动态，来自 PR 侧检索）

- **#14358 `Fix MegaMoe prefill buffer sizing`**（v0.26.0rc，open）：MegaMoe 对称 buffer 从 decode capture size 推导，
  而 prefill batch 可能超容量 → 无效输出/崩溃——**P 节点（prefill）场景直接命中**。
- **#14304/#14305 `Keep DP token counts uniform for MegaMoe`**（open，Fixes #14273）：CANN MegaMoe 要求通信域内
  所有 rank token 数一致，PD 分离下 KV consumer 跳过 DP all-reduce 导致不一致 → 错误输出。
- **#14273 `DeepSeek V4 PD output corruption with CANN 9.1.0 (works with 9.0.1)`**（open）：PD + CANN 9.1 输出损坏。
- **#13924/#13923 `BF16 model crashes with enable_fused_mc2=1 under CANN 9.1`**（open/closed，regression from #11701）。
- **#13761 `ACLNN workspace lifetime race with async scheduling, Task Queue, and Fused MC2`**（open）：workspace 生命周期竞态。
- **#12237/#12245 `Disable multistream_overlap_shared_expert when enable_fused_mc2 is enabled`**（v0.23.0 backport）。
- **#11584 `mega moe max token default value fix`**：mega_moe_max_tokens 默认值修复。
- **#11258 `Replace hardcoded max_output_size with configurable mega_moe_max_tokens`**：参数化来源。

### 结论（任务1）

1. **存在高度相关的社区问题**，尤其 GLM-5.1 系列的算子崩溃（#10700/#9186/#12933/#10944/#11531/#10370）
   与 Fused_MC2 相关报错（#9748/#14050），模式与用户场景一致（长期运行崩溃、算子 OOM、PD 分离）。
2. **修复情况参差**：多数 issue 已 closed 但**未见明确修复 PR**（可能是配置规避）；少数有 PR 但 **open 未合并**
   （#14358/#14304 等直接相关的 mega_moe 修复都在 v0.26.0rc 分支，未 backport 到 0.23.0rc1）。
3. **0.23.0rc1 上这些问题大概率未修复**——社区修复集中在 0.24+/0.26.0rc，用户要么升级，要么按规避方案走
   （详见任务2）。

---

## 任务2：GLM-5.1 崩溃排查方向（代码分析）

> 分析对象：vllm-ascend main 分支源码。详细分析见 `vllm-ascend-analysis-task2.md`。

### 崩溃链路（代码证据）

```
VLLM_ASCEND_ENABLE_FUSED_MC2=1
  → _select_a3_moe_comm_method(): EP≤32 且未装 cann_ops_transformer → MoECommType.FUSED_MC2
  → FusedMC2CommImpl.fused_experts(): torch.ops._C_ascend.dispatch_ffn_combine(...)
  → aclnnDispatchFFNCombine → CANN kernel (kernel_name=DispatchFFNCombine)
  → halMemCreate 分配 workspace（按 mega_moe_max_tokens 最坏情况预留）
```

### 核心证据

1. **workspace 按 `mega_moe_max_tokens`（默认 131072）线性预留**（`dispatch_ffn_combine_tiling.cpp` L300-306）：
   ```
   cocWorkspace = ... + maxOutputSize*K*2B + maxOutputSize*K*1B + ...
   ```
   GLM-5.1（K≈5120）→ **单算子 workspace ≈ 2GB**。
2. **每 rank token 上限**（`ascend_forward_context.py` L35-37）：
   - mega_moe: 4096/rank；**dispatch_ffn_combine: 512/rank**；MC2: 512/rank
   - 用户 P 节点 max-num-batched-tokens=8192、tp8 → 每 rank 1024 tokens **> 512 上限**
3. **prefill buffer 容量 bug**（#14358 修复）：`enable_prefill_mc2` 默认 False → 容量按 decode capture size 推导
   → P 节点 eager prefill 大 batch 超容量。
4. **DP token 不均匀**（#14304 修复）：PD 分离下 rank 间 token 数不一致 → 错误输出/崩溃。

### 排查方向（按优先级）

| 优先级 | 动作 | 说明 |
|---|---|---|
| **A** | 调小 `mega_moe_max_tokens`（如 16384） | workspace 从 ~2GB 降到 ~256MB；官方注释明示支持 |
| **A2** | P 节点开 `enable_prefill_mc2=true` | 让 buffer 按 prefill 的 max_num_batched_tokens 推导（对应 #14358 修复） |
| **B** | 安装 `cann_ops_transformer` → 走 mega_moe | 每 rank 上限 4096（>1024），更贴合负载；需 hidden/intermediate 满足约束 |
| **C** | 关 `VLLM_ASCEND_ENABLE_FUSED_MC2=0` 对照 | 隔离变量，确认崩溃是否消失 |
| **D** | 核对 `HCCL_BUFFSIZE` | tiling 要求 ≥ (M×K×topK×3+3MB)，M=8192,K=5120 → ~1GB；不足直接报错 |
| **E** | 显存碎片化治理 | 调低 gpu-memory-utilization、P 节点 MLAPO=0、检查 Mooncake buffer |
| **F** | 收集现场 | CANN 日志确认 halMemCreate 哪个 buffer；CANN dump 确认 trap 类型 |

### 结论（任务2）

**最可能根因**：P 节点启用 Fused_MC2=1 且未装 cann_ops_transformer → 走 dispatch_ffn_combine 路径，
workspace 按默认 131072 tokens 最坏情况预留 ~2GB；长期运行显存碎片化后 `halMemCreate` 分配失败
（drvRetCode=6 OOM）；叠加业务流量每 rank token 超 512 上限 + prefill buffer 容量 bug（#14358）→
kernel 异常（timeout or trap error）。建议先 A（调小 mega_moe_max_tokens），再 B（走 mega_moe），
最后 C（关 Fused_MC2 对照验证）。

---

## 验收过程中的知识库调整

| 调整 | 位置 | 说明 |
|---|---|---|
| 批量向量写（add/delete 攒批 200 条） | `vllm_kb/vectorstore.py` `delete_docs` | LanceDB 单条写随表增长退化 30-40 倍，批量后 ETA 44h→~1h |
| 嵌入跨文档攒批（64 chunks/次） | `vllm_kb/ingest.py` `_flush_embed` | 52k 次 API 调用 → ~1.4k 次 |
| 评论走 GraphQL 内联 | `vllm_kb/github_pull.py` | REST 评论配额黑洞 → GraphQL 1 点/页 |
| 限流精确等待 | `vllm_kb/github_pull.py` `_sleep_for_rate_limit` | 403/429 按 Retry-After/X-RateLimit-Reset 精确等待 |
| 任务1检索加权 | `scripts/verify_task1_final.py` | 组件过滤 + 错误签名关键词加分（适配"专有签名优先"） |

## 最终知识库规模

- canonical：66,124 条（vllm 主仓库 51,772 + vllm-ascend 14,352）
- SQLite：docs 66,124 / chunks_meta 122,487 / FTS5 全文索引
- LanceDB：93,888 chunks 向量（bge-m3 1024 维）
- 测试：133 个全部通过
