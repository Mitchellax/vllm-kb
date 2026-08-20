# 任务2：GLM-5.1 PD 分离 P 节点崩溃分析（DispatchFFNCombine / Fused_MC2）

> 分析对象：vllm-ascend main 分支源码（`D:\dsh\vllm-ascend-src\vllm-ascend-main`）
> 场景：vllm-ascend 0.23.0rc1，PD 分离（P 节点 = kv_producer），GLM-5.1 w8a8，
> Fused_MC2=1（`VLLM_ASCEND_ENABLE_FUSED_MC2=1`），长期运行（~1 天）后 P 节点崩溃。
> 崩溃特征：
> - `halMemCreate failed drvRetCode=6, driver error: out of memory`
> - `kernel_name=DispatchFFNCombine`
> - `extend info: errorStr: timeout or trap error`

---

## 1. 崩溃机制：DispatchFFNCombine 是 Fused_MC2 的融合算子

`VLLM_ASCEND_ENABLE_FUSED_MC2=1` 时，MoE 通信从 ALLTOALL/MC2 切换到融合算子：
`dispatch_ffn_combine`（w8a8）或 `mega_moe`（w8a8/w4a8/bf16）。

路径选择（`vllm_ascend/ascend_forward_context.py` `_select_a3_moe_comm_method`）：

| 条件 | 路径 |
|---|---|
| `_MEGA_MOE_SUPPORTED`（装了 `cann_ops_transformer`）且 EP≤64 | `mega_moe` |
| 未装 `cann_ops_transformer` 且 EP≤32 | `dispatch_ffn_combine` |
| 其他 | MC2 / ALLTOALL |

用户崩溃在 `DispatchFFNCombine` → **未安装 `cann_ops_transformer`，走 `dispatch_ffn_combine` 路径**。

调用链（`vllm_ascend/ops/fused_moe/moe_comm_method.py` `fused_experts`）：
```
torch.ops._C_ascend.dispatch_ffn_combine(
    x, weight1, weight2, expert_idx, scale1, scale2, bias1, bias2,
    probs, group, max_output_size=mega_moe_max_tokens, ...
)
```
→ `aclnnDispatchFFNCombine*` → CANN 算子 → kernel 执行。

## 2. 直接根因：workspace 按 max_output_size 最坏情况预留 → P 节点 OOM

`csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp` 的 workspace 计算：

```cpp
uint32_t n2 = info.K;              // K = hidden_size
uint32_t k2 = info.N / 2;
uint64_t cocWorkspace =
    (info.M + 256 - 1) / 256 * 256 * info.topK * sizeof(int32_t) +   // 路由表
    info.worldSize * info.worldSize * info.expertPerRank * sizeof(int32_t) * 2 +
    info.maxOutputSize * sizeof(float) * 2 +                          // ← max_output_size × 8B
    info.maxOutputSize * n2 * sizeof(int16_t) +                       // ← max_output_size × K × 2B
    info.maxOutputSize * info.K * sizeof(int8_t) +                    // ← max_output_size × K × 1B
    info.worldSize * sizeof(int32_t) * 16 +
    (info.expertPerRank + info.worldSize) * sizeof(int32_t) * 16;
workSpaces[0] = SYSTEM_NEED_WORKSPACE + std::max(cocWorkspace, initRoutingWorkspace);
```

`max_output_size` 来自 `ascend_config.mega_moe_max_tokens`，**默认 131072**。

对 GLM-5.1（hidden≈5120，K=5120）：
- `maxOutputSize * K * 3B = 131072 × 5120 × 3 ≈ 1.9 GB` —— 单算子 workspace 预留近 2GB！
- 加上 HCCL buffer（285 行显式检查 `HCCL_BUFFSIZE`）和 `symmetricPtr` 对称 buffer。

官方注释（`ascend_config.py` L278-283）明说：
> "workspace memory scales linearly with this value... Do not set this too large"

## 3. 触发条件：长期运行 + 业务流量 → token 数超算子容量上限

`ascend_forward_context.py` 常量：
```python
_MEGA_MOE_TOKENS_PER_RANK_LIMIT = 4096        # mega_moe 每 rank 上限
_DISPATCH_FFN_COMBINE_TOKENS_PER_RANK_LIMIT = 512   # dispatch_ffn_combine 每 rank 上限！
_MC2_TOKENS_PER_RANK_LIMIT = 512
```

`set_mc2_tokens_capacity`（L203-227）：`num_tokens_per_tp_rank = min(计算值, 上限)`。

用户 P 节点：`max-num-batched-tokens 8192, tp8` → 每 rank 1024 tokens
> `_DISPATCH_FFN_COMBINE_TOKENS_PER_RANK_LIMIT = 512` → **实际业务流量下每 rank token 数超过 512 上限**。

`ascend_config.py` L278-283 还写明：超限 token 会被**丢弃跳过**（降精度），且 workspace 按上限预留。

## 4. 三个疑点叠加（为什么"压力测试正常、业务流量崩溃"）

1. **workspace 巨大（~2GB/算子）+ 长期运行内存碎片化**：压力测试时显存整齐，业务流量（不同 prompt 长度、并发波动、KV transfer 的 Mooncake buffer）长期运行后显存碎片化，`halMemCreate` 在算子执行时申请 workspace 失败 → `drvRetCode=6, out of memory`。这正是"运行 1 天左右崩溃"的特征。

2. **每 rank token 超过 dispatch_ffn_combine 的 512 上限**：压力测试（均匀负载）与业务流量（突发、长尾）的 token 分布不同；业务流量下某 rank 收到 >512 tokens，触发算子的 drop/异常路径 → `timeout or trap error`（kernel 执行异常）。

3. **P 节点 prefill 大 batch**：PD 分离 P 节点处理 prefill，`max-num-batched-tokens 8192`，chunked prefill 下可能瞬时出现大 token 数 → 超过算子容量。

## 5. 已知问题对照（release_notes.md）

- **#8320（已知问题）**：`VLLM_ASCEND_ENABLE_FUSED_MC2` 不推荐用于 multi-DP + 大 token 场景（kv_producer/kv_both），会产生大量 padding tokens 路由到特定专家，某些 rank token 过载 → **正好命中用户 PD 场景（P 节点 = kv_producer）**。
- **#8853（已知问题）**：GLM5 和 Deepseek V3.2 分开 PD 部署时有概率空输出/乱码。
- **#8844（已知问题）**：GLM 5/5.1 PD 分离 D 节点 TP16 DP2 下 GPQA 精度不达标。
- **#6468 / #6707**：DispatchFFNCombine 优化 + 修复未对齐 UB 访问导致的向量错误（说明该算子历史上就有 kernel 级 bug）。

## 6. 排查方向建议（按优先级）

### A. 最可能直接解决：调小 `mega_moe_max_tokens`（降低 workspace 预留）
```json
--additional-config '{"enable_fused_mc2":1, "mega_moe_max_tokens": 16384}'
```
- workspace 从 ~2GB 降到 ~256MB（131072→16384，线性缩小 8 倍）
- 官方注释明确支持此参数（"Do not set this too large"）
- **副作用**：超过该值的 token 会被丢弃（精度下降）——需按业务实际每 rank 峰值 token 设定，取 max(业务峰值, 512上限×tp) 略大值。

### A2. 关键修复参考（社区 PR，命中用户场景）

**#14358 `Fix MegaMoe prefill buffer sizing`（v0.26.0rc，open 未合并）**
- CANN MegaMoe 的 symmetric 通信 buffer 从 MC2 token capacity 一次性分配；
- 当 `enable_prefill_mc2` 关闭（默认 False）时，该容量从 **decode graph capture size** 推导；
- 但 eager prefill 仍可能通过 fused MC2 选中 MegaMoe → **prefill batch 超过分配容量 → 无效输出/崩溃**；
- 用户 P 节点（prefill）max-num-batched-tokens=8192 而 capture size 通常远小 → **正是此 bug 的触发条件**。
- 对策：P 节点开启 `"enable_prefill_mc2": true`（让容量按 prefill 的 max_num_batched_tokens 推导），或升级到含此修复的版本。

**#14304 / #14305 `Keep DP token counts uniform for MegaMoe`（open 未合并）**
- CANN MegaMoe 要求通信域内所有 rank 输入 token 数一致；
- PD 分离下 KV consumer 可能跳过 DP metadata all-reduce → rank 间 token 数不一致 → 错误输出；
- 对策：关注合并进度，或 PD 场景临时关闭 Fused_MC2。

### B. 检查是否走了 dispatch_ffn_combine 而非 mega_moe
- 安装 `cann_ops_transformer` → 走 mega_moe（上限 4096/rank，支持 w8a8/w4a8/bf16，EP≤64）
- mega_moe 的每 rank 上限 4096 远大于 dispatch_ffn_combine 的 512，更贴合 P 节点 8192 tokens/8rank=1024 的负载
- 升级 CANN 到支持 mega_moe 的版本（_is_megamoe_supported_by_config 还要求 hidden∈[1024,8192] 且 %512=0，intermediate∈[1024,3072] 且 %512=0）

### C. 关闭 Fused_MC2 验证（隔离变量）
```bash
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
```
- 回退 MC2/ALLTOALL 路径，确认崩溃消失 → 锁定为 fused 算子问题
- 注意：GLM5 文档说明 Fused_MC2 与 MTP、dynamic EPLB、multistream_overlap_shared_expert 冲突，关闭后需确认这些配置

### D. 核对 HCCL_BUFFSIZE（285 行显式检查）
tiling 里要求：
```
HCCL_BUFFSIZE 需 ≥ (M × K × topK × sizeof(int8_t)) × 3 + 3MB
```
M=8192, K=5120, topK=8 → ~1GB+。若 HCCL_BUFFSIZE 不足会直接报错；够但接近上限时，长期运行碎片化后同样 OOM。

### E. 显存碎片化治理（长期运行通用手段）
- 调小 `--gpu-memory-utilization`（留出 workspace 余量）
- P 节点 `VLLM_ASCEND_ENABLE_MLAPO=0`（文档：w8a8 默认开启，更耗显存）
- 检查 Mooncake KV transfer buffer（P 节点 producer）占用
- 观察 `npu-smi info` 显存碎片（memory fragmentation）

### F. 收集崩溃现场证据
- CANN 日志（`/root/ascend/log`）：确认 halMemCreate 是哪个 buffer（workspace / symmetric / HCCL）
- `timeout or trap error` 的 kernel 现场（CANN dump）：确认是 memory fault 还是指令 trap
- 记录崩溃前最近的 token 数 / batch 分布，与 512 上限对比

## 7. 结论

**最可能根因**：PD 分离 P 节点启用 Fused_MC2=1 且未装 cann_ops_transformer → 走 dispatch_ffn_combine 路径，
该算子 workspace 按 `mega_moe_max_tokens`（默认 131072）最坏情况预留 ~2GB，且每 rank token 上限仅 512；
长期运行显存碎片化后，算子执行时 `halMemCreate` 分配 workspace 失败（drvRetCode=6 OOM），
叠加业务流量 token 数超上限 → kernel 异常（timeout or trap error）。

**验证顺序**：先 A（调小 mega_moe_max_tokens）→ 再 B（装 cann_ops_transformer 走 mega_moe）→ 最后 C（关 Fused_MC2 对照）。
