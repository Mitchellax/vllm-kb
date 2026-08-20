# GLM-5.1 PD 分离 P 节点崩溃分析：MTE out of range（repeat_interleave）

> 场景：vllm-ascend 0.18.0-a3 镜像（**正式 release 版**，2026-04-30 发布）+ #10112 patch，
> GLM-5.1 W8A8，A3 4 机 PD 分离 + **元戎（yuanrong / openYuanrong）多级缓存**（非 Mooncake），
> 压测 64K 输入 / 16K 输出 / 40~60 并发 → P 节点崩溃。
> 报错：`errorStr: The DDR address of the MTE instruction is out of range.`
> `fixp_error0 info: 0x3000012e, fixp_error1 info: 0x34, fsmld:1, tslot:3`
> python 堆栈：**`torch.repeat_interleave`（已确认在 `token_dispatcher` 路径）**；plog 无报错。
> 经验：池化分配的 DDR 池过大，元戎内存 800G→500G 后未复现。

> 注：元戎（yuanrong / openYuanrong）为独立的多级 KV 缓存实现，与 Mooncake 不同。
> 本次分析中与 Mooncake 相关的类比仅作机制参考，不视为同一组件。

---

## 1. 报错签名解读

- **`MTE instruction is out of range`（DDR 地址越界）**：AI Vector Core 的 MTE（Memory Transfer Engine）访问了超出分配范围的 DDR 地址。这是 Ascend 硬件层错误，通常意味着**某个 tensor 的 buffer 分配不足，或索引计算越界**，kernel 访问了非法 GM 地址。
- **`0x3000012e` / `0x34`**：fixp_error 硬件错误码（MTE 越界的详细分类）。
- **python 堆栈在 `torch.repeat_interleave`**：说明崩溃发生在 **repeat_interleave 展开生成的 tensor 被后续算子访问时**（repeat_interleave 本身是 host 操作，报错的是它产出的 buffer 在 device 上被 MTE 访问越界）。
- **plog 无报错**：与 #11859/#9816 等一致——崩溃在 graph replay / async 路径，plog 只有硬件错误，python 侧是异步触发的堆栈。

## 2. 知识库命中的社区问题（按相关度）

| # | 问题 | 关联点 |
|---|---|---|
| [#11859](https://github.com/vllm-project/vllm-ascend/issues/11859) ✅ | GLM5.1 16 卡 910c TP8 DP1 CP2，**60k 上下文 MTE out of range** | GLM-5.1 + 长上下文 + MTE，完全同模型同错误 |
| [#7570](https://github.com/vllm-project/vllm-ascend/issues/7570) ✅ | Qwen3.5 FullGraph MTE out of range，根因 `FusedInferAttentionScore` + `update_full_graph_params`；**PIECEWISE 正常** | graph 机制 + MTE；v0.18 之后修复 |
| [#9816](https://github.com/vllm-project/vllm-ascend/issues/9816) ✅ | DSA compress + ACLGraph **idle dummy batch** → **stale block table/slot mapping 被 graph replay 复用 → MTE OOB** | **与"池化 DDR + P 节点"最接近的根因** |
| [#10204](https://github.com/vllm-project/vllm-ascend/issues/10204) ✅ | DSV4-Flash QLI ops **序列超 300K MTE out of range** | 超长序列 → 索引溢出 |
| [#2757](https://github.com/vllm-project/vllm-ascend/issues/2757) ✅ | 100k 输入 UB 越界 | 长序列 buffer 越界 |
| [#3263](https://github.com/vllm-project/vllm-ascend/issues/3263) ✅ | MTE out of range 原题 | 报错原题 |
| [#14122](https://github.com/vllm-project/vllm-ascend/issues/14122) ⬜ | DSpark SparseAttnSharedkv decode MTE out-of-range | DSpark/稀疏注意力路径 |
| [#8015](https://github.com/vllm-project/vllm-ascend/issues/8015) ⬜ | **0.18.0rc1 A3 双机 W8A8** 197K/1K MTE out of range | 同版本同硬件同错误 |

## 3. 崩溃机制分析

### 3.1 repeat_interleave 的展开位置（v0.18.0 代码仓证据）

v0.18.0 里 `torch.repeat_interleave` 的关键路径：

1. **`vllm_ascend/worker/pcp_utils.py:796`（PCP 上下文并行）**：
   ```python
   block_table_tensor[:num_decode_reqs].repeat_interleave(ori_query_lens[:num_decode_reqs], dim=0)
   ```
   decode 请求的 block_table 按各自 query_len 展开。**64K 长序列 → 每个请求的 block_table 展开巨大**。

2. **`vllm_ascend/ops/fused_moe/token_dispatcher.py:568`（MoE token 分发，#10112 patch 区域）**：
   ```python
   global_input_tokens_local_experts_indices = torch.repeat_interleave(
       self.expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel()
   )
   ```
   按每个 expert 的 token 数展开 expert 索引。**64K 输入 × 40 并发 → 单卡 token 数巨大 → 展开索引超界**。

### 3.2 与用户经验的吻合（DDR 池 800G→500G 消失）

用户经验"池化分配的 DDR 池过大，800G→500G 后未复现"完全符合 **MTE 越界的两种触发**：

- **方案 A（buffer 不足）**：元戎池化给算子分配的 DDR workspace 不足，MTE 访问越界。
- **方案 B（索引溢出）**：池化 buffer 足够大时，`repeat_interleave` 产生的索引/offset 计算溢出（tiling 按大 buffer 切分，某 rank 的 offset 超 int32/int64 边界）→ MTE 访问到 buffer 外。

**800G→500G 消失**更符合**方案 B 的另一种形态**：池化 buffer 过大时，某些按"池大小"推导的步长/偏移（而非按实际请求 token 数）在 graph replay / dummy batch 下用了**旧值或过大的 tiling 参数**，导致 MTE 越界。缩小池后，实际使用量与推导参数重新匹配，崩溃消失。

### 3.3 与 #9816（stale block table 复用）的关系

#9816 指出：DSA/compress 路径下 **idle dummy batch 不经过正常 `_prepare_inputs()`**，请求完成后留下的 **stale block table / slot mapping 被 graph replay 复用 → MTE 越界**。

用户场景叠加了 **PD 分离（P 节点 prefill 完即释放）+ 池化（KV 复用）+ FullGraph/PIECEWISE graph**：P 节点在 40~60 并发长序列下，graph replay 复用了**池化 buffer 的旧 slot mapping**，而 buffer 已按 800G 池大小重排 → MTE 访问越界。这与"池化 DDR 池过大"的经验直接对应。

## 4. 根因结论

**最可能根因**：P 节点在 **FullGraph/ACLGraph + 元戎多级缓存（yuanrong/openYuanrong）池化** 组合下，
`torch.repeat_interleave`（**已确认在 MoE token 分发路径**，`token_dispatcher.py:568`，#10112 patch 区域）
产出的索引/buffer 在 **graph replay 或 dummy batch 路径**复用了与当前实际 token 数不匹配的
**stale 参数（过大的池化步长/旧 slot mapping）**，导致 MTE 访问 DDR 越界。
池化 buffer 越大（800G），推导参数与实际使用越失配 → 崩溃概率越高；缩小到 500G 后失配消失。

**直接证据**：
- v0.18.0 代码：`token_dispatcher.py:568`（#10112 patch 区域）和 `pcp_utils.py:796` 的 repeat_interleave；
- 社区：#9816（stale block table → MTE OOB，DSA + graph + dummy batch）、
  #7570（FullGraph 机制 + MTE，PIECEWISE 正常）、#11859（GLM-5.1 + 60k + MTE）；
- 用户经验：池化 DDR 池大小直接影响崩溃（800G 崩 / 500G 不崩）。

## 5. 排查建议（按优先级）

1. **对比 graph 模式**：P 节点 `cudagraph_mode` 从 FULL_DECODE_ONLY/FULL 改 **PIECEWISE**（#7570 明示 PIECEWISE 正常）——若崩溃消失，锁定 graph 机制问题。
2. **核对 repeat_interleave 具体调用点**：用户已确认是 `token_dispatcher.py:568`（MoE 分发）——
   聚焦 #10112 patch 的副作用：patch 引入的展开逻辑在长序列高并发下索引超界。
3. **池化参数**：元戎/池化 buffer 大小与 `max_num_batched_tokens`/`max_model_len` 的推导关系——800G 池下 tiling 的 offset 步长是否按池大小而非实际使用量。
4. **检查 0.18.0 的已知修复**：#7570/#9816/#10204/#9724 都在 0.18 之后修复，确认 0.18.0-a3 镜像是否含这些 patch；不含则升级或 cherry-pick。
5. **DSA/compress 关闭对照**：`enable_dsa_cp=false`（#9816 场景）验证是否与 DSA 路径相关。
6. **长序列复现最小化**：从 64K 降到 32K / 并发从 60 降到 20，观察崩溃阈值，反推是"总 token 数"还是"单序列长度"触发。

## 6. 与 #10112 patch 的关系

#10112 修的是 MoE 多 DP 挂起（vllm #44185），改动在 `token_dispatcher` 区域。
**用户叠加该 patch 后出现 MTE 崩溃**：两种可能——
- patch 本身与 0.18.0 的 graph/池化路径存在**未覆盖的交互**（patch 是 0.18 backport，后续版本有更多相关修复）；
- 或 patch 修复挂起后，**MoE 分发路径实际跑通了**，之前被挂起掩盖的 MTE 越界问题暴露出来。
建议：确认 patch 是否引入 `repeat_interleave` 附近逻辑变化，并与最新版 token_dispatcher 对比。
