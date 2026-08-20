# PD 分离 P 节点崩溃分析（修正版）：Communication_Error_Bind_IP_Port

> 场景：4 机 A3，3×1P + 1×1D，vllm-ascend **v0.22.1rc1**（rc 版，2026-06-30），
> 已对齐最佳实践配置，服务正常运行**两天半**后 P 节点崩溃退出。
> 定位：报错在 `token_dispatcher.py` 的 `torch.repeat_interleave` **之后**，
> 在 `multiproc_executor.py:962` 上报 `Communication_Error_Bind_IP_Port`。
> **kv_port 无异常**；业务**不使用 Ray**（多节点独立启动）。

---

## 1. 修正前分析的问题

- 上一版误判为 Mooncake kv_port 配置/端口段冲突 → **已排除**（kv_port 无异常）；
- 推荐 Ray 方案 → **业务不可用**（3×1P+1×1D 独立启动是硬约束）。

## 2. 真实代码链路（v0.22.1rc1）

```
token_dispatcher.py
  ├─ 612: global_input_tokens_local_experts_indices = torch.repeat_interleave(   ← 用户定位点
  │          self.expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
  ├─ 638/644: npu_moe_token_permute(...)                                          ← token 置换
  └─ 230-234: torch_npu.npu_moe_distribute_dispatch(**kwargs_mc2)                 ← MC2 通信算子
                 （或 dispatch_v2；kwargs 含 group_ep / moe_all_to_all_group_name）
```

**repeat_interleave 之后**进入 `npu_moe_distribute_dispatch`（MC2 跨节点 MoE 分发）——
这正是 #9447 报的 `aclnnMoeDistributeDispatchV4` 的 torch_npu 封装。

## 3. Communication_Error_Bind_IP_Port 的机制

该错误码不在 vllm 主仓（0.22.1 / main 的 network_utils.py 均无），
也不在 vllm-ascend 源码——是 **HCCL/CANN 集合通信层**的错误分类：
**HCCL 建立跨节点通信链路时 bind IP:Port 失败**。

社区证据（#9447 plog）：
```
[hccl_socket_manager] comm error, device[0]
| dest_ip(user_rank) | dest_port | src_ip(user_rank) | src_port | ...
| 10.20.0.31(29)     | 16666     | 10.20.0.2(0)      | 344832056 | ...
[Wait][LinkEstablish] wait socket establish timeout[120s]
```
**`src_port: 344832056` 端口号异常巨大**（正常端口 0~65535）——HCCL 通信端口计算/分配异常，
与用户 `port 65536`（=2^16 越界哨兵）同源：**都是端口数值溢出**。

## 4. 与用户场景（3×1P + 1×1D）的吻合

- **MoE 分发跨 4 节点**：3 个 P 节点 + 1 个 D 节点都参与 EP（专家并行）的
  `npu_moe_distribute_dispatch` 跨机通信；
- 3 个 P 节点**独立启动**（无 Ray 统一编排）→ 各自的 HCCL 通信组/端口初始化**时序不同步**；
- **"两天半后崩溃"** = 长稳后某 P 节点通信资源（HCCL socket/组）释放-重建时，
  端口分配与初始状态错位 → bind IP:Port 失败（#13527 同类：运行一段时间后 HCCL 失败挂死）；
- 与 kv_port（Mooncake KV 传输端口）**无关**——这是 **HCCL 集合通信（MoE 分发）** 的端口，
  由 HCCL 自身分配，非用户配置。

## 5. 知识库命中（按相关度）

| # | 问题 | 关联 |
|---|---|---|
| [#9447](https://github.com/vllm-project/vllm-ascend/issues/9447) ⬜ | GLM-5.1 0.18 双机 **aclnnMoeDistributeDispatchV4** 报错，plog `src_port: 344832056`（HCCL 端口异常） | **同算子 + 端口溢出直接证据** |
| [#13527](https://github.com/vllm-project/vllm-ascend/issues/13527) ⬜ | v4 flash **PD分离运行一段时间后 p0 因 hccl 失败挂死** | 同场景（PD 长稳 + HCCL 失败） |
| [#12461](https://github.com/vllm-project/vllm-ascend/issues/12461) ✅ | GLM5.2 多节点 SUSPECT REMOTE ERROR（dummy_run 路径） | 多节点通信失败 |
| [#13252](https://github.com/vllm-project/vllm-ascend/pull/13252) ✅ | fix mtp and dcp bug | 通信相关修复（已合 v0.23） |
| [#8938](https://github.com/vllm-project/vllm-ascend/issues/8938) ✅ | 多 P 节点端口抢占（ZMQ） | 多 P 节点端口冲突 |

## 6. 根因结论

**最可能根因**：3×1P + 1×1D 多节点独立启动（无 Ray）下，MoE 专家并行的
`npu_moe_distribute_dispatch`（token_dispatcher repeat_interleave → MC2 分发）跨节点
HCCL 通信链路，在长稳运行两天半后**通信资源释放-重建时端口分配异常**
（HCCL socket 端口计算溢出 → `Communication_Error_Bind_IP_Port` / `port 65536`），
导致 P 节点崩溃。**与 kv_port 配置无关**（那是 Mooncake KV 传输端口，非 HCCL 通信端口）。

**触发条件**：多 P 节点并行 MoE 分发 + 长稳运行 + 独立启动时序漂移 → 偶发，
复现不稳定（#13527 同类）。

## 7. 排查建议（按优先级）

1. **确认是哪个通信组失败**：取崩溃 P 节点 plog，找 `hccl_socket_manager` / 
   `Communication_Error_Bind_IP_Port` 附近的 **HCCL 通信组名**（EP 组 / TP 组 / DP 组），
   判断是 MoE 分发（EP）还是其他集合通信。
2. **对比 src_port/dest_port 是否异常**（#9447 是 344832056）：确认端口溢出是
   HCCL 内部计算问题还是环境变量影响。
3. **检查 HCCL 相关环境变量**：`HCCL_SOCKET_IFNAME` / `HCCL_IF_IP` / `HCCL_OP_EXPANSION_MODE`
   在 3 个 P 节点上是否一致（独立启动易配置漂移）；
   #14296 已知 `HCCL_OP_EXPANSION_MODE=AIV` 在多通信域有抢核问题。
4. **3 个 P 节点启动顺序/一致性**：确认各节点 HCCL 初始化参数（rank 映射、通信组）
   完全一致；独立启动时常见"某节点 rank 映射漂移"导致通信组 bind 异常。
5. **升级到正式版 v0.23.0**（2026-08-16 发布）：#13252（mtp/dcp）、#11342（ZMQ 端口映射）、
   #4054（Mooncake PP 端口）等修复都在 0.22.1rc1 之后，rc 版 backport 不全。
6. **复现采集**：加 `HCCL_DEBUG=INFO`（或对应开关）重跑，捕获崩溃前最后一次
   HCCL 建链的 IP:Port，对比各节点。

## 8. 版本形态

- v0.22.1rc1 = **rc（预发布）**，2026-06-30 发布（版本日历确认）；
- 正式版 v0.23.0 = **release**，2026-08-16 发布。
