# PD 分离 P 节点崩溃分析：port 65536 have already been bound

> 场景：4 机 A3，3×1P + 1×1D，vllm-ascend **v0.22.1rc1**（2026-06-30 发布，rc 版），
> 已对齐最佳实践配置，服务正常运行**两天半**后 P 节点崩溃退出。
> 报错：`port 65536 have already been bound`

---

## 1. 报错解读

- **`port 65536` 不是普通端口冲突**：TCP 端口范围是 **0~65535**，65536 = 2^16 是**越界值**。
  这意味着 ZMQ 尝试 bind 一个**计算溢出的非法端口**——不是"端口被占用"，而是"端口号本身非法"。
- **`have already been bound`**：ZMQ 对非法/重复端口 bind 失败时的报错文案（与 #8938 的
  `Address already in use` 同源，但端口号异常）。

## 2. 代码仓根因（v0.22.1rc1 mooncake_connector.py）

端口计算公式（`mooncake_connector.py` 1440-1446 / 1774-1783）：

```python
# Handshake base port
self.side_channel_port = (
    vllm_config.kv_transfer_config.kv_port
    + data_parallel_rank * tensor_parallel_size * pipeline_parallel_size * pcp_size
)
device_index = (pp_rank + pcp_rank) * tp_size + tp_rank
self.handshake_port = self.side_channel_port + device_index   # ← 287/359/1783 行
path = make_zmq_path("tcp", side_channel_host, handshake_port)
```

**`handshake_port = kv_port + dp_rank*tp*pp*pcp + device_index`**。

**65536 的产生**：当 `kv_port` 基址 + 各 rank 偏移之和 ≥ 65536 时，ZMQ bind 非法端口失败。

## 3. 与用户场景（3×1P + 1×1D）的吻合

**GLM5 官方文档（GLM5.md:1271）明示**：
> `"kv_port"`: Port for Mooncake KV transfer communication. **Each node group should use a distinct port range.**

用户是 **3 个 P 节点 + 1 个 D 节点**（4 组，每组 1P 或 1D）：
- 若 3 个 P 节点配置了**相同或相邻的 kv_port 基址**（未按节点区分端口段），
  `side_channel_port + device_index` 会在不同节点算出**相同/相邻的 handshake 端口**；
- 运行两天半后，某 P 节点**内部线程/连接重启或重连**时（Mooncake 的 kv_send/recv 线程、
  端口释放再申请），与另一节点的端口发生**跨节点冲突**，或偏移计算溢出到 65536。

**"两天半"特征**：端口冲突是**偶发**的——需要特定时序（某节点恰好释放/重连端口、
缓冲池水位触发新连接）才触发，与"长稳后崩溃"的社区模式一致（#8972/8974/8975 卡死、
#8938 端口抢占）。

## 4. 知识库命中的社区问题

| # | 问题 | 关联 |
|---|---|---|
| [#4244](https://github.com/vllm-project/vllm-ascend/issues/4244) ✅ | PD 分离 DP 场景 MooncakeConnector **未考虑 PP，kv 传输端口分配冲突**（PR #4054 修复） | **端口分配冲突的直接已知问题** |
| [#8938](https://github.com/vllm-project/vllm-ascend/issues/8938) ✅ | 1P2+KV Pool，**P0/P1 同时拉起 ZMQ 端口抢占**，Address already in use | 多 P 节点端口冲突 |
| [#11343](https://github.com/vllm-project/vllm-ascend/issues/11343) ✅ | **PD 分离+多节点 PP，ZMQ 端口映射计算错误**（PR #11342 修复）；推荐 Ray 统一拉起 | **多节点端口映射错误** |
| [#8972/8974/8975](https://github.com/vllm-project/vllm-ascend/issues/8972) ✅ | PD 分离推理服务**偶发卡死**，端口正常监听但 curl 无回应 | 长稳后偶发故障模式 |
| [#11968](https://github.com/vllm-project/vllm-ascend/issues/11968) ⬜ | 共享多卡 **HCCL_NPU_SOCKET_PORT 默认同端口**，建议端口可用性校验 | 端口默认值冲突 |

## 5. 根因结论

**最可能根因**：3 个 P 节点 + 1 个 D 节点部署中，Mooncake KV 传输的
`handshake_port = kv_port + rank 偏移 + device_index` 端口分配在**多节点共享 kv_port 基址**时
发生冲突；运行两天半后某 P 节点的连接/线程重启触发端口重新绑定，
偏移计算**溢出到非法端口 65536**（2^16），ZMQ bind 失败 → P 节点崩溃。

**两个叠加因素**：
1. **kv_port 未按节点区分端口段**（文档要求 "each node group should use a distinct port range"，
   3×1P 若共用基址则冲突）；
2. **v0.22.1rc1 是 rc 版**——#4244 的修复 PR #4054 及 #11343 的修复 #11342
   是否完整进入该 rc 需确认（rc 版常见 backport 不全）。

## 6. 排查建议（按优先级）

1. **核对 kv_port 配置**：确认 3 个 P 节点 + 1 个 D 节点的 `kv_port` 是否各自独立端口段
   （如 P0=30000、P1=30100、P2=30200、D=30300，且与 device_index 偏移不重叠）。
   计算 `kv_port + dp_rank*tp*pp*pcp + device_index` 是否 < 65536 且各节点唯一。
2. **确认 v0.22.1rc1 是否含 #4054/#11342**：查该 rc 的 git log / CHANGELOG；
   不含则升级到含修复的版本（v0.23.0 正式版，2026-08-16 发布）。
3. **多节点用 Ray 统一拉起**（#11343 官方推荐）：避免各 P 节点独立启动导致的
   端口映射不一致。
4. **崩溃现场确认**：取 P 节点崩溃前的 plog / mooncake 日志，
   确认是 `handshake_port=65536` 的哪一步（side_channel_port 还是 device_index 溢出），
   以及是否伴随某线程重启。
5. **临时规避**：调低 kv_port 基址（远离 65536 边界），并确保各节点端口段隔离。

## 7. 版本形态说明

- **v0.22.1rc1 = rc（预发布）**（版本日历确认，2026-06-30 发布）；
- 官方正式版 **v0.23.0 = release**（2026-08-16 发布）——建议升级验证。
