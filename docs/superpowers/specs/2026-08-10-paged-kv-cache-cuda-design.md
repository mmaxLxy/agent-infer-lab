# Paged KV Cache CUDA Append/Gather 设计

日期：2026-08-10

## 1. 阶段目标

实现一组能够解释分页 KV Cache数据搬运原理的 C++/CUDA算子，并用PyTorch参考实现、正确性测试、CUDA Event基准和Nsight分析证明结果可信。

证据链严格拆分：PyTorch参考实现只作为correctness oracle（正确性标准答案）；Kernel性能只比较同一CUDA扩展内的朴素版和逐项优化版；端到端性能只比较同一版本vLLM中仅改变一个变量的配置。禁止用PyTorch与vLLM的跨系统差异证明Kernel增量收益。

本阶段首先优化单层KV Cache的数据搬运，不直接修改vLLM源码。端到端集成放在独立验收步骤，避免把算子正确性、编译问题和服务问题混在一起调试。

## 2. 与真实模型的关系

本机Qwen2.5-0.5B配置：

```text
hidden_size: 896
num_attention_heads: 14
num_key_value_heads: 2
num_hidden_layers: 24
head_dim: 896 / 14 = 64
```

首版基准必须覆盖真实主形状：

```text
num_kv_heads = 2
head_dim = 64
block_size = 16
dtype = float16
```

同时保留少量泛化形状，例如 `head_dim=128`，验证实现没有把所有逻辑写死在单一模型上。

## 3. 数据布局

每一层分别保存Key Cache和Value Cache：

```text
key_cache:   [num_blocks, block_size, num_kv_heads, head_dim]
value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

新产生的Token使用连续张量表示：

```text
keys:   [num_tokens, num_kv_heads, head_dim]
values: [num_tokens, num_kv_heads, head_dim]
```

每个Token对应一个线性物理槽位：

```text
slot_mapping: [num_tokens]
block_id = slot / block_size
block_offset = slot % block_size
```

例如 `block_size=16`、`slot=35`：

```text
block_id = 35 / 16 = 2
block_offset = 35 % 16 = 3
```

表示该Token写入第2个物理块的第3个位置。

## 4. 算子语义

### 4.1 Append

Append把Decode阶段新产生的K/V写入分页缓存：

```text
for token in tokens:
    slot = slot_mapping[token]
    key_cache[slot] = keys[token]
    value_cache[slot] = values[token]
```

首版约束：

- 所有输入位于同一CUDA设备。
- Key与Value的dtype和形状一致。
- 张量连续。
- `slot_mapping`使用int64。
- 槽位必须位于 `[0, num_blocks * block_size)`。
- 同一次调用中槽位不得重复，避免并发写冲突。

### 4.2 Gather

Gather按逻辑顺序从离散物理槽位读取K/V：

```text
gathered_keys[token] = key_cache[slot_mapping[token]]
gathered_values[token] = value_cache[slot_mapping[token]]
```

它用于验证分页布局中的随机读取，并为后续Attention读取或Cache迁移实验提供基础。

## 5. 正确性基线

PyTorch参考实现只描述语义，不承担性能目标。CUDA输出必须与参考输出逐元素一致。

因为Append/Gather是纯数据复制、没有浮点运算，float16结果应当位级一致；不能用较大的误差容忍掩盖索引错误。

测试至少覆盖：

- 单Token和多Token。
- 跨block边界，例如槽位15、16、17。
- 非连续物理槽位。
- 第一个和最后一个合法槽位。
- Key和Value分别验证，防止两者写反。
- 越界槽位、重复槽位、形状不匹配和错误dtype。
- 真实形状 `num_kv_heads=2、head_dim=64`。

## 6. CUDA实现路线

### 6.1 V0朴素基线

真正的朴素基线由一个CUDA线程负责一个Token，并使用标量循环复制该Token的全部Key/Value元素。该实现优先保证语义直观，不主动构造合并访存或向量化访问。

该版本是同一扩展内所有Kernel加速比的唯一主分母。现有“一线程对应一个连续元素”的实现已经具有合并访存特征，应归入V1，不能继续称为未优化基线。

### 6.2 V1合并访存

使用一维连续元素映射，让相邻线程访问同一Token内连续的head维数据。除线程到元素的映射外，输入、输出、Stream、编译选项和计时边界与V0保持一致。

### 6.3 V2向量化

在地址满足对齐条件时，使用16字节向量化加载和存储，减少指令数量，并让一个warp访问连续地址。

不满足对齐或尾部不足16字节时必须走正确的标量路径，不能为了性能假设所有输入天然对齐。

### 6.4 V3线程布局

在保持V2向量宽度不变的情况下，改为一个CTA或Warp协作处理一个Token，减少跨Token地址跳转和重复slot读取。线程布局的收益单独报告，不能与向量化合并成一个版本。

### 6.5 V4其他优化

其他优化拆成独立变体，例如地址计算提前、`__restrict__`、Qwen主形状特化或launch参数调整。每个变体一次只增加一个变量；没有正收益的优化保留真实负结果，不并入最终版本。

本算子主要受显存带宽限制，优化重点是：

- 减少索引重复计算。
- 合并全局内存访问。
- 使用合适的向量宽度。
- 避免不必要的中间张量与主机同步。

## 7. 工程结构

计划新增：

```text
src/agent_infer_lab/kv_cache_layout.py # 纯CPU槽位映射与输入约束
cuda/kv_cache_ops.cpp              # PyTorch C++绑定与输入校验
cuda/kv_cache_kernels.cu           # Append/Gather CUDA Kernel
cuda/reference.py                  # PyTorch参考实现
cuda/test_kv_cache_ops.py          # 本地GPU正确性测试
cuda/benchmark_kv_cache_ops.py     # CUDA Event微基准
tests/test_kv_cache_layout.py      # 不依赖GPU的索引语义测试
```

CUDA扩展是可选开发组件，不加入核心Python运行时依赖。GitHub CPU CI继续只运行 `tests/`，不要求GPU、CUDA或PyTorch。

## 8. 性能测量

微基准使用CUDA Event，不使用Python墙钟直接测量异步Kernel时间：

1. 预热至少100次。
2. 正式测量至少1000次。
3. 测量前后正确同步。
4. 报告中位数或稳定均值、搬运字节数与有效带宽。
5. 主性能基线只使用同一扩展中的V0朴素CUDA Kernel；PyTorch只用于正确性验证，不参与加速比计算。

有效带宽必须按实际读写字节计算，不能只报告倍数。

报告必须同时给出相邻版本增量收益和相对V0的累计收益，避免只展示最终版本。Kernel指标与端到端TTFT、TPOT、吞吐指标分表保存，禁止互相换算。

### 8.1 实验产物

每次实验使用不可覆盖的run_id，至少保存：

```text
results/<run_id>/manifest.json
results/<run_id>/commands.sh
results/<run_id>/workload.jsonl
results/<run_id>/raw/kernel_samples.csv
results/<run_id>/raw/requests.jsonl
results/<run_id>/raw/server.log
results/<run_id>/summary/kernel.json
results/<run_id>/summary/end_to_end.json
```

`manifest.json`记录Git commit、工作区状态、GPU、驱动、CUDA、Python、PyTorch、vLLM、模型revision、完整命令、输入分布、随机种子、预热次数、正式次数和重复轮数。估算数字单独保存在“非实测”参考文档中，不得混入summary或report。

## 9. Nsight分析

使用Nsight Systems确认调用时间线和同步开销，使用Nsight Compute观察：

- Kernel执行时间。
- DRAM吞吐与峰值带宽比例。
- Global Load/Store效率。
- Warp执行效率。
- Occupancy。
- 是否存在未合并访问。

优化结论必须由指标支持，不能只凭代码看起来更复杂就宣称更快。

## 10. 完成标准

- Append与Gather均有PyTorch参考实现和CUDA实现。
- 正确性覆盖边界、随机槽位和真实Qwen形状。
- 朴素版与优化版使用相同输入比较。
- 完成V0到V4的逐项消融，同时报告相邻增量和累计收益。
- 给出真实耗时、加速比和有效带宽，不提前编造目标数字。
- Kernel与端到端使用两套指标表，PyTorch只作为correctness oracle。
- 保存完整环境、命令、输入分布和未汇总原始结果。
- 保存Nsight命令、报告路径和关键观察。
- 全量CPU测试与Ruff继续通过。
- CUDA不可用时CPU CI不会失败，但本机GPU验收必须真实通过。
