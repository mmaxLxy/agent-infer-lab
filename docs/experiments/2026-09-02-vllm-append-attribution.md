# M1：Append 微基准收益无法稳定转化为端到端收益的原因

日期：2026-09-02

## 1. 直接结论

自定义 Append 在隔离测试中确实快于 vLLM 0.23.0 原生算子，但它无法稳定改善端到端性能，主因不是 Kernel 完全无效，而是以下三点同时成立：

1. **基线不同**：11.98 倍是 2,048 Token 下 V3 相对同一扩展内刻意朴素的 V0，不是相对原生 vLLM；在相同 2,048 Token 形状下，自定义实现相对原生 vLLM 的实测加速中位数只有 2.37 倍。
2. **真实形状不同**：有效请求路径的 816 次调用中，720 次属于 1～2 Token 的 Decode 小调用，占 88.24%；2,048 Token 不是端到端 Decode 的主形状。
3. **算子占比过小**：按真实 Decode 形状加权并乘以模型 24 层，原生 Append 只占 TPOT 的 1.48%～1.80%。自定义实现能够带来的 TPOT 改善预计只有 0.34%～0.61%，低于端到端轮次波动，因而方向不稳定。

结论符合 Amdahl 定律：局部算子即使大幅加速，只要它在整个请求中的时间占比很低，对系统性能的影响就很小。

## 2. 证据来源

### 2.1 已冻结的端到端基线

M0 中 native/custom 各完成三轮，每轮 80 请求：

| 指标 | native 中位数 | custom 中位数 | 相对变化 |
|---|---:|---:|---:|
| 输出吞吐 | 298.48 tokens/s | 296.11 tokens/s | -0.79% |
| TPOT P50 | 11.97 ms | 12.16 ms | +1.61% |
| TTFT P50 | 125.78 ms | 121.61 ms | -3.32% |

三对实验的吞吐变化为 -0.79%、+5.36%、-3.71%，TPOT P50 变化为 +1.61%、-7.95%、+4.69%，方向不一致。因此不能从三轮中选择正收益轮次作为结论。

### 2.2 真实 Append 调用形状

从 M0 正确性验证日志中排除 24 次 `valid_tokens=0` 的初始化调用后：

| mapped_rows | 调用次数 | 所属阶段 |
|---:|---:|---|
| 1 | 48 | Decode |
| 2 | 672 | Decode |
| 128 | 48 | Prefill |
| 129 | 48 | Prefill |

Decode 的 1～2 Token 调用占全部有效调用的 88.24%，其中 2 Token 调用占 82.35%。原 11.98 倍微基准使用 2,048 Token，回答的是大批量搬运中的实现差异，不能代表真实 Decode 的主调用形状。

### 2.3 相同形状下的原生/自定义比较

新增 `scripts/benchmark_append_integration.py`，复现归档中的 FP16 输入形状、`[1152, 64, 1]` 输入 stride、NHD Cache 布局和 24 层模型，并运行三个独立进程。每个进程包含七轮交叉顺序测量。

2,048 Token 下：

| 指标 | 三次独立进程结果 |
|---|---:|
| 原生 vLLM device time 中位数 | 13.43 μs |
| 自定义 Append device time 中位数 | 5.36 μs |
| 自定义相对原生加速 | 2.17～2.56 倍，中位数 2.37 倍 |
| 原报告 V3 相对朴素 V0 | 11.98 倍 |

这证明 11.98 倍中的大部分收益来自击败低并行度的 V0，而不是击败已经优化过的 vLLM 原生实现。

真实 Decode 形状加权结果：

| 指标 | 中位数 | 三次进程范围 |
|---|---:|---:|
| 原生路径单层 Append | 7.59 μs | 7.40～8.99 μs |
| 自定义路径单层 Append | 5.71 μs | 5.56～5.94 μs |
| 原生 Append 占 TPOT | 1.52% | 1.48%～1.80% |
| 自定义替换预计 TPOT 变化 | -0.41% | -0.61%～-0.34% |
| 原生 Append 占 TTFT | 0.144% | 0.144%～0.153% |

负的 TPOT 变化表示预计降低。自定义路径在隔离实验中更快，因此本报告不把端到端无收益简单归因于“Python 开销完全吃掉 Kernel 收益”。Python 检查、Dispatcher 和 Kernel Launch 是固定成本，会压缩小形状收益，但主导限制仍是 Append 在整个 Decode 中的占比过小。

## 3. Amdahl 上限

固定模型为 24 层，原生 TPOT P50 为 11.97 ms。三次进程中，真实形状加权的原生 Append 每层约 7.40～8.99 μs：

```text
Append 每步耗时 = 单层 Append × 24 层
               ≈ 177.6～215.8 μs

Append 占 TPOT = 177.6～215.8 μs / 11,970 μs
               ≈ 1.48%～1.80%
```

因此：

- 即使把原生 Append 耗时完全降为零，TPOT 理论改善上限也只有 1.48%～1.80%；
- 使用当前自定义实现后，预计只节省约 0.34%～0.61% TPOT；
- 端到端三轮中的吞吐配对变化范围为 -3.71%～+5.36%，明显大于预计收益；
- TTFT 中 Append 占比只有约 0.15%，所以三轮 TTFT 的较大正负变化不可能主要由 Append 引起。

## 4. 为什么端到端方向不稳定

端到端 TPOT 和吞吐还包含 Attention、MLP、其他 CUDA Kernel、调度、Python/C++ Dispatcher、请求批处理和 GPU 频率波动。Append 预计带来的 0.34%～0.61% 改善小于这些因素的轮次变化，因此三轮中会出现一轮正收益、两轮负收益或相反情况。

这不是“12 倍优化失效”，而是测量边界发生了变化：

```text
11.98×：V3 / 朴素 V0，2,048 Token，只测一个 CUDA 算子

0.34%～0.61%：Custom / 原生 vLLM，1～2 Token 为主，测完整 Decode
```

两组数字回答不同问题，不能互相换算。

## 5. 面试表述

> 我最初在 2,048 Token 微基准中将 Append 相对朴素 V0 优化了 11.98 倍，但接入 vLLM 后没有观察到稳定的端到端加速。我进一步分析发现，11.98 倍的比较对象不是 vLLM 原生优化算子，而且真实 Decode 中 88% 的 Append 调用只处理 1～2 个 Token。三次独立测量显示，原生 Append 乘以 24 层后只占 TPOT 的约 1.48%～1.80%，当前自定义实现预计只能改善 0.34%～0.61% TPOT，小于端到端约数个百分点的轮次波动。因此根因是基线和工作负载不等价，以及 Amdahl 定律限制，而不是简单认为 Kernel 微基准加速可以线性换算成系统吞吐。

## 6. 复现与证据边界

原始结果：

- `results/system_benchmarks/2026-09-02-append-attribution/append_shape_benchmark.json`
- `results/system_benchmarks/2026-09-02-append-attribution/append_shape_benchmark_repeat2.json`
- `results/system_benchmarks/2026-09-02-append-attribution/append_shape_benchmark_repeat3.json`
- `results/system_benchmarks/2026-09-02-append-attribution/summary.json`

计时方法：CUDA Event 测量重复调用的 Device Timeline，`perf_counter` 同时记录 Host Enqueue 和最终同步墙钟时间。三个独立进程，每个进程七轮，并轮换原生/自定义执行顺序。

三次运行时的 Git HEAD 为 `7917d9a2accc354de17d7858782cd30722887d21`，工作区因新增 M1 脚本而为 dirty；`summary.json` 记录了实际脚本与 Adapter 的 SHA-256，后续归档提交不冒充运行时的干净提交。

本次尝试使用 Nsight Systems 采集时间线，但当前 WSL 会话中的 `nsys profile` 返回 `Connection refused`，因此没有生成或宣称 Nsight Systems 时间线证据。M1 的定量结论来自 CUDA Event、墙钟计时、真实验证日志和已冻结端到端结果。
