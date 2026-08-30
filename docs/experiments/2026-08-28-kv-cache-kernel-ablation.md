# Paged KV Cache Kernel 逐项消融与 Nsight Compute 分析

日期：2026-08-28

## 1. 实验目标

在同一个 PyTorch CUDA 扩展、相同输入数据、相同缓存布局和相同输出分配方式下，对 Paged KV Cache Append/Gather 的 V0、V1、V2、V3 四个版本进行逐项消融，并用两套证据回答不同问题：

- CUDA Event 多轮微基准：判断各版本是否真的更快，报告 P50/P95/P99 和逻辑有效带宽。
- Nsight Compute：解释并行度、显存利用率、Occupancy、寄存器和线程布局的变化。

PyTorch 参考实现只作为正确性标准答案，不作为性能基线；Nsight 诊断期间的单次 Duration 不替代 CUDA Event 正式结果。

## 2. Kernel 版本与归因边界

| 版本 | 实现 | 相邻版本比较的含义 |
|---|---|---|
| V0 baseline | 一个线程负责一个 Token，通过循环逐元素搬运该 Token 的 Key 和 Value | 朴素 Kernel 基线 |
| V1 coalesced | 相邻线程处理同一 Token 的相邻元素 | V0→V1 是细粒度线程并行与合并访存的组合收益 |
| V2 vectorized | 在 V1 的数据映射上，每个线程用一个 16 字节 `uint4` 搬运连续 8 个 FP16，保留标量尾部路径 | V1→V2 是向量化与减少活跃线程、地址计算和访存指令的增量收益 |
| V3 thread layout | 保留 V2 的 16 字节搬运，调整 Warp、lane group 与 Token 的映射 | V2→V3 是线程布局变化的增量收益 |

V0→V1 同时改变并行粒度和访存映射，因此不能把全部收益只称为“合并访存收益”。

## 3. 实验环境

- GPU：NVIDIA GeForce RTX 4060，8585216000 bytes（约 8 GiB）。
- 系统：Windows + WSL2，Linux 6.18.33.2。
- Python：3.12.3。
- PyTorch：2.11.0+cu129。
- CUDA Toolkit：12.9，NVCC 12.9.86。
- NVIDIA Driver：581.57。
- Compute Capability：8.9。
- Nsight Compute：2025.2.1。
- 微基准代码提交：`51065fa`。
- Nsight 诊断代码状态：`dff708802ec68adb7ac039d3909c897a58b354fb`，工作区干净。

关键环境变量：

```text
CUDA_HOME=/usr/local/cuda-12.9
CUDACXX=/usr/local/cuda-12.9/bin/nvcc
TORCH_EXTENSIONS_DIR=/mnt/d/agent-infer-lab/.torch_extensions
TORCH_CUDA_ARCH_LIST=8.9
MAX_JOBS=2
```

## 4. 正确性保障

四个 CUDA 版本均与 PyTorch correctness oracle 逐元素比较，并覆盖：

- 主形状；
- 奇数 Head/Head Dim；
- 空输入；
- 多轮迭代；
- Append 唯一 slot；
- Gather 重排和重复 slot；
- 无检查 Append 微基准接口；
- 预分配 Gather 输出接口及指针保持。

验证输出：

```text
implementations: PyTorch, V0, V1, V2, V3
shapes: main, odd-sized, multi-iteration, empty
Append: unique slots
Gather: reordered and duplicate slots
```

Python CPU 测试同时保持 `101 passed`。

## 5. CUDA Event 正式测量方法

三种形状均采用：

```text
num_tokens: 2048
num_blocks: 256
block_size: 16
dtype: FP16
warmup: 120
repeats: 1008
banks: 16
seed: 20260827
version_orders: V0/V1/V2/V3 的全部 24 种排列
independent_runs: 3
```

Gather 输出在计时前预分配；计时范围只包含无检查 CUDA 调用，排除输入验证和输出分配。表中数值为三轮独立实验各自 P50 的中位数，单位为微秒。

## 6. CUDA Event 逐项消融结果

### 6.1 对齐形状：2 KV Heads × Head Dim 64

| 操作 | V0/μs | V1/μs | V2/μs | V3/μs | V1/V0 | V2/V1 | V3/V2 | V3/V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Append | 62.464 | 7.168 | 5.504 | 5.216 | 8.71× | 1.30× | 1.055× | 11.98× |
| Gather | 64.512 | 8.192 | 5.472 | 5.120 | 7.88× | 1.50× | 1.069× | 12.60× |

该形状每个 Token 有 128 个 FP16 元素，可组成 16 个完整的 16 字节向量块。V2 相对 V1 的向量化收益明确；V3 只带来约 5% 至 7% 的小幅增量。

V3 的逻辑有效带宽约为：

- Append：402.1 GB/s。
- Gather：409.6 GB/s。

这里的“逻辑有效带宽”按 Key/Value 有效读写字节数除以 CUDA Event P50 计算，不包含元数据流量，也不是 Nsight 测得的真实 DRAM 带宽。

### 6.2 非对齐压力形状：3 KV Heads × Head Dim 65

| 操作 | V0/μs | V1/μs | V2/μs | V3/μs | 最快版本 | V3/V2 |
|---|---:|---:|---:|---:|---|---:|
| Append | 93.184 | 10.240 | 31.744 | 29.696 | V1 | 1.069× |
| Gather | 87.040 | 10.240 | 28.672 | 27.648 | V1 | 1.037× |

该形状每个 Token 的 FP16 数据跨度为 `3 × 65 × 2 = 390` bytes，`390 mod 16 = 6`。后续 Token 的起始地址通常不满足 16 字节对齐，V2/V3 大量进入标量回退路径，因此 V2/V3 明显慢于 V1。该结果证明向量化收益具有对齐前提。

### 6.3 大对齐形状：4 KV Heads × Head Dim 128

| 操作 | V0/μs | V1/μs | V2/μs | V3/μs | V2/V1 | V3/V2 | V3/V0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Append | 231.424 | 18.432 | 10.240 | 10.240 | 1.80× | 1.00× | 22.60× |
| Gather | 244.736 | 19.456 | 10.240 | 10.240 | 1.90× | 1.00× | 23.90× |

该形状有 64 个完整向量块，V2 向量化收益明显；V3 的线程布局没有继续降低 P50，说明 V3 不是所有形状下都优于 V2。

## 7. Nsight Compute 分析方法

Nsight 使用 2048 Token、2 Heads × 64 Dim 代表形状：

- V0/V1：采集 SpeedOfLight、LaunchStats、Occupancy、InstructionStats、MemoryWorkloadAnalysis、SchedulerStats 和 WarpStateStats，共 34 个 replay pass。
- V2/V3：采集 `--set basic`，共 8 个 replay pass。
- 每次只通过 Kernel 名称捕获一个目标 Kernel，避免把其他版本混入硬件指标。

诊断调用示例：

```bash
ncu \
  --target-processes all \
  --kernel-name "regex:append_kv_cache_v0_kernel" \
  --launch-count 1 \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section InstructionStats \
  --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Tables \
  --section SchedulerStats \
  --section WarpStateStats \
  --export <report-prefix> \
  python cuda/benchmark_kv_cache.py <fixed-workload-arguments>
```

## 8. Nsight：V0→V1 组合收益

### 8.1 Append

| 指标 | V0 | V1 |
|---|---:|---:|
| Grid Size | 16 | 1024 |
| Block Size | 128 | 256 |
| Waves Per SM | 0.06 | 7.11 |
| Registers Per Thread | 28 | 16 |
| Achieved Occupancy | 7.91% | 83.67% |
| DRAM Throughput | 9.36% | 58.47% |
| Memory Throughput | 24.66 GB/s | 153.10 GB/s |
| Issued Warp Per Scheduler | 0.01 | 0.36 |
| Warp Cycles Per Issued Instruction | 114.03 | 27.94 |

### 8.2 Gather

| 指标 | V0 | V1 |
|---|---:|---:|
| Grid Size | 16 | 1024 |
| Block Size | 128 | 256 |
| Waves Per SM | 0.06 | 7.11 |
| Registers Per Thread | 28 | 16 |
| Achieved Occupancy | 7.88% | 83.40% |
| DRAM Throughput | 3.81% | 51.20% |
| Memory Throughput | 10.05 GB/s | 134.31 GB/s |
| Issued Warp Per Scheduler | 0.01 | 0.29 |
| Warp Cycles Per Issued Instruction | 109.42 | 34.60 |

V0 的 Grid 只有 16 个 Block，少于 RTX 4060 的 24 个 SM，无法覆盖整块 GPU。V1 把 Token 内元素拆给大量线程，Grid 增至 1024，显著提高 Occupancy、可发射 Warp 和真实内存吞吐。V1 的总执行指令数高于 V0，因为启动了更多线程；收益来自减少单线程串行循环并提高并行执行能力，而不是减少全局总指令数。

V0 的 L1/TEX Hit Rate 虽然超过 90%，但吞吐和 Occupancy 很低。高命中率不能单独证明更快；V1 以更低命中率换取了更高的并行流式搬运吞吐。

## 9. Nsight：V2→V3 线程布局取舍

### 9.1 Append

| 指标 | V2 | V3 |
|---|---:|---:|
| Grid Size | 128 | 256 |
| Block Size | 256 | 128 |
| Waves Per SM | 0.89 | 1.07 |
| Registers Per Thread | 34 | 48 |
| Theoretical Occupancy | 100% | 83.33% |
| Achieved Occupancy | 64.88% | 59.27% |
| DRAM Throughput | 70.20% | 68.84% |

### 9.2 Gather

| 指标 | V2 | V3 |
|---|---:|---:|
| Grid Size | 128 | 256 |
| Block Size | 256 | 128 |
| Waves Per SM | 0.89 | 1.07 |
| Registers Per Thread | 34 | 48 |
| Theoretical Occupancy | 100% | 83.33% |
| Achieved Occupancy | 66.62% | 58.36% |
| DRAM Throughput | 59.85% | 64.27% |

V3 将 256 线程 Block 拆为 128 线程 Block，在总线程数不变的情况下将 Grid 翻倍，Waves Per SM 从 0.89 提高到 1.07；但更复杂的 lane group 与 Token 映射使每线程寄存器从 34 增至 48，理论和实际 Occupancy 均下降。Gather 的 DRAM 利用率有所提高，Append 基本持平略降。因此 V3 的 5% 至 7% P50 改善是有限且形状相关的布局收益，不应描述为普遍的 Occupancy 或带宽提升。

Nsight 诊断中的单次 Duration 来自不同进程且包含 replay 和计数器扰动，不用于覆盖 CUDA Event 三轮 P50 结论。

## 10. 逐项归因结论

1. **V0→V1**：最大收益来自把每 Token 的单线程串行循环改成元素级并行，并使相邻线程访问相邻元素。Nsight 显示 Grid、Occupancy、可发射 Warp 和内存吞吐同时大幅提高。
2. **V1→V2**：对齐形状下，16 字节向量化为 Append 带来约 1.30×、为 Gather 带来约 1.50×增量；非对齐形状因标量回退而退化。
3. **V2→V3**：2×64 形状下带来约 5% 至 7%小幅收益；4×128 形状下无可测收益；线程布局改善与寄存器压力相互抵消。
4. **最终选择不能只按版本号**：非对齐形状应优先 V1；大对齐形状 V2 已达到最佳 P50；V3 只在部分对齐形状取得小幅优势。

## 11. 原始证据

CUDA Event正式原始样本：

```text
results/kernel_benchmarks/v0_v1_v2_v3_h2_d64_clean_repeat{1,2,3}_2026-08-27.json
results/kernel_benchmarks/v0_v1_v2_v3_h3_d65_clean_repeat{1,2,3}_2026-08-27.json
results/kernel_benchmarks/v0_v1_v2_v3_h4_d128_clean_repeat{1,2,3}_2026-08-27.json
```

Nsight Compute原始报告、文本导出和诊断JSON：

```text
results/nsight_compute/
```

`results/nsight_compute/SHA256SUMS.txt`保存全部24份原始文件的SHA-256校验值。二进制 `.ncu-rep` 用于在 Nsight Compute 中重新打开，`.txt` 用于 Git 审阅，`.json`保存环境和输入配置。

## 12. 实验边界

- 结论仅适用于当前 RTX 4060、CUDA/PyTorch版本、缓存布局和测试形状。
- 逻辑有效带宽不等于真实DRAM带宽；硬件带宽利用率只引用Nsight指标。
- V0→V1是并行粒度和访存映射的组合变化，不单独声称“纯Coalescing收益”。
- V2要求16字节对齐，非对齐形状的标量回退成本必须单独报告。
- V3不是所有形状下的最优版本，后续生产接入应采用形状感知的选择策略。
- 本报告是Kernel级结果，不能直接表述为vLLM端到端TPOT、TTFT或吞吐收益。

## 13. 可用于项目介绍的结论

> 在同一PyTorch CUDA扩展中实现Paged KV Cache Append/Gather的V0至V3四级消融，并以PyTorch作为正确性标准答案。对2048 Token、2×64对齐形状进行三轮独立CUDA Event实验，每轮包含120次预热、1008次采样和24种版本顺序：V1的线程级并行与合并访问使Append/Gather相对V0分别达到8.71×和7.88×；V2的16字节向量化进一步达到1.30×和1.50×；V3线程布局再获得约1.055×和1.069×。Nsight显示V0→V1时Append实际内存吞吐由24.66提升至153.10 GB/s、Gather由10.05提升至134.31 GB/s，Achieved Occupancy由约7.9%提高至约83%。同时，非对齐3×65形状中V2/V3因标量回退慢于V1，4×128形状中V3与V2持平，证明优化收益具有明确的形状和对齐边界。
