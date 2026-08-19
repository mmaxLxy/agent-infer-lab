# AgentInferLab

AgentInferLab 是一个面向 Agent 长上下文负载的可复现大模型推理性能实验平台。项目将证据链拆分为两个边界：端到端实验只在同一版本 vanilla vLLM 内进行系统配置消融，Kernel 实验只在同一 CUDA 扩展内以朴素 Kernel 为基线进行逐项消融。PyTorch 参考实现仅作为 correctness oracle（正确性标准答案），不用于证明 vLLM 或自定义 Kernel 的增量性能收益。

## 项目亮点

- **可复现工作负载**：使用固定随机种子生成输入/输出长度和共享前缀比例可控的请求，通过 vLLM `/tokenize` 校准 Prompt Token 数。
- **流式推理压测**：通过 OpenAI 兼容 `/v1/completions` SSE 接口执行固定并发请求，记录请求提交、首 Token 返回和请求完成时间。
- **完整指标体系**：统一计算 TTFT、TPOT、E2E、输出吞吐、P50/P99 和请求成功率，并区分成功请求指标与失败类型。
- **Prefix Caching 实验**：分析共享前缀、上下文长度和并发度对 Prefill、缓存命中率、首 Token 延迟及吞吐的影响。
- **CUDA KV Cache 算子**：使用 C++/CUDA 实现分页 KV Cache Append/Gather；PyTorch 参考实现只负责逐元素正确性验证，性能比较统一采用同一扩展内的朴素 Kernel。
- **性能分析与工程保障**：使用 CUDA Event 测量 Kernel 延迟与有效带宽，结合 Nsight 定位访存瓶颈；通过依赖锁定、环境检测、单元测试、Ruff 和 CPU CI 保证结果可复现。

## 系统架构

```text
WorkloadConfig
      │
      ▼
确定性 RequestSpec ──► /tokenize 校准 ──► PreparedRequest
                                              │
                                              ▼
                                     固定并发请求调度器
                                              │
                                              ▼
                                  vLLM /v1/completions
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
              流式 RequestTrace                         分页 KV Cache 数据路径
                       │                                             │
                       ▼                                             ▼
          TTFT / TPOT / E2E / 吞吐                     Append / Gather CUDA Kernel
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              ▼
                              CUDA Event / Nsight / 端到端消融
```

## 实验环境

| 组件 | 配置 |
| --- | --- |
| 操作系统 | Windows + WSL2 Ubuntu 24.04 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8 GiB |
| CUDA Toolkit | 12.9 |
| Python | 3.12 |
| PyTorch | 2.11.0+cu129 |
| vLLM | 0.23.0+cu129 |
| 模型 | Qwen2.5-0.5B-Instruct |

完整环境版本与检测方法见 [`docs/environment.md`](docs/environment.md)。

## 实验证据

### 同版本 vLLM：Prefix Caching 系统消融

现有可追踪实验只比较 vLLM 0.23.0 在 Prefix Cache 关闭和开启时的系统表现。模型、请求数、并发度、输出长度、共享前缀比例和随机种子保持一致；该结果用于说明 Prefix Caching 的系统级收益，不用于证明任何自定义 CUDA Kernel 的收益。

| 输入长度 | Cache 开启累计命中率 | 输出吞吐提升 | TTFT P99 降低 | E2E P99 降低 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 72.6% | 11.9% | 40.3% | 8.0% |
| 1024 | 72.8% | 24.3% | 63.9% | 16.2% |
| 1536 | 72.8% | 38.5% | 50.6% | 28.2% |

每个配置当前只有一轮正式结果，因此定位为工程阶段基准，不外推为其他 GPU、模型或负载的通用结论。完整参数、筛选规则和原始数值见 [Prefix Cache 实验报告](docs/experiments/2026-08-10-prefix-cache.md)。

### 同一 CUDA 扩展：Kernel 逐项消融

Kernel 性能归因统一使用同一扩展、相同输入张量、相同 CUDA Stream 和相同编译选项，按以下顺序逐项增加变量：

```text
朴素 Kernel
→ 合并访存
→ 向量化
→ 线程布局
→ 地址计算或主形状特化
```

Kernel 层单独报告 CUDA Event 延迟、有效显存带宽、相邻版本增量收益和相对朴素基线的累计收益。PyTorch 参考实现只使用 `rtol=0、atol=0` 验证 FP16 数据复制是否逐元素一致，不作为性能加速比的分母。

### 证据边界

- 不使用 PyTorch 与 vLLM 的跨系统差异证明 Kernel 收益。
- 不把 Kernel 微基准加速比换算成 TTFT、TPOT 或服务吞吐收益。
- 只有在同一版本 vLLM 中仅替换目标 Kernel 后，才报告自定义 Kernel 的端到端增量。
- 没有原始样本、环境清单和完整命令支持的数字，不作为项目结论或简历结果。

## CUDA 算子设计

### Append

Append 将 Decode 阶段新产生的 Key/Value 写入分页缓存：

```text
for token in tokens:
    slot = slot_mapping[token]
    key_cache[slot] = keys[token]
    value_cache[slot] = values[token]
```

### Gather

Gather 根据逻辑顺序从离散物理槽位读取 Key/Value：

```text
for token in tokens:
    slot = slot_mapping[token]
    gathered_keys[token] = key_cache[slot]
    gathered_values[token] = value_cache[slot]
```

真实 Qwen2.5-0.5B 主形状为 `num_kv_heads=2、head_dim=64、block_size=16、dtype=float16`。优化过程中让相邻线程访问同一 Token 内连续的 Head 数据，并在满足对齐条件时使用 16 字节向量化访存，以提高全局内存访问合并度和有效带宽。

## 指标定义

```text
TTFT = first_token_at - started_at
TPOT = (completed_at - first_token_at) / (output_tokens - 1)
E2E  = completed_at - started_at
Output Throughput = 总输出 Token / 整批实验持续时间
Success Rate = 成功请求数 / 总请求数
```

P50 和 P99 使用 nearest-rank 方法。单 Token 请求没有后续 Token 间隔，因此 TPOT 记为 `None`。请求失败不会让整批实验提前丢失结果，失败类型单独汇总，延迟指标只基于成功请求计算。

## 安装

使用 uv 创建项目环境：

```bash
cd /mnt/d/agent-infer-lab
UV_LINK_MODE=copy /home/ayax/.local/bin/uv sync --no-dev --python /usr/bin/python3
```

开发环境检查：

```bash
source /home/ayax/.venvs/agent-infer-lab-dev/bin/activate
pytest -q
ruff check .
/home/ayax/.local/bin/uv lock --check
```

## 启动 vLLM

```bash
source /home/ayax/.venvs/agent-infer-lab/bin/activate

export CUDA_HOME=/usr/local/cuda-12.9
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096 \
  --enable-prefix-caching
```

服务就绪后，`http://127.0.0.1:8000/v1/models` 应返回模型信息。

## 运行推理基准

```bash
cd /mnt/d/agent-infer-lab

.venv/bin/agent-infer-bench \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 80 \
  --concurrency 8 \
  --input-tokens 2048 \
  --output-tokens 256 \
  --shared-prefix-ratio 0.75 \
  --seed 20260810
```

相同实验应固定模型、输入/输出长度、共享前缀比例、并发度和随机种子，并分别完成预热与多轮重复测量。

## 可复现性与验证

- 固定随机种子和精确 Token 长度，保证不同配置使用相同工作负载。
- 保存环境版本、实验参数、原始请求时间线和聚合结果。
- Prefix Cache 开启/关闭实验使用相同请求和模型配置。
- CUDA 微基准使用 CUDA Event，避免用 Python 墙钟误测异步 Kernel。
- Append/Gather 同时验证 Key 和 Value，并覆盖 Block 边界、非连续槽位、首尾槽位和非法输入。
- 使用 Nsight Systems 检查端到端时间线，使用 Nsight Compute 分析 DRAM 吞吐、全局访存效率、Warp 执行效率和 Occupancy。
- GitHub CPU CI 不依赖 GPU、CUDA、PyTorch 或 vLLM，本地 GPU 验收独立执行。

## 相关文档

- [环境记录](docs/environment.md)
- [Prefix Cache 实验报告](docs/experiments/2026-08-10-prefix-cache.md)
- [估算性能目标参考（非实测）](docs/experiments/estimated-performance-reference.md)
