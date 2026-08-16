# AgentInferLab

AgentInferLab 是一个面向 Agent 场景的可复现 LLM 推理基准项目。它生成长度精确、共享前缀可控的 Token 工作负载，通过 vLLM OpenAI 兼容接口执行流式固定并发请求，并汇总 TTFT、TPOT、端到端延迟、P50/P99 和输出吞吐量。

## 已实现能力

- 固定随机种子的可复现工作负载；
- 通过 vLLM `/tokenize` 构造精确长度 Token Prompt；
- 可配置共享前缀，为 Prefix Cache 实验提供输入；
- 基于 Python 标准库的 `/v1/completions` SSE 流式客户端；
- 固定并发、闭环请求调度；
- TTFT、TPOT、E2E、输出吞吐量和 P50/P99 汇总；
- 不依赖 GPU、CUDA、PyTorch 或 vLLM 的 CPU 单元测试与 CI。

## 数据流

```text
WorkloadConfig
→ RequestSpec
→ /tokenize
→ PreparedRequest
→ ThreadPoolExecutor
→ /v1/completions
→ RequestTrace
→ MetricsSummary
```

## 环境

- Windows + WSL2 Ubuntu 24.04；
- NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB；
- CUDA Toolkit 12.9；
- Python 3.12；
- PyTorch 2.11.0+cu129；
- vLLM 0.23.0+cu129；
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`。

完整版本信息见 [`docs/environment.md`](docs/environment.md)。

## 安装

项目不包含第三方运行时 Python 依赖。使用 uv 在项目目录创建本地环境：

```bash
cd /mnt/d/agent-infer-lab
UV_LINK_MODE=copy /home/ayax/.local/bin/uv sync --no-dev --python /usr/bin/python3
```

开发检查使用已有 CPU 开发环境：

```bash
source /home/ayax/.venvs/agent-infer-lab-dev/bin/activate
pytest -q
ruff check .
/home/ayax/.local/bin/uv lock --check
```

## 启动 vLLM

```bash
source /home/ayax/.venvs/agent-infer-lab/bin/activate
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export CUDA_HOME=/usr/local/cuda-12.9

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 2048
```

服务就绪后，`http://127.0.0.1:8000/v1/models` 应返回模型信息。

## 运行基准

```bash
cd /mnt/d/agent-infer-lab

.venv/bin/agent-infer-bench \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 20 \
  --concurrency 4 \
  --input-tokens 128 \
  --output-tokens 32 \
  --shared-prefix-ratio 0.5 \
  --seed 20260809
```

示例输出：

```text
requests: 20
output_tokens: 640
duration_seconds: 1.036271
output_throughput_tokens_per_second: 617.599136
ttft_p50_seconds: 0.019865
ttft_p99_seconds: 0.100700
tpot_p50_seconds: 0.005431
tpot_p99_seconds: 0.006103
e2e_p50_seconds: 0.191964
e2e_p99_seconds: 0.267913
```

该结果来自单次本地稳定态实验，用于验证基准链路，不代表跨硬件的通用性能结论。

## 指标定义

```text
TTFT = first_token_at - started_at
TPOT = (completed_at - first_token_at) / (output_tokens - 1)
E2E  = completed_at - started_at
Output Throughput = 总输出Token / 整批实验持续时间
```

P50 和 P99 使用 nearest-rank 方法。单 Token 请求没有后续 Token 间隔，因此 TPOT 为 `None`。

## 当前边界

- 当前实现固定并发闭环流量，不包含泊松到达；
- 请求失败时终止批次，不进行自动重试；
- 未实现 Goodput、SLO 达标率、GPU利用率采样或结果持久化；
- WSL 中 `pin_memory=False`，结果不能直接等同于原生 Linux 服务器；
- CUDA Kernel 优化属于下一阶段，当前项目先提供可信的优化前基线。
