# AgentInferLab 固定并发流式压测设计

日期：2026-07-27

## 1. 背景

项目已经具备两个基础模块：

- `workloads.py`：生成可复现的抽象工作负载；
- `metrics.py`：根据请求时间记录计算 TTFT、TPOT、E2E、吞吐量、P50 和 P99。

当前缺少的是连接这两个模块与真实 vLLM 服务的执行链路。本阶段将实现 Prompt 构造、vLLM 流式客户端和固定并发调度，使项目可以对本机 vLLM 服务执行可复现的端到端压测。

## 2. 目标

本阶段完成以下闭环：

```text
WorkloadConfig
→ generate_workload()
→ RequestSpec
→ 精确 Token Prompt 构造
→ 固定并发发送流式请求
→ RequestTrace
→ summarize_metrics()
→ MetricsSummary
```

完成后，用户可以指定请求数量、并发数、输入长度、输出长度、共享前缀比例和随机种子，对 vLLM 的 `/v1/completions` 接口执行真实压测，并得到可复现的推理性能指标。

## 3. 设计原则

### 3.1 精确而非估算

输入长度以 Token 数为准，不用字符数近似。客户端先调用 vLLM `/tokenize` 获取 Token ID，再直接向 `/v1/completions` 发送 Token ID 列表，避免文本解码后重新分词造成长度漂移。

### 3.2 校准与计时分离

Prompt 校准、Token 池创建和模型查询都在压测计时区间之外完成。计时只覆盖单次 HTTP 推理请求从发送到流式响应结束的过程。

### 3.3 组件职责单一

- Prompt 模块只负责生成精确输入；
- vLLM 客户端只负责 HTTP 通信和时间采集；
- Benchmark 模块只负责并发调度和指标汇总；
- 已有 Metrics 模块继续负责全部指标公式。

### 3.4 CPU 测试不依赖在线服务

单元测试通过伪造 HTTP 响应和时钟运行，不要求 GitHub Actions 具有 GPU、CUDA、PyTorch、vLLM 或网络访问。

## 4. 技术选择

### 4.1 服务接口

使用：

- `POST /tokenize`：把确定性的种子文本转换为 Token ID；
- `POST /v1/completions`：直接发送 Token ID，并接收流式生成结果。

第一版不使用 `/v1/chat/completions`。Completion 接口没有聊天模板引入的额外 Token，更适合精确控制输入长度。

### 4.2 Python 标准库

HTTP 通信使用 `http.client`，并发使用 `concurrent.futures.ThreadPoolExecutor`，不增加第三方 HTTP 依赖。

选择线程池的原因是 HTTP 请求属于 I/O 等待任务。线程等待网络响应时，其他线程仍可继续工作；第一版无需引入异步框架。

### 4.3 压测模型

使用固定并发、闭环调度：

- 线程池大小等于 `concurrency`；
- 同时最多运行 `concurrency` 个请求；
- 一个请求完成后，空闲线程领取下一个请求；
- 所有请求完成后统一汇总。

## 5. 模块设计

### 5.1 `src/agent_infer_lab/prompting.py`

#### 职责

- 调用客户端的 Tokenize 能力；
- 从确定性的 Agent 类种子文本创建 Token 池；
- 按 `RequestSpec` 构造长度精确的输入 Token ID；
- 保证共享前缀和独立后缀符合工作负载配置。

#### `PreparedRequest`

保存一次可以直接发送给 vLLM 的请求：

- `request_id`：唯一请求编号；
- `prompt_token_ids`：完整输入 Token ID；
- `output_tokens`：期望生成的最大 Token 数。

该对象不可修改，避免准备完成后请求内容被意外改变。

#### Token 池

使用项目内置的确定性 Agent 类文本，例如工具调用说明、任务背景和结构化上下文。调用 `/tokenize` 后得到基础 Token 序列。

当基础序列短于所需长度时，按顺序循环扩展；当长于所需长度时截断。该过程不能使用 Python 全局随机状态。

#### 共享前缀

对同一批请求：

```text
shared_tokens = input_tokens × shared_prefix_ratio
unique_tokens = input_tokens - shared_tokens
```

- 前 `shared_tokens` 个 Token 完全一致；
- 后 `unique_tokens` 个 Token 根据请求编号和固定随机种子选择确定性偏移；
- 每条请求最终长度必须严格等于 `input_tokens`。

### 5.2 `src/agent_infer_lab/vllm_client.py`

#### 职责

- 封装 vLLM 服务地址、模型名称和超时；
- 调用 `/tokenize`；
- 调用 `/v1/completions`；
- 解析 Server-Sent Events 流；
- 采集三个时间点；
- 返回现有 `RequestTrace`。

#### Completion 请求

请求体至少包含：

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "prompt": [1, 2, 3],
  "max_tokens": 32,
  "temperature": 0,
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

真实的 `prompt` 数组由 Prompt 模块生成。

#### 流式响应解析

客户端逐行读取响应：

- 忽略空行；
- 只处理以 `data:` 开头的 SSE 数据；
- 第一次看到包含非空生成内容的事件时记录 `first_token_at`；
- 从最终 usage 中读取实际生成 Token 数；
- 收到 `[DONE]` 时记录 `completed_at`。

第一 Token 时间不能在仅包含角色、元数据或空文本的事件上记录。

#### 计时

统一使用 `time.perf_counter()`：

- `started_at`：请求准备完成后、发送 HTTP 请求前；
- `first_token_at`：收到第一段非空生成内容时；
- `completed_at`：收到 `[DONE]` 时。

采集结果交给 `RequestTrace` 校验，不在客户端重复实现指标公式。

### 5.3 `src/agent_infer_lab/benchmark.py`

#### 职责

- 接收准备好的请求；
- 创建固定大小线程池；
- 并发调用 vLLM 客户端；
- 收集全部成功请求的 `RequestTrace`；
- 调用 `summarize_metrics()`；
- 返回 `MetricsSummary`。

#### 调度语义

例如总请求数为 20、并发数为 4：

- 初始提交 4 个请求；
- 任一请求完成后继续执行下一条；
- 任意时刻最多有 4 条请求运行；
- 20 条请求全部成功后汇总。

第一版遇到任何请求失败时终止本次实验，不把不完整批次伪装成成功结果。

### 5.4 命令行入口

提供 `agent-infer-bench` 命令，主要参数包括：

- `--base-url`：vLLM 服务地址；
- `--model`：模型名称；
- `--requests`：请求总数；
- `--concurrency`：固定并发数；
- `--input-tokens`：输入 Token 数；
- `--output-tokens`：最大输出 Token 数；
- `--shared-prefix-ratio`：共享前缀比例；
- `--seed`：固定随机种子；
- `--timeout`：单请求超时。

第一版将汇总结果打印到终端，不实现文件持久化和图表。

## 6. 数据流

```text
用户命令行参数
    ↓
WorkloadConfig
    ↓
generate_workload()
    ↓
RequestSpec[]
    ↓
PromptBuilder + /tokenize
    ↓
PreparedRequest[]
    ↓
ThreadPoolExecutor
    ↓
VllmClient.stream_completion()
    ↓
RequestTrace[]
    ↓
summarize_metrics()
    ↓
MetricsSummary
```

## 7. 错误处理

以下情况产生明确异常并终止本次实验：

- vLLM 服务无法连接；
- `/tokenize` 或 `/v1/completions` 返回非 200 状态；
- 服务返回的 JSON 或 SSE 格式无效；
- 模型名称不存在；
- Prompt 实际 Token 数与目标不一致；
- 流结束前未收到 `[DONE]`；
- 请求完成但没有收到非空生成 Token；
- 最终响应缺少可用的输出 Token 数；
- 并发数、请求数、超时或 Token 长度不合法。

错误信息应包含请求编号、接口地址、HTTP 状态和直接原因。第一版不自动重试，以免重试掩盖服务不稳定并污染延迟数据。

## 8. 测试策略

### 8.1 Prompt 测试

- 输入 Token 长度精确；
- 相同 seed 生成完全相同的请求；
- 不同请求共享正确长度的前缀；
- 独立后缀存在差异；
- 构造过程不改变 Python 全局随机状态；
- 非法 Token 长度和共享比例被拒绝。

### 8.2 vLLM 客户端测试

使用伪造连接、响应和时钟验证：

- `/tokenize` 请求体与结果解析；
- Completion 请求体包含 Token ID、`stream=true` 和确定性参数；
- 正确忽略 SSE 空行；
- 第一段非空内容触发 `first_token_at`；
- `[DONE]` 触发完成时间；
- 实际输出 Token 数进入 `RequestTrace`；
- HTTP 错误、无首 Token、无 `[DONE]` 和超时得到明确异常。

### 8.3 Benchmark 测试

- 请求总数正确；
- 同时运行数量不超过 concurrency；
- 每条请求只执行一次；
- 所有 Trace 被交给现有汇总函数；
- 任一请求失败时批次失败；
- 相同配置保持确定性。

### 8.4 验证命令

```text
pytest tests/test_prompting.py -q
pytest tests/test_vllm_client.py -q
pytest tests/test_benchmark.py -q
pytest -q
ruff check .
uv lock --check
```

## 9. 实机验收

使用本机 8 GiB GPU 和已部署的 `Qwen/Qwen2.5-0.5B-Instruct` 完成：

1. `/v1/models` 可访问；
2. `/tokenize` 可返回 Token ID；
3. 单请求流式生成成功；
4. 固定并发批次运行成功；
5. 无 OOM；
6. 输出 TTFT、TPOT、E2E、输出吞吐量、P50 和 P99；
7. 同一配置可以重复执行并保留配置记录。

## 10. 第一版明确不做

- `/v1/chat/completions`；
- 泊松到达或开放环流量；
- 自动重试；
- Goodput 和 SLO 达标率；
- GPU 显存或利用率采样；
- 失败请求统计；
- JSON、CSV 或数据库持久化；
- 图表和 Web 页面；
- CUDA 算子。

上述能力在真实压测闭环稳定后按收益排序加入。

## 11. 成功标准

- 精确 Token Prompt 可以稳定构造；
- 真实 vLLM 流式请求可以运行；
- 固定并发调度符合配置；
- 每个成功请求产生合法 `RequestTrace`；
- 已有 Metrics 模块得到真实 TTFT、TPOT、E2E、吞吐量、P50 和 P99；
- CPU 单元测试不依赖 GPU 或在线 vLLM；
- 全量 pytest、Ruff 和依赖锁检查通过；
- 本机 8 GiB GPU 实验无 OOM；
- 设计边界与实现保持一致。
