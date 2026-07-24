# 推理性能指标模块设计

日期：2026-07-22

## 1. 目标

实现一个纯 CPU、可独立测试的推理性能指标模块。

模块接收请求的原始时间记录，计算单请求指标和整批请求的汇总指标。第一版不发送 HTTP 请求，不依赖 GPU、CUDA、PyTorch 或 vLLM。

## 2. 核心数据结构

### RequestTrace

保存一次成功请求的原始数据：

- `request_id`：请求的唯一编号；
- `started_at`：请求开始时间；
- `first_token_at`：第一个输出 Token 到达时间；
- `completed_at`：完整响应结束时间；
- `output_tokens`：实际生成的 Token 数量。

该数据结构不可修改，防止采集完成后时间记录被意外改变。

### RequestMetrics

保存单个请求计算后的指标：

- `request_id`；
- `ttft_seconds`；
- `tpot_seconds`；
- `e2e_seconds`。

当请求只生成一个 Token 时，没有后续 Token 间隔，因此 `tpot_seconds` 为 `None`。

### MetricsSummary

保存整批请求的汇总结果：

- 请求数量；
- 输出 Token 总数；
- 整批请求持续时间；
- 输出吞吐量；
- TTFT 的 P50 和 P99；
- TPOT 的 P50 和 P99；
- E2E 的 P50 和 P99。

当所有请求都只生成一个 Token 时，TPOT 的 P50 和 P99 为 `None`。

## 3. 计算公式

单请求指标：

```text
TTFT = first_token_at - started_at
E2E = completed_at - started_at
TPOT = (completed_at - first_token_at) / (output_tokens - 1)
```

TPOT 只在 `output_tokens >= 2` 时计算。

整批输出吞吐量：

```text
Throughput =
所有请求的输出 Token 总数
/
(最晚完成时间 - 最早开始时间)
```

P50 和 P99 使用 nearest-rank 方法：

1. 将所有数值从小到大排序；
2. 计算位置 `ceil(百分位 × 样本数量) - 1`；
3. 返回该位置上的数值。

明确使用固定算法，避免不同统计库产生不同结果。

## 4. 对外函数

### calculate_request_metrics

```text
calculate_request_metrics(trace) -> RequestMetrics
```

接收一条原始请求记录，计算该请求的 TTFT、TPOT 和 E2E。

### summarize_metrics

```text
summarize_metrics(traces) -> MetricsSummary
```

接收一组请求记录，计算整批请求的吞吐量、P50 和 P99。

## 5. 输入校验

模块拒绝以下非法数据：

- `request_id` 不是非空字符串；
- 时间不是有限数字；
- 第一个 Token 时间早于请求开始时间；
- 完成时间早于第一个 Token 时间；
- 完成时间不晚于请求开始时间；
- `output_tokens` 不是正整数；
- 汇总输入为空；
- 一批请求中存在重复的 `request_id`。

布尔值不能作为数字或 Token 数使用。

## 6. 测试策略

测试使用可以人工计算的时间数据，覆盖：

- 数据结构不可修改；
- TTFT、TPOT 和 E2E 公式；
- 单 Token 请求的 TPOT 为 `None`；
- 多请求吞吐量；
- P50 和 P99；
- 输入为空；
- 时间顺序错误；
- 非法 Token 数；
- 重复请求编号；
- 全部单 Token 请求的 TPOT 汇总为 `None`。

完成后运行：

```text
pytest tests/test_metrics.py -q
pytest -q
ruff check .
uv lock --check
```

## 7. 第一版明确不做

第一版不实现：

- HTTP 请求发送；
- 并发调度；
- vLLM 接入；
- GPU 显存采样；
- Goodput；
- 失败请求和超时请求统计；
- 指标文件持久化；
- 图表生成。

这些功能将在核心指标公式验证通过后分阶段接入。

## 8. 成功标准

- 所有指标都能由人工构造的数据验证；
- 相同输入始终得到相同结果；
- 非法输入产生明确错误；
- CPU CI 不依赖 GPU、CUDA、PyTorch 或 vLLM；
- 后续 HTTP 客户端只负责产生 `RequestTrace`，不重复实现指标公式。
