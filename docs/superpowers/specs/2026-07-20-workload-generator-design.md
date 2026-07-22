# Agent 工作负载规格生成器设计

日期：2026-07-20

## 1. 今日目标

实现一个纯 CPU、无外部依赖的请求规格生成器。给定相同配置和随机种子时，它必须生成完全相同的请求序列，为后续 vLLM 性能对比提供公平、可复现的输入。

本阶段只定义“请求应该具有哪些长度和共享前缀特征”，不生成真实文本，也不调用 Qwen Tokenizer、GPU 或 vLLM。

## 2. 数据模型

### `WorkloadConfig`

不可变配置对象，字段如下：

- `request_count: int`：请求总数，必须大于 0。
- `input_token_choices: tuple[int, ...]`：允许选择的目标输入长度，不能为空，且每项必须大于 0。
- `output_token_choices: tuple[int, ...]`：允许选择的目标输出长度，不能为空，且每项必须大于 0。
- `shared_prefix_ratio: float`：共享前缀占输入长度的比例，取值范围为 `[0.0, 1.0]`。
- `concurrency: int`：后续实验计划使用的并发数，必须大于 0 且不能超过请求总数。
- `seed: int`：本次工作负载的随机种子。

### `RequestSpec`

不可变请求规格对象，字段如下：

- `request_id: str`：稳定且唯一的请求编号，格式为 `req-000000`。
- `input_tokens: int`：从 `input_token_choices` 中确定的目标输入长度。
- `output_tokens: int`：从 `output_token_choices` 中确定的目标输出长度。
- `shared_prefix_tokens: int`：共享前缀的目标 Token 数。

## 3. 生成接口与规则

公开接口为：

```python
generate_workload(config: WorkloadConfig) -> tuple[RequestSpec, ...]
```

生成规则：

1. 使用函数内部的 `random.Random(config.seed)`，不修改 Python 全局随机状态。
2. 每个请求分别从输入长度和输出长度候选集合中选择一个值。
3. 共享前缀长度使用 `int(input_tokens * shared_prefix_ratio)` 向下取整。
4. 按请求生成顺序分配从 `req-000000` 开始的唯一编号。
5. 返回不可变元组，避免调用方在实验开始后修改请求顺序。

## 4. 错误处理

配置对象创建时立即检查所有约束。非法请求数、并发数、空长度集合、非正长度或越界前缀比例统一抛出带明确原因的 `ValueError`。

错误必须在开始生成请求前暴露，不能静默修改用户配置。例如，并发数超过请求数时不得自动缩小并发数。

## 5. 测试策略

测试先于实现，覆盖以下行为：

1. 相同配置与种子产生完全相同的请求序列。
2. 生成数量与 `request_count` 一致。
3. 输入和输出长度只来自对应候选集合。
4. 共享前缀长度按已声明规则计算。
5. 请求编号稳定、唯一且顺序明确。
6. 每类非法配置分别抛出 `ValueError`。
7. 生成过程不改变 Python 全局随机数状态。

测试文件为 `tests/test_workloads.py`，实现文件为 `src/agent_infer_lab/workloads.py`。完成后运行项目全部 pytest 与 Ruff，确保不破坏已有 CPU CI。

## 6. 本阶段明确不做

- 不生成真实 Prompt 文本或 Token ID。
- 不加载 Qwen Tokenizer、PyTorch、CUDA 或 vLLM。
- 不实现固定间隔或泊松到达时间。
- 不写 JSON 文件或命令行入口。
- 不执行性能测试，也不产生性能结论。

真实文本转换、到达时间和 JSON 持久化将在基准测试入口需要这些能力时分别设计，避免当前模块承担无关职责。

## 7. 验收标准

- `tests/test_workloads.py` 全部通过。
- 项目全量 `pytest -q` 通过。
- `ruff check .` 通过。
- 不增加第三方运行时依赖。
- 相同配置与种子的输出可逐项比较且完全一致。
