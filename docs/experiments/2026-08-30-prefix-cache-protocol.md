# 同版本 vLLM Prefix Caching 受控实验协议

## 范围

仅验收真实 JSON 保存与系统级 Prefix Caching 消融。不接入自定义 CUDA Kernel，不把 V0～V3 微基准收益换算成端到端收益。

## 固定条件

- 本机 RTX 4060 / WSL2，已安装 vLLM 0.23.0、PyTorch 2.11.0+cu129。
- 本地 Qwen2.5-0.5B-Instruct 快照 revision `7ae557604adf67be50417f59c2c2f167def9a775`；保存全部模型文件 SHA-256。
- FP16，`max_model_len=2048`，`gpu_memory_utilization=0.65`，`max_num_seqs=16`，`max_num_batched_tokens=2048`，缓存块16 Token。
- `--enforce-eager`：两组均不使用 CUDA Graph 或 torch.compile。结论限定于此配置，不直接复用旧实验的绝对性能。
- 独立本地端口8011。只启停本次实验创建的服务进程，不影响用户原有服务。
- 每轮80请求、并发4、输入512/1024/1536 Token、输出64 Token、共享前缀75%。温度0，显式 `ignore_eos=True`，按服务端 usage 核验实际输出长度。
- 合成 Token-ID 负载，只评测长度与复用对性能的影响，不评测生成质量或真实 Agent 对话的代表性。

## 输入控制

旧 `prepare_requests` 使用循环 Token 池，原 `seed` 只影响长度选择；固定长度下更换 seed 不能证明 Prompt 不同。为避免改变旧实验，保留旧生成方式，正式新实验使用单独的 `controlled_inputs.py`。

新生成方式通过 `/tokenize` 获取有效 Token 池，使用局部 `random.Random(content_seed)` 生成公共前缀和后缀。在公共前缀后的第一个位置，为每个请求分配不同 Token，因此同长度的任意两请求最长公共前缀恰好为75%，不产生额外完整 Prompt 重复。固定 seed 不改变全局随机状态。

每轮保存全部 Token IDs、长度、重复数、两两公共前缀最小/最大值和输入 SHA-256。同一 repeat、同一长度的 ON/OFF 输入摘要必须完全相同。

## 预热和初始缓存

每个长度先运行8个不计入正式结果的预热请求，长度和输出数与正式负载一致，但共享前缀为0、content_seed独立。程序核验预热与正式请求不共享完整16 Token首块。

开启缓存时，预热完成且服务空闲后调用 `/reset_prefix_cache`。本机该API仅返回HTTP 200，不检查底层成功，因此另行核验 Engine 日志出现新的 `Successfully reset prefix cache`，否则终止实验。关闭缓存时无需重置。

每个测量批次从冷前缀缓存开始，但允许批次内部的后续请求复用前序请求的公共前缀。不同长度/轮次使用不同内容种子，并检查同一服务会话内的首缓存块不重复。不是“整个测量阶段每个请求都冷缓存”，也不是“预热过全部正式 Prompt”。

## 重复与顺序

- 三轮独立重复，每轮开启、关闭各启动一次全新服务，共6个服务会话。
- 第一/三轮 OFF→ON，第二轮 ON→OFF。
- 长度顺序轮换为512/1024/1536、1024/1536/512、1536/512/1024。
- 总计18个测量批次、1440个正式请求；3轮不能消除所有主机噪声，报告每轮数据与范围，不能夸大统计显著性。
- 失败请求保留在原始结果里，不静默重试、不挑选成功轮次替换。预热失败或输入/重置验证失败，则停止，不启动有疑问的正式批次。

## 时间与吞吐口径

- TTFT为客户端首个非空文本SSE事件到达时间减去发送开始时间；不是服务端纯Prefill耗时。
- TPOT为客户端完成时间与首次文本时间之差除以实际输出Token数减一；是平均间隔，不是逐Token间隔分布。
- 原成功请求窗口指标保留；主报告使用成功输出Token数除以包含失败等待的整轮执行耗时。
- 计时不包含输入准备、预热、环境采集、指标抓取与文件写入。
- JSON中的原始时间线重新计算聚合指标后必须一致。

## 缓存命中率

每轮前后保存完整 `/metrics` 文本，等待成功计数器更新后抓取结束快照。

`命中率 = Δvllm:prefix_cache_hits_total / Δvllm:prefix_cache_queries_total`。

本机计数器单位为Token，非请求数。关闭缓存可能没有查询增量，此时命中率为N/A，不强行除零，也不把服务端日志的历史累计百分比当成本轮命中率。

## 证据与复现

保存服务启动参数、服务日志、模型摘要、运行环境、pip freeze、源码快照ZIP及源码SHA-256、实际输入、成功/失败时间线、聚合指标、Prometheus前后快照和校验结果。若工作区有未提交变动，如实记录 dirty，并使用冻结的完整运行源码快照复现，不宣称 Git commit 单独能恢复此次运行。脚本会核验实验前后运行源码哈希不变。

运行入口：`scripts/run_prefix_ablation.py --output-dir <新的空目录路径> --model-path <本地模型快照> --port 8011`。使用CPU开发环境执行，服务由参数指定的GPU运行环境启动。输出目录不可覆盖。

## 官方方法参考

- [vLLM开发端点说明](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)：缓存重置只在开发模式启用；本实验仅监听本地回环。
- [vLLM指标定义](https://docs.vllm.ai/en/stable/design/metrics/)：Counter使用增量解释。
- 具体端点行为及Token单位以本机0.23.0的 `entrypoints/serve/dev/cache/api_router.py`、`v1/metrics/loggers.py` 为准。
