# FP8 KV Cache 性能—精度闭环（2026-09-04）

本实验只改变 KV Cache 精度，模型权重均为 FP16；不包含 AWQ。
所有结果独立归档，旧实验不覆盖、不合并。禁止把预期收益当作结果。

## 固定协议

- 硬件：RTX 4060 8 GiB；模型：本机 Qwen2.5-0.5B-Instruct 固定快照。
- vLLM 0.23.0+cu129，FLASHINFER，eager，原生 Append，Prefix Caching 关闭。
- max-model-len=2048，gpu-memory-utilization=0.65，max-num-seqs=16，max-num-batched-tokens=2048，block-size=16。
- 每次启动记录全部 24 层 q/k/v scale；FP8 首次用固定 1536-token GSM8K **训练集**文本重新触发一次 absmax 初始化。
- 保存 canonical-fp8-scales.json，所有后续 FP8 性能和精度引擎加载相同数值；初始化输入后、预热后、测量后均检查 SHA256 一致。
- 这是固定文本的单次初始化，不是完整数据集校准、更不是已验证最优的 scale。
- Worker extension 仅使用命名方法；不修改已安装 vLLM、不启用不安全 pickle RPC；HTTP 管理接口仅绑定 127.0.0.1。

## 性能

- 预检一次，不计入正式三轮。
- 正式启动顺序：FP16→FP8、FP8→FP16、FP16→FP8。
- 每次固定文本初始化后预热 32 请求，正式测量 160 请求。
- 并发 4，输入 1536 tokens、输出 64 tokens，共享前缀比例 75%（缓存关闭），seed=20260903。
- 原项目 benchmark 记录成功率、输出吞吐、TTFT、TPOT、E2E；记录测量时间窗日志和每秒 GPU 状态。
- 测量窗出现 JIT、错误或请求失败则不算成功轮次；失败尝试保留。报告每轮原值、配对变化、各自中位数，不能将两种不同的中位数口径混用。
- GPU 是桌面显示卡，未锁频或隔离 Windows 工作负载；三轮只支持本机此负载的观察，不等于生产环境显著性证明。

## 生成质量

- GSM8K 官方仓库固定 commit，数据文件 SHA256 归档。
- 随机种子 20260904，从训练集选取初始化文本，从测试集固定抽取 200 题，无测试集调 scale。
- 同一聊天模板，zero-shot，greedy，最多生成 512 tokens，分组并发 4。
- 主指标：提取最后一个 `#### 数字` 后做数值归一化的 exact match。
- 次指标：缺少格式时回退到最后一个数字；另报截断和失败，不删除困难样本。
- 对相同题目配对比较，报告净差、对错转换和配对 bootstrap 区间。200 题不是完整 GSM8K 排行榜结果。

## 条件困惑度

- 使用上述测试题前 64 题的参考解答，每题前至多 128 tokens，清除 `<<...>>` 计算器标注。
- 专用 logits processor 强制参考 token 作为下一步输入，同时读取 **修改 logits 前**的 raw logprobs。
- 每个引擎首先独立验证：正常推理和强制同一首 token 的 raw logprob 必须一致；之后验证输出 token 与参考完全一致。
- 第 2 个参考 token 起真实经过 KV Cache decode；不以整段 prefill 的 prompt-logprobs 冒充 FP8 decode 精度评估。
- PPL=exp(总负对数概率/总 token 数)，同时报告剔除每题首 token 的 decode-only 条件 PPL。
- 这是 **GSM8K 参考解答条件困惑度**，不是 WikiText PPL；评估引擎使用同步调度，不参与性能比较。
- 质量样本上下文较短，不能据此证明 1536-token 或更长上下文的质量无损。

## 运行入口（WSL）

`prepare_data.py` 冻结数据；`run_suite.py --phase smoke/performance/quality` 分阶段执行。
`run_conditional_ppl.py --kind fp16/fp8 --name <新目录>` 执行条件困惑度。
每个脚本可用 `--help` 查看必需路径。服务器和客户端 Python 分离，不依赖 shell 激活状态。
`summarize.py` 从原始结果重新计算并验证汇总；`archive.py` 记录环境、代码及文件哈希。

原始失败的 callable 探针、数据准备尝试和 smoke 结果仅供诊断，不进入正式统计。
