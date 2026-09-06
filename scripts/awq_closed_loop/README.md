# AWQ W4A16 性能—精度闭环

## 固定对象

- FP16：本机固定 `Qwen2.5-0.5B-Instruct` 快照 `7ae557...`。
- AWQ：官方 `Qwen/Qwen2.5-0.5B-Instruct-AWQ` commit `7c0528...`。
- 官方配置：4 bit、group size 128、zero point、GEMM。运行时使用 vLLM 推荐且预先锁定的 AWQ-Marlin 内核；激活与 KV Cache 均为 FP16。
- 普通 AWQ 和 AWQ-Marlin 各一次 16 请求预检，仅验证兼容性，不进入正式统计。

## 性能协议

- RTX 4060 8 GiB、vLLM 0.23.0+cu129、FLASHINFER、eager、原生 Append、关闭 Prefix Caching。
- max-model-len=2048、gpu-memory-utilization=0.65、max-num-seqs=16、max-num-batched-tokens=2048、block-size=16。
- 每轮独立启动；固定初始化后预热 32 请求，正式 160 请求；并发 4、输入 1536、输出 64、共享前缀 75%（缓存关闭）。
- 三轮交替顺序：FP16→AWQ、AWQ→FP16、FP16→AWQ。
- 测量窗出现请求失败、JIT、ERROR 或 OOM 则该尝试不计为完成；失败目录保留。
- 保存输出吞吐、TTFT/TPOT/E2E、模型加载显存、可用 KV 显存、KV token 容量和逐秒 GPU 状态。

## 精度协议

- 完整复制前次冻结数据并核对哈希，不能重新抽题。
- 200 题 GSM8K 子集：同一 token ID、greedy、max_tokens=512、并发 4；报告严格 `####` 匹配和末尾数字匹配。
- 前 64 题参考解答条件 PPL：最多 128 个参考 token，强制参考续写、读取处理器修改前 raw logprobs；另报排除首 token 的 decode-only PPL。
- 性能服务不加载参考续写处理器；PPL 专用引擎不参与性能比较。

## 结论规则

不将预期的“显存降低 65%、吞吐提升 12%、TTFT 降低 8%”写成结果。权重文件大小、GPU 模型加载显存、KV 容量、性能和精度分别报告；三轮必须全部展示。
