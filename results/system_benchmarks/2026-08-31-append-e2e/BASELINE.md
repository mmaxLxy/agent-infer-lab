# 2026-08-31 Append 端到端实验基线

M0 归档日期：2026-09-02。

本文件记录既有实验的代码、环境和证据边界。M0 没有重跑实验、修改历史 JSON/日志、启动模型服务或改变既有性能结论。

## 1. 实验代码状态

- 分支：`agent/cuda-kv-cache-ops`
- 实验记录的 Git HEAD：`a867d7a6ed67633357aba35eb19159cea2c1c670`
- 2026-09-02 M0 复核时的 Git HEAD：`a867d7a6ed67633357aba35eb19159cea2c1c670`
- 相对本地记录的远端跟踪分支：`ahead 4`；M0 没有执行 fetch 或 push。
- 实验运行时及 M0 复核时工作区均为 dirty，后续归档提交不得冒充实验运行时的干净源码版本。

M0 复核时已跟踪但未提交：

```text
.gitignore
src/agent_infer_lab/result_storage.py
src/agent_infer_lab/vllm_client.py
tests/test_vllm_client.py
```

M0 复核时未跟踪的项目内容：

```text
cuda/vllm_append_adapter.py
docs/experiments/2026-08-30-prefix-cache-controlled.md
docs/experiments/2026-08-30-prefix-cache-protocol.md
docs/experiments/2026-08-31-vllm-append-e2e.md
docs/progress/2026-08-30.md
integrations/
results/system_benchmarks/
scripts/run_prefix_ablation.py
scripts/summarize_prefix_ablation.py
src/agent_infer_lab/controlled_inputs.py
tests/test_controlled_inputs.py
tests/test_prefix_ablation_runner.py
tests/test_prefix_ablation_summary.py
```

## 2. 运行环境与模型

- GPU：NVIDIA GeForce RTX 4060，约 8 GiB，SM 8.9
- 系统：Windows + WSL2 Ubuntu
- Python：3.12
- PyTorch：2.11.0+cu129
- vLLM：0.23.0
- CUDA Toolkit：12.9
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 模型 revision：`7ae557604adf67be50417f59c2c2f167def9a775`
- 正式路径：FP16、FlashAttention、eager、单 GPU、Prefix Caching 关闭

## 3. 源码快照复核

正式归档中的 `source_snapshot.tar.gz` 已在临时目录解包核对：

- 文件数：49
- 与 2026-09-02 项目中对应文件相同：49
- 缺失或内容不一致：0
- 插件 SHA256：`2286a1e48114fe3f5faf9aaa3f9b3234ee558544dc14608b05961a8389c5b77f`
- Adapter SHA256：`ae69dc2c6318f611c344be399da0b3bc0af4f980be5278fb005c38801bdb26cd`

插件与 Adapter 哈希和正式实验审计日志一致。临时解包目录已在复核后删除。

## 4. 正确性验证口径

`verification/hook/append_verify_2533.jsonl` 包含：

- `verified` 总记录：840
- 初始化/探测记录：前 24 条，`mapped_rows=16`、`valid_tokens=0`
- 排除初始化/探测后的真实请求路径比较：816
- 真实请求路径累计有效 Token 写入：13,728
- 缓存比较失败：0

零调用进程日志包含 `process_exit`；主验证进程日志没有 `process_exit`。这不改变已落盘的逐调用缓存比较结果，但不能据此宣称主验证服务留下了可证实的优雅退出记录。

## 5. 归档内容与完整性

正式归档包含：

```text
native-r1/ ... native-r3/
custom-r1/ ... custom-r3/
verification/
source_snapshot.tar.gz
BASELINE.md
SHA256SUMS.txt
```

M0 操作前的 `SHA256SUMS.txt` 共有 25 个条目，复核结果为 25/25 通过。M0 将仓库外 verify 证据复制到 `verification/`，没有移动或删除原始暂存目录。更新后的完整文件哈希记录在同目录的 `SHA256SUMS.txt`；该清单不包含自身。

## 6. 结论边界

- 自定义 Append 已进入本次限定的真实 vLLM FP16 eager 路径。
- 有限范围正确性验证通过。
- native/custom 各三轮端到端对照未显示稳定整体加速。
- 微基准 V0→V3 收益与 native/custom 端到端结果使用不同基线，必须分别报告。
- M0 只冻结和补齐既有证据，不为 M1 性能归因或后续量化实验预先给出结论。
