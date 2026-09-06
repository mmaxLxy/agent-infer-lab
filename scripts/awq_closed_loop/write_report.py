"""Create the Chinese AWQ report from verified-summary.json."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    s = json.loads((root / "verified-summary.json").read_text())
    perf = s["performance"]
    ps = s["performance_summary"]
    q = s["quality"]
    qp = q["flexible_paired"]
    ppl = s["conditional_ppl"]
    inv = s["model_inventory"]
    t = ps["output_throughput_tokens_per_second"]
    tt = ps["ttft_p50_seconds"]
    tp = ps["tpot_p50_seconds"]
    fp_file = inv["fp16"]["files"][0]["bytes"]
    awq_file = inv["awq"]["files"][0]["bytes"]
    file_drop = inv["weight_file_reduction_percent"]
    fp_mem = perf[0]["fp16"]["model_loading_memory_gib"]
    awq_mem = perf[0]["awq"]["model_loading_memory_gib"]
    cap_fp = perf[0]["fp16"]["kv_capacity_tokens"]
    cap_awq = perf[0]["awq"]["kv_capacity_tokens"]
    rows = []
    for r in perf:
        a, b = r["fp16"], r["awq"]
        rows.append(
            f"| {r['round']} | {a['output_throughput_tokens_per_second']:.2f} / {b['output_throughput_tokens_per_second']:.2f} | {r['awq_vs_fp16_percent']['output_throughput_tokens_per_second']:+.2f}% | {a['ttft_p50_seconds'] * 1000:.2f} / {b['ttft_p50_seconds'] * 1000:.2f} | {a['tpot_p50_seconds'] * 1000:.2f} / {b['tpot_p50_seconds'] * 1000:.2f} |"
        )
    cat_fp = inv["fp16"]["tensor_bytes_by_category"]
    cat_awq = inv["awq"]["tensor_bytes_by_category"]
    linear_drop = (
        1 - (cat_awq["awq_quantized_linear_storage"] + cat_awq["other"]) / cat_fp["other"]
    ) * 100
    report = f"""# AgentInferLab：AWQ W4A16 性能与精度评估

日期：2026-09-05。硬件：RTX 4060 8 GiB。模型：Qwen2.5-0.5B-Instruct。状态：真实实验完成。

## 1. 结论

- 官方 AWQ checkpoint 为 **4-bit、group size 128、zero point、GEMM**；vLLM 使用优化的 **AWQ-Marlin** 内核，激活与 KV Cache 均为 FP16。
- 权重文件从 {fp_file / 1024**2:.2f} MiB 降至 {awq_file / 1024**2:.2f} MiB，减少 **{file_drop:.2f}%**，不是预期的 65%。
- vLLM 报告模型加载显存从 {fp_mem:.2f} GiB 降至 {awq_mem:.2f} GiB，减少 **{(1 - awq_mem / fp_mem) * 100:.2f}%**。
- 相同显存预算下，FP16 KV Cache 容量从 {cap_fp:,} 增至 {cap_awq:,} tokens，增加 **{(cap_awq / cap_fp - 1) * 100:.2f}%**。
- 三轮吞吐配对变化中位数 **{t["paired_change_percent_median"]:+.2f}%**，范围 **{t["paired_change_percent_range"][0]:+.2f}%～{t["paired_change_percent_range"][1]:+.2f}%**：当前负载没有稳定加速。
- 200题 GSM8K 数字答案匹配从 {qp["fp16_correct"]}/200（{qp["fp16_correct"] / 2:.1f}%）变为 {qp["awq_correct"]}/200（{qp["awq_correct"] / 2:.1f}%），变化 **{qp["difference_percentage_points"]:+.1f}个百分点**。
- 64题参考解答 decode-only 条件 PPL 从 {ppl["fp16"]["decode_conditional_ppl"]:.4f} 变为 {ppl["awq"]["decode_conditional_ppl"]:.4f}，相对变化 **{ppl["awq_vs_fp16_percent"]["decode_conditional_ppl"]:+.2f}%**（越低越好）。

## 2. 模型来源和存储结构

AWQ 模型来自[官方 Qwen 仓库](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ)，固定 commit `7c05280e9583b6cace2dc9d8950e21bfd3d72c19`。下载使用镜像，但响应头 `X-Repo-Commit` 与固定官方 commit 一致；所有模型文件另存 SHA256。

| checkpoint 存储 | FP16 | AWQ |
|---|---:|---:|
| safetensors 文件 | {fp_file / 1024**2:.2f} MiB | {awq_file / 1024**2:.2f} MiB |
| 嵌入/输出头张量 | {cat_fp["embedding_or_lm_head"] / 1024**2:.2f} MiB | {cat_awq["embedding_or_lm_head"] / 1024**2:.2f} MiB |
| 其余 FP16 / AWQ 线性层存储 | {cat_fp["other"] / 1024**2:.2f} MiB | {(cat_awq["awq_quantized_linear_storage"] + cat_awq["other"]) / 1024**2:.2f} MiB |

被量化主体的 checkpoint 存储约减少 {linear_drop:.2f}%，但 AWQ 文件同时保存 `model.embed_tokens.weight` 和 `lm_head.weight` 两个各约 {cat_awq["embedding_or_lm_head"] / 2 / 1024**2:.2f} MiB 的张量；FP16 文件只保存共享嵌入的一份。因此小模型的**整个文件**只减少 {file_drop:.2f}%。配置仍声明 tied embeddings，GPU 驻留显存应以 vLLM 实测为准，而不是直接用文件大小代替。

## 3. 固定实验协议

vLLM 0.23.0+cu129、FLASHINFER、eager、原生 Append、关闭 Prefix Caching；max-model-len=2048、gpu-memory-utilization=0.65、max-num-seqs=16、max-num-batched-tokens=2048、block-size=16。

每次独立启动并执行固定初始化，预热32请求；正式测量160请求，并发4、输入1536、输出64、共享前缀75%（缓存关闭）。启动顺序 FP16→AWQ、AWQ→FP16、FP16→AWQ。
正式性能共 **960/960** 请求成功，六个测量时间窗均未检出 JIT、ERROR 或 OOM。

第一次兼容性 smoke 显式使用普通 AWQ；日志提示该模型支持更快 AWQ-Marlin。随后在所有正式实验之前统一选定 AWQ-Marlin，并再次通过 smoke。两个 smoke 都不进入正式统计，选择过程记录于 `protocol-amendment.json`。

## 4. 三轮性能结果

表中为“FP16 / AWQ”；吞吐单位 tokens/s，延迟单位 ms。

| 轮次 | 输出吞吐 | AWQ吞吐变化 | TTFT P50 | TPOT P50 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

| 指标 | FP16三轮中位数 | AWQ三轮中位数 | 配对变化中位数 |
|---|---:|---:|---:|
| 输出吞吐 | {t["fp16_median"]:.2f} | {t["awq_median"]:.2f} | {t["paired_change_percent_median"]:+.2f}% |
| TTFT P50 | {tt["fp16_median"] * 1000:.2f} ms | {tt["awq_median"] * 1000:.2f} ms | {tt["paired_change_percent_median"]:+.2f}% |
| TPOT P50 | {tp["fp16_median"] * 1000:.2f} ms | {tp["awq_median"] * 1000:.2f} ms | {tp["paired_change_percent_median"]:+.2f}% |

配对变化先在每轮按 `AWQ/FP16−1` 计算，再取中位数；不能用两列中位数相除替代。AWQ主要降低矩阵权重带宽和容量，但这个0.5B模型、并发4的负载很小，内核固定开销和运行波动足以抵消收益。本实验没有进一步剖析到单个线性层，因此不把该解释写成已证明的因果。

## 5. GSM8K生成正确率

使用前次冻结且哈希一致的200题、相同聊天模板和token IDs；greedy、max_tokens=512、并发4。两组共400/400请求成功。

| 评分 | FP16 | AWQ | AWQ−FP16 |
|---|---:|---:|---:|
| 严格 `#### 数字` | {q["strict_paired"]["fp16_correct"]}/200 | {q["strict_paired"]["awq_correct"]}/200 | {q["strict_paired"]["difference_percentage_points"]:+.1f} pp |
| 末尾数字匹配 | {qp["fp16_correct"]}/200 | {qp["awq_correct"]}/200 | {qp["difference_percentage_points"]:+.1f} pp |
| 达到512-token上限 | {q["fp16"]["length_truncated"]} | {q["awq"]["length_truncated"]} | — |

末尾数字匹配的配对转换：FP16错→AWQ对 {qp["fp16_wrong_awq_correct"]} 题，FP16对→AWQ错 {qp["fp16_correct_awq_wrong"]} 题。20,000次固定种子配对bootstrap的95%百分位区间为 **[{qp["paired_bootstrap_95_percentile_ci_pp"][0]:+.1f}, {qp["paired_bootstrap_95_percentile_ci_pp"][1]:+.1f}] pp**。
严格指标主要受模型没有按 `####` 格式回答影响；末尾数字也可能误提取。全部原始回答均保留，没有人工挑样。

## 6. 参考解答条件困惑度

相同前64题、共 {ppl["fp16"]["tokens"]} 个参考token，其中 {ppl["fp16"]["decode_tokens"]} 个属于排除每题首token后的decode-only部分。专用处理器强制参考token进入真实decode，但读取修改前的raw logprobs；两个引擎都通过“正常/强制首token概率一致”和“输出token等于参考token”校验。该处理器没有进入性能服务。

| 指标 | FP16 | AWQ | 相对变化 |
|---|---:|---:|---:|
| 全部条件PPL | {ppl["fp16"]["conditional_ppl"]:.4f} | {ppl["awq"]["conditional_ppl"]:.4f} | {ppl["awq_vs_fp16_percent"]["conditional_ppl"]:+.2f}% |
| decode-only条件PPL | {ppl["fp16"]["decode_conditional_ppl"]:.4f} | {ppl["awq"]["decode_conditional_ppl"]:.4f} | {ppl["awq_vs_fp16_percent"]["decode_conditional_ppl"]:+.2f}% |

这是GSM8K参考解答的条件PPL，不是WikiText PPL；200/64题子集不能代表全部任务或长上下文精度。

## 7. 可用于简历的实测表述

在RTX 4060上完成Qwen2.5-0.5B FP16与AWQ W4A16的部署对照，接入AWQ-Marlin内核，使vLLM报告的模型驻留显存下降{(1 - awq_mem / fp_mem) * 100:.1f}%、同预算KV容量提升{(cap_awq / cap_fp - 1) * 100:.1f}%；通过三轮配对压测、200题GSM8K及逐token条件困惑度评估，识别小模型权重文件仅缩小{file_drop:.1f}%且低并发无稳定吞吐收益的适用边界。

不得写“权重显存降低65%、吞吐提升12%、TTFT降低8%”，这些预期没有被本实验支持。
"""
    with (root / "AWQ-W4A16-性能与精度评估报告-2026-09-05.md").open("x", encoding="utf-8") as f:
        f.write(report)
    print("REPORT_WRITTEN")


if __name__ == "__main__":
    main()
