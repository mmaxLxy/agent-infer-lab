"""Render a reviewable Chinese report from the verified summary."""

import argparse
import json
from pathlib import Path
from statistics import median


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    s = json.loads((root / "verified-summary.json").read_text())
    ps = s["performance_summary"]
    qs = s["quality"]
    pp = s["conditional_ppl"]
    perf = s["performance"]
    tp = ps["output_throughput_tokens_per_second"]
    tt = ps["ttft_p50_seconds"]
    to = ps["tpot_p50_seconds"]
    strict, flexible = qs["strict_paired"], qs["flexible_paired"]
    caps = {kind: sorted({r[kind]["kv_capacity_tokens"] for r in perf}) for kind in ("fp16", "fp8")}
    assert all(len(v) == 1 for v in caps.values())
    capacity_ratio = caps["fp8"][0] / caps["fp16"][0]
    rows = []
    for r in perf:
        a, b = r["fp16"], r["fp8"]
        rows.append(
            "| {} | {:.2f} / {:.2f} | {:+.2f}% | {:.2f} / {:.2f} | {:.2f} / {:.2f} |".format(
                r["round"],
                a["output_throughput_tokens_per_second"],
                b["output_throughput_tokens_per_second"],
                r["fp8_vs_fp16_percent"]["output_throughput_tokens_per_second"],
                a["ttft_p50_seconds"] * 1000,
                b["ttft_p50_seconds"] * 1000,
                a["tpot_p50_seconds"] * 1000,
                b["tpot_p50_seconds"] * 1000,
            )
        )
    oldrows = []
    oldroot = root.parent / "2026-09-03-quantization"
    for name in ("formal-r1-attempt3", "formal-r2-attempt1", "formal-r3-attempt1"):
        a, b = (
            json.loads((oldroot / name / k / "measurement.json").read_text())["result"]["metrics"]
            for k in ("fp16", "fp8")
        )
        oldrows.append(
            "| {} | {:.2f} | {:.2f} | {:+.2f}% |".format(
                name,
                a["output_throughput_tokens_per_second"],
                b["output_throughput_tokens_per_second"],
                (
                    b["output_throughput_tokens_per_second"]
                    / a["output_throughput_tokens_per_second"]
                    - 1
                )
                * 100,
            )
        )
    all_gpu = [r[k]["gpu"] for r in perf for k in ("fp16", "fp8")]
    temp_min = min(g["temperature.gpu"]["min"] for g in all_gpu)
    temp_max = max(g["temperature.gpu"]["max"] for g in all_gpu)
    memory = {
        k: median(r[k]["gpu"]["memory.used [MiB]"]["median"] for r in perf) for k in ("fp16", "fp8")
    }
    report = f"""# AgentInferLab：FP8 KV Cache 性能与精度评估

日期：2026-09-04。状态：已完成本报告限定范围内的真实测试；未实现或测试 AWQ。

## 1. 核心结果

- KV Cache 可容纳 token 数：FP16 **{caps["fp16"][0]:,}**，FP8 **{caps["fp8"][0]:,}**，为 **{capacity_ratio:.2f}×**。这是相同显存预算下的缓存容量，不是整卡显存占用减半。
- 三轮配对吞吐变化中位数 **{tp["paired_change_percent_median"]:+.2f}%**，范围 **{tp["paired_change_percent_range"][0]:+.2f}%～{tp["paired_change_percent_range"][1]:+.2f}%**。当前并发 4、1536/64-token 负载未观察到稳定端到端加速。
- 200 题 GSM8K 子集的末尾数字匹配：FP16 **{flexible["fp16_correct"]}/200（{flexible["fp16_correct"] / 2:.1f}%）**，FP8 **{flexible["fp8_correct"]}/200（{flexible["fp8_correct"] / 2:.1f}%）**，相差 **{flexible["difference_percentage_points"]:+.1f} 个百分点**。这是预先约定的次指标，不是完整 GSM8K 官方评测。
- GSM8K 参考解答条件 PPL：FP16 **{pp["fp16"]["conditional_ppl"]:.4f}**，FP8 **{pp["fp8"]["conditional_ppl"]:.4f}**；相对变化 **{pp["fp8_vs_fp16_percent"]["conditional_ppl"]:+.2f}%**。不是 WikiText PPL。
- 所有 FP8 正式性能与精度引擎使用同一份冻结的 24 层 scale，并验证评估前后一致。

## 2. 环境与公平性

RTX 4060 8 GiB；Qwen2.5-0.5B-Instruct；模型快照 `7ae557604adf67be50417f59c2c2f167def9a775`；vLLM `0.23.0+cu129`；FP16 权重。

两组均采用 FLASHINFER、原生 Append、eager、关闭 Prefix Caching；最大上下文 2048；显存预算 0.65；max-num-seqs=16；max-num-batched-tokens=2048；block-size=16。
仅 KV Cache dtype 和相应 FP8 scale 处理不同。没有修改已安装 vLLM 或安装新评估依赖。

正式顺序 FP16→FP8、FP8→FP16、FP16→FP8；每次独立启动，固定训练文本初始化后预热 32 请求，测量 160 请求。
固定并发 4，输入 1536、输出 64、共享前缀比例 75%（缓存关闭），seed=20260903。
**960/960 正式性能请求成功，六个测量时间窗均未检出 JIT/ERROR/OOM。** 首次启动编译在测量之前，退出阶段日志不混入正式时间窗。

## 3. 三轮性能结果

表中成对数值均为“FP16 / FP8”。吞吐单位 tokens/s；延迟单位 ms。

| 轮次 | 输出吞吐 | FP8 吞吐变化 | TTFT P50 | TPOT P50 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

| 指标 | FP16 三轮中位数 | FP8 三轮中位数 | 三个配对变化的中位数 |
|---|---:|---:|---:|
| 输出吞吐 tokens/s | {tp["fp16_median"]:.2f} | {tp["fp8_median"]:.2f} | {tp["paired_change_percent_median"]:+.2f}% |
| TTFT P50 ms | {tt["fp16_median"] * 1000:.2f} | {tt["fp8_median"] * 1000:.2f} | {tt["paired_change_percent_median"]:+.2f}% |
| TPOT P50 ms | {to["fp16_median"] * 1000:.2f} | {to["fp8_median"] * 1000:.2f} | {to["paired_change_percent_median"]:+.2f}% |

最后一列是先逐轮算 `FP8/FP16−1` 再取中位数，**不是**前两列相除。三轮存在运行波动，不能只选最有利的一轮。

测量中整卡显存占用的“各轮中位数再取中位数”：FP16 {memory["fp16"]:.0f} MiB，FP8 {memory["fp8"]:.0f} MiB；包含桌面和其他系统占用，不等于模型权重显存。
全部性能轮次观测温度范围 {temp_min:.0f}～{temp_max:.0f}℃。没有锁频、隔离 Windows 负载或进行因果归因剖析。

### 为什么容量翻倍却没有稳定加速？

本负载同时活跃的 token 量上限约为 `4×(1536+64)=6400`，远低于 FP16 的 {caps["fp16"][0]:,}-token 缓存容量。
因此实验没有触发“FP16 装不下、FP8 能装下”的容量优势。容量、每步计算成本和总吞吐是不同指标；不能由容量翻倍推出吞吐翻倍。
量化/反量化、后端实现和桌面 GPU 波动可能参与影响，但本实验没有证明哪一个是性能变化的主要原因。

## 4. scale 来源与一致性

原始 checkpoint 的 quantization_config 为空，权重无 scale tensor。
本机虽提示 `--calculate-kv-scales` 已弃用，但实测开关仍被执行：服务刚初始化完成时 24 层计算标志已经为 false，第一层 K scale 约 0.65125，而不是默认 1.0。
因此不能仅凭弃用警告判定该参数无效，也不能假定第一次业务请求才触发计算。

新实验首次显式重新触发一次计算：从固定 GSM8K **训练集**文本取 1536 tokens，按当前层实现的 absmax/range 生成 scale；保存后，全部 FP8 引擎加载同一组数值并关闭后续重算。
这是一段固定自然文本的单次初始化，**不是**完整训练集优化校准。测试题没有参与 scale 选择。

scale 内容指纹（对排序后的层名与 q/k/v 数值计算，不等于 JSON 文件字节哈希）：

`{s["scale_sha256"]}`

性能与生成质量会话保存初始化完成、固定初始化输入后、预热后、正式评估后的四份快照；条件困惑度会话保存初始化完成、预热后、评估后的三份快照。新旧实验 scale 语义不同，不把新精度结果直接贴到旧性能数据上。

## 5. 下游题目正确率

数据来自 [OpenAI GSM8K 官方仓库](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)。commit、文件哈希、固定抽样索引、聊天模板和 token ID 都已归档。
测试集固定随机抽取 200 题；zero-shot，greedy，max_tokens=512，分组并发 4。两组共 **400/400** 请求成功。

| 预先约定的评分规则 | FP16 | FP8 | FP8−FP16 |
|---|---:|---:|---:|
| 主指标：严格 `#### 数字` 格式匹配 | {strict["fp16_correct"]}/200（{strict["fp16_correct"] / 2:.1f}%） | {strict["fp8_correct"]}/200（{strict["fp8_correct"] / 2:.1f}%） | {strict["difference_percentage_points"]:+.1f} pp |
| 次指标：缺少格式时取末尾数字 | {flexible["fp16_correct"]}/200（{flexible["fp16_correct"] / 2:.1f}%） | {flexible["fp8_correct"]}/200（{flexible["fp8_correct"] / 2:.1f}%） | {flexible["difference_percentage_points"]:+.1f} pp |
| 因达到 512-token 上限而截断 | {qs["fp16"]["length_truncated"]} | {qs["fp8"]["length_truncated"]} | — |

模型经常输出正确数字但不遵循 `####` 格式，严格匹配更受格式遵循影响，不能直接当作数学能力本身。末尾数字规则同样有误提取风险；本报告没有通过人工挑样修正分数。
截断样本仍保留并按相同提取规则评分，不从分母删除。

末尾数字匹配的配对转换：FP16 错→FP8 对 **{flexible["fp16_wrong_fp8_correct"]}** 题，FP16 对→FP8 错 **{flexible["fp16_correct_fp8_wrong"]}** 题。
净差 {flexible["difference_percentage_points"]:+.1f} pp；固定种子 20,000 次配对 bootstrap 的 95% 百分位区间 **[{flexible["paired_bootstrap_95_percentile_ci_pp"][0]:+.1f}, {flexible["paired_bootstrap_95_percentile_ci_pp"][1]:+.1f}] pp**。
该区间描述这批题目的抽样不确定性，不覆盖引擎重复运行波动。本子集支持数字答案匹配率下降，但不能推广到所有任务或所有 FP8 配置。

## 6. 参考解答条件困惑度

固定测试子集前 64 题，每题参考解答前至多 128 tokens，清除计算器标注。
强制参考 token 逐步进入真实 decode，同时读取 logits 修改前的 raw logprobs，避免用整段 prefill 的概率冒充量化缓存 decode 评估。
每个引擎均验证“未强制”和“强制同一首 token”的原始 logprob 一致，且生成 token 完全等于参考。

| 指标 | FP16 | FP8 | 相对变化 |
|---|---:|---:|---:|
| 全部参考 token 条件 PPL | {pp["fp16"]["conditional_ppl"]:.4f} | {pp["fp8"]["conditional_ppl"]:.4f} | {pp["fp8_vs_fp16_percent"]["conditional_ppl"]:+.2f}% |
| 剔除每题首 token 的 decode-only 条件 PPL | {pp["fp16"]["decode_conditional_ppl"]:.4f} | {pp["fp8"]["decode_conditional_ppl"]:.4f} | {pp["fp8_vs_fp16_percent"]["decode_conditional_ppl"]:+.2f}% |
| 评分 token 数 | {pp["fp16"]["tokens"]} | {pp["fp8"]["tokens"]} | — |
| decode-only token 数 | {pp["fp16"]["decode_tokens"]} | {pp["fp8"]["decode_tokens"]} | — |

PPL=exp(总负对数概率/总 token 数)，越低越好。这是 **GSM8K 参考解答条件 PPL**，不是 WikiText 或无条件语言建模 PPL。
此评分专用引擎使用同步调度和参考续写处理器，**不参与性能测量**。生成正确率的服务没有加载参考续写处理器。

## 7. 与之前手工实验的关系

旧三轮每组 80 请求，新三轮每组 160 请求；新实验额外固定并验证运行时 scale。原数据保留，仅供背景参照，不合并计算中位数。

| 旧实验 | FP16 吞吐 | FP8 吞吐 | FP8 变化 |
|---|---:|---:|---:|
{chr(10).join(oldrows)}

旧 r3 FP8 缺少 post-warmup 行号标记；可由两次 tokenize 与请求数量定位测量阶段，但不能补写成当时已经记录。新实验明确记录开始/结束行号和独立时间窗。

## 8. 交付与边界

- `verified-summary.json`：从原始结果重算并通过校验的汇总。
- `canonical-fp8-scales.json`：冻结的运行时 scale；各 session 保存前后快照。
- `performance-*`：六组完整测量、预热、日志、GPU 遥测。
- `quality-*`：400 条完整生成回答、判分和统计。
- `ppl-*-final`：两组逐 token 原始 logprob、参考 token、校验和汇总。
- `data/`：固定版本数据、LICENSE、题目索引与 token ID。
- `archive/` 与 `SHA256SUMS.txt`：环境、代码快照、旧数据指纹、新数据完整性校验。
- 项目 `scripts/fp8_closed_loop/`：协议、运行脚本、评分器、单元测试、汇总和归档脚本。

局限：仅一张桌面 GPU、一个 0.5B 模型、一个正式性能负载、200 题子集和 64 题条件 PPL。
数学题精度上下文较短，**没有证明 1536-token 或更长 Agent 上下文的质量无损**；没有完整 GSM8K/WikiText、多模型、多 scale 校准方案或高缓存压力对照。
这些限制不妨碍报告本次真实观察，但禁止推广为“FP8 普遍无损且吞吐提升 5%”。

当前建议保留 FP16 作为默认配置。若继续改进 FP8，优先用独立训练/验证文本研究更有代表性的 scale 校准，达到预先约定的质量门槛后，再测更高并发或更长上下文的容量压力负载；不要为了得到正收益而反复更换测试样本。

## 9. 可用于简历的实测表述

在 RTX 4060 / Qwen2.5-0.5B 上完成 FP16 与 FP8 KV Cache 的性能—精度对照，固定并验证 24 层量化 scale，使同显存预算下 KV 缓存容量达到 {capacity_ratio:.2f}×；通过三轮配对压测确认低并发负载无稳定吞吐收益，并在 200 题 GSM8K 子集及参考解答条件困惑度上量化精度变化、归档完整复现实验。

不可写成“权重显存降低 65%”“W4A16 吞吐提升 12%”：本次没有测试 AWQ 权重量化。
"""
    with (root / "FP8-性能与精度评估报告-2026-09-04.md").open("x", encoding="utf-8") as f:
        f.write(report)
    print("REPORT_WRITTEN")


if __name__ == "__main__":
    main()
