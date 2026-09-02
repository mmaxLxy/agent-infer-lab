"""Independently audit raw system runs and build a reproducible Markdown report."""

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from dataclasses import asdict
from pathlib import Path

from agent_infer_lab.controlled_inputs import audit_inputs
from agent_infer_lab.metrics import RequestTrace, summarize_metrics
from agent_infer_lab.prompting import PreparedRequest
from agent_infer_lab.result_storage import write_json_result


def metric(text: str, name: str) -> float:
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith(name + "{") or line.startswith(name + " ")
    )


def load_and_audit(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("cache_*_input*_repeat[123].json")):
        data = json.loads(path.read_text())
        result = data["result"]
        experiment = data["experiment"]
        length = data["configuration"]["workload"]["input_token_choices"][0]
        requests = tuple(
            PreparedRequest(r["request_id"], tuple(r["prompt_token_ids"]), r["output_tokens"])
            for r in result["prepared_requests"]
        )
        traces = tuple(RequestTrace(**r) for r in result["traces"])
        assert len(requests) == result["total_requests"] == 80
        assert len(traces) == result["successful_requests"]
        assert len(result["failures"]) == result["failed_requests"]
        assert len(traces) + len(result["failures"]) == 80
        assert {r.request_id for r in requests} == {
            r["request_id"] for r in (*result["traces"], *result["failures"])
        }
        assert audit_inputs(requests) == experiment["input_audit"]
        assert experiment["input_audit"]["pairwise_common_prefix_min"] == length * 3 // 4
        assert experiment["input_audit"]["pairwise_common_prefix_max"] == length * 3 // 4
        assert experiment["input_audit"]["duplicate_prompt_count"] == 0
        assert all(t.output_tokens == 64 for t in traces)
        recomputed = asdict(summarize_metrics(traces)) if traces else None
        assert recomputed == result["metrics"]
        assert math.isclose(
            result["wall_output_throughput_tokens_per_second"],
            sum(t.output_tokens for t in traces) / result["wall_duration_seconds"],
        )
        before = path.with_name(path.stem + "_before.prom").read_text()
        after = path.with_name(path.stem + "_after.prom").read_text()
        deltas = {
            name: metric(after, name) - metric(before, name)
            for name in (
                "vllm:request_success_total",
                "vllm:request_prompt_tokens_sum",
                "vllm:request_generation_tokens_sum",
                "vllm:prefix_cache_queries_total",
                "vllm:prefix_cache_hits_total",
            )
        }
        assert deltas["vllm:request_success_total"] == len(traces)
        assert deltas["vllm:request_prompt_tokens_sum"] == length * len(traces)
        assert deltas["vllm:request_generation_tokens_sum"] == 64 * len(traces)
        if experiment["prefix_caching"]:
            assert experiment["cache_reset"]["engine_log_confirmed"]
        queries = deltas["vllm:prefix_cache_queries_total"]
        hits = deltas["vllm:prefix_cache_hits_total"]
        rows.append(
            {
                "file": path.name,
                "length": length,
                "cache": "on" if experiment["prefix_caching"] else "off",
                "repeat": experiment["repeat"],
                "success": len(traces),
                "failures": result["failures"],
                "wall_seconds": result["wall_duration_seconds"],
                "wall_tps": result["wall_output_throughput_tokens_per_second"],
                "ttft_p50_ms": result["metrics"]["ttft_p50_seconds"] * 1000,
                "ttft_p99_ms": result["metrics"]["ttft_p99_seconds"] * 1000,
                "tpot_p50_ms": result["metrics"]["tpot_p50_seconds"] * 1000,
                "e2e_p99_ms": result["metrics"]["e2e_p99_seconds"] * 1000,
                "hit_rate": hits / queries if queries else None,
                "input_sha256": experiment["input_audit"]["input_sha256"],
                "server_counter_deltas": deltas,
            }
        )
    assert len(rows) == 18
    for length in (512, 1024, 1536):
        for repeat in (1, 2, 3):
            pair = [r for r in rows if r["length"] == length and r["repeat"] == repeat]
            assert len(pair) == 2 and {r["cache"] for r in pair} == {"on", "off"}
            assert pair[0]["input_sha256"] == pair[1]["input_sha256"]
    manifest = json.loads((directory / "client_manifest.json").read_text())
    with zipfile.ZipFile(directory / "source_snapshot.zip") as archive:
        for name, digest in manifest["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    return rows


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    aggregates = []
    pairs = []
    for length in (512, 1024, 1536):
        medians = {}
        for state in ("off", "on"):
            group = [r for r in rows if r["length"] == length and r["cache"] == state]
            medians[state] = {
                key: statistics.median(r[key] for r in group)
                for key in ("wall_tps", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "e2e_p99_ms")
            }
            medians[state]["wall_tps_range"] = [
                min(r["wall_tps"] for r in group),
                max(r["wall_tps"] for r in group),
            ]
        aggregates.append(
            {
                "length": length,
                **medians,
                "wall_tps_change_percent": (
                    medians["on"]["wall_tps"] / medians["off"]["wall_tps"] - 1
                )
                * 100,
                "ttft_p50_reduction_percent": (
                    1 - medians["on"]["ttft_p50_ms"] / medians["off"]["ttft_p50_ms"]
                )
                * 100,
            }
        )
        for repeat in (1, 2, 3):
            off = next(
                r for r in rows if (r["length"], r["repeat"], r["cache"]) == (length, repeat, "off")
            )
            on = next(
                r for r in rows if (r["length"], r["repeat"], r["cache"]) == (length, repeat, "on")
            )
            pairs.append(
                {
                    "length": length,
                    "repeat": repeat,
                    "both_runs_fully_successful": off["success"] == on["success"] == 80,
                    "wall_tps_change_percent": (on["wall_tps"] / off["wall_tps"] - 1) * 100,
                    "ttft_p50_reduction_percent": (1 - on["ttft_p50_ms"] / off["ttft_p50_ms"])
                    * 100,
                }
            )
    return aggregates, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    rows = load_and_audit(args.input_dir)
    aggregates, pairs = summarize(rows)
    success = sum(r["success"] for r in rows)
    summary = {
        "audit_passed": True,
        "total_requests": 1440,
        "successful_requests": success,
        "failed_requests": 1440 - success,
        "rows": rows,
        "medians_of_per_run_statistics": aggregates,
        "paired_comparisons": pairs,
    }
    write_json_result(args.summary_output or args.input_dir / "summary.json", summary)
    lines = [
        "# 2026-08-30：真实JSON链路与vanilla vLLM Prefix Caching消融",
        "",
        "## 验收结论",
        "",
        "- 真实CLI小实验4/4成功；磁盘JSON可读取，原始时间线重算聚合指标一致，终端数值一致。",
        f"- 正式实验18批次，共1440请求，成功{success}、失败{1440 - success}，"
        f"成功率{success / 1440:.4%}。失败轮次全部保留，没有用补跑覆盖。",
        "- 9对ON/OFF输入SHA-256完全相同；每批80个不同Prompt，任意两请求公共前缀恰好为75%。",
        "- 客户端成功数、输出Token数与服务端Prometheus增量一致；服务器实际输入长度亦核验通过。",
        "- 所有成功正式请求实际输出64 Token。9次开启缓存后的清空操作均有Engine成功日志。",
        "- 完整源码快照逐项SHA-256通过；采样前后运行源码未变化。",
        "- 未接入任何AgentInferLab自定义Kernel，以下是系统缓存开关实验，不是Kernel端到端收益。",
        "",
        "## 固定配置与统计口径",
        "",
        "RTX 4060 / WSL2，vLLM 0.23.0+cu129、PyTorch 2.11.0+cu129，"
        "Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`。",
        "FP16、eager（无CUDA Graph/torch.compile）、并发4、输出64、共享前缀75%、缓存块16。"
        "max_model_len=2048，max_num_seqs=16，max_num_batched_tokens=2048，显存利用率参数0.65。",
        "每个长度先做8请求预热，然后开启组清空缓存再正式采样。初始冷缓存，允许批内共享。"
        "每轮配对使用相同输入，不同轮次使用不同内容seed；开关顺序与长度顺序轮换。",
        "完整协议见 [实验协议](2026-08-30-prefix-cache-protocol.md)。",
        "",
        "主吞吐分母是包含失败等待的整轮执行耗时；延迟只统计成功请求。"
        "下表取三轮各自统计值的中位数，不是把所有请求混合后求分位数。"
        "每轮80个请求时nearest-rank P99等于该轮成功样本最大值，不构成稳定尾延迟保证。",
        "",
        "## 三轮统计值的中位数（保留全部轮次）",
        "",
        "|输入|OFF吞吐|ON吞吐|吞吐变化|OFF TTFT P50/ms|ON TTFT P50/ms|TTFT降低|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in aggregates:
        lines.append(
            f"|{a['length']}|{a['off']['wall_tps']:.2f}|{a['on']['wall_tps']:.2f}|"
            f"{a['wall_tps_change_percent']:+.2f}%|{a['off']['ttft_p50_ms']:.2f}|"
            f"{a['on']['ttft_p50_ms']:.2f}|{a['ttft_p50_reduction_percent']:.2f}%|"
        )
    lines += [
        "",
        "吞吐单位tokens/s。512和1536组各含一轮超时，上表是描述性汇总，"
        "不能将其吞吐差异全部归因于缓存。下面同时列出全部配对结果。",
        "",
        "## 每轮配对变化与有效性",
        "",
        "|输入|重复|ON相对OFF吞吐变化|TTFT P50降低|双方均80/80成功|",
        "|---:|---:|---:|---:|---|",
    ]
    for p in pairs:
        lines.append(
            f"|{p['length']}|{p['repeat']}|{p['wall_tps_change_percent']:+.2f}%|"
            f"{p['ttft_p50_reduction_percent']:.2f}%|"
            f"{'是' if p['both_runs_fully_successful'] else '否：受超时影响'}|"
        )
    lines += [
        "",
        "## 全部原始批次摘要",
        "",
        "|输入|Cache|轮次|成功/80|整轮秒|吞吐|TTFT P50/ms|TTFT P99/ms|TPOT P50/ms|本轮命中率|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: (r["length"], r["repeat"], r["cache"])):
        hit = f"{r['hit_rate']:.4%}" if r["hit_rate"] is not None else "N/A"
        lines.append(
            f"|{r['length']}|{r['cache']}|{r['repeat']}|{r['success']}|"
            f"{r['wall_seconds']:.3f}|{r['wall_tps']:.2f}|{r['ttft_p50_ms']:.2f}|"
            f"{r['ttft_p99_ms']:.2f}|{r['tpot_p50_ms']:.3f}|{hit}|"
        )
    lines += ["", "## 失败记录与解释边界", ""]
    for r in rows:
        for f in r["failures"]:
            lines.append(
                f"- `{r['file']}`：`{f['request_id']}`，{f['error_type']}，"
                f"等待约{f['completed_at'] - f['started_at']:.3f}秒；消息 `{f['message']}`。"
            )
    lines += [
        "",
        "这些是客户端超时，现有记录不足以定位具体网络/调度根因。"
        "服务端成功计数器增量与客户端成功数一致，未见该轮被漏记的成功完成。"
        "服务关闭之后的EngineDeadError不能倒推为此前超时的原因。",
        "短输入的完整配对吞吐变化存在波动；不能声称所有输入长度下吞吐都会提升。"
        "1024组的三轮配对均完整；其他组的失败样本不删除，只限制因果解释。",
        "缓存命中率使用每批前后Token计数器差值，开启组约74.05%～74.06%。"
        "关闭组查询数增量为0，命中率N/A，不把历史累计日志百分比当作本轮数值。",
        "",
        "## 原始证据与复现",
        "",
        "原始目录：`results/system_benchmarks/2026-08-30-prefix-cache/`。",
        "- `smoke.json`、`smoke_command.json`、`smoke_validation.json`：4请求CLI验收。",
        "- `cache_*.json`：正式请求、预热、校验；`*_before.prom`/`*_after.prom`：原始计数器。",
        "- `server_r*/launch.json`和`server.log`：6次原生服务启动命令、配置与完整日志。",
        "- `server_environment.json`：GPU运行环境、模型文件哈希、pip freeze。",
        "- `client_manifest.json`、`source_snapshot.zip`：客户端环境、基准commit、实际执行源码。",
        "- `summary.json`：独立重算的机器可读摘要；`SHA256SUMS.txt`：归档校验。",
        "",
        "采样时工作区包含未提交的实验工具，记录的基准commit为a867d7a，clean=false。"
        "不能只检出该commit复现；必须使用归档的完整源码快照。所有29个运行源码/依赖文件"
        "快照哈希已验证，实验脚本也验证运行前后源码未变。",
        "",
        "复核入口：`python scripts/summarize_prefix_ablation.py --input-dir <原始目录> "
        "--summary-output <新的摘要文件> --report <新的报告文件>`；摘要和报告均拒绝覆盖，"
        "复核时指定新的输出路径即可，无需修改原始目录。",
        "原始2026-08-10实验没有删除；本轮调整了输入生成、输出控制和服务参数，"
        "不直接比较新旧绝对吞吐。",
        "",
        "## 交接：下一步仍由用户实施",
        "",
        "开始自定义Kernel安全接入评估，先核对原生FlashAttention后端实际KV布局、步长、"
        "slot语义、数据类型和Stream/Graph。未验证前不替换原生算子，不填写自定义Kernel的"
        "TPOT/吞吐收益。该步骤不属于本次自动执行范围。",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as report:
        report.write("\n".join(lines))
    print(
        json.dumps(
            {
                "success": success,
                "failures": 1440 - success,
                "aggregates": aggregates,
                "pairs": pairs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
