"""Verify and recompute all formal FP16/AWQ results."""

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from statistics import median

from run_suite import extract, numeric, save


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def one(root, stem):
    found = sorted(root.glob(stem + "-attempt*/complete.json"))
    if len(found) != 1:
        raise RuntimeError("Expected one completed " + stem)
    return found[0].parent


def pct(new, old):
    return (new / old - 1) * 100


def gpu(path):
    with path.open() as f:
        data = list(csv.DictReader(f, skipinitialspace=True))
    out = {"samples": len(data)}
    for key in (
        "memory.used [MiB]",
        "utilization.gpu [%]",
        "temperature.gpu",
        "clocks.current.graphics [MHz]",
    ):
        values = [float(re.search(r"[\d.]+", r[key]).group()) for r in data]
        out[key] = {"min": min(values), "median": median(values), "max": max(values)}
    return out


def paired(a, b, field):
    assert [x["test_index"] for x in a] == [x["test_index"] for x in b]
    changes = [int(y[field]) - int(x[field]) for x, y in zip(a, b, strict=True)]
    improved, regressed, n = changes.count(1), changes.count(-1), len(changes)
    rng = random.Random(20260905)
    boot = sorted(sum(rng.choices(changes, k=n)) / n * 100 for _ in range(20000))
    discordant = improved + regressed
    p = min(
        1.0,
        2
        * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
        / 2**discordant,
    )
    return {
        "fp16_correct": sum(x[field] for x in a),
        "awq_correct": sum(x[field] for x in b),
        "count": n,
        "difference_percentage_points": sum(changes) / n * 100,
        "fp16_wrong_awq_correct": improved,
        "fp16_correct_awq_wrong": regressed,
        "paired_bootstrap_95_percentile_ci_pp": [boot[499], boot[19499]],
        "bootstrap_replicates": 20000,
        "bootstrap_seed": 20260905,
        "exact_mcnemar_two_sided_p": p,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    cases = read(root / "data/cases.json")["cases"]
    expected = [x["test_index"] for x in cases]
    assert len(expected) == len(set(expected)) == 200
    metrics = (
        "output_throughput_tokens_per_second",
        "ttft_p50_seconds",
        "tpot_p50_seconds",
        "e2e_p50_seconds",
    )
    inventory = read(root / "model-inventory.json")
    compact = {
        kind: {key: value for key, value in inventory[kind].items() if key != "tensors"}
        for kind in ("fp16", "awq")
    }
    compact["weight_file_reduction_percent"] = inventory["weight_file_reduction_percent"]
    result = {"performance": [], "quality": {}, "conditional_ppl": {}, "model_inventory": compact}
    fixed_requests = None
    sources = []
    for repeat in range(1, 4):
        pair = {"round": repeat}
        for kind in ("fp16", "awq"):
            path = one(root, f"performance-r{repeat}-{kind}")
            sources.append(path.name)
            raw = read(path / "measurement.json")
            measured = raw["result"]
            assert measured["successful_requests"] == 160 and measured["failed_requests"] == 0
            request_values = [
                (r["prompt_token_ids"], r["output_tokens"]) for r in measured["prepared_requests"]
            ]
            assert len(request_values) == 160 and all(
                len(t) == 1536 and n == 64 for t, n in request_values
            )
            if fixed_requests is None:
                fixed_requests = request_values
            assert request_values == fixed_requests
            complete = read(path / "complete.json")
            assert complete["result"] == measured["metrics"]
            assert not complete["measurement_window"]["errors"]
            log = (path / "server.log").read_text(errors="replace")
            cap = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
            memory = re.findall(r"Model loading took ([\d.]+) GiB memory", log)
            available = re.findall(r"Available KV cache memory: ([\d.]+) GiB", log)
            assert len(cap) == len(memory) == len(available) == 1
            if kind == "awq":
                assert "Using awq_marlin kernel" in log and "quantization=awq_marlin" in log
            pair[kind] = {
                **{k: measured["metrics"][k] for k in metrics},
                "kv_capacity_tokens": int(cap[0].replace(",", "")),
                "model_loading_memory_gib": float(memory[0]),
                "available_kv_memory_gib": float(available[0]),
                "successful_requests": 160,
                "gpu": gpu(path / "gpu-during-measurement.csv"),
                "path": path.name,
            }
        pair["awq_vs_fp16_percent"] = {k: pct(pair["awq"][k], pair["fp16"][k]) for k in metrics}
        pair["kv_capacity_change_percent"] = pct(
            pair["awq"]["kv_capacity_tokens"], pair["fp16"]["kv_capacity_tokens"]
        )
        result["performance"].append(pair)
    result["performance_summary"] = {
        k: {
            "fp16_median": median(x["fp16"][k] for x in result["performance"]),
            "awq_median": median(x["awq"][k] for x in result["performance"]),
            "paired_change_percent_median": median(
                x["awq_vs_fp16_percent"][k] for x in result["performance"]
            ),
            "paired_change_percent_range": [
                min(x["awq_vs_fp16_percent"][k] for x in result["performance"]),
                max(x["awq_vs_fp16_percent"][k] for x in result["performance"]),
            ],
        }
        for k in metrics
    }
    quality = {}
    for kind in ("fp16", "awq"):
        path = one(root, "quality-r1-" + kind)
        sources.append(path.name)
        data = rows(path / "gsm8k.jsonl")
        assert (
            len(data) == 200
            and all(x["error"] is None for x in data)
            and [x["test_index"] for x in data] == expected
        )
        for case, row in zip(cases, data, strict=True):
            text = row["response"]["choices"][0]["text"]
            gold = numeric(case["gold"])
            assert row["gold"] == gold and row["strict_correct"] == (extract(text) == gold)
            assert row["flexible_correct"] == (extract(text, False) == gold)
        complete = read(path / "complete.json")
        assert complete["strict_correct"] == sum(x["strict_correct"] for x in data)
        assert complete["flexible_correct"] == sum(x["flexible_correct"] for x in data)
        quality[kind] = data
        result["quality"][kind] = complete
        ppl = root / ("ppl-" + kind + "-final")
        sources.append(ppl.name)
        summary = read(ppl / "complete.json")
        nll = rows(ppl / "conditional-nll.jsonl")
        assert len(nll) == summary["cases"] == 64
        valid = read(ppl / "raw-logprob-validation.json")
        assert abs(valid["difference"]) <= 1e-4 and valid["forced_token_ids"] == [valid["target"]]
        for row, case in zip(nll, cases[:64], strict=True):
            assert (
                row["test_index"] == case["test_index"]
                and row["token_ids"] == case["target_token_ids"]
            )
            assert math.isclose(row["nll"], -sum(row["raw_logprobs"]), abs_tol=1e-8)
            assert math.isclose(row["decode_nll"], -sum(row["raw_logprobs"][1:]), abs_tol=1e-8)
        for prefix in ("", "decode_"):
            total = sum(x[prefix + "nll"] for x in nll)
            count = sum(x[prefix + "tokens"] for x in nll)
            assert math.isclose(
                summary[prefix + "conditional_ppl"], math.exp(total / count), abs_tol=1e-8
            )
        result["conditional_ppl"][kind] = summary
    result["quality"]["strict_paired"] = paired(quality["fp16"], quality["awq"], "strict_correct")
    result["quality"]["flexible_paired"] = paired(
        quality["fp16"], quality["awq"], "flexible_correct"
    )
    result["conditional_ppl"]["awq_vs_fp16_percent"] = {
        k: pct(result["conditional_ppl"]["awq"][k], result["conditional_ppl"]["fp16"][k])
        for k in ("conditional_ppl", "decode_conditional_ppl")
    }
    result["sources"] = sources
    save(root / "verified-summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
