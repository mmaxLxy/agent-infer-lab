"""Recompute paired results from immutable experiment records (stdlib only)."""

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from statistics import median

from run_suite import extract, fingerprint, numeric, save, validate_snapshot


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def one(root, stem):
    files = sorted(root.glob(stem + "-attempt*/complete.json"))
    if len(files) != 1:
        raise RuntimeError("Expected exactly one valid result for " + stem)
    return files[0].parent


def percent(new, old):
    return (new / old - 1) * 100


def gpu_summary(path):
    with path.open() as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    answer = {"samples": len(rows)}
    for key in (
        "memory.used [MiB]",
        "utilization.gpu [%]",
        "temperature.gpu",
        "clocks.current.graphics [MHz]",
    ):
        values = [float(re.search(r"[\d.]+", row[key]).group()) for row in rows]
        answer[key] = {"min": min(values), "median": median(values), "max": max(values)}
    return answer


def paired_accuracy(a, b, key="strict_correct"):
    if [r["test_index"] for r in a] != [r["test_index"] for r in b]:
        raise RuntimeError("Quality samples differ")
    changes = [int(y[key]) - int(x[key]) for x, y in zip(a, b, strict=True)]
    improved, regressed = changes.count(1), changes.count(-1)
    n = len(changes)
    rng = random.Random(20260904)
    boot = sorted(sum(rng.choices(changes, k=n)) / n * 100 for _ in range(20000))
    discordant = improved + regressed
    p = min(
        1.0,
        2
        * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
        / 2**discordant,
    )
    return {
        "fp16_correct": sum(r[key] for r in a),
        "fp8_correct": sum(r[key] for r in b),
        "count": n,
        "difference_percentage_points": sum(changes) / n * 100,
        "fp16_wrong_fp8_correct": improved,
        "fp16_correct_fp8_wrong": regressed,
        "paired_bootstrap_95_percentile_ci_pp": [boot[499], boot[19499]],
        "bootstrap_replicates": 20000,
        "bootstrap_seed": 20260904,
        "exact_mcnemar_two_sided_p": p,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    dataset = read(root / "data/cases.json")["cases"]
    expected_ids = [c["test_index"] for c in dataset]
    assert len(expected_ids) == len(set(expected_ids)) == 200
    canonical = read(root / "canonical-fp8-scales.json")
    validate_snapshot(canonical)
    scale_hash = fingerprint(canonical)
    result = {"scale_sha256": scale_hash, "performance": [], "quality": {}, "conditional_ppl": {}}
    sources = []
    fixed_requests = None
    metric_names = [
        "output_throughput_tokens_per_second",
        "ttft_p50_seconds",
        "tpot_p50_seconds",
        "e2e_p50_seconds",
    ]
    for repeat in range(1, 4):
        pair = {"round": repeat}
        for kind in ("fp16", "fp8"):
            path = one(root, f"performance-r{repeat}-{kind}")
            sources.append(path.name)
            raw = read(path / "measurement.json")
            measured = raw["result"]
            request_values = [
                (r["prompt_token_ids"], r["output_tokens"]) for r in measured["prepared_requests"]
            ]
            assert len(request_values) == 160 and all(
                len(t) == 1536 and n == 64 for t, n in request_values
            )
            if fixed_requests is None:
                fixed_requests = request_values
            assert request_values == fixed_requests, "Performance input token IDs differ"
            assert measured["successful_requests"] == 160 and measured["failed_requests"] == 0
            assert raw["configuration"]["workload"] == {
                "request_count": 160,
                "input_token_choices": [1536],
                "output_token_choices": [64],
                "shared_prefix_ratio": 0.75,
                "concurrency": 4,
                "seed": 20260903,
            }
            completion = read(path / "complete.json")
            assert measured["metrics"] == completion["result"]
            assert not completion["measurement_window"]["errors"]
            snaps = [
                read(path / name)
                for name in (
                    "scales-after-initialization-input.json",
                    "scales-after-warmup.json",
                    "scales-after-measurement.json",
                )
            ]
            assert len({fingerprint(s) for s in snaps}) == 1
            if kind == "fp8":
                assert fingerprint(snaps[-1]) == scale_hash
            log = (path / "server.log").read_text(errors="replace")
            capacities = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
            assert len(capacities) == 1
            pair[kind] = {
                **{k: measured["metrics"][k] for k in metric_names},
                "kv_capacity_tokens": int(capacities[0].replace(",", "")),
                "successful_requests": measured["successful_requests"],
                "gpu": gpu_summary(path / "gpu-during-measurement.csv"),
                "path": path.name,
            }
        pair["fp8_vs_fp16_percent"] = {
            k: percent(pair["fp8"][k], pair["fp16"][k]) for k in metric_names
        }
        result["performance"].append(pair)
    result["performance_summary"] = {
        k: {
            "fp16_median": median(p["fp16"][k] for p in result["performance"]),
            "fp8_median": median(p["fp8"][k] for p in result["performance"]),
            "paired_change_percent_median": median(
                p["fp8_vs_fp16_percent"][k] for p in result["performance"]
            ),
            "paired_change_percent_range": [
                min(p["fp8_vs_fp16_percent"][k] for p in result["performance"]),
                max(p["fp8_vs_fp16_percent"][k] for p in result["performance"]),
            ],
        }
        for k in metric_names
    }
    quality_rows = {}
    for kind in ("fp16", "fp8"):
        path = one(root, "quality-r1-" + kind)
        sources.append(path.name)
        rows = records(path / "gsm8k.jsonl")
        quality_rows[kind] = rows
        assert len(rows) == 200 and all(r["error"] is None for r in rows)
        assert [r["test_index"] for r in rows] == expected_ids
        for case, row in zip(dataset, rows, strict=True):
            text = row["response"]["choices"][0]["text"]
            assert row["gold"] == numeric(case["gold"])
            assert row["strict_correct"] == (extract(text) == numeric(case["gold"]))
            assert row["flexible_correct"] == (extract(text, False) == numeric(case["gold"]))
        summary = read(path / "complete.json")
        assert summary["strict_correct"] == sum(r["strict_correct"] for r in rows)
        assert summary["flexible_correct"] == sum(r["flexible_correct"] for r in rows)
        result["quality"][kind] = summary
        if kind == "fp8":
            assert summary["scale_sha256"] == scale_hash
        path = root / ("ppl-" + kind + "-final")
        sources.append(path.name)
        summary = read(path / "complete.json")
        nll_rows = records(path / "conditional-nll.jsonl")
        assert len(nll_rows) == 64 and summary["cases"] == 64
        validation = read(path / "raw-logprob-validation.json")
        assert abs(validation["difference"]) <= 1e-4 and validation["forced_logprob"] < 0
        assert validation["forced_token_ids"] == [validation["target"]]
        for row, case in zip(nll_rows, dataset[:64], strict=True):
            assert (
                row["test_index"] == case["test_index"]
                and row["token_ids"] == case["target_token_ids"]
            )
            assert len(row["raw_logprobs"]) == row["tokens"] == len(row["token_ids"])
            assert math.isclose(row["nll"], -sum(row["raw_logprobs"]), abs_tol=1e-8)
            assert math.isclose(row["decode_nll"], -sum(row["raw_logprobs"][1:]), abs_tol=1e-8)
        for prefix in ("", "decode_"):
            nll, count = (
                sum(r[prefix + "nll"] for r in nll_rows),
                sum(r[prefix + "tokens"] for r in nll_rows),
            )
            assert math.isclose(
                summary[prefix + "conditional_ppl"], math.exp(nll / count), abs_tol=1e-8
            )
        if kind == "fp8":
            assert summary["scale_sha256"] == scale_hash
        result["conditional_ppl"][kind] = summary
    result["quality"]["strict_paired"] = paired_accuracy(quality_rows["fp16"], quality_rows["fp8"])
    result["quality"]["flexible_paired"] = paired_accuracy(
        quality_rows["fp16"], quality_rows["fp8"], "flexible_correct"
    )
    result["conditional_ppl"]["fp8_vs_fp16_percent"] = {
        k: percent(result["conditional_ppl"]["fp8"][k], result["conditional_ppl"]["fp16"][k])
        for k in ("conditional_ppl", "decode_conditional_ppl")
    }
    result["sources"] = sources
    save(root / "verified-summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
