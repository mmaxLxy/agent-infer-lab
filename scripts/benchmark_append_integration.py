"""Benchmark native vLLM and custom Append on observed serving shapes.

This script is intentionally GPU-only. It compares the same FP16 strided
inputs and NHD cache layout seen in the archived vLLM verification run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import time
from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import vllm
from vllm import _custom_ops as ops

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = (
    ROOT
    / "results"
    / "system_benchmarks"
    / "2026-08-31-append-e2e"
    / "verification"
    / "hook"
    / "append_verify_2533.jsonl"
)


def load_custom_extension():
    path = ROOT / "cuda" / "vllm_append_adapter.py"
    spec = importlib.util.spec_from_file_location("ail_append_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_adapter()


def observed_shapes(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "verified" and row.get("valid_tokens", 0) > 0:
                counts[int(row["mapped_rows"])] += 1
    if not counts:
        raise RuntimeError(f"no serving shapes found in {path}")
    return counts


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def benchmark(
    operation: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iterations):
        operation()
    end_event.record()
    end_event.synchronize()
    device_us = start_event.elapsed_time(end_event) * 1000.0 / iterations

    torch.cuda.synchronize()
    wall_start = time.perf_counter_ns()
    for _ in range(iterations):
        operation()
    enqueue_end = time.perf_counter_ns()
    torch.cuda.synchronize()
    sync_end = time.perf_counter_ns()
    return {
        "device_timeline_us_per_call": device_us,
        "host_enqueue_us_per_call": (enqueue_end - wall_start) / 1000.0 / iterations,
        "wall_with_final_sync_us_per_call": (sync_end - wall_start) / 1000.0 / iterations,
    }


def summarize(rounds: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for metric in rounds[0]:
        values = [row[metric] for row in rounds]
        result[metric] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "p20": percentile(values, 0.2),
            "p80": percentile(values, 0.8),
        }
    return result


def native_python_path(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    key_cache, value_cache = cache.unbind(1)
    ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slots,
        "auto",
        scale,
        scale,
    )


def custom_python_path(
    extension,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    if key.dtype != torch.float16 or value.dtype != torch.float16:
        raise RuntimeError("expected FP16 inputs")
    if cache.dtype != torch.float16 or cache.ndim != 5:
        raise RuntimeError("expected FP16 five-dimensional cache")
    if not key.is_cuda or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("expected eager CUDA execution")
    if slots.ndim != 1 or slots.dtype != torch.int64:
        raise RuntimeError("expected one-dimensional int64 slots")
    key_cache, value_cache = cache.unbind(1)
    extension.append_fp16_(key, value, key_cache, value_cache, slots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations-small", type=int, default=2000)
    parser.add_argument("--iterations-large", type=int, default=500)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--tpot-ms", type=float, default=11.97)
    parser.add_argument("--ttft-ms", type=float, default=125.78)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output.exists():
        raise FileExistsError(args.output)

    torch.manual_seed(20260902)
    torch.cuda.manual_seed_all(20260902)
    extension = load_custom_extension()
    shape_counts = observed_shapes(args.audit)
    shapes = sorted({*shape_counts, 2048})

    results: dict[str, dict[str, object]] = {}
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    for tokens in shapes:
        # The archived real input stride was [1152, 64, 1]. Slicing an
        # [tokens, 18, 64] QKV backing tensor reproduces that stride.
        qkv = torch.randn(tokens, 18, 64, device="cuda", dtype=torch.float16)
        key = qkv[:, 14:16, :]
        value = qkv[:, 16:18, :]
        cache = torch.zeros(
            max(256, (tokens + 15) // 16),
            2,
            16,
            2,
            64,
            device="cuda",
            dtype=torch.float16,
        )
        slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
        key_cache, value_cache = cache.unbind(1)

        expected = torch.zeros_like(cache)
        expected_key, expected_value = expected.unbind(1)
        ops.reshape_and_cache_flash(
            key,
            value,
            expected_key,
            expected_value,
            slots,
            "auto",
            scale,
            scale,
        )
        actual = torch.zeros_like(cache)
        actual_key, actual_value = actual.unbind(1)
        extension.append_fp16_(key, value, actual_key, actual_value, slots)
        torch.cuda.synchronize()
        if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
            raise RuntimeError(f"native/custom mismatch for tokens={tokens}")

        iterations = args.iterations_small if tokens <= 129 else args.iterations_large
        operations = {
            "native_direct": partial(
                ops.reshape_and_cache_flash,
                key,
                value,
                key_cache,
                value_cache,
                slots,
                "auto",
                scale,
                scale,
            ),
            "custom_direct": partial(
                extension.append_fp16_,
                key,
                value,
                key_cache,
                value_cache,
                slots,
            ),
            "native_python_path": partial(native_python_path, key, value, cache, slots, scale),
            "custom_python_path": partial(custom_python_path, extension, key, value, cache, slots),
        }
        rounds = {name: [] for name in operations}
        names = list(operations)
        for round_index in range(args.rounds):
            ordered_names = names[round_index % len(names) :] + names[: round_index % len(names)]
            if round_index % 2:
                ordered_names.reverse()
            for name in ordered_names:
                rounds[name].append(
                    benchmark(
                        operations[name],
                        warmup=args.warmup,
                        iterations=iterations,
                    )
                )
        results[str(tokens)] = {
            "iterations": iterations,
            "correctness": "bitwise_equal",
            "input_shape": list(key.shape),
            "input_stride": list(key.stride()),
            "cache_shape": list(cache.shape),
            "cache_stride": list(cache.stride()),
            "raw_rounds": rounds,
            "summary": {name: summarize(rows) for name, rows in rounds.items()},
        }

    def metric(tokens: int, operation: str, name: str) -> float:
        return float(
            results[str(tokens)]["summary"][operation][name]["median"]  # type: ignore[index]
        )

    decode_counts = {tokens: count for tokens, count in shape_counts.items() if tokens <= 2}
    prefill_counts = {tokens: count for tokens, count in shape_counts.items() if tokens > 2}

    def weighted(counts: dict[int, int], operation: str, name: str) -> float:
        total = sum(counts.values())
        weighted_sum = sum(
            metric(tokens, operation, name) * count for tokens, count in counts.items()
        )
        return weighted_sum / total

    decode_native_us = weighted(
        decode_counts, "native_python_path", "wall_with_final_sync_us_per_call"
    )
    decode_custom_us = weighted(
        decode_counts, "custom_python_path", "wall_with_final_sync_us_per_call"
    )
    prefill_native_us = weighted(
        prefill_counts, "native_python_path", "wall_with_final_sync_us_per_call"
    )
    derived = {
        "observed_shape_counts": dict(sorted(shape_counts.items())),
        "observed_calls": sum(shape_counts.values()),
        "decode_shape_counts": decode_counts,
        "prefill_shape_counts": prefill_counts,
        "decode_native_weighted_us_per_layer_call": decode_native_us,
        "decode_custom_weighted_us_per_layer_call": decode_custom_us,
        "decode_native_append_us_per_model_step": decode_native_us * args.layers,
        "decode_custom_append_us_per_model_step": decode_custom_us * args.layers,
        "native_append_share_of_tpot_percent": (
            decode_native_us * args.layers / (args.tpot_ms * 1000.0) * 100.0
        ),
        "custom_append_share_of_tpot_percent": (
            decode_custom_us * args.layers / (args.tpot_ms * 1000.0) * 100.0
        ),
        "maximum_tpot_reduction_if_native_append_were_free_percent": (
            decode_native_us * args.layers / (args.tpot_ms * 1000.0) * 100.0
        ),
        "expected_tpot_change_from_custom_minus_native_percent": (
            (decode_custom_us - decode_native_us) * args.layers / (args.tpot_ms * 1000.0) * 100.0
        ),
        "prefill_native_weighted_us_per_layer_call": prefill_native_us,
        "native_append_share_of_ttft_percent": (
            prefill_native_us * args.layers / (args.ttft_ms * 1000.0) * 100.0
        ),
        "headline_v0_us_tokens_2048": 62.464,
        "headline_v3_us_tokens_2048": 5.216,
        "headline_v3_vs_v0_speedup": 11.98,
    }
    payload = {
        "schema_version": 1,
        "purpose": "M1 Append end-to-end attribution",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "measurement": {
            "rounds": args.rounds,
            "warmup_per_round": args.warmup,
            "iterations_small": args.iterations_small,
            "iterations_large": args.iterations_large,
            "layers": args.layers,
            "archived_native_tpot_p50_ms": args.tpot_ms,
            "archived_native_ttft_p50_ms": args.ttft_ms,
            "timing_note": (
                "CUDA Event measures device timeline for repeated calls; host enqueue and "
                "wall measurements use perf_counter around the same repeated calls."
            ),
        },
        "results": results,
        "derived": derived,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(derived, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
