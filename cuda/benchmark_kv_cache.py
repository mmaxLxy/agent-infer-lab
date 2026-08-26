"""Run reproducible CUDA Event benchmarks for KV cache kernels."""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from load_extension import load_kv_cache_extension

_VERSIONS = ("v0", "v1", "v2")
_VERSION_ORDERS = (
    ("v0", "v1", "v2"),
    ("v1", "v2", "v0"),
    ("v2", "v0", "v1"),
    ("v2", "v1", "v0"),
    ("v1", "v0", "v2"),
    ("v0", "v2", "v1"),
)
_VARIANT_DEFINITIONS = {
    "v0": (
        "Scalar-per-token baseline: one thread loops "
        "over all Key and Value elements for one token."
    ),
    "v1": (
        "Flat-element coalesced implementation: adjacent "
        "threads process adjacent elements of one token."
    ),
    "v2": (
        "Vectorized chunk implementation: one thread "
        "moves eight contiguous FP16 values as one "
        "16-byte uint4 vector when alignment permits."
    ),
}
_COMPARISON_INTERPRETATIONS = {
    "v1_vs_v0": (
        "Combined benefit of fine-grained thread "
        "parallelism and coalesced access mapping."
    ),
    "v2_vs_v1": (
        "Combined benefit of 16-byte vector chunking, "
        "reduced thread count, address calculations, "
        "and memory instructions."
    ),
    "v2_vs_v0": (
        "Cumulative benefit from the scalar-per-token "
        "baseline to the V2 vectorized implementation."
    ),
}
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse benchmark configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark V0, V1, and V2 paged KV cache "
            "CUDA kernels with CUDA Events."
        )
    )

    parser.add_argument(
        "--operation",
        choices=("append", "gather", "both"),
        default="both",
    )
    parser.add_argument(
        "--num-tokens",
        type=positive_integer,
        default=2048,
    )
    parser.add_argument(
        "--num-blocks",
        type=positive_integer,
        default=256,
    )
    parser.add_argument(
        "--block-size",
        type=positive_integer,
        default=16,
    )
    parser.add_argument(
        "--num-kv-heads",
        type=positive_integer,
        default=2,
    )
    parser.add_argument(
        "--head-dim",
        type=positive_integer,
        default=64,
    )
    parser.add_argument(
        "--warmup",
        type=positive_integer,
        default=100,
    )
    parser.add_argument(
        "--repeats",
        type=positive_integer,
        default=1000,
    )
    parser.add_argument(
        "--banks",
        type=positive_integer,
        default=16,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260824,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    capacity_tokens = (
        args.num_blocks * args.block_size
    )
    if args.num_tokens > capacity_tokens:
        parser.error(
            "num_tokens cannot exceed "
            "num_blocks * block_size"
        )

    return args


def command_output(
    command: list[str],
) -> str:
    """Run a diagnostic command without failing the benchmark."""

    try:
        completed = subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"

    output = completed.stdout.strip()
    if output:
        return output

    error_output = completed.stderr.strip()
    if error_output:
        return error_output

    if completed.returncode == 0:
        return ""

    return f"exit_code={completed.returncode}"


def collect_environment() -> dict[str, Any]:
    """Collect the environment needed to reproduce the run."""

    properties = torch.cuda.get_device_properties(0)
    git_status = command_output(
        ["git", "status", "--short"]
    )

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": (
            properties.total_memory
        ),
        "compute_capability": list(
            torch.cuda.get_device_capability(0)
        ),
        "driver_version": command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ),
        "nvcc_version": command_output(
            ["nvcc", "--version"]
        ),
        "git_commit": command_output(
            ["git", "rev-parse", "HEAD"]
        ),
        "git_status_short": git_status,
        "git_worktree_clean": not bool(git_status),
        "environment_variables": {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "CUDACXX": os.environ.get("CUDACXX"),
            "TORCH_EXTENSIONS_DIR": os.environ.get(
                "TORCH_EXTENSIONS_DIR"
            ),
            "TORCH_CUDA_ARCH_LIST": os.environ.get(
                "TORCH_CUDA_ARCH_LIST"
            ),
            "MAX_JOBS": os.environ.get("MAX_JOBS"),
        },
    }


def build_banks(
    args: argparse.Namespace,
) -> list[dict[str, torch.Tensor]]:
    """Create deterministic input and output tensor banks."""

    generator = torch.Generator(
        device="cuda"
    )
    generator.manual_seed(args.seed)

    capacity_tokens = (
        args.num_blocks * args.block_size
    )
    cache_shape = (
        args.num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
    )
    input_shape = (
        args.num_tokens,
        args.num_kv_heads,
        args.head_dim,
    )

    banks: list[dict[str, torch.Tensor]] = []

    for _ in range(args.banks):
        keys = torch.randn(
            input_shape,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        values = torch.randn(
            input_shape,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )

        key_cache = torch.randn(
            cache_shape,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        value_cache = torch.randn(
            cache_shape,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )

        append_slots = torch.randperm(
            capacity_tokens,
            dtype=torch.int64,
            device="cuda",
            generator=generator,
        )[: args.num_tokens].contiguous()

        gather_slots = torch.randint(
            low=0,
            high=capacity_tokens,
            size=(args.num_tokens,),
            dtype=torch.int64,
            device="cuda",
            generator=generator,
        )

        gathered_keys = torch.empty(
            input_shape,
            dtype=torch.float16,
            device="cuda",
        )
        gathered_values = torch.empty_like(
            gathered_keys
        )

        banks.append(
            {
                "keys": keys,
                "values": values,
                "key_cache": key_cache,
                "value_cache": value_cache,
                "append_slots": append_slots,
                "gather_slots": gather_slots,
                "gathered_keys": gathered_keys,
                "gathered_values": gathered_values,
            }
        )

    return banks


def build_functions(
    extension: object,
) -> dict[str, dict[str, Any]]:
    """Resolve benchmark functions before timing starts."""

    return {
        "append": {
            "v0": (
                extension
                .append_kv_cache_v0_unchecked_
            ),
            "v1": (
                extension
                .append_kv_cache_v1_unchecked_
            ),
            "v2": (
                extension
                .append_kv_cache_v2_unchecked_
            ),
        },
        "gather": {
            "v0": (
                extension
                .gather_kv_cache_v0_unchecked_out_
            ),
            "v1": (
                extension
                .gather_kv_cache_v1_unchecked_out_
            ),
            "v2": (
                extension
                .gather_kv_cache_v2_unchecked_out_
            ),
        },
    }


def operation_arguments(
    operation: str,
    bank: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Return prebuilt arguments for one operation."""

    if operation == "append":
        return (
            bank["keys"],
            bank["values"],
            bank["key_cache"],
            bank["value_cache"],
            bank["append_slots"],
        )

    return (
        bank["key_cache"],
        bank["value_cache"],
        bank["gather_slots"],
        bank["gathered_keys"],
        bank["gathered_values"],
    )


def version_order(
    index: int,
) -> tuple[str, ...]:
    """Rotate all version orders to balance position effects."""

    return _VERSION_ORDERS[
        index % len(_VERSION_ORDERS)
    ]


def warm_up(
    *,
    operation: str,
    functions: dict[str, Any],
    banks: list[dict[str, torch.Tensor]],
    warmup: int,
) -> None:
    """Warm up all variants before recording samples."""

    for index in range(warmup):
        bank = banks[index % len(banks)]

        for version in version_order(index):
            arguments = operation_arguments(
                operation,
                bank,
            )
            functions[version](*arguments)

    torch.cuda.synchronize()


def nearest_rank(
    values: list[float],
    percentile: float,
) -> float:
    """Calculate a nearest-rank percentile."""

    ordered = sorted(values)
    index = (
        math.ceil(percentile * len(ordered))
        - 1
    )
    return ordered[index]


def summarize_samples(
    samples: list[dict[str, Any]],
    useful_bytes: int,
) -> dict[str, Any]:
    """Summarize raw CUDA Event measurements."""

    milliseconds = [
        float(sample["milliseconds"])
        for sample in samples
    ]

    p50_ms = nearest_rank(
        milliseconds,
        0.50,
    )
    p95_ms = nearest_rank(
        milliseconds,
        0.95,
    )
    p99_ms = nearest_rank(
        milliseconds,
        0.99,
    )

    standard_deviation = 0.0
    if len(milliseconds) >= 2:
        standard_deviation = statistics.stdev(
            milliseconds
        )

    return {
        "sample_count": len(milliseconds),
        "mean_ms": statistics.fmean(
            milliseconds
        ),
        "standard_deviation_ms": (
            standard_deviation
        ),
        "minimum_ms": min(milliseconds),
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "maximum_ms": max(milliseconds),
        "effective_bandwidth_gbps_at_p50": (
            useful_bytes
            / (p50_ms / 1000.0)
            / 1_000_000_000.0
        ),
    }


def compare_versions(
    summaries: dict[str, dict[str, Any]],
    *,
    baseline: str,
    candidate: str,
    interpretation: str,
) -> dict[str, Any]:
    """Compare one candidate with one baseline at P50."""

    baseline_p50 = summaries[
        baseline
    ]["p50_ms"]
    candidate_p50 = summaries[
        candidate
    ]["p50_ms"]

    return {
        "baseline": baseline,
        "candidate": candidate,
        "interpretation": interpretation,
        "p50_speedup": (
            baseline_p50 / candidate_p50
        ),
        "p50_latency_reduction_percent": (
            (
                1.0
                - candidate_p50 / baseline_p50
            )
            * 100.0
        ),
    }


def measure_operation(
    *,
    operation: str,
    functions: dict[str, Any],
    banks: list[dict[str, torch.Tensor]],
    warmup: int,
    repeats: int,
    useful_bytes: int,
) -> dict[str, Any]:
    """Measure V0, V1, and V2 with interleaved CUDA Events."""

    warm_up(
        operation=operation,
        functions=functions,
        banks=banks,
        warmup=warmup,
    )

    event_records: list[dict[str, Any]] = []

    for repeat_index in range(repeats):
        bank_index = repeat_index % len(banks)
        bank = banks[bank_index]

        for order_index, version in enumerate(
            version_order(repeat_index)
        ):
            arguments = operation_arguments(
                operation,
                bank,
            )
            start_event = torch.cuda.Event(
                enable_timing=True
            )
            end_event = torch.cuda.Event(
                enable_timing=True
            )

            start_event.record()
            functions[version](*arguments)
            end_event.record()

            event_records.append(
                {
                    "repeat_index": repeat_index,
                    "order_index": order_index,
                    "bank_index": bank_index,
                    "version": version,
                    "start_event": start_event,
                    "end_event": end_event,
                }
            )

    torch.cuda.synchronize()

    samples_by_version: dict[
        str,
        list[dict[str, Any]],
    ] = {
        version: []
        for version in _VERSIONS
    }

    for record in event_records:
        milliseconds = record[
            "start_event"
        ].elapsed_time(
            record["end_event"]
        )

        sample = {
            "repeat_index": record[
                "repeat_index"
            ],
            "order_index": record[
                "order_index"
            ],
            "bank_index": record[
                "bank_index"
            ],
            "milliseconds": milliseconds,
            "effective_bandwidth_gbps": (
                useful_bytes
                / (milliseconds / 1000.0)
                / 1_000_000_000.0
            ),
        }

        samples_by_version[
            record["version"]
        ].append(sample)

    summaries = {
        version: summarize_samples(
            samples_by_version[version],
            useful_bytes,
        )
        for version in _VERSIONS
    }

    comparisons = {
        "v1_vs_v0": compare_versions(
            summaries,
            baseline="v0",
            candidate="v1",
            interpretation=(
                _COMPARISON_INTERPRETATIONS[
                    "v1_vs_v0"
                ]
            ),
        ),
        "v2_vs_v1": compare_versions(
            summaries,
            baseline="v1",
            candidate="v2",
            interpretation=(
                _COMPARISON_INTERPRETATIONS[
                    "v2_vs_v1"
                ]
            ),
        ),
        "v2_vs_v0": compare_versions(
            summaries,
            baseline="v0",
            candidate="v2",
            interpretation=(
                _COMPARISON_INTERPRETATIONS[
                    "v2_vs_v0"
                ]
            ),
        ),
    }

    return {
        "operation": operation,
        "useful_bytes_per_call": useful_bytes,
        "samples": samples_by_version,
        "summary": summaries,
        "comparisons": comparisons,
    }


def resolve_output_path(
    requested: Path | None,
) -> Path:
    """Resolve the raw-result JSON path."""

    if requested is not None:
        if requested.is_absolute():
            return requested
        return _PROJECT_ROOT / requested

    timestamp = datetime.now(
        UTC
    ).strftime("%Y%m%dT%H%M%SZ")

    return (
        _PROJECT_ROOT
        / "results"
        / "kernel_benchmarks"
        / (
            "kv_cache_v0_v1_v2_"
            f"{timestamp}.json"
        )
    )


def print_result(
    result: dict[str, Any],
) -> None:
    """Print a compact human-readable summary."""

    operation = result["operation"]
    print(f"{operation}:")

    for version in _VERSIONS:
        summary = result["summary"][version]
        print(
            f"  {version}: "
            f"p50={summary['p50_ms']:.6f} ms, "
            f"p95={summary['p95_ms']:.6f} ms, "
            f"p99={summary['p99_ms']:.6f} ms, "
            "effective_bandwidth="
            f"{summary['effective_bandwidth_gbps_at_p50']:.3f} GB/s"
        )

    for comparison_name in (
        "v1_vs_v0",
        "v2_vs_v1",
        "v2_vs_v0",
    ):
        comparison = result[
            "comparisons"
        ][comparison_name]

        baseline = comparison[
            "baseline"
        ].upper()
        candidate = comparison[
            "candidate"
        ].upper()

        print(
            f"  {candidate}/{baseline} "
            "p50 speedup: "
            f"{comparison['p50_speedup']:.3f}x"
        )
        print(
            f"  {candidate} p50 latency "
            "reduction: "
            f"{comparison['p50_latency_reduction_percent']:.2f}%"
        )


def main() -> None:
    """Run the requested benchmark and persist raw samples."""

    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA must be available for this benchmark"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    extension = load_kv_cache_extension()
    functions = build_functions(extension)
    banks = build_banks(args)

    operations = (
        ("append", "gather")
        if args.operation == "both"
        else (args.operation,)
    )

    element_count = (
        args.num_tokens
        * args.num_kv_heads
        * args.head_dim
    )
    element_size_bytes = torch.tensor(
        [],
        dtype=torch.float16,
    ).element_size()

    useful_bytes = (
        4
        * element_count
        * element_size_bytes
    )

    results = []

    for operation in operations:
        result = measure_operation(
            operation=operation,
            functions=functions[operation],
            banks=banks,
            warmup=args.warmup,
            repeats=args.repeats,
            useful_bytes=useful_bytes,
        )
        results.append(result)
        print_result(result)

    output_path = resolve_output_path(
        args.output
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 2,
        "created_at_utc": datetime.now(
            UTC
        ).isoformat(),
        "measurement_scope": (
            "CUDA Event elapsed time around "
            "unchecked CUDA calls with "
            "preallocated Gather outputs"
        ),
        "bandwidth_definition": (
            "Logical useful Key/Value bytes "
            "read and written per call; "
            "metadata traffic is excluded"
        ),
        "variant_definitions": (
            _VARIANT_DEFINITIONS
        ),
        "comparison_interpretations": (
            _COMPARISON_INTERPRETATIONS
        ),
        "command": [sys.executable, *sys.argv],
        "configuration": {
            "operation": args.operation,
            "versions": list(_VERSIONS),
            "version_orders": [
                list(order)
                for order in _VERSION_ORDERS
            ],
            "num_tokens": args.num_tokens,
            "num_blocks": args.num_blocks,
            "block_size": args.block_size,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "banks": args.banks,
            "seed": args.seed,
            "cache_capacity_tokens": (
                args.num_blocks
                * args.block_size
            ),
            "dtype": "torch.float16",
        },
        "environment": collect_environment(),
        "results": results,
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"raw_result: {output_path}")


if __name__ == "__main__":
    main()
