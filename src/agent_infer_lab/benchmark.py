
"""Fixed-concurrency execution for reproducible vLLM benchmarks."""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from agent_infer_lab.metrics import MetricsSummary, RequestTrace, summarize_metrics
from agent_infer_lab.prompting import PreparedRequest, prepare_requests
from agent_infer_lab.result_storage import collect_run_metadata, write_json_result
from agent_infer_lab.vllm_client import VllmClient
from agent_infer_lab.workloads import WorkloadConfig, generate_workload

SendRequest = Callable[[PreparedRequest], RequestTrace]
Clock = Callable[[], float]


@dataclass(frozen=True)
class RequestFailure:
    """Failure information and execution timestamps for one request."""

    request_id: str
    error_type: str
    message: str
    started_at: float | None = None
    completed_at: float | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated metrics, raw requests, traces, and failures."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    metrics: MetricsSummary | None
    failures: tuple[RequestFailure, ...]
    traces: tuple[RequestTrace, ...] = ()
    prepared_requests: tuple[PreparedRequest, ...] = ()
    wall_duration_seconds: float | None = None


RequestOutcome = RequestTrace | RequestFailure


def _request_failure(
    request: PreparedRequest,
    error: Exception,
    *,
    started_at: float,
    completed_at: float,
) -> RequestFailure:
    """Convert an exception chain into stable failure information."""

    root_error: BaseException = error
    while root_error.__cause__ is not None:
        root_error = root_error.__cause__

    return RequestFailure(
        request_id=request.request_id,
        error_type=type(root_error).__name__,
        message=str(root_error) or repr(root_error),
        started_at=started_at,
        completed_at=completed_at,
    )


def run_benchmark(
    requests: tuple[PreparedRequest, ...],
    *,
    concurrency: int,
    send_request: SendRequest,
    clock: Clock = time.perf_counter,
) -> BenchmarkResult:
    """Execute requests and retain raw data for result persistence."""

    if not requests:
        raise ValueError("requests must not be empty")
    if (
        not isinstance(concurrency, int)
        or isinstance(concurrency, bool)
        or concurrency <= 0
    ):
        raise ValueError("concurrency must be a positive integer")
    if concurrency > len(requests):
        raise ValueError("concurrency cannot exceed request count")

    request_ids = tuple(request.request_id for request in requests)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request IDs must be unique")

    def execute_one(request: PreparedRequest) -> RequestOutcome:
        started_at = clock()

        try:
            trace = send_request(request)
            if not isinstance(trace, RequestTrace):
                raise TypeError("send_request must return a RequestTrace")
            if trace.request_id != request.request_id:
                raise ValueError(
                    "returned trace request_id must match the request"
                )
            return trace
        except Exception as error:  # noqa: BLE001
            return _request_failure(
                request,
                error,
                started_at=started_at,
                completed_at=clock(),
            )

    batch_started_at = clock()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        outcomes = tuple(executor.map(execute_one, requests))

    wall_duration_seconds = clock() - batch_started_at

    traces = tuple(
        outcome
        for outcome in outcomes
        if isinstance(outcome, RequestTrace)
    )
    failures = tuple(
        outcome
        for outcome in outcomes
        if isinstance(outcome, RequestFailure)
    )
    metrics = summarize_metrics(traces) if traces else None

    return BenchmarkResult(
        total_requests=len(requests),
        successful_requests=len(traces),
        failed_requests=len(failures),
        success_rate=len(traces) / len(requests),
        metrics=metrics,
        failures=failures,
        traces=traces,
        prepared_requests=requests,
        wall_duration_seconds=wall_duration_seconds,
    )


def execute_benchmark(
    config: WorkloadConfig,
    client: VllmClient,
) -> BenchmarkResult:
    """Prepare exact prompts, execute requests, and summarize results."""

    specs = generate_workload(config)
    requests = prepare_requests(specs, client.tokenize)
    return run_benchmark(
        requests,
        concurrency=config.concurrency,
        send_request=client.stream_completion,
    )


def _wall_output_throughput(
    result: BenchmarkResult,
) -> float | None:
    """Count successful output tokens over the full execution duration."""

    duration = result.wall_duration_seconds
    if duration is None or duration <= 0:
        return None

    total_output_tokens = (
        result.metrics.total_output_tokens
        if result.metrics is not None
        else 0
    )
    return total_output_tokens / duration


def build_result_payload(
    config: WorkloadConfig,
    client: VllmClient,
    result: BenchmarkResult,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Build a versioned document containing configuration and raw results."""

    serialized_result = asdict(result)
    serialized_result["wall_output_throughput_tokens_per_second"] = (
        _wall_output_throughput(result)
    )

    return {
        "schema_version": 1,
        "experiment_type": "vllm_fixed_concurrency",
        "configuration": {
            "workload": asdict(config),
            "client": {
                "base_url": client.base_url,
                "model": client.model,
                "timeout_seconds": client.timeout,
            },
        },
        "metadata": metadata,
        "metric_definitions": {
            "timestamps": (
                "Client-side time.perf_counter readings in seconds; "
                "not UTC timestamps or server-side GPU timings."
            ),
            "metrics_scope": (
                "The metrics object uses successful requests only."
            ),
            "wall_duration_seconds": (
                "Time from executor creation through executor shutdown, "
                "including successful and failed request execution; "
                "excluding prompt preparation, metadata collection, "
                "result aggregation, and JSON writing."
            ),
            "wall_output_throughput_tokens_per_second": (
                "Output tokens from successful requests divided by "
                "wall_duration_seconds. Partial output from failed "
                "requests is not counted."
            ),
            "failure_timestamps": (
                "Worker execution start and exception capture time; "
                "executor queue wait is not part of an individual "
                "failure duration."
            ),
        },
        "result": serialized_result,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed-concurrency streaming vLLM benchmark."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=_positive_int, default=20)
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument(
        "--input-tokens",
        type=_positive_int,
        nargs="+",
        default=[128],
    )
    parser.add_argument(
        "--output-tokens",
        type=_positive_int,
        nargs="+",
        default=[32],
    )
    parser.add_argument("--shared-prefix-ratio", type=_ratio, default=0.5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Save configuration and raw results to a new JSON file.",
    )
    return parser


def _format_optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _print_metrics(metrics: MetricsSummary | None) -> None:
    if metrics is None:
        print("successful_metrics: N/A")
        return

    print("metrics_scope: successful_requests_only")
    print(f"output_tokens: {metrics.total_output_tokens}")
    print(f"duration_seconds: {metrics.duration_seconds:.6f}")
    print(
        "output_throughput_tokens_per_second: "
        f"{metrics.output_throughput_tokens_per_second:.6f}"
    )
    print(f"ttft_p50_seconds: {metrics.ttft_p50_seconds:.6f}")
    print(f"ttft_p99_seconds: {metrics.ttft_p99_seconds:.6f}")
    print(f"tpot_p50_seconds: {_format_optional(metrics.tpot_p50_seconds)}")
    print(f"tpot_p99_seconds: {_format_optional(metrics.tpot_p99_seconds)}")
    print(f"e2e_p50_seconds: {metrics.e2e_p50_seconds:.6f}")
    print(f"e2e_p99_seconds: {metrics.e2e_p99_seconds:.6f}")


def _print_failures(failures: tuple[RequestFailure, ...]) -> None:
    if not failures:
        print("failure_types: none")
        return

    failure_counts = Counter(failure.error_type for failure in failures)
    print("failure_types:")
    for error_type, count in sorted(failure_counts.items()):
        print(f"  {error_type}: {count}")

    print("failure_details:")
    for failure in failures[:10]:
        print(
            f"  {failure.request_id}: "
            f"{failure.error_type}: {failure.message}"
        )
    if len(failures) > 10:
        print(f"  ... {len(failures) - 10} more failures")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)

    output_path = None
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        if output_path.exists():
            parser.error(f"output path already exists: {output_path}")

    config = WorkloadConfig(
        request_count=args.requests,
        input_token_choices=tuple(args.input_tokens),
        output_token_choices=tuple(args.output_tokens),
        shared_prefix_ratio=args.shared_prefix_ratio,
        concurrency=args.concurrency,
        seed=args.seed,
    )
    client = VllmClient(
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
    )

    metadata = None
    if output_path is not None:
        metadata = collect_run_metadata(
            [
                sys.executable,
                "-m",
                "agent_infer_lab.benchmark",
                *arguments,
            ]
        )

    result = execute_benchmark(config, client)

    print(f"total_requests: {result.total_requests}")
    print(f"successful_requests: {result.successful_requests}")
    print(f"failed_requests: {result.failed_requests}")
    print(f"success_rate: {result.success_rate:.6f}")
    print(
        "wall_duration_seconds: "
        f"{_format_optional(result.wall_duration_seconds)}"
    )
    print(
        "wall_output_throughput_tokens_per_second: "
        f"{_format_optional(_wall_output_throughput(result))}"
    )
    _print_metrics(result.metrics)
    _print_failures(result.failures)

    if output_path is not None and metadata is not None:
        payload = build_result_payload(
            config,
            client,
            result,
            metadata,
        )
        saved_path = write_json_result(output_path, payload)
        print(f"raw_result: {saved_path}")


if __name__ == "__main__":
    main()
