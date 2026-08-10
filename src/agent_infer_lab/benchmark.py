"""Fixed-concurrency execution for reproducible vLLM benchmarks."""

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from agent_infer_lab.metrics import MetricsSummary, RequestTrace, summarize_metrics
from agent_infer_lab.prompting import PreparedRequest, prepare_requests
from agent_infer_lab.vllm_client import VllmClient
from agent_infer_lab.workloads import WorkloadConfig, generate_workload

SendRequest = Callable[[PreparedRequest], RequestTrace]


@dataclass(frozen=True)
class RequestFailure:
    """Failure information collected from one request."""

    request_id: str
    error_type: str
    message: str


@dataclass(frozen=True)
class BenchmarkResult:
    """Successful metrics and failures collected from one benchmark run."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    metrics: MetricsSummary | None
    failures: tuple[RequestFailure, ...]


RequestOutcome = RequestTrace | RequestFailure


def _request_failure(
    request: PreparedRequest,
    error: Exception,
) -> RequestFailure:
    """Convert an exception chain into stable failure information."""

    root_error: BaseException = error
    while root_error.__cause__ is not None:
        root_error = root_error.__cause__

    return RequestFailure(
        request_id=request.request_id,
        error_type=type(root_error).__name__,
        message=str(root_error) or repr(root_error),
    )


def run_benchmark(
    requests: tuple[PreparedRequest, ...],
    *,
    concurrency: int,
    send_request: SendRequest,
) -> BenchmarkResult:
    """Run every request and collect both successes and failures."""

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

    def execute_one(request: PreparedRequest) -> RequestOutcome:
        try:
            return send_request(request)
        except Exception as error:  # noqa: BLE001
            return _request_failure(request, error)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        outcomes = tuple(executor.map(execute_one, requests))

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
    args = build_parser().parse_args(argv)
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
    result = execute_benchmark(config, client)

    print(f"total_requests: {result.total_requests}")
    print(f"successful_requests: {result.successful_requests}")
    print(f"failed_requests: {result.failed_requests}")
    print(f"success_rate: {result.success_rate:.6f}")
    _print_metrics(result.metrics)
    _print_failures(result.failures)


if __name__ == "__main__":
    main()
