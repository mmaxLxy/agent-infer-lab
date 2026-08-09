"""Fixed-concurrency execution for reproducible vLLM benchmarks."""

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from agent_infer_lab.metrics import MetricsSummary, RequestTrace, summarize_metrics
from agent_infer_lab.prompting import PreparedRequest, prepare_requests
from agent_infer_lab.vllm_client import VllmClient
from agent_infer_lab.workloads import WorkloadConfig, generate_workload

SendRequest = Callable[[PreparedRequest], RequestTrace]


def run_benchmark(
    requests: tuple[PreparedRequest, ...],
    *,
    concurrency: int,
    send_request: SendRequest,
) -> MetricsSummary:
    """Run all requests with a fixed maximum concurrency."""

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

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        traces = tuple(executor.map(send_request, requests))
    return summarize_metrics(traces)


def execute_benchmark(
    config: WorkloadConfig,
    client: VllmClient,
) -> MetricsSummary:
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
    summary = execute_benchmark(config, client)

    print(f"requests: {summary.request_count}")
    print(f"output_tokens: {summary.total_output_tokens}")
    print(f"duration_seconds: {summary.duration_seconds:.6f}")
    print(
        "output_throughput_tokens_per_second: "
        f"{summary.output_throughput_tokens_per_second:.6f}"
    )
    print(f"ttft_p50_seconds: {summary.ttft_p50_seconds:.6f}")
    print(f"ttft_p99_seconds: {summary.ttft_p99_seconds:.6f}")
    print(f"tpot_p50_seconds: {_format_optional(summary.tpot_p50_seconds)}")
    print(f"tpot_p99_seconds: {_format_optional(summary.tpot_p99_seconds)}")
    print(f"e2e_p50_seconds: {summary.e2e_p50_seconds:.6f}")
    print(f"e2e_p99_seconds: {summary.e2e_p99_seconds:.6f}")


if __name__ == "__main__":
    main()
