"""Pure CPU calculations for inference latency and throughput metrics."""

import math
from dataclasses import dataclass


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _nearest_rank(values: tuple[float, ...], percentile: float) -> float:
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return ordered[index]


@dataclass(frozen=True)
class RequestTrace:
    """Raw timestamps and output size for one successful request."""

    request_id: str
    started_at: float
    first_token_at: float
    completed_at: float
    output_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        for name, value in (
            ("started_at", self.started_at),
            ("first_token_at", self.first_token_at),
            ("completed_at", self.completed_at),
        ):
            if not _is_finite_number(value):
                raise ValueError(f"{name} must be a finite number")
        if self.first_token_at < self.started_at:
            raise ValueError("first_token_at cannot be earlier than started_at")
        if self.completed_at < self.first_token_at:
            raise ValueError("completed_at cannot be earlier than first_token_at")
        if self.completed_at <= self.started_at:
            raise ValueError("completed_at must be later than started_at")
        if (
            not isinstance(self.output_tokens, int)
            or isinstance(self.output_tokens, bool)
            or self.output_tokens <= 0
        ):
            raise ValueError("output_tokens must be a positive integer")


@dataclass(frozen=True)
class RequestMetrics:
    """Derived latency metrics for one request."""

    request_id: str
    ttft_seconds: float
    tpot_seconds: float | None
    e2e_seconds: float


@dataclass(frozen=True)
class MetricsSummary:
    """Aggregate latency percentiles and output throughput for one batch."""

    request_count: int
    total_output_tokens: int
    duration_seconds: float
    output_throughput_tokens_per_second: float
    ttft_p50_seconds: float
    ttft_p99_seconds: float
    tpot_p50_seconds: float | None
    tpot_p99_seconds: float | None
    e2e_p50_seconds: float
    e2e_p99_seconds: float


def calculate_request_metrics(trace: RequestTrace) -> RequestMetrics:
    """Calculate TTFT, TPOT, and end-to-end latency for one request."""

    ttft_seconds = trace.first_token_at - trace.started_at
    e2e_seconds = trace.completed_at - trace.started_at
    tpot_seconds = None
    if trace.output_tokens >= 2:
        tpot_seconds = (
            trace.completed_at - trace.first_token_at
        ) / (trace.output_tokens - 1)
    return RequestMetrics(
        request_id=trace.request_id,
        ttft_seconds=ttft_seconds,
        tpot_seconds=tpot_seconds,
        e2e_seconds=e2e_seconds,
    )


def summarize_metrics(traces: tuple[RequestTrace, ...]) -> MetricsSummary:
    """Aggregate successful request traces using nearest-rank percentiles."""

    if not traces:
        raise ValueError("traces must not be empty")

    request_ids = tuple(trace.request_id for trace in traces)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_id values must be unique")

    request_metrics = tuple(calculate_request_metrics(trace) for trace in traces)
    ttft_values = tuple(metric.ttft_seconds for metric in request_metrics)
    e2e_values = tuple(metric.e2e_seconds for metric in request_metrics)
    tpot_values = tuple(
        metric.tpot_seconds
        for metric in request_metrics
        if metric.tpot_seconds is not None
    )

    earliest_start = min(trace.started_at for trace in traces)
    latest_completion = max(trace.completed_at for trace in traces)
    duration_seconds = latest_completion - earliest_start
    total_output_tokens = sum(trace.output_tokens for trace in traces)

    return MetricsSummary(
        request_count=len(traces),
        total_output_tokens=total_output_tokens,
        duration_seconds=duration_seconds,
        output_throughput_tokens_per_second=(
            total_output_tokens / duration_seconds
        ),
        ttft_p50_seconds=_nearest_rank(ttft_values, 0.50),
        ttft_p99_seconds=_nearest_rank(ttft_values, 0.99),
        tpot_p50_seconds=(
            _nearest_rank(tpot_values, 0.50) if tpot_values else None
        ),
        tpot_p99_seconds=(
            _nearest_rank(tpot_values, 0.99) if tpot_values else None
        ),
        e2e_p50_seconds=_nearest_rank(e2e_values, 0.50),
        e2e_p99_seconds=_nearest_rank(e2e_values, 0.99),
    )
