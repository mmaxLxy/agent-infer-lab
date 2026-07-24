from dataclasses import FrozenInstanceError

import pytest

from agent_infer_lab.metrics import (
    MetricsSummary,
    RequestMetrics,
    RequestTrace,
    calculate_request_metrics,
    summarize_metrics,
)


def make_trace(
    request_id: str = "req-000001",
    started_at: float = 10.0,
    first_token_at: float = 10.5,
    completed_at: float = 12.5,
    output_tokens: int = 5,
) -> RequestTrace:
    return RequestTrace(
        request_id=request_id,
        started_at=started_at,
        first_token_at=first_token_at,
        completed_at=completed_at,
        output_tokens=output_tokens,
    )


def test_metric_records_are_immutable() -> None:
    trace = make_trace()
    metrics = calculate_request_metrics(trace)
    summary = summarize_metrics((trace,))

    with pytest.raises(FrozenInstanceError):
        trace.output_tokens = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        metrics.ttft_seconds = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        summary.request_count = 2  # type: ignore[misc]


def test_calculate_request_metrics_uses_expected_formulas() -> None:
    metrics = calculate_request_metrics(make_trace())

    assert metrics == RequestMetrics(
        request_id="req-000001",
        ttft_seconds=0.5,
        tpot_seconds=0.5,
        e2e_seconds=2.5,
    )


def test_single_token_request_has_no_tpot() -> None:
    metrics = calculate_request_metrics(
        make_trace(first_token_at=10.4, completed_at=10.4, output_tokens=1)
    )

    assert metrics.ttft_seconds == pytest.approx(0.4)
    assert metrics.tpot_seconds is None
    assert metrics.e2e_seconds == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", ""),
        ("request_id", "   "),
        ("started_at", float("nan")),
        ("first_token_at", float("inf")),
        ("completed_at", True),
        ("first_token_at", 9.5),
        ("completed_at", 10.25),
    ],
)
def test_request_trace_rejects_invalid_fields(field: str, value: object) -> None:
    values: dict[str, object] = {
        "request_id": "req-000001",
        "started_at": 10.0,
        "first_token_at": 10.5,
        "completed_at": 12.5,
        "output_tokens": 5,
    }
    values[field] = value

    with pytest.raises(ValueError):
        RequestTrace(**values)  # type: ignore[arg-type]


def test_request_trace_requires_positive_duration() -> None:
    with pytest.raises(ValueError):
        make_trace(started_at=10.0, first_token_at=10.0, completed_at=10.0)


@pytest.mark.parametrize("output_tokens", [0, -1, True])
def test_request_trace_rejects_invalid_output_tokens(
    output_tokens: object,
) -> None:
    with pytest.raises(ValueError):
        make_trace(output_tokens=output_tokens)  # type: ignore[arg-type]


def test_summarize_metrics_calculates_percentiles_and_throughput() -> None:
    traces = (
        make_trace("req-1", 0.0, 1.0, 5.0, 5),
        make_trace("req-2", 1.0, 3.0, 7.0, 3),
        make_trace("req-3", 2.0, 5.0, 11.0, 4),
        make_trace("req-4", 3.0, 7.0, 15.0, 5),
    )

    summary = summarize_metrics(traces)

    assert summary == MetricsSummary(
        request_count=4,
        total_output_tokens=17,
        duration_seconds=15.0,
        output_throughput_tokens_per_second=pytest.approx(17 / 15),
        ttft_p50_seconds=2.0,
        ttft_p99_seconds=4.0,
        tpot_p50_seconds=2.0,
        tpot_p99_seconds=2.0,
        e2e_p50_seconds=6.0,
        e2e_p99_seconds=12.0,
    )


def test_summarize_metrics_is_deterministic() -> None:
    traces = (make_trace(),)

    assert summarize_metrics(traces) == summarize_metrics(traces)


def test_summarize_metrics_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        summarize_metrics(())


def test_summarize_metrics_rejects_duplicate_request_ids() -> None:
    traces = (
        make_trace("req-duplicate", 0.0, 0.5, 1.0, 2),
        make_trace("req-duplicate", 1.0, 1.5, 2.0, 2),
    )

    with pytest.raises(ValueError):
        summarize_metrics(traces)


def test_all_single_token_requests_have_no_tpot_percentiles() -> None:
    traces = (
        make_trace("req-1", 0.0, 1.0, 1.0, 1),
        make_trace("req-2", 1.0, 3.0, 3.0, 1),
    )

    summary = summarize_metrics(traces)

    assert summary.tpot_p50_seconds is None
    assert summary.tpot_p99_seconds is None

def test_summarize_metrics_uses_nearest_rank_for_p99() -> None:
    traces = tuple(
        make_trace(
            request_id=f"req-{index:03d}",
            started_at=0.0,
            first_token_at=float(index),
            completed_at=float(index + 1),
            output_tokens=2,
        )
        for index in range(1, 102)
    )

    summary = summarize_metrics(traces)

    assert summary.ttft_p99_seconds == 100.0


def test_summarize_metrics_excludes_single_token_requests_from_tpot() -> None:
    traces = (
        make_trace("req-single", 0.0, 1.0, 1.0, 1),
        make_trace("req-multi-1", 0.0, 1.0, 5.0, 3),
        make_trace("req-multi-2", 0.0, 1.0, 9.0, 3),
    )

    summary = summarize_metrics(traces)

    assert summary.tpot_p50_seconds == 2.0
    assert summary.tpot_p99_seconds == 4.0
