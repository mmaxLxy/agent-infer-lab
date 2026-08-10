import threading
import time

import pytest

from agent_infer_lab import benchmark
from agent_infer_lab.benchmark import (
    BenchmarkResult,
    RequestFailure,
    execute_benchmark,
    run_benchmark,
)
from agent_infer_lab.metrics import MetricsSummary, RequestTrace
from agent_infer_lab.prompting import PreparedRequest
from agent_infer_lab.workloads import WorkloadConfig


def make_requests(count: int = 4) -> tuple[PreparedRequest, ...]:
    return tuple(
        PreparedRequest(f"req-{index}", (index + 1,), 2)
        for index in range(count)
    )


def make_trace(request: PreparedRequest) -> RequestTrace:
    index = int(request.request_id.removeprefix("req-"))
    started = float(index)
    return RequestTrace(
        request_id=request.request_id,
        started_at=started,
        first_token_at=started + 0.1,
        completed_at=started + 1.0,
        output_tokens=2,
    )


def test_run_benchmark_executes_each_request_and_summarizes() -> None:
    seen: list[str] = []

    def send(request: PreparedRequest) -> RequestTrace:
        seen.append(request.request_id)
        return make_trace(request)

    result = run_benchmark(
        make_requests(),
        concurrency=2,
        send_request=send,
    )

    assert sorted(seen) == ["req-0", "req-1", "req-2", "req-3"]
    assert result.total_requests == 4
    assert result.successful_requests == 4
    assert result.failed_requests == 0
    assert result.success_rate == 1.0
    assert result.failures == ()
    assert result.metrics is not None
    assert result.metrics.request_count == 4
    assert result.metrics.total_output_tokens == 8


def test_run_benchmark_never_exceeds_concurrency() -> None:
    lock = threading.Lock()
    running = 0
    peak = 0

    def send(request: PreparedRequest) -> RequestTrace:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)

        time.sleep(0.01)

        with lock:
            running -= 1
        return make_trace(request)

    run_benchmark(
        make_requests(6),
        concurrency=2,
        send_request=send,
    )

    assert peak == 2


def test_run_benchmark_collects_partial_failure() -> None:
    seen: list[str] = []

    def send(request: PreparedRequest) -> RequestTrace:
        seen.append(request.request_id)
        if request.request_id == "req-1":
            try:
                raise ConnectionRefusedError(111, "Connection refused")
            except ConnectionRefusedError as error:
                raise RuntimeError("service failed") from error
        return make_trace(request)

    result = run_benchmark(
        make_requests(3),
        concurrency=2,
        send_request=send,
    )

    assert sorted(seen) == ["req-0", "req-1", "req-2"]
    assert result.total_requests == 3
    assert result.successful_requests == 2
    assert result.failed_requests == 1
    assert result.success_rate == pytest.approx(2 / 3)
    assert result.metrics is not None
    assert result.metrics.request_count == 2
    assert len(result.failures) == 1
    assert result.failures[0].request_id == "req-1"
    assert result.failures[0].error_type == "ConnectionRefusedError"
    assert "Connection refused" in result.failures[0].message


def test_run_benchmark_handles_all_requests_failing() -> None:
    def send(_: PreparedRequest) -> RequestTrace:
        raise TimeoutError("request timed out")

    result = run_benchmark(
        make_requests(3),
        concurrency=2,
        send_request=send,
    )

    assert result.total_requests == 3
    assert result.successful_requests == 0
    assert result.failed_requests == 3
    assert result.success_rate == 0.0
    assert result.metrics is None
    assert len(result.failures) == 3
    assert all(
        failure.error_type == "TimeoutError"
        for failure in result.failures
    )


@pytest.mark.parametrize("concurrency", [0, -1, True])
def test_run_benchmark_rejects_invalid_concurrency(
    concurrency: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        run_benchmark(
            make_requests(),
            concurrency=concurrency,  # type: ignore[arg-type]
            send_request=make_trace,
        )


def test_run_benchmark_rejects_excessive_concurrency() -> None:
    with pytest.raises(ValueError, match="cannot exceed request count"):
        run_benchmark(
            make_requests(2),
            concurrency=3,
            send_request=make_trace,
        )


def test_run_benchmark_rejects_empty_requests() -> None:
    with pytest.raises(ValueError, match="requests must not be empty"):
        run_benchmark(
            (),
            concurrency=1,
            send_request=make_trace,
        )


def test_execute_benchmark_connects_existing_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WorkloadConfig(
        request_count=2,
        input_token_choices=(4,),
        output_token_choices=(2,),
        shared_prefix_ratio=0.5,
        concurrency=1,
        seed=7,
    )
    requests = (
        PreparedRequest("req-000000", (1, 2, 3, 4), 2),
        PreparedRequest("req-000001", (1, 2, 5, 6), 2),
    )
    expected_metrics = MetricsSummary(
        request_count=2,
        total_output_tokens=4,
        duration_seconds=2.0,
        output_throughput_tokens_per_second=2.0,
        ttft_p50_seconds=0.1,
        ttft_p99_seconds=0.2,
        tpot_p50_seconds=0.9,
        tpot_p99_seconds=1.8,
        e2e_p50_seconds=1.0,
        e2e_p99_seconds=2.0,
    )
    expected_result = BenchmarkResult(
        total_requests=2,
        successful_requests=2,
        failed_requests=0,
        success_rate=1.0,
        metrics=expected_metrics,
        failures=(),
    )

    class FakeClient:
        def tokenize(self, _: str) -> tuple[int, ...]:
            return (1, 2, 3)

        def stream_completion(
            self,
            request: PreparedRequest,
        ) -> RequestTrace:
            return make_trace(request)

    monkeypatch.setattr(
        benchmark,
        "prepare_requests",
        lambda specs, tokenize: requests,
    )
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda prepared, *, concurrency, send_request: expected_result,
    )

    assert execute_benchmark(
        config,
        FakeClient(),  # type: ignore[arg-type]
    ) == expected_result


def test_main_prints_successful_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metrics = MetricsSummary(
        request_count=2,
        total_output_tokens=8,
        duration_seconds=2.0,
        output_throughput_tokens_per_second=4.0,
        ttft_p50_seconds=0.1,
        ttft_p99_seconds=0.2,
        tpot_p50_seconds=0.03,
        tpot_p99_seconds=0.04,
        e2e_p50_seconds=1.0,
        e2e_p99_seconds=1.2,
    )
    result = BenchmarkResult(
        total_requests=2,
        successful_requests=2,
        failed_requests=0,
        success_rate=1.0,
        metrics=metrics,
        failures=(),
    )
    monkeypatch.setattr(
        benchmark,
        "execute_benchmark",
        lambda config, client: result,
    )

    benchmark.main(
        [
            "--model",
            "test-model",
            "--requests",
            "2",
            "--concurrency",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert "total_requests: 2" in output
    assert "successful_requests: 2" in output
    assert "failed_requests: 0" in output
    assert "success_rate: 1.000000" in output
    assert "metrics_scope: successful_requests_only" in output
    assert "output_throughput_tokens_per_second: 4.000000" in output
    assert "ttft_p99_seconds: 0.200000" in output
    assert "failure_types: none" in output


def test_main_prints_failures_without_successful_metrics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failures = (
        RequestFailure(
            request_id="req-0",
            error_type="ConnectionRefusedError",
            message="[Errno 111] Connection refused",
        ),
        RequestFailure(
            request_id="req-1",
            error_type="ConnectionRefusedError",
            message="[Errno 111] Connection refused",
        ),
        RequestFailure(
            request_id="req-2",
            error_type="TimeoutError",
            message="request timed out",
        ),
    )
    result = BenchmarkResult(
        total_requests=3,
        successful_requests=0,
        failed_requests=3,
        success_rate=0.0,
        metrics=None,
        failures=failures,
    )
    monkeypatch.setattr(
        benchmark,
        "execute_benchmark",
        lambda config, client: result,
    )

    benchmark.main(
        [
            "--model",
            "test-model",
            "--requests",
            "3",
            "--concurrency",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert "total_requests: 3" in output
    assert "successful_requests: 0" in output
    assert "failed_requests: 3" in output
    assert "success_rate: 0.000000" in output
    assert "successful_metrics: N/A" in output
    assert "ConnectionRefusedError: 2" in output
    assert "TimeoutError: 1" in output
    assert "req-0: ConnectionRefusedError" in output
