import threading
import time

import pytest

from agent_infer_lab import benchmark
from agent_infer_lab.benchmark import execute_benchmark, run_benchmark
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

    summary = run_benchmark(
        make_requests(),
        concurrency=2,
        send_request=send,
    )

    assert sorted(seen) == ["req-0", "req-1", "req-2", "req-3"]
    assert summary.request_count == 4
    assert summary.total_output_tokens == 8


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


def test_run_benchmark_propagates_request_failure() -> None:
    def send(_: PreparedRequest) -> RequestTrace:
        raise RuntimeError("service failed")

    with pytest.raises(RuntimeError, match="service failed"):
        run_benchmark(
            make_requests(2),
            concurrency=1,
            send_request=send,
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
    expected = MetricsSummary(
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
        lambda prepared, *, concurrency, send_request: expected,
    )

    assert execute_benchmark(
        config,
        FakeClient(),  # type: ignore[arg-type]
    ) == expected


def test_main_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = MetricsSummary(
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
    monkeypatch.setattr(
        benchmark,
        "execute_benchmark",
        lambda config, client: summary,
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
    assert "requests: 2" in output
    assert "output_throughput_tokens_per_second: 4.000000" in output
    assert "ttft_p99_seconds: 0.200000" in output
