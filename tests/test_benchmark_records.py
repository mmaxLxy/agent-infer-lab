"""CPU tests for raw benchmark records and execution timing."""

import pytest

from agent_infer_lab.benchmark import (
    BenchmarkResult,
    RequestFailure,
    run_benchmark,
)
from agent_infer_lab.metrics import RequestTrace
from agent_infer_lab.prompting import PreparedRequest


def make_requests() -> tuple[PreparedRequest, ...]:
    return (
        PreparedRequest("req-1", (11, 12, 13), 2),
        PreparedRequest("req-2", (21, 22, 23), 2),
    )


def make_trace(
    request: PreparedRequest,
    started_at: float,
) -> RequestTrace:
    return RequestTrace(
        request_id=request.request_id,
        started_at=started_at,
        first_token_at=started_at + 0.25,
        completed_at=started_at + 1.0,
        output_tokens=2,
    )


def test_run_benchmark_preserves_requests_and_successful_traces() -> None:
    requests = make_requests()
    expected_traces = (
        make_trace(requests[0], 11.0),
        make_trace(requests[1], 13.0),
    )
    trace_by_id = {
        trace.request_id: trace
        for trace in expected_traces
    }
    clock_values = iter([10.0, 11.0, 13.0, 20.0])

    def send(request: PreparedRequest) -> RequestTrace:
        return trace_by_id[request.request_id]

    result = run_benchmark(
        requests,
        concurrency=1,
        send_request=send,
        clock=clock_values.__next__,
    )

    assert result.prepared_requests == requests
    assert result.prepared_requests[0].prompt_token_ids == (11, 12, 13)
    assert result.prepared_requests[1].prompt_token_ids == (21, 22, 23)
    assert result.traces == expected_traces
    assert result.successful_requests == 2
    assert result.failed_requests == 0
    assert result.failures == ()
    assert result.wall_duration_seconds == pytest.approx(10.0)

    assert result.metrics is not None
    assert result.metrics.duration_seconds == pytest.approx(3.0)
    assert next(clock_values, None) is None


def test_run_benchmark_records_failure_times_and_full_duration() -> None:
    requests = make_requests()
    successful_trace = make_trace(requests[0], 101.0)
    clock_values = iter([100.0, 101.0, 103.0, 108.0, 110.0])

    def send(request: PreparedRequest) -> RequestTrace:
        if request.request_id == "req-2":
            try:
                raise TimeoutError("request timed out")
            except TimeoutError as error:
                raise RuntimeError("client request failed") from error

        return successful_trace

    result = run_benchmark(
        requests,
        concurrency=1,
        send_request=send,
        clock=clock_values.__next__,
    )

    assert result.total_requests == 2
    assert result.successful_requests == 1
    assert result.failed_requests == 1
    assert result.success_rate == pytest.approx(0.5)
    assert result.prepared_requests == requests
    assert result.traces == (successful_trace,)

    assert len(result.failures) == 1
    failure = result.failures[0]

    assert failure.request_id == "req-2"
    assert failure.error_type == "TimeoutError"
    assert failure.message == "request timed out"
    assert failure.started_at == pytest.approx(103.0)
    assert failure.completed_at == pytest.approx(108.0)

    assert result.wall_duration_seconds == pytest.approx(10.0)
    assert result.metrics is not None
    assert result.metrics.duration_seconds == pytest.approx(1.0)
    assert result.metrics.total_output_tokens == 2
    assert next(clock_values, None) is None


def test_run_benchmark_preserves_records_when_every_request_fails() -> None:
    requests = make_requests()
    clock_values = iter([10.0, 11.0, 14.0, 15.0, 20.0, 21.0])

    def send(request: PreparedRequest) -> RequestTrace:
        raise ConnectionRefusedError(
            f"connection refused for {request.request_id}"
        )

    result = run_benchmark(
        requests,
        concurrency=1,
        send_request=send,
        clock=clock_values.__next__,
    )

    assert result.total_requests == 2
    assert result.successful_requests == 0
    assert result.failed_requests == 2
    assert result.success_rate == 0.0
    assert result.metrics is None
    assert result.traces == ()
    assert result.prepared_requests == requests
    assert result.wall_duration_seconds == pytest.approx(11.0)

    assert [
        failure.request_id
        for failure in result.failures
    ] == ["req-1", "req-2"]

    assert [
        (failure.started_at, failure.completed_at)
        for failure in result.failures
    ] == [(11.0, 14.0), (15.0, 20.0)]

    assert next(clock_values, None) is None


def test_run_benchmark_rejects_duplicate_ids_before_sending() -> None:
    requests = (
        PreparedRequest("req-duplicate", (11, 12), 2),
        PreparedRequest("req-duplicate", (21, 22), 2),
    )
    sent_ids: list[str] = []

    def send(request: PreparedRequest) -> RequestTrace:
        sent_ids.append(request.request_id)
        return make_trace(request, 10.0)

    with pytest.raises(ValueError, match="request IDs must be unique"):
        run_benchmark(
            requests,
            concurrency=1,
            send_request=send,
        )

    assert sent_ids == []


@pytest.mark.parametrize("invalid_result", [None, "not a trace"])
def test_run_benchmark_records_invalid_return_type_as_failure(
    invalid_result: object,
) -> None:
    request = make_requests()[0]

    def send(_: PreparedRequest) -> object:
        return invalid_result

    result = run_benchmark(
        (request,),
        concurrency=1,
        send_request=send,  # type: ignore[arg-type]
    )

    assert result.total_requests == 1
    assert result.successful_requests == 0
    assert result.failed_requests == 1
    assert result.metrics is None
    assert result.traces == ()
    assert result.prepared_requests == (request,)

    failure = result.failures[0]

    assert failure.request_id == request.request_id
    assert failure.error_type == "TypeError"
    assert failure.message == "send_request must return a RequestTrace"
    assert failure.started_at is not None
    assert failure.completed_at is not None
    assert failure.completed_at >= failure.started_at


def test_run_benchmark_records_mismatched_trace_id_as_failure() -> None:
    request = make_requests()[0]
    wrong_trace = RequestTrace(
        request_id="unexpected-request",
        started_at=10.0,
        first_token_at=10.25,
        completed_at=11.0,
        output_tokens=2,
    )

    def send(_: PreparedRequest) -> RequestTrace:
        return wrong_trace

    result = run_benchmark(
        (request,),
        concurrency=1,
        send_request=send,
    )

    assert result.successful_requests == 0
    assert result.failed_requests == 1
    assert result.metrics is None
    assert result.traces == ()

    failure = result.failures[0]

    assert failure.request_id == request.request_id
    assert failure.error_type == "ValueError"
    assert failure.message == (
        "returned trace request_id must match the request"
    )


def test_new_record_fields_allow_existing_construction() -> None:
    failure = RequestFailure(
        request_id="req-1",
        error_type="TimeoutError",
        message="request timed out",
    )
    result = BenchmarkResult(
        total_requests=1,
        successful_requests=0,
        failed_requests=1,
        success_rate=0.0,
        metrics=None,
        failures=(failure,),
    )

    assert failure.started_at is None
    assert failure.completed_at is None
    assert result.traces == ()
    assert result.prepared_requests == ()
    assert result.wall_duration_seconds is None
