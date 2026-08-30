"""CPU tests for benchmark CLI JSON output."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent_infer_lab import benchmark
from agent_infer_lab.benchmark import BenchmarkResult, RequestFailure
from agent_infer_lab.metrics import RequestTrace, summarize_metrics
from agent_infer_lab.prompting import PreparedRequest
from agent_infer_lab.vllm_client import VllmClient
from agent_infer_lab.workloads import WorkloadConfig


def make_arguments(output_path: Path | None = None) -> list[str]:
    arguments = [
        "--model",
        "test-model",
        "--requests",
        "2",
        "--concurrency",
        "1",
        "--input-tokens",
        "3",
        "--output-tokens",
        "2",
        "--shared-prefix-ratio",
        "0.0",
        "--seed",
        "7",
    ]

    if output_path is not None:
        arguments.extend(["--output", str(output_path)])

    return arguments


def make_result(*, all_failed: bool = False) -> BenchmarkResult:
    requests = (
        PreparedRequest("req-1", (11, 12, 13), 2),
        PreparedRequest("req-2", (21, 22, 23), 2),
    )
    successful_trace = RequestTrace(
        request_id="req-1",
        started_at=101.0,
        first_token_at=101.25,
        completed_at=102.0,
        output_tokens=2,
    )
    second_failure = RequestFailure(
        request_id="req-2",
        error_type="TimeoutError",
        message="request timed out",
        started_at=103.0,
        completed_at=108.0,
    )

    traces = (successful_trace,)
    failures = (second_failure,)

    if all_failed:
        traces = ()
        failures = (
            RequestFailure(
                request_id="req-1",
                error_type="ConnectionRefusedError",
                message="connection refused",
                started_at=101.0,
                completed_at=102.0,
            ),
            second_failure,
        )

    return BenchmarkResult(
        total_requests=2,
        successful_requests=len(traces),
        failed_requests=len(failures),
        success_rate=len(traces) / 2,
        metrics=summarize_metrics(traces) if traces else None,
        failures=failures,
        traces=traces,
        prepared_requests=requests,
        wall_duration_seconds=10.0,
    )


def test_main_saves_configuration_metadata_and_raw_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "nested" / "run.json"
    arguments = make_arguments(target)
    result = make_result()
    stages: list[str] = []
    recorded_commands: list[list[str]] = []
    metadata = {
        "git_commit": "test-commit",
        "git_worktree_clean": True,
        "client_environment": {"test_environment": True},
    }

    def collect_metadata(command: list[str]) -> dict[str, object]:
        stages.append("metadata")
        recorded_commands.append(list(command))
        return metadata

    def execute(
        config: WorkloadConfig,
        client: VllmClient,
    ) -> BenchmarkResult:
        stages.append("execute")
        assert config.request_count == 2
        assert config.concurrency == 1
        assert config.input_token_choices == (3,)
        assert config.output_token_choices == (2,)
        assert config.seed == 7
        assert client.model == "test-model"
        return result

    monkeypatch.setattr(
        benchmark,
        "collect_run_metadata",
        collect_metadata,
    )
    monkeypatch.setattr(benchmark, "execute_benchmark", execute)

    benchmark.main(arguments)

    assert stages == ["metadata", "execute"]
    assert recorded_commands == [
        [
            sys.executable,
            "-m",
            "agent_infer_lab.benchmark",
            *arguments,
        ]
    ]

    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["experiment_type"] == "vllm_fixed_concurrency"
    assert payload["metadata"] == metadata
    assert payload["configuration"]["workload"] == {
        "request_count": 2,
        "input_token_choices": [3],
        "output_token_choices": [2],
        "shared_prefix_ratio": 0.0,
        "concurrency": 1,
        "seed": 7,
    }
    assert payload["configuration"]["client"] == {
        "base_url": "http://127.0.0.1:8000",
        "model": "test-model",
        "timeout_seconds": 60.0,
    }

    saved_result = payload["result"]

    assert saved_result["total_requests"] == 2
    assert saved_result["successful_requests"] == 1
    assert saved_result["failed_requests"] == 1
    assert saved_result["success_rate"] == pytest.approx(0.5)
    assert saved_result["prepared_requests"] == [
        {
            "request_id": "req-1",
            "prompt_token_ids": [11, 12, 13],
            "output_tokens": 2,
        },
        {
            "request_id": "req-2",
            "prompt_token_ids": [21, 22, 23],
            "output_tokens": 2,
        },
    ]
    assert saved_result["traces"] == [
        {
            "request_id": "req-1",
            "started_at": 101.0,
            "first_token_at": 101.25,
            "completed_at": 102.0,
            "output_tokens": 2,
        }
    ]
    assert saved_result["failures"] == [
        {
            "request_id": "req-2",
            "error_type": "TimeoutError",
            "message": "request timed out",
            "started_at": 103.0,
            "completed_at": 108.0,
        }
    ]

    assert saved_result["wall_duration_seconds"] == pytest.approx(10.0)
    assert saved_result[
        "wall_output_throughput_tokens_per_second"
    ] == pytest.approx(0.2)
    assert saved_result["metrics"]["duration_seconds"] == pytest.approx(1.0)
    assert saved_result["metrics"][
        "output_throughput_tokens_per_second"
    ] == pytest.approx(2.0)

    assert "metric_definitions" in payload
    output = capsys.readouterr().out
    assert f"raw_result: {target.resolve()}" in output
    assert "wall_output_throughput_tokens_per_second: 0.200000" in output


def test_main_rejects_existing_output_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "existing.json"
    original_bytes = b'{"run": "original"}\n'
    target.write_bytes(original_bytes)

    def forbidden(
        *args: object,
        **kwargs: object,
    ) -> None:
        pytest.fail("Existing output must be rejected before execution.")

    monkeypatch.setattr(benchmark, "collect_run_metadata", forbidden)
    monkeypatch.setattr(benchmark, "execute_benchmark", forbidden)

    with pytest.raises(SystemExit) as error:
        benchmark.main(make_arguments(target))

    assert error.value.code == 2
    assert target.read_bytes() == original_bytes
    assert "output path already exists" in capsys.readouterr().err


def test_main_saves_all_failed_run_with_zero_wall_throughput(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "all_failed.json"
    result = make_result(all_failed=True)

    monkeypatch.setattr(
        benchmark,
        "collect_run_metadata",
        lambda command: {"test_environment": True},
    )
    monkeypatch.setattr(
        benchmark,
        "execute_benchmark",
        lambda config, client: result,
    )

    benchmark.main(make_arguments(target))

    payload = json.loads(target.read_text(encoding="utf-8"))
    saved_result = payload["result"]

    assert saved_result["successful_requests"] == 0
    assert saved_result["failed_requests"] == 2
    assert saved_result["success_rate"] == 0.0
    assert saved_result["metrics"] is None
    assert saved_result["traces"] == []
    assert len(saved_result["prepared_requests"]) == 2
    assert len(saved_result["failures"]) == 2
    assert saved_result["wall_duration_seconds"] == pytest.approx(10.0)
    assert saved_result["wall_output_throughput_tokens_per_second"] == 0.0


def test_main_without_output_skips_metadata_and_json_writing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = make_result()

    def forbidden(
        *args: object,
        **kwargs: object,
    ) -> None:
        pytest.fail("No output option means no metadata or JSON writing.")

    monkeypatch.setattr(benchmark, "collect_run_metadata", forbidden)
    monkeypatch.setattr(benchmark, "write_json_result", forbidden)
    monkeypatch.setattr(
        benchmark,
        "execute_benchmark",
        lambda config, client: result,
    )

    benchmark.main(make_arguments())

    output = capsys.readouterr().out

    assert "total_requests: 2" in output
    assert "wall_output_throughput_tokens_per_second: 0.200000" in output
    assert "raw_result:" not in output


@pytest.mark.parametrize("duration", [None, 0.0, -1.0])
def test_payload_uses_null_for_unknown_or_invalid_wall_duration(
    duration: float | None,
) -> None:
    result = replace(
        make_result(),
        wall_duration_seconds=duration,
    )
    config = WorkloadConfig(
        request_count=2,
        input_token_choices=(3,),
        output_token_choices=(2,),
        shared_prefix_ratio=0.0,
        concurrency=1,
        seed=7,
    )
    client = VllmClient(
        base_url="http://127.0.0.1:8000",
        model="test-model",
    )

    payload = benchmark.build_result_payload(
        config,
        client,
        result,
        {"test_environment": True},
    )
    decoded = json.loads(json.dumps(payload, allow_nan=False))

    assert decoded["result"][
        "wall_output_throughput_tokens_per_second"
    ] is None


def test_main_does_not_overwrite_file_created_during_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "created_during_run.json"
    original_bytes = b'{"run": "another-process"}\n'
    result = make_result()

    def execute(
        config: WorkloadConfig,
        client: VllmClient,
    ) -> BenchmarkResult:
        target.write_bytes(original_bytes)
        return result

    monkeypatch.setattr(
        benchmark,
        "collect_run_metadata",
        lambda command: {"test_environment": True},
    )
    monkeypatch.setattr(benchmark, "execute_benchmark", execute)

    with pytest.raises(FileExistsError):
        benchmark.main(make_arguments(target))

    assert target.read_bytes() == original_bytes
