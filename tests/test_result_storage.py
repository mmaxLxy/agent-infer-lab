"""CPU tests for experiment metadata and JSON result persistence."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_infer_lab import result_storage


def test_write_json_result_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "experiment.json"
    payload = {
        "experiment": "共享前缀实验",
        "request_count": 2,
        "success_rate": 1.0,
        "latencies": [0.1, 0.2],
        "optional_metric": None,
    }

    saved_path = result_storage.write_json_result(target, payload)

    assert saved_path == target.resolve()
    assert target.is_file()

    text = target.read_text(encoding="utf-8")

    assert json.loads(text) == payload
    assert "共享前缀实验" in text
    assert text.endswith("\n")


def test_write_json_result_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "existing.json"

    result_storage.write_json_result(target, {"run": "first"})
    original_bytes = target.read_bytes()

    with pytest.raises(FileExistsError):
        result_storage.write_json_result(target, {"run": "second"})

    assert target.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_write_json_result_rejects_nonfinite_numbers(
    tmp_path: Path,
    invalid_value: float,
) -> None:
    target = tmp_path / "invalid.json"

    with pytest.raises(ValueError):
        result_storage.write_json_result(
            target,
            {"latency": invalid_value},
        )

    assert not target.exists()


def test_write_json_result_rejects_unsupported_objects(
    tmp_path: Path,
) -> None:
    target = tmp_path / "unsupported.json"

    with pytest.raises(TypeError):
        result_storage.write_json_result(
            target,
            {"unsupported": object()},
        )

    assert not target.exists()


@pytest.mark.parametrize(
    ("git_status", "expected_clean"),
    [
        ("", True),
        (" M src/agent_infer_lab/benchmark.py", False),
        (None, None),
    ],
)
def test_collect_run_metadata_records_git_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_status: str | None,
    expected_clean: bool | None,
) -> None:
    def fake_git_output(
        repository_root: Path,
        *arguments: str,
    ) -> str | None:
        assert repository_root == tmp_path.resolve()

        if arguments == ("rev-parse", "HEAD"):
            return "example-commit"

        assert arguments == (
            "status",
            "--short",
            "--untracked-files=all",
        )
        return git_status

    monkeypatch.setattr(
        result_storage,
        "_git_output",
        fake_git_output,
    )
    monkeypatch.setattr(
        result_storage,
        "collect_environment",
        lambda: {"test_environment": True},
    )

    command = ["python", "-m", "agent_infer_lab.benchmark"]

    metadata = result_storage.collect_run_metadata(
        command,
        repository_root=tmp_path,
    )

    assert metadata["git_commit"] == "example-commit"
    assert metadata["git_status_short"] == git_status
    assert metadata["git_worktree_clean"] is expected_clean
    assert metadata["client_environment"] == {
        "test_environment": True,
    }
    assert metadata["repository_root"] == str(tmp_path.resolve())
    assert metadata["command"] == command
    assert metadata["source_sha256"] == {}

    timestamp = datetime.fromisoformat(
        str(metadata["captured_at_utc"])
    )
    assert timestamp.utcoffset() == UTC.utcoffset(timestamp)

    command.append("--model")
    assert metadata["command"] == [
        "python",
        "-m",
        "agent_infer_lab.benchmark",
    ]


def test_source_hashes_cover_code_and_dependency_files(
    tmp_path: Path,
) -> None:
    files = {
        "src/agent_infer_lab/example.py": b"value = 1\n",
        "cuda/example.cu": b"// CUDA source\n",
        "cuda/example.cpp": b"// C++ source\n",
        "cuda/example.h": b"// Header\n",
        "cuda/example.cuh": b"// CUDA header\n",
        "pyproject.toml": b"[project]\n",
        "uv.lock": b"version = 1\n",
    }

    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    (tmp_path / "cuda" / "compiled.so").write_bytes(b"binary")
    (tmp_path / "cuda" / "notes.txt").write_text(
        "not source code",
        encoding="utf-8",
    )

    hashes = result_storage._source_hashes(tmp_path)

    assert hashes == {
        relative_path: hashlib.sha256(content).hexdigest()
        for relative_path, content in files.items()
    }

    source = tmp_path / "src" / "agent_infer_lab" / "example.py"
    source.write_bytes(b"value = 2\n")

    updated_hashes = result_storage._source_hashes(tmp_path)

    assert (
        updated_hashes["src/agent_infer_lab/example.py"]
        != hashes["src/agent_infer_lab/example.py"]
    )


def test_source_hashes_allow_missing_directories(tmp_path: Path) -> None:
    assert result_storage._source_hashes(tmp_path) == {}


def test_git_output_preserves_leading_spaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert command == [
            "git",
            "-C",
            str(tmp_path),
            "status",
            "--short",
        ]
        assert kwargs["timeout"] == 10
        assert kwargs["check"] is False

        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=" M example.py\r\n",
            stderr="",
        )

    monkeypatch.setattr(
        result_storage.subprocess,
        "run",
        fake_run,
    )

    output = result_storage._git_output(
        tmp_path,
        "status",
        "--short",
    )

    assert output == " M example.py"


def test_git_output_returns_none_on_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        result_storage.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["git"],
            returncode=128,
            stdout="",
            stderr="not a git repository",
        ),
    )

    assert result_storage._git_output(
        tmp_path,
        "rev-parse",
        "HEAD",
    ) is None


@pytest.mark.parametrize(
    "error",
    [
        OSError("git executable not found"),
        subprocess.TimeoutExpired(cmd="git", timeout=10),
    ],
)
def test_git_output_returns_none_on_command_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def failing_run(
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        raise error

    monkeypatch.setattr(
        result_storage.subprocess,
        "run",
        failing_run,
    )

    assert result_storage._git_output(
        tmp_path,
        "rev-parse",
        "HEAD",
    ) is None
