"""Environment metadata and JSON persistence for benchmark results."""

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from agent_infer_lab.environment import collect_environment


def _git_output(
    repository_root: Path,
    *arguments: str,
) -> str | None:
    """Read Git information without changing the repository."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None

    return completed.stdout.rstrip("\r\n")


def _source_hashes(
    repository_root: Path,
) -> dict[str, str]:
    """Calculate SHA-256 fingerprints for code and dependency files."""

    source_files: set[Path] = set()

    for directory_name in ("src", "cuda"):
        directory = repository_root / directory_name

        if not directory.is_dir():
            continue

        for pattern in ("*.py", "*.cpp", "*.cu", "*.h", "*.cuh"):
            source_files.update(directory.rglob(pattern))

    for filename in ("pyproject.toml", "uv.lock"):
        path = repository_root / filename

        if path.is_file():
            source_files.add(path)

    return {
        path.relative_to(repository_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(source_files)
    }


def collect_run_metadata(
    command: Sequence[str],
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Capture client-side environment and code state before a run."""

    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[2]

    repository_root = repository_root.resolve()

    git_commit = _git_output(
        repository_root,
        "rev-parse",
        "HEAD",
    )
    git_status = _git_output(
        repository_root,
        "status",
        "--short",
        "--untracked-files=all",
    )

    git_worktree_clean = None

    if git_status is not None:
        git_worktree_clean = git_status == ""

    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python_executable": sys.executable,
        "command": list(command),
        "repository_root": str(repository_root),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "git_worktree_clean": git_worktree_clean,
        "source_sha256": _source_hashes(repository_root),
        "client_environment": collect_environment(),
    }


def write_json_result(
    output_path: str | Path,
    payload: Mapping[str, object],
) -> Path:
    """Create a JSON result file without overwriting an existing result."""

    serialized = json.dumps(
        dict(payload),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        output_file.write(serialized)
        output_file.write("\n")

    return target
