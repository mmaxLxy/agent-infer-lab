"""CPU-safe development environment inspection."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path

CommandStatus = dict[str, object]
VersionReader = Callable[[str], str | None]
CommandProbe = Callable[[str, tuple[str, ...], tuple[str, ...]], CommandStatus]
EnvironmentCollector = Callable[[], dict[str, object]]


def _read_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _find_command(name: str, extra_paths: Sequence[str]) -> str | None:
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved

    for candidate in extra_paths:
        if Path(candidate).is_file():
            return candidate
    return None


def _probe_command(
    name: str,
    args: tuple[str, ...],
    extra_paths: tuple[str, ...] = (),
) -> CommandStatus:
    executable = _find_command(name, extra_paths)
    if executable is None:
        return {
            "available": False,
            "path": None,
            "output": None,
            "error": f"{name} not found",
        }

    try:
        completed = subprocess.run(
            [executable, *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "path": executable,
            "output": None,
            "error": str(exc),
        }

    output = (completed.stdout or completed.stderr).strip() or None
    error = None if completed.returncode == 0 else f"exit code {completed.returncode}"
    return {
        "available": completed.returncode == 0,
        "path": executable,
        "output": output,
        "error": error,
    }


def collect_environment(
    *,
    version_reader: VersionReader = _read_package_version,
    command_probe: CommandProbe = _probe_command,
) -> dict[str, object]:
    """Collect environment details without importing optional GPU packages."""
    tools = {
        "nvcc": command_probe("nvcc", ("--version",), ("/usr/local/bin/nvcc",)),
        "nsys": command_probe("nsys", ("--version",), ("/usr/local/bin/nsys",)),
        "ncu": command_probe("ncu", ("--version",), ("/usr/local/bin/ncu",)),
    }
    gpu = command_probe(
        "nvidia-smi",
        (
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ),
        ("/usr/lib/wsl/lib/nvidia-smi",),
    )
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
        },
        "packages": {
            "torch": version_reader("torch"),
            "vllm": version_reader("vllm"),
        },
        "tools": tools,
        "gpu": gpu,
    }


def _gpu_available(report: dict[str, object]) -> bool:
    gpu = report.get("gpu")
    return isinstance(gpu, dict) and gpu.get("available") is True


def _print_human_report(report: dict[str, object]) -> None:
    python_info = report["python"]
    platform_info = report["platform"]
    packages = report["packages"]
    tools = report["tools"]
    gpu = report["gpu"]

    print(f"Python: {python_info['version']} ({python_info['executable']})")
    print(f"Platform: {platform_info['platform']}")
    print(f"GPU: {'available' if gpu['available'] else 'unavailable'}")
    for name, version in packages.items():
        print(f"Package {name}: {version or 'not installed'}")
    for name, status in tools.items():
        state = "available" if status["available"] else "unavailable"
        print(f"Tool {name}: {state}")


def main(
    argv: Sequence[str] | None = None,
    *,
    collect: EnvironmentCollector = collect_environment,
) -> int:
    parser = argparse.ArgumentParser(description="Inspect the AgentInferLab environment.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Return a non-zero status when no working NVIDIA GPU is detected.",
    )
    args = parser.parse_args(argv)

    report = collect()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human_report(report)

    return 1 if args.require_gpu and not _gpu_available(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
