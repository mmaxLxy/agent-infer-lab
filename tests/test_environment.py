import json


def _missing_probe(
    name: str,
    args: tuple[str, ...],
    extra_paths: tuple[str, ...],
) -> dict[str, object]:
    del args, extra_paths
    return {
        "available": False,
        "path": None,
        "output": None,
        "error": f"{name} not found",
    }


def _cpu_only_report() -> dict[str, object]:
    return {
        "python": {"version": "3.12.3", "executable": "/usr/bin/python3"},
        "platform": {"system": "Linux", "release": "test", "platform": "test"},
        "packages": {"torch": None, "vllm": None},
        "tools": {},
        "gpu": {"available": False, "path": None, "output": None, "error": "not found"},
    }


def test_collect_environment_handles_missing_optional_dependencies() -> None:
    from agent_infer_lab.environment import collect_environment

    report = collect_environment(
        version_reader=lambda name: None,
        command_probe=_missing_probe,
    )

    assert report["packages"] == {"torch": None, "vllm": None}
    assert report["gpu"]["available"] is False
    assert all(tool["available"] is False for tool in report["tools"].values())


def test_default_mode_succeeds_without_gpu(capsys) -> None:
    from agent_infer_lab.environment import main

    exit_code = main(["--json"], collect=_cpu_only_report)

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["gpu"]["available"] is False


def test_strict_mode_fails_without_gpu(capsys) -> None:
    from agent_infer_lab.environment import main

    exit_code = main(["--json", "--require-gpu"], collect=_cpu_only_report)

    capsys.readouterr()
    assert exit_code == 1


def test_strict_mode_succeeds_with_gpu(capsys) -> None:
    from agent_infer_lab.environment import main

    report = _cpu_only_report()
    report["gpu"] = {
        "available": True,
        "path": "/usr/bin/nvidia-smi",
        "output": "NVIDIA GPU",
        "error": None,
    }

    exit_code = main(["--json", "--require-gpu"], collect=lambda: report)

    capsys.readouterr()
    assert exit_code == 0
