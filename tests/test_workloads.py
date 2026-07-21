from dataclasses import FrozenInstanceError

import pytest

from agent_infer_lab.workloads import WorkloadConfig


def valid_config() -> WorkloadConfig:
    return WorkloadConfig(
        request_count=4,
        input_token_choices=(512, 2048),
        output_token_choices=(64, 128),
        shared_prefix_ratio=0.5,
        concurrency=2,
        seed=2026,
    )


def test_workload_config_is_immutable() -> None:
    config = valid_config()

    with pytest.raises(FrozenInstanceError):
        config.request_count = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_count", 0),
        ("request_count", True),
        ("input_token_choices", ()),
        ("input_token_choices", [512]),
        ("input_token_choices", (512, 0)),
        ("output_token_choices", ()),
        ("output_token_choices", (-1,)),
        ("shared_prefix_ratio", -0.1),
        ("shared_prefix_ratio", 1.1),
        ("concurrency", 0),
        ("concurrency", 5),
        ("seed", True),
    ],
)
def test_workload_config_rejects_invalid_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "request_count": 4,
        "input_token_choices": (512, 2048),
        "output_token_choices": (64, 128),
        "shared_prefix_ratio": 0.5,
        "concurrency": 2,
        "seed": 2026,
    }
    values[field] = value

    with pytest.raises(ValueError):
        WorkloadConfig(**values)  # type: ignore[arg-type]
