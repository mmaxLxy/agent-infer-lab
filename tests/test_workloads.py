import random
from dataclasses import FrozenInstanceError

import pytest

from agent_infer_lab.workloads import RequestSpec, WorkloadConfig, generate_workload


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


def test_request_spec_is_immutable() -> None:
    request = RequestSpec("req-000000", 10, 4, 3)

    with pytest.raises(FrozenInstanceError):
        request.input_tokens = 20  # type: ignore[misc]


def test_generate_workload_is_deterministic() -> None:
    config = valid_config()

    assert generate_workload(config) == generate_workload(config)


def test_generate_workload_builds_expected_fixed_requests() -> None:
    config = WorkloadConfig(
        request_count=3,
        input_token_choices=(10,),
        output_token_choices=(4,),
        shared_prefix_ratio=0.35,
        concurrency=2,
        seed=7,
    )

    assert generate_workload(config) == (
        RequestSpec("req-000000", 10, 4, 3),
        RequestSpec("req-000001", 10, 4, 3),
        RequestSpec("req-000002", 10, 4, 3),
    )


def test_generate_workload_uses_only_allowed_lengths() -> None:
    config = valid_config()

    requests = generate_workload(config)

    assert len(requests) == config.request_count
    assert all(item.input_tokens in config.input_token_choices for item in requests)
    assert all(item.output_tokens in config.output_token_choices for item in requests)
    assert len({item.request_id for item in requests}) == config.request_count


def test_generate_workload_does_not_change_global_random_state() -> None:
    random.seed(99)
    state_before = random.getstate()

    generate_workload(valid_config())

    assert random.getstate() == state_before
