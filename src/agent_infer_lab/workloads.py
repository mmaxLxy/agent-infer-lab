"""Deterministic request specifications for inference benchmarks."""

import random
from dataclasses import dataclass


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_token_choices(name: str, values: tuple[int, ...]) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    if any(not _is_positive_int(value) for value in values):
        raise ValueError(f"{name} must contain only positive integers")


@dataclass(frozen=True)
class WorkloadConfig:
    """Validated settings used to generate one reproducible workload."""

    request_count: int
    input_token_choices: tuple[int, ...]
    output_token_choices: tuple[int, ...]
    shared_prefix_ratio: float
    concurrency: int
    seed: int

    def __post_init__(self) -> None:
        if not _is_positive_int(self.request_count):
            raise ValueError("request_count must be a positive integer")
        _validate_token_choices("input_token_choices", self.input_token_choices)
        _validate_token_choices("output_token_choices", self.output_token_choices)
        if isinstance(self.shared_prefix_ratio, bool) or not isinstance(
            self.shared_prefix_ratio, (int, float)
        ):
            raise ValueError("shared_prefix_ratio must be a number")
        if not 0.0 <= self.shared_prefix_ratio <= 1.0:
            raise ValueError("shared_prefix_ratio must be between 0.0 and 1.0")
        if not _is_positive_int(self.concurrency):
            raise ValueError("concurrency must be a positive integer")
        if self.concurrency > self.request_count:
            raise ValueError("concurrency cannot exceed request_count")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class RequestSpec:
    """One immutable request description for a later benchmark backend."""

    request_id: str
    input_tokens: int
    output_tokens: int
    shared_prefix_tokens: int


def generate_workload(config: WorkloadConfig) -> tuple[RequestSpec, ...]:
    """Generate deterministic requests without changing global random state."""

    generator = random.Random(config.seed)
    requests = []
    for index in range(config.request_count):
        input_tokens = generator.choice(config.input_token_choices)
        output_tokens = generator.choice(config.output_token_choices)
        requests.append(
            RequestSpec(
                request_id=f"req-{index:06d}",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                shared_prefix_tokens=int(input_tokens * config.shared_prefix_ratio),
            )
        )
    return tuple(requests)
