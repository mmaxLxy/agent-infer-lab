# Deterministic Workload Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-CPU generator that returns the same immutable request specifications for the same workload configuration and seed.

**Architecture:** `workloads.py` owns immutable configuration and request data classes plus one pure generation function. It uses a local Python random-number generator, performs validation before generation, and returns a tuple without touching files, tokenizers, GPUs, or global random state.

**Tech Stack:** Python 3.12 standard library, pytest 9.1.1, Ruff 0.15.22.

## Global Constraints

- Add no third-party runtime dependency.
- Implement only request specifications; do not generate text, token IDs, arrival times, or JSON.
- Keep CPU CI independent of PyTorch, CUDA, vLLM, and Hugging Face.
- Use test-first development and run each test once before its implementation exists.

---

### Task 1: Immutable configuration and validation

**Files:**
- Create: `src/agent_infer_lab/workloads.py`
- Create: `tests/test_workloads.py`

**Interfaces:**
- Consumes: Six configuration values supplied directly by Python callers.
- Produces: `WorkloadConfig`, an immutable validated data class.

- [ ] **Step 1: Write the first failing configuration tests**

```python
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
```

- [ ] **Step 2: Run the tests and verify the RED state**

Run in the CPU development environment:

```bash
pytest tests/test_workloads.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'agent_infer_lab.workloads'`. This proves the new tests detect that the feature is absent.

- [ ] **Step 3: Add the minimal immutable configuration implementation**

Create `src/agent_infer_lab/workloads.py`:

```python
"""Deterministic request specifications for inference benchmarks."""

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
```

- [ ] **Step 4: Run the configuration tests and verify GREEN**

```bash
pytest tests/test_workloads.py -q
```

Expected: all configuration tests pass.

- [ ] **Step 5: Commit the configuration slice**

```bash
git add src/agent_infer_lab/workloads.py tests/test_workloads.py
git commit -m "feat: validate workload configuration"
```

### Task 2: Deterministic request generation

**Files:**
- Modify: `src/agent_infer_lab/workloads.py`
- Modify: `tests/test_workloads.py`

**Interfaces:**
- Consumes: `WorkloadConfig` from Task 1.
- Produces: `RequestSpec` and `generate_workload(config) -> tuple[RequestSpec, ...]`.

- [ ] **Step 1: Extend the imports and append failing generation tests**

Replace the import block at the top of `tests/test_workloads.py` with:

```python
import random
from dataclasses import FrozenInstanceError

import pytest

from agent_infer_lab.workloads import RequestSpec, WorkloadConfig, generate_workload
```

Then append these tests:

```python
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
```

- [ ] **Step 2: Run only the generation tests and verify RED**

```bash
pytest tests/test_workloads.py -q
```

Expected: collection fails because `RequestSpec` and `generate_workload` do not exist yet.

- [ ] **Step 3: Add the minimal generation implementation**

Add `import random` above the existing `dataclasses` import, then append to `src/agent_infer_lab/workloads.py`:

```python
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
```

- [ ] **Step 4: Run the workload tests and verify GREEN**

```bash
pytest tests/test_workloads.py -q
```

Expected: all workload tests pass.

- [ ] **Step 5: Run the complete quality gate**

```bash
pytest -q
ruff check .
uv lock --check
```

Expected: every test passes, Ruff reports `All checks passed!`, and the lock file is current.

- [ ] **Step 6: Commit the generator slice**

```bash
git add src/agent_infer_lab/workloads.py tests/test_workloads.py
git commit -m "feat: generate deterministic workloads"
```
