"""Controlled system inputs must be identical across a paired ablation."""

import random

import pytest

from agent_infer_lab.controlled_inputs import (
    audit_inputs,
    common_prefix_length,
    prepare_controlled_requests,
)
from agent_infer_lab.workloads import WorkloadConfig, generate_workload


def tokenize(_: str) -> tuple[int, ...]:
    return tuple(range(100, 612))


@pytest.mark.parametrize("length", [512, 1024, 1536])
def test_controlled_inputs_have_exact_shared_prefix_and_no_duplicates(length: int) -> None:
    config = WorkloadConfig(80, (length,), (64,), 0.75, 4, 7)
    requests = prepare_controlled_requests(generate_workload(config), tokenize, content_seed=8)
    audit = audit_inputs(requests)
    assert audit["input_lengths"] == [length]
    assert audit["duplicate_prompt_count"] == 0
    assert audit["pairwise_common_prefix_min"] == length * 3 // 4
    assert audit["pairwise_common_prefix_max"] == length * 3 // 4
    assert all(request.output_tokens == 64 for request in requests)


def test_content_seed_changes_fixed_length_prompts_without_global_rng_mutation() -> None:
    specs = generate_workload(WorkloadConfig(4, (32,), (4,), 0.5, 2, 7))
    state = random.getstate()
    first = prepare_controlled_requests(specs, tokenize, content_seed=1)
    assert first == prepare_controlled_requests(specs, tokenize, content_seed=1)
    assert first != prepare_controlled_requests(specs, tokenize, content_seed=2)
    assert state == random.getstate()
    assert audit_inputs(first)["input_sha256"] == audit_inputs(first)["input_sha256"]


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_prefix_ratio_extremes(ratio: float) -> None:
    specs = generate_workload(WorkloadConfig(4, (32,), (4,), ratio, 2, 7))
    requests = prepare_controlled_requests(specs, tokenize, content_seed=1)
    assert audit_inputs(requests)["pairwise_common_prefix_min"] == int(32 * ratio)


def test_controlled_inputs_reject_small_token_pool() -> None:
    specs = generate_workload(WorkloadConfig(4, (32,), (4,), 0.5, 2, 7))
    with pytest.raises(ValueError, match="Token pool"):
        prepare_controlled_requests(specs, lambda _: (1, 2), content_seed=1)


@pytest.mark.parametrize("seed", [True, 1.5])
def test_controlled_inputs_reject_invalid_seed(seed: object) -> None:
    specs = generate_workload(WorkloadConfig(4, (32,), (4,), 0.5, 2, 7))
    with pytest.raises(ValueError, match="content_seed"):
        prepare_controlled_requests(specs, tokenize, content_seed=seed)  # type: ignore[arg-type]


def test_prefix_comparison_handles_equal_and_short_inputs() -> None:
    assert common_prefix_length((1, 2), (1, 2, 3)) == 2
    assert common_prefix_length((1, 2), (1, 3)) == 1
    assert common_prefix_length((), (1,)) == 0
