"""Compare versioned CUDA kernels with PyTorch references."""

import torch
from load_extension import load_kv_cache_extension
from reference import (
    append_kv_cache_reference,
    gather_kv_cache_reference,
)


def run_case(
    extension: object,
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    append_slots: list[int],
    gather_slots: list[int],
) -> None:
    num_tokens = len(append_slots)

    keys = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    values = torch.randn_like(keys)

    cache_shape = (
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
    )

    initial_key_cache = torch.randn(
        cache_shape,
        dtype=torch.float16,
        device="cuda",
    )
    initial_value_cache = torch.randn_like(
        initial_key_cache
    )

    append_slot_mapping = torch.tensor(
        append_slots,
        dtype=torch.int64,
        device="cuda",
    )

    expected_key_cache = initial_key_cache.clone()
    expected_value_cache = (
        initial_value_cache.clone()
    )

    append_kv_cache_reference(
        keys,
        values,
        expected_key_cache,
        expected_value_cache,
        append_slot_mapping,
    )

    v0_key_cache = initial_key_cache.clone()
    v0_value_cache = initial_value_cache.clone()

    extension.append_kv_cache_v0_(
        keys,
        values,
        v0_key_cache,
        v0_value_cache,
        append_slot_mapping,
    )

    v1_key_cache = initial_key_cache.clone()
    v1_value_cache = (
        initial_value_cache.clone()
    )

    extension.append_kv_cache_v1_(
        keys,
        values,
        v1_key_cache,
        v1_value_cache,
        append_slot_mapping,
    )

    v2_key_cache = initial_key_cache.clone()
    v2_value_cache = (
        initial_value_cache.clone()
    )

    extension.append_kv_cache_v2_(
        keys,
        values,
        v2_key_cache,
        v2_value_cache,
        append_slot_mapping,
    )

    for actual_key_cache in (
        v0_key_cache,
        v1_key_cache,
        v2_key_cache,
    ):
        torch.testing.assert_close(
            actual_key_cache,
            expected_key_cache,
            rtol=0,
            atol=0,
        )

    for actual_value_cache in (
        v0_value_cache,
        v1_value_cache,
        v2_value_cache,
    ):
        torch.testing.assert_close(
            actual_value_cache,
            expected_value_cache,
            rtol=0,
            atol=0,
        )

    gather_slot_mapping = torch.tensor(
        gather_slots,
        dtype=torch.int64,
        device="cuda",
    )

    key_cache_before_gather = (
        expected_key_cache.clone()
    )
    value_cache_before_gather = (
        expected_value_cache.clone()
    )

    expected_keys, expected_values = (
        gather_kv_cache_reference(
            expected_key_cache,
            expected_value_cache,
            gather_slot_mapping,
        )
    )

    v0_keys, v0_values = (
        extension.gather_kv_cache_v0(
            expected_key_cache,
            expected_value_cache,
            gather_slot_mapping,
        )
    )

    v1_keys, v1_values = (
        extension.gather_kv_cache_v1(
            expected_key_cache,
            expected_value_cache,
            gather_slot_mapping,
        )
    )

    v2_keys, v2_values = (
        extension.gather_kv_cache_v2(
            expected_key_cache,
            expected_value_cache,
            gather_slot_mapping,
        )
    )

    for actual_keys in (
        v0_keys,
        v1_keys,
        v2_keys,
    ):
        torch.testing.assert_close(
            actual_keys,
            expected_keys,
            rtol=0,
            atol=0,
        )

    for actual_values in (
        v0_values,
        v1_values,
        v2_values,
    ):
        torch.testing.assert_close(
            actual_values,
            expected_values,
            rtol=0,
            atol=0,
        )

    torch.testing.assert_close(
        v0_keys,
        v1_keys,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        v1_keys,
        v2_keys,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        v0_values,
        v1_values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        v1_values,
        v2_values,
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(
        expected_key_cache,
        key_cache_before_gather,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        expected_value_cache,
        value_cache_before_gather,
        rtol=0,
        atol=0,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA must be available for this test"
        )

    torch.manual_seed(20260821)
    torch.cuda.manual_seed_all(20260821)

    extension = load_kv_cache_extension()

    run_case(
        extension,
        num_blocks=8,
        block_size=16,
        num_kv_heads=2,
        head_dim=64,
        append_slots=[
            0,
            15,
            16,
            17,
            35,
            127,
        ],
        gather_slots=[
            35,
            0,
            127,
            16,
            15,
            35,
        ],
    )

    run_case(
        extension,
        num_blocks=4,
        block_size=16,
        num_kv_heads=3,
        head_dim=65,
        append_slots=[
            1,
            16,
            63,
        ],
        gather_slots=[
            63,
            1,
            63,
            16,
        ],
    )

    run_case(
        extension,
        num_blocks=2,
        block_size=8,
        num_kv_heads=1,
        head_dim=7,
        append_slots=[],
        gather_slots=[],
    )

    torch.cuda.synchronize()

    print(
        "Versioned CUDA KV cache correctness "
        "tests passed"
    )
    print(
        "implementations: PyTorch, V0, V1, V2"
    )
    print("shapes: main, odd-sized, empty")
    print("Append: unique slots")
    print(
        "Gather: reordered and duplicate slots"
    )


if __name__ == "__main__":
    main()
