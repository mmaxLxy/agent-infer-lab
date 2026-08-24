"""Validate unchecked CUDA interfaces used by Kernel benchmarks."""

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
    """Compare benchmark interfaces with PyTorch references."""

    num_append_tokens = len(append_slots)

    keys = torch.randn(
        num_append_tokens,
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

    for version in ("v0", "v1"):
        actual_key_cache = initial_key_cache.clone()
        actual_value_cache = (
            initial_value_cache.clone()
        )

        append_function = getattr(
            extension,
            f"append_kv_cache_{version}_unchecked_",
        )

        append_result = append_function(
            keys,
            values,
            actual_key_cache,
            actual_value_cache,
            append_slot_mapping,
        )

        if append_result is not None:
            raise AssertionError(
                f"{version} Append must modify "
                "the caches in place"
            )

        torch.testing.assert_close(
            actual_key_cache,
            expected_key_cache,
            rtol=0,
            atol=0,
        )
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

    expected_keys, expected_values = (
        gather_kv_cache_reference(
            expected_key_cache,
            expected_value_cache,
            gather_slot_mapping,
        )
    )

    output_shape = (
        len(gather_slots),
        num_kv_heads,
        head_dim,
    )

    for version in ("v0", "v1"):
        actual_keys = torch.empty(
            output_shape,
            dtype=torch.float16,
            device="cuda",
        )
        actual_values = torch.empty_like(
            actual_keys
        )

        key_pointer_before = (
            actual_keys.data_ptr()
        )
        value_pointer_before = (
            actual_values.data_ptr()
        )

        gather_function = getattr(
            extension,
            (
                f"gather_kv_cache_{version}"
                "_unchecked_out_"
            ),
        )

        gather_result = gather_function(
            expected_key_cache,
            expected_value_cache,
            gather_slot_mapping,
            actual_keys,
            actual_values,
        )

        if gather_result is not None:
            raise AssertionError(
                f"{version} Gather must write "
                "into preallocated outputs"
            )

        if (
            actual_keys.data_ptr()
            != key_pointer_before
        ):
            raise AssertionError(
                f"{version} Gather replaced "
                "the preallocated Key output"
            )

        if (
            actual_values.data_ptr()
            != value_pointer_before
        ):
            raise AssertionError(
                f"{version} Gather replaced "
                "the preallocated Value output"
            )

        torch.testing.assert_close(
            actual_keys,
            expected_keys,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_values,
            expected_values,
            rtol=0,
            atol=0,
        )


def main() -> None:
    """Run valid benchmark-interface cases."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA must be available for this test"
        )

    torch.manual_seed(20260824)
    torch.cuda.manual_seed_all(20260824)

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
        "CUDA benchmark interface "
        "correctness tests passed"
    )
    print(
        "implementations: PyTorch, V0, V1"
    )
    print(
        "interfaces: unchecked Append, "
        "preallocated Gather"
    )
    print(
        "shapes: main, odd-sized, empty"
    )
    print(
        "preallocated output pointers: preserved"
    )


if __name__ == "__main__":
    main()
