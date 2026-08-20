"""Validate boundary and error cases for CUDA KV cache Gather."""

from collections.abc import Callable

import torch
from load_extension import load_kv_cache_extension
from reference import gather_kv_cache_reference

NUM_BLOCKS = 8
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_DIM = 64
CAPACITY_TOKENS = NUM_BLOCKS * BLOCK_SIZE


def make_cache() -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    key_cache = torch.randn(
        NUM_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.float16,
        device="cuda",
    )
    value_cache = torch.randn_like(key_cache)

    return key_cache, value_cache


def check_valid_case(
    extension: object,
    slots: list[int],
) -> None:
    key_cache, value_cache = make_cache()

    key_cache_before = key_cache.clone()
    value_cache_before = value_cache.clone()

    slot_mapping = torch.tensor(
        slots,
        dtype=torch.int64,
        device="cuda",
    )

    expected_keys, expected_values = (
        gather_kv_cache_reference(
            key_cache,
            value_cache,
            slot_mapping,
        )
    )

    actual_keys, actual_values = (
        extension.gather_kv_cache(
            key_cache,
            value_cache,
            slot_mapping,
        )
    )

    expected_shape = (
        len(slots),
        NUM_KV_HEADS,
        HEAD_DIM,
    )

    if actual_keys.shape != expected_shape:
        raise AssertionError(
            "unexpected gathered Key shape: "
            f"{tuple(actual_keys.shape)}"
        )

    if actual_values.shape != expected_shape:
        raise AssertionError(
            "unexpected gathered Value shape: "
            f"{tuple(actual_values.shape)}"
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

    torch.testing.assert_close(
        key_cache,
        key_cache_before,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        value_cache,
        value_cache_before,
        rtol=0,
        atol=0,
    )


def expect_runtime_error(
    label: str,
    expected_message: str,
    action: Callable[[], None],
) -> None:
    try:
        action()
    except RuntimeError as error:
        if expected_message not in str(error):
            raise AssertionError(
                f"{label}: unexpected error: {error}"
            ) from error
    else:
        raise AssertionError(
            f"{label}: expected RuntimeError"
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA must be available for this test"
        )

    torch.manual_seed(20260820)
    torch.cuda.manual_seed_all(20260820)

    extension = load_kv_cache_extension()

    check_valid_case(extension, [])
    check_valid_case(extension, [0])
    check_valid_case(
        extension,
        [
            15,
            16,
            17,
            CAPACITY_TOKENS - 1,
        ],
    )
    check_valid_case(
        extension,
        [35, 0, 35, 127],
    )

    key_cache, value_cache = make_cache()

    valid_slots = torch.tensor(
        [0, 17],
        dtype=torch.int64,
        device="cuda",
    )

    negative_slot = torch.tensor(
        [-1, 2],
        dtype=torch.int64,
        device="cuda",
    )
    expect_runtime_error(
        "negative slot",
        "slots must be in",
        lambda: extension.gather_kv_cache(
            key_cache,
            value_cache,
            negative_slot,
        ),
    )

    out_of_range_slot = torch.tensor(
        [0, CAPACITY_TOKENS],
        dtype=torch.int64,
        device="cuda",
    )
    expect_runtime_error(
        "out-of-range slot",
        "slots must be in",
        lambda: extension.gather_kv_cache(
            key_cache,
            value_cache,
            out_of_range_slot,
        ),
    )

    int32_slots = torch.tensor(
        [0, 17],
        dtype=torch.int32,
        device="cuda",
    )
    expect_runtime_error(
        "wrong slot dtype",
        "slot_mapping must use torch.int64",
        lambda: extension.gather_kv_cache(
            key_cache,
            value_cache,
            int32_slots,
        ),
    )

    two_dimensional_slots = valid_slots.view(
        1,
        2,
    )
    expect_runtime_error(
        "wrong slot shape",
        "slot_mapping must be one-dimensional",
        lambda: extension.gather_kv_cache(
            key_cache,
            value_cache,
            two_dimensional_slots,
        ),
    )

    float32_key_cache = key_cache.to(
        torch.float32
    )
    expect_runtime_error(
        "wrong Key cache dtype",
        "caches must use torch.float16",
        lambda: extension.gather_kv_cache(
            float32_key_cache,
            value_cache,
            valid_slots,
        ),
    )

    wrong_value_cache = value_cache[
        :,
        :,
        :,
        :-1,
    ].contiguous()
    expect_runtime_error(
        "wrong Value cache shape",
        "value_cache must have the same shape as key_cache",
        lambda: extension.gather_kv_cache(
            key_cache,
            wrong_value_cache,
            valid_slots,
        ),
    )

    noncontiguous_key_cache = torch.randn(
        NUM_BLOCKS,
        BLOCK_SIZE,
        HEAD_DIM,
        NUM_KV_HEADS,
        dtype=torch.float16,
        device="cuda",
    ).transpose(2, 3)

    expect_runtime_error(
        "non-contiguous Key cache",
        "key_cache must be contiguous",
        lambda: extension.gather_kv_cache(
            noncontiguous_key_cache,
            value_cache,
            valid_slots,
        ),
    )

    cpu_slots = torch.tensor(
        [0, 17],
        dtype=torch.int64,
        device="cpu",
    )
    expect_runtime_error(
        "CPU slot mapping",
        "slot_mapping must be a CUDA tensor",
        lambda: extension.gather_kv_cache(
            key_cache,
            value_cache,
            cpu_slots,
        ),
    )

    torch.cuda.synchronize()

    print("CUDA KV cache Gather boundary tests passed")
    print(
        "valid cases: empty, single, "
        "block boundary, duplicate reads"
    )
    print("invalid cases: 8")


if __name__ == "__main__":
    main()
