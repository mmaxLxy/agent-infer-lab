"""Validate boundary and error cases for CUDA KV cache Append."""

from collections.abc import Callable

import torch
from load_extension import load_kv_cache_extension
from reference import append_kv_cache_reference

NUM_BLOCKS = 8
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_DIM = 64
CAPACITY_TOKENS = NUM_BLOCKS * BLOCK_SIZE


def make_inputs(
    num_tokens: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    keys = torch.randn(
        num_tokens,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.float16,
        device="cuda",
    )
    values = torch.randn_like(keys)

    key_cache = torch.full(
        (
            NUM_BLOCKS,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
        ),
        -7.0,
        dtype=torch.float16,
        device="cuda",
    )
    value_cache = torch.full(
        (
            NUM_BLOCKS,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
        ),
        11.0,
        dtype=torch.float16,
        device="cuda",
    )

    return (
        keys,
        values,
        key_cache,
        value_cache,
    )


def check_valid_case(
    extension: object,
    slots: list[int],
) -> None:
    (
        keys,
        values,
        initial_key_cache,
        initial_value_cache,
    ) = make_inputs(len(slots))

    slot_mapping = torch.tensor(
        slots,
        dtype=torch.int64,
        device="cuda",
    )

    expected_key_cache = (
        initial_key_cache.clone()
    )
    expected_value_cache = (
        initial_value_cache.clone()
    )

    append_kv_cache_reference(
        keys,
        values,
        expected_key_cache,
        expected_value_cache,
        slot_mapping,
    )

    actual_key_cache = initial_key_cache.clone()
    actual_value_cache = (
        initial_value_cache.clone()
    )

    extension.append_kv_cache_(
        keys,
        values,
        actual_key_cache,
        actual_value_cache,
        slot_mapping,
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

    torch.manual_seed(20260812)
    torch.cuda.manual_seed_all(20260812)

    extension = load_kv_cache_extension()

    check_valid_case(extension, [])
    check_valid_case(extension, [0])
    check_valid_case(
        extension,
        [15, 16, 17, CAPACITY_TOKENS - 1],
    )

    (
        keys,
        values,
        key_cache,
        value_cache,
    ) = make_inputs(2)

    duplicate_slots = torch.tensor(
        [17, 17],
        dtype=torch.int64,
        device="cuda",
    )
    expect_runtime_error(
        "duplicate slots",
        "slot_mapping must contain unique slots",
        lambda: extension.append_kv_cache_(
            keys,
            values,
            key_cache,
            value_cache,
            duplicate_slots,
        ),
    )

    negative_slot = torch.tensor(
        [-1, 2],
        dtype=torch.int64,
        device="cuda",
    )
    expect_runtime_error(
        "negative slot",
        "slots must be in",
        lambda: extension.append_kv_cache_(
            keys,
            values,
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
        lambda: extension.append_kv_cache_(
            keys,
            values,
            key_cache,
            value_cache,
            out_of_range_slot,
        ),
    )

    int32_slots = torch.tensor(
        [0, 1],
        dtype=torch.int32,
        device="cuda",
    )
    expect_runtime_error(
        "wrong slot dtype",
        "slot_mapping must use torch.int64",
        lambda: extension.append_kv_cache_(
            keys,
            values,
            key_cache,
            value_cache,
            int32_slots,
        ),
    )

    float32_keys = keys.to(torch.float32)
    expect_runtime_error(
        "wrong Key dtype",
        "keys and caches must use torch.float16",
        lambda: extension.append_kv_cache_(
            float32_keys,
            values,
            key_cache,
            value_cache,
            torch.tensor(
                [0, 1],
                dtype=torch.int64,
                device="cuda",
            ),
        ),
    )

    wrong_values = values[:, :, :-1].contiguous()
    expect_runtime_error(
        "wrong Value shape",
        "values must have the same shape as keys",
        lambda: extension.append_kv_cache_(
            keys,
            wrong_values,
            key_cache,
            value_cache,
            torch.tensor(
                [0, 1],
                dtype=torch.int64,
                device="cuda",
            ),
        ),
    )

    noncontiguous_keys = torch.randn(
        2,
        HEAD_DIM,
        NUM_KV_HEADS,
        dtype=torch.float16,
        device="cuda",
    ).transpose(1, 2)

    expect_runtime_error(
        "non-contiguous Keys",
        "keys must be contiguous",
        lambda: extension.append_kv_cache_(
            noncontiguous_keys,
            values,
            key_cache,
            value_cache,
            torch.tensor(
                [0, 1],
                dtype=torch.int64,
                device="cuda",
            ),
        ),
    )

    cpu_slots = torch.tensor(
        [0, 1],
        dtype=torch.int64,
        device="cpu",
    )
    expect_runtime_error(
        "CPU slot mapping",
        "slot_mapping must be a CUDA tensor",
        lambda: extension.append_kv_cache_(
            keys,
            values,
            key_cache,
            value_cache,
            cpu_slots,
        ),
    )

    torch.cuda.synchronize()

    print("CUDA KV cache Append boundary tests passed")
    print("valid cases: empty, single, block boundary")
    print("invalid cases: 8")


if __name__ == "__main__":
    main()
