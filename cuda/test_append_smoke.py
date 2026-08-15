"""Validate the naive CUDA paged KV cache Append kernel."""

import torch
from load_extension import load_kv_cache_extension
from reference import append_kv_cache_reference


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available for this test")

    torch.manual_seed(20260812)
    torch.cuda.manual_seed_all(20260812)

    extension = load_kv_cache_extension()

    num_blocks = 8
    block_size = 16
    num_kv_heads = 2
    head_dim = 64
    num_tokens = 6

    keys = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    values = torch.randn_like(keys)

    slot_mapping = torch.tensor(
        [0, 15, 16, 17, 35, 127],
        dtype=torch.int64,
        device="cuda",
    )

    cache_shape = (
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
    )

    initial_key_cache = torch.full(
        cache_shape,
        -7.0,
        dtype=torch.float16,
        device="cuda",
    )
    initial_value_cache = torch.full(
        cache_shape,
        11.0,
        dtype=torch.float16,
        device="cuda",
    )

    expected_key_cache = initial_key_cache.clone()
    expected_value_cache = initial_value_cache.clone()

    append_kv_cache_reference(
        keys,
        values,
        expected_key_cache,
        expected_value_cache,
        slot_mapping,
    )

    actual_key_cache = initial_key_cache.clone()
    actual_value_cache = initial_value_cache.clone()

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

    print("Naive CUDA KV cache Append test passed")
    print("cache shape:", list(cache_shape))
    print("slot mapping:", slot_mapping.cpu().tolist())
    print(
        "copied FP16 values:",
        keys.numel() + values.numel(),
    )


if __name__ == "__main__":
    main()
