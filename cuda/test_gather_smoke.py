"""Validate the CUDA paged KV cache Gather kernel."""

import torch
from load_extension import load_kv_cache_extension
from reference import gather_kv_cache_reference


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available for this test")

    torch.manual_seed(20260815)
    torch.cuda.manual_seed_all(20260815)

    extension = load_kv_cache_extension()

    num_blocks = 8
    block_size = 16
    num_kv_heads = 2
    head_dim = 64

    cache_shape = (
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
    )

    key_cache = torch.randn(
        cache_shape,
        dtype=torch.float16,
        device="cuda",
    )
    value_cache = torch.randn_like(key_cache)

    key_cache_before = key_cache.clone()
    value_cache_before = value_cache.clone()

    slot_mapping = torch.tensor(
        [35, 0, 127, 16, 15, 35],
        dtype=torch.int64,
        device="cuda",
    )

    expected_keys, expected_values = gather_kv_cache_reference(
        key_cache,
        value_cache,
        slot_mapping,
    )

    actual_keys, actual_values = extension.gather_kv_cache(
        key_cache,
        value_cache,
        slot_mapping,
    )

    expected_shape = (
        slot_mapping.numel(),
        num_kv_heads,
        head_dim,
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

    torch.cuda.synchronize()

    print("CUDA KV cache Gather correctness test passed")
    print("cache shape:", list(cache_shape))
    print("slot mapping:", slot_mapping.cpu().tolist())
    print("gathered shape:", list(actual_keys.shape))
    print(
        "copied FP16 values:",
        actual_keys.numel() + actual_values.numel(),
    )


if __name__ == "__main__":
    main()
