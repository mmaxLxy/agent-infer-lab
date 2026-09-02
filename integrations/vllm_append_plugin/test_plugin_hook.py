"""Exercise the hook in an isolated process before launching an actual server."""

import os
from types import SimpleNamespace

import torch
from ail_vllm_append_plugin import register
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl


def main() -> None:
    mode = os.environ["AIL_APPEND_MODE"]
    before = FlashAttentionImpl.do_kv_cache_update
    register()
    register()  # Registration must be idempotent.
    after = FlashAttentionImpl.do_kv_cache_update
    assert (before is after) == (mode == "native")
    impl = object.__new__(FlashAttentionImpl)
    impl.attn_type = AttentionType.DECODER
    impl.kv_cache_dtype = "auto"
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    layer = SimpleNamespace(_k_scale=scale, _v_scale=scale)

    for stream in (torch.cuda.default_stream(), torch.cuda.Stream()):
        with torch.cuda.stream(stream):
            for n in (1, 6, 33):
                qkv = torch.randn(n + 2, 18, 64, device="cuda", dtype=torch.float16)
                k, v = qkv[:, 14:16], qkv[:, 16:18]
                slots = torch.randperm(128, device="cuda", dtype=torch.int64)[:n].contiguous()
                if n > 1:
                    slots[n // 2] = -1
                cache = torch.zeros(8, 2, 16, 2, 64, device="cuda", dtype=torch.float16)
                expected = torch.zeros_like(cache)
                for row, slot in enumerate(slots.cpu().tolist()):
                    if slot >= 0:
                        block, offset = divmod(slot, 16)
                        expected[block, 0, offset].copy_(k[row])
                        expected[block, 1, offset].copy_(v[row])
                impl.do_kv_cache_update(layer, k, v, cache, slots)
                stream.synchronize()
                assert torch.equal(cache, expected)
    print(f"Hook mode={mode}: PASS (6 cases; default and non-default stream)")


if __name__ == "__main__":
    main()
