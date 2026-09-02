"""FP16/NHD vLLM Append adapter using the V3 thread layout.

Supports strided token rows and strided cache blocks.
Negative slots are skipped; nonnegative slots must be unique and in range.
This is an integration adapter, not a new ablation version.
"""

from functools import lru_cache

from torch.utils.cpp_extension import load_inline

_CPP = r"""
void append_fp16_(
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slots
);
"""

_CUDA = r"""
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

using I = std::int64_t;
using Half = at::Half;

__global__ void append_v3_strided_kernel(
    const Half* keys,
    const Half* values,
    Half* key_cache,
    Half* value_cache,
    const I* slots,
    I num_tokens,
    I width,
    I chunks,
    int group,
    I block_size,
    I capacity,
    I key_row_stride,
    I value_row_stride,
    I key_block_stride,
    I value_block_stride
) {
    const int lane = threadIdx.x % 32;
    const I warp = static_cast<I>(blockIdx.x) * (blockDim.x / 32)
        + threadIdx.x / 32;
    const I token = warp * (32 / group) + lane / group;

    if (token >= num_tokens) {
        return;
    }

    const I slot = slots[token];
    if (slot < 0) {
        return;
    }
    CUDA_KERNEL_ASSERT(slot < capacity);

    const I block = slot / block_size;
    const I offset = slot % block_size;

    for (I chunk = lane % group; chunk < chunks; chunk += group) {
        const I element = chunk * 8;
        const I remaining = width - element;

        const Half* ki = keys + token * key_row_stride + element;
        const Half* vi = values + token * value_row_stride + element;
        Half* ko = key_cache + block * key_block_stride
            + offset * width + element;
        Half* vo = value_cache + block * value_block_stride
            + offset * width + element;

        const std::uintptr_t addresses =
            reinterpret_cast<std::uintptr_t>(ki)
            | reinterpret_cast<std::uintptr_t>(vi)
            | reinterpret_cast<std::uintptr_t>(ko)
            | reinterpret_cast<std::uintptr_t>(vo);

        if (remaining >= 8 && (addresses & 15) == 0) {
            const uint4 k = *reinterpret_cast<const uint4*>(ki);
            const uint4 v = *reinterpret_cast<const uint4*>(vi);
            *reinterpret_cast<uint4*>(ko) = k;
            *reinterpret_cast<uint4*>(vo) = v;
        } else {
            const I count = remaining < 8 ? remaining : 8;
            for (I j = 0; j < count; ++j) {
                ko[j] = ki[j];
                vo[j] = vi[j];
            }
        }
    }
}

void append_fp16_(
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slots
) {
    for (const auto& tensor :
         {keys, values, key_cache, value_cache, slots}) {
        TORCH_CHECK(tensor.is_cuda(), "all tensors must be CUDA");
        TORCH_CHECK(
            tensor.device() == keys.device(),
            "all tensors must use the same device"
        );
    }

    for (const auto& tensor :
         {keys, values, key_cache, value_cache}) {
        TORCH_CHECK(
            tensor.scalar_type() == at::kHalf,
            "this adapter supports FP16 only"
        );
    }

    TORCH_CHECK(keys.dim() == 3 && values.dim() == 3,
                "inputs must be [tokens, heads, dim]");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                "caches must be [blocks, block_size, heads, dim]");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(),
                "cache shapes must match");
    TORCH_CHECK(slots.dim() == 1 && slots.is_contiguous(),
                "slots must be a contiguous vector");
    TORCH_CHECK(slots.scalar_type() == at::kLong,
                "slots must be int64");

    const I n = slots.numel();
    const I heads = keys.size(1);
    const I dim = keys.size(2);
    const I width = heads * dim;
    const I block_size = key_cache.size(1);

    TORCH_CHECK(heads > 0 && dim > 0 && block_size > 0,
                "heads, dim and block_size must be positive");
    TORCH_CHECK(keys.size(0) >= n && values.size(0) >= n,
                "inputs must contain all mapped token rows");
    TORCH_CHECK(values.size(1) == heads && values.size(2) == dim,
                "Key and Value head shapes must match");
    TORCH_CHECK(
        key_cache.size(2) == heads && key_cache.size(3) == dim,
        "input and cache head shapes must match"
    );

    for (const auto& tensor : {keys, values}) {
        TORCH_CHECK(
            tensor.stride(2) == 1
            && tensor.stride(1) == dim
            && tensor.stride(0) >= width,
            "inputs need contiguous elements within each token"
        );
    }

    for (const auto& tensor : {key_cache, value_cache}) {
        TORCH_CHECK(
            tensor.stride(3) == 1
            && tensor.stride(2) == dim
            && tensor.stride(1) == width
            && tensor.stride(0) >= block_size * width,
            "only NHD caches with nonoverlapping blocks are supported"
        );
    }

    if (n == 0) {
        return;
    }

    TORCH_CHECK(key_cache.size(0) > 0, "cache must have capacity");
    const c10::cuda::CUDAGuard guard(keys.device());

    const I chunks = (width + 7) / 8;
    int group = 1;
    while (group < chunks && group < 32) {
        group *= 2;
    }

    constexpr int threads = 128;
    const I tokens_per_block = (threads / 32) * (32 / group);
    const I blocks = (n + tokens_per_block - 1) / tokens_per_block;

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(keys.get_device()).stream();

    append_v3_strided_kernel<<<blocks, threads, 0, stream>>>(
        keys.data_ptr<Half>(),
        values.data_ptr<Half>(),
        key_cache.data_ptr<Half>(),
        value_cache.data_ptr<Half>(),
        slots.data_ptr<I>(),
        n, width, chunks, group, block_size,
        key_cache.size(0) * block_size,
        keys.stride(0), values.stride(0),
        key_cache.stride(0), value_cache.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def load_adapter():
    """Build a separate extension without changing the V0-V3 extension."""
    return load_inline(
        name="agent_infer_lab_vllm_append",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["append_fp16_"],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        with_cuda=True,
        verbose=True,
    )


def main() -> None:
    import torch
    from vllm import _custom_ops as ops

    extension = load_adapter()
    torch.manual_seed(20260831)
    shapes = [
        (0, 2, 64),
        (1, 2, 64),
        (6, 2, 64),
        (33, 2, 64),
        (17, 3, 65),
        (9, 4, 128),
    ]
    streams = [torch.cuda.default_stream(), torch.cuda.Stream()]
    adapter_passed = 0
    native_passed = 0
    adapter_only_passed = 0

    for stream_index, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            for n, heads, dim in shapes:
                shift = int(dim % 8 != 0)
                count = (n + 2) * 3 * heads * dim
                storage = torch.randn(
                    count + shift, device="cuda", dtype=torch.float16
                )
                qkv = storage[shift:].view(n + 2, 3 * heads, dim)
                keys = qkv[:, heads:2 * heads, :]
                values = qkv[:, 2 * heads:, :]
                slots = torch.randperm(
                    128, device="cuda", dtype=torch.int64
                )[:n].contiguous()
                if n > 1:
                    slots[n // 2] = -1

                actual = torch.zeros(
                    8, 2, 16, heads, dim,
                    device="cuda", dtype=torch.float16,
                )
                expected = torch.zeros_like(actual)
                ak, av = actual.unbind(1)
                for token, slot in enumerate(slots.cpu().tolist()):
                    if slot >= 0:
                        block, offset = divmod(slot, 16)
                        expected[block, 0, offset].copy_(keys[token])
                        expected[block, 1, offset].copy_(values[token])

                # Test the adapter on EVERY case, including empty/misaligned.
                # Synchronize before comparison to attribute errors correctly.
                stream.synchronize()
                extension.append_fp16_(keys, values, ak, av, slots)
                stream.synchronize()
                torch.testing.assert_close(
                    actual, expected, rtol=0, atol=0
                )
                adapter_passed += 1

                # Only these generated cases meet the tested native contract.
                # This is not a universal predicate for all vLLM input layouts.
                native_case = n > 0 and shift == 0 and dim % 8 == 0
                if native_case:
                    native = torch.zeros_like(actual)
                    nk, nv = native.unbind(1)
                    scale = torch.ones(
                        (), device="cuda", dtype=torch.float32
                    )
                    ops.reshape_and_cache_flash(
                        keys, values, nk, nv, slots,
                        "auto", scale, scale,
                    )
                    stream.synchronize()
                    torch.testing.assert_close(
                        native, expected, rtol=0, atol=0
                    )
                    native_passed += 1
                    scope = "native + adapter + reference"
                else:
                    adapter_only_passed += 1
                    scope = "adapter + reference; native not invoked"

                print(
                    f"PASS stream={stream_index} tokens={n} "
                    f"heads={heads} dim={dim}: {scope}",
                    flush=True,
                )

    assert adapter_passed == 12
    assert native_passed == 8
    assert adapter_only_passed == 4
    print("Adapter / reference: PASS (12 cases)")
    print("Native / adapter / reference: PASS (8 supported cases)")
    print("Adapter-only boundary checks: PASS (4 cases)")
    print("streams: default and non-default")
    print("verified: strides, padding, negative slots, empty, odd-sized")


if __name__ == "__main__":
    main()
