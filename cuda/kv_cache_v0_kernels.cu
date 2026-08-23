#include <cstdint>
#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void append_kv_cache_v0_kernel(
    const at::Half* keys,
    const at::Half* values,
    at::Half* key_cache,
    at::Half* value_cache,
    const std::int64_t* slot_mapping,
    std::int64_t num_tokens,
    std::int64_t values_per_token
) {
    const std::int64_t token_index =
        static_cast<std::int64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    if (token_index >= num_tokens) {
        return;
    }

    const std::int64_t slot =
        slot_mapping[token_index];

    const std::int64_t input_start =
        token_index * values_per_token;
    const std::int64_t cache_start =
        slot * values_per_token;

    for (
        std::int64_t value_offset = 0;
        value_offset < values_per_token;
        ++value_offset
    ) {
        const std::int64_t input_index =
            input_start + value_offset;
        const std::int64_t cache_index =
            cache_start + value_offset;

        key_cache[cache_index] =
            keys[input_index];
        value_cache[cache_index] =
            values[input_index];
    }
}

__global__ void gather_kv_cache_v0_kernel(
    const at::Half* key_cache,
    const at::Half* value_cache,
    at::Half* gathered_keys,
    at::Half* gathered_values,
    const std::int64_t* slot_mapping,
    std::int64_t num_tokens,
    std::int64_t values_per_token
) {
    const std::int64_t token_index =
        static_cast<std::int64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    if (token_index >= num_tokens) {
        return;
    }

    const std::int64_t slot =
        slot_mapping[token_index];

    const std::int64_t cache_start =
        slot * values_per_token;
    const std::int64_t output_start =
        token_index * values_per_token;

    for (
        std::int64_t value_offset = 0;
        value_offset < values_per_token;
        ++value_offset
    ) {
        const std::int64_t cache_index =
            cache_start + value_offset;
        const std::int64_t output_index =
            output_start + value_offset;

        gathered_keys[output_index] =
            key_cache[cache_index];
        gathered_values[output_index] =
            value_cache[cache_index];
    }
}

}  // namespace

void append_kv_cache_v0_cuda(
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping
) {
    const c10::cuda::CUDAGuard device_guard(
        keys.device()
    );

    const std::int64_t num_tokens =
        keys.size(0);

    if (num_tokens == 0) {
        return;
    }

    const std::int64_t values_per_token =
        keys.size(1)
        * keys.size(2);

    constexpr int threads_per_block = 128;

    const int num_thread_blocks =
        static_cast<int>(
            (
                num_tokens
                + threads_per_block
                - 1
            )
            / threads_per_block
        );

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            keys.get_device()
        ).stream();

    append_kv_cache_v0_kernel<<<
        num_thread_blocks,
        threads_per_block,
        0,
        stream
    >>>(
        keys.data_ptr<at::Half>(),
        values.data_ptr<at::Half>(),
        key_cache.data_ptr<at::Half>(),
        value_cache.data_ptr<at::Half>(),
        slot_mapping.data_ptr<std::int64_t>(),
        num_tokens,
        values_per_token
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor>
gather_kv_cache_v0_cuda(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping
) {
    const c10::cuda::CUDAGuard device_guard(
        key_cache.device()
    );

    torch::Tensor gathered_keys =
        torch::empty(
            {
                slot_mapping.size(0),
                key_cache.size(2),
                key_cache.size(3),
            },
            key_cache.options()
        );

    torch::Tensor gathered_values =
        torch::empty(
            {
                slot_mapping.size(0),
                value_cache.size(2),
                value_cache.size(3),
            },
            value_cache.options()
        );

    const std::int64_t num_tokens =
        slot_mapping.size(0);

    if (num_tokens == 0) {
        return std::make_tuple(
            gathered_keys,
            gathered_values
        );
    }

    const std::int64_t values_per_token =
        key_cache.size(2)
        * key_cache.size(3);

    constexpr int threads_per_block = 128;

    const int num_thread_blocks =
        static_cast<int>(
            (
                num_tokens
                + threads_per_block
                - 1
            )
            / threads_per_block
        );

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            key_cache.get_device()
        ).stream();

    gather_kv_cache_v0_kernel<<<
        num_thread_blocks,
        threads_per_block,
        0,
        stream
    >>>(
        key_cache.data_ptr<at::Half>(),
        value_cache.data_ptr<at::Half>(),
        gathered_keys.data_ptr<at::Half>(),
        gathered_values.data_ptr<at::Half>(),
        slot_mapping.data_ptr<std::int64_t>(),
        num_tokens,
        values_per_token
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(
        gathered_keys,
        gathered_values
    );
}
