#include <cstdint>
#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

using Vector = uint4;

constexpr std::int64_t kValuesPerVector =
    sizeof(Vector) / sizeof(at::Half);

__device__ bool is_vector_aligned(
    const at::Half* pointer
) {
    return (
        reinterpret_cast<std::uintptr_t>(
            pointer
        )
        % alignof(Vector)
    ) == 0;
}

__global__ void append_kv_cache_v2_kernel(
    const at::Half* keys,
    const at::Half* values,
    at::Half* key_cache,
    at::Half* value_cache,
    const std::int64_t* slot_mapping,
    std::int64_t num_tokens,
    std::int64_t values_per_token,
    std::int64_t chunks_per_token
) {
    const std::int64_t work_index =
        static_cast<std::int64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    const std::int64_t num_work_items =
        num_tokens * chunks_per_token;

    if (work_index >= num_work_items) {
        return;
    }

    const std::int64_t token_index =
        work_index / chunks_per_token;
    const std::int64_t chunk_index =
        work_index % chunks_per_token;

    const std::int64_t value_offset =
        chunk_index * kValuesPerVector;

    const std::int64_t remaining_values =
        values_per_token - value_offset;

    const std::int64_t slot =
        slot_mapping[token_index];

    const std::int64_t input_index =
        token_index * values_per_token
        + value_offset;

    const std::int64_t cache_index =
        slot * values_per_token
        + value_offset;

    const at::Half* key_input =
        keys + input_index;
    const at::Half* value_input =
        values + input_index;

    at::Half* key_output =
        key_cache + cache_index;
    at::Half* value_output =
        value_cache + cache_index;

    const bool can_use_vector =
        remaining_values >= kValuesPerVector
        && is_vector_aligned(key_input)
        && is_vector_aligned(value_input)
        && is_vector_aligned(key_output)
        && is_vector_aligned(value_output);

    if (can_use_vector) {
        const Vector key_vector =
            *reinterpret_cast<const Vector*>(
                key_input
            );
        const Vector value_vector =
            *reinterpret_cast<const Vector*>(
                value_input
            );

        *reinterpret_cast<Vector*>(
            key_output
        ) = key_vector;
        *reinterpret_cast<Vector*>(
            value_output
        ) = value_vector;

        return;
    }

    const std::int64_t scalar_count =
        remaining_values < kValuesPerVector
        ? remaining_values
        : kValuesPerVector;

    for (
        std::int64_t index = 0;
        index < scalar_count;
        ++index
    ) {
        key_output[index] =
            key_input[index];
        value_output[index] =
            value_input[index];
    }
}

__global__ void gather_kv_cache_v2_kernel(
    const at::Half* key_cache,
    const at::Half* value_cache,
    at::Half* gathered_keys,
    at::Half* gathered_values,
    const std::int64_t* slot_mapping,
    std::int64_t num_tokens,
    std::int64_t values_per_token,
    std::int64_t chunks_per_token
) {
    const std::int64_t work_index =
        static_cast<std::int64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    const std::int64_t num_work_items =
        num_tokens * chunks_per_token;

    if (work_index >= num_work_items) {
        return;
    }

    const std::int64_t token_index =
        work_index / chunks_per_token;
    const std::int64_t chunk_index =
        work_index % chunks_per_token;

    const std::int64_t value_offset =
        chunk_index * kValuesPerVector;

    const std::int64_t remaining_values =
        values_per_token - value_offset;

    const std::int64_t slot =
        slot_mapping[token_index];

    const std::int64_t cache_index =
        slot * values_per_token
        + value_offset;

    const std::int64_t output_index =
        token_index * values_per_token
        + value_offset;

    const at::Half* key_input =
        key_cache + cache_index;
    const at::Half* value_input =
        value_cache + cache_index;

    at::Half* key_output =
        gathered_keys + output_index;
    at::Half* value_output =
        gathered_values + output_index;

    const bool can_use_vector =
        remaining_values >= kValuesPerVector
        && is_vector_aligned(key_input)
        && is_vector_aligned(value_input)
        && is_vector_aligned(key_output)
        && is_vector_aligned(value_output);

    if (can_use_vector) {
        const Vector key_vector =
            *reinterpret_cast<const Vector*>(
                key_input
            );
        const Vector value_vector =
            *reinterpret_cast<const Vector*>(
                value_input
            );

        *reinterpret_cast<Vector*>(
            key_output
        ) = key_vector;
        *reinterpret_cast<Vector*>(
            value_output
        ) = value_vector;

        return;
    }

    const std::int64_t scalar_count =
        remaining_values < kValuesPerVector
        ? remaining_values
        : kValuesPerVector;

    for (
        std::int64_t index = 0;
        index < scalar_count;
        ++index
    ) {
        key_output[index] =
            key_input[index];
        value_output[index] =
            value_input[index];
    }
}

}  // namespace

void append_kv_cache_v2_cuda(
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

    const std::int64_t chunks_per_token =
        (
            values_per_token
            + kValuesPerVector
            - 1
        )
        / kValuesPerVector;

    const std::int64_t num_work_items =
        num_tokens * chunks_per_token;

    constexpr int threads_per_block = 256;

    const int num_thread_blocks =
        static_cast<int>(
            (
                num_work_items
                + threads_per_block
                - 1
            )
            / threads_per_block
        );

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            keys.get_device()
        ).stream();

    append_kv_cache_v2_kernel<<<
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
        values_per_token,
        chunks_per_token
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_kv_cache_v2_out_cuda(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping,
    torch::Tensor gathered_keys,
    torch::Tensor gathered_values
) {
    const c10::cuda::CUDAGuard device_guard(
        key_cache.device()
    );

    const std::int64_t num_tokens =
        slot_mapping.size(0);

    if (num_tokens == 0) {
        return;
    }

    const std::int64_t values_per_token =
        key_cache.size(2)
        * key_cache.size(3);

    const std::int64_t chunks_per_token =
        (
            values_per_token
            + kValuesPerVector
            - 1
        )
        / kValuesPerVector;

    const std::int64_t num_work_items =
        num_tokens * chunks_per_token;

    constexpr int threads_per_block = 256;

    const int num_thread_blocks =
        static_cast<int>(
            (
                num_work_items
                + threads_per_block
                - 1
            )
            / threads_per_block
        );

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            key_cache.get_device()
        ).stream();

    gather_kv_cache_v2_kernel<<<
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
        values_per_token,
        chunks_per_token
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor>
gather_kv_cache_v2_cuda(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping
) {
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

    gather_kv_cache_v2_out_cuda(
        key_cache,
        value_cache,
        slot_mapping,
        gathered_keys,
        gathered_values
    );

    return std::make_tuple(
        gathered_keys,
        gathered_values
    );
}
