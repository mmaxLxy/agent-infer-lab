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

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 128;

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

int lane_group_size_for_chunks(
    std::int64_t chunks_per_token
) {
    int lane_group_size = 1;

    while (
        lane_group_size < chunks_per_token
        && lane_group_size < kWarpSize
    ) {
        lane_group_size *= 2;
    }

    return lane_group_size;
}

__global__ void append_kv_cache_v3_kernel(
    const at::Half* keys,
    const at::Half* values,
    at::Half* key_cache,
    at::Half* value_cache,
    const std::int64_t* slot_mapping,
    std::int64_t num_tokens,
    std::int64_t values_per_token,
    std::int64_t chunks_per_token,
    int lane_group_size,
    int tokens_per_warp
) {
    const int warp_index_in_block =
        threadIdx.x / kWarpSize;
    const int lane_index =
        threadIdx.x % kWarpSize;

    const int warps_per_block =
        blockDim.x / kWarpSize;

    const std::int64_t global_warp_index =
        static_cast<std::int64_t>(blockIdx.x)
            * warps_per_block
        + warp_index_in_block;

    const int token_group_index =
        lane_index / lane_group_size;
    const int lane_index_in_group =
        lane_index % lane_group_size;

    if (token_group_index >= tokens_per_warp) {
        return;
    }

    const std::int64_t token_index =
        global_warp_index * tokens_per_warp
        + token_group_index;

    if (token_index >= num_tokens) {
        return;
    }

    const std::int64_t slot =
        slot_mapping[token_index];

    for (
        std::int64_t chunk_index =
            lane_index_in_group;
        chunk_index < chunks_per_token;
        chunk_index += lane_group_size
    ) {
        const std::int64_t value_offset =
            chunk_index * kValuesPerVector;

        const std::int64_t remaining_values =
            values_per_token - value_offset;

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

            continue;
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
}

__global__ void gather_kv_cache_v3_kernel(
    const at::Half* key_cache,
    const at::Half* value_cache,
    at::Half* gathered_keys,
    at::Half* gathered_values,
    const std::int64_t* slot_mapping,
    std::int64_t num_tokens,
    std::int64_t values_per_token,
    std::int64_t chunks_per_token,
    int lane_group_size,
    int tokens_per_warp
) {
    const int warp_index_in_block =
        threadIdx.x / kWarpSize;
    const int lane_index =
        threadIdx.x % kWarpSize;

    const int warps_per_block =
        blockDim.x / kWarpSize;

    const std::int64_t global_warp_index =
        static_cast<std::int64_t>(blockIdx.x)
            * warps_per_block
        + warp_index_in_block;

    const int token_group_index =
        lane_index / lane_group_size;
    const int lane_index_in_group =
        lane_index % lane_group_size;

    if (token_group_index >= tokens_per_warp) {
        return;
    }

    const std::int64_t token_index =
        global_warp_index * tokens_per_warp
        + token_group_index;

    if (token_index >= num_tokens) {
        return;
    }

    const std::int64_t slot =
        slot_mapping[token_index];

    for (
        std::int64_t chunk_index =
            lane_index_in_group;
        chunk_index < chunks_per_token;
        chunk_index += lane_group_size
    ) {
        const std::int64_t value_offset =
            chunk_index * kValuesPerVector;

        const std::int64_t remaining_values =
            values_per_token - value_offset;

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

            continue;
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
}

}  // namespace

void append_kv_cache_v3_cuda(
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

    const int lane_group_size =
        lane_group_size_for_chunks(
            chunks_per_token
        );

    const int tokens_per_warp =
        kWarpSize / lane_group_size;

    const std::int64_t num_warps =
        (
            num_tokens
            + tokens_per_warp
            - 1
        )
        / tokens_per_warp;

    constexpr int warps_per_block =
        kThreadsPerBlock / kWarpSize;

    const int num_thread_blocks =
        static_cast<int>(
            (
                num_warps
                + warps_per_block
                - 1
            )
            / warps_per_block
        );

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            keys.get_device()
        ).stream();

    append_kv_cache_v3_kernel<<<
        num_thread_blocks,
        kThreadsPerBlock,
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
        chunks_per_token,
        lane_group_size,
        tokens_per_warp
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_kv_cache_v3_out_cuda(
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

    const int lane_group_size =
        lane_group_size_for_chunks(
            chunks_per_token
        );

    const int tokens_per_warp =
        kWarpSize / lane_group_size;

    const std::int64_t num_warps =
        (
            num_tokens
            + tokens_per_warp
            - 1
        )
        / tokens_per_warp;

    constexpr int warps_per_block =
        kThreadsPerBlock / kWarpSize;

    const int num_thread_blocks =
        static_cast<int>(
            (
                num_warps
                + warps_per_block
                - 1
            )
            / warps_per_block
        );

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            key_cache.get_device()
        ).stream();

    gather_kv_cache_v3_kernel<<<
        num_thread_blocks,
        kThreadsPerBlock,
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
        chunks_per_token,
        lane_group_size,
        tokens_per_warp
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor>
gather_kv_cache_v3_cuda(
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

    gather_kv_cache_v3_out_cuda(
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
