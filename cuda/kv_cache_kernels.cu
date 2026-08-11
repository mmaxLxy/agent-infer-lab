#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void resolve_slots_kernel(
    const std::int64_t* slots,
    std::int64_t* locations,
    std::int64_t num_slots,
    std::int64_t block_size
) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (index >= num_slots) {
        return;
    }

    const std::int64_t slot = slots[index];

    locations[index * 2] = slot / block_size;
    locations[index * 2 + 1] = slot % block_size;
}

}  // namespace

torch::Tensor resolve_slots_cuda(
    torch::Tensor slots,
    std::int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(slots.device());

    torch::Tensor locations = torch::empty(
        {slots.size(0), 2},
        slots.options()
    );

    const std::int64_t num_slots = slots.numel();

    if (num_slots == 0) {
        return locations;
    }

    constexpr int threads_per_block = 256;
    const int num_thread_blocks = static_cast<int>(
        (num_slots + threads_per_block - 1)
        / threads_per_block
    );

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            slots.get_device()
        ).stream();

    resolve_slots_kernel<<<
        num_thread_blocks,
        threads_per_block,
        0,
        stream
    >>>(
        slots.data_ptr<std::int64_t>(),
        locations.data_ptr<std::int64_t>(),
        num_slots,
        block_size
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return locations;
}
