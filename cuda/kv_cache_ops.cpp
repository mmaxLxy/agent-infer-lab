#include <cstdint>

#include <torch/extension.h>

torch::Tensor resolve_slots_cuda(
    torch::Tensor slots,
    std::int64_t block_size
);

torch::Tensor resolve_slots(
    torch::Tensor slots,
    std::int64_t block_size
) {
    TORCH_CHECK(
        slots.is_cuda(),
        "slots must be a CUDA tensor"
    );
    TORCH_CHECK(
        slots.scalar_type() == at::kLong,
        "slots must use torch.int64"
    );
    TORCH_CHECK(
        slots.dim() == 1,
        "slots must be a one-dimensional tensor"
    );
    TORCH_CHECK(
        slots.is_contiguous(),
        "slots must be contiguous"
    );
    TORCH_CHECK(
        block_size > 0,
        "block_size must be positive"
    );

    return resolve_slots_cuda(slots, block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "resolve_slots",
        &resolve_slots,
        "Resolve linear KV cache slots on CUDA"
    );
}
