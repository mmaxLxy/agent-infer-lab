#include <cstdint>
#include <tuple>

#include <torch/extension.h>

torch::Tensor resolve_slots_cuda(
    torch::Tensor slots,
    std::int64_t block_size
);

void append_kv_cache_cuda(
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping
);

std::tuple<torch::Tensor, torch::Tensor>
gather_kv_cache_cuda(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping
);

namespace {

void check_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(
        tensor.is_cuda(),
        name,
        " must be a CUDA tensor"
    );
    TORCH_CHECK(
        tensor.is_contiguous(),
        name,
        " must be contiguous"
    );
}

void check_unique_slots(
    const torch::Tensor& slot_mapping
) {
    const std::int64_t num_slots =
        slot_mapping.numel();

    if (num_slots < 2) {
        return;
    }

    const auto sort_result =
        at::sort(slot_mapping);
    const torch::Tensor sorted_slots =
        std::get<0>(sort_result);

    const torch::Tensor previous_slots =
        sorted_slots.slice(
            0,
            0,
            num_slots - 1
        );
    const torch::Tensor next_slots =
        sorted_slots.slice(
            0,
            1,
            num_slots
        );

    const bool has_duplicate =
        at::any(
            at::eq(
                previous_slots,
                next_slots
            )
        ).item<bool>();

    TORCH_CHECK(
        !has_duplicate,
        "slot_mapping must contain unique slots"
    );
}

void check_slot_bounds(
    const torch::Tensor& slot_mapping,
    std::int64_t capacity_tokens
) {
    if (slot_mapping.numel() == 0) {
        return;
    }

    const std::int64_t minimum_slot =
        slot_mapping
            .min()
            .item<std::int64_t>();
    const std::int64_t maximum_slot =
        slot_mapping
            .max()
            .item<std::int64_t>();

    TORCH_CHECK(
        minimum_slot >= 0
            && maximum_slot < capacity_tokens,
        "slots must be in [0, ",
        capacity_tokens,
        ")"
    );
}

}  // namespace

torch::Tensor resolve_slots(
    torch::Tensor slots,
    std::int64_t block_size
) {
    check_cuda_contiguous(slots, "slots");

    TORCH_CHECK(
        slots.scalar_type() == at::kLong,
        "slots must use torch.int64"
    );
    TORCH_CHECK(
        slots.dim() == 1,
        "slots must be a one-dimensional tensor"
    );
    TORCH_CHECK(
        block_size > 0,
        "block_size must be positive"
    );

    return resolve_slots_cuda(
        slots,
        block_size
    );
}

void append_kv_cache(
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping
) {
    check_cuda_contiguous(keys, "keys");
    check_cuda_contiguous(values, "values");
    check_cuda_contiguous(
        key_cache,
        "key_cache"
    );
    check_cuda_contiguous(
        value_cache,
        "value_cache"
    );
    check_cuda_contiguous(
        slot_mapping,
        "slot_mapping"
    );

    TORCH_CHECK(
        keys.dim() == 3,
        "keys must have shape "
        "[num_tokens, num_kv_heads, head_dim]"
    );
    TORCH_CHECK(
        values.sizes() == keys.sizes(),
        "values must have the same shape as keys"
    );
    TORCH_CHECK(
        key_cache.dim() == 4,
        "key_cache must have shape "
        "[num_blocks, block_size, "
        "num_kv_heads, head_dim]"
    );
    TORCH_CHECK(
        value_cache.sizes()
            == key_cache.sizes(),
        "value_cache must have the same "
        "shape as key_cache"
    );
    TORCH_CHECK(
        keys.size(1) == key_cache.size(2)
            && keys.size(2)
                == key_cache.size(3),
        "input KV dimensions must match "
        "cache KV dimensions"
    );
    TORCH_CHECK(
        slot_mapping.dim() == 1,
        "slot_mapping must be "
        "one-dimensional"
    );
    TORCH_CHECK(
        slot_mapping.size(0)
            == keys.size(0),
        "slot_mapping length must match "
        "num_tokens"
    );
    TORCH_CHECK(
        slot_mapping.scalar_type()
            == at::kLong,
        "slot_mapping must use torch.int64"
    );
    TORCH_CHECK(
        keys.scalar_type() == at::kHalf,
        "keys and caches must use "
        "torch.float16"
    );
    TORCH_CHECK(
        values.scalar_type()
                == keys.scalar_type()
            && key_cache.scalar_type()
                == keys.scalar_type()
            && value_cache.scalar_type()
                == keys.scalar_type(),
        "all Key and Value tensors must "
        "use the same dtype"
    );
    TORCH_CHECK(
        values.device() == keys.device()
            && key_cache.device()
                == keys.device()
            && value_cache.device()
                == keys.device()
            && slot_mapping.device()
                == keys.device(),
        "all tensors must be on the same "
        "CUDA device"
    );

    const std::int64_t capacity_tokens =
        key_cache.size(0)
        * key_cache.size(1);

    check_slot_bounds(
        slot_mapping,
        capacity_tokens
    );

    if (slot_mapping.numel() > 0) {
        check_unique_slots(slot_mapping);
    }

    append_kv_cache_cuda(
        keys,
        values,
        key_cache,
        value_cache,
        slot_mapping
    );
}

std::tuple<torch::Tensor, torch::Tensor>
gather_kv_cache(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping
) {
    check_cuda_contiguous(
        key_cache,
        "key_cache"
    );
    check_cuda_contiguous(
        value_cache,
        "value_cache"
    );
    check_cuda_contiguous(
        slot_mapping,
        "slot_mapping"
    );

    TORCH_CHECK(
        key_cache.dim() == 4,
        "key_cache must have shape "
        "[num_blocks, block_size, "
        "num_kv_heads, head_dim]"
    );
    TORCH_CHECK(
        value_cache.sizes()
            == key_cache.sizes(),
        "value_cache must have the same "
        "shape as key_cache"
    );
    TORCH_CHECK(
        slot_mapping.dim() == 1,
        "slot_mapping must be "
        "one-dimensional"
    );
    TORCH_CHECK(
        slot_mapping.scalar_type()
            == at::kLong,
        "slot_mapping must use torch.int64"
    );
    TORCH_CHECK(
        key_cache.scalar_type()
            == at::kHalf,
        "caches must use torch.float16"
    );
    TORCH_CHECK(
        value_cache.scalar_type()
            == key_cache.scalar_type(),
        "Key and Value caches must use "
        "the same dtype"
    );
    TORCH_CHECK(
        value_cache.device()
                == key_cache.device()
            && slot_mapping.device()
                == key_cache.device(),
        "all tensors must be on the same "
        "CUDA device"
    );

    const std::int64_t capacity_tokens =
        key_cache.size(0)
        * key_cache.size(1);

    check_slot_bounds(
        slot_mapping,
        capacity_tokens
    );

    return gather_kv_cache_cuda(
        key_cache,
        value_cache,
        slot_mapping
    );
}

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    module
) {
    module.def(
        "resolve_slots",
        &resolve_slots,
        "Resolve linear KV cache slots "
        "on CUDA"
    );
    module.def(
        "append_kv_cache_",
        &append_kv_cache,
        "Append Key and Value tensors "
        "into a paged KV cache"
    );
    module.def(
        "gather_kv_cache",
        &gather_kv_cache,
        "Gather Key and Value tensors "
        "from a paged KV cache"
    );
}
