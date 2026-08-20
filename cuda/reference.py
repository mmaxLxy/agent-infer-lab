"""PyTorch reference operations for a paged KV cache."""

import torch


def append_kv_cache_reference(
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Append Key and Value vectors into a paged KV cache in place."""

    if keys.ndim != 3:
        raise ValueError(
            "keys must have shape "
            "[num_tokens, num_kv_heads, head_dim]"
        )
    if values.shape != keys.shape:
        raise ValueError(
            "values must have the same shape as keys"
        )
    if key_cache.ndim != 4:
        raise ValueError(
            "key_cache must have shape "
            "[num_blocks, block_size, num_kv_heads, head_dim]"
        )
    if value_cache.shape != key_cache.shape:
        raise ValueError(
            "value_cache must have the same shape as key_cache"
        )
    if keys.shape[1:] != key_cache.shape[2:]:
        raise ValueError(
            "input KV dimensions must match cache KV dimensions"
        )
    if slot_mapping.ndim != 1:
        raise ValueError(
            "slot_mapping must be one-dimensional"
        )
    if slot_mapping.shape[0] != keys.shape[0]:
        raise ValueError(
            "slot_mapping length must match num_tokens"
        )
    if slot_mapping.dtype != torch.int64:
        raise ValueError(
            "slot_mapping must use torch.int64"
        )
    if keys.dtype != torch.float16:
        raise ValueError(
            "keys and caches must use torch.float16"
        )
    if not (
        values.dtype
        == key_cache.dtype
        == value_cache.dtype
        == keys.dtype
    ):
        raise ValueError(
            "all Key and Value tensors must use the same dtype"
        )

    devices = {
        keys.device,
        values.device,
        key_cache.device,
        value_cache.device,
        slot_mapping.device,
    }
    if len(devices) != 1:
        raise ValueError(
            "all tensors must be on the same device"
        )

    if not all(
        tensor.is_contiguous()
        for tensor in (
            keys,
            values,
            key_cache,
            value_cache,
            slot_mapping,
        )
    ):
        raise ValueError(
            "all tensors must be contiguous"
        )

    capacity_tokens = (
        key_cache.shape[0] * key_cache.shape[1]
    )

    if slot_mapping.numel() > 0:
        minimum_slot = int(slot_mapping.min().item())
        maximum_slot = int(slot_mapping.max().item())

        if minimum_slot < 0 or maximum_slot >= capacity_tokens:
            raise ValueError(
                f"slots must be in [0, {capacity_tokens})"
            )

        unique_slots = torch.unique(slot_mapping)
        if unique_slots.numel() != slot_mapping.numel():
            raise ValueError(
                "slot_mapping must contain unique slots"
            )

    flat_key_cache = key_cache.view(
        capacity_tokens,
        key_cache.shape[2],
        key_cache.shape[3],
    )
    flat_value_cache = value_cache.view(
        capacity_tokens,
        value_cache.shape[2],
        value_cache.shape[3],
    )

    flat_key_cache.index_copy_(
        0,
        slot_mapping,
        keys,
    )
    flat_value_cache.index_copy_(
        0,
        slot_mapping,
        values,
    )


def gather_kv_cache_reference(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather Key and Value vectors from a paged KV cache."""

    if key_cache.ndim != 4:
        raise ValueError(
            "key_cache must have shape "
            "[num_blocks, block_size, num_kv_heads, head_dim]"
        )
    if value_cache.shape != key_cache.shape:
        raise ValueError(
            "value_cache must have the same shape as key_cache"
        )
    if slot_mapping.ndim != 1:
        raise ValueError(
            "slot_mapping must be one-dimensional"
        )
    if slot_mapping.dtype != torch.int64:
        raise ValueError(
            "slot_mapping must use torch.int64"
        )
    if key_cache.dtype != torch.float16:
        raise ValueError(
            "caches must use torch.float16"
        )
    if value_cache.dtype != key_cache.dtype:
        raise ValueError(
            "Key and Value caches must use the same dtype"
        )

    devices = {
        key_cache.device,
        value_cache.device,
        slot_mapping.device,
    }
    if len(devices) != 1:
        raise ValueError(
            "all tensors must be on the same device"
        )

    if not all(
        tensor.is_contiguous()
        for tensor in (
            key_cache,
            value_cache,
            slot_mapping,
        )
    ):
        raise ValueError(
            "all tensors must be contiguous"
        )

    capacity_tokens = (
        key_cache.shape[0] * key_cache.shape[1]
    )

    if slot_mapping.numel() > 0:
        minimum_slot = int(slot_mapping.min().item())
        maximum_slot = int(slot_mapping.max().item())

        if minimum_slot < 0 or maximum_slot >= capacity_tokens:
            raise ValueError(
                f"slots must be in [0, {capacity_tokens})"
            )

    flat_key_cache = key_cache.view(
        capacity_tokens,
        key_cache.shape[2],
        key_cache.shape[3],
    )
    flat_value_cache = value_cache.view(
        capacity_tokens,
        value_cache.shape[2],
        value_cache.shape[3],
    )

    gathered_keys = flat_key_cache.index_select(
        0,
        slot_mapping,
    )
    gathered_values = flat_value_cache.index_select(
        0,
        slot_mapping,
    )

    return gathered_keys, gathered_values
