"""Opt-in process-local Append hook. Never modifies installed vLLM files.

Modes: native (default, no hook), verify (native shadow comparison), custom.
Only the vLLM 0.23.0 FlashAttention decoder FP16/auto eager path is supported.
Unsupported inputs fail explicitly; there is no silent native fallback.
"""

import atexit
import hashlib
import importlib.util
import json
import os
import time
from functools import wraps
from pathlib import Path


def register() -> None:
    mode = os.environ.get("AIL_APPEND_MODE", "native")
    if mode not in {"native", "verify", "custom"}:
        raise ValueError("AIL_APPEND_MODE must be native, verify, or custom")
    if mode == "native":
        return

    import torch
    import vllm
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    if vllm.__version__.split("+")[0] != "0.23.0":
        raise RuntimeError("This experimental hook is pinned to vLLM 0.23.0")

    original = FlashAttentionImpl.do_kv_cache_update
    installed_mode = getattr(original, "_ail_append_mode", None)
    if installed_mode is not None:
        if installed_mode != mode:
            raise RuntimeError("Use a fresh process when changing Append mode")
        return

    root = Path(os.environ["AGENT_INFER_LAB_ROOT"]).resolve()
    adapter_path = root / "cuda" / "vllm_append_adapter.py"
    audit_dir = Path(os.environ["AIL_APPEND_AUDIT_DIR"]).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"append_{mode}_{os.getpid()}.jsonl"
    # Refuse an accidental restart that would overwrite an earlier process log.
    audit_path.touch(exist_ok=False)
    stats = {"calls": 0, "empty_calls": 0, "verified_calls": 0, "verified_valid_tokens": 0}
    extension = None

    def record(event: str, **fields) -> None:
        row = {
            "event": event,
            "mode": mode,
            "pid": os.getpid(),
            "time_unix": time.time(),
            **fields,
        }
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    record(
        "registered",
        vllm_version=vllm.__version__,
        adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        plugin_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hook="FlashAttentionImpl.do_kv_cache_update",
    )
    atexit.register(lambda: record("process_exit", **stats))

    @wraps(original)
    def update(self, layer, key, value, kv_cache, slot_mapping):
        nonlocal extension
        if self.attn_type != AttentionType.DECODER:
            raise RuntimeError("Append experiment only supports decoder attention")
        if self.kv_cache_dtype not in {"auto", "float16"}:
            raise RuntimeError("Quantized/BF16 KV caches are outside this experiment")
        if key.dtype != torch.float16 or value.dtype != torch.float16:
            raise RuntimeError("Append experiment requires FP16 inputs")
        if kv_cache.dtype != torch.float16 or kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
            raise RuntimeError("Expected FP16 cache [blocks, 2, block_size, heads, dim]")
        if not key.is_cuda or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Append experiment requires eager CUDA execution")
        if slot_mapping.ndim != 1 or slot_mapping.dtype != torch.int64:
            raise RuntimeError("Expected one-dimensional int64 slot mapping")

        stats["calls"] += 1
        n = slot_mapping.numel()
        if n == 0:
            stats["empty_calls"] += 1
            return

        if extension is None:
            spec = importlib.util.spec_from_file_location("ail_append_adapter", adapter_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Cannot load the local Append adapter")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            extension = module.load_adapter()
            record(
                "first_nonempty_call",
                input_shape=list(key.shape),
                input_stride=list(key.stride()),
                cache_shape=list(kv_cache.shape),
                cache_stride=list(kv_cache.stride()),
                mapped_rows=n,
            )
            print(
                f"[AIL_APPEND] mode={mode}: custom Append invoked; audit={audit_path}", flush=True
            )

        shadow = None
        valid_slots = []
        if mode == "verify":
            # Synchronization/copies are deliberately verification-only.
            host_slots = slot_mapping.detach().cpu().tolist()
            valid_slots = [slot for slot in host_slots if slot >= 0]
            capacity = kv_cache.shape[0] * kv_cache.shape[2]
            if any(slot >= capacity for slot in valid_slots):
                raise RuntimeError("Nonnegative slot exceeds cache capacity")
            if len(valid_slots) != len(set(valid_slots)):
                raise RuntimeError("Repeated write slots are not supported")
            if n > key.shape[0] or n > value.shape[0]:
                raise RuntimeError("Mapped rows exceed input rows")
            # Preserve real block strides, including any inter-layer gaps.
            shadow = torch.empty_strided(
                kv_cache.shape, kv_cache.stride(), dtype=kv_cache.dtype, device=kv_cache.device
            )
            shadow.copy_(kv_cache)
            original(self, layer, key, value, shadow, slot_mapping)
            torch.cuda.current_stream(key.device).synchronize()

        kc, vc = kv_cache.unbind(1)
        extension.append_fp16_(key, value, kc, vc, slot_mapping)

        if mode == "verify":
            torch.cuda.current_stream(key.device).synchronize()
            # Compare raw FP16 bits: untouched uninitialized cache can contain NaNs.
            if not torch.equal(kv_cache.view(torch.int16), shadow.view(torch.int16)):
                record("verification_failed", call=stats["calls"], mapped_rows=n)
                raise RuntimeError("Native/custom KV cache bitwise mismatch")
            stats["verified_calls"] += 1
            stats["verified_valid_tokens"] += len(valid_slots)
            record("verified", call=stats["calls"], valid_tokens=len(valid_slots), mapped_rows=n)
        elif stats["calls"] % 256 == 0:
            record("custom_progress", **stats)

    update._ail_append_mode = mode
    FlashAttentionImpl.do_kv_cache_update = update
