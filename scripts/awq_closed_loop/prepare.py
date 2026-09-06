"""Freeze AWQ experiment metadata and inventory model tensors."""

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

from safetensors import safe_open

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def inventory(root):
    files = sorted(root.glob("*.safetensors"))
    categories = {}
    dtypes = {}
    tensors = []
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for name in f:
                item = f.get_slice(name)
                shape = list(item.get_shape())
                dtype = item.get_dtype()
                size = math.prod(shape) * DTYPE_BYTES[dtype]
                if any(part in name for part in ("qweight", "qzeros", "scales", "g_idx")):
                    category = "awq_quantized_linear_storage"
                elif "embed_tokens" in name or "lm_head" in name:
                    category = "embedding_or_lm_head"
                else:
                    category = "other"
                categories[category] = categories.get(category, 0) + size
                dtypes[dtype] = dtypes.get(dtype, 0) + size
                tensors.append(
                    {
                        "name": name,
                        "shape": shape,
                        "dtype": dtype,
                        "bytes": size,
                        "category": category,
                    }
                )
    return {
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in files],
        "tensor_bytes_by_category": categories,
        "tensor_bytes_by_dtype": dtypes,
        "tensor_bytes_total": sum(x["bytes"] for x in tensors),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline-model", type=Path, required=True)
    p.add_argument("--awq-model", type=Path, required=True)
    args = p.parse_args()
    base, awq = inventory(args.baseline_model), inventory(args.awq_model)
    save(
        args.root / "model-inventory.json",
        {
            "fp16": base,
            "awq": awq,
            "weight_file_reduction_percent": (
                1 - awq["files"][0]["bytes"] / base["files"][0]["bytes"]
            )
            * 100,
        },
    )
    data_hashes = {p.name: sha(p) for p in sorted((args.root / "data").iterdir()) if p.is_file()}
    awq_config = json.loads((args.awq_model / "config.json").read_text())
    save(
        args.root / "experiment-manifest.json",
        {
            "date": "2026-09-05",
            "official_model": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            "official_commit": "7c05280e9583b6cace2dc9d8950e21bfd3d72c19",
            "download_endpoint": "https://hf-mirror.com (official commit verified by X-Repo-Commit)",
            "baseline_model": str(args.baseline_model),
            "awq_model": str(args.awq_model),
            "quantization_config": awq_config["quantization_config"],
            "data_sha256": data_hashes,
            "versions": {
                name: version(name) for name in ("vllm", "torch", "transformers", "safetensors")
            },
            "performance": "3 paired rounds; A-B/B-A/A-B; 32 warmup + 160 measured; C4, input1536, output64, prefix cache off",
            "quality": "Same frozen 200 GSM8K samples and same extraction as FP8 experiment",
            "conditional_ppl": "Same first 64 frozen samples; raw pre-processor teacher-forced logprobs",
            "awq_runtime_backend": "awq_marlin (optimized W4A16; ordinary awq smoke excluded)",
            "claim_policy": "Report measured values; do not reuse expected 65%/12%/8% claims",
        },
    )
    print(
        json.dumps(
            {
                "fp16_bytes": base["files"][0]["bytes"],
                "awq_bytes": awq["files"][0]["bytes"],
                "reduction_percent": (1 - awq["files"][0]["bytes"] / base["files"][0]["bytes"])
                * 100,
                "awq_categories": awq["tensor_bytes_by_category"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
