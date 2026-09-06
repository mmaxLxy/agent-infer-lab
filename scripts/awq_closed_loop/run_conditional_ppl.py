"""Teacher-forced GSM8K reference-solution conditional PPL for FP16/AWQ."""

import argparse
import json
import math
import os
from pathlib import Path


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-model", required=True)
    p.add_argument("--awq-model", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--kind", choices=("fp16", "awq"), required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int, default=64)
    args = p.parse_args()
    out = args.root / args.name
    out.mkdir(exist_ok=False)
    os.environ.update(
        {
            "AIL_APPEND_MODE": "native",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
        }
    )
    from vllm import LLM, SamplingParams

    model = args.awq_model if args.kind == "awq" else args.baseline_model
    config = dict(
        model=model,
        dtype="half",
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        max_model_len=2048,
        gpu_memory_utilization=0.65,
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        block_size=16,
        seed=20260905,
        enforce_eager=True,
        enable_prefix_caching=False,
        logits_processors=["awq_runtime:GoldContinuationProcessor"],
        logprobs_mode="raw_logprobs",
        async_scheduling=False,
    )
    if args.kind == "awq":
        config["quantization"] = "awq_marlin"
    save(
        out / "protocol.json",
        {
            "kind": args.kind,
            "engine": config,
            "limit": args.limit,
            "metric": "GSM8K gold-solution conditional PPL; raw pre-processor logprobs",
            "decode_metric": "Excludes each case's first target token",
            "performance_measurement": False,
        },
    )
    data = json.loads((args.root / "data/cases.json").read_text())
    llm = LLM(**config)
    llm.generate(
        [{"prompt_token_ids": data["initialization_tokens"]}],
        SamplingParams(temperature=0, max_tokens=64, ignore_eos=True),
        use_tqdm=False,
    )
    first = data["cases"][0]
    token = first["target_token_ids"][0]
    prompt = {"prompt_token_ids": first["prompt_token_ids"]}
    natural = llm.generate(
        [prompt],
        SamplingParams(temperature=0, max_tokens=1, logprob_token_ids=[token]),
        use_tqdm=False,
    )[0].outputs[0]
    forced = llm.generate(
        [prompt],
        SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=0,
            ignore_eos=True,
            extra_args={"gold_tokens": [token]},
        ),
        use_tqdm=False,
    )[0].outputs[0]
    natural_lp = natural.logprobs[0][token].logprob
    forced_lp = forced.logprobs[0][token].logprob
    validation = {
        "target": token,
        "natural_logprob": natural_lp,
        "forced_logprob": forced_lp,
        "difference": forced_lp - natural_lp,
        "forced_token_ids": list(forced.token_ids),
    }
    save(out / "raw-logprob-validation.json", validation)
    if list(forced.token_ids) != [token] or abs(natural_lp - forced_lp) > 1e-4 or not forced_lp < 0:
        raise RuntimeError("Raw logprob validation failed")
    print("RAW_LOGPROB_VALIDATION_PASS", flush=True)
    rows = []
    with (out / "conditional-nll.jsonl").open("x") as f:
        selected = data["cases"][: args.limit]
        for offset in range(0, len(selected), 4):
            batch = selected[offset : offset + 4]
            prompts = [{"prompt_token_ids": c["prompt_token_ids"]} for c in batch]
            params = [
                SamplingParams(
                    temperature=0,
                    max_tokens=len(c["target_token_ids"]),
                    ignore_eos=True,
                    logprobs=0,
                    extra_args={"gold_tokens": c["target_token_ids"]},
                )
                for c in batch
            ]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for case, result in zip(batch, outputs, strict=True):
                generated = result.outputs[0]
                targets = case["target_token_ids"]
                if list(generated.token_ids) != targets:
                    raise RuntimeError("Teacher-forced token mismatch")
                lps = [
                    step[token_id].logprob
                    for step, token_id in zip(generated.logprobs, targets, strict=True)
                ]
                if not all(math.isfinite(v) and v <= 1e-6 for v in lps):
                    raise RuntimeError("Invalid raw logprob")
                row = {
                    "test_index": case["test_index"],
                    "token_ids": targets,
                    "raw_logprobs": lps,
                    "nll": -sum(lps),
                    "decode_nll": -sum(lps[1:]),
                    "tokens": len(lps),
                    "decode_tokens": len(lps) - 1,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
            print(f"PPL {args.kind} {len(rows)}/{args.limit}", flush=True)
    tokens = sum(r["tokens"] for r in rows)
    decode_tokens = sum(r["decode_tokens"] for r in rows)
    nll = sum(r["nll"] for r in rows)
    decode_nll = sum(r["decode_nll"] for r in rows)
    summary = {
        "kind": args.kind,
        "cases": len(rows),
        "tokens": tokens,
        "decode_tokens": decode_tokens,
        "nll": nll,
        "decode_nll": decode_nll,
        "conditional_ppl": math.exp(nll / tokens),
        "decode_conditional_ppl": math.exp(decode_nll / decode_tokens),
        "raw_logprob_validation": "passed",
    }
    save(out / "complete.json", summary)
    print("PPL_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
