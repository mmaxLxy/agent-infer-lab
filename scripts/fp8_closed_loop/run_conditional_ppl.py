"""Teacher-forced GSM8K reference-solution conditional perplexity.

Only separate quality engines load the forcing processor. Performance engines
never use it. Raw logprobs are computed by vLLM before processor modifications.
"""

import argparse
import json
import math
import os
from pathlib import Path

from run_suite import fingerprint, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--kind", choices=("fp16", "fp8"), required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int, default=64)
    args = p.parse_args()
    out = args.root / args.name
    out.mkdir(exist_ok=False)
    os.environ["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent) + ":" + os.environ.get("PYTHONPATH", "")
    )
    os.environ["AIL_APPEND_MODE"] = "native"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    from vllm import LLM, SamplingParams

    data = json.loads((args.root / "data/cases.json").read_text())
    config = dict(
        model=args.model,
        dtype="half",
        kv_cache_dtype="fp8" if args.kind == "fp8" else "auto",
        calculate_kv_scales=args.kind == "fp8",
        attention_backend="FLASHINFER",
        max_model_len=2048,
        gpu_memory_utilization=0.65,
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        block_size=16,
        seed=20260903,
        enforce_eager=True,
        enable_prefix_caching=False,
        worker_extension_cls="fp8_runtime.ScaleWorkerExtension",
        logits_processors=["fp8_runtime:GoldContinuationProcessor"],
        logprobs_mode="raw_logprobs",
        async_scheduling=False,
    )
    save(
        out / "protocol.json",
        {
            "engine": config,
            "metric": "GSM8K gold-solution conditional PPL, not standard WikiText PPL",
            "decode_metric": "Exclude first target token (prefill prediction); remaining tokens use KV-cache decode",
            "limit": args.limit,
            "max_target_tokens": 128,
            "note": "Synchronous scheduling is used only in this teacher-forced quality test; not a performance measurement",
        },
    )
    llm = LLM(**config)

    def snapshot():
        return llm.collective_rpc("ail_snapshot")[0]

    save(out / "scales-after-init.json", snapshot())
    if args.kind == "fp8":
        llm.collective_rpc(
            "ail_load_scales", args=((args.root / "canonical-fp8-scales.json").read_text(),)
        )
    llm.generate(
        [{"prompt_token_ids": data["initialization_tokens"]}],
        SamplingParams(temperature=0, max_tokens=64, ignore_eos=True),
        use_tqdm=False,
    )
    before = snapshot()
    save(out / "scales-after-warmup.json", before)

    # Independent check: one-token raw logprob with and without forcing must match.
    first = data["cases"][0]
    token = first["target_token_ids"][0]
    prompt = {"prompt_token_ids": first["prompt_token_ids"]}
    normal = llm.generate(
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
    natural_lp = normal.logprobs[0][token].logprob
    forced_lp = forced.logprobs[0][token].logprob
    save(
        out / "raw-logprob-validation.json",
        {
            "target": token,
            "natural_logprob": natural_lp,
            "forced_logprob": forced_lp,
            "difference": forced_lp - natural_lp,
            "forced_token_ids": list(forced.token_ids),
        },
    )
    if list(forced.token_ids) != [token] or abs(natural_lp - forced_lp) > 1e-4 or not forced_lp < 0:
        raise RuntimeError("Raw pre-processor logprob validation failed")
    print("RAW_LOGPROB_VALIDATION_PASS", flush=True)

    records = []
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
            results = llm.generate(prompts, params, use_tqdm=False)
            for case, result in zip(batch, results, strict=True):
                generated = result.outputs[0]
                targets = case["target_token_ids"]
                if list(generated.token_ids) != targets:
                    raise RuntimeError("Teacher forcing token mismatch")
                lps = [
                    step[token].logprob
                    for step, token in zip(generated.logprobs, targets, strict=True)
                ]
                if not all(math.isfinite(v) and v <= 1e-6 for v in lps):
                    raise RuntimeError("Invalid logprobs")
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
                records.append(row)
            print(f"PPL {args.kind} {len(records)}/{args.limit}", flush=True)
    after = snapshot()
    save(out / "scales-after-evaluation.json", after)
    if fingerprint(before) != fingerprint(after):
        raise RuntimeError("Scales changed during scoring")
    tokens = sum(r["tokens"] for r in records)
    dtokens = sum(r["decode_tokens"] for r in records)
    nll = sum(r["nll"] for r in records)
    dnll = sum(r["decode_nll"] for r in records)
    summary = {
        "kind": args.kind,
        "cases": len(records),
        "tokens": tokens,
        "decode_tokens": dtokens,
        "nll": nll,
        "decode_nll": dnll,
        "conditional_ppl": math.exp(nll / tokens),
        "decode_conditional_ppl": math.exp(dnll / dtokens),
        "scale_sha256": fingerprint(after),
        "raw_logprob_validation": "passed",
    }
    save(out / "complete.json", summary)
    print("PPL_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
