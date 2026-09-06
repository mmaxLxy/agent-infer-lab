"""Download immutable GSM8K source, freeze samples and tokenizer outputs."""

import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path


def save(path, obj):
    encoded = json.dumps(obj, ensure_ascii=False, indent=2)
    with path.open("x", encoding="utf-8") as f:
        f.write(encoded + "\n")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "AgentInferLab-reproducibility-audit"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    commit = (
        json.loads((args.source_dir / "manifest.json").read_text())["commit"]
        if args.source_dir
        else json.loads(
            fetch("https://api.github.com/repos/openai/grade-school-math/commits/master")
        )["sha"]
    )
    manifest = {
        "repository": "https://github.com/openai/grade-school-math",
        "commit": commit,
        "selection_seed": 20260904,
        "sources": {},
    }
    data = {}
    for name in ("train.jsonl", "test.jsonl", "LICENSE"):
        tail = name if name == "LICENSE" else "grade_school_math/data/" + name
        url = "https://raw.githubusercontent.com/openai/grade-school-math/" + commit + "/" + tail
        raw = (args.source_dir / name).read_bytes() if args.source_dir else fetch(url)
        with (args.out / name).open("xb") as f:
            f.write(raw)
        manifest["sources"][name] = {"url": url, "sha256": hashlib.sha256(raw).hexdigest()}
        if name.endswith("jsonl"):
            data[name] = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rng = random.Random(20260904)
    train_ids = rng.sample(range(len(data["train.jsonl"])), 16)
    test_ids = rng.sample(range(len(data["test.jsonl"])), 200)

    def clean(text):
        return re.sub(r"<<.*?>>", "", text)

    init_text = "\n\n".join(
        data["train.jsonl"][i]["question"] + "\n" + clean(data["train.jsonl"][i]["answer"])
        for i in train_ids
    )
    init_ids = tokenizer.encode(init_text, add_special_tokens=False)
    if len(init_ids) < 1536:
        raise RuntimeError("Initialization corpus unexpectedly short")
    instruction = "Solve the math problem step by step. End your answer with #### followed by the final numerical answer."
    cases = []
    for index in test_ids:
        row = data["test.jsonl"][index]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": row["question"] + "\n\n" + instruction},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(prompt_ids, "keys"):
            prompt_ids = prompt_ids["input_ids"]
        prompt_ids = list(prompt_ids)
        if not all(isinstance(t, int) for t in prompt_ids):
            raise TypeError("Expected one flat token-ID sequence")
        target = tokenizer.encode(clean(row["answer"]), add_special_tokens=False)[:128]
        if len(prompt_ids) + 512 > 2048 or len(target) < 2:
            raise RuntimeError("Case exceeds fixed context or target is empty")
        cases.append(
            {
                "test_index": index,
                "question": row["question"],
                "answer": row["answer"],
                "gold": row["answer"].split("####")[-1].strip(),
                "prompt_token_ids": prompt_ids,
                "target_token_ids": target,
            }
        )
    manifest.update(
        {
            "train_count": len(data["train.jsonl"]),
            "test_count": len(data["test.jsonl"]),
            "initialization_train_indices": train_ids,
            "test_indices": test_ids,
            "generation_cases": 200,
            "conditional_ppl_cases": 64,
            "initialization": "One fixed 1536-token natural-language training prompt; one-shot absolute-max scales, not an optimized corpus calibration",
            "generation_protocol": "zero-shot chat, temperature=0, max_tokens=512; strict #### numeric EM primary, final-number fallback secondary",
            "ppl_protocol": "first 64 selected cases, first <=128 gold-solution tokens; force continuation, score raw pre-processor logprobs; report full and decode-only (exclude token 1) conditional PPL; not WikiText PPL",
            "instruction": instruction,
        }
    )
    save(args.out / "manifest.json", manifest)
    save(args.out / "cases.json", {"initialization_tokens": init_ids[:1536], "cases": cases})
    print(
        json.dumps(
            {
                "commit": commit,
                "cases": len(cases),
                "max_prompt_length": max(len(c["prompt_token_ids"]) for c in cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
