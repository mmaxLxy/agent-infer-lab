"""Archive environment, code, model/data hashes, and all AWQ artifacts."""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def cmd(command, cwd=None):
    r = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=180)
    return {"command": command, "exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline-model", type=Path, required=True)
    p.add_argument("--awq-model", type=Path, required=True)
    args = p.parse_args()
    out = args.root / "archive"
    out.mkdir(exist_ok=False)
    here = Path(__file__).parent
    packages = {}
    for name in (
        "vllm",
        "torch",
        "transformers",
        "safetensors",
        "compressed-tensors",
        "datasets",
        "lm_eval",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    environment = {
        "python": sys.executable,
        "python_version": sys.version,
        "packages": packages,
        "pip_check": cmd([sys.executable, "-m", "pip", "check"]),
        "pip_freeze": cmd([sys.executable, "-m", "pip", "freeze"]),
        "git_head": cmd(["git", "rev-parse", "HEAD"], args.repo),
        "git_status": cmd(["git", "status", "--short"], args.repo),
        "gpu_after_experiments": cmd(["nvidia-smi"]),
        "cuda": cmd(["/usr/local/cuda-12.9/bin/nvcc", "--version"]),
    }
    save(out / "environment.json", environment)
    checks = {
        "protocol": cmd(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(here),
                "-p",
                "test_protocol.py",
                "-v",
            ]
        ),
        "runtime_cpu": cmd([sys.executable, str(here / "runtime_cpu_checks.py")]),
    }
    save(out / "test-results.json", checks)
    if environment["pip_check"]["exit_code"] or any(x["exit_code"] for x in checks.values()):
        raise RuntimeError("Checks failed")
    models = {}
    for kind, root in (("fp16", args.baseline_model), ("awq", args.awq_model)):
        models[kind] = {
            p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(root.iterdir())
            if p.is_file()
        }
    save(out / "model-files.json", models)
    names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    tokenizer = {name: {kind: models[kind].get(name) for kind in ("fp16", "awq")} for name in names}
    tokenizer["all_available_sha256_identical"] = all(
        x["fp16"] and x["awq"] and x["fp16"]["sha256"] == x["awq"]["sha256"]
        for x in tokenizer.values()
        if isinstance(x, dict)
    )
    save(out / "tokenizer-equivalence.json", tokenizer)
    sources = []
    for sub in ("src", "scripts/awq_closed_loop"):
        sources += [
            x
            for x in (args.repo / sub).rglob("*")
            if x.is_file() and x.suffix in (".py", ".json", ".md")
        ]
    sources += [x for x in (args.repo / "pyproject.toml", args.repo / "README.md") if x.is_file()]
    hashes = {}
    with zipfile.ZipFile(out / "source-snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(sources)):
            rel = path.relative_to(args.repo).as_posix()
            z.write(path, rel)
            hashes[rel] = sha(path)
        base = Path(distribution("vllm").locate_file("vllm"))
        for rel in (
            "model_executor/layers/quantization/awq.py",
            "model_executor/layers/quantization/awq_marlin.py",
            "v1/sample/sampler.py",
            "v1/sample/logits_processor/interface.py",
        ):
            path = base / rel
            if path.is_file():
                z.write(path, "installed-vllm/" + rel)
                hashes["installed-vllm/" + rel] = sha(path)
    save(out / "source-sha256.json", hashes)
    prior = args.root.parent / "2026-09-04-fp8-closed-loop" / "data"
    comparison = []
    for path in sorted((args.root / "data").iterdir()):
        if path.is_file():
            comparison.append(
                {
                    "file": path.name,
                    "current": sha(path),
                    "prior": sha(prior / path.name),
                    "identical": sha(path) == sha(prior / path.name),
                }
            )
    save(out / "data-equivalence.json", comparison)
    if not all(x["identical"] for x in comparison):
        raise RuntimeError("Frozen data changed")
    target = args.root / "SHA256SUMS.txt"
    artifacts = sorted(x for x in args.root.rglob("*") if x.is_file() and x != target)
    with target.open("x", encoding="utf-8") as f:
        for path in artifacts:
            f.write(sha(path) + "  " + path.relative_to(args.root).as_posix() + "\n")
    print(json.dumps({"artifacts": len(artifacts), "sources": len(hashes), "pip_check": 0}))


if __name__ == "__main__":
    main()
