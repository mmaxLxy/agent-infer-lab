"""Read-only environment capture and exclusive-write reproducibility archive."""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

from run_suite import save


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
    return {
        "command": cmd,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    args = p.parse_args()
    archive = args.root / "archive"
    archive.mkdir(exist_ok=False)
    packages = {}
    for name in (
        "vllm",
        "torch",
        "transformers",
        "safetensors",
        "flashinfer-python",
        "compressed-tensors",
        "datasets",
        "lm_eval",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    env = {
        "python": sys.executable,
        "python_version": sys.version,
        "packages": packages,
        "pip_check": command([sys.executable, "-m", "pip", "check"]),
        "pip_freeze": command([sys.executable, "-m", "pip", "freeze"]),
        "git_head": command(["git", "rev-parse", "HEAD"], args.repo),
        "git_status": command(["git", "status", "--short"], args.repo),
        "gpu_after_experiments": command(["nvidia-smi"]),
        "cuda": command(["/usr/local/cuda-12.9/bin/nvcc", "--version"]),
    }
    save(archive / "environment.json", env)
    checks = {
        "protocol": command(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(Path(__file__).parent),
                "-p",
                "test_protocol.py",
                "-v",
            ]
        ),
        "runtime_cpu": command(
            [sys.executable, str(Path(__file__).with_name("runtime_cpu_checks.py"))]
        ),
    }
    save(archive / "test-results.json", checks)
    if any(v["exit_code"] for v in checks.values()) or env["pip_check"]["exit_code"]:
        raise RuntimeError("Environment or CPU checks failed")
    model_files = sorted(p for p in args.model.iterdir() if p.is_file())
    save(archive / "model-sha256.json", {p.name: sha(p) for p in model_files})
    sources = set()
    for sub in ("src", "scripts/fp8_closed_loop", "cuda"):
        sources.update(
            p
            for p in (args.repo / sub).rglob("*")
            if p.is_file() and p.suffix in (".py", ".md", ".cu", ".cpp", ".toml")
        )
    sources.update(
        p for p in (args.repo / "pyproject.toml", args.repo / "README.md") if p.is_file()
    )
    hashes = {}
    with zipfile.ZipFile(
        archive / "source-snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as z:
        for path in sorted(sources):
            rel = path.relative_to(args.repo).as_posix()
            z.write(path, rel)
            hashes[rel] = sha(path)
        base = Path(distribution("vllm").locate_file("vllm"))
        for rel in (
            "config/cache.py",
            "model_executor/layers/attention/attention.py",
            "v1/sample/sampler.py",
            "v1/sample/logits_processor/interface.py",
            "v1/worker/gpu_model_runner.py",
            "entrypoints/serve/dev/rpc/api_router.py",
        ):
            path = base / rel
            if path.is_file():
                name = "installed-vllm/" + rel
                z.write(path, name)
                hashes[name] = sha(path)
    save(archive / "source-sha256.json", hashes)
    oldroot = args.root.parent / "2026-09-03-quantization"
    oldfiles = sorted(p for p in oldroot.rglob("*") if p.is_file())
    prior_checks = []
    for manifest in oldfiles:
        if manifest.name not in ("SHA256SUMS.txt", "SMOKE_SHA256SUMS.txt"):
            continue
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(maxsplit=1)
            target_path = args.repo / relative.lstrip("*")
            if not target_path.is_file():
                target_path = manifest.parent / relative.lstrip("*")
            if not target_path.is_file():
                raise RuntimeError("Cannot resolve historical checksum path: " + relative)
            actual = sha(target_path)
            prior_checks.append({"file": str(target_path), "pass": actual == expected})
    save(archive / "prior-integrity-check.json", prior_checks)
    if not all(r["pass"] for r in prior_checks):
        raise RuntimeError("Prior artifact checksum mismatch")
    save(
        archive / "prior-artifacts-sha256.json",
        {p.relative_to(oldroot).as_posix(): sha(p) for p in oldfiles},
    )
    target = args.root / "SHA256SUMS.txt"
    artifacts = sorted(p for p in args.root.rglob("*") if p.is_file() and p != target)
    with target.open("x", encoding="utf-8") as f:
        for path in artifacts:
            f.write(sha(path) + "  " + path.relative_to(args.root).as_posix() + "\n")
    print(
        json.dumps(
            {
                "artifacts": len(artifacts),
                "sources": len(hashes),
                "pip_check": env["pip_check"]["exit_code"],
            }
        )
    )


if __name__ == "__main__":
    main()
