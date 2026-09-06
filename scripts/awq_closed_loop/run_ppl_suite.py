"""Sequential PPL smoke and formal runs."""

import argparse
import os
import subprocess
from pathlib import Path

from run_suite import SERVER_PY, stop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-model", required=True)
    p.add_argument("--awq-model", required=True)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    env = dict(os.environ)
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["PATH"] = "/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = str(Path(__file__).parent.resolve())
    for name, kind, count in (
        ("ppl-awq-smoke", "awq", 4),
        ("ppl-fp16-final", "fp16", 64),
        ("ppl-awq-final", "awq", 64),
    ):
        if (args.root / name / "complete.json").exists():
            print("SKIP " + name, flush=True)
            continue
        if (args.root / name).exists():
            raise RuntimeError("Incomplete result is preserved: " + name)
        cmd = [
            SERVER_PY,
            str(Path(__file__).with_name("run_conditional_ppl.py")),
            "--baseline-model",
            args.baseline_model,
            "--awq-model",
            args.awq_model,
            "--root",
            str(args.root),
            "--kind",
            kind,
            "--name",
            name,
            "--limit",
            str(count),
        ]
        print("START " + name, flush=True)
        with (args.root / (name + "-process.log")).open("x") as stream:
            process = subprocess.Popen(
                cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                process.wait(timeout=900)
                if process.returncode or not (args.root / name / "complete.json").exists():
                    raise RuntimeError(name + " failed")
                print("DONE " + name, flush=True)
            finally:
                stop(process)
    print("ALL_PPL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
