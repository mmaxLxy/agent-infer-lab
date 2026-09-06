"""Fresh paired FP16/AWQ performance and GSM8K generation-quality suite."""

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SERVER_PY = "/home/ayax/.venvs/agent-infer-lab/bin/python"
CLIENT_PY = "/home/ayax/.venvs/agent-infer-lab-dev/bin/python"
NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?"


def save(path, value):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(encoded + "\n")


def numeric(value):
    try:
        return str(Decimal(value.replace(",", "")).normalize())
    except (InvalidOperation, AttributeError):
        return None


def extract(text, strict=True):
    found = re.findall(r"####\s*(" + NUMBER + ")", text)
    if not found and not strict:
        found = re.findall(NUMBER, text)
    return numeric(found[-1]) if found else None


def http(port, path, body=None, timeout=300):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
    return json.loads(data) if data else None


def stop(process):
    if process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 10)):
        if process.poll() is not None:
            break
        os.killpg(process.pid, sig)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=timeout)


def new_session(root, stem):
    for attempt in range(1, 100):
        path = root / (stem + "-attempt" + str(attempt))
        if not path.exists():
            path.mkdir()
            return path
    raise RuntimeError("Too many attempts")


def run(args, kind, stem, quality=False, smoke=False):
    out = new_session(args.output, stem)
    model = args.awq_model if kind == "awq" else args.baseline_model
    cases = json.loads((args.output / "data/cases.json").read_text())
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(args.repo / "src"),
            "CUDA_HOME": "/usr/local/cuda-12.9",
            "AIL_APPEND_MODE": "native",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
        }
    )
    env["PATH"] = "/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    command = [
        SERVER_PY,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model),
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "half",
        "--kv-cache-dtype",
        "auto",
        "--attention-backend",
        "FLASHINFER",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.65",
        "--max-num-seqs",
        "16",
        "--max-num-batched-tokens",
        "2048",
        "--block-size",
        "16",
        "--seed",
        "20260905",
        "--enforce-eager",
        "--no-enable-prefix-caching",
    ]
    if kind == "awq":
        command += ["--quantization", "awq_marlin"]
    save(out / "launch.json", {"kind": kind, "model": str(model), "command": command})
    try:
        http(args.port, "/health", timeout=2)
    except OSError:
        pass
    else:
        raise RuntimeError("Experiment port occupied")
    process = None
    telemetry = None
    gpu_stream = None
    log = out / "server.log"
    print("START " + out.name, flush=True)
    with log.open("x") as stream:
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            started = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Server exited; inspect " + str(log))
                try:
                    http(args.port, "/health", timeout=2)
                    break
                except OSError:
                    if time.monotonic() - started > 600:
                        raise TimeoutError("Server startup timed out") from None
                    time.sleep(1)

            def complete(tokens, max_tokens=64, ignore_eos=True):
                return http(
                    args.port,
                    "/v1/completions",
                    {
                        "model": MODEL_NAME,
                        "prompt": tokens,
                        "max_tokens": max_tokens,
                        "temperature": 0,
                        "stream": False,
                        "ignore_eos": ignore_eos,
                    },
                )

            save(out / "initialization-output.json", complete(cases["initialization_tokens"]))

            def bench(name, requests):
                cmd = [
                    CLIENT_PY,
                    "-m",
                    "agent_infer_lab.benchmark",
                    "--base-url",
                    "http://127.0.0.1:" + str(args.port),
                    "--model",
                    MODEL_NAME,
                    "--requests",
                    str(requests),
                    "--concurrency",
                    "4",
                    "--input-tokens",
                    "1536",
                    "--output-tokens",
                    "64",
                    "--shared-prefix-ratio",
                    "0.75",
                    "--seed",
                    "20260903",
                    "--timeout",
                    "180",
                    "--output",
                    str(out / (name + ".json")),
                ]
                result = subprocess.run(
                    cmd, cwd=args.repo, env=env, capture_output=True, text=True, timeout=900
                )
                (out / (name + "-client.log")).write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError("Benchmark failed")
                payload = json.loads((out / (name + ".json")).read_text())
                if (
                    payload["result"]["successful_requests"] != requests
                    or payload["result"]["failed_requests"]
                ):
                    raise RuntimeError("Benchmark request failure")
                return payload

            bench("warmup", 4 if smoke else 32)
            if quality:
                rows = []
                with (out / "gsm8k.jsonl").open("x") as target:

                    def evaluate(case):
                        try:
                            response = complete(case["prompt_token_ids"], 512, False)
                            text = response["choices"][0]["text"]
                            gold = numeric(case["gold"])
                            return {
                                "test_index": case["test_index"],
                                "gold": gold,
                                "strict_answer": extract(text),
                                "flexible_answer": extract(text, False),
                                "strict_correct": extract(text) == gold,
                                "flexible_correct": extract(text, False) == gold,
                                "response": response,
                                "error": None,
                            }
                        except Exception as exc:
                            return {
                                "test_index": case["test_index"],
                                "gold": numeric(case["gold"]),
                                "strict_correct": False,
                                "flexible_correct": False,
                                "error": repr(exc),
                            }

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                        for offset in range(0, 200, 4):
                            batch = list(pool.map(evaluate, cases["cases"][offset : offset + 4]))
                            for row in batch:
                                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                            target.flush()
                            rows.extend(batch)
                            print(
                                "QUALITY {} {}/200 correct={}".format(
                                    kind, len(rows), sum(r["flexible_correct"] for r in rows)
                                ),
                                flush=True,
                            )
                summary = {
                    "kind": kind,
                    "total": len(rows),
                    "success": sum(r["error"] is None for r in rows),
                    "strict_correct": sum(r["strict_correct"] for r in rows),
                    "flexible_correct": sum(r["flexible_correct"] for r in rows),
                    "length_truncated": sum(
                        r.get("response", {}).get("choices", [{}])[0].get("finish_reason")
                        == "length"
                        for r in rows
                    ),
                }
            else:
                gpu_stream = (out / "gpu-during-measurement.csv").open("x")
                telemetry = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,name,memory.used,utilization.gpu,temperature.gpu,pstate,clocks.current.graphics,clocks.current.memory,power.draw",
                        "--format=csv",
                        "-l",
                        "1",
                    ],
                    stdout=gpu_stream,
                    stderr=subprocess.STDOUT,
                )
                start_line = len(log.read_text(errors="replace").splitlines())
                measured = bench("measurement", 16 if smoke else 160)
                end_line = len(log.read_text(errors="replace").splitlines())
                telemetry.terminate()
                telemetry.wait(timeout=10)
                telemetry = None
                gpu_stream.close()
                gpu_stream = None
                window = log.read_text(errors="replace").splitlines()[start_line:end_line]
                (out / "measurement-server-window.log").write_text("\n".join(window) + "\n")
                bad = [
                    line
                    for line in window
                    if re.search(
                        r"JIT compilation during inference|Traceback|CUDA out of memory|ERROR", line
                    )
                ]
                if bad:
                    raise RuntimeError("Contaminated measurement: " + repr(bad))
                summary = {
                    "kind": kind,
                    "result": measured["result"]["metrics"],
                    "success": measured["result"]["successful_requests"],
                    "measurement_window": {
                        "start_line_exclusive": start_line,
                        "end_line_inclusive": end_line,
                        "errors": bad,
                    },
                }
                print(
                    "PERF {} throughput={:.2f}".format(
                        kind, summary["result"]["output_throughput_tokens_per_second"]
                    ),
                    flush=True,
                )
            save(out / "complete.json", summary)
            return {"path": str(out), "summary": summary}
        except Exception as exc:
            save(out / "failure.json", {"error": repr(exc)})
            raise
        finally:
            if telemetry is not None:
                telemetry.terminate()
                telemetry.wait(timeout=10)
            if gpu_stream is not None:
                gpu_stream.close()
            if process is not None:
                stop(process)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--baseline-model", type=Path, required=True)
    p.add_argument("--awq-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8013)
    p.add_argument("--phase", choices=("smoke", "performance", "quality"), required=True)
    args = p.parse_args()
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    if args.phase == "smoke":
        print(json.dumps(run(args, "awq", "smoke-awq", smoke=True)["summary"], indent=2))
        return
    order = (
        [(i, k) for i in range(1, 4) for k in (("fp16", "awq") if i % 2 else ("awq", "fp16"))]
        if args.phase == "performance"
        else [(1, "fp16"), (1, "awq")]
    )
    for repeat, kind in order:
        stem = f"{args.phase}-r{repeat}-{kind}"
        if list(args.output.glob(stem + "-attempt*/complete.json")):
            print("SKIP " + stem, flush=True)
            continue
        item = run(args, kind, stem, quality=args.phase == "quality")
        with (args.output / "session-index.jsonl").open("a") as f:
            f.write(json.dumps(item) + "\n")
    print("PHASE_COMPLETE " + args.phase, flush=True)


if __name__ == "__main__":
    main()
