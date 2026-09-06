"""Fixed-scale, loopback-only FP16/FP8 performance and generation-quality suite."""

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
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


def save(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def numeric(text):
    try:
        return str(Decimal(text.replace(",", "")).normalize())
    except (InvalidOperation, AttributeError):
        return None


def extract(text, strict=True):
    found = re.findall(r"####\s*(" + NUMBER + ")", text)
    if not found and not strict:
        found = re.findall(NUMBER, text)
    return numeric(found[-1]) if found else None


def fingerprint(snapshot):
    value = [
        {"layer": r["layer"], **{k: r[k] for k in ("_q_scale", "_k_scale", "_v_scale")}}
        for r in sorted(snapshot["rows"], key=lambda row: row["layer"])
    ]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_snapshot(snapshot):
    if len(snapshot["rows"]) != 24:
        raise RuntimeError("Expected 24 attention layers")
    for row in snapshot["rows"]:
        for k in ("_q_scale", "_k_scale", "_v_scale"):
            if len(row[k]) != 1 or not math.isfinite(row[k][0]) or row[k][0] <= 0:
                raise RuntimeError("Invalid scale in " + row["layer"])


def http(port, path, body=None, timeout=300):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read()
    return json.loads(content) if content else None


def rpc(port, method, args=None):
    answer = http(port, "/collective_rpc", {"method": method, "args": args or [], "timeout": 60})[
        "results"
    ]
    if len(answer) != 1:
        raise RuntimeError("Expected one worker")
    validate_snapshot(answer[0])
    return answer[0]


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
    for i in range(1, 100):
        out = root / (stem + "-attempt" + str(i))
        if not out.exists():
            out.mkdir()
            return out
    raise RuntimeError("Too many attempts")


def complete(port, tokens, max_tokens=64, ignore_eos=True):
    return http(
        port,
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


def run(args, kind, stem, quality=False):
    out = new_session(args.output, stem)
    log = out / "server.log"
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(args.repo / "src") + ":" + str(Path(__file__).parent.resolve()),
            "CUDA_HOME": "/usr/local/cuda-12.9",
            "AIL_APPEND_MODE": "native",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
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
        str(args.model),
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "half",
        "--kv-cache-dtype",
        "fp8" if kind == "fp8" else "auto",
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
        "20260903",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--worker-extension-cls",
        "fp8_runtime.ScaleWorkerExtension",
    ]
    if kind == "fp8":
        command.append("--calculate-kv-scales")
    save(
        out / "launch.json",
        {
            "command": command,
            "kind": kind,
            "extra_environment": {
                k: env[k] for k in ("AIL_APPEND_MODE", "PYTHONPATH", "VLLM_SERVER_DEV_MODE")
            },
        },
    )
    cases = json.loads((args.output / "data/cases.json").read_text())
    canonical_path = args.output / "canonical-fp8-scales.json"
    try:
        http(args.port, "/health", timeout=2)
    except OSError:
        pass
    else:
        raise RuntimeError("Experiment port occupied")
    telemetry = None
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
            start = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Server exited: " + str(log))
                try:
                    http(args.port, "/health", timeout=2)
                    break
                except OSError:
                    if time.monotonic() - start > 600:
                        raise TimeoutError("Server startup timed out") from None
                    time.sleep(1)
            initial = rpc(args.port, "ail_snapshot")
            save(out / "scales-after-init.json", initial)
            print(
                "INIT {} pending={} first_k={}".format(
                    kind,
                    sum(r["calculate"] for r in initial["rows"]),
                    initial["rows"][0]["_k_scale"],
                ),
                flush=True,
            )
            if kind == "fp8" and canonical_path.exists():
                rpc(args.port, "ail_load_scales", [canonical_path.read_text()])
            elif kind == "fp8":
                rpc(args.port, "ail_arm_scales")
            init_response = complete(args.port, cases["initialization_tokens"])
            save(out / "initialization-output.json", init_response)
            ready = rpc(args.port, "ail_snapshot")
            save(out / "scales-after-initialization-input.json", ready)
            if any(r["calculate"] for r in ready["rows"]):
                raise RuntimeError("Scale calculation did not finish")
            if kind == "fp8" and not canonical_path.exists():
                save(canonical_path, ready)
            expected = fingerprint(ready)
            if kind == "fp8" and expected != fingerprint(json.loads(canonical_path.read_text())):
                raise RuntimeError("Canonical scale mismatch")

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
                    cmd, cwd=args.repo, env=env, capture_output=True, text=True, timeout=600
                )
                with (out / (name + "-client.log")).open("x") as f:
                    f.write(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError("Benchmark command failed")
                payload = json.loads((out / (name + ".json")).read_text())
                if payload["result"]["successful_requests"] != requests:
                    raise RuntimeError("Failed benchmark requests")
                return payload

            bench("warmup", 32)
            warm = rpc(args.port, "ail_snapshot")
            save(out / "scales-after-warmup.json", warm)
            if fingerprint(warm) != expected:
                raise RuntimeError("Scales changed during warmup")
            if quality:
                results = []
                with (out / "gsm8k.jsonl").open("x") as target:

                    def evaluate(case):
                        try:
                            response = complete(args.port, case["prompt_token_ids"], 512, False)
                            choice = response["choices"][0]
                            answer = extract(choice["text"])
                            flexible = extract(choice["text"], False)
                            return {
                                "test_index": case["test_index"],
                                "gold": numeric(case["gold"]),
                                "strict_answer": answer,
                                "flexible_answer": flexible,
                                "strict_correct": answer == numeric(case["gold"]),
                                "flexible_correct": flexible == numeric(case["gold"]),
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
                        for offset in range(0, len(cases["cases"]), 4):
                            batch = list(pool.map(evaluate, cases["cases"][offset : offset + 4]))
                            for row in batch:
                                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                            target.flush()
                            results.extend(batch)
                            print(
                                "QUALITY {} {}/200 correct={}".format(
                                    kind, len(results), sum(r["strict_correct"] for r in results)
                                ),
                                flush=True,
                            )
                summary = {
                    "kind": kind,
                    "total": len(results),
                    "success": sum(r["error"] is None for r in results),
                    "strict_correct": sum(r["strict_correct"] for r in results),
                    "flexible_correct": sum(r["flexible_correct"] for r in results),
                    "length_truncated": sum(
                        r.get("response", {}).get("choices", [{}])[0].get("finish_reason")
                        == "length"
                        for r in results
                    ),
                }
            else:
                # GPU sampling runs only around measurement, separately from server shutdown.
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
                measured = bench("measurement", 160)
                end_line = len(log.read_text(errors="replace").splitlines())
                telemetry.terminate()
                telemetry.wait(timeout=10)
                telemetry = None
                gpu_stream.close()
                window = log.read_text(errors="replace").splitlines()[start_line:end_line]
                with (out / "measurement-server-window.log").open("x") as f:
                    f.write("\n".join(window) + "\n")
                bad = [
                    line
                    for line in window
                    if re.search(
                        r"JIT compilation during inference|Traceback|CUDA out of memory|ERROR", line
                    )
                ]
                if bad:
                    raise RuntimeError("Measurement contaminated: " + repr(bad))
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
            final = rpc(args.port, "ail_snapshot")
            save(out / "scales-after-measurement.json", final)
            if fingerprint(final) != expected:
                raise RuntimeError("Scales changed during evaluation")
            summary["scale_sha256"] = expected
            save(out / "complete.json", summary)
            return {"path": str(out), "summary": summary}
        except Exception as exc:
            save(out / "failure.json", {"error": repr(exc)})
            raise
        finally:
            if telemetry is not None:
                telemetry.terminate()
                telemetry.wait(timeout=10)
            stop(process)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8013)
    p.add_argument("--phase", choices=("smoke", "performance", "quality"), required=True)
    args = p.parse_args()
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    if args.phase == "smoke":
        result = run(args, "fp8", "smoke-fp8")
        print(json.dumps(result["summary"], indent=2))
        return
    if args.phase == "performance":
        order = [
            (i, k) for i in range(1, 4) for k in (("fp16", "fp8") if i % 2 else ("fp8", "fp16"))
        ]
    else:
        order = [(1, "fp16"), (1, "fp8")]
    for repeat, kind in order:
        stem = f"{args.phase}-r{repeat}-{kind}"
        done = list(args.output.glob(stem + "-attempt*/complete.json"))
        if done:
            print("SKIP completed " + stem, flush=True)
            continue
        item = run(args, kind, stem, quality=args.phase == "quality")
        with (args.output / "session-index.jsonl").open("a") as f:
            f.write(json.dumps(item) + "\n")
    print("PHASE_COMPLETE " + args.phase, flush=True)


if __name__ == "__main__":
    main()
