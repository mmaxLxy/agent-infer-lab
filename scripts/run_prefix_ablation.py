"""Run an isolated, recorded vanilla-vLLM Prefix Cache experiment (Linux)."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from http.client import HTTPConnection, HTTPException
from pathlib import Path

from agent_infer_lab.benchmark import build_result_payload, run_benchmark
from agent_infer_lab.controlled_inputs import (
    audit_inputs,
    common_prefix_length,
    prepare_controlled_requests,
)
from agent_infer_lab.metrics import summarize_metrics
from agent_infer_lab.result_storage import collect_run_metadata, write_json_result
from agent_infer_lab.vllm_client import VllmClient
from agent_infer_lab.workloads import WorkloadConfig, generate_workload

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def http(port: int, path: str, method: str = "GET") -> str:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        if response.status != 200:
            raise RuntimeError(f"{method} {path}: {response.status} {body[:300]}")
        return body
    finally:
        connection.close()


def counter(text: str, name: str) -> float:
    values = []
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            values.append(float(line.rsplit(" ", 1)[1]))
    return sum(values)


def command_output(command: list[str]) -> dict:
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def reset_cache(port: int, log: Path) -> dict:
    before = log.read_text(errors="replace").count("Successfully reset prefix cache")
    http(port, "/reset_prefix_cache", "POST")
    for _ in range(50):
        after = log.read_text(errors="replace").count("Successfully reset prefix cache")
        if after > before:
            return {"http_status": 200, "engine_log_confirmed": True}
        time.sleep(0.1)
    raise RuntimeError("HTTP reset returned but engine success was not confirmed")


def wait_metrics(port: int, successes: float) -> str:
    for _ in range(100):
        text = http(port, "/metrics")
        if counter(text, "vllm:request_success_total") >= successes:
            return text
        time.sleep(0.1)
    raise RuntimeError("Server metrics did not catch up to completed requests")


def validate_result(payload: dict) -> dict:
    result = payload["result"]
    from agent_infer_lab.metrics import RequestTrace

    traces = tuple(RequestTrace(**trace) for trace in result["traces"])
    expected = asdict(summarize_metrics(traces)) if traces else None
    if expected != result["metrics"]:
        raise RuntimeError("Saved aggregate does not match saved traces")
    requests = result["prepared_requests"]
    request_ids = {item["request_id"] for item in requests}
    outcome_ids = [item["request_id"] for item in (*result["traces"], *result["failures"])]
    if len(outcome_ids) != len(requests) or set(outcome_ids) != request_ids:
        raise RuntimeError("Input and outcome records do not match")
    return {
        "recomputed_metrics_match": True,
        "all_request_ids_accounted_for": True,
        "successful_requests": len(traces),
        "failed_requests": len(result["failures"]),
    }


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-python", default="/home/ayax/.venvs/agent-infer-lab/bin/python")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "CUDA_HOME": "/usr/local/cuda-12.9",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_LOG_STATS_INTERVAL": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
        }
    )
    env["PATH"] = "/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    metadata = collect_run_metadata([sys.executable, *sys.argv])
    write_json_result(out / "client_manifest.json", metadata)
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in metadata["source_sha256"]:
            archive.write(ROOT / relative, relative)
    model_files = {}
    for path in sorted(args.model_path.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                model_files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    runtime = subprocess.run(
        [
            args.server_python,
            "-c",
            "import json,torch,vllm; "
            "from agent_infer_lab.environment import collect_environment; "
            "print(json.dumps({'environment':collect_environment(),"
            "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
            "'vllm':vllm.__version__,'gpu':torch.cuda.get_device_name(0)}))",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    write_json_result(
        out / "server_environment.json",
        {
            "runtime": json.loads(runtime.stdout.splitlines()[-1]),
            "model_snapshot": str(args.model_path),
            "model_revision": args.model_path.name,
            "model_file_sha256": model_files,
            "pip_freeze": command_output([args.server_python, "-m", "pip", "freeze"]),
            "gpu_before": command_output(["nvidia-smi"]),
            "environment_overrides": {
                key: env[key]
                for key in (
                    "CUDA_HOME",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "VLLM_SERVER_DEV_MODE",
                    "VLLM_LOG_STATS_INTERVAL",
                    "VLLM_NO_USAGE_STATS",
                    "DO_NOT_TRACK",
                )
            },
        },
    )
    try:
        http(args.port, "/v1/models")
    except OSError:
        pass
    else:
        raise RuntimeError("Dedicated experiment port already in use")

    run_index = []
    prior_prefixes = []
    input_digests = {}
    for repeat in range(1, args.repeats + 1):
        states = [False, True] if repeat % 2 else [True, False]
        lengths = [512, 1024, 1536]
        lengths = lengths[repeat - 1 :] + lengths[: repeat - 1]
        for enabled in states:
            state = "on" if enabled else "off"
            session = out / f"server_r{repeat}_{state}"
            session.mkdir()
            log = session / "server.log"
            command = [
                args.server_python,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                str(args.model_path),
                "--served-model-name",
                MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--dtype",
                "half",
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
                "--enforce-eager",
                "--seed",
                "20260830",
                "--enable-prefix-caching" if enabled else "--no-enable-prefix-caching",
            ]
            write_json_result(
                session / "launch.json",
                {"command": command, "prefix_caching": enabled, "repeat": repeat},
            )
            with log.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    for tick in range(300):
                        if process.poll() is not None:
                            raise RuntimeError(f"Server exited; inspect {log}")
                        try:
                            models = http(args.port, "/v1/models")
                            break
                        except (OSError, HTTPException):
                            if tick % 15 == 0:
                                print(
                                    f"Waiting for server repeat={repeat} cache={state}", flush=True
                                )
                            time.sleep(1)
                    else:
                        raise TimeoutError("Server startup exceeded 300 seconds")
                    (session / "models.json").write_text(models, encoding="utf-8")
                    client = VllmClient(
                        f"http://127.0.0.1:{args.port}", MODEL, timeout=30, ignore_eos=True
                    )

                    if repeat == 1 and not enabled:
                        smoke_command = [
                            sys.executable,
                            "-m",
                            "agent_infer_lab.benchmark",
                            "--base-url",
                            client.base_url,
                            "--model",
                            MODEL,
                            "--requests",
                            "4",
                            "--concurrency",
                            "2",
                            "--input-tokens",
                            "128",
                            "--output-tokens",
                            "32",
                            "--output",
                            str(out / "smoke.json"),
                        ]
                        smoke = command_output(smoke_command)
                        write_json_result(out / "smoke_command.json", smoke)
                        if smoke["returncode"] != 0:
                            raise RuntimeError("CLI JSON smoke failed")
                        payload = json.loads((out / "smoke.json").read_text())
                        validation = validate_result(payload)
                        if validation["successful_requests"] != 4:
                            raise RuntimeError("CLI smoke did not complete all 4 requests")
                        for key in ("total_requests", "successful_requests", "failed_requests"):
                            if f"{key}: {payload['result'][key]}" not in smoke["stdout"]:
                                raise RuntimeError("CLI counts do not match JSON")
                        for key in ("duration_seconds", "output_throughput_tokens_per_second"):
                            value = payload["result"]["metrics"][key]
                            if f"{key}: {value:.6f}" not in smoke["stdout"]:
                                raise RuntimeError("CLI metrics do not match JSON")
                        write_json_result(out / "smoke_validation.json", validation)
                        print("Real 4-request CLI JSON smoke: PASS", flush=True)

                    for length in lengths:
                        name = f"cache_{state}_input{length}_repeat{repeat}"
                        config = WorkloadConfig(80, (length,), (64,), 0.75, 4, 20260830)
                        content_seed = 20260830 + repeat * 10000 + length
                        requests = prepare_controlled_requests(
                            generate_workload(config), client.tokenize, content_seed=content_seed
                        )
                        audit = audit_inputs(requests)
                        if (
                            audit["duplicate_prompt_count"] != 0
                            or audit["pairwise_common_prefix_min"] != length * 3 // 4
                            or audit["pairwise_common_prefix_max"] != length * 3 // 4
                        ):
                            raise RuntimeError("Input audit failed")
                        pair = (repeat, length)
                        if pair in input_digests and input_digests[pair] != audit["input_sha256"]:
                            raise RuntimeError("ON/OFF inputs differ")
                        input_digests[pair] = audit["input_sha256"]
                        warm_config = WorkloadConfig(8, (length,), (64,), 0.0, 4, 20260830)
                        warm = prepare_controlled_requests(
                            generate_workload(warm_config),
                            client.tokenize,
                            content_seed=content_seed + 1000000,
                        )
                        if any(
                            common_prefix_length(r.prompt_token_ids, w.prompt_token_ids) >= 16
                            for r in requests
                            for w in warm
                        ):
                            raise RuntimeError(
                                "Warmup shares a full cache block with measured inputs"
                            )
                        prefix = requests[0].prompt_token_ids[:16]
                        if any(prefix == p for p in prior_prefixes):
                            raise RuntimeError("Measured prefix reused within a server session")
                        prior_prefixes.append(prefix)
                        warm_result = run_benchmark(
                            warm, concurrency=4, send_request=client.stream_completion
                        )
                        write_json_result(out / f"{name}_warmup.json", asdict(warm_result))
                        if warm_result.failed_requests:
                            raise RuntimeError("Warmup failed; no measured run was started")
                        time.sleep(2)
                        reset = reset_cache(args.port, log) if enabled else {"disabled": True}
                        time.sleep(1)
                        before = http(args.port, "/metrics")
                        (out / f"{name}_before.prom").write_text(before, encoding="utf-8")
                        gpu_before = command_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=temperature.gpu,"
                                "clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu",
                                "--format=csv,noheader",
                            ]
                        )
                        print(f"Measuring {name}", flush=True)
                        result = run_benchmark(
                            requests, concurrency=4, send_request=client.stream_completion
                        )
                        payload = build_result_payload(config, client, result, metadata)
                        payload["experiment"] = {
                            "prefix_caching": enabled,
                            "repeat": repeat,
                            "content_seed": content_seed,
                            "generator": "controlled_inputs_v1",
                            "ignore_eos": True,
                            "input_audit": audit,
                            "cache_reset": reset,
                            "warmup_requests": 8,
                            "server_launch": str(session / "launch.json"),
                            "initial_cache": "cold; reuse allowed within measured batch",
                            "gpu_before": gpu_before,
                        }
                        # Preserve outcomes even when later metrics collection fails.
                        write_json_result(out / f"{name}.json", payload)
                        validation = validate_result(payload)
                        after = wait_metrics(
                            args.port,
                            counter(before, "vllm:request_success_total")
                            + result.successful_requests,
                        )
                        (out / f"{name}_after.prom").write_text(after, encoding="utf-8")
                        queries = counter(after, "vllm:prefix_cache_queries_total") - counter(
                            before, "vllm:prefix_cache_queries_total"
                        )
                        hits = counter(after, "vllm:prefix_cache_hits_total") - counter(
                            before, "vllm:prefix_cache_hits_total"
                        )
                        validation.update(
                            {
                                "prefix_queries_delta": queries,
                                "prefix_hits_delta": hits,
                                "prefix_hit_rate": hits / queries if queries else None,
                                "output_lengths": sorted({t.output_tokens for t in result.traces}),
                                "fixed_output_length_passed": all(
                                    t.output_tokens == 64 for t in result.traces
                                ),
                            }
                        )
                        write_json_result(out / f"{name}_validation.json", validation)
                        run_index.append({"file": f"{name}.json", "validation": validation})
                        print(
                            f"{name}: {result.successful_requests}/80 success; "
                            f"cache hit rate={validation['prefix_hit_rate']}",
                            flush=True,
                        )
                    prior_prefixes.clear()
                finally:
                    stop_server(process)
            time.sleep(2)
    write_json_result(out / "run_index.json", {"runs": run_index})
    final = collect_run_metadata([sys.executable, *sys.argv])
    if metadata["source_sha256"] != final["source_sha256"]:
        raise RuntimeError("Source files changed during the experiment")
    print(f"COMPLETE: {len(run_index)} measured runs in {out}", flush=True)


if __name__ == "__main__":
    main()
