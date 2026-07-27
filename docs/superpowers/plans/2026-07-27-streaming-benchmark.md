# Fixed-Concurrency Streaming Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an exact-token, fixed-concurrency benchmark client that sends streaming `/v1/completions` requests to vLLM and reports real inference latency and throughput metrics.

**Architecture:** Convert existing deterministic `RequestSpec` values into immutable exact-token `PreparedRequest` values, send each request through an isolated standard-library HTTP connection, and schedule calls with a fixed-size thread pool. Reuse the existing `RequestTrace` and `summarize_metrics()` implementation instead of duplicating metric formulas.

**Tech Stack:** Python 3.12, `http.client`, `concurrent.futures.ThreadPoolExecutor`, vLLM `/tokenize`, vLLM `/v1/completions`, pytest, Ruff, uv.

## Global Constraints

- Add no runtime dependency; use only the Python 3.12 standard library.
- Send exact Token ID lists to `/v1/completions`; do not decode and re-tokenize prompts.
- Keep `/tokenize` calibration outside each inference request's measured interval.
- Use `time.perf_counter()` for request timing.
- Use fixed-concurrency closed-loop scheduling.
- CPU tests must not require GPU, CUDA, PyTorch, vLLM, or network access.
- Stop the batch when a request fails; do not add retries in this version.
- Do not implement Chat Completions, Poisson arrivals, Goodput, GPU sampling, result persistence, charts, or CUDA kernels in this version.
- Per user preference, implement each small component before adding its tests; do not include a deliberate red-test demonstration.

---

## File Map

- Create `src/agent_infer_lab/prompting.py`: exact-token prompt construction.
- Create `tests/test_prompting.py`: deterministic length and shared-prefix checks.
- Create `src/agent_infer_lab/vllm_client.py`: `/tokenize`, streaming Completion, SSE parsing, and timestamps.
- Create `tests/test_vllm_client.py`: fake HTTP and fake clock tests.
- Create `src/agent_infer_lab/benchmark.py`: fixed-concurrency scheduling, end-to-end orchestration, and CLI.
- Create `tests/test_benchmark.py`: concurrency, delegation, failure, and CLI output checks.
- Modify `pyproject.toml`: register `agent-infer-bench`.
- Create `docs/progress/2026-07-27.md`: record the completed stage after real verification.

### Task 1: Exact-Token Prompt Construction

**Files:**
- Create: `src/agent_infer_lab/prompting.py`
- Create: `tests/test_prompting.py`

**Interfaces:**
- Consumes: `RequestSpec` from `agent_infer_lab.workloads`.
- Produces: `PreparedRequest`, `prepare_requests(specs, tokenize, seed_text=...)`.

- [ ] **Step 1: Create the Prompt module**

Create `src/agent_infer_lab/prompting.py` with:

```python
"""Exact-token prompt construction for reproducible inference requests."""

from collections.abc import Callable
from dataclasses import dataclass

from agent_infer_lab.workloads import RequestSpec

DEFAULT_SEED_TEXT = """
You are an inference agent. Read the task, inspect the available tools,
compare the evidence, and return a concise answer with explicit reasoning.
Tool names: search, retrieve, calculate, summarize, verify, and respond.
Context contains system instructions, user messages, tool results, and notes.
Always preserve identifiers, numbers, ordering, and requested output format.
"""

Tokenize = Callable[[str], tuple[int, ...]]


def _is_token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _take_cyclic(
    token_pool: tuple[int, ...],
    length: int,
    *,
    offset: int = 0,
) -> tuple[int, ...]:
    return tuple(
        token_pool[(offset + index) % len(token_pool)] for index in range(length)
    )


@dataclass(frozen=True)
class PreparedRequest:
    """One immutable request containing exact input Token IDs."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    output_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.prompt_token_ids, tuple) or not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be a non-empty tuple")
        if any(not _is_token_id(token_id) for token_id in self.prompt_token_ids):
            raise ValueError("prompt_token_ids must contain non-negative integers")
        if (
            not isinstance(self.output_tokens, int)
            or isinstance(self.output_tokens, bool)
            or self.output_tokens <= 0
        ):
            raise ValueError("output_tokens must be a positive integer")


def prepare_requests(
    specs: tuple[RequestSpec, ...],
    tokenize: Tokenize,
    *,
    seed_text: str = DEFAULT_SEED_TEXT,
) -> tuple[PreparedRequest, ...]:
    """Create exact-length deterministic prompts from request specifications."""

    if not specs:
        raise ValueError("specs must not be empty")
    if not isinstance(seed_text, str) or not seed_text.strip():
        raise ValueError("seed_text must be a non-empty string")

    token_pool = tokenize(seed_text)
    if not isinstance(token_pool, tuple) or not token_pool:
        raise ValueError("tokenize must return a non-empty tuple")
    if any(not _is_token_id(token_id) for token_id in token_pool):
        raise ValueError("tokenize must return non-negative integer Token IDs")

    max_shared_tokens = max(spec.shared_prefix_tokens for spec in specs)
    shared_pool = _take_cyclic(token_pool, max_shared_tokens)

    prepared = []
    for index, spec in enumerate(specs):
        if not 0 <= spec.shared_prefix_tokens <= spec.input_tokens:
            raise ValueError(
                "shared_prefix_tokens must be between zero and input_tokens"
            )

        unique_tokens = spec.input_tokens - spec.shared_prefix_tokens
        shared_prefix = shared_pool[: spec.shared_prefix_tokens]
        unique_offset = (index + 1) * 17
        unique_suffix = _take_cyclic(
            token_pool,
            unique_tokens,
            offset=unique_offset,
        )
        prompt_token_ids = shared_prefix + unique_suffix
        if len(prompt_token_ids) != spec.input_tokens:
            raise RuntimeError("prepared prompt length does not match input_tokens")

        prepared.append(
            PreparedRequest(
                request_id=spec.request_id,
                prompt_token_ids=prompt_token_ids,
                output_tokens=spec.output_tokens,
            )
        )
    return tuple(prepared)
```

- [ ] **Step 2: Add Prompt tests**

Create `tests/test_prompting.py` with:

```python
from dataclasses import FrozenInstanceError

import pytest

from agent_infer_lab.prompting import PreparedRequest, prepare_requests
from agent_infer_lab.workloads import RequestSpec


def make_specs() -> tuple[RequestSpec, ...]:
    return (
        RequestSpec("req-000000", 8, 4, 3),
        RequestSpec("req-000001", 8, 5, 3),
        RequestSpec("req-000002", 6, 2, 3),
    )


def tokenize(_: str) -> tuple[int, ...]:
    return (10, 11, 12, 13, 14, 15, 16)


def test_prepare_requests_preserves_exact_lengths_and_outputs() -> None:
    prepared = prepare_requests(make_specs(), tokenize)

    assert tuple(len(request.prompt_token_ids) for request in prepared) == (8, 8, 6)
    assert tuple(request.output_tokens for request in prepared) == (4, 5, 2)


def test_prepare_requests_builds_shared_prefixes() -> None:
    prepared = prepare_requests(make_specs(), tokenize)

    assert prepared[0].prompt_token_ids[:3] == prepared[1].prompt_token_ids[:3]
    assert prepared[1].prompt_token_ids[:3] == prepared[2].prompt_token_ids[:3]


def test_prepare_requests_builds_request_specific_suffixes() -> None:
    prepared = prepare_requests(make_specs(), tokenize)

    assert prepared[0].prompt_token_ids[3:] != prepared[1].prompt_token_ids[3:]


def test_prepare_requests_is_deterministic() -> None:
    assert prepare_requests(make_specs(), tokenize) == prepare_requests(
        make_specs(),
        tokenize,
    )


def test_prepare_requests_calls_tokenize_once() -> None:
    calls: list[str] = []

    def recording_tokenize(text: str) -> tuple[int, ...]:
        calls.append(text)
        return (1, 2, 3)

    prepare_requests(make_specs(), recording_tokenize, seed_text="agent context")

    assert calls == ["agent context"]


def test_prepared_request_is_immutable() -> None:
    request = PreparedRequest("req-1", (1, 2), 3)

    with pytest.raises(FrozenInstanceError):
        request.output_tokens = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("specs", "seed_text", "message"),
    [
        ((), "valid", "specs must not be empty"),
        (make_specs(), "", "seed_text must be a non-empty string"),
    ],
)
def test_prepare_requests_rejects_invalid_inputs(
    specs: tuple[RequestSpec, ...],
    seed_text: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_requests(specs, tokenize, seed_text=seed_text)


@pytest.mark.parametrize("tokens", [(), (1, True), (1, -1)])
def test_prepare_requests_rejects_invalid_tokenizer_output(
    tokens: tuple[object, ...],
) -> None:
    def invalid_tokenize(_: str) -> tuple[int, ...]:
        return tokens  # type: ignore[return-value]

    with pytest.raises(ValueError):
        prepare_requests(make_specs(), invalid_tokenize)


def test_prepare_requests_rejects_invalid_shared_prefix() -> None:
    specs = (RequestSpec("req-1", 4, 2, 5),)

    with pytest.raises(ValueError, match="shared_prefix_tokens"):
        prepare_requests(specs, tokenize)
```

- [ ] **Step 3: Run focused tests and lint**

Run:

```bash
pytest tests/test_prompting.py -q
ruff check src/agent_infer_lab/prompting.py tests/test_prompting.py
```

Expected:

```text
All Prompt tests pass.
All checks passed!
```

- [ ] **Step 4: Commit the Prompt component**

```bash
git add src/agent_infer_lab/prompting.py tests/test_prompting.py
git commit -m "feat: build exact-token prompts"
```

### Task 2: vLLM HTTP and Streaming Client

**Files:**
- Create: `src/agent_infer_lab/vllm_client.py`
- Create: `tests/test_vllm_client.py`

**Interfaces:**
- Consumes: `PreparedRequest`.
- Produces: `VllmClient.tokenize(text) -> tuple[int, ...]` and `VllmClient.stream_completion(request) -> RequestTrace`.

- [ ] **Step 1: Create the vLLM client**

Create `src/agent_infer_lab/vllm_client.py` with:

```python
"""Standard-library HTTP client for vLLM tokenization and streaming inference."""

import http.client
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import SplitResult, urlsplit

from agent_infer_lab.metrics import RequestTrace
from agent_infer_lab.prompting import PreparedRequest

Clock = Callable[[], float]


class VllmClientError(RuntimeError):
    """Raised when vLLM communication or response validation fails."""


def _is_token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True)
class VllmClient:
    """Client for one vLLM model served through an OpenAI-compatible API."""

    base_url: str
    model: str
    timeout: float = 60.0
    clock: Clock = field(default=time.perf_counter, repr=False, compare=False)

    def __post_init__(self) -> None:
        parsed = self._parsed_url()
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a positive number")

    def _parsed_url(self) -> SplitResult:
        return urlsplit(self.base_url)

    def _endpoint(self, suffix: str) -> str:
        prefix = self._parsed_url().path.rstrip("/")
        return f"{prefix}{suffix}" or "/"

    def _connection(self) -> http.client.HTTPConnection:
        parsed = self._parsed_url()
        connection_type = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        return connection_type(parsed.hostname, parsed.port, timeout=self.timeout)

    @staticmethod
    def _json_body(payload: dict[str, Any]) -> bytes:
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    @staticmethod
    def _read_json_response(
        response: http.client.HTTPResponse,
        *,
        endpoint: str,
    ) -> dict[str, Any]:
        raw_body = response.read()
        if response.status != 200:
            detail = raw_body.decode("utf-8", errors="replace")
            raise VllmClientError(
                f"{endpoint} returned HTTP {response.status}: {detail}"
            )
        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise VllmClientError(f"{endpoint} returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise VllmClientError(f"{endpoint} must return a JSON object")
        return payload

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Tokenize text with the served model."""

        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")

        endpoint = self._endpoint("/tokenize")
        connection = self._connection()
        try:
            connection.request(
                "POST",
                endpoint,
                body=self._json_body({"model": self.model, "prompt": text}),
                headers={"Content-Type": "application/json"},
            )
            payload = self._read_json_response(
                connection.getresponse(),
                endpoint=endpoint,
            )
        except OSError as error:
            raise VllmClientError(f"cannot connect to {self.base_url}") from error
        finally:
            connection.close()

        tokens = payload.get("tokens")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(not _is_token_id(token_id) for token_id in tokens)
        ):
            raise VllmClientError("/tokenize returned invalid tokens")
        return tuple(tokens)

    def stream_completion(self, request: PreparedRequest) -> RequestTrace:
        """Send one streaming Completion request and collect raw timings."""

        endpoint = self._endpoint("/v1/completions")
        payload = {
            "model": self.model,
            "prompt": list(request.prompt_token_ids),
            "max_tokens": request.output_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        connection = self._connection()
        started_at = self.clock()
        first_token_at: float | None = None
        completed_at: float | None = None
        completion_tokens: int | None = None

        try:
            connection.request(
                "POST",
                endpoint,
                body=self._json_body(payload),
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                detail = response.read().decode("utf-8", errors="replace")
                raise VllmClientError(
                    f"{request.request_id} {endpoint} returned "
                    f"HTTP {response.status}: {detail}"
                )

            while raw_line := response.readline():
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    completed_at = self.clock()
                    break

                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise VllmClientError(
                        f"{request.request_id} received invalid SSE JSON"
                    ) from error
                if not isinstance(event, dict):
                    raise VllmClientError(
                        f"{request.request_id} received a non-object SSE event"
                    )

                usage = event.get("usage")
                if isinstance(usage, dict):
                    value = usage.get("completion_tokens")
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value > 0
                    ):
                        completion_tokens = value

                choices = event.get("choices")
                if isinstance(choices, list) and choices:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        text = choice.get("text")
                        if (
                            first_token_at is None
                            and isinstance(text, str)
                            and text
                        ):
                            first_token_at = self.clock()
        except (OSError, UnicodeDecodeError) as error:
            raise VllmClientError(
                f"{request.request_id} failed while reading {endpoint}"
            ) from error
        finally:
            connection.close()

        if completed_at is None:
            raise VllmClientError(f"{request.request_id} ended without [DONE]")
        if first_token_at is None:
            raise VllmClientError(
                f"{request.request_id} completed without a generated token"
            )
        if completion_tokens is None:
            raise VllmClientError(
                f"{request.request_id} completed without completion token usage"
            )

        return RequestTrace(
            request_id=request.request_id,
            started_at=started_at,
            first_token_at=first_token_at,
            completed_at=completed_at,
            output_tokens=completion_tokens,
        )
```

- [ ] **Step 2: Add fake HTTP client tests**

Create `tests/test_vllm_client.py` with:

```python
import json
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from agent_infer_lab.prompting import PreparedRequest
from agent_infer_lab.vllm_client import VllmClient, VllmClientError


class FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = b"",
        lines: tuple[bytes, ...] = (),
    ) -> None:
        self.status = status
        self._body = body
        self._lines: Iterator[bytes] = iter(lines)

    def read(self) -> bytes:
        return self._body

    def readline(self) -> bytes:
        return next(self._lines, b"")


class FakeConnection:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[tuple[str, str, bytes, dict[str, str]]] = []
        self.closed = False

    def request(
        self,
        method: str,
        endpoint: str,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        self.requests.append((method, endpoint, body, headers))

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


def make_client(
    connection: FakeConnection,
    *,
    clock_values: tuple[float, ...] = (10.0, 10.2, 11.0),
) -> VllmClient:
    values = iter(clock_values)
    client = VllmClient(
        "http://127.0.0.1:8000",
        "Qwen/Qwen2.5-0.5B-Instruct",
        clock=lambda: next(values),
    )
    object.__setattr__(client, "_connection", lambda: connection)
    return client


def test_tokenize_sends_model_and_prompt() -> None:
    response = FakeResponse(body=json.dumps({"tokens": [1, 2, 3]}).encode())
    connection = FakeConnection(response)
    client = make_client(connection)

    assert client.tokenize("agent prompt") == (1, 2, 3)

    method, endpoint, body, headers = connection.requests[0]
    assert method == "POST"
    assert endpoint == "/tokenize"
    assert json.loads(body) == {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "agent prompt",
    }
    assert headers["Content-Type"] == "application/json"
    assert connection.closed


def test_stream_completion_builds_trace_from_sse() -> None:
    lines = (
        b"\n",
        b'data: {"choices":[{"text":""}]}\n',
        b'data: {"choices":[{"text":"CUDA"}]}\n',
        b'data: {"choices":[],"usage":{"completion_tokens":4}}\n',
        b"data: [DONE]\n",
    )
    connection = FakeConnection(FakeResponse(lines=lines))
    client = make_client(connection)
    request = PreparedRequest("req-000001", (7, 8, 9), 4)

    trace = client.stream_completion(request)

    assert trace.request_id == "req-000001"
    assert trace.started_at == 10.0
    assert trace.first_token_at == 10.2
    assert trace.completed_at == 11.0
    assert trace.output_tokens == 4

    method, endpoint, body, headers = connection.requests[0]
    assert method == "POST"
    assert endpoint == "/v1/completions"
    assert json.loads(body) == {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": [7, 8, 9],
        "max_tokens": 4,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert headers["Accept"] == "text/event-stream"


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (
            (
                b'data: {"choices":[{"text":"x"}]}\n',
                b'data: {"choices":[],"usage":{"completion_tokens":1}}\n',
            ),
            "without \\[DONE\\]",
        ),
        (
            (
                b'data: {"choices":[],"usage":{"completion_tokens":1}}\n',
                b"data: [DONE]\n",
            ),
            "without a generated token",
        ),
        (
            (
                b'data: {"choices":[{"text":"x"}]}\n',
                b"data: [DONE]\n",
            ),
            "without completion token usage",
        ),
    ],
)
def test_stream_completion_rejects_incomplete_streams(
    lines: tuple[bytes, ...],
    message: str,
) -> None:
    connection = FakeConnection(FakeResponse(lines=lines))
    client = make_client(connection)

    with pytest.raises(VllmClientError, match=message):
        client.stream_completion(PreparedRequest("req-1", (1,), 1))


def test_stream_completion_reports_http_error() -> None:
    connection = FakeConnection(FakeResponse(status=404, body=b"model not found"))
    client = make_client(connection, clock_values=(10.0,))

    with pytest.raises(VllmClientError, match="HTTP 404"):
        client.stream_completion(PreparedRequest("req-1", (1,), 1))


def test_tokenize_reports_invalid_tokens() -> None:
    connection = FakeConnection(
        FakeResponse(body=json.dumps({"tokens": [1, True]}).encode())
    )
    client = make_client(connection)

    with pytest.raises(VllmClientError, match="invalid tokens"):
        client.tokenize("prompt")


def test_https_uses_https_connection() -> None:
    client = VllmClient("https://example.com:8443", "model")

    with patch(
        "agent_infer_lab.vllm_client.http.client.HTTPSConnection"
    ) as connection_type:
        client._connection()

    connection_type.assert_called_once_with("example.com", 8443, timeout=60.0)
```

- [ ] **Step 3: Run focused tests and lint**

Run:

```bash
pytest tests/test_vllm_client.py -q
ruff check src/agent_infer_lab/vllm_client.py tests/test_vllm_client.py
```

Expected:

```text
All vLLM client tests pass.
All checks passed!
```

- [ ] **Step 4: Commit the vLLM client**

```bash
git add src/agent_infer_lab/vllm_client.py tests/test_vllm_client.py
git commit -m "feat: add streaming vllm client"
```

### Task 3: Fixed-Concurrency Runner and CLI

**Files:**
- Create: `src/agent_infer_lab/benchmark.py`
- Create: `tests/test_benchmark.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `WorkloadConfig`, `PreparedRequest`, `VllmClient`, and `summarize_metrics`.
- Produces: `run_benchmark()`, `execute_benchmark()`, and `agent-infer-bench`.

- [ ] **Step 1: Create the Benchmark runner**

Create `src/agent_infer_lab/benchmark.py` with:

```python
"""Fixed-concurrency execution for reproducible vLLM benchmarks."""

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from agent_infer_lab.metrics import MetricsSummary, RequestTrace, summarize_metrics
from agent_infer_lab.prompting import PreparedRequest, prepare_requests
from agent_infer_lab.vllm_client import VllmClient
from agent_infer_lab.workloads import WorkloadConfig, generate_workload

SendRequest = Callable[[PreparedRequest], RequestTrace]


def run_benchmark(
    requests: tuple[PreparedRequest, ...],
    *,
    concurrency: int,
    send_request: SendRequest,
) -> MetricsSummary:
    """Run all requests with a fixed maximum concurrency."""

    if not requests:
        raise ValueError("requests must not be empty")
    if (
        not isinstance(concurrency, int)
        or isinstance(concurrency, bool)
        or concurrency <= 0
    ):
        raise ValueError("concurrency must be a positive integer")
    if concurrency > len(requests):
        raise ValueError("concurrency cannot exceed request count")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        traces = tuple(executor.map(send_request, requests))
    return summarize_metrics(traces)


def execute_benchmark(
    config: WorkloadConfig,
    client: VllmClient,
) -> MetricsSummary:
    """Prepare exact prompts, execute requests, and summarize results."""

    specs = generate_workload(config)
    requests = prepare_requests(specs, client.tokenize)
    return run_benchmark(
        requests,
        concurrency=config.concurrency,
        send_request=client.stream_completion,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed-concurrency streaming vLLM benchmark."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=_positive_int, default=20)
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument(
        "--input-tokens",
        type=_positive_int,
        nargs="+",
        default=[128],
    )
    parser.add_argument(
        "--output-tokens",
        type=_positive_int,
        nargs="+",
        default=[32],
    )
    parser.add_argument("--shared-prefix-ratio", type=_ratio, default=0.5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def _format_optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = WorkloadConfig(
        request_count=args.requests,
        input_token_choices=tuple(args.input_tokens),
        output_token_choices=tuple(args.output_tokens),
        shared_prefix_ratio=args.shared_prefix_ratio,
        concurrency=args.concurrency,
        seed=args.seed,
    )
    client = VllmClient(
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
    )
    summary = execute_benchmark(config, client)

    print(f"requests: {summary.request_count}")
    print(f"output_tokens: {summary.total_output_tokens}")
    print(f"duration_seconds: {summary.duration_seconds:.6f}")
    print(
        "output_throughput_tokens_per_second: "
        f"{summary.output_throughput_tokens_per_second:.6f}"
    )
    print(f"ttft_p50_seconds: {summary.ttft_p50_seconds:.6f}")
    print(f"ttft_p99_seconds: {summary.ttft_p99_seconds:.6f}")
    print(f"tpot_p50_seconds: {_format_optional(summary.tpot_p50_seconds)}")
    print(f"tpot_p99_seconds: {_format_optional(summary.tpot_p99_seconds)}")
    print(f"e2e_p50_seconds: {summary.e2e_p50_seconds:.6f}")
    print(f"e2e_p99_seconds: {summary.e2e_p99_seconds:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register the CLI**

In `pyproject.toml`, change:

```toml
[project.scripts]
agent-infer-env = "agent_infer_lab.environment:main"
```

to:

```toml
[project.scripts]
agent-infer-bench = "agent_infer_lab.benchmark:main"
agent-infer-env = "agent_infer_lab.environment:main"
```

- [ ] **Step 3: Add Benchmark tests**

Create `tests/test_benchmark.py` with:

```python
import threading
import time

import pytest

from agent_infer_lab import benchmark
from agent_infer_lab.benchmark import execute_benchmark, run_benchmark
from agent_infer_lab.metrics import MetricsSummary, RequestTrace
from agent_infer_lab.prompting import PreparedRequest
from agent_infer_lab.workloads import WorkloadConfig


def make_requests(count: int = 4) -> tuple[PreparedRequest, ...]:
    return tuple(
        PreparedRequest(f"req-{index}", (index + 1,), 2)
        for index in range(count)
    )


def make_trace(request: PreparedRequest) -> RequestTrace:
    index = int(request.request_id.removeprefix("req-"))
    started = float(index)
    return RequestTrace(
        request_id=request.request_id,
        started_at=started,
        first_token_at=started + 0.1,
        completed_at=started + 1.0,
        output_tokens=2,
    )


def test_run_benchmark_executes_each_request_and_summarizes() -> None:
    seen: list[str] = []

    def send(request: PreparedRequest) -> RequestTrace:
        seen.append(request.request_id)
        return make_trace(request)

    summary = run_benchmark(make_requests(), concurrency=2, send_request=send)

    assert sorted(seen) == ["req-0", "req-1", "req-2", "req-3"]
    assert summary.request_count == 4
    assert summary.total_output_tokens == 8


def test_run_benchmark_never_exceeds_concurrency() -> None:
    lock = threading.Lock()
    running = 0
    peak = 0

    def send(request: PreparedRequest) -> RequestTrace:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.01)
        with lock:
            running -= 1
        return make_trace(request)

    run_benchmark(make_requests(6), concurrency=2, send_request=send)

    assert peak == 2


def test_run_benchmark_propagates_request_failure() -> None:
    def send(_: PreparedRequest) -> RequestTrace:
        raise RuntimeError("service failed")

    with pytest.raises(RuntimeError, match="service failed"):
        run_benchmark(make_requests(2), concurrency=1, send_request=send)


@pytest.mark.parametrize("concurrency", [0, -1, True])
def test_run_benchmark_rejects_invalid_concurrency(
    concurrency: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        run_benchmark(
            make_requests(),
            concurrency=concurrency,  # type: ignore[arg-type]
            send_request=make_trace,
        )


def test_execute_benchmark_connects_existing_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WorkloadConfig(2, (4,), (2,), 0.5, 1, 7)
    requests = (
        PreparedRequest("req-000000", (1, 2, 3, 4), 2),
        PreparedRequest("req-000001", (1, 2, 5, 6), 2),
    )
    expected = MetricsSummary(
        request_count=2,
        total_output_tokens=4,
        duration_seconds=2.0,
        output_throughput_tokens_per_second=2.0,
        ttft_p50_seconds=0.1,
        ttft_p99_seconds=0.2,
        tpot_p50_seconds=0.9,
        tpot_p99_seconds=1.8,
        e2e_p50_seconds=1.0,
        e2e_p99_seconds=2.0,
    )

    class FakeClient:
        def tokenize(self, _: str) -> tuple[int, ...]:
            return (1, 2, 3)

        def stream_completion(self, request: PreparedRequest) -> RequestTrace:
            return make_trace(request)

    monkeypatch.setattr(benchmark, "prepare_requests", lambda specs, tokenize: requests)
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda prepared, *, concurrency, send_request: expected,
    )

    assert execute_benchmark(config, FakeClient()) == expected  # type: ignore[arg-type]


def test_main_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = MetricsSummary(
        request_count=2,
        total_output_tokens=8,
        duration_seconds=2.0,
        output_throughput_tokens_per_second=4.0,
        ttft_p50_seconds=0.1,
        ttft_p99_seconds=0.2,
        tpot_p50_seconds=0.03,
        tpot_p99_seconds=0.04,
        e2e_p50_seconds=1.0,
        e2e_p99_seconds=1.2,
    )
    monkeypatch.setattr(benchmark, "execute_benchmark", lambda config, client: summary)

    benchmark.main(["--model", "test-model", "--requests", "2", "--concurrency", "1"])

    output = capsys.readouterr().out
    assert "requests: 2" in output
    assert "output_throughput_tokens_per_second: 4.000000" in output
    assert "ttft_p99_seconds: 0.200000" in output
```

- [ ] **Step 4: Run focused tests, all tests, lint, and lock check**

Run:

```bash
pytest tests/test_benchmark.py -q
pytest -q
ruff check .
/home/ayax/.local/bin/uv lock --check
```

Expected:

```text
All Benchmark tests pass.
All project tests pass.
All checks passed!
Resolved 8 packages.
```

- [ ] **Step 5: Commit the runner and CLI**

```bash
git add pyproject.toml src/agent_infer_lab/benchmark.py tests/test_benchmark.py
git commit -m "feat: run fixed-concurrency benchmarks"
```

### Task 4: Real vLLM Verification and Progress Record

**Files:**
- Create: `docs/progress/2026-07-27.md`

**Interfaces:**
- Consumes: the running local vLLM server and `agent-infer-bench`.
- Produces: one reproducible real-GPU benchmark result and a progress record.

- [ ] **Step 1: Start the vLLM service in terminal 1**

```bash
cd /mnt/d/agent-infer-lab
source /home/ayax/.venvs/agent-infer-lab/bin/activate
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export CUDA_HOME=/usr/local/cuda-12.9
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 2048
```

Expected:

```text
Application startup complete.
The service listens on 127.0.0.1:8000 without OOM.
```

- [ ] **Step 2: Verify service and exact tokenization in terminal 2**

```bash
cd /mnt/d/agent-infer-lab
source /home/ayax/.venvs/agent-infer-lab-dev/bin/activate
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
curl -s http://127.0.0.1:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Agent inference benchmark"}' \
  | python -m json.tool
```

Expected:

```text
/v1/models contains Qwen/Qwen2.5-0.5B-Instruct.
/tokenize returns a non-empty tokens array.
```

- [ ] **Step 3: Run a safe smoke benchmark**

```bash
agent-infer-bench \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 4 \
  --concurrency 1 \
  --input-tokens 64 \
  --output-tokens 16 \
  --shared-prefix-ratio 0.5 \
  --seed 20260727
```

Expected:

```text
requests: 4
output_tokens: a positive number
duration_seconds: a positive number
TTFT, TPOT, E2E, and throughput fields are printed.
```

- [ ] **Step 4: Run the first fixed-concurrency experiment**

```bash
agent-infer-bench \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 20 \
  --concurrency 4 \
  --input-tokens 128 \
  --output-tokens 32 \
  --shared-prefix-ratio 0.5 \
  --seed 20260727
```

Expected:

```text
20 requests complete without OOM.
All latency percentiles and output throughput are printed.
```

- [ ] **Step 5: Re-run project verification**

```bash
pytest -q
ruff check .
/home/ayax/.local/bin/uv lock --check
git status --short --untracked-files=all
```

Expected:

```text
All tests pass.
All checks passed.
The dependency lock is current.
Only the intended progress document may be untracked.
```

- [ ] **Step 6: Record verified facts**

Create `docs/progress/2026-07-27.md` only after the real run. Record:

- branch and commit IDs;
- exact vLLM start command;
- exact benchmark command;
- model and GPU;
- request count and concurrency;
- input/output Token settings and shared-prefix ratio;
- TTFT P50/P99;
- TPOT P50/P99;
- E2E P50/P99;
- output throughput;
- whether OOM occurred;
- pytest, Ruff, and uv results;
- observed warnings and remaining limitations.

Do not invent benchmark numbers.

- [ ] **Step 7: Commit the verified record**

```bash
git add docs/progress/2026-07-27.md
git commit -m "docs: record streaming benchmark results"
```

### Task 5: Push, CI, and Merge

**Files:**
- No source changes expected.

**Interfaces:**
- Consumes: completed feature branch and local verification evidence.
- Produces: merged GitHub PR with passing CPU CI.

- [ ] **Step 1: Confirm the final branch state**

```bash
git branch --show-current
git status --short --untracked-files=all
git log --oneline --decorate -5
```

Expected:

```text
Current branch is agent/streaming-benchmark.
Worktree is clean.
Recent commits correspond to design, Prompt, client, runner, and progress.
```

- [ ] **Step 2: Push the branch**

```bash
git \
  -c credential.helper='!/mnt/d/tools/gh/bin/gh auth git-credential' \
  -c http.proxy=http://127.0.0.1:7897 \
  -c https.proxy=http://127.0.0.1:7897 \
  push -u origin agent/streaming-benchmark
```

Expected:

```text
The remote branch is created or updated successfully.
```

- [ ] **Step 3: Create a normal PR**

```bash
HTTP_PROXY=http://127.0.0.1:7897 \
HTTPS_PROXY=http://127.0.0.1:7897 \
/mnt/d/tools/gh/bin/gh pr create \
  --base main \
  --head agent/streaming-benchmark \
  --title "feat: add fixed-concurrency streaming benchmark" \
  --body "Implements exact-token prompt construction, vLLM streaming requests, fixed-concurrency scheduling, CPU tests, and a verified local GPU run."
```

Expected:

```text
A non-draft PR URL is returned.
```

- [ ] **Step 4: Check CI and merge directly after success**

```bash
HTTP_PROXY=http://127.0.0.1:7897 \
HTTPS_PROXY=http://127.0.0.1:7897 \
/mnt/d/tools/gh/bin/gh pr checks --watch
```

After every required check passes:

```bash
HTTP_PROXY=http://127.0.0.1:7897 \
HTTPS_PROXY=http://127.0.0.1:7897 \
/mnt/d/tools/gh/bin/gh pr merge --merge
```

Expected:

```text
CPU CI passes and the PR is merged without a Draft-to-Ready step.
```
