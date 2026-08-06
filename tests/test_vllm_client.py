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
