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
        except (OSError, http.client.HTTPException) as error:
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
        except (
            OSError,
            UnicodeDecodeError,
            http.client.HTTPException,
        ) as error:
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
