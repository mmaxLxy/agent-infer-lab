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
