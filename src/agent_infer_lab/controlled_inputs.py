"""Exact-prefix synthetic Token workloads for controlled system experiments."""

import hashlib
import json
import random
import string
from collections.abc import Sequence

from agent_infer_lab.prompting import PreparedRequest, Tokenize
from agent_infer_lab.workloads import RequestSpec

TOKEN_POOL_TEXT = (
    "Synthetic inference performance workload. "
    + " ".join(str(number) for number in range(512))
    + " "
    + " ".join(a + b for a in string.ascii_lowercase for b in string.ascii_lowercase)
)


def prepare_controlled_requests(
    specs: tuple[RequestSpec, ...],
    tokenize: Tokenize,
    *,
    content_seed: int,
) -> tuple[PreparedRequest, ...]:
    """Use a shared random prefix and distinct first suffix Tokens.

    These are synthetic valid Token-ID sequences, not a language-quality test.
    Unlike the legacy cyclic generator, the seed changes actual input content.
    """
    if not specs:
        raise ValueError("specs must not be empty")
    if isinstance(content_seed, bool) or not isinstance(content_seed, int):
        raise ValueError("content_seed must be an integer")
    if len({spec.request_id for spec in specs}) != len(specs):
        raise ValueError("request IDs must be unique")
    for spec in specs:
        if not 0 <= spec.shared_prefix_tokens <= spec.input_tokens:
            raise ValueError("invalid shared prefix length")
    tokens = tokenize(TOKEN_POOL_TEXT)
    if not tokens or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens
    ):
        raise ValueError("tokenize must return valid non-negative Token IDs")
    pool = tuple(dict.fromkeys(tokens))
    if len(pool) < len(specs):
        raise ValueError("Token pool needs at least one distinct Token per request")
    rng = random.Random(content_seed)
    prefix = tuple(rng.choices(pool, k=max(s.shared_prefix_tokens for s in specs)))
    markers = rng.sample(pool, len(specs))
    requests = []
    for index, spec in enumerate(specs):
        suffix_length = spec.input_tokens - spec.shared_prefix_tokens
        suffix = ()
        if suffix_length:
            suffix = (markers[index], *rng.choices(pool, k=suffix_length - 1))
        requests.append(
            PreparedRequest(
                spec.request_id, prefix[: spec.shared_prefix_tokens] + suffix, spec.output_tokens
            )
        )
    return tuple(requests)


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return min(len(left), len(right))


def audit_inputs(requests: tuple[PreparedRequest, ...]) -> dict[str, object]:
    """Record duplicate inputs, observed LCP range and a stable input digest."""
    prompts = [request.prompt_token_ids for request in requests]
    lengths = [
        common_prefix_length(prompts[i], prompts[j])
        for i in range(len(prompts))
        for j in range(i + 1, len(prompts))
    ]
    encoded = json.dumps(
        [[r.request_id, list(r.prompt_token_ids), r.output_tokens] for r in requests],
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "request_count": len(requests),
        "input_lengths": sorted({len(p) for p in prompts}),
        "duplicate_prompt_count": len(prompts) - len(set(prompts)),
        "pairwise_common_prefix_min": min(lengths) if lengths else None,
        "pairwise_common_prefix_max": max(lengths) if lengths else None,
        "input_sha256": hashlib.sha256(encoded).hexdigest(),
    }
