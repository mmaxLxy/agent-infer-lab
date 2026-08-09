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
