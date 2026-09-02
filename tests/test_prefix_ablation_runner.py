"""CPU checks for the experiment runner's HTTP and counter helpers."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prefix_runner", Path(__file__).resolve().parents[1] / "scripts/run_prefix_ablation.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_counter_sums_labels_without_matching_other_names() -> None:
    text = "\n".join(
        [
            "# HELP vllm:prefix_cache_hits_total counter",
            'vllm:prefix_cache_hits_total{engine="0"} 10.0',
            'vllm:prefix_cache_hits_total{engine="1"} 20.0',
            'vllm:prefix_cache_hits_created{engine="0"} 999.0',
        ]
    )
    assert runner.counter(text, "vllm:prefix_cache_hits_total") == 30.0
    assert runner.counter(text, "missing") == 0.0


@pytest.mark.parametrize("status", [200, 500])
def test_http_uses_direct_connection_and_always_closes(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    events = []

    class Connection:
        def __init__(self, host: str, port: int, timeout: int) -> None:
            events.append((host, port, timeout))

        def request(self, method: str, path: str) -> None:
            events.append((method, path))

        def getresponse(self):
            class Response:
                def read(self):
                    return b"response"

            response = Response()
            response.status = status
            return response

        def close(self):
            events.append("closed")

    monkeypatch.setattr(runner, "HTTPConnection", Connection)
    if status == 200:
        assert runner.http(8011, "/metrics") == "response"
    else:
        with pytest.raises(RuntimeError, match="500"):
            runner.http(8011, "/metrics")
    assert events == [("127.0.0.1", 8011, 5), ("GET", "/metrics"), "closed"]
