"""Keep every paired run, including failures, in the system experiment summary."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prefix_summary", Path(__file__).resolve().parents[1] / "scripts/summarize_prefix_ablation.py"
)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def example_rows() -> list[dict]:
    return [
        {
            "length": length,
            "repeat": repeat,
            "cache": cache,
            "success": 80,
            "wall_tps": repeat * 100 * (2 if cache == "on" else 1),
            "ttft_p50_ms": 10 if cache == "on" else 20,
            "ttft_p99_ms": 30,
            "tpot_p50_ms": 5,
            "e2e_p99_ms": 100,
        }
        for length in (512, 1024, 1536)
        for repeat in (1, 2, 3)
        for cache in ("off", "on")
    ]


def test_summary_uses_medians_and_keeps_failed_pairs() -> None:
    rows = example_rows()
    rows[0]["success"] = 79
    rows[0]["wall_tps"] = 1
    aggregates, pairs = summary.summarize(rows)
    assert len(rows) == 18
    assert len(aggregates) == 3
    assert len(pairs) == 9
    assert aggregates[0]["off"]["wall_tps"] == 200
    assert aggregates[0]["off"]["wall_tps_range"] == [1, 300]
    assert aggregates[0]["wall_tps_change_percent"] == pytest.approx(100)
    assert aggregates[0]["ttft_p50_reduction_percent"] == pytest.approx(50)
    assert pairs[0]["both_runs_fully_successful"] is False
    assert pairs[0]["wall_tps_change_percent"] == pytest.approx(19900)
    assert all(p["both_runs_fully_successful"] for p in pairs[1:])


def test_counter_only_sums_the_requested_metric() -> None:
    text = 'hits_total{engine="0"} 10\nhits_total{engine="1"} 20\nhits_created 999\n'
    assert summary.metric(text, "hits_total") == 30
    assert summary.metric(text, "missing") == 0
