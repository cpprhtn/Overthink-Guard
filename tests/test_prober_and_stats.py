import json

import pytest

from overthink_guard.analysis import Probe, ProbeTracker, read_probe_answer
from overthink_guard.storage import ShadowStats


@pytest.mark.parametrize(
    ("continuation", "expected"),
    [("45}$.", "45"), (r"\frac{1}{2}}", r"\frac{1}{2}"), (" X = 3 }", "x=3"), ("45 and more", None), ("}", None)],
)
def test_read_probe_answer(continuation, expected):
    assert read_probe_answer(continuation) == expected


def test_tracker_needs_k_identical_non_empty_answers_in_a_row():
    tracker = ProbeTracker(converge_k=3)
    answers = ["24", "45", None, "45", "45", "45", "45"]
    fired = [tracker.add(Probe(400 * (i + 1), a, 0.2)) for i, a in enumerate(answers)]
    assert fired == [False, False, False, False, False, True, False]
    assert tracker.decision.at_tokens == 2400
    assert tracker.seconds == pytest.approx(1.4)


def record(tokens, tier0_stop, tier0_match, tier2=None, elapsed=10.0):
    return {
        "ts": 0,
        "model": "m",
        "thinking_tokens": tokens,
        "elapsed_seconds": elapsed,
        "tier0": {"stop_at": tier0_stop, "match": tier0_match},
        "tier2": tier2,
    }


def test_summary_aggregates_both_tiers(tmp_path):
    stats = ShadowStats(tmp_path / "shadow.jsonl")
    stats.add(record(1000, None, None, {"stop_at": 400, "match": True, "probes": 2, "probe_seconds": 0.5}))
    stats.add(record(1000, 900, False, {"stop_at": 800, "match": False, "probes": 2, "probe_seconds": 0.5}))
    stats.add(record(500, None, None))
    summary = stats.summary()
    assert summary["requests"] == 3
    assert summary["tier0"] == {
        "requests": 3,
        "would_stop": 1,
        "saved_tokens": 100,
        "saved_ratio": 0.04,
        "mismatch": 1,
        "unknown": 0,
    }
    assert summary["tier2"]["saved_ratio"] == 0.4
    assert summary["tier2"]["mismatch"] == 1
    assert summary["tier2"]["overhead_ratio"] == 0.05


def test_records_persist_and_reload_skipping_bad_lines(tmp_path):
    path = tmp_path / "nested" / "shadow.jsonl"
    ShadowStats(path).add(record(100, None, None))
    with path.open("a", encoding="utf-8") as f:
        f.write("not json\n")
    assert ShadowStats(path).summary()["requests"] == 1
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["thinking_tokens"] == 100


def test_memory_only_stats_write_nothing(tmp_path):
    stats = ShadowStats(None)
    stats.add(record(100, None, None))
    assert stats.summary()["requests"] == 1
    assert list(tmp_path.iterdir()) == []
