from __future__ import annotations

from casm.benchmark_harness_v2 import (
    BALANCED_GRAPH_SEED,
    TASK_SEED,
    apply_control_ratios,
    exact_balanced_graphs,
    task_groups,
)


def test_task_manifest_is_deterministic():
    a = task_groups(TASK_SEED, 4, True)
    b = task_groups(TASK_SEED, 4, True)
    assert a.keys() == b.keys()
    for key in a:
        assert [(x.text, x.answer) for x in a[key]] == [(x.text, x.answer) for x in b[key]]


def test_balanced_graph_manifest_is_exactly_balanced():
    for hard in (False, True):
        rows = exact_balanced_graphs(BALANCED_GRAPH_SEED, hard, 11)
        labels = [x.answer for x in rows]
        assert labels.count("yes") == 11
        assert labels.count("no") == 11


def test_state_tracking_manifest_exposes_initial_state():
    rows = task_groups(TASK_SEED, 32, True)["state_process"]
    assert rows
    for ex in rows:
        assert "initial " in ex.text.lower()
        assert "answer " in ex.text.lower()


def test_control_ratios_are_applied_by_sequence_length():
    models = {
        "control": {"efficiency": [
            {"seq_len": 96, "prefill_tokens_per_second": 10.0, "decode_tokens_per_second": 5.0},
            {"seq_len": 192, "prefill_tokens_per_second": 20.0, "decode_tokens_per_second": 4.0},
        ]},
        "candidate": {"efficiency": [
            {"seq_len": 96, "prefill_tokens_per_second": 15.0, "decode_tokens_per_second": 10.0},
            {"seq_len": 192, "prefill_tokens_per_second": 10.0, "decode_tokens_per_second": 8.0},
        ]},
    }
    apply_control_ratios(models, "control")
    rows = models["candidate"]["efficiency"]
    assert rows[0]["prefill_vs_control"] == 1.5
    assert rows[0]["decode_vs_control"] == 2.0
    assert rows[1]["prefill_vs_control"] == 0.5
    assert rows[1]["decode_vs_control"] == 2.0
