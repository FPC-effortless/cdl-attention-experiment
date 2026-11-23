"""Protocol checks that do not execute or inspect a trained candidate."""
import math

import numpy as np
import pytest
import torch

from fusion.preservation import completed_runs, fixed_windows, preservation_gate


def test_fixed_sampling_has_no_duplicate_scored_positions():
    values = np.arange(55770, dtype=np.uint16)
    starts, windows = fixed_windows(values)
    again_starts, again = fixed_windows(values)
    assert starts == again_starts and torch.equal(windows, again)
    assert windows.shape == (64, 129)
    positions = [position for start in starts for position in range(start + 1, start + 129)]
    assert len(positions) == len(set(positions)) == 8192
    assert min(positions) >= 1 and max(positions) < len(values)
    for row, start in enumerate(starts):
        assert windows[row, 0] == start and windows[row, -1] == start + 128


def test_gate_uses_relative_nll_and_keeps_perplexity_separate():
    # A 5% NLL increase at NLL2 is a >10% perplexity increase. The locked
    # criterion is NLL, so confusing these quantities changes the outcome.
    gate = preservation_gate(2.1, 2.0)
    assert gate["passed"]
    assert gate["relative_mean_nll_change"] == pytest.approx(.05)
    assert gate["character_perplexity_ratio"] == pytest.approx(math.exp(.1))
    assert preservation_gate(2.2, 2.)["passed"]
    assert not preservation_gate(2.21, 2.)["passed"]
    assert preservation_gate(1.9, 2.)["passed"]
    with pytest.raises(ValueError):
        preservation_gate(float("nan"), 2.)


def test_incomplete_campaign_is_rejected_before_model_evaluation(tmp_path):
    with pytest.raises(RuntimeError, match="incomplete"):
        completed_runs(tmp_path)
