"""Critical comparison: a non-neural learner of exact pair equivalence."""
from .outcome_coder import controlled_pair_outcomes
from .symbolic_control import EquivalenceCodebook, run_control


def test_full_pair_graph_has_six_components_and_no_weights():
    vocabulary, outcomes = controlled_pair_outcomes()
    coder = EquivalenceCodebook(vocabulary, outcomes)
    assert coder.n_codes == 6
    assert coder.trainable_parameters == 0
    assert coder.pair_supervision_count == 276
    assert coder.code("owns") == coder.code("possesses")
    assert coder.code("owns") != coder.code("lives")
    assert coder.code("UNKNOWN") is None


def test_symbolic_control_recovers_all_gold_positions_with_same_task_stream():
    report = run_control(seeds=(101,102), lengths=(8,64,256,1024), trials=8)
    assert report["codebook_size"] == 6
    assert len(report["rows"]) == 4
    assert all(r["mean_recall_at_1"] == 1.0 for r in report["rows"])
