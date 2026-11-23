"""Protocol-boundary and family-clustering tests; no learned model scoring."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize import (adequacy, canonical_digest, paired_family_bootstrap,
                       read_interrupted_trace, require_completed_evaluation,
                       scalar_ratio_gate, threshold, verify_panel)


def archived_row(step):
    import json
    return json.dumps({'step': step, 'cpu_seconds': step / 10, 'wall_seconds': step / 5,
                       'generated_chars': 7, 'physical_token_positions': 80}).encode()


@pytest.mark.parametrize('suffix,completed,partial', [
    (b'\n', 2, 0),
    (b'', 2, 0),
    (b'\n{"step":3,"cpu_seconds":', 2, len(b'{"step":3,"cpu_seconds":')),
])
def test_interrupted_trace_retains_only_verified_complete_rows(tmp_path, suffix, completed, partial):
    path = tmp_path / 'train.jsonl'
    path.write_bytes(archived_row(1) + b'\n' + archived_row(2) + suffix)
    rows, partial_bytes = read_interrupted_trace(path)
    assert len(rows) == completed and partial_bytes == partial
    assert sum(row['generated_chars'] for row in rows) == 14
    assert rows[-1]['cpu_seconds'] == .2


@pytest.mark.parametrize('body', [
    archived_row(1) + b'\n{"step":2\n',
    archived_row(1) + b'\n{"step":2\n' + archived_row(2),
    archived_row(1) + b'\n\n',
    b'{"step":1',
    archived_row(1) + b'\n' + archived_row(3),
])
def test_interrupted_trace_rejects_completed_or_interior_corruption(tmp_path, body):
    path = tmp_path / 'train.jsonl'
    path.write_bytes(body)
    with pytest.raises(ValueError):
        read_interrupted_trace(path)


def test_integer_thresholds_do_not_round_failed_counts_up():
    assert threshold(122, 128, '.95')
    assert not threshold(121, 128, '.95')
    assert threshold(99, 100, '.99')
    with pytest.raises(ValueError):
        threshold(0, 0, '.99')


def test_nll_gate_includes_exact_decimal_boundary():
    assert (2.1/2-1) > .05  # Demonstrates the locked scorer's subtraction risk.
    assert scalar_ratio_gate(2.1, 2., .05)['passed']
    assert not scalar_ratio_gate(2.1000001, 2., .05)['passed']
    assert scalar_ratio_gate(2.2, 2., .10)['passed']
    with pytest.raises(ValueError):
        scalar_ratio_gate(float('nan'), 2., .05)


def test_bootstrap_resamples_families_without_tripling_sample_size():
    continuous = {seed: {'a': True, 'b': True, 'c': False, 'd': False} for seed in (17, 29, 41)}
    reset = {seed: {'a': False, 'b': False, 'c': False, 'd': False} for seed in (17, 29, 41)}
    result = paired_family_bootstrap(continuous, reset)
    assert result['families'] == 4
    assert result['mean_accuracy_gain'] == .5
    assert result['percentile_95_interval'] == [0., 1.]
    assert not result['observed_effect_gate_passed']
    require_same = paired_family_bootstrap({s: dict(reversed(list(v.items()))) for s, v in continuous.items()}, reset)
    assert result == require_same
    del reset[17]['d']
    with pytest.raises(ValueError, match='family identity'):
        paired_family_bootstrap(continuous, reset)


def test_regime_gate_cannot_hide_a_negative_seed():
    continuous = {s: {str(i): s != 41 for i in range(128)} for s in (17, 29, 41)}
    reset = {s: {str(i): s == 41 for i in range(128)} for s in (17, 29, 41)}
    result = paired_family_bootstrap(continuous, reset)
    assert result['mean_accuracy_gain'] > .1
    assert result['percentile_95_interval'][0] > 0
    assert not result['conditions']['no_negative_seed_gain']
    assert not result['observed_effect_gate_passed']


def test_training_adequacy_requires_every_declared_condition():
    counts = {'meaningful_post_answer_operations_exact': 99, 'meaningful_post_answer_operations_total': 100,
              'query_support_joint_exact': 99, 'queries_total': 100, 'answers_exact': 95, 'answers_total': 100,
              'whole_worlds_exact': 90, 'worlds': 100}
    assert adequacy(counts, True)['passed']
    assert not adequacy(counts, False)['passed']
    assert not adequacy({**counts, 'whole_worlds_exact': 89}, True)['passed']


def test_independent_trace_reconciliation_accepts_real_schema_and_rejects_forged_reads():
    import copy
    from evaluate import evaluate_panel
    from test_evaluate_002 import OutputSensitiveFixture, fixture_family
    families = [fixture_family(with_hold=True)]
    for reset in (False, True):
        panel = evaluate_panel(OutputSensitiveFixture(), families, reset_after_answer=reset)
        panel.update(checkpoint_sha256='fixture-checkpoint', data_sha256='fixture-data',
                     family_sha256=canonical_digest(families), parameters={'total': 1077834, 'active': 997674},
                     neural_state_tensor_bytes=10592)
        counts, outcomes = verify_panel(panel, families, checkpoint_sha256='fixture-checkpoint',
                                        data_sha256='fixture-data', reset_after_answer=reset)
        assert counts['whole_session_pairs_exact'] == int(not reset)
        assert outcomes == {'fixture': not reset}
        altered = copy.deepcopy(panel)
        ask = next(step for step in altered['rows'][0]['steps'] if step['type'] == 'ask')
        ask['selected_values'] = [4]
        with pytest.raises(ValueError, match='strict read differs'):
            verify_panel(altered, families, checkpoint_sha256='fixture-checkpoint',
                         data_sha256='fixture-data', reset_after_answer=reset)


def test_panel_files_cannot_substitute_for_successful_endpoint_completion():
    names = [f'{arm}_seed{seed}' for seed in (17, 29, 41) for arm in ('continuous', 'reset_training', 'parent')]
    lock = {'endpoints': [{'name': name} for name in names]}
    receipt = {'status': 'completed', 'evaluation_lock_sha256': 'lock',
               'workers': [{'name': name, 'returncode': 0} for name in names]}
    require_completed_evaluation(receipt, lock, 'lock')
    with pytest.raises(ValueError, match='every fixed endpoint'):
        require_completed_evaluation({**receipt, 'workers': receipt['workers'][:-1]}, lock, 'lock')
    failed = [dict(row) for row in receipt['workers']]
    failed[0]['returncode'] = 1
    with pytest.raises(ValueError, match='every fixed endpoint'):
        require_completed_evaluation({**receipt, 'workers': failed}, lock, 'lock')
    with pytest.raises(ValueError, match='campaign did not complete'):
        require_completed_evaluation({**receipt, 'status': 'failed'}, lock, 'lock')
