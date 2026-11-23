"""Apply the prospectively fixed gates to saved 002 evidence; no model calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from common import HERE, PARENT, SEEDS, ARMS, parent_checkpoint, sha256, write_json


BOOTSTRAP_SEED = 20029017
BOOTSTRAP_SAMPLES = 20000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_interrupted_trace(path):
    """Retain complete archived rows; only the final unfinished write may be partial."""
    rows, partial_bytes = [], 0
    lines = Path(path).read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            require(index == len(lines) - 1 and not line.endswith((b'\n', b'\r')),
                    'malformed complete or interior interrupted trace line')
            partial_bytes = len(line)
            break
        require(isinstance(row, dict) and row.get('step') == len(rows) + 1,
                'noncontiguous interrupted trace')
        require(all(type(row.get(key)) in (int, float) and math.isfinite(row[key])
                    for key in ('cpu_seconds', 'wall_seconds')), 'invalid interrupted trace clocks')
        require(0 <= row['cpu_seconds'] < 3600 and row['wall_seconds'] >= 0,
                'invalid interrupted trace clocks')
        require(all(type(row.get(key)) is int and row[key] >= 0
                    for key in ('generated_chars', 'physical_token_positions')),
                'invalid interrupted trace counters')
        if rows:
            require(row['cpu_seconds'] >= rows[-1]['cpu_seconds']
                    and row['wall_seconds'] >= rows[-1]['wall_seconds'],
                    'nonmonotonic interrupted trace clocks')
        rows.append(row)
    require(0 < len(rows) <= 1200, 'interrupted trace has no admissible completed rows')
    return rows, partial_bytes


def fraction(numerator, denominator):
    require(type(numerator) is int and type(denominator) is int and 0 <= numerator <= denominator and denominator > 0,
            'invalid exact count denominator')
    return Fraction(numerator, denominator)


def threshold(numerator, denominator, minimum):
    return fraction(numerator, denominator) >= Fraction(str(minimum))


def scalar_ratio_gate(observed, reference, allowed):
    """Compare the stored decimal values inclusively, avoiding subtractive error."""
    require(math.isfinite(observed) and math.isfinite(reference) and reference > 0, 'invalid NLL')
    ratio = Fraction(str(observed)) / Fraction(str(reference))
    return {'mean_nll': observed, 'reference_nll': reference,
            'relative_nll_change': float(ratio - 1), 'allowed_relative_nll_increase': allowed,
            'passed': ratio <= 1 + Fraction(str(allowed))}


def paired_family_bootstrap(continuous, reset, *, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """Resample families, retaining all correlated seed comparisons per family."""
    require(set(continuous) == set(reset) and len(continuous) == 3, 'three paired seeds required')
    seeds = sorted(continuous)
    identities = sorted(continuous[seeds[0]])
    require(bool(identities), 'empty paired family panel')
    for model_seed in seeds:
        require(set(continuous[model_seed]) == set(identities) == set(reset[model_seed]), 'family identity mismatch')
    matrix = np.array([[int(continuous[s][fid]) - int(reset[s][fid]) for fid in identities] for s in seeds], dtype=np.int8)
    per_seed = {str(s): float(Fraction(int(row.sum()), len(identities))) for s, row in zip(seeds, matrix)}
    exact_mean = Fraction(int(matrix.sum()), matrix.size)
    # Division by three occurs after summing all retained seed contrasts. No seed
    # is sampled as an additional independent family or selected for advantage.
    family_sums = matrix.sum(axis=0)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(identities), size=(samples, len(identities)))
    boot = family_sums[draws].sum(axis=1) / matrix.size
    low, high = np.quantile(boot, [0.025, 0.975]).tolist()
    conditions = {'mean_gain_at_least_10_points': exact_mean >= Fraction(1, 10),
                  'no_negative_seed_gain': bool((matrix.sum(axis=1) >= 0).all()),
                  'paired_family_interval_excludes_zero': low > 0}
    return {'unit': 'paired family; all three seed contrasts retained together',
            'families': len(identities), 'seeds': seeds, 'resamples': samples, 'rng_seed': seed,
            'mean_accuracy_gain': float(exact_mean), 'seed_accuracy_gains': per_seed,
            'percentile_95_interval': [low, high], 'conditions': conditions,
            'observed_effect_gate_passed': all(conditions.values()),
            'scope': 'conditional on the three fixed starting endpoints; seeds are not extra independent families'}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def operation_matches(predicted, reference):
    op, entity, relation, value = (reference[k] for k in ('operation', 'entity', 'relation', 'value'))
    return predicted[0] == op and (op == 0 or
            (list(predicted[1:3]) == [entity, relation] and (op != 1 or predicted[3] == value)))


def verify_panel(payload, families, *, checkpoint_sha256, data_sha256, reset_after_answer):
    """Recompute qualification counts from actual trace proposals and fixed labels."""
    require(payload['checkpoint_sha256'] == checkpoint_sha256, 'panel checkpoint identity')
    require(payload['data_sha256'] == data_sha256 and payload['family_sha256'] == canonical_digest(families), 'panel data identity')
    policy = payload['policy']
    require(policy['reader'] == 'strict' and policy['memory_enabled_override'] is None
            and policy['reset_after_answer'] is reset_after_answer, 'panel deployment policy')
    require(payload['parameters']['total'] == 1077834 and payload['parameters']['active'] == 997674
            and payload['neural_state_tensor_bytes'] == 10592, 'panel architecture budget')
    labels = {(f['family_id'], side): (f, world) for f in families for side, world in enumerate(f['worlds'])}
    require(len(payload['rows']) == len(labels), 'panel world denominator')
    counts = {name: 0 for name in ('answers_exact', 'answers_total', 'queries_total', 'query_support_joint_exact',
                                  'meaningful_post_answer_operations_exact', 'meaningful_post_answer_operations_total',
                                  'whole_worlds_exact', 'whole_session_pairs_exact')}
    pair_worlds, seen = defaultdict(list), set()
    for row in payload['rows']:
        identity = row['family_id'], row['side']
        require(identity in labels and identity not in seen, 'world identity mismatch or duplication')
        seen.add(identity)
        family, reference = labels[identity]
        require(row['kind'] == family['kind'] and len(row['steps']) == len(reference['steps']), 'world geometry')
        ledger, seen_answer, whole = {}, False, True
        local = {key: 0 for key in counts if key not in ('whole_worlds_exact', 'whole_session_pairs_exact')}
        for truth, actual in zip(reference['steps'], row['steps']):
            require((actual['type'], actual['text']) == (truth['type'], truth['text']), 'trace text changed')
            if truth['type'] == 'observe':
                predicted = actual['predicted_operation']
                require(len(predicted) == 4, 'operation shape')
                op, entity, relation, value = predicted
                if op == 1:
                    ledger[entity, relation] = value
                elif op == 2:
                    ledger.pop((entity, relation), None)
                serialized = [[e, r, v] for (e, r), v in sorted(ledger.items())]
                require(actual['post_event_ledger'] == serialized, 'trace ledger differs from actual proposed operations')
                exact = operation_matches(predicted, truth)
                ledger_exact = serialized == truth['post_event_ledger']
                require(actual['operation_exact'] == exact and actual['ledger_exact'] == ledger_exact, 'operation/ledger scoring flag')
                whole &= exact and ledger_exact
                if seen_answer and truth['operation'] != 0:
                    local['meaningful_post_answer_operations_total'] += 1
                    local['meaningful_post_answer_operations_exact'] += int(exact)
            else:
                entity, plan = actual['predicted_query_entity'], actual['predicted_query_plan']
                key = (entity, int(plan in (1, 2)))
                keys, values = [list(key)], [ledger.get(key, 16)]
                if plan == 2:
                    if values[0] != 16:
                        key = (values[0], 0)
                        keys.append(list(key)); values.append(ledger.get(key, 16))
                    else:
                        values.append(16)
                require(actual['selected_keys'] == keys and actual['selected_values'] == values, 'strict read differs from actual candidate ledger')
                query = entity == truth['query_entity'] and plan == truth['query_plan']
                support = keys == truth['supports'] and values == truth['evidence_values']
                answer = actual['generated'] == truth['answer_text']
                require(actual['query_exact'] == query and actual['support_exact'] == support
                        and actual['answer_exact'] == answer, 'question scoring flag')
                require(actual['state_reset_after_answer'] is reset_after_answer, 'actual reset policy')
                local['queries_total'] += 1
                local['query_support_joint_exact'] += int(query and support)
                local['answers_total'] += 1
                local['answers_exact'] += int(answer)
                whole &= query and support and answer
                seen_answer = True
        require(row['whole_world_exact'] == whole, 'complete world flag')
        for name, value in local.items():
            require(row[name] == value, 'world count differs: ' + name)
            counts[name] += value
        counts['whole_worlds_exact'] += int(whole)
        pair_worlds[row['family_id']].append(bool(whole))
    outcomes = {}
    for family in families:
        fid = family['family_id']
        require(len(pair_worlds[fid]) == 2, 'missing pair member')
        outcomes[fid] = all(pair_worlds[fid])
    counts['whole_session_pairs_exact'] = sum(outcomes.values())
    summary = payload['summary']
    for name, value in counts.items():
        require(summary[name] == value, 'aggregate count differs: ' + name)
    require(summary['worlds'] == len(labels) and summary['families'] == len(families), 'aggregate denominators')
    recorded = {row['family_id']: row['both_worlds_exact'] for row in payload['family_rows']}
    require(len(recorded) == len(payload['family_rows']) and recorded == outcomes, 'paired-family score flags')
    return {**counts, 'worlds': len(labels), 'families': len(families)}, outcomes


def adequacy(counts, completed):
    conditions = {
        'complete_finite_fit': bool(completed),
        'meaningful_postanswer_operations_99pct': threshold(counts['meaningful_post_answer_operations_exact'], counts['meaningful_post_answer_operations_total'], '.99'),
        'query_and_support_99pct': threshold(counts['query_support_joint_exact'], counts['queries_total'], '.99'),
        'generated_answers_95pct': threshold(counts['answers_exact'], counts['answers_total'], '.95'),
        'complete_worlds_90pct': threshold(counts['whole_worlds_exact'], counts['worlds'], '.90'),
    }
    return {'conditions': conditions, 'passed': all(conditions.values()),
            'scope': 'training-distribution adequacy on the fixed 128-family TRAIN-pool subset under assigned training state policy; pool families are not necessarily sampled during fitting, so this is not a pure memorization-fit gate'}


def absolute_gates(counts):
    conditions = {'answers_95pct': threshold(counts['answers_exact'], counts['answers_total'], '.95'),
                  'complete_pairs_95pct': threshold(counts['whole_session_pairs_exact'], counts['families'], '.95')}
    return {'conditions': conditions, 'passed': all(conditions.values())}


def require_completed_evaluation(exit_receipt, lock, lock_hash):
    expected = {f'{arm}_seed{seed}' for seed in SEEDS for arm in (*ARMS, 'parent')}
    rows = exit_receipt['workers']
    require(exit_receipt['status'] == 'completed'
            and exit_receipt['evaluation_lock_sha256'] == lock_hash,
            'evaluation campaign did not complete under its declared lock')
    require(len(rows) == len(expected) and {row['name'] for row in rows} == expected
            and all(type(row['returncode']) is int and row['returncode'] == 0 for row in rows),
            'every fixed endpoint requires a successful evaluation worker')
    require(len(lock['endpoints']) == len(expected) and {row['name'] for row in lock['endpoints']} == expected,
            'evaluation lock endpoint inventory differs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=HERE/'evidence')
    parser.add_argument('--output', type=Path, default=HERE/'evidence/QUALIFICATION.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'refusing to replace completed qualification')
    evidence = args.evidence
    consumed = {}
    def read(path):
        path = Path(path)
        consumed[str(path)] = sha256(path)
        return json.loads(path.read_text())
    lock = read(evidence/'TRAINING_LOCK.json')
    review = read(evidence/'PRETRAIN_REVIEW.json')
    require(review['pass'] is True, 'pretraining review not admitted')
    for relative, expected in lock['source_sha256'].items():
        require(sha256(HERE.parent/relative) == expected, 'locked dependency changed: ' + relative)
    evaluation_lock_path = evidence/'EVALUATION_LOCK.json'
    evaluation_lock = read(evaluation_lock_path)
    evaluation_lock_hash = sha256(evaluation_lock_path)
    require_completed_evaluation(read(evidence/'EVALUATION_EXIT.json'), evaluation_lock, evaluation_lock_hash)
    require(evaluation_lock['completion']['training_lock_sha256'] == sha256(evidence/'TRAINING_LOCK.json'),
            'evaluation used a different training lock')
    for field in ('source_sha256', 'data_sha256', 'legacy_lock_sha256'):
        for relative, expected in evaluation_lock[field].items():
            require(sha256(HERE.parent/relative) == expected, 'evaluation-locked input changed: ' + relative)
    require(evaluation_lock['source_sha256'].get(str(Path(__file__).resolve().relative_to(HERE.parent))) == sha256(Path(__file__)),
            'this summarizer was not bound by the evaluation lock')
    manifests = read(evidence/'data/DATA_MANIFEST.json')
    datasets = {}
    for split in ('train_probe', 'test', 'fresh', 'long', 'dev'):
        path = evidence/'data'/f'{split}.json'
        data = read(path)
        require(sha256(path) == manifests['split_files'][split]['sha256'], 'split manifest mismatch')
        datasets[split] = (data['families'], sha256(path))
    results, outcome_maps, fit_records = {}, {}, {}
    expected_counts = {'train_probe': 128, 'test': 128, 'fresh': 128, 'long': 32, 'dev': 64}
    for seed in SEEDS:
        expected_parent = sha256(parent_checkpoint(seed))
        require(lock['parent_sha256'][str(seed)] == expected_parent, 'frozen parent changed')
        for arm in (*ARMS, 'parent'):
            name = f'{arm}_seed{seed}'
            checkpoint_hash = expected_parent
            if arm != 'parent':
                directory = evidence/'runs'/name
                fit = read(directory/'RESULT.json')
                require(fit['status'] == 'completed' and fit['steps'] == 1200 and fit['teacher_steps'] == 100
                        and fit['seed'] == seed and fit['arm'] == arm and fit['parent_sha256'] == expected_parent,
                        'incomplete or unmatched fixed fit')
                checkpoint_hash = sha256(directory/'final.pt')
                require(fit['checkpoint_sha256'] == checkpoint_hash, 'final checkpoint hash mismatch')
                logs = [json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines() if line.strip()]
                consumed[str(directory/'train.jsonl')] = sha256(directory/'train.jsonl')
                require([row['step'] for row in logs] == list(range(1, 1201)), 'training update sequence')
                require(all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) for row in logs), 'nonfinite fit evidence')
                require(all(row['teacher_typed_execution'] == (row['step'] <= 100) for row in logs), 'typed warmup policy')
                fit_records[name] = fit
            worker = read(evidence/'evaluation'/name/'RESULT.json')
            require(worker['status'] == 'completed' and worker['checkpoint_sha256'] == checkpoint_hash
                    and worker['evaluation_lock_sha256'] == evaluation_lock_hash, 'endpoint evaluation incomplete or mismatched')
            require(set(worker['panel_sha256']) == set(expected_counts), 'endpoint omitted a fixed panel')
            for split, expected_hash in worker['panel_sha256'].items():
                require(sha256(evidence/'evaluation'/name/f'{split}.json') == expected_hash, 'completed panel changed')
            require(sha256(evidence/'regression'/name/'RESULT.json') == worker['regression_result_sha256'],
                    'completed old regression receipt changed')
            entry = {'arm': arm, 'seed': seed, 'checkpoint_sha256': checkpoint_hash, 'panels': {}}
            for split, (families, data_hash) in datasets.items():
                require(len(families) == expected_counts[split], 'fixed family count')
                path = evidence/'evaluation'/name/f'{split}.json'
                # DEV is optional and never participates in a qualification gate.
                if split == 'dev' and not path.exists():
                    continue
                reset = split == 'train_probe' and arm == 'reset_training'
                counts, outcomes = verify_panel(read(path), families, checkpoint_sha256=checkpoint_hash,
                                                 data_sha256=data_hash, reset_after_answer=reset)
                entry['panels'][split] = counts
                if split == 'test':
                    outcome_maps[name] = outcomes
            if arm != 'parent':
                entry['training_adequacy'] = adequacy(entry['panels']['train_probe'], True)
            results[name] = entry
    recovery = evaluation_lock['completion'].get('recovery')
    interrupted_costs = []
    if recovery is not None:
        for interrupted in recovery['recovered']:
            archived = (HERE/interrupted['failed_attempt']).resolve()
            archived.relative_to(HERE.resolve())
            log = archived/'train.jsonl'
            lines, partial_bytes = read_interrupted_trace(log)
            original_trace = interrupted['original_trace']
            require(original_trace['completed_updates'] == len(lines)
                    and original_trace['trailing_partial_line_bytes'] == partial_bytes
                    and original_trace['training_loop_cpu_seconds_lower_bound'] == lines[-1]['cpu_seconds']
                    and all(original_trace['reported_counters_lower_bound'].get(key, 0)
                            == sum(row[key] for row in lines)
                            for key in ('generated_chars', 'physical_token_positions')),
                    'archived trace differs from admitted recovery receipt')
            consumed[str(log)] = sha256(log)
            interrupted_costs.append({'seed': interrupted['seed'], 'arm': interrupted['arm'],
                'archived_attempt': str(archived.relative_to(HERE)), 'recorded_updates': len(lines),
                'trailing_partial_line_bytes': partial_bytes,
                'recorded_cpu_seconds_lower_bound': lines[-1]['cpu_seconds'],
                'recorded_generated_characters': sum(row.get('generated_chars', 0) for row in lines),
                'recorded_physical_token_positions': sum(row.get('physical_token_positions', 0) for row in lines),
                'scope': 'discarded interrupted attempt; additional real work, not extra training of the retained endpoint'})
    fairness = {}
    for seed in SEEDS:
        a, b = (fit_records[f'{arm}_seed{seed}'] for arm in ARMS)
        fields = ('initial_tensor_sha256', 'batch_trace_sha256', 'train_data_sha256', 'shakespeare_train_sha256',
                  'input_chars', 'answer_targets', 'teacher_decode_chars', 'rehearsal_targets', 'steps',
                  'native_lr', 'interface_lr', 'weight_decay', 'gradient_clip', 'batch_families')
        checks = {field: a[field] == b[field] for field in fields}
        require(all(checks.values()), 'matched-arm input/supervision mismatch')
        fairness[str(seed)] = {'matched_fields': checks, 'generated_characters': {arm: fit_records[f'{arm}_seed{seed}']['generated_chars'] for arm in ARMS},
                               'note': 'generated lengths and realized physical work may differ'}
    for seed in SEEDS:
        parent_counts = results[f'parent_seed{seed}']['panels']['test']
        for arm in ARMS:
            entry = results[f'{arm}_seed{seed}']
            counts = entry['panels']['test']
            change = fraction(counts['whole_session_pairs_exact'], counts['families']) - fraction(parent_counts['whole_session_pairs_exact'], parent_counts['families'])
            conditions = absolute_gates(counts)['conditions']
            conditions['whole_pair_gain_over_parent_10_points'] = change >= Fraction(1, 10)
            conditions['training_adequacy'] = entry['training_adequacy']['passed']
            entry['familiar_repair'] = {'conditions': conditions, 'whole_pair_gain_over_parent': float(change), 'passed': all(conditions.values())}
            entry['novel_wording'] = absolute_gates(entry['panels']['fresh'])
            entry['long_extrapolation'] = absolute_gates(entry['panels']['long'])
    effect = paired_family_bootstrap(
        {seed: outcome_maps[f'continuous_seed{seed}'] for seed in SEEDS},
        {seed: outcome_maps[f'reset_training_seed{seed}'] for seed in SEEDS})
    effect['all_six_training_adequacy_passed'] = all(results[f'{arm}_seed{s}']['training_adequacy']['passed'] for arm in ARMS for s in SEEDS)
    effect['qualified_regime_advantage'] = effect['observed_effect_gate_passed'] and effect['all_six_training_adequacy_passed']
    for seed in SEEDS:
        parent_path = evidence/'regression'/f'parent_seed{seed}'/'RESULT.json'
        original_path = PARENT/'evidence/evaluation'/f'memory_seed{seed}'/'RESULT.json'
        original = read(original_path)
        baseline = read(parent_path) if parent_path.exists() else original
        require(baseline['checkpoint_sha256'] == sha256(parent_checkpoint(seed)) and baseline['panels'] == original['panels'], 'reused old-parent receipt differs')
        for arm in ARMS:
            entry = results[f'{arm}_seed{seed}']
            observed = read(evidence/'regression'/f'{arm}_seed{seed}'/'RESULT.json')
            require(observed['checkpoint_sha256'] == entry['checkpoint_sha256'] and observed['corpus_sha256'] == original['corpus_sha256'], 'old regression provenance')
            conditions, comparisons = {}, {}
            for panel in ('test', 'fresh', 'long'):
                current, reference = observed['panels'][panel], baseline['panels'][panel]
                now, before = current['modes']['strict'], reference['modes']['strict']
                a = fraction(now['exact_answers'], current['worlds'])
                h = fraction(now['strict_whole_history_pairs'], current['families'])
                old_a = fraction(before['exact_answers'], reference['worlds'])
                old_h = fraction(before['strict_whole_history_pairs'], reference['families'])
                comparisons[panel] = {'answer_accuracy': float(a), 'whole_pair_accuracy': float(h),
                                      'answer_change_vs_parent': float(a-old_a), 'whole_pair_change_vs_parent': float(h-old_h)}
                if panel == 'fresh':
                    conditions['old_fresh_answer_drop_at_most_5_points'] = a >= old_a-Fraction(1, 20)
                    conditions['old_fresh_whole_pair_drop_at_most_5_points'] = h >= old_h-Fraction(1, 20)
                else:
                    conditions[f'old_{panel}_answer_floor'] = a >= (Fraction(99, 100) if panel == 'test' else Fraction(95, 100))
                    conditions[f'old_{panel}_whole_pair_floor'] = h >= Fraction(95, 100)
            entry['old_regressions'] = {'conditions': conditions, 'panels': comparisons, 'passed': all(conditions.values()),
                                        'parent_receipt_reused_sha256': sha256(original_path), 'scope': 'already-exposed regression panels'}
    language = read(evidence/'LANGUAGE_PRESERVATION.json')
    full_nll = language['references']['original_connected_cog']['mean_nll']
    parent_language = language['references']['paired_fused001_parents']
    language_discrepancies = []
    require(len(language['runs']) == 6, 'language endpoint count')
    for row in language['runs']:
        name = f"{row['arm']}_seed{row['seed']}"
        require(name in results and row['checkpoint_sha256'] == results[name]['checkpoint_sha256'], 'language checkpoint identity')
        nll = row['metrics']['mean_nll']
        windows = row['metrics']['per_window']
        weighted = sum(w['nll_sum'] for w in windows)/sum(w['targets'] for w in windows)
        require(math.isclose(nll, weighted, abs_tol=1e-12, rel_tol=0), 'language weighted NLL')
        parent_nll = parent_language[str(row['seed'])]['metrics']['mean_nll']
        gates = {'paired_parent_nondegradation': scalar_ratio_gate(nll, parent_nll, .05),
                 'original_full_recovery': scalar_ratio_gate(nll, full_nll, .10)}
        for key, gate in gates.items():
            if gate['passed'] != row[key]['passed']:
                language_discrepancies.append({'run': name, 'gate': key, 'locked_scorer_flag': row[key]['passed'],
                                              'inclusive_protocol_flag': gate['passed'], **gate})
        results[name]['language'] = gates
    continuous = [results[f'continuous_seed{s}'] for s in SEEDS]
    decisions = {
        'all_continuous_training_adequacy': all(r['training_adequacy']['passed'] for r in continuous),
        'all_continuous_familiar_repair': all(r['familiar_repair']['passed'] for r in continuous),
        'specific_regime_advantage_qualified': effect['qualified_regime_advantage'],
        'all_continuous_novel_wording_absolute_gates': all(r['novel_wording']['passed'] for r in continuous),
        'all_continuous_old_regressions': all(r['old_regressions']['passed'] for r in continuous),
        'all_continuous_paired_parent_language_preservation': all(r['language']['paired_parent_nondegradation']['passed'] for r in continuous),
        'all_continuous_original_full_language_recovery': all(r['language']['original_full_recovery']['passed'] for r in continuous),
    }
    decisions['targeted_familiar_repair_qualified'] = all(decisions[k] for k in (
        'all_continuous_familiar_repair', 'all_continuous_old_regressions', 'all_continuous_paired_parent_language_preservation'))
    decisions['broader_wording_repair_qualified'] = decisions['targeted_familiar_repair_qualified'] and decisions['all_continuous_novel_wording_absolute_gates']
    output = {'schema': 'cog-fusion-002-qualification-v1', 'protocol_sha256': sha256(HERE/'PROTOCOL.md'),
              'summarizer_sha256': sha256(Path(__file__)), 'source_receipt_sha256': consumed,
              'fairness': fairness, 'runs': results, 'paired_regime_effect': effect, 'decisions': decisions,
              'operational_recovery': {'receipt': recovery, 'interrupted_attempt_costs': interrupted_costs,
                  'successful_fit_cpu_seconds': sum(fit['cpu_seconds'] for fit in fit_records.values()),
                  'additional_recorded_interrupted_cpu_seconds_lower_bound': sum(row['recorded_cpu_seconds_lower_bound'] for row in interrupted_costs)},
              'language_gate_arithmetic': {'method': 'inclusive rational comparison of the stored decimal NLL values',
                  'locked_scorer_discrepancies': language_discrepancies},
              'limits': ['training adequacy is distinct from generalization',
                         'failed finite-budget optimization does not establish architecture impossibility',
                         'frozen-parent gains combine extra training data and regime; matched-arm contrast isolates carry training',
                         'no architecture-wide, attention-superiority or efficiency-superiority verdict']}
    write_json(args.output, output)
    print(json.dumps({'output': str(args.output), 'decisions': decisions,
                      'paired_regime_effect': effect, 'language_arithmetic_discrepancies': language_discrepancies}, indent=2))


if __name__ == '__main__':
    main()
