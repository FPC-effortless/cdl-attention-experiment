"""Independent saved-evidence reconciliation; no Torch or study helpers imported."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def trace(path, *, partial=False):
    rows, tail = [], 0
    lines = Path(path).read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            check(partial and index == len(lines) - 1 and not line.endswith((b'\n', b'\r')),
                  'malformed trace line')
            tail = len(line)
            break
        check(row['step'] == len(rows) + 1, 'noncontiguous trace')
        check(all(math.isfinite(v) for v in row.values() if type(v) in (int, float)), 'nonfinite trace value')
        check(0 <= row['cpu_seconds'] < 3600 and row['wall_seconds'] >= 0, 'invalid trace clock')
        if rows:
            check(all(row[k] >= rows[-1][k] for k in ('cpu_seconds', 'wall_seconds')), 'nonmonotonic clock')
        rows.append(row)
    check(bool(rows), 'empty trace')
    return rows, tail


def supplied_counts(families):
    worlds = [world for family in families for world in family['worlds']]
    batch, counts = len(worlds), Counter()
    previous, seen_answer = None, False
    for position in range(len(worlds[0]['steps'])):
        steps = [world['steps'][position] for world in worlds]
        kind = steps[0]['type']
        check(all(step['type'] == kind for step in steps), 'batch schedules differ')
        lengths = [len(step['text']) + 1 for step in steps]
        counts.update(input_chars=sum(lengths), input_physical_positions=batch * max(lengths),
                      semantic_encoder_calls=1)
        if kind == 'observe':
            counts['write_total'] += batch
            if seen_answer:
                counts['post_answer_write_total'] += batch
            if previous == 'ask':
                counts['first_post_answer_write_total'] += batch
        else:
            check(kind == 'ask', 'unsupported step type')
            lengths = [len(step['answer_text']) for step in steps]
            teacher_lengths = [len('\nAnswer: ') + n - 1 for n in lengths]
            counts.update(answer_targets=sum(lengths), teacher_decode_chars=sum(teacher_lengths),
                          teacher_decode_physical_positions=batch * max(teacher_lengths),
                          generation_prefix_chars=batch * len('\nAnswer: '), query_total=batch,
                          generated_answer_total=batch, teacher_decoder_calls=1)
            seen_answer = True
        previous = kind
    counts.update(rehearsal_targets=512, rehearsal_physical_positions=512, rehearsal_calls=1)
    return counts


def reconcile_trace(rows, seed, arm, families, lm_length):
    rng, totals, expected_totals, counts = random.Random(20090017 + seed), Counter(), Counter(), Counter()
    batch_hash, family_hash, lm_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for row in rows:
        indices = [rng.randrange(len(families)) for _ in range(4)]
        windows = [rng.randrange(lm_length - 129) for _ in range(4)]
        batch_hash.update(json.dumps([indices, windows]).encode())
        family_hash.update(json.dumps(indices).encode())
        lm_hash.update(json.dumps(windows).encode())
        counts.update(indices)
        expected = supplied_counts([families[index] for index in indices])
        check(all(type(row.get(key)) is int and row[key] == value for key, value in expected.items()),
              'supplied input/target/padding count differs')
        check(row['teacher_typed_execution'] is (row['step'] <= 100), 'typed warmup differs')
        check(row['answer_boundaries_reset'] == (24 if arm == 'reset_training' else 0), 'reset policy differs')
        generated, calls = row['generated_chars'], row['generation_decoder_calls']
        check(type(generated) is int and type(calls) is int, 'dynamic counts are not integers')
        maxima = calls - 3
        check(3 <= maxima <= 36 and maxima + 21 <= generated <= 8 * maxima, 'rollout count bounds differ')
        physical = sum(expected[key] for key in ('input_physical_positions', 'teacher_decode_physical_positions',
                       'generation_prefix_chars', 'rehearsal_physical_positions')) + generated
        check(row['physical_token_positions'] == physical, 'physical position sum differs')
        for exact, total in (('write_exact', 'write_total'), ('post_answer_write_exact', 'post_answer_write_total'),
                             ('first_post_answer_write_exact', 'first_post_answer_write_total'),
                             ('query_exact', 'query_total'), ('generated_answer_exact', 'generated_answer_total')):
            check(type(row[exact]) is int and 0 <= row[exact] <= row[total], 'logged accuracy count out of bounds')
        totals.update({key: value for key, value in row.items() if type(value) is int and key != 'step'})
        expected_totals.update(expected)
    return {'totals': dict(totals), 'supplied_totals': dict(expected_totals),
            'batch_trace_sha256': batch_hash.hexdigest(), 'family_draw_sha256': family_hash.hexdigest(),
            'lm_start_sha256': lm_hash.hexdigest(), 'distinct_training_families_seen': len(counts),
            'full_family_sample_counts': [counts[index] for index in range(len(families))],
            'probe_sampled_families': sum(index in counts for index in range(128)),
            'probe_unsampled_families': sum(index not in counts for index in range(128))}


def main():
    output = ROOT / 'review/INCOMPLETE_EVIDENCE_RECONCILIATION.json'
    check(not output.exists(), 'review output already exists')
    audit_path = ROOT / 'evidence/INCOMPLETE_CAMPAIGN_AUDIT.json'
    report = read(audit_path)
    check(report['campaign_complete'] is False and report['campaign_status'] == 'incomplete'
          and report['qualification_status'] == 'not_evaluated_incomplete_campaign', 'campaign scope changed')
    check(report['audit_status'] == 'completed' and report['individual_completed_fit_integrity_pass'],
          'individual integrity audit did not complete')
    check(report['model_inference_calls'] == 0 and report['probe_evaluations'] == 0
          and report['held_data_files_read'] == [], 'audit inference/data scope differs')
    for path, expected in report['input_sha256'].items():
        check(sha(RESEARCH / path) == expected, 'bound input hash differs: ' + path)
    lock = read(ROOT / 'evidence/TRAINING_LOCK.json')
    check(len(lock['source_sha256']) == report['locked_hashes_verified'] == 22, 'lock count differs')
    check(all(sha(RESEARCH / path) == value for path, value in lock['source_sha256'].items()), 'locked bytes differ')
    check(sha(ROOT / 'summarize.py') == 'dbe5be3c7aba00b257022f0f38239eb4b5d44125e2c5b718ee684eaa32d74392',
          'frozen summary differs')
    exits = read(ROOT / 'evidence/CAMPAIGN_EXIT.json')
    names = {f"{row['arm']}_seed{row['seed']}" for row in exits if row['returncode'] == 0}
    check(len(names) == 5 and names == set(report['completed_fit_names']) == {r['name'] for r in report['runs']},
          'five-endpoint inventory differs')
    families = read(ROOT / 'evidence/data/train.json')['families']
    check(len(families) == 2048, 'TRAIN pool size differs')
    lm_length = (RESEARCH / 'cog_fusion_001/evidence/reproduction_inputs/train.bin').stat().st_size // 2
    total_counters, completed_cpu, partial_cpu, reviews = Counter(), 0.0, 0.0, []
    for entry in report['runs']:
        folder = ROOT / 'evidence/runs' / entry['name']
        result, identity = read(folder / 'RESULT.json'), read(folder / 'IDENTITY.json')
        check(all(result.get(key) == value for key, value in identity.items()), 'result identity differs')
        check(result['status'] == 'completed' and result['steps'] == entry['steps'] == 1200, 'endpoint budget differs')
        check(entry['checkpoint_sha256'] == result['checkpoint_sha256'] == sha(folder / 'final.pt'), 'checkpoint identity differs')
        check(entry['optimizer_sha256'] == sha(folder / 'optimizer.pt'), 'optimizer identity differs')
        rows, tail = trace(folder / 'train.jsonl')
        check(len(rows) == 1200 and tail == 0, 'complete trace geometry differs')
        replay = reconcile_trace(rows, entry['seed'], entry['arm'], families, lm_length)
        check(replay['totals'] == entry['counters'] == entry['accounting']['reported_integer_counter_sums'], 'counter totals differ')
        check(replay['supplied_totals'] == entry['accounting']['expected_supplied_counts'], 'supplied totals differ')
        check(all(result.get(key) == value for key, value in replay['totals'].items()), 'result totals differ')
        for key in ('batch_trace_sha256', 'distinct_training_families_seen'):
            check(result[key] == report['replays'][str(entry['seed'])][key] == replay[key], 'replayed ' + key + ' differs')
        for key in ('family_draw_sha256', 'lm_start_sha256', 'full_family_sample_counts'):
            check(entry['coverage'][key] == replay[key], 'exposure replay differs')
        for key in ('cpu_seconds', 'wall_seconds'):
            check(math.isfinite(result[key]) and result[key] >= rows[-1][key] and entry[key] == result[key], 'completion clock differs')
        tensor = entry['checkpoints']
        check(tensor['pass'] and tensor['saved_python_rng_matches_full_replay']
              and tensor['total_parameters'] == 1077834 and tensor['active_parameters'] == 997674
              and tensor['inactive_board_parameters'] == 80160 and tensor['active_optimizer_states'] == 85
              and tensor['optimizer_parameter_step_range'] == [1200, 1200]
              and tensor['inactive_board_bitwise_unchanged'] and tensor['inactive_board_optimizer_states'] == 0
              and tensor['inactive_board_parent_sha256'] == tensor['inactive_board_final_sha256'], 'tensor audit claims differ')
        total_counters.update(replay['totals']); completed_cpu += result['cpu_seconds']
        reviews.append({'name': entry['name'], 'rows_checked': len(rows), 'cpu_seconds': result['cpu_seconds'],
                        'physical_token_positions': replay['totals']['physical_token_positions'],
                        'distinct_train_families': replay['distinct_training_families_seen'],
                        'probe_sampled_families': replay['probe_sampled_families'],
                        'probe_unsampled_families': replay['probe_unsampled_families']})
    partial_reviews = []
    check({entry['attempt'] for entry in report['incomplete_attempts']} == {'original', 'permitted_retry'}, 'partial inventory differs')
    for entry in report['incomplete_attempts']:
        path = RESEARCH / entry['trace_used']
        check(sha(path) == entry['trace_sha256'], 'partial trace identity differs')
        rows, tail = trace(path, partial=True)
        check(len(rows) == entry['observed_updates'] and tail == entry['trailing_partial_line_bytes'], 'partial geometry differs')
        replay = reconcile_trace(rows, 17, 'continuous', families, lm_length)
        check(replay['totals'] == entry['counters_lower_bound'], 'partial counter totals differ')
        check(entry['cpu_seconds_lower_bound'] == rows[-1]['cpu_seconds'], 'partial CPU differs')
        for key in ('batch_trace_sha256', 'family_draw_sha256', 'lm_start_sha256', 'full_family_sample_counts'):
            check(entry['reconstructed_draws'][key] == replay[key], 'partial expected draw replay differs')
        original = entry['attempt'] == 'original'
        check(entry['exit_code'] == (-9 if original else None) and entry['os_termination_confirmed'] is original
              and entry['process_status'] == ('terminated_by_recorded_signal' if original else 'unknown'), 'process status claim differs')
        total_counters.update(replay['totals']); partial_cpu += rows[-1]['cpu_seconds']
        partial_reviews.append({'attempt': entry['attempt'], 'rows_checked': len(rows),
                                'cpu_seconds_lower_bound': rows[-1]['cpu_seconds'],
                                'physical_positions_lower_bound': replay['totals']['physical_token_positions'],
                                'process_status': entry['process_status']})
    for key, value in [('completed_training_loop_cpu_seconds', completed_cpu),
                       ('incomplete_attempt_training_loop_cpu_seconds_lower_bound', partial_cpu),
                       ('observed_training_loop_cpu_seconds_lower_bound', completed_cpu + partial_cpu)]:
        check(math.isclose(report['costs'][key], value, rel_tol=0, abs_tol=1e-8), 'CPU sum differs')
    check(dict(total_counters) == report['observed_counters_lower_bound'], 'combined observed counters differ')
    for before_after in ('held_artifacts_before', 'held_artifacts_after'):
        check(not any(report[before_after].values()), 'held artifacts recorded')
        check(not any((ROOT / 'evidence' / name).exists() for name in report[before_after]), 'held artifacts now exist')
    check(not (ROOT / 'evidence/RECOVERY_RESULT.json').exists(), 'recovery completion appeared')
    receipt = {'schema': 'cog-fusion-002-independent-incomplete-evidence-reconciliation-v1', 'pass': True,
               'reviewed_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
               'audited_report_sha256': sha(audit_path), 'bound_input_hashes_verified': len(report['input_sha256']),
               'locked_sources_verified': 22, 'completed_fit_reviews': reviews, 'incomplete_attempt_reviews': partial_reviews,
               'canonical_rows_checked': sum(r['rows_checked'] for r in reviews),
               'incomplete_rows_checked': sum(r['rows_checked'] for r in partial_reviews),
               'completed_training_loop_cpu_seconds': completed_cpu,
               'incomplete_training_loop_cpu_seconds_lower_bound': partial_cpu,
               'observed_training_loop_cpu_seconds_lower_bound': completed_cpu + partial_cpu,
               'observed_physical_token_positions_lower_bound': total_counters['physical_token_positions'],
               'campaign_complete': False, 'qualification_evaluated': False, 'model_inference_calls': 0,
               'method': 'independent JSON/log/hash and Python-draw replay; no imported study/audit helpers or Torch',
               'limits': ['Tensor checks were source-reviewed and their saved claims and bound payload hashes reconciled; this second review does not deserialize the weights again.',
                          'Unsaved generated training strings are not reconstructed; dynamic counts receive internal-consistency checks.',
                          'Missing control and terminal receipts leave retry process status unknown.',
                          'CPU totals exclude initialization and unreported work; partial counts are observed lower bounds.',
                          'Individual artifact integrity does not qualify the incomplete campaign or establish learned performance.']}
    with output.open('x') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(output), 'pass': True,
                      'rows_checked': receipt['canonical_rows_checked'] + receipt['incomplete_rows_checked'],
                      'cpu_lower_bound': completed_cpu + partial_cpu,
                      'physical_positions_lower_bound': total_counters['physical_token_positions']}))


if __name__ == '__main__':
    main()
