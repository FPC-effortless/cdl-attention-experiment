"""Reconcile saved final evidence; no model execution or candidate imports."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math

import torch


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def check(condition, description):
    if not condition:
        raise AssertionError(description)


def close(actual, expected, description, atol=1e-12):
    check(math.isclose(actual, expected, rel_tol=0, abs_tol=atol), description)


def review_panel(corpus, path):
    saved = read(path)
    labels = {(family['family_id'], side): world
              for family in corpus for side, world in enumerate(family['worlds'])}
    check(len(saved['rows']) == len(labels), f'{path}: world count')
    seen, pairs = set(), defaultdict(list)
    for row in saved['rows']:
        identity = (row['family_id'], row['side'])
        check(identity not in seen, f'{path}: duplicate identity')
        seen.add(identity)
        world = labels[identity]
        check(row['target'] == world['answer_text'], f'{path}: target')
        check(row['query'] == world['query'], f'{path}: question')
        ledger, op_count, ledgers_match = {}, 0, True
        check(len(row['predicted_operations']) == len(world['events']), f'{path}: event count')
        for index, (prediction, event) in enumerate(zip(row['predicted_operations'], world['events'])):
            action, entity, relation, value = prediction
            correct = action == event['operation']
            if action != 0:
                correct = correct and entity == event['entity'] and relation == event['relation']
            if action == 1:
                correct = correct and value == event['value']
                ledger[entity, relation] = value
            elif action == 2:
                ledger.pop((entity, relation), None)
            op_count += int(correct)
            serialized = [[e, r, v] for (e, r), v in sorted(ledger.items())]
            ledgers_match &= serialized == world['post_event_ledgers'][index]
        check(row['operations_exact'] == op_count, f'{path}: operation count')
        check(row['operations_total'] == len(world['events']), f'{path}: operation denominator')
        check(row['all_operations_exact'] == (op_count == len(world['events'])), f'{path}: operation flag')
        check(row['all_post_event_ledgers_exact'] == ledgers_match, f'{path}: ledger flag')
        check(row['final_ledger'] == serialized, f'{path}: final ledger')
        query_match = (row['predicted_query_entity'] == world['query_entity']
                       and row['predicted_query_plan'] == world['query_plan'])
        check(row['query_exact'] == query_match, f'{path}: query flag')
        plan = row['predicted_query_plan']
        key = (row['predicted_query_entity'], int(plan in (1, 2)))
        keys, values = [list(key)], [ledger.get(key, 16)]
        if plan == 2:
            if values[0] != 16:
                key = (values[0], 0)
                keys.append(list(key))
                values.append(ledger.get(key, 16))
            else:
                values.append(16)
        for name, answer in row['answers'].items():
            check(answer['correct'] == (answer['generated'] == world['answer_text']), f'{path}: exact answer')
            check(answer['support_exact'] == (answer['keys'] == world['supports']), f'{path}: support flag')
            if name in ('strict', 'acute_memory_off'):
                check(answer['keys'] == keys and answer['values'] == values, f'{path}: actual strict read')
                check(answer['status'] == ('missing' if values[-1] == 16 else 'ok'), f'{path}: strict status')
        pairs[row['family_id']].append(row)
    summary = saved['summary']
    check(summary['worlds'] == len(seen) and summary['families'] == len(pairs), f'{path}: denominators')
    for field in ('operations_exact', 'operations_total', 'all_operations_exact', 'all_post_event_ledgers_exact', 'query_exact'):
        check(summary[field] == sum(row[field] for row in saved['rows']), f'{path}: {field}')
    for mode, reported in summary['modes'].items():
        actual = {k: 0 for k in ('exact_answers', 'complete_pairs', 'strict_whole_history_pairs',
                                 'answer_changing_complete', 'answer_changing_total',
                                 'invariant_complete', 'invariant_total', 'support_exact')}
        by_kind = {}
        for pair in pairs.values():
            check(len(pair) == 2, f'{path}: pair membership')
            changing = pair[0]['target'] != pair[1]['target']
            correct_count = sum(r['answers'][mode]['correct'] for r in pair)
            complete = correct_count == 2
            whole = complete and all(r['all_operations_exact'] and r['all_post_event_ledgers_exact']
                                      and r['query_exact'] and r['answers'][mode]['support_exact'] for r in pair)
            actual['exact_answers'] += correct_count
            actual['complete_pairs'] += int(complete)
            actual['strict_whole_history_pairs'] += int(whole)
            actual['support_exact'] += sum(r['answers'][mode]['support_exact'] for r in pair)
            group = 'answer_changing' if changing else 'invariant'
            actual[group + '_total'] += 1
            actual[group + '_complete'] += int(complete)
            kind = by_kind.setdefault(pair[0]['kind'], {'pairs': 0, 'complete_pairs': 0, 'strict_pairs': 0, 'exact_answers': 0})
            kind['pairs'] += 1
            kind['complete_pairs'] += int(complete)
            kind['strict_pairs'] += int(whole)
            kind['exact_answers'] += correct_count
        for key, value in actual.items():
            check(reported[key] == value, f'{path}: aggregate {mode}/{key}')
        check(reported['by_kind'] == by_kind, f'{path}: task kinds')
        check(reported['max_candidates_per_query'] == max(row['answers'][mode]['candidates_fetched'] for row in saved['rows']),
              f'{path}: maximum candidate count')
    return summary


def main():
    torch.set_num_threads(1)
    review = {'date_utc': '2026-11-23', 'mode': 'saved evidence reconciliation only; no inference or fitting',
              'checks': {}, 'source_hashes': {}}
    for name in ('evidence/TRAINING_LOCK.json', 'evidence/composition/PROSPECTIVE_LOCK.json'):
        for path, expected in read(name)['source_sha256'].items():
            actual = sha(path)
            check(actual == expected, f'locked source changed: {path}')
            review['source_hashes'][path] = actual
    review['checks']['all_primary_and_numeric_locked_hashes_match'] = True
    corpus = read('evidence/corpus.json')
    panels = {}
    for arm in ('memory', 'no_memory'):
        for seed in (17, 29, 41):
            name = f'{arm}_seed{seed}'
            directory = f'evidence/evaluation/{name}'
            results = read(directory + '/RESULT.json')
            panels[name] = {}
            for panel in ('test', 'fresh', 'long'):
                summary = review_panel(corpus[panel], directory + '/' + panel + '.json')
                check(summary == results['panels'][panel], f'{name}/{panel}: top-level summary')
                panels[name][panel] = {mode: {field: stat[field] for field in
                    ('exact_answers', 'complete_pairs', 'strict_whole_history_pairs')}
                    for mode, stat in summary['modes'].items()}
            if arm == 'memory':
                intervention = read(directory + '/interventions.json')
                for row in intervention['rows']:
                    family = next(f for f in corpus['test'] if f['family_id'] == row['family_id'])
                    truth = [w['answer_text'] for w in family['worlds']]
                    check(row['targets'] == truth, f'{name}: intervention targets')
                    check(len(row['generated_by_state_origin']) == 2, f'{name}: both state origins')
                    correct = sum(output == truth for output in row['generated_by_state_origin'])
                    check(row['directional_pairs_correct'] == correct and row['both_correct'] == (correct == 2),
                          f'{name}: intervention counts')
    review['checks']['primary_panels'] = panels
    review['checks']['primary_rows_reconciled'] = 6 * (256 + 256 + 64)
    numeric = read('evidence/composition/RESULTS.json')
    durable = read('evidence/composition/DURABLE_RESULTS.json')
    protocol = read('evidence/composition/DURABLE_PROTOCOL.json')
    check(numeric['lock_sha256'] == sha('evidence/composition/PROSPECTIVE_LOCK.json'), 'numeric lock digest')
    check(numeric['protocol_sha256'] == sha('COMPOSITION_PROTOCOL.md'), 'numeric protocol digest')
    check(durable['protocol_sha256'] == sha('evidence/composition/DURABLE_PROTOCOL.json'), 'durable protocol digest')
    numeric_rows, durable_rows, midway, erased, rollback, max_difference = [], 0, 0, 0, 0, 0.0
    for result, durable_seed in zip(numeric['seeds'], durable['seeds']):
        seed = result['seed']
        check(seed == durable_seed['seed'], 'numeric/durable seed order')
        directory = f'evidence/composition/seed{seed}'
        check(result == read(directory + '/RESULT.json'), 'numeric per-seed result')
        check(durable_seed == read(directory + '/DURABLE_RESULT.json'), 'durable per-seed result')
        arrays = torch.load(ROOT / directory / 'heldout_outputs.pt', weights_only=True, map_location='cpu')
        inputs, target = arrays['inputs'], arrays['target']
        check(tuple(inputs.shape) == (4096, 2, 8), 'held numeric shape')
        check(torch.equal(inputs.mean(1), target), 'numeric target arithmetic')
        digest = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
        check(digest == numeric['test_input_sha256'], 'numeric held input hash')
        recomputed = {name: float((output - target).square().mean()) for name, output in arrays['outputs'].items()}
        for name, mse in recomputed.items():
            close(mse, result['mse'][name], f'{seed}/{name}: saved-output MSE')
        for arm, digest in result['checkpoint_sha256'].items():
            check(sha(directory + '/' + arm + '.pt') == digest, f'{seed}/{arm}: checkpoint hash')
        check(result['completed_updates_per_arm'] == 1000 and result['training_pairs_per_arm'] == 64000,
              'numeric fit budget')
        check(recomputed['connected'] < .002 and recomputed['reset_control'] / recomputed['connected'] >= 10
              and recomputed['acute_reset'] > .03, 'numeric loss gates')
        check(result['gradient']['finite'] and result['gradient']['abs_mean'] > 0, 'recorded first-input gradient gate')
        indices = [row['index'] for row in durable_seed['rows']]
        check(indices == protocol['first_indices'], 'prospective durable selection')
        for row in durable_seed['rows']:
            index = row['index']
            check(torch.equal(torch.tensor(row['input']), inputs[index]), 'durable actual input')
            check(torch.equal(torch.tensor(row['target']), target[index]), 'durable target')
            difference = float((torch.tensor(row['stored_output']) - arrays['outputs']['connected'][index]).abs().max())
            close(difference, row['maximum_absolute_output_difference'], 'durable output difference')
            check(difference <= protocol['neural_output_tolerance'], 'durable learned output parity')
            check(row['restart_between_reads'] == bool(index % 2), 'durable midread selection')
            check(row['intermediate_controller_equal'] == (True if index % 2 else None), 'recorded midread equality')
            check(row['postcommit_controller_equal'] and row['derived_value_survives_controller_reset']
                  and row['untouched_record_preserved'] and row['pass'], 'recorded durable gates')
            max_difference = max(max_difference, difference)
            midway += int(row['restart_between_reads'])
            durable_rows += 1
        check([(row['index'], row['source']) for row in durable_seed['source_erasures']] == [(0, 'a'), (15, 'b')],
              'source erasure selection')
        check(all(row['invalidated'] for row in durable_seed['source_erasures']), 'recorded erasure gates')
        check(durable_seed['before_commit_rollback'] and durable_seed['pass'], 'recorded rollback gate')
        erased += len(durable_seed['source_erasures'])
        rollback += int(durable_seed['before_commit_rollback'])
        numeric_rows.append({'seed': seed, 'mse_recomputed': recomputed,
                             'reset_to_connected_ratio': recomputed['reset_control'] / recomputed['connected']})
    review['checks']['numeric'] = numeric_rows
    review['checks']['numeric_paired_cpu_seconds_sum'] = sum(row['paired_cpu_seconds'] for row in numeric['seeds'])
    review['checks']['durable'] = {'reconciled_rows': durable_rows, 'midread_restores': midway,
        'postcommit_restores': durable_rows, 'controller_clear_then_reuse': durable_rows,
        'maximum_saved_neural_output_difference': max_difference, 'source_erasure_rejections': erased,
        'before_commit_rollbacks': rollback,
        'scope': 'input/output parity independently recomputed; restart/erasure/rollback checks reconciled from owner receipts, not rerun'}
    language = read('evidence/LANGUAGE_PRESERVATION.json')
    metrics = {name: value for name, value in language['references'].items()}
    metrics.update({run['name']: run['metrics'] for run in language['runs']})
    for name, metric in metrics.items():
        close(sum(row['nll_sum'] for row in metric['per_window']) / sum(row['targets'] for row in metric['per_window']),
              metric['mean_nll'], f'{name}: weighted language NLL')
    review['checks']['language_nll'] = {name: metric['mean_nll'] for name, metric in metrics.items()}
    interaction = read('evidence/INTERACTION_RESULTS.json')
    review['checks']['interaction'] = {}
    for seed, result in interaction.items():
        questions = [row for row in result['transcript'] if row['type'] == 'ask'
                     and row['result'].get('generation_mode') == 'neural']
        check(len(questions) == result['learned_question_count'], 'live question denominator')
        correct = sum(row['result']['text'] == row['expected'] for row in questions)
        check(correct == result['learned_questions_correct'], 'live exact count')
        review['checks']['interaction'][seed] = {'correct': correct, 'questions': len(questions),
            'historical_tokens_replayed': result['counters']['historical_tokens_replayed']}
    for path in ('fusion/full_model.py', 'fusion/composition.py', 'tests/test_full_model.py',
                 'review/FULL_MODEL_REVIEW.xml', 'evidence/composition/RESULTS.json',
                 'evidence/composition/DURABLE_RESULTS.json', 'RESULT.md', 'COMPOSITION_RESULT.md'):
        review['source_hashes'][path] = sha(path)
    review['pass'] = True
    path = ROOT / 'review/FINAL_EVIDENCE_REVIEW.json'
    path.write_text(json.dumps(review, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'pass': True, 'primary_rows_reconciled': review['checks']['primary_rows_reconciled'],
                      'numeric': numeric_rows, 'durable': review['checks']['durable'], 'receipt': str(path)}, indent=2))


if __name__ == '__main__':
    main()
