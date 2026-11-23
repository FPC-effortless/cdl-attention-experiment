"""Independent data/provenance admission. No candidate inference or fitting."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.with_name('cog_fusion_001')
COUNTS = {'train': 2048, 'dev': 64, 'test': 128, 'fresh': 128, 'long': 32}
KINDS = ('direct', 'twohop', 'correction', 'irrelevant', 'retraction', 'bridge', 'missing', 'cancelled')
PARENT_HASHES = {
    17: '69180f8e9d05574eff19c78cf003f7e4aa21e5442e45a90ec2c09bffff553421',
    29: '5f9699aa14f44c2bbb30f176d9a1886767e971a48235d7e5dab435a13f6d7522',
    41: '5640ccce618b25c434b664934fc7e63986629cd6ec7bc5a790f785cfb7ee4332',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, description):
    if not condition:
        raise AssertionError(description)


def literal(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('missing literal ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT/'evidence/data')
    parser.add_argument('--tests', type=Path, default=ROOT/'evidence/pretrain_tests.xml')
    parser.add_argument('--output', type=Path, default=ROOT/'evidence/PRETRAIN_REVIEW.json')
    args = parser.parse_args()
    corpus = json.loads((args.data_root/'corpus.json').read_text())
    manifest = json.loads((args.data_root/'DATA_MANIFEST.json').read_text())
    check(set(corpus) == set(COUNTS), 'partition names')
    check(manifest['family_counts'] == COUNTS, 'declared family counts')
    check(sha(args.data_root/'corpus.json') == manifest['corpus_sha256'], 'corpus digest')
    entities, unknown = manifest['entities'], manifest['unknown']
    check(len(entities) == len(set(entities)) == 16 and unknown == 16, 'finite vocabulary')
    fillers = literal(PARENT/'fusion/data.py', 'FILLERS')
    banks = {key: manifest[key] for key in ('templates_train', 'templates_new_held', 'old_held_templates_excluded_from_train')}
    sets = {key: {text for group in bank.values() for text in group} for key, bank in banks.items()}
    for left, right in (('templates_train', 'templates_new_held'), ('templates_train', 'old_held_templates_excluded_from_train'),
                        ('templates_new_held', 'old_held_templates_excluded_from_train')):
        check(not sets[left] & sets[right], 'wording-bank overlap')
    parent_receipts = {}
    for seed, expected in PARENT_HASHES.items():
        path = PARENT/'evidence/runs'/f'memory_seed{seed}'/'final.pt'
        check(sha(path) == expected, 'parent checkpoint changed')
        saved = torch.load(path, weights_only=True, map_location='cpu')
        check(saved['memory_enabled'] is True and saved['config']['native_board_enabled'] is False, 'parent memory mode')
        check(saved['schema'] == 'cog-fusion-001-v1', 'parent schema')
        check(len(saved['chars']) == 65, 'parent vocabulary count')
        chars = set(saved['chars'])
        unique = {}
        for name, value in saved['model'].items():
            unique.setdefault((value.data_ptr(), value.numel(), tuple(value.stride())), (name, value.numel()))
        total = sum(size for name, size in unique.values())
        inactive = sum(size for name, size in unique.values() if name.startswith('native.record_board.'))
        check(total == 1077834 and inactive == 80160 and total-inactive == 997674, 'parent parameter counts')
        parent_receipts[str(seed)] = {'sha256': expected, 'stored_parameters': total, 'active_parameters': total-inactive}
    old_lock = json.loads((PARENT/'evidence/TRAINING_LOCK.json').read_text())
    for relative, expected in old_lock['source_sha256'].items():
        check(sha(PARENT/relative) == expected, '001 locked dependency changed: ' + relative)
    seen_semantics, seen_text = {}, {}
    summaries = {}
    for split, families in corpus.items():
        check(len(families) == COUNTS[split], 'actual family count')
        check(digest(families) == manifest['split_family_sha256'][split], 'split family digest')
        split_path = args.data_root/manifest['split_files'][split]['filename']
        check(sha(split_path) == manifest['split_files'][split]['sha256'], 'split file digest')
        payload = json.loads(split_path.read_text())
        check(payload['split'] == split and payload['families'] == families, 'separate split content')
        bank = banks['templates_new_held' if split == 'fresh' else 'templates_train']
        counts = Counter(families=len(families), worlds=2*len(families))
        kind_counts = Counter()
        turn_plans, turn_entities, turn_lengths = [set() for _ in range(3)], [set() for _ in range(3)], [set() for _ in range(3)]
        family_ids = set()
        for family in families:
            fid = family['family_id']
            check(fid not in family_ids, 'duplicate family ID')
            family_ids.add(fid)
            check(family['kind'] in KINDS, 'unknown family kind')
            kind_counts[family['kind']] += 1
            check(len(family['worlds']) == 2, 'paired world count')
            paired_text, answers_pair = [], []
            for world in family['worlds']:
                ledger, semantic, texts, answers = {}, [], [], []
                observations, questions, seen_answer, previous = 0, 0, False, None
                ask_positions = []
                for step in world['steps']:
                    check(set(step['text']) <= chars, 'unsupported input character')
                    texts.append((step['type'], step['text']))
                    if step['type'] == 'observe':
                        observations += 1
                        counts['observations'] += 1
                        op, entity, relation, value = (step[k] for k in ('operation', 'entity', 'relation', 'value'))
                        check(op in (0, 1, 2) and 0 <= entity < 16 and relation in (0, 1) and 0 <= value <= 16, 'typed observation range')
                        if op == 1:
                            ledger[entity, relation] = value
                        elif op == 2:
                            ledger.pop((entity, relation), None)
                        if op:
                            semantic.append(('observe', op, entity, relation, value if op == 1 else 16))
                        expected_ledger = [[e, r, v] for (e, r), v in sorted(ledger.items())]
                        check(step['post_event_ledger'] == expected_ledger, 'post-event ledger label')
                        typ, index = step['template_family'], step['template_index']
                        if typ == 'neutral':
                            expected_text = fillers[index]
                            check((op, entity, relation, value) == (0, 0, 0, 16), 'neutral operation')
                        else:
                            expected_text = bank[typ][index].format(e=entities[entity], v='unknown' if value == 16 else entities[value])
                        check(step['text'] == expected_text, 'observation wording-bank mismatch')
                        if seen_answer:
                            counts['post_answer_observations'] += 1
                            counts['meaningful_post_answer_observations'] += int(op != 0)
                        if previous == 'ask':
                            counts['immediate_post_answer_observations'] += 1
                            counts['meaningful_immediate_post_answer_observations'] += int(op != 0)
                    elif step['type'] == 'ask':
                        entity, plan = step['query_entity'], step['query_plan']
                        check(0 <= entity < 16 and plan in (0, 1, 2), 'query range')
                        semantic.append(('ask', entity, plan))
                        key = (entity, int(plan in (1, 2)))
                        keys, values = [list(key)], [ledger.get(key, 16)]
                        if plan == 2:
                            if values[0] != 16:
                                key = (values[0], 0)
                                keys.append(list(key))
                                values.append(ledger.get(key, 16))
                            else:
                                values.append(16)
                        value = values[-1]
                        answer = ('unknown' if value == 16 else entities[value]) + '.'
                        check((step['answer'], step['answer_text'], step['supports'], step['evidence_values']) == (value, answer, keys, values), 'query/support/answer label')
                        check(step['current_ledger'] == [[e, r, v] for (e, r), v in sorted(ledger.items())], 'query ledger label')
                        typ = ('q_owner', 'q_container', 'q_twohop')[plan]
                        check(step['template_family'] == typ and step['text'] == step['query'] == bank[typ][step['template_index']].format(e=entities[entity]), 'query wording')
                        check(len(answer) <= 12 and set(answer) <= chars, 'answer interface adequacy')
                        turn_plans[questions].add(plan)
                        turn_entities[questions].add(entity)
                        turn_lengths[questions].add(len(answer))
                        questions += 1
                        counts['answers'] += 1
                        answers.append(answer)
                        ask_positions.append(observations)
                        seen_answer = True
                    else:
                        raise AssertionError('unsupported step type')
                    previous = step['type']
                expected_observations = 72 if split == 'long' else 8
                expected_positions = [3, 37, 72] if split == 'long' else [3, 5, 8]
                check(observations == expected_observations and questions == 3 and ask_positions == expected_positions, 'history geometry')
                check(world['answers'] == answers, 'world answer list')
                check(world['final_ledger'] == [[e, r, v] for (e, r), v in sorted(ledger.items())], 'final ledger')
                for signature, seen in ((digest(semantic), seen_semantics), (digest(texts), seen_text)):
                    check(signature not in seen or seen[signature] == (split, fid), 'cross-family/split duplicate')
                    seen[signature] = (split, fid)
                paired_text.append(digest(texts))
                answers_pair.append(answers)
            check(paired_text[0] != paired_text[1], 'identical input-world pair')
            counts['session_pairs_with_any_answer_change'] += int(answers_pair[0] != answers_pair[1])
            counts['question_pairs_answer_changing'] += sum(a != b for a, b in zip(*answers_pair))
        check(kind_counts == Counter({kind: len(families)//8 for kind in KINDS}), 'kind balance')
        counts['question_pairs_invariant'] = len(families)*3-counts['question_pairs_answer_changing']
        counts['session_pairs_with_all_answers_invariant'] = len(families)-counts['session_pairs_with_any_answer_change']
        for key, value in manifest['paired_outcomes'][split].items():
            check(counts[key] == value, 'paired-outcome denominator')
        if split != 'long':
            check(counts['observations'] == len(families)*16 and counts['answers'] == len(families)*6,
                  'normal event denominators')
            check(counts['post_answer_observations'] == len(families)*10 and counts['immediate_post_answer_observations'] == len(families)*4,
                  'normal post-answer denominators')
        if split == 'train':
            check(all(values == {0, 1, 2} for values in turn_plans), 'TRAIN plan coverage')
            check(all(values == set(range(8, 16)) for values in turn_entities), 'TRAIN queried object coverage')
            check(all(values == {4, 5, 6, 8} for values in turn_lengths), 'TRAIN answer length coverage')
        summaries[split] = {**dict(counts), 'kind_counts': dict(kind_counts),
            'plans_by_turn': [sorted(v) for v in turn_plans], 'query_entities_by_turn': [sorted(v) for v in turn_entities],
            'answer_lengths_by_turn': [sorted(v) for v in turn_lengths]}
    probe_path = args.data_root/'train_probe.json'
    probe = json.loads(probe_path.read_text())
    check(probe['source_split'] == 'train' and probe['source_indices'] == list(range(128)), 'probe selection')
    check(probe['families'] == corpus['train'][:128], 'probe content')
    check(Counter(f['kind'] for f in probe['families']) == Counter({kind: 16 for kind in KINDS}), 'probe balance')
    check(sha(probe_path) == manifest['split_files']['train_probe']['sha256'], 'probe digest')
    tests = ET.parse(args.tests).getroot()
    suites = [tests] if tests.tag == 'testsuite' else list(tests.findall('testsuite'))
    totals = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    check(totals['tests'] >= 15 and all(totals[k] == 0 for k in ('failures', 'errors', 'skipped')), 'engineering tests incomplete or failed')
    calibration = json.loads((ROOT/'evidence/CALIBRATION_TIMING.json').read_text())
    check(calibration['fit_updates'] == 0 and all(r['finite_gradients'] for r in calibration['rows']), 'resource calibration policy')
    paths = [ROOT/'PROTOCOL.md', ROOT/'common.py', ROOT/'data.py', ROOT/'train.py', ROOT/'evaluate.py',
             ROOT/'run_campaign.py', ROOT/'runtime_eval.py', *sorted((ROOT/'tests').glob('*.py')),
             Path(__file__), args.tests, args.data_root/'DATA_MANIFEST.json', args.data_root/'corpus.json',
             args.data_root/'train.json', probe_path, ROOT/'evidence/CALIBRATION_TIMING.json']
    receipt = {'schema': 'cog-fusion-002-pretrain-review-v1', 'time_utc': datetime.now(timezone.utc).isoformat(),
        'pass': True, 'scope': 'independent label/partition/provenance replay and reconciled engineering receipts; no held inference or fitting',
        'data': summaries, 'parent_models': parent_receipts, 'engineering_tests': totals,
        'old_locked_dependencies_unchanged': True, 'train_probe_first_128_balanced': True,
        'duplicate_or_identical_pair_failures': 0, 'label_failures': 0,
        'source_sha256': {str(path.relative_to(ROOT.parent)): sha(path) for path in paths},
        'limits': ['data use fixed identity vocabulary and eight structural kinds',
                   'TRAIN probe is an adequacy diagnostic, not a generalization panel',
                   'template/index semantic replay validates constructed synthetic intent, not unrestricted language truth']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'pass': True, 'output': str(args.output), 'engineering_tests': totals,
                      'counts': {k: {f: v[f] for f in ('families', 'worlds', 'observations', 'answers')} for k, v in summaries.items()}}, indent=2))


if __name__ == '__main__':
    main()
