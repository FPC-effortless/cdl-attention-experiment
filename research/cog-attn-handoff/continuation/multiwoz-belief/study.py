"""Select on development and score the locked policy on official dialogue turns."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.sparse import hstack

HERE = Path(__file__).resolve().parent
BASE = Path(os.getenv('COG_CYCLE_DIR', HERE.parent/'cog-051-060'))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent/'delta-authority-048/recipe'))
import dialogue_transfer as original
from belief import DialogueState, EMPTY, HOLD

def read(path):
    return json.loads(path.read_bytes())

def save(path, data):
    path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n')

def observations(split):
    source = BASE/'checkpoints/dialogue/models.joblib'
    manifest = dict(line.split('  ')[::-1] for line in (BASE/'SHA256SUMS').read_text().splitlines())
    for path in (source, BASE/'dialogue_transfer.py', BASE/'results/dialogue-transfer.json'):
        assert sha256(path.read_bytes()).hexdigest() == manifest[str(path.relative_to(BASE))]
    store = BASE.parent/'delta-authority-048/recipe/state_store.py'
    assert sha256(store.read_bytes()).hexdigest() == read(BASE/'prior-inputs.json')['recipe/state_store.py']
    vectors, models = joblib.load(source)
    hashes = {}
    rows = original.corpus(split, hashes)
    report = read(BASE/'results/dialogue-transfer.json')
    assert hashes == {k:v for k,v in report['source'].items() if k.startswith(split+'/')}
    matrix = hstack([v.transform([r[2][field] for r in rows]) for field,v in vectors],format='csr')
    labels = {}
    thresholds = {}
    score = []
    for slot in original.SLOTS:
        model = models[slot][0]
        labels[slot] = model.classes_.tolist()
        score.append(model.predict_proba(matrix))
        thresholds[slot] = report['slots'][slot+'/direct']['gated']['threshold']
    grouped = defaultdict(list)
    for i, (dialogue, turn, _, delta) in enumerate(rows):
        grouped[dialogue].append((turn, [p[i] for p in score], delta))
    return grouped, labels, thresholds

def evaluate(data, config, keep=False):
    grouped, labels, thresholds = data
    counts = {}
    trace = []
    for dialogue, rows in grouped.items():
        runtime = DialogueState(labels, thresholds, **config)
        stats = Counter()
        truth = [EMPTY] * len(labels)
        previous = truth.copy()
        for turn, scores, delta in rows:
            patch = runtime.observe(turn, scores)
            verified = runtime.verified()
            estimates = runtime.estimates()
            values = [estimates[k][0] for k in labels]
            committed = [verified.get(k, EMPTY) for k in labels]
            truth = [old if d == HOLD else d for old,d in zip(truth,delta)]
            changed = any(d != HOLD for d in delta)
            match = values == truth
            correct_verified = committed == truth
            stats.update(turns=1, joint=int(match), changed=int(changed),
                         correct_changed=int(changed and match), stale=int(not changed and not match),
                         verified_joint=int(correct_verified), helpful=int(match and not correct_verified),
                         harmful=int(not match and correct_verified), diverged=int(values!=committed),
                         writes=len(patch), proposed_transitions=sum(a!=b for a,b in zip(previous, values)))
            for j, slot in enumerate(labels):
                good = slot in patch and (patch[slot] or EMPTY) == delta[j]
                stats['true_writes'] += int(good)
                stats['positives'] += int(delta[j]!=HOLD)
                mass = runtime.mass[j]
                # A held label outside the trained vocabulary receives zero predicted mass.
                true_mass = mass[labels[slot].index(truth[j])] if truth[j] in labels[slot] else 0.
                stats['brier_sum'] += float(1 + mass @ mass - 2 * true_mass)
            if keep:
                trace.append([dialogue,turn,delta,truth.copy(),patch,committed,values,
                              [round(estimates[k][2],8) for k in labels]])
            previous = values
        counts[dialogue] = dict(stats)
        runtime.close()
    total = Counter()
    for row in counts.values():
        total.update(row)
    total['slot_brier'] = total['brier_sum'] / (len(labels)*total['turns'])
    return dict(total=dict(total), per_dialogue=counts), trace

def select(out):
    data = observations('dev')
    choices = dict(soft=[dict(method='soft',power=p,hold_weight=w)
                         for p in (1,2) for w in (.25,.5,1,2)],
                   hard=[dict(method='hard',cutoff=t) for t in (0,.25,.5,.75)])
    table = {}
    winners = {}
    for kind, options in choices.items():
        attempts = []
        for config in options:
            result,_ = evaluate(data,config)
            attempts.append(dict(config=config, total=result['total']))
        best = max(attempts,key=lambda r:(r['total']['joint'], r['total']['correct_changed']))
        table[kind] = attempts
        winners[kind] = best['config']
    save(out/'selection.json', dict(selected=winners, development=table))

def compare(candidate, baseline):
    keys = sorted(baseline)
    assert set(keys)==set(candidate)
    differences = np.array([baseline[k]['stale']-candidate[k]['stale'] for k in keys])
    rng = np.random.default_rng(61051)
    draws = rng.choice(differences,(5000,len(keys)),replace=True).sum(axis=1)
    return dict(fewer_stale=int(differences.sum()), ci=np.quantile(draws,[.025,.975]).tolist())

def score(out, lock):
    data = observations('test')
    runs = dict(verified=dict(method='verified'), unrestricted=dict(method='hard',cutoff=0),
                unmodified_soft=dict(method='soft'), **read(lock)['selected'])
    metrics = {}
    traces = {}
    for arm, config in runs.items():
        metrics[arm], traces[arm] = evaluate(data,config,keep=True)
    origin = read(BASE/'results/dialogue-session.json')['arms']
    for arm, source in [('verified','direct/gated'),('unrestricted','direct/unrestricted')]:
        for dialogue, counts in metrics[arm]['per_dialogue'].items():
            old = origin[source]['dialogues'][dialogue]
            assert (counts['joint'], counts['stale']) == (old['joint_correct'], old['later_hold_wrong'])
    comparisons = {arm: {base:compare(m['per_dialogue'],metrics[base]['per_dialogue'])
                         for base in ('verified','unrestricted')} for arm,m in metrics.items()}
    assert all([r[:6] for r in rows] == [r[:6] for r in traces['verified']] for rows in traces.values())
    report = dict(selection_digest=sha256(lock.read_bytes()).hexdigest(), configs=runs, results=metrics,
                  comparison=comparisons, trace_columns=['dialogue','turn','gold_delta','gold_state',
                  'verified_patch','verified_state','read_state','read_confidence'])
    save(out/'results.json', report)
    save(out/'traces.json', traces)
    save(out/'metrics.json', {'joint':metrics['soft']['total']['joint']})

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['select','score'])
    parser.add_argument('--out',type=Path,default=Path(os.getenv('TRAINLAB_OUTPUT_DIR',HERE)))
    parser.add_argument('--lock',type=Path,default=HERE/'selection.json')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='select':
        select(args.out)
    else:
        score(args.out,args.lock)
