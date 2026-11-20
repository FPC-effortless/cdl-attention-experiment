"""Independently reconstruct the original verified write stream and recount display errors."""
from collections import Counter
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/'cog-051-060/results'
SLOTS = ('hotel-area', 'hotel-pricerange')

def load(path):
    return json.loads(path.read_bytes())

def run():
    preds = load(PRIOR/'dialogue-predictions.json')
    gates = load(PRIOR/'dialogue-transfer.json')['slots']
    ledger = {}
    for slot in SLOTS:
        key = slot+'/direct'
        for dialogue, turn, gold, value, score in preds[key]:
            entry = ledger.setdefault((dialogue,turn), {'delta':[], 'patch':{}})
            entry['delta'].append(gold)
            if score>0 and score>=gates[key]['gated']['threshold']:
                entry['patch'][slot] = None if value=='<empty>' else value
    states = {}
    for key, entry in sorted(ledger.items()):
        state = states.setdefault(key[0],{})
        state.update(entry['patch'])
        entry['verified'] = [state.get(slot) or '<empty>' for slot in SLOTS]
    report = load(HERE/'results.json')
    arms = load(HERE/'traces.json')
    for arm, rows in arms.items():
        assert len(ledger) == len(rows) == len({(r[0],r[1]) for r in rows})
        truth = {}
        count = Counter()
        for dialogue, turn, delta, target, patch, committed, values, _ in rows:
            source = ledger[dialogue,turn]
            assert (delta,patch,committed) == (source['delta'],source['patch'],source['verified'])
            old = truth.get(dialogue,['<empty>']*2)
            truth[dialogue] = [a if d=='<hold>' else d for a,d in zip(old,delta)]
            assert target == truth[dialogue]
            count['joint'] += values==target
            count['stale'] += delta==['<hold>']*2 and values!=target
            count['writes'] += len(patch)
        assert all(report['results'][arm]['total'][k]==v for k,v in count.items())
    return dict(arms=len(arms), user_turns=len(ledger), verified_slot_writes=sum(len(v['patch']) for v in ledger.values()))

if __name__=='__main__':
    print(json.dumps(run()))
