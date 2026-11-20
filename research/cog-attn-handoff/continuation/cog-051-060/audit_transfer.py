"""Recount supplemental authentic predictions and validate recorded costs and checkpoints."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def load(name):
    return json.loads((RESULTS / name).read_text())


def main():
    counts = {}
    kitchen = load('kitchen.json')
    for name, arm in kitchen['arms'].items():
        rows = arm['rows']
        assert len({r[0] for r in rows}) == kitchen['validation'] == len(rows)
        take = [r for r in rows if r[3] >= arm['cutoff']]
        assert arm['all'] == dict(n=len(rows), correct=sum(r[1] == r[2] for r in rows),
                                  admitted=len(take), tp=sum(r[1] == r[2] for r in take))
        counts['kitchen_' + name] = len(rows)

    dialogue = load('dialogue-transfer.json')
    predictions = load('dialogue-predictions.json')
    for key, rows in predictions.items():
        assert len({(d, t) for d, t, *_ in rows}) == len(rows) == dialogue['counts'][2]
        for arm in dialogue['slots'][key].values():
            take = [r for r in rows if r[-1] > 0 and r[-1] >= arm['threshold']]
            assert (len(take), sum(r[2] == r[3] for r in take), sum(r[2] != '<hold>' for r in rows)) == (
                arm['admitted'], arm['tp'], arm['positives'])
    counts['dialogue_predictions'] = sum(map(len, predictions.values()))

    confirmed = load('authentic-session.json')
    arrivals = load('confirmation.json')['rows']
    assert len(arrivals) == confirmed['counts']['verification_arrivals']
    assert {r[2] for r in arrivals if r[5]} == {r[0] for r in confirmed['confirmed']}
    assert len(confirmed['confirmed']) == confirmed['counts']['committed']
    counts['authentic_commits'] = len(confirmed['confirmed'])

    for run in load('authentic-replays.json'):
        path = RESULTS / str(run['seed']) / 'issue-predictions.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run['matched_predictions']
    timings = load('online-cost.json')
    for row in timings:
        assert row['score_ns'] == row['build_ns'] + 12 * row['online_ns']
        assert row['wrong'] == 0
    counts['online_operations'] = sum(r['requests'] for r in timings)

    for seed in (5101, 5107, 5113):
        receipt = load(f'{seed}/acquisition-candidate/receipt.json')['artifacts']
        files = list((HERE / 'checkpoints' / str(seed)).glob('*'))
        assert {p.name for p in files} == {'csm.npz', 'roles.joblib', 'gru.pt', 'transformer.pt'}
        for path in files:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt[path.name]
    return counts


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
