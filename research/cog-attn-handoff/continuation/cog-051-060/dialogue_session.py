"""Replay published prospective deltas through the original SQLite transaction module."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODULES = ROOT.parent/'delta-authority-048/recipe'
sys.path.insert(0, str(MODULES))
from state_store import World


def replay(rows, settings):
    totals = Counter()
    episodes = {}
    for (dialogue, turn), observations in sorted(rows.items()):
        if dialogue not in episodes:
            episodes[dialogue] = (World(), {}, Counter())
        db, target, local = episodes[dialogue]
        patch = {}
        for slot, (change, pred, score) in observations.items():
            if change != '<hold>':
                if change == '<empty>':
                    target.pop(slot, None)
                else:
                    target[slot] = change
            if score > 0 and score >= settings[slot]:
                patch[slot] = None if pred == '<empty>' else pred
        if patch:
            assert db.commit(str(turn), db.epoch(), patch) == 'committed'
        match = db.state() == target
        changed = any(v[0] != '<hold>' for v in observations.values())
        local['turns'] += 1
        local['joint_correct'] += match
        local['changed_turns'] += changed
        local['changed_correct'] += changed and match
        local['later_hold_wrong'] += not changed and not match
        local['writes'] += len(patch)
    per_dialogue = {}
    for dialogue, (db, _, local) in episodes.items():
        per_dialogue[dialogue] = local
        totals.update(local)
        db.db.close()
    return dict(total=totals, dialogues=per_dialogue)


def main():
    models = json.loads((RESULTS/'dialogue-transfer.json').read_text())['slots']
    observations = json.loads((RESULTS/'dialogue-predictions.json').read_text())
    slots = ('hotel-area', 'hotel-pricerange')
    outputs = {}
    for head in ('direct', 'factored'):
        rows = {}
        for slot in slots:
            for dialogue, turn, change, pred, p in observations[slot+'/'+head]:
                rows.setdefault((dialogue, turn), {})[slot] = change, pred, p
        assert all(set(turn) == set(slots) for turn in rows.values())
        for mode in ('gated', 'unrestricted'):
            settings = {slot: models[slot+'/'+head][mode]['threshold'] for slot in slots}
            outputs[head+'/'+mode] = replay(rows, settings)
    outputs['hold'] = replay(rows, dict.fromkeys(slots, 2))
    digest = hashlib.sha256((MODULES/'state_store.py').read_bytes()).hexdigest()
    expected = json.loads((ROOT/'prior-inputs.json').read_text())['recipe/state_store.py']
    assert digest == expected
    (RESULTS/'dialogue-session.json').write_text(json.dumps(dict(state_store=digest, arms=outputs)))


if __name__ == '__main__':
    main()
