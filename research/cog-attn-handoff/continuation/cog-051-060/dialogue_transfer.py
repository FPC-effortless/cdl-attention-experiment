"""Predict authentic user-turn deltas from text, and freeze gates on development."""
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('MULTIWOZ_DIR', '/workspace/scratch/multiwoz/data/MultiWOZ_2.2'))
SLOTS = ('hotel-area', 'hotel-pricerange')
HOLD = '<hold>'
EMPTY = '<empty>'


def corpus(split, hashes):
    rows = []
    for shard in sorted((SOURCE / split).glob('dialogues_*.json')):
        blob = shard.read_bytes()
        hashes[str(shard.relative_to(SOURCE))] = hashlib.sha256(blob).hexdigest()
        for dialogue in json.loads(blob):
            history = []
            prior = [EMPTY] * len(SLOTS)
            for turn in dialogue['turns']:
                utterance = turn['utterance'].lower()
                if turn['speaker'] == 'USER':
                    labels = {k: v for frame in turn['frames']
                              for k, v in frame['state']['slot_values'].items()}
                    state = [labels.get(slot, [EMPTY])[0] for slot in SLOTS]
                    delta = [v if v != old else HOLD for v, old in zip(state, prior)]
                    rows.append((dialogue['dialogue_id'], int(turn['turn_id']),
                                 [utterance, ' '.join(history[-2:])], delta))
                    prior = state
                history.append(utterance)
    return rows


def features(rows, ntrain):
    blocks = []
    vectors = []
    for field in (0, 1):
        docs = [r[2][field] for r in rows]
        for analyzer, grams in [('word', (1, 2)), ('char', (3, 5))]:
            vector = TfidfVectorizer(analyzer=analyzer, ngram_range=grams, min_df=5,
                                     max_features=32000, sublinear_tf=True)
            vector.fit(docs[:ntrain])
            blocks.append(vector.transform(docs))
            vectors.append((field, vector))
    return hstack(blocks, format='csr'), vectors


def classifier(x, y, xt):
    model = LogisticRegression(max_iter=250, C=8, solver='lbfgs')
    model.fit(x, y)
    return model.classes_, model.predict_proba(xt), model


def threshold(scores, predicted, truth):
    idx = np.argsort(-scores, kind='stable')
    count = np.arange(1, len(idx) + 1)
    correct = np.cumsum(predicted[idx] == truth[idx])
    complete_tie = np.r_[scores[idx][:-1] != scores[idx][1:], True]
    usable = np.flatnonzero((correct >= .99 * count) & complete_tie & (scores[idx] > 0))
    if not len(usable):
        return 2.
    return float(scores[idx][usable[np.argmax(correct[usable])]])


def metrics(score, pred, truth, threshold, groups):
    take = (score >= threshold) & (score > 0)
    ok = take & (pred == truth)
    positive = truth != HOLD
    num = np.bincount(groups, weights=ok)
    den = np.bincount(groups, weights=take, minlength=len(num))
    pos = np.bincount(groups, weights=positive, minlength=len(num))
    rng = np.random.default_rng(53056)
    ids = rng.integers(len(num), size=(1000, len(num)))
    def interval(bottom):
        a = num[ids].sum(axis=1)
        b = bottom[ids].sum(axis=1)
        return np.quantile(a[b > 0]/b[b > 0], [.025, .975]).tolist() if (b > 0).any() else []
    return dict(threshold=threshold, tp=int(ok.sum()), admitted=int(take.sum()),
                positives=int(positive.sum()), precision_ci=interval(den), recall_ci=interval(pos))


def main():
    hashes = {}
    train, dev, test = (corpus(split, hashes) for split in ('train', 'dev', 'test'))
    assert train and dev and test
    parts = [{r[0] for r in part} for part in (train, dev, test)]
    assert sum(map(len, parts)) == len(set.union(*parts))
    n, z = len(train), len(dev)
    rows = train + dev + test
    x, vectors = features(rows, n)
    fitted = {}
    groups = np.unique([r[0] for r in test], return_inverse=True)[1]
    out = dict(source=hashes, counts=[len(p) for p in (train, dev, test)], slots={})
    predictions = {}
    for slot, name in enumerate(SLOTS):
        y = np.array([r[3][slot] for r in rows])
        change = y[:n] != HOLD
        classes, prob, direct_model = classifier(x[:n], y[:n], x[n:])
        ix = prob.argmax(1)
        direct = (classes[ix], prob.max(1) * (classes[ix] != HOLD))
        event, event_p, event_model = classifier(x[:n], change, x[n:])
        values, val_p, value_model = classifier(x[:n][change], y[:n][change], x[n:])
        fitted[name] = (direct_model, event_model, value_model)
        scores = val_p.max(1) * event_p[:, list(event).index(True)]
        factored = (values[val_p.argmax(1)], scores * (scores > event_p[:, list(event).index(False)]))
        for arm, (pred, score) in [('direct', direct), ('factored', factored)]:
            t = threshold(score[:z], pred[:z], y[n:n+z])
            key = name + '/' + arm
            out['slots'][key] = {mode: metrics(score[z:], pred[z:], y[n+z:], gate, groups)
                                 for mode, gate in [('gated', t), ('unrestricted', 0.)]}
            predictions[key] = [[r[0], r[1], r[3][slot], p, float(s)]
                                for r, p, s in zip(test, pred[z:], score[z:])]
    saved = ROOT / 'checkpoints' / 'dialogue'
    saved.mkdir(parents=True, exist_ok=True)
    joblib.dump((vectors, fitted), saved / 'models.joblib', compress=3)
    (ROOT/'results/dialogue-transfer.json').write_text(json.dumps(out))
    (ROOT/'results/dialogue-predictions.json').write_text(json.dumps(predictions, separators=(',', ':')))


if __name__ == '__main__':
    main()
