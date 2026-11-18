"""Select contiguous expressions from text and bind their meanings from labeled examples."""
import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from roles import Reader, WORDS, V, choose, data, outcome, State

ROOT = Path(__file__).parent
SEEDS = (11, 23, 37)
TRAIN = [
    'for {a} and {b} use relation {p} and ignore relation {d}',
    'ignore relation {d} then use relation {p} for {a} and {b}',
    'compare {a} to {b} using {p} despite the note containing {d}',
    'the note contains {d} but {a} and {b} are compared using {p}',
]
TEST = {
    'familiar': TRAIN,
    'target_last': ['set aside relation {d} when comparing {a} and {b} using {p}'],
    'target_first': ['use relation {p} for {a} and {b} even if a note mentions {d}'],
    'contrast_reversed': ['the relation between {a} and {b} is {d} in the discarded draft but is {p} in the accepted draft'],
    'parallel_clauses': ['for {a} and {b} reject {d} as the relation and accept {p} as the relation'],
}
TRAIN_WORDS = 'dax wug tiv zom kef yul nar siv bel mun'.split()
TEST_WORDS = 'vor nes hal gix fep lun zep jas mok rin'.split()
SIZES = (1, 2, 3)
DEMONSTRATION = 'for {a} and {b} use relation {p} and ignore relation {d}'


def phrase(rng, vocabulary, length):
    return ' '.join(rng.sample(vocabulary, length))


def render(frame, p, d, a, b):
    return frame.format(p=p, d=d, a=a, b=b)


def spans(text):
    words = text.split()
    return [(start, start + size) for start in range(len(words))
            for size in SIZES if start + size <= len(words)]


def find(text, key):
    words = text.split()
    parts = key.split()
    fits = [(start, start + len(parts)) for start in range(len(words))
            if words[start:start + len(parts)] == parts]
    assert len(fits) == 1
    return fits[0]


def features(words, start, end):
    def word(index):
        if not 0 <= index < len(words):
            return '<edge>'
        return '<record>' if words[index] in WORDS else words[index]

    result = {
        'len=' + str(end - start): 1,
        'left=' + word(start - 1): 1,
        'left2=' + word(start - 2): 1,
        'right=' + word(end): 1,
        'right2=' + word(end + 1): 1,
        'both=' + word(start - 1) + '|' + word(end): 1,
        'records_left': sum(w in WORDS for w in words[:start]),
    }
    for token in words[start:end]:
        result['inside=' + token] = 1
    return result


def train(seed):
    rng = random.Random(seed)
    inputs, labels = [], []
    for _ in range(750):
        size = rng.choice(SIZES)
        tokens = rng.sample(TRAIN_WORDS, 2 * size)
        target, foil = ' '.join(tokens[:size]), ' '.join(tokens[size:])
        text = render(rng.choice(TRAIN), target, foil, *rng.sample(WORDS, 2))
        correct = find(text, target)
        for start, end in spans(text):
            inputs.append(features(text.split(), start, end))
            labels.append(int((start, end) == correct))
    vec = DictVectorizer()
    model = LogisticRegression(max_iter=250, class_weight='balanced', random_state=seed)
    model.fit(vec.fit_transform(inputs), labels)
    return vec, model


def freeze():
    panels = {}
    for ci, (name, frames) in enumerate(TEST.items()):
        rng = random.Random(34001 + ci)
        support, _, targets, _ = data(34200 + ci, 128, 'train')
        records = []
        for i, (s, t) in enumerate(zip(support, targets)):
            names = [next(w for w in WORDS if V[w] in s[2 * slot]) for slot in t]
            others = [next(w for w in WORDS if V[w] in s[2 * slot])
                      for slot in range(8) if slot not in t]
            size = SIZES[i % len(SIZES)]
            words = rng.sample(TEST_WORDS, 3 * size)
            keys = [' '.join(words[j:j + size]) for j in range(0, len(words), size)]
            orientation = i % 2
            frame = rng.choice(frames)
            queries = [render(frame, key, keys[2], *names) for key in keys[:2]]
            demonstrations = []
            for j, key in enumerate(keys):
                a, b = others[j * 2:j * 2 + 2]
                distractor = keys[(j + 1) % 3]
                output = (a, b)[(j % 2) ^ orientation]
                demonstrations.append(render(DEMONSTRATION, key, distractor, a, b)
                                      + ' . selected record ' + output)
            records.append(dict(s=s.tolist(), t=t.tolist(), q=queries, keys=keys,
                                orientation=orientation, demos=demonstrations, length=size))
        panels[name] = records
    with gzip.open(ROOT/'panels.gz', 'wt') as handle:
        json.dump(panels, handle)
    (ROOT/'protocol.json').write_text(json.dumps(dict(
        training=TRAIN, testing=TEST, seeds=SEEDS,
        gate='>=95% exact paired spans, routes and conditional outcomes per seed and family',
        hash=hashlib.sha256((ROOT/'panels.gz').read_bytes()).hexdigest(),
        scope='single investigator, synthetic, supervised spans in training, fixed textual outcome format, known record vocabulary'), indent=2))
    return panels


def resolve(text, vec, model):
    candidates = spans(text)
    words = text.split()
    logits = model.decision_function(vec.transform([features(words, a, b) for a, b in candidates]))
    index = int(choose(logits))
    if index < 0:
        return None
    return candidates[index]


def key_for(text, span):
    return ' '.join(text.split()[slice(*span)]) if span else None


def bind(demos, keys, vec, model, supplied, reverse=False):
    values = {}
    for text, key in zip(demos, keys):
        instruction, output = text.split(' . selected record ')
        a, b = [w for w in instruction.split() if w in WORDS]
        predicted = key if supplied else key_for(instruction, resolve(instruction, vec, model))
        if predicted is not None:
            direction = (a, b).index(output) ^ reverse
            values.setdefault(predicted, set()).add(direction)
    return {k: next(iter(v)) for k, v in values.items() if len(v) == 1}


def exact_effects(slot, gold, targets, gate):
    for present in targets:
        cells = [None] * 10
        cells[present], cells[8], cells[9] = ('v', 4), ('v', 7), ('v', 3)
        for op in ('copy', 'link'):
            allow = gold == present
            dest = (('v', 7) if op == 'copy' else ('l', 8)) if allow else ('v', 3)
            expected = cells.copy()
            expected[8], expected[9] = ('v', 91), dest
            value = 7 if allow and op == 'copy' else 91 if allow else 3
            if outcome(State(tuple(cells)), op, slot, gate) != (allow, tuple(expected), value):
                return False
    return True


def main():
    panels = freeze()
    rows, raw = [], {}
    for seed in SEEDS:
        vec, model = train(seed)
        np.savez(ROOT/f'selector-{seed}.npz', coef=model.coef_, intercept=model.intercept_,
                 features=vec.get_feature_names_out())
        reader = Reader(seed, True)
        checkpoint = np.load(ROOT/f'recurrent-{seed}.npz')
        reader.E, reader.a, reader.b = (checkpoint[k] for k in ('E', 'a', 'b'))
        gate = np.load(ROOT/f'commit-{seed}.npz')['predicate']
        enc = np.zeros((len(WORDS), 6), int)
        enc[:, 0], enc[:, 1] = V['entry'], [V[w] for w in WORDS]
        word_vectors = reader.forward(enc)[0]
        for cohort, cases in panels.items():
            support = np.array([x['s'] for x in cases])
            bank = reader.forward(support)[0].reshape(len(cases), 8, 2, 32).sum(2)
            loc = choose(word_vectors @ bank.transpose(0, 2, 1))
            for arm in ('learned', 'given_demo_spans', 'given_query_spans', 'oracle', 'reverse'):
                predictions, picked, effects = [], [], []
                for i, ex in enumerate(cases):
                    bindings = bind(ex['demos'], ex['keys'], vec, model,
                                    arm in ('given_demo_spans', 'oracle', 'reverse'), arm == 'reverse')
                    pair, found, valid = [], [], []
                    for j, text in enumerate(ex['q']):
                        gold_span = find(text, ex['keys'][j])
                        selected = gold_span if arm in ('given_query_spans', 'oracle', 'reverse') else resolve(text, vec, model)
                        selected_key = key_for(text, selected)
                        names = [w for w in text.split() if w in WORDS]
                        index = bindings.get(selected_key)
                        slot = int(loc[i, WORDS.index(names[index])]) if index is not None else -1
                        gold = ex['t'][j ^ ex['orientation']]
                        pair.append(slot)
                        found.append(selected == gold_span)
                        valid.append(exact_effects(slot, gold, ex['t'], gate))
                    predictions.append(pair)
                    picked.append(found)
                    effects.append(valid)
                arr = np.array(predictions)
                gold = np.array([[e['t'][j ^ e['orientation']] for j in range(2)] for e in cases])
                result = dict(seed=seed, cohort=cohort, arm=arm,
                              spans=int(np.array(picked).all(1).sum()), routes=int((arr==gold).all(1).sum()),
                              actions=int(np.array(effects).all(1).sum()), abstain=int((arr<0).sum()))
                rows.append(result)
                raw[f'{seed}_{cohort}_{arm}'] = arr
    np.savez_compressed(ROOT/'predictions.npz', **raw)
    (ROOT/'results.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
