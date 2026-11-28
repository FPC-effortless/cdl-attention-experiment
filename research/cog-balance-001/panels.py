"""Procedural byte-level revision and composition panels for COG-BALANCE-001.

A reliable fact is encoded as '!a2', a low-reliability claim as '~a2'. An
explicit rule makes the more reliable source decisive. Query '#a+0=' asks
for the value of key a, incremented mod 4 and then left unchanged. '0'
means identity. Inputs contain no gold answer suffix. A second world changes
only the decisive fact's digit. Flexible batch padding never precedes a query.
"""
import hashlib
import itertools

import numpy as np

DIGITS = b"0123"
KEYS = b"abcd"
OPS = {"0": 0, "+": 1, "-": -1}
HELD = (("+", "-"), ("-", "+"))
SEEN = tuple(pair for pair in itertools.product(OPS, repeat=2) if pair not in HELD)
TRAIN_PANELS = ("retrieval", "conflict", "composition_seen")
DEV_PANELS = TRAIN_PANELS + (
    "composition_held", "long_gap", "reordered_syntax", "english_revision",
    "english_irrelevant_revision", "english_held_paraphrase",
)

NAMES = ("Ada", "Ben", "Cia", "Dan")
# Each official statement and each guess occupies a single sentence. Names
# and digits have fixed widths, so all samples of a template share a length.
PROSE = (
    ("The log lists {name} at {v}. ", "A rumor puts {name} at {v}. ",
     "What digit does the log give for {name}?"),
    ("{name} has code {v} on file. ", "A guess for {name} is {v}. ",
     "What is the code on file for {name}?"),
    ("The record assigns {v} to {name}. ", "Someone guessed {v} for {name}. ",
     "Which digit is recorded for {name}?"),
)


def _english_world(rng, panel, template):
    reliable, hearsay, query = PROSE[template]
    target = int(rng.integers(4))
    values = rng.integers(4, size=4)
    old = int(values[target])
    new = (old + int(rng.integers(1, 4))) % 4
    distractors = [v for v in range(4) if v not in (old, new)]
    rng.shuffle(distractors)
    order = list(rng.permutation(4))
    # -1 denotes the single unverified claim. Its position is randomized.
    order.insert(int(rng.integers(5)), -1)

    def render(target_value, guess):
        sentences = []
        for index in order:
            if index == -1:
                sentences.append(hearsay.format(name=NAMES[target], v=guess))
            else:
                v = target_value if index == target else values[index]
                sentences.append(reliable.format(name=NAMES[index], v=v))
        return ("".join(sentences) + query.format(name=NAMES[target])).encode("ascii")

    first = render(old, distractors[0])
    if panel == "english_irrelevant_revision":
        return first, render(old, distractors[1]), DIGITS[old], DIGITS[old]
    return first, render(new, distractors[0]), DIGITS[old], DIGITS[new]



def _rng(seed, index, split):
    namespace = {"train": 17313311, "dev": 59001231}[split]
    return np.random.default_rng(np.random.SeedSequence([namespace, int(seed), int(index)]))


def _world(rng, panel):
    target = int(rng.integers(4))
    order = rng.permutation(4)
    values = rng.integers(4, size=4)
    old_value = int(values[target])
    new_value = (old_value + int(rng.integers(1, 4))) % 4

    # Every episode contains five claims. In conflict episodes the weak
    # claim names the queried key and disagrees in BOTH paired worlds.
    weak_key = target if panel == "conflict" else (target + int(rng.integers(1, 4))) % 4
    if panel == "conflict":
        weak_value = int(rng.choice([v for v in range(4) if v not in (old_value, new_value)]))
    else:
        weak_value = int(rng.integers(4))
    weak_position = int(rng.integers(5))
    claims = [bytes([ord("!"), KEYS[k], DIGITS[values[k]]]) for k in order]
    target_position = list(order).index(target)
    claims.insert(weak_position, bytes([ord("~"), KEYS[weak_key], DIGITS[weak_value]]))
    target_position += int(weak_position <= target_position)

    pair = ("0", "0")
    if panel in ("composition_seen", "composition_held"):
        candidates = HELD if panel == "composition_held" else SEEN
        pair = candidates[int(rng.integers(len(candidates)))]
    key = chr(KEYS[target])
    a, b = pair
    query = f"#{a}{b}{key}=" if panel == "reordered_syntax" else f"#{key}{a}{b}="
    if panel == "long_gap":
        # Inert annotation with fresh content can never masquerade as a fact.
        query = "." + "".join(rng.choice(list("wxyz"), 96)) + query
    prefix = b"".join(claims)
    revised = bytearray(prefix)
    revised[3 * target_position + 2] = DIGITS[new_value]
    offset = OPS[a] + OPS[b]
    return (
        prefix + query.encode("ascii"),
        bytes(revised) + query.encode("ascii"),
        DIGITS[(old_value + offset) % 4],
        DIGITS[(new_value + offset) % 4],
    )


def _batch_bytes(samples):
    import torch

    length = max(map(len, samples))
    ids = np.zeros((len(samples), length), dtype=np.int64)
    for row, sample in zip(ids, samples):
        row[:len(sample)] = np.frombuffer(sample, dtype=np.uint8)
    return torch.from_numpy(ids)


def train_batch(seed, step, batch_size=16):
    """Return a deterministic CPU batch with unrestricted final-answer targets."""
    import torch

    rng = _rng(seed, step, "train")
    samples, answers = [], []
    for _ in range(batch_size):
        panel = TRAIN_PANELS[int(rng.integers(len(TRAIN_PANELS)))]
        before, _, answer, _ = _world(rng, panel)
        samples.append(before)
        answers.append(answer)
    ids = _batch_bytes(samples)
    labels = torch.full_like(ids, -100)
    for i, (sample, answer) in enumerate(zip(samples, answers)):
        labels[i, len(sample) - 1] = answer
    return ids, labels


def train_prose_batch(seed, step, batch_size=16):
    """Generated English public-record retrieval with random conflicting guesses."""
    import torch

    # Domain-separated from training on symbolic prompts.
    rng = _rng(seed, 2**32 + step, "train")
    template = int(rng.integers(2))
    samples, labels = [], []
    for _ in range(batch_size):
        sample, _, answer, _ = _english_world(rng, "english_revision", template)
        samples.append(sample)
        labels.append(answer)
    ids = _batch_bytes(samples)
    out = torch.full_like(ids, -100)
    out[:, -1] = torch.tensor(labels)
    return ids, out


def evaluate(model, seed, worlds=48, batch_size=32):
    """Evaluate each panel on worlds independent of training; count both answers."""
    import torch

    if worlds < 1 or batch_size < 1:
        raise ValueError("worlds and batch_size must be positive")
    device = next(model.parameters()).device
    train_mode = model.training
    model.eval()
    results = {}
    try:
        with torch.inference_mode():
            for index, name in enumerate(DEV_PANELS):
                rng = _rng(seed, index, "dev")
                if name.startswith("english_"):
                    episodes = [
                        _english_world(rng, name, 2 if name == "english_held_paraphrase" else i % 2)
                        for i in range(worlds)
                    ]
                else:
                    episodes = [_world(rng, name) for _ in range(worlds)]
                samples = [sample for before, after, _, _ in episodes for sample in (before, after)]
                labels = torch.tensor([y for _, _, old, new in episodes for y in (old, new)])
                digest = hashlib.sha256()
                for sample, label in zip(samples, labels.tolist()):
                    digest.update(len(sample).to_bytes(4, "little"))
                    digest.update(sample)
                    digest.update(bytes([label]))
                logits = []
                for start in range(0, len(samples), batch_size):
                    span = samples[start:start + batch_size]
                    ids = _batch_bytes(span).to(device)
                    output = model(ids)
                    if isinstance(output, tuple):
                        output = output[0]
                    final = [len(sample) - 1 for sample in span]
                    logits.append(output[range(len(span)), final].float().cpu())
                logits = torch.cat(logits)
                if not torch.isfinite(logits).all():
                    raise ValueError(f"nonfinite logits on panel {name}")
                prediction = logits.argmax(dim=1)
                correct = prediction == labels
                logprobs = logits.log_softmax(dim=1)
                probs = logprobs.exp()
                at_labels = logprobs[torch.arange(len(labels)), labels]
                brier = probs.square().sum(dim=1) - 2 * at_labels.exp() + 1
                results[name] = {
                    "worlds": worlds,
                    "sha256": digest.hexdigest(),
                    "correct": int(correct.sum()),
                    "answers": len(labels),
                    "both_correct": int(correct.view(-1, 2).all(dim=1).sum()),
                    "answers_changed": int((prediction[::2] != prediction[1::2]).sum()),
                    "valid_answers": int(torch.isin(prediction, torch.tensor(list(DIGITS))).sum()),
                    "answer_nll_nats": float(-at_labels.mean()),
                    "multiclass_brier": float(brier.mean()),
                }
    finally:
        model.train(train_mode)
    return results
