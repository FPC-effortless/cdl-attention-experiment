"""Independent, generated challenge panels for byte-only revision and addressing.

Runs held synthetic evaluations after model training. All four people have
one official and one weak claim, in shuffled order. Identical-history
question swaps explicitly test dependence on the named subject. The grammar
oracle is only an evaluation check. Individual model inputs are ordinary
bytes and never contain the target or annotations as extra input fields.
"""
from collections import Counter
import hashlib
import re

import numpy as np


NAMES = ("Ada", "Ben", "Cia", "Dan")
NEW_NAMES = ("Ayo", "Ibi", "Kemi", "Seyi")
VALUES = b"0123"
SEED = 73_071_031
PANELS = (
    "balanced_familiar",
    "balanced_new_names",
    "balanced_new_phrasing",
    "balanced_new_names_and_phrasing",
    "balanced_irrelevant_official_edit",
    "balanced_irrelevant_rumor_edit",
    "balanced_query_swap",
    "symbol_value_key_order",
)
FAMILIAR = (
    "The log lists {n} at {v}. ", "A rumor puts {n} at {v}. ",
    "What digit does the log give for {n}?",
)
NOVEL = (
    "{v} is the signed entry for {n}. ", "{v} is a guess about {n}. ",
    "Retrieve the signed entry belonging to {n}.",
)


def _world(rng, panel):
    """One before/after episode with balanced source frequency for all subjects."""
    names = NEW_NAMES if panel in ("balanced_new_names", "balanced_new_names_and_phrasing") else NAMES
    templates = NOVEL if panel in ("balanced_new_phrasing", "balanced_new_names_and_phrasing") else FAMILIAR
    official = rng.integers(4, size=4).tolist()
    rumor = rng.integers(4, size=4).tolist()
    subject = int(rng.integers(4))
    other = (subject + int(rng.integers(1, 4))) % 4
    shifted = lambda v: (v + int(rng.integers(1, 4))) % 4
    order = rng.permutation(8)
    # Both versions in the question swap must have different valid answers.
    if panel == "balanced_query_swap" and official[other] == official[subject]:
        official[other] = shifted(official[other])

    def render(q):
        text = []
        for index in order:
            weak, who = divmod(int(index), 4)
            value = (rumor if weak else official)[who]
            if panel == "symbol_value_key_order":
                text.append(f"{'~' if weak else '!'}{value}{'abcd'[who]}")
            else:
                text.append(templates[weak].format(n=names[who], v=value))
        text.append(f"#00{'abcd'[q]}=" if panel == "symbol_value_key_order" else templates[2].format(n=names[q]))
        return "".join(text).encode("ascii")

    before, a = render(subject), official[subject]
    if panel == "balanced_query_swap":
        subject = other
    elif panel == "balanced_irrelevant_official_edit":
        official[other] = shifted(official[other])
    elif panel == "balanced_irrelevant_rumor_edit":
        rumor[subject] = shifted(rumor[subject])
    else:
        official[subject] = shifted(official[subject])
    return before, render(subject), VALUES[a], VALUES[official[subject]]


def _digest(episodes):
    digest = hashlib.sha256()
    for before, after, a, b in episodes:
        for data, label in ((before, a), (after, b)):
            digest.update(len(data).to_bytes(4, "little"))
            digest.update(data)
            digest.update(bytes([label]))
    return digest.hexdigest()


def _counts(episodes, pred):
    labels = np.array([world[2:] for world in episodes])
    pred = np.array(pred).reshape(labels.shape)
    correct = pred == labels
    return {
        "worlds": len(labels), "correct": int(correct.sum()), "answers": labels.size,
        "both_correct": int(correct.all(axis=1).sum()),
        "answers_changed": int((pred[:, 0] != pred[:, 1]).sum()),
        "target_changed": int((labels[:, 0] != labels[:, 1]).sum()),
        "sha256": _digest(episodes),
    }


def _episodes(panel, worlds, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9449]))
    # The four surface variants share exactly the same underlying worlds.
    return [_world(rng, panel) for _ in range(worlds)]


def evaluate_confirmation(model, worlds=32, seed=SEED, batch_size=12):
    """Return per-panel unrestricted top-1 raw-byte predictions and pair counts."""
    import torch

    if min(worlds, batch_size) <= 0:
        raise ValueError("worlds and batch_size must be positive")
    device = next(model.parameters()).device
    mode = model.training
    model.eval()
    out = {}
    try:
        with torch.inference_mode():
            for panel in PANELS:
                cases = _episodes(panel, worlds, seed)
                texts = [text for case in cases for text in case[:2]]
                decisions = []
                for start in range(0, len(texts), batch_size):
                    group = texts[start:start + batch_size]
                    ids = torch.zeros((len(group), max(map(len, group))), dtype=torch.long, device=device)
                    for i, item in enumerate(group):
                        ids[i, :len(item)] = torch.as_tensor(list(item), dtype=torch.long, device=device)
                    logits = model(ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    scores = logits[range(len(group)), [len(t)-1 for t in group]]
                    if not torch.isfinite(scores).all():
                        raise ValueError(f"nonfinite output on {panel}")
                    decisions.extend(scores.argmax(-1).tolist())
                out[panel] = _counts(cases, decisions)
    finally:
        model.train(mode)
    return out


def _parse(raw):
    if raw.startswith((b"!", b"~")):
        history, question = raw.split(b"#")
        clauses = re.findall(rb"([!~])([0-3])([a-d])", history)
        if clauses:
            evidence = [(subject, mark == b"!", digit) for mark, digit, subject in clauses]
        else:
            evidence = [(subject, mark == b"!", digit) for mark, subject, digit
                        in re.findall(rb"([!~])([a-d])([0-3])", history)]
        return evidence, re.search(rb"[a-d]", question).group(), question
    clauses = raw.split(b". ")
    query = clauses.pop()
    subjects = rb"(?:Ada|Ben|Cia|Dan|Ayo|Ibi|Kemi|Seyi)"
    target = re.search(subjects, query).group()
    facts = []
    for clause in clauses:
        subject = re.search(subjects, clause).group()
        digit = re.search(rb"[0-3]", clause).group()
        weak = any(marker in clause.lower() for marker in (b"rumor", b"guess"))
        facts.append((subject, not weak, digit))
    return facts, target, b""


def _rules(raw):
    facts, target, ops = _parse(raw)
    official = {name: digit[0] for name, auth, digit in facts if auth}
    # Deliberately never consult target when selecting the most-mentioned name.
    # Lexical tie-break is fixed; changing the query has no influence.
    count = Counter(name for name, _, _ in facts)
    frequent = min(count, key=lambda n: (-count[n], n))
    read = official[target]
    return {
        "full_input_grammar": VALUES[(read - 48 + ops.count(b"+") - ops.count(b"-")) % 4],
        "ignore_query_most_mentioned_official": official[frequent],
        "ignore_operations_read_target": read,
        "copy_last_any_claim": facts[-1][2][0],
        "uniform_hashed_input": VALUES[hashlib.sha256(raw).digest()[0] % 4],
        "constant_zero": 48,
    }


def audit_shortcuts(worlds=64, original_seed=610003):
    from panels import DEV_PANELS, _english_world, _rng, _world as original_world

    def apply(cases):
        preds = {}
        for case in cases:
            for text in case[:2]:
                for rule, answer in _rules(text).items():
                    preds.setdefault(rule, []).append(answer)
        measures = {rule: _counts(cases, p) for rule, p in preds.items()}
        assert measures["full_input_grammar"]["both_correct"] == len(cases)
        return measures

    original = {}
    for j, panel in enumerate(DEV_PANELS):
        rng = _rng(original_seed, j, "dev")
        if panel.startswith("english_"):
            cases = [_english_world(rng, panel, 2 if panel == "english_held_paraphrase" else i % 2)
                     for i in range(worlds)]
        else:
            cases = [original_world(rng, panel) for _ in range(worlds)]
        original[panel] = apply(cases)
    confirmation = {p: apply(_episodes(p, worlds, SEED)) for p in PANELS}
    return {"worlds_per_panel": worlds, "existing": original, "confirmation": confirmation}


if __name__ == "__main__":
    import json
    print(json.dumps(audit_shortcuts(), indent=2))
