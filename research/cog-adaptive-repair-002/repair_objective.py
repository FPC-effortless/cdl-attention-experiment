"""COG byte-level revision training. Trace labels and pairing are training metadata."""
from dataclasses import dataclass
import re

import torch
from torch.nn import functional as F

from panels import DIGITS, NAMES, OPS, PROSE, TRAIN_PANELS, _batch_bytes, _english_world, _rng, _world

REGIMES = ("final", "trace", "pair", "trace_pair")
OFFICIAL_LOG = re.compile(rb"The log lists (\w+) at ([0-3])")
OFFICIAL_FILE = re.compile(rb"(\w+) has code ([0-3]) on file")
RUMOR = re.compile(rb"A rumor puts (\w+) at ([0-3])")
GUESS = re.compile(rb"A guess for (\w+) is ([0-3])")
QUERY_NAME = re.compile(rb"for (\w+)\?$")


@dataclass
class RepairBatch:
    ids: torch.Tensor
    final_targets: torch.Tensor
    trace_targets: torch.Tensor


def trace_targets(prompt):
    """Return prefix-determined (position, byte label) pairs and terminal label.

    At a claim boundary, report the most recent authoritative value for that
    claim's key if known. A conflicting weak claim cannot change that value.
    At a query, report the retrieved value, then each available operation's
    result. Each target uses only bytes at or before its own position.
    """
    facts, events = {}, []
    if prompt.startswith((b"!", b"~")):
        claims, query = prompt.split(b"#")
        if len(claims) % 3 or len(query) != 4 or query[-1:] != b"=":
            raise ValueError("Invalid symbolic training grammar")
        for start in range(0, len(claims), 3):
            role, key, digit = claims[start:start + 3]
            if role == ord("!"):
                facts[key] = digit
            if key in facts:
                events.append((start + 2, facts[key]))
        pos = len(claims) + 1
        value = facts[query[0]] - ord("0")
        events.append((pos, DIGITS[value]))
        for delta_pos, op in enumerate(query[1:3], start=1):
            value = (value + OPS[chr(op)]) % 4
            events.append((pos + delta_pos, DIGITS[value]))
        return events, DIGITS[value]

    clauses = prompt.split(b". ")
    query = clauses.pop()
    pos = 0
    for clause in clauses:
        official = OFFICIAL_LOG.fullmatch(clause) or OFFICIAL_FILE.fullmatch(clause)
        match = official or RUMOR.fullmatch(clause) or GUESS.fullmatch(clause)
        if match is None:
            raise ValueError("Unrecognized English training clause")
        key, digit = match.groups()
        if official:
            facts[key] = digit[0]
        if key in facts:
            events.append((pos + len(clause), facts[key]))  # full stop after claim
        pos += len(clause) + 2
    match = QUERY_NAME.search(query)
    if match is None:
        raise ValueError("Unrecognized English training question")
    answer = facts[match.group(1)]
    events.append((pos + match.end(1) - 1, answer))  # final character of visible name
    return events, answer


def balanced_english_world(rng, template, intervention):
    """Four official records and four rumors; target frequency is constant.

    Query-swap pairs have byte-identical histories. A relevant revision
    changes only the requested official record. An irrelevant revision changes
    only its rumor. Rumor values independently range over all four digits.
    """
    reliable, rumor, question = PROSE[template]
    target, other = map(int, rng.choice(4, size=2, replace=False))
    values = rng.integers(4, size=4)
    guesses = rng.integers(4, size=4)
    order = rng.permutation(8)
    if intervention == "query" and values[other] == values[target]:
        values[other] = (values[target] + rng.integers(1, 4)) % 4

    def render(key, facts, hearsay):
        sentences = []
        for entry in order:
            i = entry % 4
            text, digit = (reliable, facts[i]) if entry < 4 else (rumor, hearsay[i])
            sentences.append(text.format(name=NAMES[i], v=digit))
        return ("".join(sentences) + question.format(name=NAMES[key])).encode("ascii")

    before = render(target, values, guesses)
    new_target = other if intervention == "query" else target
    new_values, new_guesses = values.copy(), guesses.copy()
    if intervention == "fact":
        new_values[target] = (values[target] + rng.integers(1,4)) % 4
    elif intervention == "rumor":
        new_guesses[target] = (guesses[target] + rng.integers(1,4)) % 4
    elif intervention != "query":
        raise ValueError(f"Unknown intervention {intervention!r}")
    after = render(new_target, new_values, new_guesses)
    return before, after, DIGITS[values[target]], DIGITS[new_values[new_target]]


def repair_batch(seed, step, batch_size=8, balanced=False):
    """Every arm sees the same even-sized batch of adjacent before/after worlds."""
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Use an even batch_size of at least two for paired training")
    english = step % 2 == 1
    rng = _rng(seed, step + 2**33, "train")
    template = int(rng.integers(2)) if english else None
    samples, targets, traces = [], [], []
    for _ in range(batch_size // 2):
        if english and balanced:
            # 3/8 each decisive fact and query revisions, 1/4 rumor controls.
            choice = int(rng.integers(8))
            intervention = "fact" if choice < 3 else ("query" if choice < 6 else "rumor")
            episode = balanced_english_world(rng, template, intervention)
        elif english:
            # Credible revisions outnumber irrelevant revisions three to one.
            panel = "english_irrelevant_revision" if rng.integers(4) == 0 else "english_revision"
            episode = _english_world(rng, panel, template)
        else:
            panel = TRAIN_PANELS[int(rng.integers(len(TRAIN_PANELS)))]
            episode = _world(rng, panel)
        for prompt, answer in zip(episode[:2], episode[2:]):
            events, decoded = trace_targets(prompt)
            if decoded != answer or any(i >= len(prompt)-1 for i, _ in events):
                raise ValueError("Causal traces disagree with generator")
            samples.append(prompt)
            targets.append(answer)
            traces.append(events)

    if len({len(s) for s in samples}) != 1:
        raise ValueError("Batch must use one sequence length for the final-position readout")
    ids = _batch_bytes(samples)
    auxiliary = torch.full_like(ids, -100)
    for i, events in enumerate(traces):
        positions, labels = zip(*events)
        auxiliary[i, list(positions)] = torch.tensor(labels)
    return RepairBatch(ids, torch.tensor(targets), auxiliary)


def intervention_loss(last, targets):
    """Require old-vs-new log odds to move in the correct direction.

    For irrelevant revisions penalize the Jensen-Shannon divergence of the
    full 288-way output distributions. The model never receives the other
    member of its pair as an inference input.
    """
    left, right = last[::2], last[1::2]
    a, b = targets[::2, None], targets[1::2, None]
    left_gap = left.gather(1, a) - left.gather(1, b)
    right_gap = right.gather(1, a) - right.gather(1, b)
    change_loss = F.softplus(-0.5 * (left_gap - right_gap)).flatten()
    logp, logq = F.log_softmax(left, dim=-1), F.log_softmax(right, dim=-1)
    logm = torch.logaddexp(logp, logq) - 0.6931471805599453
    stable_loss = 0.5 * ((logp.exp() * (logp-logm)).sum(-1) +
                         (logq.exp() * (logq-logm)).sum(-1))
    return torch.where(a.flatten() != b.flatten(), change_loss, stable_loss).mean()


def repair_loss(logits, batch, regime):
    """One fixed unit of cognitive objective weight shared by active terms."""
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime {regime!r}")
    use_trace = regime in ("trace", "trace_pair")
    use_pair = regime in ("pair", "trace_pair")
    last = logits[:, -1]
    targets = batch.final_targets.to(last.device)
    loss = (1 - 0.25 * (int(use_trace) + int(use_pair))) * F.cross_entropy(last, targets)
    if use_trace:
        labels = batch.trace_targets.to(last.device)
        mask = labels != -100
        loss = loss + 0.25 * F.cross_entropy(logits[mask], labels[mask])
    if use_pair:
        loss = loss + 0.25 * intervention_loss(last, targets)
    return loss
