"""LRR-005: attention-free causal role registers and indexed token-position reads.

Training is answer-position supervised and uses a *soft* all-position contrast;
inference is a hard, incremental register/index procedure, not self-attention.
This is a restricted synthetic mechanism experiment, not a language model.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = {
    "own": ("owns", "keeps", "possesses", "holds"),
    "live": ("inhabits", "occupies", "resides", "dwells"),
    "repair": ("repairs", "fixes", "mends", "restores"),
    "teach": ("teaches", "trains", "tutors", "coaches"),
}
NAMES = ("mira", "niko", "ada", "lena", "sora", "timo", "lumi", "kira", "jude", "ravi")
FILLERS = ("now", "quite", "recently", "quickly", "often", "truly")
VOCAB = ("<pad>", "<unk>", ".", "?", "what", "does") + NAMES + FILLERS + tuple(
    word for words in FAMILIES.values() for word in words
)
ID = {word: i for i, word in enumerate(VOCAB)}
RESERVED = {(1, 2), (3, 0)}


@dataclass(frozen=True)
class Example:
    words: tuple[str, ...]
    answer: int
    family: str
    subject: str
    question_alias: str
    facts: tuple[tuple[str, str, str, str, int], ...]


def make_example(seed: int, n_facts: int = 7, split: str = "train",
                 layout: str = "regular", negative: bool = False) -> Example:
    """Semantically marked metadata is *evaluator only*, never passed to model."""
    if n_facts < 5 or split not in ("train", "dev", "test"):
        raise ValueError("unsupported fixture")
    if layout not in ("regular", "shift", "novel_fillers"):
        raise ValueError("unsupported layout")
    rng = random.Random(seed)
    family = rng.choice(tuple(FAMILIES))
    entity = rng.choice(NAMES)
    aliases = FAMILIES[family]
    if split == "train":
        options = [(a, b) for i, a in enumerate(aliases)
                   for j, b in enumerate(aliases) if (i, j) not in RESERVED]
        fact_alias, query_alias = rng.choice(options)
    else:
        a, b = (3, 0) if split == "dev" else (1, 2)
        fact_alias, query_alias = aliases[a], aliases[b]
    gold_value = f"value{seed}gold"
    facts = [(entity, family, fact_alias, gold_value)]
    for other_family in FAMILIES:
        if other_family != family:
            facts.append((entity, other_family,
                          rng.choice(FAMILIES[other_family]),
                          f"value{seed}wrong{other_family}"))
    facts.append((rng.choice([name for name in NAMES if name != entity]),
                  family, rng.choice(aliases), f"value{seed}other"))
    while len(facts) < n_facts:
        name = rng.choice(NAMES)
        other_family = rng.choice(tuple(FAMILIES))
        if (name, other_family) == (entity, family):
            continue
        facts.append((name, other_family, rng.choice(FAMILIES[other_family]),
                      f"value{seed}n{len(facts)}"))
    rng.shuffle(facts)
    words: list[str] = []
    located: list[tuple[str, str, str, str, int]] = []
    filler_set = FILLERS[3:] if layout == "novel_fillers" else FILLERS[:3]
    def filler():
        return rng.choice(filler_set)
    for name, fam, predicate, value in facts:
        words.append(name)
        if rng.random() < 0.5:
            words.append(filler())
        words.append(predicate)
        if rng.random() < 0.5:
            words.append(filler())
        if layout in ("shift", "novel_fillers") and rng.random() < 0.6:
            words.append(filler())
        words.append(value)
        located.append((name, fam, predicate, value, len(words) - 1))
        words.append(".")
    if negative:
        # A queried entity not observed in the history: easy no-match control.
        present = {f[0] for f in located}
        options = [name for name in NAMES if name not in present]
        entity = rng.choice(options) if options else "unseen_entity"
    words.extend(("what", "does", entity))
    if rng.random() < 0.4:
        words.append(filler())
    words.extend((query_alias, "?"))
    answer = next(f[-1] for f in located if f[3] == gold_value)
    return Example(tuple(words), -1 if negative else answer, family,
                   entity, query_alias, tuple(located))


def make_training_pair(seed: int, step: int, pair_offset: int) -> tuple[Example, Example]:
    """Two correct, conflicting relation queries over the same fact history.

    The second query's ordered (fact alias, question alias) is **also**
    filtered against RESERVED; this closes the original exploratory leakage.
    """
    base = make_example(seed, 7, "train")
    candidates = [f for f in base.facts if f[0] == base.subject and f[1] != base.family]
    fact = candidates[(step + pair_offset) % len(candidates)]
    aliases = FAMILIES[fact[1]]
    original_index = aliases.index(fact[2])
    permitted = [q for j, q in enumerate(aliases) if (original_index, j) not in RESERVED]
    words = list(base.words)
    words[-2] = permitted[(step + pair_offset) % len(permitted)]
    alternative = Example(tuple(words), fact[-1], fact[1],
                          base.subject, words[-2], base.facts)
    if base.answer == alternative.answer:
        raise AssertionError("paired outcomes must differ")
    return base, alternative


def assert_pair_split(example: Example, *, training: bool) -> None:
    fact = [f for f in example.facts if f[-1] == example.answer]
    if len(fact) != 1:
        raise AssertionError("ambiguous or missing gold value")
    words = FAMILIES[example.family]
    pair = (words.index(fact[0][2]), words.index(example.question_alias))
    if training and pair in RESERVED:
        raise AssertionError("train/test alias-pair leakage")
    if not training and pair != (1, 2):
        raise AssertionError("confirmatory test does not use reserved pair")


def vectorize(examples: list[Example]):
    length = max(len(ex.words) for ex in examples)
    x = torch.zeros(len(examples), length, dtype=torch.long)
    y = torch.tensor([ex.answer for ex in examples], dtype=torch.long)
    q = torch.tensor([len(ex.words) - 1 for ex in examples], dtype=torch.long)
    for i, ex in enumerate(examples):
        x[i, :len(ex.words)] = torch.tensor([ID.get(w, 1) for w in ex.words])
    return x, y, q


class CausalRoleRegister(nn.Module):
    """Copy gates + hard relation codes; no per-query history token scores."""
    def __init__(self, width: int = 32, codes: int = 8):
        super().__init__()
        self.codes = codes
        self.emb = nn.Embedding(len(VOCAB), width, padding_idx=0)
        self.conv = nn.Conv1d(width, width, 3)
        self.norm = nn.LayerNorm(width)
        self.role = nn.Linear(width, 3)
        self.code = nn.Linear(width, codes)
        self.null_logit = nn.Parameter(torch.tensor(-1.5))

    def token_features(self, x):
        h = self.emb(x)
        return self.norm(h + F.gelu(self.conv(
            F.pad(h.transpose(1, 2), (2, 0))
        ).transpose(1, 2)))

    def forward(self, x):
        h = self.token_features(x)
        gates = self.role(h).sigmoid()
        local_codes = self.code(h).softmax(-1)
        entity = torch.zeros(x.shape[0], len(VOCAB), device=x.device)
        relation = torch.ones(x.shape[0], self.codes, device=x.device) / self.codes
        entities, relations = [], []
        for t in range(x.shape[1]):
            # CRITICAL: The value address is read from state BEFORE x_t updates it.
            entities.append(entity)
            relations.append(relation)
            ge, gr = gates[:, t, :1], gates[:, t, 1:2]
            entity = ge * F.one_hot(x[:, t], num_classes=len(VOCAB)).float() + (1-ge) * entity
            relation = gr * local_codes[:, t] + (1-gr) * relation
        return torch.stack(entities, 1), torch.stack(relations, 1), gates

    def answer_loss(self, x, answer, query_index, pairs=None, credit: float = 0.0):
        e, r, g = self(x)
        rows = torch.arange(len(x))
        qe, qr = e[rows, query_index], r[rows, query_index]
        score = ((e * qe[:, None, :]).sum(-1) *
                 (r * qr[:, None, :]).sum(-1) * g[:, :, 2]).clamp_min(1e-9)
        valid = torch.arange(x.shape[1])[None, :] < query_index[:, None]
        scores = torch.where(valid, score.log(), torch.full_like(score, -1e9))
        scores = torch.cat((scores, self.null_logit.expand(len(x), 1)), -1)
        targets = torch.where(answer >= 0, answer, torch.full_like(answer, x.shape[1]))
        ce = F.cross_entropy(scores, targets)
        regularizer = .018 * (g * valid[:, :, None]).mean()
        pos = g[rows, answer.clamp_min(0), 2].clamp_min(1e-9)
        pos_penalty = (-torch.log(pos) * (answer >= 0).float()).sum() / max(1, int((answer >= 0).sum()))
        loss = ce + regularizer + .06 * F.softplus(pos_penalty)
        local = torch.tensor(0., device=x.device)
        if credit and pairs:
            pair_idx = torch.tensor(pairs, dtype=torch.long)
            pa, pb = qr[pair_idx[:, 0]], qr[pair_idx[:, 1]]
            similarity = (pa * pb).sum(-1)
            local = -torch.log1p(-similarity.clamp(max=1-1e-6)).mean()
            loss = loss + credit * local
        return loss, {"ce": float(ce.detach()), "local_credit": float(local.detach())}

    @torch.no_grad()
    def route(self, words: tuple[str, ...]):
        self.eval()
        x = torch.tensor([[ID.get(w, 1) for w in words]])
        h = self.token_features(x)
        gates = self.role(h).sigmoid()[0].tolist()
        codes = self.code(h)[0].argmax(-1).tolist()
        entity = relation = None
        posting: dict[tuple[str, int], int] = {}
        writes = 0
        for i, token in enumerate(words):
            # Query is the final '?', read registers without consuming '?'.
            if i == len(words) - 1:
                break
            if gates[i][2] > .5 and entity is not None and relation is not None:
                posting[(entity, relation)] = i
                writes += 1
            if gates[i][0] > .5:
                entity = token
            if gates[i][1] > .5:
                relation = codes[i]
        answer = posting.get((entity, relation))
        if answer is not None and answer >= len(words) - 1:
            raise AssertionError("future-token leakage")
        return answer, {"writes": writes, "distinct_addresses": len(posting),
                        "abstained": answer is None, "index_probes": 1}


def fit(seed: int, *, steps=500, batch_size=32, credit=0.0):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = CausalRoleRegister()
    optim = torch.optim.AdamW(model.parameters(), lr=.004)
    started = time.perf_counter()
    training_pairs_checked = 0
    for step in range(steps):
        samples: list[Example] = []
        pairs = []
        for i in range(batch_size // 2):
            a, b = make_training_pair(seed * 1_000_000 + step * 1_000 + i, step, i)
            assert_pair_split(a, training=True)
            assert_pair_split(b, training=True)
            training_pairs_checked += 2
            samples.extend((a, b))
            pairs.append((len(samples) - 2, len(samples) - 1))
        x, answer, q = vectorize(samples)
        model.train()
        optim.zero_grad(set_to_none=True)
        loss, diagnostics = model.answer_loss(x, answer, q, pairs, credit)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optim.step()
    model.eval()
    return model, {"seed": seed, "steps": steps, "parameters": sum(p.numel() for p in model.parameters()),
                   "train_documents": steps * batch_size // 2,
                   "train_questions": steps * batch_size,
                   "anti_leakage_checked": training_pairs_checked,
                   "wall_seconds": time.perf_counter() - started,
                   "final_objective": float(loss.detach()), **diagnostics}
