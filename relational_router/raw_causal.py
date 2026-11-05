"""LRR-003: answer-supervised causal local role addressing over unmarked text.

No fact/role/value labels or equivalence pairs are supplied at runtime.
The grammar, finite known relation capacity and exact name equality are priors.
The training-only soft position-ranking loss is NOT an inference mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

ALIASES = {
    "ownership": ("owns", "keeps", "possesses", "holds"),
    "residence": ("inhabits", "occupies", "resides", "dwells"),
    "repair": ("repairs", "fixes", "mends", "restores"),
    "teaching": ("teaches", "trains", "tutors", "coaches"),
}
NAMES = ("mira", "niko", "ada", "lena", "sora", "timo",
         "lumi", "kira", "jude", "ravi", "nora", "elina")
VOCAB = ("<pad>", "<unk>", ".", "?", "what", "does", "now") + NAMES + tuple(
    word for words in ALIASES.values() for word in words)
IDS = {word: i for i, word in enumerate(VOCAB)}
PUNCTUATION = (".", "?")


def tokenize(text: str) -> tuple[str, ...]:
    """Only punctuation/whitespace normalization; no semantic fact parser."""
    return tuple(text.lower().replace(".", " . ").replace("?", " ? ").split())


@dataclass(frozen=True)
class Example:
    text: str
    answer: str
    gold_index: int
    family: str           # evaluator only, never passed to model
    fact_alias: str       # evaluator only
    question_alias: str   # evaluator only

    @property
    def tokens(self) -> tuple[str, ...]:
        return tokenize(self.text)


def generate(seed: int, n_facts: int, split: str = "train",
             *, new_entity: bool = False, update: bool = False) -> Example:
    """Raw clause strings; split omits two ordered synonym pairs per family."""
    if n_facts < (6 if update else 5):
        raise ValueError("not enough facts for hard distractors")
    if split not in ("train", "dev", "test"):
        raise ValueError("unrecognized split")
    rng = random.Random(seed)
    entity = f"nonce_{seed}_name" if new_entity else rng.choice(NAMES)
    family = rng.choice(tuple(ALIASES))
    words = ALIASES[family]
    if split == "train":
        allowed = [(a, b) for i, a in enumerate(words)
                   for j, b in enumerate(words) if (i, j) not in ((0, 1), (2, 3))]
        fact_alias, question_alias = rng.choice(allowed)
    elif split == "dev":
        fact_alias, question_alias = words[0], words[1]
    else:
        fact_alias, question_alias = words[2], words[3]

    answer = f"value_{seed}_gold"
    # Each example receives an entirely new value token; its embedding is UNK.
    facts = [(entity, fact_alias, answer, family)]
    others = [x for x in ALIASES if x != family]
    for i, other in enumerate(others):
        facts.append((entity, rng.choice(ALIASES[other]),
                      f"value_{seed}_wrongrel_{i}", other))
    other_entity = rng.choice([x for x in NAMES if x != entity])
    facts.append((other_entity, rng.choice(words),
                  f"value_{seed}_wrongentity", family))
    while len(facts) < n_facts - int(update):
        other = rng.choice(tuple(ALIASES))
        name = rng.choice(NAMES)
        if (name, other) == (entity, family):
            continue
        facts.append((name, rng.choice(ALIASES[other]),
                      f"value_{seed}_distractor_{len(facts)}", other))
    rng.shuffle(facts)
    if update:
        # Model must resolve newest compatible write rather than old one.
        older = f"value_{seed}_old"
        facts = [(entity, fact_alias, older, family)] + [
            f for f in facts if f[2] != answer] + [
            (entity, fact_alias, answer, family)]

    tokens: list[str] = []
    for subject, predicate, value, _ in facts:
        tokens.extend((subject, predicate, value, "."))
    tokens.extend(("what", "does", entity, question_alias, "?"))
    if tokens.count(answer) != 1:
        raise AssertionError("answer must identify exactly one training token")
    position = tokens.index(answer)
    if position >= len(tokens) - 1:
        raise AssertionError("future answer")
    return Example(" ".join(tokens), answer, position,
                   family, fact_alias, question_alias)


def examples_to_batch(examples: list[Example]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    texts = [e.tokens for e in examples]
    length = max(len(t) for t in texts)
    ids = torch.zeros((len(texts), length), dtype=torch.long)
    answer = torch.tensor([e.gold_index for e in examples], dtype=torch.long)
    query = torch.tensor([len(e.tokens) - 1 for e in examples], dtype=torch.long)
    for row, tokens in enumerate(texts):
        ids[row, :len(tokens)] = torch.tensor(
            [IDS.get(t, IDS["<unk>"]) for t in tokens], dtype=torch.long)
    return ids, answer, query


class CausalRoleIndexer(nn.Module):
    """Independent local causal filters, discrete relation codes, hard lookup.

    A small learned offset-classifier finds entity and predicate surfaces in
    the previous four raw positions. It does not compare current query to
    all earlier tokens. At query-time only one hash address is retrieved.
    """

    def __init__(self, width: int = 48, window: int = 4, codes: int = 4):
        super().__init__()
        self.width, self.window, self.codes = width, window, codes
        self.emb = nn.Embedding(len(VOCAB), width, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(width, width, 3, dilation=d) for d in (1, 2, 4))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.convs)
        self.entity_offset = nn.Linear(width, window)
        self.relation_offset = nn.Linear(width, window)
        self.lexical_code = nn.Embedding(len(VOCAB), codes)
        self.write_gate = nn.Linear(width, 1)

    def forward(self, ids: torch.Tensor):
        h = self.emb(ids)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(F.pad(h.transpose(1, 2), (2 * conv.dilation[0], 0)))
            h = norm(h + F.gelu(x.transpose(1, 2)))
        # Only past tokens; there is no positional information from the future.
        prev = torch.stack(
            [F.pad(ids, (lag, 0))[:, :ids.shape[1]]
             for lag in range(1, self.window + 1)], dim=2)
        return (self.entity_offset(h), self.relation_offset(h),
                self.lexical_code(prev), self.write_gate(h).squeeze(-1), prev)

    def answer_position_loss(self, ids: torch.Tensor,
                             answer: torch.Tensor, query: torch.Tensor,
                             balance_weight: float = 0.2) -> torch.Tensor:
        entity_logits, relation_logits, code_logits, gate, previous = self(ids)
        b, seq = ids.shape
        rows = torch.arange(b, device=ids.device)
        entity_p = entity_logits.softmax(-1)
        relation_p = relation_logits.softmax(-1)
        code_p = code_logits.softmax(-1)

        query_entity = entity_p[rows, query]
        query_window = previous[rows, query]
        equality = ((query_window[:, None, :, None] == previous[:, :, None, :]) &
                    (query_window[:, None, :, None] != 0) &
                    (previous[:, :, None, :] != 0)).float()
        same_entity = (equality * query_entity[:, None, :, None] *
                       entity_p[:, :, None, :]).sum(dim=(-1, -2))

        read_relation = (relation_p[rows, query, :, None] *
                         code_p[rows, query]).sum(dim=1)
        write_relation = (relation_p[..., None] * code_p).sum(dim=2)
        same_relation = (write_relation * read_relation[:, None, :]).sum(-1)

        match = (same_entity * same_relation * gate.sigmoid()).clamp(1e-10, 1)
        causal = torch.arange(seq, device=ids.device)[None, :] < query[:, None]
        logits = torch.where(causal, match.log(),
                             torch.full_like(match, -1e9))
        answer_loss = F.cross_entropy(logits, answer)

        gold_relation = write_relation[rows, answer]
        usage = 0.0
        for dist in (read_relation, gold_relation):
            marg = dist.mean(dim=0)
            usage = usage + (
                marg * (marg.clamp_min(1e-8) * self.codes).log()).sum()

        offset_entropy = 0.0
        for dist in (entity_p[rows, answer], entity_p[rows, query],
                     relation_p[rows, answer], relation_p[rows, query]):
            offset_entropy = offset_entropy - (
                dist * dist.clamp_min(1e-9).log()).sum(-1).mean()
        gate_positive = F.softplus(-gate[rows, answer]).mean()
        gate_budget = (gate.sigmoid() * causal).sum() / causal.sum()
        return (answer_loss + 0.1 * gate_positive + 0.03 * gate_budget +
                balance_weight * usage + 0.005 * offset_entropy)

    @torch.no_grad()
    def route(self, text: str, *, relation_only: bool = False,
              literal_relation: bool = False) -> dict:
        """Inference consumes only unmarked text, never gold annotations."""
        self.eval()
        t0 = time.perf_counter_ns()
        tokens = tokenize(text)
        if not tokens or tokens[-1] != "?":
            raise ValueError("a query ending in ? is required")
        x = torch.tensor([[IDS.get(w, IDS["<unk>"]) for w in tokens]],
                         dtype=torch.long)
        with torch.inference_mode():
            entity_logit, relation_logit, lexical_logit, gate, _ = self(x)
        ent_off = entity_logit[0].argmax(-1).tolist()
        rel_off = relation_logit[0].argmax(-1).tolist()
        rel_code = lexical_logit[0].argmax(-1).tolist()
        admissions = 0
        index: dict[tuple[str, int | str], int] = {}

        def key_at(pos: int) -> tuple[str, int | str] | None:
            e = pos - 1 - ent_off[pos]
            r = pos - 1 - rel_off[pos]
            if e < 0 or r < 0:
                return None
            entity = "*" if relation_only else tokens[e]
            relation = tokens[r] if literal_relation else rel_code[pos][rel_off[pos]]
            return (entity, relation)

        question = len(tokens) - 1
        for i in range(question):
            if float(gate[0, i].sigmoid()) < 0.5:
                continue
            address = key_at(i)
            if address is None:
                continue
            index[address] = i  # latest write wins
            admissions += 1
        address = key_at(question)
        answer_index = index.get(address) if address is not None else None
        elapsed = (time.perf_counter_ns() - t0) / 1000.0
        return {
            "token_index": answer_index,
            "token": tokens[answer_index] if answer_index is not None else None,
            "query_address": address,
            "admissions": admissions,
            "distinct_addresses": len(index),
            "index_probes": int(address is not None),
            "elapsed_total_cpu_us": elapsed,
        }


def train_model(seed: int, *, steps: int = 700, batch_size: int = 48,
                learning_rate: float = 0.002, width: int = 48,
                balance_weight: float = 0.2) -> tuple[CausalRoleIndexer, dict]:
    """No relation labels, span labels or precomputed alias equivalences."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = CausalRoleIndexer(width=width)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    last_loss = 0.0
    for step in range(steps):
        data = [
            generate(seed * 10_000_000 + step * 1_000 + i,
                     n_facts=random.Random(seed + step * 1_000 + i).randint(5, 8),
                     split="train")
            for i in range(batch_size)
        ]
        ids, gold, query = examples_to_batch(data)
        model.train()
        optim.zero_grad(set_to_none=True)
        loss = model.answer_position_loss(
            ids, gold, query, balance_weight=balance_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optim.step()
        last_loss = float(loss.detach().item())
    return model, {
        "train_seed": seed,
        "updates": steps,
        "training_documents": steps * batch_size,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "final_training_objective": last_loss,
        "loss": "answer-position contrastive ranking and address balance",
    }
