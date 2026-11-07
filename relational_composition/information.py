"""LRC-002 conditional prediction diagnostics, not a compression codec.

Oracle routes and answer masks are privileged positive-control inputs. Only the
learned_lm arm is an unscaffolded raw-text LM with learned indexed routing.
"""
from dataclasses import replace
import math
import random

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .data import batch, encode, make_example


ARMS = ("learned_lm", "local_lm", "oracle_lm", "oracle_answer",
        "local_answer", "oracle_random_answer")
SUFFIXES = "abcdefghijklmnopqrst"


def example(seed, *, condition="iid", random_answer=False):
    ex = make_example(seed, "dev" if condition != "iid" else "train", condition)
    if random_answer:
        # Independent label stream; neither input text nor route metadata changes.
        # Seed is never a model input; fresh worlds/label seeds are used at eval.
        label = "n" + random.Random(seed ^ 0x5A71F903).choice(SUFFIXES)
        ex = replace(ex, text=ex.text[:ex.answer_start] + label + "\n", answer=label)
    return ex


def oracle_routes(examples, tokens, layers):
    """Supply both supporting fact endpoints at answer-prediction positions.

    This already resolves the graph chain. It tests endpoint decoding capacity,
    NOT discovery of multi-hop structure, composition or learned addressing.
    """
    routes = torch.full((layers, *tokens.shape, 2), -1, dtype=torch.long, device=tokens.device)
    for row, ex in enumerate(examples):
        routes[:, row, ex.answer_start - 1:, :] = torch.tensor(ex.fact_ends, device=tokens.device)
    return routes


def forward_options(arm, examples, tokens, layers, *, corrupt=False):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if arm.startswith("oracle"):
        routes = oracle_routes(examples, tokens, layers)
        if corrupt:
            # Same input text and query positions; substitute unrelated earlier
            # occurrences. Does not leak answer labels or change gather budget.
            for row, ex in enumerate(examples):
                routes[:, row, ex.answer_start - 1:] = torch.tensor(
                    [(p + ex.answer_start // 2) % (ex.answer_start - 1) for p in ex.fact_ends])
        return {"oracle_operands": routes}
    if arm.startswith("local"):
        return {"memory": "disabled"}
    return {"memory": "shuffled" if corrupt else "normal"}


@torch.no_grad()
def generate_answers(model, examples, arm):
    """Greedy two-byte generation: no gold answer prefix is fed back."""
    sequences = [encode(e.text[:e.answer_start]) for e in examples]
    answers = [[] for _ in examples]
    for _ in range(2):
        tokens = pad_sequence([torch.tensor(s) for s in sequences], batch_first=True)
        options = forward_options(arm, examples, tokens, model.config.layers)
        out = model(tokens, **options)
        for row, seq in enumerate(sequences):
            value = int(out.logits[row, len(seq) - 1].argmax())
            answers[row].append(value)
            # PAD is not a language symbol; stop that sample with an explicit
            # nonzero invalid sentinel if sampled, counted as a wrong answer.
            seq.append(value if value else 257 - 1)
    return [values == encode(e.answer) for values, e in zip(answers, examples)]


@torch.no_grad()
def measure(model, examples, arm, *, chunk_size=8):
    model.eval()
    records = []
    for start in range(0, len(examples), chunk_size):
        part = examples[start:start + chunk_size]
        x, y, mask = batch(part)
        normal = model(x, **forward_options(arm, part, x, model.config.layers))
        corrupt = model(x, **forward_options(arm, part, x, model.config.layers, corrupt=True))
        loss = F.cross_entropy(normal.logits.flatten(0, 1), y.flatten(),
                               ignore_index=0, reduction="none").reshape_as(y)
        bad_loss = F.cross_entropy(corrupt.logits.flatten(0, 1), y.flatten(),
                                   ignore_index=0, reduction="none").reshape_as(y)
        generated = generate_answers(model, part, arm)
        for row, ex in enumerate(part):
            bits = float(loss[row][mask[row]].sum() / math.log(2))
            corrupt_bits = float(bad_loss[row][mask[row]].sum() / math.log(2))
            records.append(dict(answer_bits=bits, corrupt_answer_bits=corrupt_bits,
                memory_gain_bits=corrupt_bits - bits, generated_correct=generated[row],
                text_nll=float(loss[row].sum() / (y[row] != 0).sum()),
                teacher_forced_correct=bool(((normal.logits[row].argmax(-1) == y[row]) | ~mask[row]).all())))
    mean = lambda k: sum(r[k] for r in records) / len(records)
    return dict(examples=len(records), answer_bits=mean("answer_bits"),
                memory_gain_bits=mean("memory_gain_bits"), text_nll=mean("text_nll"),
                generated_accuracy=mean("generated_correct"),
                teacher_forced_accuracy=mean("teacher_forced_correct"), records=records)
