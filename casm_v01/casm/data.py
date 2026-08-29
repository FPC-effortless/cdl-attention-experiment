from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch

PAD = 256
BOS = 257
EOS = 258
SEP = 259
VOCAB_SIZE = 260


def encode(text: str, max_len: int | None = None) -> List[int]:
    raw = list(text.encode("utf-8", errors="replace"))
    ids = [BOS] + raw + [EOS]
    if max_len is not None:
        ids = ids[:max_len]
        if ids[-1] != EOS:
            ids[-1] = EOS
    return ids


def decode(ids: Sequence[int]) -> str:
    b = bytes([i for i in ids if 0 <= i < 256])
    return b.decode("utf-8", errors="replace")


@dataclass
class Example:
    text: str
    task: str
    answer: str


def _names(n: int) -> List[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    out = []
    for i in range(n):
        out.append(alphabet[i % 26] + str(i // 26))
    return out


def associative_recall(rng: random.Random, hard: bool = False) -> Example:
    keys = _names(12 if hard else 7)
    rng.shuffle(keys)
    facts = []
    mapping = {}
    for k in keys:
        v = rng.randint(10, 99)
        mapping[k] = v
        facts.append(f"{k}={v}")
    q = rng.choice(keys)
    rng.shuffle(facts)
    text = f"task associative recall\nfacts {' '.join(facts)}\nquery {q}\nanswer {mapping[q]}"
    return Example(text, "assoc", str(mapping[q]))


def state_tracking(rng: random.Random, hard: bool = False) -> Example:
    items = ["red", "blue", "green", "gold", "white"]
    locs = ["box", "desk", "shelf", "bag", "tray", "room"]
    n = 12 if hard else 6
    state = {x: rng.choice(locs) for x in items}
    events = []
    for _ in range(n):
        obj = rng.choice(items)
        dst = rng.choice(locs)
        state[obj] = dst
        events.append(f"move {obj} to {dst}")
    q = rng.choice(items)
    text = f"task state tracking\n{' ; '.join(events)}\nwhere {q}\nanswer {state[q]}"
    return Example(text, "state", state[q])


def arithmetic(rng: random.Random, hard: bool = False) -> Example:
    if hard:
        a, b, c = [rng.randint(10, 99) for _ in range(3)]
        op = rng.choice(["plusminus", "sum3"])
        if op == "plusminus":
            ans = a + b - c
            expr = f"{a}+{b}-{c}"
        else:
            ans = a + b + c
            expr = f"{a}+{b}+{c}"
    else:
        a, b = rng.randint(0, 49), rng.randint(0, 49)
        if rng.random() < 0.5:
            ans, expr = a + b, f"{a}+{b}"
        else:
            hi, lo = max(a, b), min(a, b)
            ans, expr = hi - lo, f"{hi}-{lo}"
    text = f"task arithmetic\ncompute {expr}\nanswer {ans}"
    return Example(text, "arith", str(ans))


def rule_induction(rng: random.Random, hard: bool = False) -> Example:
    if hard:
        a = rng.randint(1, 9)
        mul = rng.choice([2, 3])
        add = rng.randint(1, 5)
        seq = [a]
        for _ in range(4):
            seq.append(seq[-1] * mul + add)
        ans = seq[-1] * mul + add
    else:
        start = rng.randint(0, 20)
        step = rng.randint(1, 9)
        seq = [start + i * step for i in range(5)]
        ans = seq[-1] + step
    text = f"task rule induction\nsequence {' '.join(map(str, seq))}\nnext\nanswer {ans}"
    return Example(text, "rule", str(ans))


def graph_reachability(rng: random.Random, hard: bool = False) -> Example:
    nodes = _names(9 if hard else 6)
    src = nodes[0]
    dst = nodes[4 if hard else 3]
    edges = []
    for i in range(4 if hard else 3):
        edges.append((nodes[i], nodes[i + 1]))
    for _ in range(5 if hard else 2):
        a, b = rng.sample(nodes, 2)
        if (a, b) not in edges:
            edges.append((a, b))
    rng.shuffle(edges)
    edge_txt = " ".join(f"{a}->{b}" for a, b in edges)
    text = f"task graph\nedges {edge_txt}\nreachable {src} {dst}\nanswer yes"
    return Example(text, "graph", "yes")


def reverse_copy(rng: random.Random, hard: bool = False) -> Example:
    n = 14 if hard else 7
    chars = "abcdefgxyz"
    s = "".join(rng.choice(chars) for _ in range(n))
    ans = s[::-1]
    text = f"task reverse\ninput {s}\nanswer {ans}"
    return Example(text, "reverse", ans)


TASKS = [associative_recall, state_tracking, arithmetic, rule_induction, graph_reachability, reverse_copy]


def generate_example(rng: random.Random, hard: bool = False) -> Example:
    fn = rng.choice(TASKS)
    return fn(rng, hard=hard)


def make_batch(
    batch_size: int,
    seq_len: int,
    seed: int,
    hard: bool = False,
    return_answer_mask: bool = False,
):
    """Pack tasks and optionally mark answer bytes in the token sequence."""
    rng = random.Random(seed)
    rows: List[List[int]] = []
    answer_rows: List[List[bool]] = []
    examples: List[Example] = []
    marker = b"answer "
    for _ in range(batch_size):
        row = [BOS]
        ans_mask = [False]
        first = True
        while len(row) < seq_len - 1:
            ex = generate_example(rng, hard=hard)
            body_bytes = ex.text.encode("utf-8", errors="replace")
            body = list(body_bytes)
            needed = len(body) + (0 if first else 1)
            if len(row) + needed + 1 > seq_len:
                break
            if not first:
                row.append(SEP)
                ans_mask.append(False)
            row.extend(body)
            body_mask = [False] * len(body)
            marker_at = body_bytes.rfind(marker)
            if marker_at >= 0:
                answer_start = marker_at + len(marker)
                answer_len = len(ex.answer.encode("utf-8", errors="replace"))
                for j in range(answer_start, min(answer_start + answer_len, len(body_mask))):
                    body_mask[j] = True
            ans_mask.extend(body_mask)
            examples.append(ex)
            first = False
        row.append(EOS)
        ans_mask.append(False)
        if len(row) < seq_len:
            pad_n = seq_len - len(row)
            row.extend([PAD] * pad_n)
            ans_mask.extend([False] * pad_n)
        rows.append(row[:seq_len])
        answer_rows.append(ans_mask[:seq_len])
    tok = torch.tensor(rows, dtype=torch.long)
    if return_answer_mask:
        return tok, examples, torch.tensor(answer_rows, dtype=torch.bool)
    return tok, examples


def gzip_teacher_distributions(tokens: torch.Tensor, chunk_size: int, memory_slots: int, state_slots: int, temperature: float = 1.0) -> torch.Tensor:
    """Cold-start ranking from incremental gzip description length."""
    import gzip

    src = tokens.detach().cpu() if tokens.device.type != "cpu" else tokens.detach()
    b, _ = src.shape
    x = src[:, :-1]
    total = x.shape[1]
    starts = list(range(0, total, chunk_size))
    out = torch.zeros((b, len(starts), state_slots + memory_slots), dtype=torch.float32)

    def raw(ids):
        vals = [int(v) for v in ids.tolist() if int(v) != PAD]
        return b"".join(v.to_bytes(2, "little", signed=False) for v in vals)

    for bi in range(b):
        chunks = [raw(x[bi, s:min(s + chunk_size, total)]) for s in starts]
        for j in range(len(chunks) - 1):
            future = chunks[j + 1]
            first = max(0, j - memory_slots + 1)
            retained = list(range(first, j + 1))
            ring_offset = memory_slots - len(retained)
            scores = torch.full((state_slots + memory_slots,), -1e9, dtype=torch.float32)
            for off, ci in enumerate(retained):
                mem = chunks[ci]
                delta = len(gzip.compress(mem + future, compresslevel=6)) - len(gzip.compress(mem, compresslevel=6)) if mem and future else 1e6
                scores[state_slots + ring_offset + off] = -float(delta)
            finite = scores > -1e8
            if finite.any():
                out[bi, j, finite] = torch.softmax(scores[finite] / max(temperature, 1e-6), dim=0)
            else:
                out[bi, j] = 1.0 / out.shape[-1]
        if len(chunks):
            out[bi, len(chunks) - 1, state_slots + memory_slots - 1] = 1.0
    return out
