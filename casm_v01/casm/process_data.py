from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch

from .data import BOS, EOS, PAD, Example


@dataclass
class ProcessExample:
    example: Example
    traces: List[str]


def _names(n: int) -> List[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return [alphabet[i % 26] + str(i // 26) for i in range(n)]


def _checkpoints(n: int, steps: int = 3) -> List[int]:
    return [max(1, math.ceil((i + 1) * n / steps)) for i in range(steps)]


def associative_process(rng: random.Random, hard: bool = False) -> ProcessExample:
    keys = _names(12 if hard else 7)
    rng.shuffle(keys)
    mapping = {k: rng.randint(10, 99) for k in keys}
    facts = [f"{k}={v}" for k, v in mapping.items()]
    q = rng.choice(keys)
    rng.shuffle(facts)
    ans = str(mapping[q])
    ex = Example(
        f"task associative recall\nfacts {' '.join(facts)}\nquery {q}\nanswer {ans}",
        "assoc",
        ans,
    )
    return ProcessExample(ex, [f"query {q}", f"fact {q}={ans}", f"answer {ans}"])


def state_process(rng: random.Random, hard: bool = False) -> ProcessExample:
    """State tracking with all information required for the answer visible.

    The legacy generator sampled random initial locations but omitted them from
    the prompt. This corrected process curriculum serializes the initial state
    and then provides verifier-computed intermediate states for the queried
    object at three event checkpoints.
    """
    items = ["red", "blue", "green", "gold", "white"]
    locs = ["box", "desk", "shelf", "bag", "tray", "room"]
    n = 12 if hard else 6
    initial = {x: rng.choice(locs) for x in items}
    state = dict(initial)
    events = []
    snapshots = []
    for _ in range(n):
        obj = rng.choice(items)
        dst = rng.choice(locs)
        state[obj] = dst
        events.append((obj, dst))
        snapshots.append(dict(state))
    q = rng.choice(items)
    initial_txt = " ".join(f"{k}={initial[k]}" for k in items)
    event_txt = " ; ".join(f"move {obj} to {dst}" for obj, dst in events)
    ans = state[q]
    ex = Example(
        f"task state tracking\ninitial {initial_txt}\n{event_txt}\nwhere {q}\nanswer {ans}",
        "state",
        ans,
    )
    traces = [f"{q}={snapshots[i - 1][q]}" for i in _checkpoints(n)]
    return ProcessExample(ex, traces)


def arithmetic_process(rng: random.Random, hard: bool = False) -> ProcessExample:
    if hard:
        a, b, c = [rng.randint(10, 99) for _ in range(3)]
        if rng.random() < 0.5:
            first = a + b
            ans = first - c
            expr = f"{a}+{b}-{c}"
            traces = [f"partial {a}+{b}={first}", f"partial {first}-{c}={ans}", f"answer {ans}"]
        else:
            first = a + b
            ans = first + c
            expr = f"{a}+{b}+{c}"
            traces = [f"partial {a}+{b}={first}", f"partial {first}+{c}={ans}", f"answer {ans}"]
    else:
        a, b = rng.randint(0, 49), rng.randint(0, 49)
        if rng.random() < 0.5:
            ans = a + b
            expr = f"{a}+{b}"
            traces = [f"left {a}", f"compute {expr}={ans}", f"answer {ans}"]
        else:
            hi, lo = max(a, b), min(a, b)
            ans = hi - lo
            expr = f"{hi}-{lo}"
            traces = [f"left {hi}", f"compute {expr}={ans}", f"answer {ans}"]
    ans_s = str(ans)
    return ProcessExample(Example(f"task arithmetic\ncompute {expr}\nanswer {ans_s}", "arith", ans_s), traces)


def rule_process(rng: random.Random, hard: bool = False) -> ProcessExample:
    if hard:
        a = rng.randint(1, 9)
        mul = rng.choice([2, 3])
        add = rng.randint(1, 5)
        seq = [a]
        for _ in range(4):
            seq.append(seq[-1] * mul + add)
        ans = seq[-1] * mul + add
        traces = [f"rule multiply {mul}", f"rule multiply {mul} add {add}", f"answer {ans}"]
    else:
        start = rng.randint(0, 20)
        step = rng.randint(1, 9)
        seq = [start + i * step for i in range(5)]
        ans = seq[-1] + step
        traces = [f"difference {step}", f"last {seq[-1]} plus {step}", f"answer {ans}"]
    ans_s = str(ans)
    ex = Example(
        f"task rule induction\nsequence {' '.join(map(str, seq))}\nnext\nanswer {ans_s}",
        "rule",
        ans_s,
    )
    return ProcessExample(ex, traces)


def _reachable_within(nodes: Sequence[str], edges: Sequence[tuple[str, str]], src: str, depth: int) -> List[str]:
    reached = {src}
    frontier = {src}
    for _ in range(depth):
        nxt = {b for a, b in edges if a in frontier and b not in reached}
        if not nxt:
            break
        reached.update(nxt)
        frontier = nxt
    return sorted(reached)


def graph_process(rng: random.Random, hard: bool = False) -> ProcessExample:
    nodes = _names(10 if hard else 7)
    reachable = rng.random() < 0.5
    edges: set[tuple[str, str]] = set()
    if reachable:
        src, dst = nodes[0], nodes[-1]
        middle = nodes[1:-1].copy()
        rng.shuffle(middle)
        chain_len = 5 if hard else 3
        chain = [src] + middle[:chain_len] + [dst]
        edges.update(zip(chain[:-1], chain[1:]))
        target_edges = 14 if hard else 8
        while len(edges) < target_edges:
            a, b = rng.sample(nodes, 2)
            edges.add((a, b))
        ans = "yes"
    else:
        cut = len(nodes) // 2
        left, right = nodes[:cut], nodes[cut:]
        src, dst = left[0], right[-1]
        target_edges = 14 if hard else 8
        attempts = 0
        while len(edges) < target_edges and attempts < target_edges * 30:
            comp = left if rng.random() < 0.5 else right
            if len(comp) >= 2:
                a, b = rng.sample(comp, 2)
                edges.add((a, b))
            attempts += 1
        ans = "no"
    edge_list = list(edges)
    rng.shuffle(edge_list)
    edge_txt = " ".join(f"{a}->{b}" for a, b in edge_list)
    ex = Example(
        f"task graph\nedges {edge_txt}\nreachable {src} {dst}\nanswer {ans}",
        "graph",
        ans,
    )
    max_depth = len(nodes) - 1
    depths = _checkpoints(max_depth)
    traces = []
    for i, d in enumerate(depths):
        reached = _reachable_within(nodes, edge_list, src, d)
        suffix = f" answer {ans}" if i == len(depths) - 1 else ""
        traces.append(f"reachable {' '.join(reached)}{suffix}")
    return ProcessExample(ex, traces)


def reverse_process(rng: random.Random, hard: bool = False) -> ProcessExample:
    n = 14 if hard else 7
    chars = "abcdefgxyz"
    s = "".join(rng.choice(chars) for _ in range(n))
    rev = s[::-1]
    cps = _checkpoints(len(rev))
    traces = [f"reverse {rev[:i]}" for i in cps]
    ex = Example(f"task reverse\ninput {s}\nanswer {rev}", "reverse", rev)
    return ProcessExample(ex, traces)


PROCESS_TASKS: List[Callable[[random.Random, bool], ProcessExample]] = [
    associative_process,
    state_process,
    arithmetic_process,
    rule_process,
    graph_process,
    reverse_process,
]


def generate_process_example(rng: random.Random, hard: bool = False) -> ProcessExample:
    return rng.choice(PROCESS_TASKS)(rng, hard)


def make_process_batch(
    batch_size: int,
    seq_len: int,
    seed: int,
    *,
    hard: bool = False,
    reasoning_steps: int = 3,
):
    """One verifier-complete episode per row plus latent-process metadata."""
    if reasoning_steps != 3:
        raise ValueError("process curriculum currently defines exactly 3 verifier checkpoints")
    rng = random.Random(seed)
    rows: List[List[int]] = []
    masks: List[List[bool]] = []
    anchors: List[int] = []
    process_examples: List[ProcessExample] = []
    marker = b"answer "

    for _ in range(batch_size):
        for _attempt in range(2000):
            pex = generate_process_example(rng, hard=hard)
            body = pex.example.text.encode("utf-8", errors="replace")
            if len(body) + 2 <= seq_len:
                break
        else:
            raise RuntimeError(f"could not sample process episode fitting seq_len={seq_len}")

        marker_at = body.rfind(marker)
        if marker_at < 0:
            raise ValueError("process episode has no answer marker")
        answer_start = 1 + marker_at + len(marker)
        anchor = answer_start - 1
        answer_len = len(pex.example.answer.encode("utf-8", errors="replace"))

        row = [BOS] + list(body) + [EOS]
        mask = [False] * len(row)
        for j in range(answer_start, min(answer_start + answer_len, len(row) - 1)):
            mask[j] = True
        pad = seq_len - len(row)
        row.extend([PAD] * pad)
        mask.extend([False] * pad)
        rows.append(row)
        masks.append(mask)
        anchors.append(anchor)
        process_examples.append(pex)

    return (
        torch.tensor(rows, dtype=torch.long),
        process_examples,
        torch.tensor(masks, dtype=torch.bool),
        torch.tensor(anchors, dtype=torch.long),
    )


def corrected_state_long(rng: random.Random, n_events: int) -> Example:
    items = ["red", "blue", "green", "gold", "white", "black", "silver", "orange"]
    locs = ["box", "desk", "shelf", "bag", "tray", "room", "vault", "yard", "cart", "case"]
    initial = {x: rng.choice(locs) for x in items}
    state = dict(initial)
    events = []
    for _ in range(n_events):
        obj = rng.choice(items)
        dst = rng.choice(locs)
        state[obj] = dst
        events.append(f"move {obj} to {dst}")
    q = rng.choice(items)
    initial_txt = " ".join(f"{k}={initial[k]}" for k in items)
    return Example(
        f"task state tracking\ninitial {initial_txt}\n{' ; '.join(events)}\nwhere {q}\nanswer {state[q]}",
        "state_long",
        state[q],
    )


def associative_long(rng: random.Random, n_keys: int) -> Example:
    keys = [f"k{i}" for i in range(n_keys)]
    mapping = {k: rng.randint(10, 99) for k in keys}
    facts = [f"{k}={v}" for k, v in mapping.items()]
    rng.shuffle(facts)
    q = rng.choice(keys)
    ans = str(mapping[q])
    return Example(
        f"task associative recall\nfacts {' '.join(facts)}\nquery {q}\nanswer {ans}",
        "assoc_long",
        ans,
    )
