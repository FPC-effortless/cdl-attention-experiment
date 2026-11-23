"""Label-free in-memory execution used identically for training and evaluation.

Training may explicitly substitute supervised operations. Public candidate
execution only receives texts and logits. The archive holds typed identities;
its deterministic rules are part of the architecture, not learned behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

from .data import UNKNOWN


def pad_texts(texts: list[str], chars: list[str] | tuple[str, ...]):
    stoi = {c: i for i, c in enumerate(chars)}
    lengths = torch.tensor([len(s) for s in texts], dtype=torch.long)
    out = torch.full((len(texts), int(lengths.max())), stoi[" "], dtype=torch.long)
    for i, text in enumerate(texts):
        try:
            out[i, :len(text)] = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        except KeyError as exc:
            raise ValueError(f"character outside inherited tokenizer: {exc.args[0]!r}") from exc
    return out, lengths


def predicted_operation(semantics: dict, index: int) -> tuple[int, int, int, int]:
    return tuple(int(semantics[k][index].argmax()) for k in ("operation", "entity", "relation", "value"))


def execute(ledger: dict, operation: tuple[int, int, int, int]):
    op, e, r, v = operation
    if op == 1:
        ledger[(e, r)] = v
    elif op == 2:
        ledger.pop((e, r), None)


def bucket(entity: int, relation: int) -> int:
    return (entity + 7 * relation) % 16


@dataclass
class Resolution:
    values: list[int]
    keys: list[tuple[int, int]]
    status: str
    candidates_fetched: int = 0
    codebook_comparisons: int = 0
    index_records: int = 0


def strict_read(ledger: dict, entity: int, plan: int) -> Resolution:
    key = (entity, 1 if plan in (1, 2) else 0)
    keys = [key]
    value = ledger.get(key, UNKNOWN)
    values = [value]
    if plan == 2:
        if value != UNKNOWN:
            keys.append((value, 0))
            value = ledger.get(keys[-1], UNKNOWN)
        values.append(value)
    return Resolution(values, keys, "ok" if value != UNKNOWN else "missing", len(keys))


def hybrid_read(model, feature: torch.Tensor, ledger: dict, plan: int, *, tie_tolerance=1e-7) -> Resolution:
    # This small reference builds an index per query and explicitly charges all
    # indexed records. Runtime may maintain it incrementally in a future version.
    index: dict[int, list[tuple[int, int]]] = {}
    for key in ledger:
        index.setdefault(bucket(*key), []).append(key)
    values, keys, fetched = [], [], 0
    previous = None
    for hop in range(2 if plan == 2 else 1):
        logits, vector = model.route_logits(feature, previous_value=previous, hop=hop)
        addresses = logits[0].topk(2).indices.tolist()
        candidates = []
        for address in addresses:
            group = index.get(address, [])
            if len(group) > 2:
                return Resolution([UNKNOWN], keys, "overflow", fetched, 16 * (hop + 1), len(ledger))
            candidates.extend(group)
        fetched += len(candidates)
        if not candidates:
            return Resolution([UNKNOWN], keys, "missing", fetched, 16 * (hop + 1), len(ledger))
        # <=4 actual fetched candidate records; there is no all-record top-k.
        es = torch.tensor([[k[0] for k in candidates]], dtype=torch.long)
        rs = torch.tensor([[k[1] for k in candidates]], dtype=torch.long)
        scores = model.candidate_scores(vector, es, rs)[0]
        ordered = scores.argsort(descending=True)
        if len(candidates) > 1 and abs(float(scores[ordered[0]] - scores[ordered[1]])) <= tie_tolerance:
            return Resolution([UNKNOWN], keys, "tie", fetched, 16 * (hop + 1), len(ledger))
        key = candidates[int(ordered[0])]
        value = ledger[key]
        keys.append(key)
        values.append(value)
        if value == UNKNOWN:
            break
        previous = torch.tensor([value], dtype=torch.long)
    return Resolution(values, keys, "ok" if values[-1] != UNKNOWN else "missing", fetched,
                      16 * len(values), len(ledger))


def evidence_tensors(resolutions: list[Resolution]):
    values = torch.full((len(resolutions), 2), UNKNOWN, dtype=torch.long)
    masks = torch.zeros(len(resolutions), 2, dtype=torch.bool)
    for i, res in enumerate(resolutions):
        vals = res.values[-2:]
        values[i, :len(vals)] = torch.tensor(vals)
        masks[i, :len(vals)] = True
    return values, masks


def greedy_generate(model, state, resolutions, *, max_chars=12, memory_enabled=None):
    """Generate from the whole inherited vocabulary; no choices or gold length.

    Each emitted character, including the terminator, is consumed exactly once.
    A finished row is removed from subsequent decoder calls, so its saved state
    cannot contain invisible characters while other rows are still answering.
    """
    if type(max_chars) is not int or not 1 <= max_chars <= 12:
        raise ValueError("generation is bounded to one through twelve characters")

    def select_rows(value, indices):
        if isinstance(value, torch.Tensor):
            return value.index_select(0, indices.to(value.device))
        if isinstance(value, dict):
            return {key: select_rows(item, indices) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(select_rows(item, indices) for item in value)
        if isinstance(value, list):
            return [select_rows(item, indices) for item in value]
        raise TypeError("decoder state must consist of tensors and containers")

    def publish_rows(previous, updated, indices):
        if isinstance(previous, torch.Tensor):
            return previous.index_copy(0, indices.to(previous.device), updated)
        if isinstance(previous, dict):
            return {key: publish_rows(item, updated[key], indices) for key, item in previous.items()}
        if isinstance(previous, tuple):
            return tuple(publish_rows(item, updated[i], indices) for i, item in enumerate(previous))
        if isinstance(previous, list):
            return [publish_rows(item, updated[i], indices) for i, item in enumerate(previous)]
        raise TypeError("decoder state must consist of tensors and containers")

    chars = list(model.chars)
    prefix = "\nAnswer: "
    tokens, _ = pad_texts([prefix] * len(resolutions), chars)
    values, masks = evidence_tensors(resolutions)
    logits, state = model.decode(tokens, state, values, masks, memory_enabled=memory_enabled)
    next_logits = logits[:, -1]
    generated = [""] * len(resolutions)
    complete = [False] * len(resolutions)
    for _ in range(max_chars):
        active = torch.tensor([i for i, done in enumerate(complete) if not done],
                              dtype=torch.long, device=next_logits.device)
        ids = next_logits.index_select(0, active).argmax(-1)
        for i, token in zip(active.tolist(), ids.tolist()):
            char = chars[token]
            generated[i] += char
            if char == ".":
                complete[i] = True
        logits, updated = model.decode(ids[:, None], select_rows(state, active),
                                       values.index_select(0, active), masks.index_select(0, active),
                                       memory_enabled=memory_enabled)
        state = publish_rows(state, updated, active)
        next_logits = next_logits.index_copy(0, active, logits[:, -1])
        if all(complete):
            break
    return generated, state
