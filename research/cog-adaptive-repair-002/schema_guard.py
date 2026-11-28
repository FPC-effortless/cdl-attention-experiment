"""Fail-closed policy for the narrow annotated English templates.

Only a trained role assignment that agrees with a source-template signature
may read or write the 32-slot auxiliary store. An unsupported completed clause
invalidates the stored knowledge for this stream, including after overflow.
The original language model supplies all outputs without an eligible read.
"""
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from panels import DIGITS, PROSE
from structured_memory import (BOUNDARY, CAPACITY, CLAUSE_BYTES, KEY_BYTES,
                               READ, TRUST, WEAK, StructuredCOG, fuse, parse_many)

# Matching is case-insensitive for English literals, exact on punctuation.
# Include every non-whitespace byte, so unexpected characters cannot disappear.
LEX = re.compile(rb"[a-zA-Z]+|[0-9]+|[^\sA-Za-z0-9]", re.ASCII)


@dataclass(frozen=True)
class Signature:
    action: int
    parts: tuple


def compile_signatures():
    signatures = []
    for family in PROSE[:2]:
        for action, fmt in zip((TRUST, WEAK, READ), family):
            # Placeholders come directly from the training annotation fields.
            rendered = fmt.format(name="SUBJECTPLACEHOLDER", v="VALUEPLACEHOLDER").strip().encode("ascii")
            parts = tuple(LEX.findall(rendered.lower()))
            if parts.count(b"subjectplaceholder") != 1 or parts.count(b"valueplaceholder") != int(action != READ):
                raise ValueError("Unrecognized annotation layout")
            signatures.append(Signature(action, parts))
    return tuple(signatures)


SIGNATURES = compile_signatures()


def validate(raw, parsed):
    action, subject, value, _ = parsed
    if subject is None or len(subject) > KEY_BYTES or not subject.isalpha():
        return False
    if not all(32 <= b <= 126 or b in (9,10,13) for b in raw):
        return False
    actual = LEX.findall(raw.lower())
    for sig in SIGNATURES:
        if action != sig.action or len(actual) != len(sig.parts):
            continue
        for part, observed in zip(sig.parts, actual):
            if part == b"subjectplaceholder":
                valid = observed == subject
            elif part == b"valueplaceholder":
                valid = len(observed) == 1 and observed[0] in DIGITS and observed[0] == value
            else:
                valid = observed == part
            if not valid:
                break
        else:
            return True
    return False


class GuardState(NamedTuple):
    entries: tuple
    pending: bytes
    overflow: bool


class GuardedStore:
    def __init__(self, tagger, state=None):
        self.tagger = tagger
        entries, pending, overflow = state if state is not None else ((), b"", False)
        self.bank = OrderedDict(entries)
        self.pending = bytearray(pending)
        self.overflow = overflow

    def snapshot(self):
        return GuardState(tuple(self.bank.items()), bytes(self.pending), self.overflow)

    def apply(self, raw, parsed):
        if not validate(raw, parsed):
            self.bank.clear()
            return None
        action, key, value, _ = parsed
        if action == TRUST:
            if key not in self.bank and len(self.bank) == CAPACITY:
                self.bank.popitem(last=False)
            self.bank[key] = value
        if action == READ:
            return self.bank.get(key)
        return None

    def step(self, b):
        if not 0 <= b <= 255:
            self.bank.clear()
            self.pending.clear()
            self.overflow = False
            return None
        if len(self.pending) < CLAUSE_BYTES:
            self.pending.append(b)
        else:
            self.overflow = True
        if b not in BOUNDARY:
            return None
        raw = bytes(self.pending).strip()
        self.pending.clear()
        if self.overflow:
            self.overflow = False
            self.bank.clear()
            return None
        return self.apply(raw, parse_many(self.tagger, [raw])[raw])


def guarded_actions(raw, parsed, bank=None):
    """Cached offline decisions; includes the same overflow/invalidations as step."""
    s = GuardedStore(None)
    if bank is not None:
        s.bank = bank
    start = 0
    result = {}
    for i, b in enumerate(raw):
        if b in BOUNDARY:
            if i - start + 1 > CLAUSE_BYTES:
                s.bank.clear()
            else:
                clause = raw[start:i+1].strip()
                guess = s.apply(clause, parsed[clause])
                if guess is not None:
                    result[i] = guess
            start = i+1
    return result


class GuardedStructuredCOG(nn.Module):
    def __init__(self, base, tagger):
        super().__init__()
        self.base = base
        self.tagger = tagger
        self.disable_read = False

    @torch.inference_mode()
    def forward(self, x, state=None, carry=False):
        prior = None if state is None else state[0]
        logits, base_state = self.base(x, prior, carry=True)
        next_rows, outputs = [], []
        for i, row in enumerate(x.tolist()):
            s = GuardedStore(self.tagger, None if state is None else state[1][i])
            edits = {}
            for j, b in enumerate(row):
                value = s.step(b)
                if value is not None:
                    edits[j] = value
            outputs.append(logits[i] if self.disable_read else fuse(logits[i], edits))
            next_rows.append(s.snapshot())
        outputs = torch.stack(outputs)
        return (outputs, (base_state, tuple(next_rows))) if carry else outputs


def build_adapter(base, tagger, mode="guarded"):
    if mode == "guarded":
        return GuardedStructuredCOG(base, tagger)
    if mode == "raw":
        return StructuredCOG(base, tagger)
    raise ValueError("mode must be guarded or raw")
