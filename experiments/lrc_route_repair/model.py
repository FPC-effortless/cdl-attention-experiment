"""Learned fine keys and bounded content-aware retrieval for sentence LRC.

This is an architectural/training revision of the sentence-level adapter, not
a drop-in byte-language-model implementation. Inference takes only tokenized
records and a query. Supervision is confined to train.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from lrc_vb001_comparison.source.study import SentenceEncoder
from .index import BoundedKeyIndex


@dataclass
class PreparedMemory:
    values: torch.Tensor
    keys: torch.Tensor
    codes: torch.Tensor
    indexes: list[BoundedKeyIndex]
    code_divisor: int


@dataclass
class RouteOutput:
    logits: torch.Tensor
    selected: torch.Tensor
    writer_codes: torch.Tensor
    read_codes: torch.Tensor
    inspected: torch.Tensor
    returned: torch.Tensor
    overflow: torch.Tensor
    bucket_size: torch.Tensor
    probes: torch.Tensor
    ties: torch.Tensor


class LearnedKeyRouter(nn.Module):
    def __init__(self, width=32, codes=16):
        super().__init__()
        self.encoder = SentenceEncoder(width)
        self.writer = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.first = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.second = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.register_buffer("codebook", torch.zeros(codes, width))

    def write(self, records):
        values = self.encoder(records)
        return values, F.normalize(self.writer(values), dim=-1)

    def query_first(self, query):
        state = self.encoder(query)
        return state, F.normalize(self.first(state), dim=-1)

    def query_second(self, state, value):
        return F.normalize(self.second(torch.cat((state, value), -1)), dim=-1)


class ContentAddressedLRC(nn.Module):
    """Two dependent hard reads; the answer sees only the final selected value.

    The fine key preserves distinctions that an intentionally coarser address
    may merge. The shared learned codebook supplies integer index addresses.
    Ties in content scores fail closed rather than selecting by record order.
    No forward argument accepts support labels, person IDs, or era metadata.
    """
    def __init__(self, width=32, codes=16, probes=2, capacity=2):
        super().__init__()
        if not 1 <= probes <= codes or capacity < 1:
            raise ValueError("invalid probe/candidate budget")
        self.router = LearnedKeyRouter(width, codes)
        self.answer = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4))
        self.probe_count = probes
        self.capacity = capacity

    def prepare_memory(self, records, *, code_divisor=1):
        if records.ndim != 3 or code_divisor < 1:
            raise ValueError("expected batch/record/token tensor and positive divisor")
        values, keys = self.router.write(records)
        codes = (keys @ self.router.codebook.T).argmax(-1) // code_divisor
        indexes = [BoundedKeyIndex(row, max_candidates=self.capacity) for row in codes]
        return PreparedMemory(values, keys, codes, indexes, code_divisor)

    def query_memory(self, memory, query, *, resolver="content", probe_count=None):
        """Score only returned candidates, never all record keys at query time.

        resolver='ordinal' is an explicitly named diagnostic ablation: choose
        the lowest occurrence number from the bounded returned set. It is
        never the default. code_divisor=2 is a paired-collision stress test.
        """
        if resolver not in ("content", "ordinal"):
            raise ValueError("unknown collision resolver")
        count = self.probe_count if probe_count is None else probe_count
        if not 1 <= count <= self.router.codebook.shape[0]:
            raise ValueError("invalid probe count")
        states, first = self.router.query_first(query)
        batch = len(query)
        if len(memory.indexes) != batch:
            raise ValueError("query batch differs from prepared memory")
        selected = torch.full((batch, 2), -1, dtype=torch.long, device=query.device)
        inspected = torch.zeros_like(selected)
        returned = torch.zeros_like(selected)
        overflow = torch.zeros_like(selected)
        bucket_size = torch.zeros_like(selected)
        probes = torch.zeros_like(selected)
        ties = torch.zeros_like(selected)
        read_codes = torch.full((batch, 2, count), -1, dtype=torch.long, device=query.device)
        final = torch.zeros_like(states)
        for row in range(batch):
            read_key = first[row]
            for hop in range(2):
                # Stable code-index tie breaking depends on the codebook only,
                # never the record order. At most count codes are considered.
                scores = read_key @ self.router.codebook.T
                requested = scores.argsort(descending=True, stable=True)[:count].tolist()
                requested = list(dict.fromkeys(code // memory.code_divisor for code in requested))
                candidates = []
                for offset, code in enumerate(requested):
                    hit = memory.indexes[row].lookup(code)
                    read_codes[row, hop, offset] = code
                    probes[row, hop] += 1
                    overflow[row, hop] += int(hit.overflow)
                    bucket_size[row, hop] += hit.bucket_size
                    candidates.extend(hit.candidates)
                candidates = list(dict.fromkeys(candidates))
                returned[row, hop] = len(candidates)
                if len(candidates) > count * self.capacity:
                    raise AssertionError("candidate budget violated")
                chosen = -1
                if candidates and resolver == "ordinal":
                    chosen = min(candidates)
                elif candidates:
                    candidate_keys = memory.keys[row, candidates]
                    candidate_scores = candidate_keys @ read_key
                    inspected[row, hop] = len(candidate_scores)
                    winners = candidate_scores.eq(candidate_scores.max()).nonzero().flatten()
                    if len(winners) == 1:
                        chosen = candidates[int(winners[0])]
                    else:
                        ties[row, hop] = 1
                selected[row, hop] = chosen
                value = memory.values[row, chosen] if chosen >= 0 else torch.zeros_like(states[row])
                if hop == 0:
                    if chosen < 0:
                        # A dependent chain cannot continue without a bridge.
                        # Keep hop 2 null with zero probes and a zero final value.
                        break
                    read_key = self.router.query_second(states[row], value)
                else:
                    final[row] = value
        return RouteOutput(self.answer(final), selected, memory.codes, read_codes,
                           inspected, returned, overflow, bucket_size, probes, ties)

    def forward(self, records, query):
        return self.query_memory(self.prepare_memory(records), query)
