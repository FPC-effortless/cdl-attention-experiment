"""LRR-003B: incremental causal state and bounded repeated-query inference.

The underlying causal convolution has receptive field:
  R = 1 + 2*(1 + 2 + 4) = 15 tokens.
Retain R-1=14 prior tokens for correct next-token local computation.
The index is built incrementally, not reconstructed for every question.

This supports *a separate query boundary*, NOT natural-language question
recognition inside a mixed stream. It does not discover entity semantics.
"""
from __future__ import annotations

import time

import torch

from .raw_causal import CausalRoleIndexer, IDS, tokenize


class CachedCausalIndex:
    def __init__(self, model: CausalRoleIndexer):
        if len(model.convs) != 3:
            raise ValueError("unexpected encoder architecture")
        self.model = model.eval()
        self.receptive_field = 1 + sum(2 * conv.dilation[0] for conv in model.convs)
        self.tail_size = self.receptive_field - 1
        self.history_tokens: list[str] = []
        self.tail: tuple[str, ...] = ()
        self.postings: dict[tuple[str, int], list[int]] = {}
        self.write_count = 0

    def _forward(self, tokens: tuple[str, ...]):
        x = torch.tensor([[IDS.get(word, IDS["<unk>"]) for word in tokens]],
                         dtype=torch.long)
        with torch.inference_mode():
            return self.model(x)

    @staticmethod
    def _address(tokens, ent_argmax, rel_argmax, codes_argmax, j):
        ei = j - 1 - ent_argmax[j]
        ri = j - 1 - rel_argmax[j]
        if ei < 0 or ri < 0:
            return None
        return (tokens[ei], codes_argmax[j][rel_argmax[j]])

    @torch.no_grad()
    def append(self, raw_text: str) -> dict:
        """Append new raw observed tokens once and update persistent index."""
        incoming = tokenize(raw_text)
        if not incoming:
            return {"appended": 0, "writes": 0}
        if "?" in incoming:
            raise ValueError("questions belong to query(), not observed history")
        begin = time.perf_counter_ns()
        context = self.tail + incoming
        ent, rel, codes, gate, _ = self._forward(context)
        ent_ids = ent[0].argmax(-1).tolist()
        rel_ids = rel[0].argmax(-1).tolist()
        code_ids = codes[0].argmax(-1).tolist()
        start = len(self.tail)
        old_length = len(self.history_tokens)
        new_writes = 0
        for i in range(start, len(context)):
            if float(gate[0, i].sigmoid()) < 0.5:
                continue
            address = self._address(context, ent_ids, rel_ids, code_ids, i)
            if address is None:
                continue
            absolute = old_length + (i - start)
            self.postings.setdefault(address, []).append(absolute)
            new_writes += 1
        self.history_tokens.extend(incoming)
        self.tail = tuple(context[-self.tail_size:])
        self.write_count += new_writes
        return {
            "appended": len(incoming),
            "writes": new_writes,
            "observed_total": len(self.history_tokens),
            "distinct_addresses": len(self.postings),
            "elapsed_ingest_cpu_us": (time.perf_counter_ns()-begin)/1000.0,
        }

    @torch.no_grad()
    def query(self, question_text: str, *, k: int = 1) -> dict:
        """Only encode local question/suffix, then one posting-list lookup."""
        if k < 1:
            raise ValueError("k must be positive")
        incoming = tokenize(question_text)
        if not incoming or incoming[-1] != "?":
            raise ValueError("query must end with question mark")
        begin = time.perf_counter_ns()
        context = self.tail + incoming
        ent, rel, codes, _, _ = self._forward(context)
        ent_ids = ent[0].argmax(-1).tolist()
        rel_ids = rel[0].argmax(-1).tolist()
        code_ids = codes[0].argmax(-1).tolist()
        pos = len(context) - 1
        address = self._address(context, ent_ids, rel_ids, code_ids, pos)
        positions = tuple(reversed(self.postings.get(address, [])[-k:])) if address else ()
        if any(i >= len(self.history_tokens) for i in positions):
            raise AssertionError("cache returned future/question position")
        elapsed = (time.perf_counter_ns()-begin)/1000.0
        return {
            "token_indices": positions,
            "tokens": tuple(self.history_tokens[i] for i in positions),
            "query_address": address,
            "index_probes": int(address is not None),
            "observed_history_length": len(self.history_tokens),
            "encoded_token_count": len(context),
            "elapsed_query_cpu_us": elapsed,
        }
