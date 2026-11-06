"""LRR-005 delayed mechanistic integrity gate: chunked causal register cache.

The model's single causal Conv1d kernel of length 3 needs a 2-token suffix.
This cache implements exactly the same hard gate/write order as route(words)
while retaining the built address table across appended observed tokens.
It is NOT a learned semantic repair.
"""
from __future__ import annotations
import time
import torch

from .lrr005_register import CausalRoleRegister, ID


class CachedRoleRegister:
    def __init__(self, model: CausalRoleRegister):
        self.model = model.eval()
        self.tail: tuple[str, ...] = ()
        self.tail_size = self.model.conv.kernel_size[0] - 1
        self.entity: str | None = None
        self.relation: int | None = None
        self.postings: dict[tuple[str, int], int] = {}
        self.tokens: list[str] = []
        self.writes = 0

    def _advance(self, incoming, *, entity, relation, postings,
                 absolute_start, commit: bool):
        words = self.tail + tuple(incoming)
        # A history length zero still permits an empty append call.
        if not incoming:
            return entity, relation, 0, {}
        x = torch.tensor([[ID.get(word, 1) for word in words]], dtype=torch.long)
        with torch.inference_mode():
            h = self.model.token_features(x)
            gates = self.model.role(h).sigmoid()[0].tolist()
            codes = self.model.code(h)[0].argmax(-1).tolist()
        offset = len(self.tail)
        newly_written = {}
        nwrites = 0
        for j in range(offset, len(words)):
            token = words[j]
            absolute_position = absolute_start + (j - offset)
            if gates[j][2] > .5 and entity is not None and relation is not None:
                key = (entity, relation)
                if commit:
                    postings[key] = absolute_position
                else:
                    newly_written[key] = absolute_position
                nwrites += 1
            if gates[j][0] > .5:
                entity = token
            if gates[j][1] > .5:
                relation = codes[j]
        return entity, relation, nwrites, newly_written

    @torch.no_grad()
    def append(self, words: tuple[str, ...]):
        if not words:
            return {"appended": 0, "new_writes": 0}
        if "?" in words:
            raise ValueError("query tokens must not be permanently appended as facts")
        old_length = len(self.tokens)
        next_entity, next_relation, nwrites, _ = self._advance(
            words, entity=self.entity, relation=self.relation,
            postings=self.postings, absolute_start=old_length, commit=True)
        self.entity, self.relation = next_entity, next_relation
        self.tokens.extend(words)
        self.tail = (self.tail + words)[-self.tail_size:]
        self.writes += nwrites
        return {"appended": len(words), "new_writes": nwrites,
                "total_tokens": len(self.tokens), "total_writes": self.writes}

    @torch.no_grad()
    def query(self, words: tuple[str, ...]):
        if not words or words[-1] != "?":
            raise ValueError("query must end with ?")
        start = time.perf_counter_ns()
        # The terminal '?' is only a query boundary. Source hard route()
        # does not consume or write it before reading the final address.
        entity, relation, _, local = self._advance(
            words[:-1], entity=self.entity, relation=self.relation,
            postings=self.postings, absolute_start=len(self.tokens), commit=False)
        key = (entity, relation)
        result = local.get(key, self.postings.get(key))
        if result is not None and result >= len(self.tokens)+len(words)-1:
            raise AssertionError("future token")
        return {"token_index": result, "address": key, "index_probes": 1,
                "elapsed_query_cpu_us": (time.perf_counter_ns()-start)/1000.0,
                "encoded_tokens": len(self.tail)+len(words)-1,
                "history_length": len(self.tokens)}
