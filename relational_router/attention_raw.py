"""Parameter-matched causal Transformer pointer comparator for LRR-003.

This *does* use self-attention and dense query-to-position pointer scoring,
on the same answer-derived positions and training-document streams as LRR.
"""
from __future__ import annotations

import math
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .raw_causal import IDS, VOCAB, tokenize, generate, examples_to_batch


class CausalAttentionPointer(nn.Module):
    def __init__(self, width: int = 48, feedforward: int = 80, heads: int = 4):
        super().__init__()
        self.width = width
        self.embed = nn.Embedding(len(VOCAB), width, padding_idx=0)
        self.layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feedforward,
            dropout=0.0, batch_first=True)
        self.query_proj = nn.Linear(width, width)
        self.key_proj = nn.Linear(width, width)

    def pointer_logits(self, ids: torch.Tensor, query_index: torch.Tensor) -> torch.Tensor:
        batch, length = ids.shape
        # Fixed, nonlearned sinusoidal position encoding; no maximum context
        # position or positional embedding is trained on the short data.
        position = torch.arange(length, device=ids.device, dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, self.width, 2, device=ids.device,
                                       dtype=torch.float32) *
                          (-math.log(10000.0) / self.width))
        encoding = torch.zeros(length, self.width, device=ids.device)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        hidden = self.embed(ids) + encoding[None, :, :]
        causal_mask = torch.ones((length, length), device=ids.device,
                                  dtype=torch.bool).triu(1)
        hidden = self.layer(hidden, src_mask=causal_mask,
                            src_key_padding_mask=ids.eq(0))
        rows = torch.arange(batch, device=ids.device)
        question = self.query_proj(hidden[rows, query_index])
        keys = self.key_proj(hidden)
        scores = (keys * question[:, None, :]).sum(-1) / math.sqrt(self.width)
        eligible = (torch.arange(length, device=ids.device)[None, :] <
                    query_index[:, None]) & ids.ne(0)
        return scores.masked_fill(~eligible, -1e9)

    @torch.no_grad()
    def route(self, text: str) -> dict:
        self.eval()
        begin = time.perf_counter_ns()
        tokens = tokenize(text)
        if not tokens or tokens[-1] != "?":
            raise ValueError("query must end in ?")
        ids = torch.tensor([[IDS.get(t, IDS["<unk>"]) for t in tokens]],
                           dtype=torch.long)
        indices = torch.tensor([len(tokens) - 1], dtype=torch.long)
        with torch.inference_mode():
            logits = self.pointer_logits(ids, indices)
        answer = int(logits.argmax(-1).item())
        return {
            "token_index": answer,
            "token": tokens[answer],
            "index_probes": len(tokens) - 1,
            "elapsed_total_cpu_us": (time.perf_counter_ns() - begin) / 1000.0,
        }


def train_attention(seed: int, *, steps: int = 700,
                    batch_size: int = 48, width: int = 48,
                    learning_rate: float = 0.002) -> tuple[CausalAttentionPointer, dict]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = CausalAttentionPointer(width=width)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lossval = float("nan")
    for step in range(steps):
        data = [
            generate(seed * 10_000_000 + step * 1_000 + i,
                     n_facts=random.Random(seed + step * 1_000 + i).randint(5, 8),
                     split="train")
            for i in range(batch_size)
        ]
        ids, answer, query_index = examples_to_batch(data)
        model.train()
        optim.zero_grad(set_to_none=True)
        logits = model.pointer_logits(ids, query_index)
        loss = F.cross_entropy(logits, answer)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optim.step()
        lossval = float(loss.detach().item())
    return model, {
        "train_seed": seed,
        "updates": steps,
        "training_documents": steps * batch_size,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "final_training_objective": lossval,
        "loss": "attention pointer gold-index cross entropy",
    }
