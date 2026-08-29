from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .data import PAD
from .model import CASM, CASMConfig


class CASMRecurrent(CASM):
    """CASM with parameter-shared recurrent memory reasoning.

    `reasoning_steps=1` is designed to match the ordinary Q/K memory path.
    Larger values repeatedly reuse the same router, value projection, memory
    gate, and memory FFN. Parameter count is therefore unchanged while
    effective latent computation depth increases.
    """

    def __init__(self, cfg: CASMConfig, reasoning_steps: int = 1):
        super().__init__(cfg)
        if reasoning_steps < 1:
            raise ValueError("reasoning_steps must be >= 1")
        self.reasoning_steps = int(reasoning_steps)

    def reason(
        self,
        h: torch.Tensor,
        candidates: torch.Tensor,
        valid: torch.Tensor,
        *,
        capture_states: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Run shared retrieve/update iterations over a fixed memory set."""
        states: List[torch.Tensor] = []
        scores_all: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        z = h
        for _ in range(self.reasoning_steps):
            if capture_states:
                states.append(z)
            retrieved, scores = self.router.retrieve(z, candidates, valid)
            gate = torch.sigmoid(self.memory_gate(torch.cat([z, retrieved], dim=-1)))
            z = z + gate * retrieved
            z = z + self.memory_ffn(self.memory_ffn_norm(z))
            scores_all.append(scores)
            gates.append(gate)
        return z, states, scores_all, gates

    @torch.no_grad()
    def reason_detached(
        self,
        h: torch.Tensor,
        candidates: torch.Tensor,
        valid: torch.Tensor,
        *,
        capture_states: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        z, states, _, _ = self.reason(
            h, candidates, valid, capture_states=capture_states
        )
        return z, states

    def forward(
        self,
        tokens: torch.Tensor,
        return_aux: bool = True,
        external_teacher: Optional[torch.Tensor] = None,
        teacher_alpha: float = 0.0,
        target_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # This experiment intentionally disables the old compression target.
        # Keep the signature compatible with CASM, but reject accidental use.
        if external_teacher is not None or teacher_alpha:
            raise ValueError("CASMRecurrent does not use the old compression teacher")

        b, t = tokens.shape
        if t < 2:
            raise ValueError("Need at least two tokens")

        x_in = tokens[:, :-1]
        targets = tokens[:, 1:]
        total_len = x_in.shape[1]
        ring, ring_valid, state = self._init_memory(
            b, tokens.device, self.embed.weight.dtype
        )

        logits_chunks: List[torch.Tensor] = []
        target_chunks: List[torch.Tensor] = []
        mtp_losses: List[torch.Tensor] = []
        verifier_losses: List[torch.Tensor] = []
        router_entropies: List[torch.Tensor] = []
        memory_gate_means: List[torch.Tensor] = []

        for start in range(0, total_len, self.cfg.chunk_size):
            end = min(start + self.cfg.chunk_size, total_len)
            ids = x_in[:, start:end]
            y = targets[:, start:end]

            h = self.embed(ids)
            for block in self.blocks:
                h = block(h)
            h = self.norm(h)

            candidates, cand_valid = self._candidates(ring, ring_valid, state)
            if self.cfg.use_memory:
                h, _, score_steps, gate_steps = self.reason(h, candidates, cand_valid)
                memory_gate_means.extend(g.mean() for g in gate_steps)
                if return_aux:
                    for scores in score_steps:
                        if scores.ndim == 2:
                            weights_r = F.softmax(
                                scores.masked_fill(~cand_valid, -1e9), dim=-1
                            )
                        else:
                            mask = cand_valid[:, None, :].expand_as(scores)
                            weights_r = F.softmax(scores.masked_fill(~mask, -1e9), dim=-1)
                        ent = -(weights_r * weights_r.clamp_min(1e-8).log()).sum(dim=-1)
                        router_entropies.append(ent.mean())
            else:
                h = h + self.memory_ffn(self.memory_ffn_norm(h))

            logits = self.lm_head(h)
            logits_chunks.append(logits)
            target_chunks.append(y)

            if return_aux and self.cfg.mtp_horizons > 1:
                for horizon, head in enumerate(self.mtp_heads, start=2):
                    valid_len = h.shape[1] - (horizon - 1)
                    if valid_len <= 0:
                        continue
                    pred = head(h[:, :valid_len])
                    tgt_start = start + horizon
                    tgt_end = min(start + h.shape[1] + 1, tokens.shape[1])
                    tgt = tokens[:, tgt_start:tgt_end]
                    if tgt.shape[1] == valid_len and (tgt != PAD).any():
                        mtp_losses.append(
                            F.cross_entropy(
                                pred.transpose(1, 2), tgt, ignore_index=PAD
                            )
                        )

            valid_tok = ids != PAD
            last_idx = valid_tok.long().sum(dim=1).clamp_min(1) - 1
            summary = h[torch.arange(b, device=h.device), last_idx]
            state = self.state.update(state, summary)
            new_mem = torch.tanh(self.memory_in(summary))
            ring = torch.cat([ring[:, 1:], new_mem[:, None, :]], dim=1)
            ring_valid = torch.cat(
                [
                    ring_valid[:, 1:],
                    torch.ones(b, 1, device=tokens.device, dtype=torch.bool),
                ],
                dim=1,
            )

            if return_aux and end < total_len:
                nxt_end = min(end + self.cfg.chunk_size, total_len)
                next_ids = x_in[:, end:nxt_end]
                if next_ids.shape[1] > 0:
                    next_emb = self.embed(next_ids).mean(dim=1)
                    pos = self.verify(torch.cat([summary, next_emb], dim=-1)).squeeze(-1)
                    if b > 1:
                        neg_emb = torch.roll(next_emb, shifts=1, dims=0)
                        neg = self.verify(
                            torch.cat([summary, neg_emb], dim=-1)
                        ).squeeze(-1)
                        verifier_losses.append(
                            0.5
                            * (
                                F.binary_cross_entropy_with_logits(
                                    pos, torch.ones_like(pos)
                                )
                                + F.binary_cross_entropy_with_logits(
                                    neg, torch.zeros_like(neg)
                                )
                            )
                        )

        logits_all = torch.cat(logits_chunks, dim=1)
        targets_all = torch.cat(target_chunks, dim=1)
        token_nll = F.cross_entropy(
            logits_all.transpose(1, 2),
            targets_all,
            ignore_index=PAD,
            reduction="none",
        )
        valid_targets = targets_all != PAD
        if target_weights is not None and target_weights.shape != targets_all.shape:
            raise ValueError(
                f"target_weights {target_weights.shape} must match targets {targets_all.shape}"
            )
        weights = (
            valid_targets.to(token_nll.dtype)
            if target_weights is None
            else target_weights.to(token_nll.dtype) * valid_targets.to(token_nll.dtype)
        )
        lm_loss = (token_nll * weights).sum() / weights.sum().clamp_min(1.0)
        zero = lm_loss.new_zeros(())
        mtp_loss = torch.stack(mtp_losses).mean() if mtp_losses else zero
        verifier_loss = (
            torch.stack(verifier_losses).mean() if verifier_losses else zero
        )
        total = (
            lm_loss
            + self.cfg.mtp_loss_weight * mtp_loss
            + self.cfg.verifier_loss_weight * verifier_loss
        )
        return {
            "loss": total,
            "lm_loss": lm_loss,
            "compression_loss": zero,
            "compression_predictor_loss": zero,
            "mtp_loss": mtp_loss,
            "verifier_loss": verifier_loss,
            "router_entropy": (
                torch.stack(router_entropies).mean() if router_entropies else zero
            ),
            "mean_compression_gain": zero,
            "compression_gain_std": zero,
            "memory_gate_mean": (
                torch.stack(memory_gate_means).mean() if memory_gate_means else zero
            ),
            "logits": logits_all,
        }
