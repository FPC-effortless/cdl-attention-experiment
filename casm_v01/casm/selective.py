from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CASM, CASMConfig


class SelectiveCASM(CASM):
    """CASM v0.2 candidate: compression-trained Q/K with selective episodic writes.

    The inference read path remains ordinary Q/K. Compression remains a training
    signal. Episodic memory uses separate erase and write controls and a soft
    gated shift register so write=1 exactly recovers append/shift behavior while
    write=0 preserves the existing memory bank.
    """

    def __init__(self, cfg: CASMConfig):
        super().__init__(cfg)
        d = cfg.memory_dim
        hidden = max(32, cfg.d_model // 2)
        self.write_gate_net = nn.Sequential(
            nn.Linear(cfg.d_model + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.erase_gate_net = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.write_loss_weight = 0.05
        self.strength_log_scale = 0.5
        self.write_force = 0.0
        nn.init.constant_(self.write_gate_net[-1].bias, -1.5)
        nn.init.constant_(self.erase_gate_net[-1].bias, -4.0)
        nn.init.normal_(self.write_gate_net[-1].weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.erase_gate_net[-1].weight, mean=0.0, std=1e-3)

    def _init_selective_memory(self, batch: int, device: torch.device, dtype: torch.dtype):
        ring, _, state = self._init_memory(batch, device, dtype)
        strength = torch.zeros(batch, self.cfg.memory_slots, device=device, dtype=dtype)
        return ring, strength, state

    def _candidates_selective(self, ring: torch.Tensor, strength: torch.Tensor, state: torch.Tensor):
        state_strength = torch.ones(state.shape[:2], device=state.device, dtype=state.dtype)
        candidates = torch.cat([state, ring], dim=1)
        strengths = torch.cat([state_strength, strength], dim=1)
        valid = strengths > 1e-4
        return candidates, strengths, valid

    def _scores_with_strength(self, query, memory, strengths, valid):
        scores = self.router.scores(query, memory)
        prior = self.strength_log_scale * strengths.clamp_min(1e-4).log()
        if scores.ndim == 3:
            prior = prior[:, None, :]
            valid_mask = valid[:, None, :].expand_as(scores)
        else:
            valid_mask = valid
        return (scores + prior).masked_fill(~valid_mask, -1e9)

    def _retrieve_selective(self, query, memory, strengths, valid):
        scores = self._scores_with_strength(query, memory, strengths, valid)
        weights = F.softmax(scores, dim=-1)
        if query.ndim == 3:
            mem = torch.einsum("btm,bmd->btd", weights, memory)
        else:
            mem = torch.einsum("bm,bmd->bd", weights, memory)
        return self.router.value(mem), scores

    def _observed_surprise(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Only within-chunk next bytes are used: no future boundary token leakage.
        if logits.shape[1] <= 1:
            return torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
        l = logits[:, :-1]
        y = targets[:, :-1]
        nll = F.cross_entropy(l.transpose(1, 2), y, ignore_index=256, reduction="none")
        mask = (y != 256).to(nll.dtype)
        return (nll * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)

    def _novelty(self, new_mem, ring, strength, state):
        candidates, _, valid = self._candidates_selective(ring, strength, state)
        q = F.normalize(new_mem, dim=-1)
        k = F.normalize(candidates, dim=-1)
        sim = torch.einsum("bd,bmd->bm", q, k).masked_fill(~valid, -1.0)
        return (1.0 - sim.max(dim=-1).values).clamp(0.0, 2.0)

    def _selective_write(self, ring, strength, state, summary, new_mem, observed_surprise):
        novelty = self._novelty(new_mem, ring, strength, state)
        surprise_feat = torch.log1p(observed_surprise.clamp_min(0.0))
        write_feat = torch.cat([summary, surprise_feat[:, None], novelty[:, None]], dim=-1)
        learned_write = torch.sigmoid(self.write_gate_net(write_feat)).squeeze(-1)
        write_prob = float(self.write_force) + (1.0 - float(self.write_force)) * learned_write

        ne = new_mem[:, None, :].expand_as(ring)
        erase_feat = torch.cat([ne, ring, ne * ring, (ne - ring).abs()], dim=-1)
        erase_prob = torch.sigmoid(self.erase_gate_net(erase_feat)).squeeze(-1)
        erase_prob = erase_prob * write_prob[:, None]
        ring_erased = ring * (1.0 - erase_prob[:, :, None])
        strength_erased = strength * (1.0 - erase_prob)

        shifted_ring = torch.cat([ring_erased[:, 1:], new_mem[:, None, :]], dim=1)
        shifted_strength = torch.cat(
            [strength_erased[:, 1:], torch.ones_like(strength_erased[:, :1])], dim=1
        )
        w = write_prob[:, None, None]
        ring_new = (1.0 - w) * ring_erased + w * shifted_ring
        ws = write_prob[:, None]
        strength_new = (1.0 - ws) * strength_erased + ws * shifted_strength
        return (
            ring_new,
            strength_new.clamp(0.0, 1.0),
            write_prob,
            learned_write,
            erase_prob.mean(dim=-1),
            novelty,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_aux: bool = True,
        external_teacher: Optional[torch.Tensor] = None,
        teacher_alpha: float = 0.0,
        target_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, t = tokens.shape
        if t < 2:
            raise ValueError("Need at least two tokens")
        x_in = tokens[:, :-1]
        targets = tokens[:, 1:]
        total_len = x_in.shape[1]
        ring, ring_strength, state = self._init_selective_memory(b, tokens.device, self.embed.weight.dtype)

        logits_chunks: List[torch.Tensor] = []
        target_chunks: List[torch.Tensor] = []
        compression_losses: List[torch.Tensor] = []
        compression_predictor_losses: List[torch.Tensor] = []
        mtp_losses: List[torch.Tensor] = []
        verifier_losses: List[torch.Tensor] = []
        write_losses: List[torch.Tensor] = []
        router_entropies: List[torch.Tensor] = []
        gain_means: List[torch.Tensor] = []
        gain_stds: List[torch.Tensor] = []
        memory_gate_means: List[torch.Tensor] = []
        write_gate_means: List[torch.Tensor] = []
        erase_gate_means: List[torch.Tensor] = []
        novelty_means: List[torch.Tensor] = []
        strength_means: List[torch.Tensor] = []

        for chunk_idx, start in enumerate(range(0, total_len, self.cfg.chunk_size)):
            end = min(start + self.cfg.chunk_size, total_len)
            ids = x_in[:, start:end]
            y = targets[:, start:end]
            h = self.embed(ids)
            for block in self.blocks:
                h = block(h)
            h = self.norm(h)

            past_candidates, past_strengths, past_valid = self._candidates_selective(ring, ring_strength, state)
            if self.cfg.use_memory:
                retrieved, _ = self._retrieve_selective(h, past_candidates, past_strengths, past_valid)
                gate = torch.sigmoid(self.memory_gate(torch.cat([h, retrieved], dim=-1)))
                h = h + gate * retrieved
                memory_gate_means.append(gate.mean())

            h = h + self.memory_ffn(self.memory_ffn_norm(h))
            logits = self.lm_head(h)
            logits_chunks.append(logits)
            target_chunks.append(y)

            if return_aux and self.cfg.mtp_horizons > 1:
                for horizon, head in enumerate(self.mtp_heads, start=2):
                    valid_len = h.shape[1] - (horizon - 1)
                    if valid_len > 0:
                        pred = head(h[:, :valid_len])
                        tgt_start = start + horizon
                        tgt_end = min(start + h.shape[1] + 1, tokens.shape[1])
                        tgt = tokens[:, tgt_start:tgt_end]
                        if tgt.shape[1] == valid_len and (tgt != 256).any():
                            mtp_losses.append(F.cross_entropy(pred.transpose(1, 2), tgt, ignore_index=256))

            valid_tok = ids != 256
            last_idx = valid_tok.long().sum(dim=1).clamp_min(1) - 1
            summary = h[torch.arange(b, device=h.device), last_idx]
            new_mem = torch.tanh(self.memory_in(summary))
            observed_surprise = self._observed_surprise(logits, y)

            state = self.state.update(state, summary)
            ring, ring_strength, write_prob, learned_write, erase_prob, novelty = self._selective_write(
                ring, ring_strength, state, summary, new_mem, observed_surprise
            )
            write_gate_means.append(write_prob.mean())
            erase_gate_means.append(erase_prob.mean())
            novelty_means.append(novelty.mean())
            strength_means.append(ring_strength.mean())

            candidates, cand_strengths, cand_valid = self._candidates_selective(ring, ring_strength, state)
            if self.cfg.use_memory:
                scores = self._scores_with_strength(summary, candidates, cand_strengths, cand_valid)
                weights = F.softmax(scores, dim=-1)
                router_entropies.append((-(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)).mean())
            else:
                scores = torch.zeros(b, candidates.shape[1], device=tokens.device)

            future_start = end
            future_end = min(future_start + self.cfg.compression_future_tokens, tokens.shape[1])
            need_aux = (
                self.cfg.compression_loss_weight > 0.0
                or self.cfg.compression_predictor_loss_weight > 0.0
                or (external_teacher is not None and teacher_alpha > 0.0)
            )
            if return_aux and self.cfg.use_memory and need_aux and future_end > future_start:
                future = tokens[:, future_start:future_end]
                target_dist, gain, predictor_loss = self.router.compression_target(summary, candidates, future, cand_valid)
                finite_gain = gain.masked_fill(~cand_valid, 0.0)
                valid_count = cand_valid.sum(dim=-1).clamp_min(1).to(gain.dtype)
                gain_mean_per = finite_gain.sum(dim=-1) / valid_count
                centered = (gain - gain_mean_per[:, None]).masked_fill(~cand_valid, 0.0)
                gain_std = torch.sqrt((centered.pow(2).sum(dim=-1) / valid_count).clamp_min(1e-12))
                gain_stds.append(gain_std.mean())

                if external_teacher is not None and teacher_alpha > 0.0 and chunk_idx < external_teacher.shape[1]:
                    ext = external_teacher[:, chunk_idx, :].to(target_dist.device, target_dist.dtype)
                    confidence = (gain_std.detach() / 0.08).clamp(0.0, 1.0)
                    ext_alpha = torch.maximum(torch.full_like(confidence, float(teacher_alpha)), 1.0 - confidence)
                    target_dist = ext_alpha[:, None] * ext + (1.0 - ext_alpha[:, None]) * target_dist
                    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                pred_log = F.log_softmax(scores, dim=-1)
                compression_losses.append(-(target_dist * pred_log).sum(dim=-1).mean())
                compression_predictor_losses.append(predictor_loss)
                gain_means.append(finite_gain.sum() / cand_valid.sum().clamp_min(1))

                # Future-verified target: did this chunk's compressed state reduce
                # code length of actual future bytes? The deployed gate predicts
                # this from causal surprise/novelty/current-state features only.
                single_valid = torch.ones(b, 1, device=tokens.device, dtype=torch.bool)
                _, new_gain, _ = self.router.compression_target(summary, new_mem[:, None, :], future, single_valid)
                write_target = torch.sigmoid((new_gain[:, 0].detach() - 0.05) / 0.05)
                write_losses.append(F.binary_cross_entropy(learned_write, write_target))

            if return_aux and end < total_len:
                nxt_end = min(end + self.cfg.chunk_size, total_len)
                next_ids = x_in[:, end:nxt_end]
                if next_ids.shape[1] > 0:
                    next_emb = self.embed(next_ids).mean(dim=1)
                    pos = self.verify(torch.cat([summary, next_emb], dim=-1)).squeeze(-1)
                    if b > 1:
                        neg_emb = torch.roll(next_emb, shifts=1, dims=0)
                        neg = self.verify(torch.cat([summary, neg_emb], dim=-1)).squeeze(-1)
                        verifier_losses.append(
                            0.5 * (
                                F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                                + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
                            )
                        )

        logits_all = torch.cat(logits_chunks, dim=1)
        targets_all = torch.cat(target_chunks, dim=1)
        token_nll = F.cross_entropy(logits_all.transpose(1, 2), targets_all, ignore_index=256, reduction="none")
        valid_targets = targets_all != 256
        if target_weights is None:
            weights = valid_targets.to(token_nll.dtype)
        else:
            if target_weights.shape != targets_all.shape:
                raise ValueError(f"target_weights {target_weights.shape} must match targets {targets_all.shape}")
            weights = target_weights.to(token_nll.dtype) * valid_targets.to(token_nll.dtype)
        lm_loss = (token_nll * weights).sum() / weights.sum().clamp_min(1.0)

        zero = lm_loss.new_zeros(())
        compression_loss = torch.stack(compression_losses).mean() if compression_losses else zero
        compression_predictor_loss = torch.stack(compression_predictor_losses).mean() if compression_predictor_losses else zero
        mtp_loss = torch.stack(mtp_losses).mean() if mtp_losses else zero
        verifier_loss = torch.stack(verifier_losses).mean() if verifier_losses else zero
        write_loss = torch.stack(write_losses).mean() if write_losses else zero
        total = (
            lm_loss
            + self.cfg.compression_loss_weight * compression_loss
            + self.cfg.compression_predictor_loss_weight * compression_predictor_loss
            + self.cfg.mtp_loss_weight * mtp_loss
            + self.cfg.verifier_loss_weight * verifier_loss
            + self.write_loss_weight * write_loss
        )
        return {
            "loss": total,
            "lm_loss": lm_loss,
            "compression_loss": compression_loss,
            "compression_predictor_loss": compression_predictor_loss,
            "mtp_loss": mtp_loss,
            "verifier_loss": verifier_loss,
            "write_loss": write_loss,
            "router_entropy": torch.stack(router_entropies).mean() if router_entropies else zero,
            "mean_compression_gain": torch.stack(gain_means).mean() if gain_means else zero,
            "compression_gain_std": torch.stack(gain_stds).mean() if gain_stds else zero,
            "memory_gate_mean": torch.stack(memory_gate_means).mean() if memory_gate_means else zero,
            "write_gate_mean": torch.stack(write_gate_means).mean() if write_gate_means else zero,
            "erase_gate_mean": torch.stack(erase_gate_means).mean() if erase_gate_means else zero,
            "memory_novelty_mean": torch.stack(novelty_means).mean() if novelty_means else zero,
            "ring_strength_mean": torch.stack(strength_means).mean() if strength_means else zero,
            "logits": logits_all,
        }
