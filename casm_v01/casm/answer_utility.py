from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .data import PAD
from .model import CASM


def answer_utility_qk_loss(
    model: CASM,
    tokens: torch.Tensor,
    answer_target_mask: torch.Tensor,
    temperature: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Train live Q/K scores toward causal answer utility.

    For each teacher-forced answer byte, every memory candidate that was
    available *before the current chunk* is evaluated counterfactually. Its
    utility is the reduction in the model's actual answer-token NLL relative
    to the same local hidden state with no retrieved memory.

    The counterfactual evaluator and recurrent replay are detached. The
    auxiliary loss therefore updates the router's Q/K geometry, while the
    ordinary LM objective remains responsible for the backbone and memory
    representations.
    """
    if not model.cfg.use_memory:
        z = model.embed.weight.new_zeros(())
        return {"loss": z, "mean_gain": z, "gain_std": z, "positions": z}

    x_in = tokens[:, :-1]
    targets = tokens[:, 1:]
    if answer_target_mask.shape != targets.shape:
        raise ValueError(
            f"answer_target_mask {answer_target_mask.shape} must match targets {targets.shape}"
        )

    b, total_len = x_in.shape
    ring, ring_valid, state = model._init_memory(
        b, tokens.device, model.embed.weight.dtype
    )

    loss_sum = model.embed.weight.new_zeros(())
    pos_count = 0
    gain_values = []
    gain_std_values = []

    for start in range(0, total_len, model.cfg.chunk_size):
        end = min(start + model.cfg.chunk_size, total_len)
        ids = x_in[:, start:end]
        y = targets[:, start:end]

        # Replay the exact inference state without building a second backbone
        # gradient graph. Q/K gets a separate gradient below from live scores.
        with torch.no_grad():
            h_local = model.embed(ids)
            for block in model.blocks:
                h_local = block(h_local)
            h_local = model.norm(h_local)
            past_candidates, past_valid = model._candidates(ring, ring_valid, state)

        # Do not supervise the first chunk. Persistent-state zero placeholders
        # exist there, but no episodic information has actually been observed.
        eligible = ring_valid.any(dim=-1)
        answer_here = answer_target_mask[:, start:end] & eligible[:, None] & (y != PAD)
        pos = answer_here.nonzero(as_tuple=False)

        if pos.numel() > 0:
            # Live scores: gradients flow only through Q/K because replayed
            # hidden states and memories are detached.
            live_scores = model.router.scores(
                h_local.detach(), past_candidates.detach()
            )
            bi, ti = pos[:, 0], pos[:, 1]
            pred_scores = live_scores[bi, ti]
            valid = past_valid[bi]

            with torch.no_grad():
                qh = h_local[bi, ti]
                cand = past_candidates[bi]
                cand_value = model.router.value(cand)
                qexp = qh[:, None, :].expand(-1, cand_value.shape[1], -1)
                gate = torch.sigmoid(
                    model.memory_gate(torch.cat([qexp, cand_value], dim=-1))
                )
                h_cf = qexp + gate * cand_value
                h_cf = h_cf + model.memory_ffn(model.memory_ffn_norm(h_cf))
                logits_cf = model.lm_head(h_cf)

                tgt = y[bi, ti]
                tgt_cf = tgt[:, None].expand(-1, cand_value.shape[1])
                cf_nll = F.cross_entropy(
                    logits_cf.reshape(-1, model.cfg.vocab_size),
                    tgt_cf.reshape(-1),
                    reduction="none",
                ).view_as(tgt_cf)

                h_base = qh + model.memory_ffn(model.memory_ffn_norm(qh))
                base_nll = F.cross_entropy(
                    model.lm_head(h_base), tgt, reduction="none"
                )
                gain = (base_nll[:, None] - cf_nll).masked_fill(~valid, -1e9)
                target_dist = F.softmax(gain / max(temperature, 1e-6), dim=-1)

                finite_gain = gain.masked_fill(~valid, 0.0)
                count = valid.sum(dim=-1).clamp_min(1).to(finite_gain.dtype)
                mean_per = finite_gain.sum(dim=-1) / count
                centered = (gain - mean_per[:, None]).masked_fill(~valid, 0.0)
                gain_std = torch.sqrt(
                    (centered.pow(2).sum(dim=-1) / count).clamp_min(1e-12)
                )
                gain_values.append(mean_per.mean())
                gain_std_values.append(gain_std.mean())

            pred_log = F.log_softmax(pred_scores.masked_fill(~valid, -1e9), dim=-1)
            per_pos = -(target_dist * pred_log).sum(dim=-1)
            loss_sum = loss_sum + per_pos.sum()
            pos_count += int(per_pos.numel())

        # Advance recurrent memory exactly as the model does at inference.
        with torch.no_grad():
            past_candidates, past_valid = model._candidates(ring, ring_valid, state)
            retrieved, _ = model.router.retrieve(h_local, past_candidates, past_valid)
            gate = torch.sigmoid(
                model.memory_gate(torch.cat([h_local, retrieved], dim=-1))
            )
            h = h_local + gate * retrieved
            h = h + model.memory_ffn(model.memory_ffn_norm(h))
            valid_tok = ids != PAD
            last_idx = valid_tok.long().sum(dim=1).clamp_min(1) - 1
            summary = h[torch.arange(b, device=h.device), last_idx]
            state = model.state.update(state, summary)
            new_mem = torch.tanh(model.memory_in(summary))
            ring = torch.cat([ring[:, 1:], new_mem[:, None, :]], dim=1)
            ring_valid = torch.cat(
                [
                    ring_valid[:, 1:],
                    torch.ones(b, 1, device=tokens.device, dtype=torch.bool),
                ],
                dim=1,
            )

    zero = model.embed.weight.new_zeros(())
    if pos_count == 0:
        return {"loss": zero, "mean_gain": zero, "gain_std": zero, "positions": zero}
    return {
        "loss": loss_sum / pos_count,
        "mean_gain": torch.stack(gain_values).mean() if gain_values else zero,
        "gain_std": torch.stack(gain_std_values).mean() if gain_std_values else zero,
        "positions": zero.new_tensor(float(pos_count)),
    }
