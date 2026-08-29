from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .data import PAD
from .recurrent_model import CASMRecurrent


def deep_answer_loss(
    model: CASMRecurrent,
    tokens: torch.Tensor,
    answer_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Supervise intermediate recurrent states with the model's own LM head.

    The ordinary model loss already supervises the final recurrent state. This
    auxiliary objective applies next-byte cross entropy only to recurrent steps
    before the final step, and only where the *target* byte belongs to the gold
    answer span. No additional decoder parameters are introduced.

    For an answer token at absolute position ``j``, logits from hidden position
    ``j-1`` predict that byte. Therefore the target mask is ``answer_mask[:,1:]``.
    The first answer byte is predicted from the final byte of the literal
    ``answer `` marker, before any gold answer byte is visible causally.
    """
    if answer_mask.shape != tokens.shape:
        raise ValueError("answer_mask must match tokens")
    if model.reasoning_steps < 2:
        raise ValueError("deep supervision requires at least two reasoning steps")

    b, t = tokens.shape
    if t < 2:
        raise ValueError("Need at least two tokens")

    x_in = tokens[:, :-1]
    targets = tokens[:, 1:]
    total_len = x_in.shape[1]
    ring, ring_valid, state = model._init_memory(
        b, tokens.device, model.embed.weight.dtype
    )

    step_loss_sums: List[torch.Tensor] = [
        model.embed.weight.new_zeros(()) for _ in range(model.reasoning_steps - 1)
    ]
    step_correct: List[torch.Tensor] = [
        model.embed.weight.new_zeros(()) for _ in range(model.reasoning_steps - 1)
    ]
    step_count: List[torch.Tensor] = [
        model.embed.weight.new_zeros(()) for _ in range(model.reasoning_steps - 1)
    ]

    for start in range(0, total_len, model.cfg.chunk_size):
        end = min(start + model.cfg.chunk_size, total_len)
        ids = x_in[:, start:end]
        y = targets[:, start:end]
        target_mask = answer_mask[:, start + 1 : end + 1] & (y != PAD)

        h = model.embed(ids)
        for block in model.blocks:
            h = block(h)
        h = model.norm(h)

        candidates, cand_valid = model._candidates(ring, ring_valid, state)
        if model.cfg.use_memory:
            final_h, pre_states, _, _ = model.reason(
                h, candidates, cand_valid, capture_states=True
            )
            # pre_states contains z before each reasoning update. Convert to
            # post-update states: z1, z2, ..., zR.
            post_states = list(pre_states[1:]) + [final_h]
        else:
            final_h = h + model.memory_ffn(model.memory_ffn_norm(h))
            post_states = [final_h for _ in range(model.reasoning_steps)]

        if target_mask.any():
            count = target_mask.sum().to(dtype=final_h.dtype)
            for ri, z in enumerate(post_states[:-1]):
                logits = model.lm_head(z)
                nll = F.cross_entropy(
                    logits.transpose(1, 2), y, ignore_index=PAD, reduction="none"
                )
                step_loss_sums[ri] = step_loss_sums[ri] + nll[target_mask].sum()
                pred = logits.argmax(dim=-1)
                step_correct[ri] = step_correct[ri] + (
                    pred[target_mask] == y[target_mask]
                ).to(final_h.dtype).sum()
                step_count[ri] = step_count[ri] + count

        valid_tok = ids != PAD
        last_idx = valid_tok.long().sum(dim=1).clamp_min(1) - 1
        summary = final_h[torch.arange(b, device=tokens.device), last_idx]
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

    valid_steps = [i for i, c in enumerate(step_count) if float(c.detach()) > 0]
    if not valid_steps:
        zero = model.embed.weight.new_zeros(())
        return zero, {
            "deep_answer_acc": zero.detach(),
            "deep_answer_nll": zero.detach(),
            "supervised_steps": zero.detach(),
        }

    losses = [step_loss_sums[i] / step_count[i].clamp_min(1.0) for i in valid_steps]
    # Later recurrent states get modestly more weight, while the actual final
    # state remains governed by the ordinary LM objective.
    weights = torch.arange(1, len(losses) + 1, device=tokens.device, dtype=losses[0].dtype)
    weights = weights / weights.sum()
    loss = torch.stack(losses).mul(weights).sum()

    accs = [step_correct[i] / step_count[i].clamp_min(1.0) for i in valid_steps]
    metrics = {
        "deep_answer_acc": torch.stack(accs).mean().detach(),
        "deep_answer_nll": torch.stack(losses).mean().detach(),
        "supervised_steps": loss.new_tensor(float(len(valid_steps))).detach(),
    }
    for out_i, src_i in enumerate(valid_steps, start=1):
        metrics[f"step{src_i + 1}_answer_nll"] = losses[out_i - 1].detach()
        metrics[f"step{src_i + 1}_answer_acc"] = accs[out_i - 1].detach()
    return loss, metrics
