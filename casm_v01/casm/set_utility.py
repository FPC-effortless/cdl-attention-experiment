from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .data import PAD
from .recurrent_model import CASMRecurrent


def set_conditioned_utility_loss(
    model: CASMRecurrent,
    tokens: torch.Tensor,
    answer_target_mask: torch.Tensor,
    *,
    temperature: float = 0.7,
    min_gain_std: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Supervise recurrent Q/K with leave-one-memory-out answer utility.

    For every eligible teacher-forced answer byte:

        U_i = NLL(answer | all valid memories except i)
              - NLL(answer | all valid memories)

    The full recurrent reasoning loop is used for both the factual and
    counterfactual evaluations. Thus a candidate is valuable when removing it
    from the *set* harms the final answer after all recurrent retrieve/update
    steps, rather than when isolated injection happens to lower loss.

    Counterfactual computation is detached. Q/K is trained at each recurrent
    step from detached working-state snapshots, so the auxiliary objective does
    not directly update values, memory writes, gates, backbone, or LM head.

    Near-uniform/noisy targets are skipped. Informative utility vectors are
    z-normalized before softmax so supervision depends on relative causal
    contribution rather than raw scale.
    """
    zero = model.embed.weight.new_zeros(())
    if not model.cfg.use_memory:
        return {
            "loss": zero,
            "mean_gain": zero,
            "gain_std": zero,
            "positions": zero,
            "informative_positions": zero,
            "informative_fraction": zero,
        }

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

    loss_sum = zero.clone()
    loss_terms = 0
    total_positions = 0
    informative_positions = 0
    gain_means = []
    gain_stds = []

    for start in range(0, total_len, model.cfg.chunk_size):
        end = min(start + model.cfg.chunk_size, total_len)
        ids = x_in[:, start:end]
        y = targets[:, start:end]

        with torch.no_grad():
            h_local = model.embed(ids)
            for block in model.blocks:
                h_local = block(h_local)
            h_local = model.norm(h_local)
            candidates, cand_valid = model._candidates(ring, ring_valid, state)

        # Persistent zero state exists before any episodic observation, so do
        # not supervise until at least one real memory has been written.
        eligible = ring_valid.any(dim=-1)
        answer_here = (
            answer_target_mask[:, start:end]
            & eligible[:, None]
            & (y != PAD)
        )
        pos = answer_here.nonzero(as_tuple=False)

        if pos.numel() > 0:
            bi, ti = pos[:, 0], pos[:, 1]
            qh = h_local[bi, ti]
            cand = candidates[bi]
            valid = cand_valid[bi]
            tgt = y[bi, ti]
            total_positions += int(pos.shape[0])

            with torch.no_grad():
                full_h, step_states = model.reason_detached(
                    qh, cand, valid, capture_states=True
                )
                full_nll = F.cross_entropy(
                    model.lm_head(full_h), tgt, reduction="none"
                )

                n_pos, n_cand = valid.shape
                gains = qh.new_full((n_pos, n_cand), -1e9)
                for j in range(n_cand):
                    valid_j = valid.clone()
                    valid_j[:, j] = False
                    # State slots make the candidate set non-empty, but keep a
                    # generic guard for future configurations.
                    still_valid = valid_j.any(dim=-1)
                    if not still_valid.any():
                        continue
                    ablated_h, _ = model.reason_detached(
                        qh, cand, valid_j, capture_states=False
                    )
                    ablated_nll = F.cross_entropy(
                        model.lm_head(ablated_h), tgt, reduction="none"
                    )
                    gains[:, j] = torch.where(
                        valid[:, j], ablated_nll - full_nll, gains[:, j]
                    )

                finite = gains.masked_fill(~valid, 0.0)
                count = valid.sum(dim=-1).clamp_min(1).to(finite.dtype)
                mean = finite.sum(dim=-1) / count
                centered = (gains - mean[:, None]).masked_fill(~valid, 0.0)
                std = torch.sqrt(
                    (centered.pow(2).sum(dim=-1) / count).clamp_min(1e-12)
                )
                info = std >= float(min_gain_std)
                informative_positions += int(info.sum())
                gain_means.append(mean.mean())
                gain_stds.append(std.mean())

                z = centered / std[:, None].clamp_min(1e-6)
                z = z.clamp(min=-8.0, max=8.0)
                z = z.masked_fill(~valid, -1e9)
                target_dist = F.softmax(
                    z / max(float(temperature), 1e-6), dim=-1
                )

            if info.any():
                # Each recurrent step gets its own live Q/K score gradient at
                # the detached state reached by the factual full-memory path.
                for step_state in step_states:
                    live_scores = model.router.scores(
                        step_state.detach(), cand.detach()
                    )
                    pred_log = F.log_softmax(
                        live_scores.masked_fill(~valid, -1e9), dim=-1
                    )
                    per_pos = -(target_dist * pred_log).sum(dim=-1)
                    loss_sum = loss_sum + per_pos[info].sum()
                    loss_terms += int(info.sum())

        # Advance the recurrent state exactly as deployed inference does.
        with torch.no_grad():
            candidates, cand_valid = model._candidates(ring, ring_valid, state)
            h, _ = model.reason_detached(
                h_local, candidates, cand_valid, capture_states=False
            )
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

    if loss_terms == 0:
        return {
            "loss": zero,
            "mean_gain": torch.stack(gain_means).mean() if gain_means else zero,
            "gain_std": torch.stack(gain_stds).mean() if gain_stds else zero,
            "positions": zero.new_tensor(float(total_positions)),
            "informative_positions": zero,
            "informative_fraction": zero,
        }

    return {
        "loss": loss_sum / loss_terms,
        "mean_gain": torch.stack(gain_means).mean() if gain_means else zero,
        "gain_std": torch.stack(gain_stds).mean() if gain_stds else zero,
        "positions": zero.new_tensor(float(total_positions)),
        "informative_positions": zero.new_tensor(float(informative_positions)),
        "informative_fraction": zero.new_tensor(
            informative_positions / max(1, total_positions)
        ),
    }
