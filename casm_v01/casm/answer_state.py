from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import EOS, PAD
from .model import CASMConfig
from .recurrent_model import CASMRecurrent


@dataclass
class AnswerStateOutput:
    logits_steps: List[torch.Tensor]
    z_steps: List[torch.Tensor]


class CASMAnswerState(nn.Module):
    """CASM with a persistent non-autoregressive answer state.

    The prompt is encoded only through the final byte of the literal ``answer ``
    marker. Gold answer bytes are never consumed by the answer-state recurrence.
    A shared update block jointly refines latent reasoning state ``z`` and a fixed
    bank of answer slots ``y``. Answer slots are decoded in parallel through the
    core model's tied language-model head.

    ``reasoning_steps`` changes effective depth only. Parameters are shared across
    steps, so one-step and multi-step variants have identical parameter counts.
    """

    def __init__(
        self,
        cfg: CASMConfig,
        *,
        reasoning_steps: int = 3,
        answer_slots: int = 20,
    ) -> None:
        super().__init__()
        if reasoning_steps < 1:
            raise ValueError("reasoning_steps must be >= 1")
        if answer_slots < 2:
            raise ValueError("answer_slots must be >= 2")
        self.cfg = cfg
        self.reasoning_steps = int(reasoning_steps)
        self.answer_slots = int(answer_slots)

        self.core = CASMRecurrent(cfg, reasoning_steps=1)
        d = cfg.d_model
        self.answer_pos = nn.Parameter(torch.empty(answer_slots, d))
        self.answer_seed = nn.Linear(d, d, bias=False)
        self.answer_feedback = nn.Linear(d, d, bias=False)
        self.answer_context = nn.Linear(d, d, bias=False)
        self.answer_gate = nn.Linear(2 * d, d)
        self.answer_update = nn.Sequential(
            nn.Linear(3 * d, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, d),
        )
        self.answer_mixer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.answer_norm = nn.LayerNorm(d)
        nn.init.normal_(self.answer_pos, mean=0.0, std=0.02)
        for module in [
            self.answer_seed,
            self.answer_feedback,
            self.answer_context,
            self.answer_gate,
        ]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for module in self.answer_update.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _encode_to_answer_anchor(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompt prefixes and capture state/candidates at ``answer ``.

        Tokens strictly after each row's anchor are replaced by PAD before any
        transformer computation. This makes answer-state predictions invariant
        to the supplied gold answer suffix and prevents teacher-forcing leakage.
        """
        b, t = tokens.shape
        if anchors.shape != (b,):
            raise ValueError("anchors must have shape [batch]")
        if t < 2:
            raise ValueError("Need at least two tokens")

        x_in = tokens[:, :-1]
        total_len = x_in.shape[1]
        ring, ring_valid, state = self.core._init_memory(
            b, tokens.device, self.core.embed.weight.dtype
        )
        captured_z: List[torch.Tensor | None] = [None] * b
        captured_candidates: List[torch.Tensor | None] = [None] * b
        captured_valid: List[torch.Tensor | None] = [None] * b

        for start in range(0, total_len, self.cfg.chunk_size):
            end = min(start + self.cfg.chunk_size, total_len)
            ids = x_in[:, start:end].clone()
            abs_pos = torch.arange(start, end, device=tokens.device)[None, :]
            ids = ids.masked_fill(abs_pos > anchors[:, None], PAD)

            h = self.core.embed(ids)
            for block in self.core.blocks:
                h = block(h)
            h = self.core.norm(h)

            candidates, cand_valid = self.core._candidates(ring, ring_valid, state)
            for bi in range(b):
                a = int(anchors[bi])
                if captured_z[bi] is None and start <= a < end:
                    local = a - start
                    captured_z[bi] = h[bi, local]
                    captured_candidates[bi] = candidates[bi]
                    captured_valid[bi] = cand_valid[bi]

            if all(x is not None for x in captured_z):
                break

            if self.cfg.use_memory:
                h_mem, _, _, _ = self.core.reason(h, candidates, cand_valid)
            else:
                h_mem = h + self.core.memory_ffn(self.core.memory_ffn_norm(h))
            valid_tok = ids != PAD
            last_idx = valid_tok.long().sum(dim=1).clamp_min(1) - 1
            summary = h_mem[torch.arange(b, device=tokens.device), last_idx]
            state = self.core.state.update(state, summary)
            new_mem = torch.tanh(self.core.memory_in(summary))
            ring = torch.cat([ring[:, 1:], new_mem[:, None, :]], dim=1)
            ring_valid = torch.cat(
                [ring_valid[:, 1:], valid_tok.any(dim=1, keepdim=True)],
                dim=1,
            )

        if any(x is None for x in captured_z):
            raise RuntimeError("failed to capture every answer anchor")
        z = torch.stack([x for x in captured_z if x is not None], dim=0)
        candidates = torch.stack(
            [x for x in captured_candidates if x is not None], dim=0
        )
        valid = torch.stack([x for x in captured_valid if x is not None], dim=0)
        return z, candidates, valid

    def forward(self, tokens: torch.Tensor, anchors: torch.Tensor) -> AnswerStateOutput:
        z, candidates, valid = self._encode_to_answer_anchor(tokens, anchors)
        b = z.shape[0]
        pos = self.answer_pos[None, :, :].expand(b, -1, -1)
        y = self.answer_seed(z)[:, None, :] + pos

        logits_steps: List[torch.Tensor] = []
        z_steps: List[torch.Tensor] = []
        for _ in range(self.reasoning_steps):
            y_summary = y.mean(dim=1)
            q = z + self.answer_feedback(y_summary)
            if self.cfg.use_memory:
                retrieved, _ = self.core.router.retrieve(q[:, None, :], candidates, valid)
                retrieved = retrieved[:, 0]
                gate = torch.sigmoid(
                    self.core.memory_gate(torch.cat([z, retrieved], dim=-1))
                )
                z = z + gate * retrieved
            z = z + self.core.memory_ffn(self.core.memory_ffn_norm(z))

            z_b = self.answer_context(z)[:, None, :].expand(-1, self.answer_slots, -1)
            proposal = self.answer_update(torch.cat([y, z_b, pos], dim=-1))
            gate_y = torch.sigmoid(self.answer_gate(torch.cat([y, z_b], dim=-1)))
            y = y + gate_y * proposal
            y = self.answer_mixer(y)

            logits_steps.append(self.core.lm_head(self.answer_norm(y)))
            z_steps.append(z)
        return AnswerStateOutput(logits_steps=logits_steps, z_steps=z_steps)


def answer_targets(
    answers: Sequence[str],
    answer_slots: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    rows: List[List[int]] = []
    for answer in answers:
        ids = list(answer.encode("utf-8", errors="replace")) + [EOS]
        if len(ids) > answer_slots:
            raise ValueError(
                f"answer requires {len(ids)} slots but answer_slots={answer_slots}"
            )
        rows.append(ids + [PAD] * (answer_slots - len(ids)))
    return torch.tensor(rows, dtype=torch.long, device=device)


def answer_state_loss(
    output: AnswerStateOutput,
    targets: torch.Tensor,
    *,
    step_weights: Sequence[float] | None = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if not output.logits_steps:
        raise ValueError("no answer-state logits")
    if step_weights is None:
        step_weights = [float(i + 1) for i in range(len(output.logits_steps))]
    if len(step_weights) != len(output.logits_steps):
        raise ValueError("step_weights must match reasoning steps")
    weights = torch.tensor(
        step_weights,
        dtype=output.logits_steps[0].dtype,
        device=output.logits_steps[0].device,
    )
    weights = weights / weights.sum().clamp_min(1e-8)
    losses = [
        F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=PAD)
        for logits in output.logits_steps
    ]
    total = sum(w * loss for w, loss in zip(weights, losses))
    return total, losses


def decode_answer_logits(logits: torch.Tensor) -> List[str]:
    ids = logits.argmax(dim=-1).detach().cpu().tolist()
    answers: List[str] = []
    for row in ids:
        out: List[int] = []
        for tok in row:
            if tok == EOS:
                break
            if 0 <= tok < 256:
                out.append(tok)
            elif tok == PAD:
                break
        answers.append(bytes(out).decode("utf-8", errors="replace"))
    return answers
