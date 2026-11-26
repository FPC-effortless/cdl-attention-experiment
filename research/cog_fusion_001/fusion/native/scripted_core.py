"""Scripted form of the production board recurrence, with identical tensors.

Only Python dispatch is removed. Native hard routing, straight-through
surrogates, read-before-write order, and torch's GRUCell operator are retained.
The traced/oracle Python reference remains in model.py for diagnostics.
"""

from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


@torch.jit.script
def _hard_categorical(logits: Tensor) -> Tensor:
    probability = logits.float().softmax(-1)
    hard = F.one_hot(probability.argmax(-1), probability.size(-1)).to(probability.dtype)
    return hard + (probability - probability.detach())


@torch.jit.script
def recurrent_core(
    qe_base: Tensor, qr_base: Tensor, we_base: Tensor, wr_base: Tensor,
    wc_base: Tensor, wv_base: Tensor, controller_inputs: Tensor,
    memory: Tensor, controller: Tensor,
    query_feedback_weight: Tensor, writer_feedback_weight: Tensor,
    gru_weight_ih: Tensor, gru_weight_hh: Tensor,
    gru_bias_ih: Tensor, gru_bias_hh: Tensor,
    read_to_value_weight: Tensor,
    n_entities: int, n_relations: int, memory_dim: int,
    feedback_enabled: bool, memory_enabled: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    current_memory = memory.float()
    c = controller.float()
    reads = torch.jit.annotate(List[Tensor], [])
    controllers = torch.jit.annotate(List[Tensor], [])
    for t in range(qe_base.size(1)):
        q_feedback = F.linear(c, query_feedback_weight)
        qe_delta, qr_delta = q_feedback.split([n_entities, n_relations], -1)
        qe = qe_base[:, t] + qe_delta
        qr = qr_base[:, t] + qr_delta
        query = (_hard_categorical(qe).unsqueeze(-1)
                 * _hard_categorical(qr).unsqueeze(-2)).flatten(-2)
        if memory_enabled:
            read = (query.unsqueeze(-1) * current_memory).sum(-2)
        else:
            read = current_memory.new_zeros((qe_base.size(0), memory_dim))
        if feedback_enabled and memory_enabled:
            internal_read = read
        else:
            internal_read = torch.zeros_like(read)
        cell_input = torch.cat((controller_inputs[:, t].float(), internal_read), -1)
        # This is the same native operator called by nn.GRUCell.forward.
        c = torch.gru_cell(cell_input, c, gru_weight_ih, gru_weight_hh,
                           gru_bias_ih, gru_bias_hh).float()

        w_feedback = F.linear(c, writer_feedback_weight)
        we_delta, wr_delta, wc_delta, wv_delta = w_feedback.split(
            [n_entities, n_relations, 1, memory_dim], -1)
        we = we_base[:, t] + we_delta
        wr = wr_base[:, t] + wr_delta
        wc = wc_base[:, t] + wc_delta
        wv = wv_base[:, t] + wv_delta + F.linear(internal_read, read_to_value_weight)
        address = (_hard_categorical(we).unsqueeze(-1)
                   * _hard_categorical(wr).unsqueeze(-2)).flatten(-2)
        probability = wc.sigmoid()
        hard_commit = (probability >= 0.5).to(probability.dtype)
        commit = hard_commit + (probability - probability.detach())
        value = wv.tanh()
        if memory_enabled:
            weight = (commit * address).unsqueeze(-1)
            current_memory = ((1.0 - weight) * current_memory
                              + weight * value.unsqueeze(-2))
        reads.append(read)
        controllers.append(c)
    return (torch.stack(reads, dim=1), torch.stack(controllers, dim=1),
            current_memory.clone(), c.clone())
