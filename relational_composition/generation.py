"""Safe autoregressive generation with PAD masking, EOS, and ragged prompts.

This API works with the existing V3 and V4 cached decoding classes, without
changing their learned weights. It never feeds PAD=0 back into the model.
Ended rows feed a fixed valid EOS token during remaining steps; they are
independent documents and cannot affect active rows' inference.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence
import torch


@dataclass(frozen=True)
class GenerationResult:
    generated: torch.Tensor
    lengths: torch.Tensor
    terminated: torch.Tensor
    steps: int


@torch.inference_mode()
def greedy_generate(stream, prompt: torch.Tensor, *,
                    max_new_tokens: int, eos_token_id: int | None = None,
                    max_context_tokens: int | None = None) -> GenerationResult:
    if prompt.ndim != 2 or prompt.shape[1] == 0:
        raise ValueError('expected nonempty [batch, length] prompt')
    if max_new_tokens < 0:
        raise ValueError('negative max_new_tokens')
    vocab = stream.config.vocab
    if vocab <= 1:
        raise ValueError('model has no non-PAD tokens')
    if eos_token_id is not None and not 1 <= eos_token_id < vocab:
        raise ValueError('EOS must be an in-vocabulary non-PAD token')
    if max_context_tokens is not None and prompt.shape[1] + max_new_tokens > max_context_tokens:
        raise ValueError('requested sequence exceeds explicit context budget')
    batch=prompt.shape[0]
    if max_new_tokens == 0:
        return GenerationResult(prompt.new_empty((batch,0)),
               prompt.new_zeros((batch,)),
               torch.zeros(batch,dtype=torch.bool,device=prompt.device),0)
    logits=stream.prefill(prompt)
    output=torch.zeros((batch,max_new_tokens),dtype=torch.long,device=prompt.device)
    lengths=torch.zeros(batch,dtype=torch.long,device=prompt.device)
    done=torch.zeros(batch,dtype=torch.bool,device=prompt.device)
    fallback=eos_token_id if eos_token_id is not None else 1
    consumed=0
    for i in range(max_new_tokens):
        scores=logits.clone()
        scores[:,0]=float('-inf')  # Never decode padding/undefined state.
        ids=scores.argmax(-1)
        output[:,i]=torch.where(done,torch.zeros_like(ids),ids)
        lengths+=(~done).to(lengths.dtype)
        if eos_token_id is not None:
            done |= (ids==eos_token_id)
        consumed=i+1
        if i+1 == max_new_tokens or bool(done.all()):
            break
        ids=torch.where(done,torch.full_like(ids,fallback),ids)
        logits=stream.step(ids)
    return GenerationResult(output[:,:consumed],lengths,done,consumed)


@torch.inference_mode()
def grouped_greedy_generate(stream_factory: Callable[[],object],
                            prompts: Sequence[Sequence[int]], *,
                            max_new_tokens: int, eos_token_id: int | None = None,
                            max_context_tokens: int | None = None):
    """Group equal-length prompts, preserving original order and row isolation.

    This avoids padding a model that expects one strictly causal document per
    row; it makes no claim that grouping beats a fused variable-length kernel.
    """
    if not prompts:
        return []
    grouped=defaultdict(list)
    for i,prompt in enumerate(prompts):
        if len(prompt)<1:
            raise ValueError('empty prompt')
        grouped[len(prompt)].append((i,prompt))
    output=[None]*len(prompts)
    for length,entries in grouped.items():
        tokens=torch.tensor([p for _,p in entries],dtype=torch.long)
        result=greedy_generate(stream_factory(),tokens,
                max_new_tokens=max_new_tokens,eos_token_id=eos_token_id,
                max_context_tokens=max_context_tokens)
        for row,(i,_) in enumerate(entries):
            output[i]=result.generated[row,:int(result.lengths[row])].tolist()
    return output
