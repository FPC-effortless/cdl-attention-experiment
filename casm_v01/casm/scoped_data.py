from __future__ import annotations

import random
from typing import List

import torch

from .data import BOS, EOS, PAD, Example, generate_example


def make_episode_batch(
    batch_size: int,
    seq_len: int,
    seed: int,
    *,
    hard: bool = False,
    return_answer_mask: bool = False,
):
    """Build one independent task episode per batch row.

    The existing packed curriculum concatenates independent tasks with SEP while
    CASM carries episodic/persistent memory across the separator. Evaluation,
    however, starts each problem with clean state. This control removes that
    train/eval scope mismatch: every row contains exactly one task and therefore
    one memory lifetime.

    Episodes that do not fit are rejected and resampled rather than truncated,
    because truncating can silently remove the query or answer.
    """
    if seq_len < 4:
        raise ValueError("seq_len too small")

    rng = random.Random(seed)
    rows: List[List[int]] = []
    masks: List[List[bool]] = []
    examples: List[Example] = []
    marker = b"answer "

    for _ in range(batch_size):
        for _attempt in range(1000):
            ex = generate_example(rng, hard=hard)
            body_bytes = ex.text.encode("utf-8", errors="replace")
            if len(body_bytes) + 2 <= seq_len:
                break
        else:
            raise RuntimeError(
                f"Could not sample an episode fitting seq_len={seq_len}"
            )

        row = [BOS] + list(body_bytes) + [EOS]
        ans_mask = [False] * len(row)
        marker_at = body_bytes.rfind(marker)
        if marker_at < 0:
            raise ValueError("generated episode has no answer marker")
        answer_start = 1 + marker_at + len(marker)
        answer_len = len(ex.answer.encode("utf-8", errors="replace"))
        for j in range(answer_start, min(answer_start + answer_len, len(row) - 1)):
            ans_mask[j] = True

        pad_n = seq_len - len(row)
        row.extend([PAD] * pad_n)
        ans_mask.extend([False] * pad_n)
        rows.append(row)
        masks.append(ans_mask)
        examples.append(ex)

    tok = torch.tensor(rows, dtype=torch.long)
    if return_answer_mask:
        return tok, examples, torch.tensor(masks, dtype=torch.bool)
    return tok, examples
