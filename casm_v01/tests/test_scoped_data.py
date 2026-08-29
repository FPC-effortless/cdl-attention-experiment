import torch

from casm.data import BOS, EOS, PAD, SEP
from casm.scoped_data import make_episode_batch


def test_episode_batch_has_one_scope_per_row():
    toks, examples, mask = make_episode_batch(
        16, 385, 20261201, hard=True, return_answer_mask=True
    )
    assert toks.shape == mask.shape == (16, 385)
    assert len(examples) == 16
    assert torch.all(toks[:, 0] == BOS)
    # No packed-task separator: every row is one independent memory lifetime.
    assert not (toks == SEP).any()
    for row in toks:
        eos = (row == EOS).nonzero(as_tuple=False)
        assert len(eos) == 1
        eos_i = int(eos[0])
        assert torch.all(row[eos_i + 1 :] == PAD)


def test_episode_answer_mask_matches_literal_answer():
    toks, examples, mask = make_episode_batch(
        12, 385, 20261202, hard=False, return_answer_mask=True
    )
    for row, ex, row_mask in zip(toks, examples, mask):
        answer_bytes = bytes(row[row_mask].tolist()).decode("utf-8")
        assert answer_bytes == ex.answer
        assert not row_mask[0]
        assert not row_mask[row == PAD].any()
