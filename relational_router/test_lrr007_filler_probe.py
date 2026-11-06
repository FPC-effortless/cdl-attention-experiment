"""Independently check LRR-007P fast local encoder equals frozen hard route."""
import torch

from .lrr005_register import CausalRoleRegister
from .lrr007_dual_credit import heldout_example
from .lrr007_filler_probe import FILLER_MAPS,changed_words,hard_batch


def test_filler_changes_are_position_preserving():
    ex=heldout_example(122334,32,split="test",layout="regular")
    for name,m in FILLER_MAPS.items():
        new=changed_words(ex,m)
        assert len(new.words)==len(ex.words)
        assert new.answer==ex.answer
        assert new.words[new.answer]==ex.words[ex.answer]
        assert new.question_alias==ex.question_alias


def test_batch_hard_execution_matches_original_source():
    torch.manual_seed(23)
    model=CausalRoleRegister().eval()
    for n in (7,32,128):
        cases=[heldout_example(134*1000+n+i,n,split="test") for i in range(5)]
        for mapping in FILLER_MAPS.values():
            modified=[changed_words(ex,mapping) for ex in cases]
            expected=[model.route(ex.words)[0] for ex in modified]
            assert hard_batch(model,modified)==expected


def test_no_answer_pointer_is_outside_causal_history():
    model=CausalRoleRegister().eval()
    ex=heldout_example(4567,7,split="test")
    selection=hard_batch(model,[ex])[0]
    assert selection is None or 0<=selection<len(ex.words)-1
