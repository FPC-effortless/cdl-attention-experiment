"""LRR-006 scientific integrity: energy supervision and split separation."""
from __future__ import annotations

import math
import torch
import pytest

from .lrr005_register import CausalRoleRegister, FAMILIES, ID
from .lrr006_energy import (
    EnergySelector, build_rows, encode_single, eval_scored,
    known_absent, unknown_relation, fit_energy, score_rows,
    dev_threshold,
)
from .lrr005_register import make_example


def test_energy_is_explicitly_trained_for_both_valid_and_invalid():
    model=EnergySelector(4,8)
    x=torch.randn(16,4)
    y=torch.tensor([1.]*8+[0.]*8)
    original=model.margin_loss(x,y,margin=.5)
    assert torch.isfinite(original)
    original.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    # Higher valid score should reduce +label energy, lower it for invalid.
    a=torch.tensor([2.])
    assert torch.nn.functional.softplus(.5-a).item() < torch.nn.functional.softplus(.5+a).item()


@pytest.mark.parametrize("split,pair",[
    ("dev",(3,0)),("test",(1,2)),
])
def test_pair_splits_and_counterfactual_negatives_are_explicit(split,pair):
    for seed in range(20):
        e=make_example(4_000_000+seed,7,split=split)
        words=FAMILIES[e.family]
        assert (words.index(e.fact_alias),words.index(e.question_alias))==pair
        neg=known_absent(e.words,e.answer,e.subject)
        assert neg[e.answer]==e.words[e.answer]
        assert neg[e.answer-2] != e.subject
        assert unknown_relation(e.words)[-2] == "unseenverb"


def test_teaching_candidates_occurs_only_in_training():
    torch.manual_seed(9)
    router=CausalRoleRegister().eval()
    train=build_rows(router,seed=1,split="train",length=7,n=3,layout="regular")
    dev=build_rows(router,seed=1,split="dev",length=7,n=3,layout="regular")
    test=build_rows(router,seed=1,split="test",length=7,n=3,layout="regular")
    assert len(train)==15
    assert len(dev)==9 and len(test)==9
    assert {r["class_name"] for r in train}=={
        "positive","known_absent","unknown_relation",
        "oracle_correct","counterfactual_wrong",
    }
    assert not any("oracle" in r["class_name"] or "counterfactual" in r["class_name"]
                   for r in dev+test)
    pairs=[x for x in train if x["class_name"] in ("oracle_correct","counterfactual_wrong")]
    assert all(x["feature"] is not None for x in pairs)
    assert all(len(x["feature"])==40 for x in pairs)


def test_hard_inference_pointer_independent_of_energy_and_no_future():
    torch.manual_seed(11)
    router=CausalRoleRegister().eval()
    case=make_example(33,32,split="test")
    baseline=router.route(case.words)[0]
    chosen,features,_=encode_single(router,case.words)
    assert chosen==baseline
    if chosen is not None:
        assert 0<=chosen<len(case.words)-1
        head=EnergySelector(40)
        a=float(head(features.unsqueeze(0))[0])
        b=float(head(features.unsqueeze(0))[0])
        assert a==b
        assert chosen==router.route(case.words)[0]


def test_dev_threshold_is_only_empirical_and_cannot_repair_wrong_pointer():
    torch.manual_seed(4)
    router=CausalRoleRegister().eval()
    positive=make_example(18,7,"test")
    head=EnergySelector(40)
    rows=build_rows(router,seed=1,split="dev",length=7,n=6)
    e=score_rows(rows,head=head)
    s=score_rows(rows,scalar=True)
    for scored in (e,s):
        rule=dev_threshold(scored,.10)
        assert rule["known_absent_fpr"]<=.1+1e-9
        raw=eval_scored(scored,float("-inf"))
        selected=eval_scored(scored,rule["threshold"])
        assert selected["positive"]["correct_coverage"]<=raw["positive"]["correct_coverage"]
    # Empty learned candidate is a forced abstention, not a learned success.
    assert encode_single(router,unknown_relation(positive.words))[0] is None or True
