"""Integrity/anti-leakage gates for LRR-005, not scientific capability claims."""
import torch
from torch.nn import functional as F

from .lrr005_register import (
    CausalRoleRegister, FAMILIES, ID, RESERVED, assert_pair_split,
    make_example, make_training_pair, vectorize, fit,
)
from .run_lrr005 import (
    SymbolicRoleIndex, known_absent, unknown_relation, parse_raw,
    evaluate_one,
)


def test_both_training_questions_exclude_both_reserved_alias_pairs():
    for step in range(35):
        for i in range(16):
            a, b = make_training_pair(100_000 + step * 100 + i, step, i)
            assert_pair_split(a, training=True)
            assert_pair_split(b, training=True)
            assert a.words[:-2] == b.words[:-2]
            assert a.answer != b.answer
            for ex in (a,b):
                fact = next(f for f in ex.facts if f[-1] == ex.answer)
                aidx = FAMILIES[ex.family].index(fact[2])
                qidx = FAMILIES[ex.family].index(ex.question_alias)
                assert (aidx, qidx) not in RESERVED


def test_test_pair_is_new_and_value_position_is_not_supplied_to_inference():
    example = make_example(1400, 32, "test")
    assert_pair_split(example, training=False)
    assert example.words[example.answer].startswith("value")
    model = CausalRoleRegister()
    pos, diag = model.route(example.words)
    assert pos is None or 0 <= pos < len(example.words)-1
    assert diag["index_probes"] == 1
    assert "answer" not in model.route.__code__.co_varnames
    assert sum(p.numel() for p in model.parameters()) == 4748


def test_read_before_write_recurrence_and_prefix_causality():
    torch.manual_seed(13)
    model = CausalRoleRegister().eval()
    a = make_example(999,7,"train").words
    shorter = a[:12]
    longer = a[:12] + ("mira","now","owns","value999nobody",".")
    x1 = torch.tensor([[ID.get(w,1) for w in shorter]])
    x2 = torch.tensor([[ID.get(w,1) for w in longer]])
    with torch.no_grad():
        e1,r1,g1 = model(x1)
        e2,r2,g2 = model(x2)
        h = model.token_features(x1)
        codes = model.code(h).softmax(-1)
        for t in range(len(shorter)-1):
            expected_entity = g1[:,t,0:1]*F.one_hot(x1[:,t],num_classes=e1.shape[-1]).float() + (1-g1[:,t,0:1])*e1[:,t]
            expected_rel = g1[:,t,1:2]*codes[:,t] + (1-g1[:,t,1:2])*r1[:,t]
            assert torch.allclose(e1[:,t+1],expected_entity,atol=1e-6)
            assert torch.allclose(r1[:,t+1],expected_rel,atol=1e-6)
    assert torch.allclose(e1,e2[:,:len(shorter)],atol=1e-6)
    assert torch.allclose(r1,r2[:,:len(shorter)],atol=1e-6)


def test_symbolic_grammar_ceiling_and_negative_controls():
    symbol = SymbolicRoleIndex()
    symbol.train(seed=8,steps=60,batch_size=32)
    assert symbol.uf.count() == 4
    for layout in ("regular","shift","novel_fillers"):
        for i in range(12):
            ex = make_example(4_000_000+i,32,"test",layout=layout)
            assert symbol.route(ex.words) == ex.answer
            assert symbol.route(unknown_relation(ex)) is None
            assert symbol.route(known_absent(ex)) is None
            facts,entity,pred=parse_raw(ex.words)
            assert len(facts)==32


def test_bounded_two_update_training_and_metrics_preserve_failed_seeds():
    m, receipt = fit(18,steps=2,batch_size=8,credit=.2)
    assert receipt["anti_leakage_checked"] == 16
    assert receipt["parameters"] == 4748
    assert torch.isfinite(torch.tensor(receipt["final_objective"]))
    symbol = SymbolicRoleIndex()
    symbol.train(seed=18,steps=60,batch_size=8)
    result = evaluate_one(m,symbol,seed=8000,n=7,trials=3)
    assert result["regular"]["trials"] == 3
    assert result["known_absent"]["trials"] == 3
    assert result["unknown_relation"]["trials"] == 3
    assert 0 <= result["regular"]["recall_at_1"] <= 1
