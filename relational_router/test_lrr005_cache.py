"""Independent late mechanical gate: persistent chunk equivalence at hard inference."""
from copy import deepcopy

import pytest
import torch

from .lrr005_register import CausalRoleRegister, make_example
from .lrr005_cache import CachedRoleRegister


def split_question(words):
    pos=words.index("what")
    return words[:pos], words[pos:]


@pytest.mark.parametrize("model_variant", ["random","all-write","no-write"])
@pytest.mark.parametrize("facts", [7, 32, 128])
def test_chunked_and_full_hard_route_agree(facts, model_variant):
    torch.manual_seed(121)
    m=CausalRoleRegister().eval()
    if model_variant == "all-write":
        with torch.no_grad():
            m.role.weight.zero_()
            m.role.bias.fill_(3.0)
    if model_variant == "no-write":
        with torch.no_grad():
            m.role.weight.zero_()
            m.role.bias.fill_(-3.0)

    ex=make_example(77000+facts,facts,split="test",layout="shift")
    history,question=split_question(ex.words)
    full_position,diag=m.route(ex.words)
    whole=CachedRoleRegister(m)
    whole.append(history)
    whole_answer=whole.query(question)
    assert whole_answer["token_index"]==full_position
    assert whole_answer["encoded_tokens"]<=m.conv.kernel_size[0]-1+len(question)-1
    assert whole_answer["index_probes"]==1

    for size in (1,3,7,19,10000):
        state=CachedRoleRegister(m)
        for i in range(0,len(history),size):
            state.append(history[i:i+size])
        assert state.tokens==list(history)
        assert state.writes==whole.writes
        assert state.postings==whole.postings
        assert state.query(question)["token_index"]==full_position
        prior=deepcopy(state.postings)
        prior_tokens=tuple(state.tokens)
        for _ in range(3):
            assert state.query(question)["token_index"]==full_position
        assert state.postings==prior
        assert tuple(state.tokens)==prior_tokens
    if model_variant=="all-write":
        assert whole.writes>=len(history)//2
    if model_variant=="no-write":
        assert whole.writes==0


def test_cache_rejects_question_as_history_and_missing_query_marker():
    state=CachedRoleRegister(CausalRoleRegister())
    with pytest.raises(ValueError):
        state.append(("what","does","mira","own","?"))
    with pytest.raises(ValueError):
        state.query(("what","does","mira","owns"))


def test_two_append_batches_hold_correct_prior_register_and_indices():
    torch.manual_seed(22)
    model=CausalRoleRegister().eval()
    first=make_example(991,7,split="dev")
    second=make_example(992,7,split="dev")
    h1,q1=split_question(first.words)
    h2,q2=split_question(second.words)
    state=CachedRoleRegister(model)
    state.append(h1)
    p1=state.query(q1)
    state.append(h2)
    p2=state.query(q2)
    # Same state/corpus should be reproduced by one contiguous prefill.
    entire=CachedRoleRegister(model)
    entire.append(h1+h2)
    assert state.postings==entire.postings
    assert state.tokens==entire.tokens
    assert p2["token_index"]==entire.query(q2)["token_index"]
    assert p1["token_index"] is None or p1["token_index"] < len(h1)
