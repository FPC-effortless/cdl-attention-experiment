import copy, math
from collections import Counter
import torch
import pytest
from .lrr008_invariance import InvariantRegister, augmented_batch, objective, FORBIDDEN, alias_pair
from .lrr009_curriculum import schedule
from .lrr010_entropy import entropy_objective, semantic_examples, semantic_statistics
from .run_lrr010 import negative
from .lrr008_invariance import heldout, SymbolicIndex

@pytest.mark.parametrize('mode',['clean','augmentation'])
def test_zero_coefficient_loss_and_all_gradient_parity(mode):
    torch.set_num_threads(1);torch.manual_seed(998)
    model=InvariantRegister();_,x,y,q,mask,noise=augmented_batch(998,0,2)
    old,_=objective(model,x,y,q,mask,noise,mode)
    new,_=entropy_objective(model,x,y,q,mask,noise,mode,0.)
    assert torch.allclose(old,new,atol=1e-6,rtol=1e-6)
    a=torch.autograd.grad(old,tuple(model.parameters()),retain_graph=True)
    b=torch.autograd.grad(new,tuple(model.parameters()))
    for x,y in zip(a,b):assert torch.allclose(x,y,atol=1e-6,rtol=1e-5)


def test_only_the_marginal_entropy_is_added():
    torch.manual_seed(990);m=InvariantRegister()
    _,x,y,q,mask,noise=augmented_batch(999,2,2)
    zero,z=entropy_objective(m,x,y,q,mask,noise,'augmentation',0.)
    val,t=entropy_objective(m,x,y,q,mask,noise,'augmentation',.1)
    assert abs(float((val-zero).detach())+.1*z['marginal_entropy'])<1e-5
    assert 0<=t['marginal_entropy']<=math.log(8)+1e-6
    # Sample certainty and batch diversity are mathematically different.
    p=torch.eye(4)
    hbar=-(p.mean(0)*p.mean(0).log()).sum()
    hs=-(p*p.clamp_min(1e-9).log()).sum(1).mean()
    assert hbar>1 and hs==0


def test_semantic_panel_balanced_legal_unique_histories():
    examples=semantic_examples(99001,2)
    assert len(examples)==32
    assert set(Counter((e.family,e.question_alias) for e in examples).values())=={2}
    assert len({e.words[:e.words.index('what')] for e in examples})==32
    assert all(alias_pair(e) not in FORBIDDEN for e in examples)


def semantic_rows(codes):
    return [{'true_family':str(i//8),'query_hard_relation_code':codes[i//8],
             'gold_address_match':True,'collision_pairs':0,'nonvalue_writes':0,
             'writes':1,'correct':True} for i in range(32)]


def test_semantics_not_equal_to_occupancy_or_accuracy():
    separated=semantic_statistics(semantic_rows([0,1,2,3]))
    merged=semantic_statistics(semantic_rows([0,0,1,2]))
    collapsed=semantic_statistics(semantic_rows([0,0,0,0]))
    assert not separated['semantic_collapse'] and abs(separated['hard_effective_codes']-4)<1e-7
    assert merged['semantic_collapse'] and not merged['severe_starvation']
    assert collapsed['semantic_collapse'] and collapsed['severe_starvation']
    # We intentionally set retrieval correct in ALL test rows: semantic
    # collapse classification must not simply be a low-accuracy threshold.
    assert collapsed['recall']==1.0


def test_negative_support_removed_with_known_query():
    ex=heldout(99003,32,'regular',split='dev');n=negative(ex,'known_absent')
    records,entity,_=SymbolicIndex.parse(n.words)
    from .lrr005_register import FAMILIES
    assert not any(a==entity and b in FAMILIES[ex.family] for a,b,_ in records)
    assert n.words[-2]==ex.words[-2] and n.answer==-1
    u=negative(ex,'unknown_relation');assert u.words[-2]=='lrr010_unknown_relation'


def test_all_schedules_have_exact_multiset():
    a=schedule('interleaved',4)
    assert Counter(a)==Counter(schedule('clean_first',4))==Counter(schedule('augmentation_first',4))


def test_model_inference_is_not_modified():
    torch.manual_seed(992)
    m=InvariantRegister();ex=heldout(99123,7,'regular')
    pred,d=m.route(ex.words)
    assert pred is None or 0<=pred<len(ex.words)-1
    assert d['index_probes']==1 and sum(p.numel() for p in m.parameters())==4748
