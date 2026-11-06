"""LRR-009 order intervention and data-identity integrity checks."""
from collections import Counter
import hashlib, tempfile
from pathlib import Path
import torch
from .lrr005_register import CausalRoleRegister, ID, vectorize
from .lrr008_invariance import objective, FORBIDDEN, alias_pair, latest_records, heldout
from .lrr009_curriculum import *

def test_schedules_have_identical_event_multisets_and_distinct_order():
    schedules=[schedule(a,11) for a in ARMS]
    assert all(Counter(s)==Counter(schedules[0]) for s in schedules)
    assert len({tuple(s) for s in schedules})==3
    assert [m for _,m in schedules[1]][:11]==['clean']*11
    assert [m for _,m in schedules[2]][:11]==['augmentation']*11
    assert schedules[0][:4]==[(0,'clean'),(0,'augmentation'),(1,'clean'),(1,'augmentation')]

def test_source_provenance_matches_archived_runtime_and_wrapper():
    base=Path(__file__).parent
    expected={'lrr005_register.py':'cdf9855fa689fa246ef2965b176beba27024c24f2341ceec6915aed2ab3b1740',
              'lrr008_invariance.py':'9ed44cf7a2e3914de1fb5ebb1d65e59191969536f7398d16f49c932608f43a93'}
    for f,h in expected.items():assert hashlib.sha256((base/f).read_bytes()).hexdigest()==h

def test_bank_latest_labels_reserved_pairs_and_noise_placement():
    bank,hashes=make_bank(999,4,8)
    assert len(bank)==len(hashes)==4
    for examples,x,y,q,mask,noise in bank:
        assert len(examples)==48
        for i,ex in enumerate(examples):
            assert alias_pair(ex) not in FORBIDDEN
            assert latest_records(ex)[(ex.subject,ex.family)][-1]==ex.answer
            assert not mask[i,ex.answer]
        assert not mask[:24].any()
        for i in range(24):assert torch.equal(x[i],x[i+24])
    assert len({event_digest(schedule(a,4),hashes,sorted_events=True) for a in ARMS})==1

def test_unchanged_objective_and_zero_noise_forward():
    torch.manual_seed(991);m=InvariantRegister()
    bank,_=make_bank(996,1);ex,x,y,q,mask,noise=bank[0]
    e1,r1,g1=m(x);e2,r2,g2=m.augmented_forward(x,torch.zeros_like(mask),noise)
    assert torch.equal(e1,e2) and torch.equal(r1,r2) and torch.equal(g1,g2)
    l,met=objective(m,x,y,q,mask,noise,'augmentation')
    l.backward();assert torch.isfinite(l) and torch.isfinite(m.code.weight.grad).all()
    assert sum(p.numel() for p in m.parameters())==4748

def test_hard_original_runtime_and_causal_prefix_parity():
    torch.manual_seed(900);m=InvariantRegister().eval()
    examples=[heldout(130_999_000_000+i,7,'untrained_in_vocab','dev') for i in range(5)]
    for ex,diag in zip(examples,predictions(m,examples)):
        assert m.route(ex.words)[0]==diag['pred']
        assert diag['pred'] is None or diag['pred']<len(ex.words)-1
    a=examples[0].words[:10];b=a+('mira','now','owns','value_nonce','.')
    x=torch.tensor([[ID.get(w,1) for w in a]]);y=torch.tensor([[ID.get(w,1) for w in b]])
    with torch.no_grad():h1=m.token_features(x);h2=m.token_features(y)
    assert torch.allclose(h1,h2[:,:10],atol=1e-6)

def test_short_three_schedule_execution_records_every_optimizer_step():
    torch.set_num_threads(1);bank,h=make_bank(998,2)
    receipts=[]
    with tempfile.TemporaryDirectory() as td:
        for arm in ARMS:
            _,receipt,cases=fit_schedule(997,arm,bank,h,td,dev_trials=3)
            assert receipt['updates']==4 and len(receipt['milestones'])==4
            assert receipt['clean_updates']==receipt['augmented_updates']==2
            assert all(r['optimizer_step_min']==r['step']==r['optimizer_step_max'] for r in receipt['milestones'])
            receipts.append(receipt)
    for field in ('init_tensor_sha256','bank_sha256','event_multiset_sha256'):
        assert len({r[field] for r in receipts})==1
    assert len({r['chronological_event_sha256'] for r in receipts})==3
