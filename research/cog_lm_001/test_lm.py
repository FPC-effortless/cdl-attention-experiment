import copy
import math

import pytest
import torch
from model import Config, LM, SlotBank, Successors, mix, successor_ids


@pytest.mark.parametrize('arm',['transformer','mamba2','local','cog'])
def test_causal_cached_and_restorable(arm):
    torch.set_num_threads(1)
    torch.manual_seed(45)
    m=LM(Config(arm)).eval()
    x=torch.randint(1,2048,(2,48))
    with torch.no_grad():
        a,_=m(x)
        prefix,state=m(x[:,:19])
        clone=copy.deepcopy(state)
        pieces=[]
        for i in range(19,48):
            p,state=m(x[:,i:i+1],state)
            pieces.append(p)
        torch.testing.assert_close(torch.cat([prefix,*pieces],1),a,atol=2e-5,rtol=2e-5)
        again,_=m(x[:,19:20],clone)
        torch.testing.assert_close(again,pieces[0])
        perm=x.clone();perm[:,23:]=torch.randint(1,2048,(2,25))
        b,_=m(perm)
        torch.testing.assert_close(a[:,:23],b[:,:23],atol=2e-5,rtol=2e-5)
        assert torch.allclose(a.exp().sum(-1),torch.ones_like(a[:,:,0]),atol=2e-6)


def test_observe_then_lookup_revision_and_eviction():
    s=Successors(20)
    assert s.apply([3,4,5,3,4])[-1]==5
    assert s.apply([6,3,4])[-1]==6
    s.apply(list(range(20,55)))
    assert s.apply([3,4])[-1]==-1
    assert len(s.table)==20
    assert successor_ids(torch.tensor([[3,4,5],[3,4,8]]),256).eq(-1).all()


def test_memories_train_and_resets_isolate_documents():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    m=LM(Config('cog'))
    x=torch.tensor([[3,4,5,6]*12])
    z,_=m(x)
    loss=-z[0,36,3]
    loss.backward()
    assert all(torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in
               (m.bank.write.weight,m.bank.route.weight,m.bank.readout.weight,m.gate.weight))
    with torch.no_grad():
        a,_=m(x)
        m(torch.tensor([[3,4,7]*18]))
        b,_=m(x)
        torch.testing.assert_close(a,b)


def test_text_generation_survives_process_checkpoint(tmp_path):
    from argparse import Namespace
    from dataclasses import asdict
    from pathlib import Path
    from infer import infer
    torch.manual_seed(19)
    model=LM(Config('cog'))
    checkpoint=tmp_path/'model.pt'
    torch.save({'config':asdict(model.c),'state':model.state_dict(),'seed':19},checkpoint)
    state=tmp_path/'session.pt'
    args=Namespace(weights=checkpoint,tokenizer=Path(__file__).with_name('tokenizer.json'),seed=7,
                   prompt='The current direction is',resume_state=None,temperature=.8,tokens=8,save_state=None)
    complete=infer(args)
    args.tokens=3
    args.save_state=state
    prefix=infer(args)
    assert prefix[-1]!=0
    args.prompt=None;args.resume_state=state;args.save_state=None;args.tokens=5
    suffix=infer(args)
    assert prefix+suffix==complete
    checkpoint2=tmp_path/'other.pt'
    torch.save({'config':asdict(model.c),'state':model.state_dict(),'seed':20},checkpoint2)
    args.weights=checkpoint2
    with pytest.raises(ValueError,match='mismatch'):
        infer(args)


def test_reference_index_agrees_on_causal_successors_before_eviction():
    # Same posting rule as prior LRC CausalIndex, with latest (rank 0) as operand.
    # A posting for (t-2,t-1) becomes available only after observing its successor t.
    from bisect import bisect_left
    rng=torch.Generator().manual_seed(7)
    seq=torch.randint(0,5,(190,),generator=rng).tolist()
    s=Successors(256)
    postings={}
    for i,t in enumerate(seq):
        if i>=2:
            postings.setdefault(tuple(seq[i-2:i]),[]).append(i)
        key=tuple(seq[max(0,i-1):i+1])
        known=postings.get(key,[])
        before=bisect_left(known,i+1)
        expected=seq[known[before-1]] if before else -1
        assert s.ingest(t)==expected


def test_foreign_slot_probe_leaves_unwritten_prefix_unchanged():
    torch.set_num_threads(1)
    torch.manual_seed(89)
    m=LM(Config('cog')).eval()
    a=torch.randint(0,2048,(2,32))
    with torch.no_grad():
        p,_=m(a)
        q,_=m(a,intervention='foreign_slots')
        torch.testing.assert_close(p[:,:16],q[:,:16])
        assert (p[:,16:]-q[:,16:]).abs().max()>.0001
