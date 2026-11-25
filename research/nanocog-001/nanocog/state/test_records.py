"""Fixed/math invariants and optional source-only integration with actual PULSE."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch

from . import CLEAR, HOLD, SET, StableRecordMemory, TypedRecordBoard, WritePacket

torch.set_num_threads(1)


def item(addr,op,val):
    return WritePacket(*(torch.tensor(x,dtype=torch.long) for x in (addr,op,val)))


def eq(a,b):
    assert all(torch.equal(x,y) for x,y in zip(a._parts(),b._parts()))


def test_exact_typed_custody_and_revision():
    b=TypedRecordBoard(); s=b.empty(2)
    s0=s.detach()
    # box 1 -> owner 6, cargo record 5 -> stable box 1, unresolved until occupied.
    s,_=b.apply(s,item([1,5],[SET,SET],[6,9]))
    assert b.read(s,torch.tensor([1,5])).resolved.tolist()==[6,-1]
    s,_=b.apply(s,item([5,1],[SET,SET],[9,2]))
    assert b.read(s,torch.tensor([5,5])).resolved.tolist()==[6,2]
    pre=s.detach()
    s,rec=b.apply(s,item([1,5],[SET,SET],[8,2]))
    assert rec.legal.tolist()==[False,False]       # Holder type versus box link type.
    assert rec.changed.tolist()==[False,False]
    assert torch.equal(pre.value,s.value) and torch.equal(pre.revision,s.revision)
    assert torch.equal(pre.last_change,s.last_change)
    assert s.clock.tolist()==[3,3]
    s,rec=b.apply(s,item([1,1],[SET,CLEAR],[6,0]))
    assert rec.changed.tolist()==[False,True]
    assert s.revision[:,1].tolist()==[1,2]
    assert b.read(s,torch.tensor([5,5])).resolved.tolist()==[6,-1]
    assert (s0.value==-1).all()
    assert b.validate_state(s) is s
    with pytest.raises(ValueError,match="chronology"):
        b.validate_state(replace(s,clock=torch.tensor([0,0])))


def test_differential_512_packets_with_invalid_types_and_inactive_rows():
    rng=random.Random(33419)
    b=TypedRecordBoard(); batch=4; s=b.empty(batch)
    val=[[-1]*8 for _ in range(batch)]
    rev=[[0]*8 for _ in range(batch)]
    at=[[-1]*8 for _ in range(batch)]
    clocks=[0]*batch
    values=lambda i,v: (0<=v<8) if i<4 else (8<=v<12)
    for _ in range(128):
        a=[rng.randint(-2,9) for _ in range(batch)]
        op=[rng.randint(0,4) for _ in range(batch)]
        v=[rng.randint(-2,13) for _ in range(batch)]
        live=[rng.choice((True,True,False)) for _ in range(batch)]
        assert not any(x.data_ptr()==y.data_ptr()
                       for x,y in zip(s._parts(),b.empty(batch)._parts()))
        new,receipt=b.apply(s,item(a,op,v),torch.tensor(live))
        legal=[];changed=[]
        for i,(ar,rr,vl,active) in enumerate(zip(a,op,v,live)):
            ok=active and (rr==HOLD or 0<=ar<8 and (rr==CLEAR or rr==SET and values(ar,vl)))
            change=ok and rr!=HOLD and val[i][ar]!=(-1 if rr==CLEAR else vl)
            if change:
                val[i][ar]=-1 if rr==CLEAR else vl
                rev[i][ar]+=1
                at[i][ar]=clocks[i]
            clocks[i]+=int(active)
            legal.append(ok);changed.append(change)
        assert new.value.tolist()==val and new.revision.tolist()==rev and new.last_change.tolist()==at
        assert clocks==new.clock.tolist() and legal==receipt.legal.tolist() and changed==receipt.changed.tolist()
        b.validate_state(new)
        query=[rng.randint(-1,8) for _ in range(batch)]
        solved=[]
        for i,k in enumerate(query):
            resolved=-1 if not 0<=k<8 else val[i][k]
            if resolved>=8:resolved=val[i][resolved-8]
            solved.append(resolved)
        assert b.read(new,torch.tensor(query)).resolved.tolist()==solved
        s=new


def test_custom_schema_rejects_second_link_and_never_peeks():
    legal=torch.ones(3,4,dtype=torch.bool)
    links=torch.tensor([-1,-1,1,2])
    b=TypedRecordBoard(legal,links); s=b.empty(1)
    for a,v in [(0,2),(1,3),(2,0)]:
        s,_=b.apply(s,item([a],[SET],[v]))
    assert b.read(s,torch.tensor([0])).value.item()==2
    assert b.read(s,torch.tensor([0])).resolved.item()==-1  # Explicit one-hop bound.
    assert b.read(s,torch.tensor([1])).target.item()==2


def force(module,idx):
    module.weight.data.zero_();module.bias.data.fill_(-9);module.bias.data[idx]=9


def test_strict_before_chunk_replay_and_padding_nan():
    torch.manual_seed(4997)
    m=StableRecordMemory(16).eval()
    force(m.write_operation,SET);force(m.write_value,0)
    with torch.no_grad():
        m.write_address.weight.zero_()
        m.read_address.weight.zero_()
    x=torch.randn(3,7,16)
    live=torch.tensor([[True]*7,[True]*4+[False]*3,[False]+[True]*6])
    x[~live]=torch.nan
    y,st,trace=m(x,valid=live,return_state=True,return_aux=True)
    assert trace['read_valid'][:,0].tolist()==[False,False,False]
    assert trace['write_changed'].sum(1).tolist()==[1,1,1]
    assert st.clock.tolist()==[7,4,6]
    assert y.isfinite().all() and y[~live].eq(0).all()
    first,mid=m(x[:,:4],valid=live[:,:4],return_state=True)
    second,done=m(x[:,4:],mid,valid=live[:,4:],return_state=True)
    assert torch.equal(y,torch.cat((first,second),dim=1));eq(st,done)
    alt=x.clone();alt[:,4:]=17
    changed,_,_=m(alt,valid=live,return_state=True,return_aux=True)
    assert torch.equal(changed[:,:4],y[:,:4])
    idx=torch.tensor([2,0,2]); cloned=mid.index_select(idx)
    out_reordered, last=m(x[idx,4:],cloned,valid=live[idx,4:],return_state=True)
    assert torch.equal(out_reordered,second[idx]);eq(last,done.index_select(idx))


def test_loss_reaches_payload_but_not_hard_proposal():
    torch.manual_seed(5)
    m=StableRecordMemory(16)
    force(m.write_operation,SET)
    m.write_address.weight.data.zero_()
    m.read_address.weight.data.zero_()
    y=m(torch.rand(2,5,16))
    y.square().sum().backward()
    assert m.payload.weight.grad is not None
    for head in (m.write_address,m.write_operation,m.write_value,m.read_address):
        assert head.weight.grad is None
    assert m.key.grad is None


@pytest.mark.skipif(not os.environ.get('PULSE006_SOURCE'),reason='optional exact upstream source only')
def test_actual_archived_pulse_source_integration():
    root=Path(os.environ['PULSE006_SOURCE'])
    p=root/'parent'/'networks.py'
    q=root/'parent'/'world.py'
    expected=("8edd054a26a0fe1842d67d3cfaa5d3e9fc8ec4d07c853f93cd1b6c1f7e809ece",
              "fb0e2cb85789435f910aa122a5e36be091be1b985c671c32e602c707f91d86c2")
    assert tuple(hashlib.sha256(x.read_bytes()).hexdigest() for x in (p,q))==expected
    run=r"""
import torch
from networks import Pulse
from nanocog.state import Pulse006TypedAdapter
torch.set_num_threads(1)
torch.manual_seed(404)
m=Pulse().eval(); m.enable_handoff('event')
a=Pulse006TypedAdapter(m).eval()
for layer,i in [(m.addr,0),(m.effect,4),(m.lookup,0)]:
    layer.weight.data.zero_();layer.bias.data.fill_(-9);layer.bias.data[i]=9
m.commit.weight.data.zero_();m.commit.bias.data.fill_(9)
x=torch.randint(0,258,(2,5,18)); lengths=torch.tensor([5,3])
with torch.inference_mode():
    result,st,trace=a(x,lengths,return_trace=True)
    alter=x.clone();alter[0,4:]=199;alter[1,2:]=199
    altered,after,_=a(alter,lengths,return_trace=True)
assert result.resolved.tolist()==[4,4] and altered.resolved.tolist()==[4,4]
assert st.clock.tolist()==[4,2] and st.revision[:,0].tolist()==[1,1]
assert len(trace)==4 and all(torch.equal(x,y) for x,y in zip(st._parts(),after._parts()))
"""
    env=dict(os.environ)
    env["PYTHONPATH"]=os.pathsep.join((str(root/'parent'),str(Path(__file__).resolve().parents[2])))
    subprocess.run([sys.executable,"-c",run],env=env,check=True,timeout=30)


def test_custom_link_map_with_default_permits_is_preserved():
    links = torch.full((12,), -1, dtype=torch.long)
    b = TypedRecordBoard(links=links)
    assert torch.equal(b.links, links)


def test_import_rejects_impossible_revision_history():
    b = TypedRecordBoard()
    s = b.empty(1)
    s, _ = b.apply(s, item([0], [SET], [1]))
    s, _ = b.apply(s, item([1], [SET], [2]))
    b.validate_state(s)
    same_event = s.last_change.clone()
    same_event[0, 1] = 0
    with pytest.raises(ValueError, match="chronology|duplicate"):
        b.validate_state(replace(s, last_change=same_event))
    impossible = s.revision.clone()
    impossible[0, 1] = 2
    with pytest.raises(ValueError, match="chronology"):
        b.validate_state(replace(s, revision=impossible))
