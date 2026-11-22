"""Inert reference checks only. No data access or optimization."""
import torch
import pytest
from .router import BoundedRoute

torch.set_num_threads(1)

def close(a,b):
    torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-6)

def state_close(a,b):
    close(a.values,b.values)
    torch.testing.assert_close(a.positions,b.positions)
    torch.testing.assert_close(a.cursor,b.cursor)

def model(**kwargs):
    torch.manual_seed(2901)
    return BoundedRoute(10,block=3,**kwargs)

@pytest.mark.parametrize("training",[True,False])
def test_causal_candidate_and_chunks(training):
    m=model(retained_blocks=4).train(training)
    torch.manual_seed(3704)
    x=torch.randn(3,24,10)
    with torch.no_grad():
        y,s,idx=m.forward_with_candidates(x)
        assert idx.shape==(3,24,2,2)
        assert (idx[:,0]==-1).all() and (y[:,0]==0).all()
        assert ((idx<0)|(idx<torch.arange(24)[None,:,None,None])).all()
        assert ((idx<0)|(idx//3>=torch.arange(24)[None,:,None,None]//3-3)).all()
        x2=x.clone();x2[:,12:]=300*torch.randn_like(x[:,12:])
        y2,_,idx2=m.forward_with_candidates(x2)
        close(y[:,:12],y2[:,:12]);torch.testing.assert_close(idx[:,:12],idx2[:,:12])
        chunks=[];traces=[];st=None
        for sub in (x[:,:1],x[:,1:9],x[:,9:17],x[:,17:]):
            z,st,q=m.forward_with_candidates(sub,st)
            chunks.append(z);traces.append(q)
        close(torch.cat(chunks,1),y);torch.testing.assert_close(torch.cat(traces,1),idx)
        state_close(st,s)

def test_expiry_collision_nonpersistent_disable():
    m=model(retained_blocks=3).eval()
    with torch.no_grad():
        for layer in (m.publisher,m.subscriber):
            layer.weight.zero_()
        m.admit.weight.zero_();m.admit.bias.fill_(3)
        x=torch.randn(1,3,10)
        _,s,idx=m.forward_with_candidates(x)
        assert idx[0,2,0].tolist()==[1,0]
        assert s.positions[0,0].tolist()==[2,1]  # old position overwritten despite retention horizon
        backup=s.detached()
        a,st,di=m.forward_with_candidates(torch.zeros(1,6,10),s,disable=True)
        assert a.eq(0).all() and di.eq(-1).all() and st.positions.eq(-1).all()
        assert st.cursor.tolist()==[9]
        state_close(s,backup)

def test_training_gradients_and_indexed_parity():
    m=model()
    x=torch.randn(2,28,10)
    y,st,idx=m.forward_with_candidates(x)
    y.square().mean().backward()
    for n in ('publisher','subscriber','payload','project','gain','admit'):
        d=getattr(m,n).weight.grad
        assert d is not None and torch.isfinite(d).all() and d.abs().sum()>0,n
    with torch.no_grad():
        y2,s2,i2=m.eval().forward_with_candidates(x)
        close(y,y2);state_close(st,s2);torch.testing.assert_close(idx,i2)

def test_same_publisher_cache_content_blind_read():
    m=model().eval()
    x=torch.randn(2,21,10)
    with torch.no_grad():
        y,s,idx=m.forward_with_candidates(x)
        m.route_mode='content_blind'
        z,t,di=m.forward_with_candidates(x)
        state_close(s,t)
        assert not torch.equal(idx,di)
        assert (y-z).abs().sum()>0
        assert (((di<0)|(di<torch.arange(21)[None,:,None,None]))).all()

def test_fork_preserves_parent_and_selected_rows():
    m=model().eval()
    x=torch.randn(2,13,10)
    with torch.no_grad():
        _,s=m(x[:1,:6]);backup=s.detached()
        f=s.index_select(torch.tensor([0,0]))
        y,st=m(x[:,6:],f)
        state_close(s,backup)
        for i in range(2):
            zz,ss=m(x[i:i+1,6:],s)
            close(zz,y[i:i+1]);state_close(ss,st.index_select(torch.tensor([i])))

def test_candidate_trace_matches_independent_fifo_simulator():
    m=model(retained_blocks=4).eval()
    x=torch.randn(2,25,10)
    with torch.no_grad():
        _,state,actual=m.forward_with_candidates(x)
        read=m.subscriber(x).tolist()
        send=m.publisher(x).tolist()
        pub=m.admit(x).squeeze(-1).tolist()
    expected=[]
    final=[]
    for batch in range(len(x)):
        buckets=[[] for _ in range(4)]
        trace=[]
        for i in range(x.shape[1]):
            for j in range(4):
                buckets[j]=[p for p in buckets[j] if p//3>=i//3-3]
            chosen=sorted(range(4),key=lambda j:(-read[batch][i][j],j))[:2]
            trace.append([(buckets[j]+[-1,-1])[:2] for j in chosen])
            if pub[batch][i]>=0:
                j=max(range(4),key=lambda j:(send[batch][i][j],-j))
                buckets[j]=([i]+buckets[j])[:2]
        final.append([([p for p in b if p//3>=x.shape[1]//3-3]+[-1,-1])[:2] for b in buckets])
        expected.append(trace)
    assert actual.tolist()==expected
    assert state.positions.tolist()==final
