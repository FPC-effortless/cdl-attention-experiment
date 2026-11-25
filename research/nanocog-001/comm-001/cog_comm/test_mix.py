
import pytest
import torch

from .mix import LocalMix, LocalState, SummaryMix, SummaryState


def split_run(module,x,cuts):
    s=None;y=[]
    for a,b in zip(cuts,cuts[1:]):
        z,s=module(x[:,a:b],s)
        y.append(z)
    return torch.cat(y,1),s


def test_local_simple_shifted_tap_oracle_and_dynamic_gate():
    torch.manual_seed(19)
    m=LocalMix(6,window=3).double()
    with torch.no_grad():
        m.depthwise.copy_(torch.randn_like(m.depthwise))
    x=torch.randn(2,15,6,dtype=torch.float64)
    actual,state=m(x)
    expected=[]
    with torch.no_grad():
        for i in range(x.size(1)):
            recent=x[:,max(0,i-m.window+1):i+1]
            n=recent.size(1)
            gate=m.offset_gate(x[:,i])[:,-n:].softmax(1)
            mix=(recent*m.depthwise[:,-n:].T[None]*gate[:,:,None]).sum(1)
            expected.append(m.pointwise(mix))
    torch.testing.assert_close(actual,torch.stack(expected,1),rtol=1e-12,atol=1e-12)
    assert torch.equal(state.history,x[:,-2:])
    assert state.history.untyped_storage().nbytes()==state.history.numel()*state.history.element_size()
    # Changing ignored older events, not the query or its live taps, is silent.
    older=x.clone();older[:, :7]+=51
    changed,_=m(older)
    assert torch.equal(changed[:,9:],actual[:,9:])


def test_summary_block_exclusive_reference_and_first_packet():
    torch.manual_seed(201)
    m=SummaryMix(7,block=3).double()
    x=torch.randn(2,19,7,dtype=torch.float64)
    actual,state=m(x)
    expected=[]
    for i in range(x.size(1)):
        past=x[:,:3*(i//3)]
        if not past.shape[1]:
            expected.append(torch.zeros_like(x[:,i]));continue
        w=m.weight(past).sigmoid()
        mean=(w*m.value(past)).sum(1)/w.sum(1)
        expected.append(m.out(mean*m.query(x[:,i]).sigmoid()))
    torch.testing.assert_close(actual,torch.stack(expected,dim=1),rtol=1e-12,atol=1e-12)
    assert torch.equal(actual[:,:3],torch.zeros_like(actual[:,:3]))
    assert state.phase==1 and state.complete.shape==state.pending.shape==(2,2,7)
    # Same block's earlier events remain excluded from this query.
    modified=x.clone()
    modified[:,9:11]+=21
    mod,_=m(modified)
    assert torch.equal(mod[:,11],actual[:,11])
    assert not torch.allclose(mod[:,12],actual[:,12])


@pytest.mark.parametrize('module',[LocalMix(9),SummaryMix(9,block=8),LocalMix(9,window=1),SummaryMix(9,block=1)])
def test_chunk_reset_and_future_prefix(module):
    torch.manual_seed(78939)
    x=torch.randn(3,32,9)
    expected,st=module(x)
    for cuts in [(0,1,3,7,8,9,15,24,31,32),tuple(range(33)),(0,0,14,14,19,32)]:
        got,cache=split_run(module,x,cuts)
        torch.testing.assert_close(got,expected,atol=2e-6,rtol=2e-6)
        if isinstance(st,SummaryState):
            assert cache.phase==st.phase
            for q,r in zip((st.complete,st.pending),(cache.complete,cache.pending)):
                torch.testing.assert_close(q,r,atol=3e-6,rtol=2e-6)
        else:assert torch.equal(cache.history,st.history)
    y,s=module(x[:,:10])
    changed=x.clone();changed[:,10:]*=-10
    torch.testing.assert_close(module(changed)[0][:,:10],y,atol=2e-6,rtol=2e-6)
    empty,same=module(x[:,:0],s)
    assert same is s and empty.shape==(3,0,9)
    r,_=module(x[:,10:],None)
    standalone,_=split_run(module,x[:,10:],(0,3,11,22))
    torch.testing.assert_close(standalone,r,atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('module',[LocalMix(5),SummaryMix(5)])
def test_row_fork_and_bad_state(module):
    torch.manual_seed(12)
    x=torch.randn(3,12,5)
    _,start=module(x[:,:7])
    act,_=module(x[:,7:],start)
    ids=torch.tensor([2,0,2,1]);fork=start.index_select(ids).detach()
    y,_=module(x[ids,7:],fork)
    torch.testing.assert_close(y,act[ids],rtol=2e-6,atol=2e-6)
    with pytest.raises(ValueError,match='state'):
        module(x[:1,7:],start)
    with pytest.raises(ValueError,match='width'):
        module(x[:,:,:4])


def test_summary_state_midblock_and_fixed_size():
    m=SummaryMix(7,block=4)
    x=torch.randn(2,13,7)
    _,partial=m(x[:,:3])
    assert partial.phase==3
    assert not partial.complete.any() and partial.pending[:,1,:].gt(0).all()
    y,st=m(x[:,3:4],partial)
    assert st.phase==0 and not st.pending.any() and not y.any()
    _,larger=m(x,st)
    for f in (st.complete,st.pending,larger.complete,larger.pending):
        assert f.untyped_storage().nbytes()==4*7*4
