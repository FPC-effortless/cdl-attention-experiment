import pytest
import torch
from .model import build_decoder, set_routing


@pytest.mark.parametrize("arm", ["A","B","C"])
@torch.inference_mode()
def test_chunk_and_suffix_parity(arm):
    x=torch.randint(100,(2,19),generator=torch.Generator().manual_seed(40))
    m=build_decoder(100,arm,width=32,layers=2).eval()
    full,_=m(x)
    a,st=m(x[:,:9])
    b,_=m(x[:,9:],st)
    alt=x.clone();alt[:,9:]=(alt[:,9:]+17)%100
    z,_=m(alt)
    rebuild,_=m(alt[:,9:],st)
    torch.testing.assert_close(torch.cat((a,b),1),full,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(z[:,:9],full[:,:9],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(z[:,9:],rebuild,atol=2e-6,rtol=2e-5)


def test_shared_initialization_and_disabled_route():
    with torch.inference_mode():
        a,b,c=(build_decoder(100,v,width=32,layers=2).eval() for v in "ABC")
        aa,bb,cc=(m.state_dict() for m in (a,b,c))
        assert all(torch.equal(v,bb[n]) and torch.equal(v,cc[n]) for n,v in aa.items())
        assert all(torch.equal(v,cc[n]) for n,v in bb.items())
        x=torch.randint(100,(2,17))
        set_routing(c,False)
        torch.testing.assert_close(b(x)[0],c(x)[0],atol=2e-6,rtol=2e-5)


def test_valid_gradients_in_ordinary_path():
    x=torch.randint(100,(3,17),generator=torch.Generator().manual_seed(22))
    m=build_decoder(100,"C",width=32,layers=2)
    logits,_=m(x)
    logits[:,10:].square().mean().backward()
    for base in ("local","summary","route"):
        g=[p.grad for n,p in m.named_parameters() if f".{base}." in n]
        assert len(g) and any(t is not None and t.isfinite().all() and t.abs().sum()>0 for t in g)
        assert all(t is None or t.isfinite().all() for t in g)


def test_local_channel_cannot_distinguish_matched_query_suffix():
    from .data import generate, ANSWERS, model_view
    from .objective import Tokenizer
    f=generate("train",4,82001,delays=(0,))
    t=Tokenizer.from_train(model_view(f),ANSWERS,minimum=1)
    x=torch.tensor([t.encode(w.prompt) for w in f[0].worlds])
    assert not torch.equal(x[0],x[1]) and torch.equal(x[0,-8:],x[1,-8:])
    with torch.inference_mode():
        y,_=build_decoder(t.size,"A")(x)
    torch.testing.assert_close(y[0,-1],y[1,-1],atol=0,rtol=0)


def test_late_padding_does_not_change_earlier_logits():
    x=torch.randint(100,(1,93),generator=torch.Generator().manual_seed(44))
    pad=torch.cat((x,torch.zeros((1,13),dtype=torch.long)),1)
    for arm in "ABC":
        m=build_decoder(100,arm).eval()
        with torch.inference_mode():
            y,_=m(x)
            z,_=m(pad)
        torch.testing.assert_close(y,z[:,:x.shape[1]],atol=2e-6,rtol=2e-5)
