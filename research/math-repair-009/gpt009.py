"""GPT-only storage-layout compatibility adapter; native trainer is unmodified."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
import train009 as base
from data009 import T,sha

BASE_LOCK=base.source_lock
def loss(m,x,y):
    return m(x.contiguous(),y.contiguous())[1]
def logits(m,x):
    x=x.contiguous()
    return m(x,x)[0]
def lock():
    return dict(BASE_LOCK(), **{'gpt009.py':sha(Path(__file__))})
base.loss=loss
base.logits=logits
base.source_lock=lock

def smoke(a):
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    m=base.make('gpt',a.seed);pf=base.preflight(m);m.train()
    data=torch.randint(256,(4,T+1),generator=torch.Generator().manual_seed(a.seed+791))
    x,y=data[:,:-1],data[:,1:]
    assert not x.is_contiguous() and not y.is_contiguous() and y.shape==(4,T)
    failed=False
    try:
        y.view(-1)
    except RuntimeError:
        failed=True
    assert failed,'fixture must reproduce the original target-stride failure'
    def gradients(l):
        m.zero_grad(set_to_none=True);l.backward()
        return [p.grad.detach().clone() for p in m.parameters() if p.grad is not None]
    l=loss(m,x,y);g=gradients(l);l0=m(x.contiguous(),y.contiguous())[1];g0=gradients(l0)
    assert torch.isfinite(l) and torch.equal(l,l0) and len(g)==len(g0)
    assert all(torch.equal(a,b) for a,b in zip(g,g0))
    with torch.no_grad():
        z=logits(m,x);z0=m(x.contiguous(),x.contiguous())[0]
        assert torch.equal(z,z0)
    opt=torch.optim.AdamW(m.parameters(),lr=.0006,betas=(.9,.95));opt.step()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);p=out/'smoke_checkpoint.pt'
    torch.save(dict(weights=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),lock=lock()),p)
    cp=torch.load(p,weights_only=True,map_location='cpu');assert cp['lock']==lock()
    m2=base.make('gpt',a.seed);m2.load_state_dict(cp['weights']);m2.train();opt2=torch.optim.AdamW(m2.parameters(),lr=.0006,betas=(.9,.95));opt2.load_state_dict(cp['optimizer'])
    torch.set_rng_state(cp['rng'])
    for mm,oo in [(m,opt),(m2,opt2)]:
        oo.zero_grad(set_to_none=True);loss(mm,x,y).backward();oo.step()
    assert all(torch.equal(v,m2.state_dict()[k]) for k,v in m.state_dict().items())
    p.unlink()
    receipt=dict(seed=a.seed,upstream_revision=base.GPT_REV,parameters=pf['parameters'],causal_error=pf['causal_error'],production_microbatch=[4,T],noncontiguous_fixture=True,original_view_failure_reproduced=True,loss=float(l.detach()),loss_equal=True,gradient_equal=True,logits_equal=True,next_update_equal_after_safe_restore=True,source=lock(),classification='engineering preflight only; zero scientific training fits')
    (out/'gpt_smoke.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','validate','smoke']);p.add_argument('--arm',choices=['gpt'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',type=int);p.add_argument('--data');p.add_argument('--out',required=True);p.add_argument('--resume');p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    if a.mode=='smoke':smoke(a)
    elif a.mode=='train':base.train(a)
    else:base.final(a)
