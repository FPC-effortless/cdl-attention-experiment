"""Engineering-only AdamW continuation check, NOT scientific training fits."""
import json, tempfile, platform
from pathlib import Path
import torch, numpy as np
from torch.nn import functional as F
from contextual_cog import make,MODES
from model import count
ARMS=('HH','A0',*MODES)
SEEDS=(9001,9011,9029,9041)
def adam(m):
    return torch.optim.AdamW([{'params':[p for p in m.parameters() if p.ndim>=2],'weight_decay':.1},{'params':[p for p in m.parameters() if p.ndim<2],'weight_decay':0}],lr=.0006,betas=(.9,.95))
def step(m,opt,ids):
    m.train();opt.zero_grad(set_to_none=True)
    z=m(ids[:,:-1]);loss=F.cross_entropy(z.flatten(0,1),ids[:,1:].flatten());assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step();return float(loss.detach())
def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    rows=[]
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'checkpoint.pt'
        for arm in ARMS:
            for seed in SEEDS:
                gen=torch.Generator().manual_seed(seed+77);x,y=[torch.randint(0,256,(2,65),generator=gen) for _ in range(2)]
                m=make(arm,seed);o=adam(m);p=count(m);l1=step(m,o,x);rng=torch.get_rng_state().clone();torch.save(dict(weights=m.state_dict(),optimizer=o.state_dict(),rng=rng,step=1,curve=[dict(step=1,dev_bpb=float(np.float64(l1)))]),path)
                cp=torch.load(path,map_location='cpu',weights_only=True);assert cp['curve'][0]['dev_bpb']==l1 and type(cp['curve'][0]['dev_bpb']) is float
                l2=step(m,o,y);expected=m.state_dict();expected_opt=o.state_dict();m2=make(arm,seed);o2=adam(m2);m2.load_state_dict(cp['weights']);o2.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng']);l2r=step(m2,o2,y)
                assert l2==l2r and equal(expected,m2.state_dict()) and equal(expected_opt,o2.state_dict()),(arm,seed,l2,l2r)
                with torch.no_grad():
                    m2.eval();z,st=m2(x[:,:-1],carry=True);za,sa=m2(x[:,:19],carry=True);zb,sb=m2(x[:,19:-1],sa,True);err=float((z-torch.cat([za,zb],1)).abs().max());assert err<3e-5,(arm,seed,err)
                rows.append(dict(arm=arm,seed=seed,parameters=p,finite_gradient=True,exact_resume=True,exact_optimizer=True,chunk_maxerr=err));print(arm,seed,'PASS',flush=True)
    report=dict(status='ENGINEERING_ONLY',training_fits=0,fixture='random 2x64 next-byte targets; two optimizer steps and one replay per configuration',torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),rows=rows)
    Path(__file__).with_name('smoke_resume.json').write_text(json.dumps(report,indent=2)+'\n');print('ENGINEERING PASS',len(rows))
if __name__=='__main__':main()
