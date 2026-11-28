"""Engineering-only stress at peak AdamW LR from fresh HH weights; not a fit."""
import argparse,hashlib,json,math,tempfile
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from contextual_cog import make
RATES=(.0012,.002,.003)
SEEDS=(9101,9109)
CORPUS='2955e5b6e772cf4539615b0585c1b522f7cf168c3c8e47c95b29dc97dda40c35'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify(root):
    assert sha(root/'manifest.json')==CORPUS
    m=json.loads((root/'manifest.json').read_text());assert m['split']=='train'
    assert all(sha(root/k)==v for k,v in m['files'].items())

def same(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def opt(m,rate):
    return torch.optim.AdamW([{'params':[p for p in m.parameters() if p.ndim>=2],'weight_decay':.1},{'params':[p for p in m.parameters() if p.ndim<2],'weight_decay':0}],lr=rate,betas=(.9,.95))

def step(m,o,arr):
    m.train();o.zero_grad(set_to_none=True);loss=0.
    for x in arr.split(4):
        z=m(x[:,:-1]);l=F.cross_entropy(z.flatten(0,1),x[:,1:].flatten())/4;assert torch.isfinite(l);l.backward();loss+=float(l.detach())
    g=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(g)
    with torch.no_grad():old=[p.detach().clone() for p in m.parameters()]
    o.step()
    with torch.no_grad():
        assert all(torch.isfinite(p).all() for p in m.parameters())
        den=sum(float(x.double().square().sum()) for x in old);num=sum(float((p-x).double().square().sum()) for p,x in zip(m.parameters(),old))
    return dict(loss=loss,unclipped_gradient_norm=float(g),relative_parameter_update_norm=math.sqrt(num/den))

def check(root,rate,seed):
    starts=np.load(root/'train_starts.npy',allow_pickle=False);order=np.random.Generator(np.random.PCG64(seed+900090)).permutation(starts)[:32].reshape(2,16)
    raw=np.memmap(root/'train.bin',dtype=np.uint8,mode='r');idx=np.arange(769)
    xs=[torch.from_numpy(raw[row[:,None]+idx].astype(np.int64)) for row in order]
    m=make('HH',seed);assert sum(p.numel() for p in m.parameters())==1077504;o=opt(m,rate)
    first=step(m,o,xs[0])
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'checkpoint.pt';torch.save(dict(weights=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state()),p);c=torch.load(p,weights_only=True,map_location='cpu')
        second=step(m,o,xs[1]);m2=make('HH',seed);o2=opt(m2,rate);m2.load_state_dict(c['weights']);o2.load_state_dict(c['optimizer']);torch.set_rng_state(c['rng']);replay=step(m2,o2,xs[1]);assert second==replay and same(m.state_dict(),m2.state_dict()) and same(o.state_dict(),o2.state_dict())
    return dict(seed=seed,rate=rate,first=first,second=second,exact_next_update=True,scientific_fit=False,targets_per_update=12288,stress='cold start at peak LR, deliberately bypasses schedule; ordinary fit retains original warmup')

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--rate',type=float,choices=RATES);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);verify(a.data)
    import model,contextual_cog
    assert sha(model.__file__)=='eaf5ae7973957bc68330fe754a8aa6cdb1199189413fee6995aa749f809e75c8'
    assert sha(contextual_cog.__file__)=='fcb483fc6b8440a05c952898fa57161c634183587a988947155afc9147da18d3'
    rows=[]
    for r in (RATES if a.rate is None else (a.rate,)):
        for s in (SEEDS if a.seed is None else (a.seed,)):
            z=check(a.data,r,s);rows.append(z);print(json.dumps(z),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='ENGINEERING_ONLY',scientific_fits=0,torch=str(torch.__version__),numpy=np.__version__,corpus=CORPUS,source_sha256=sha(__file__),rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
