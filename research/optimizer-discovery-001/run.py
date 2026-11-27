"""Independent development screen. Uses immutable parent training corpus and model."""
import argparse,hashlib,importlib,inspect,json,math,os,platform,resource,sys,tempfile,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'math-repair-009'
ARCH=HERE.parent/'algorithm-discovery-001'
sys.path.insert(0,str(ARCH));sys.path.insert(1,str(PARENT))
from contextual_cog import make
from optimizers import make_bundle,RECIPES,RATES
import data009,recover009,train009
STEPS=1024
SEEDS=(9101,9109)
T=data009.T;B=data009.B
EXPECTED={'model.py':(ARCH,'31c3555521469425d884c700aed01b48c4fab50e'),'contextual_cog.py':(ARCH,'5001c137ffae2b96241904e5b4c061b91249b446'),'train009.py':(PARENT,'5d0d726338a3fd635c25a9fbbd7e7a1f4cec822d'),'data009.py':(PARENT,'08afe12ab312f04b23b4b841fb294d2a9895c560')}
VENDOR={'soap':('soap','a6362cc9ed2f50ed299204eb3bf07715e7280757'),'schedulefree':('adamw_schedulefree','e8f5a5d52f497577fa229b3c4341feca251bb6e9')}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def blob(p):
    b=Path(p).read_bytes();return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def sources(recipe):
    lock={}
    for n,(d,h) in EXPECTED.items():assert blob(d/n)==h,(n,'changed parent');lock[n]=sha(d/n)
    for n in ('optimizers.py','run.py','PROTOCOL.md','aggregate.py'):lock[n]=sha(HERE/n)
    if recipe in VENDOR:
        mod,h=VENDOR[recipe];p=Path(importlib.import_module(mod).__file__);assert blob(p)==h,(mod,'wrong vendor source');lock[mod]=sha(p)
    return lock
def eq(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(eq(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return a==b
def engineering(recipe,rate,seed):
    # SOAP must exercise the periodic basis refresh before checking reload.
    rows=[]
    for arm in ('HH','context_joint'):
        m=make(arm,seed);o=make_bundle(m,recipe,rate,STEPS);gen=torch.Generator().manual_seed(seed+77);n=12 if recipe=='soap' else 3
        xs=[torch.randint(0,256,(2,65),generator=gen) for _ in range(n+1)]
        def one(m,o,x,i):
            m.train();o.train();o.zero_grad();z=m(x[:,:-1]);l=F.cross_entropy(z.flatten(0,1),x[:,1:].flatten());l.backward();g=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(l) and torch.isfinite(g);o.step(i);return float(l.detach())
        for i,x in enumerate(xs[:-1]):one(m,o,x,i)
        o.eval();m.eval()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'c.pt';torch.save(dict(weights=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state()),p);c=torch.load(p,weights_only=True,map_location='cpu')
            l1=one(m,o,xs[-1],n);m2=make(arm,seed);o2=make_bundle(m2,recipe,rate,STEPS);m2.load_state_dict(c['weights']);o2.load_state_dict(c['optimizer']);torch.set_rng_state(c['rng']);l2=one(m2,o2,xs[-1],n)
            assert l1==l2 and eq(m.state_dict(),m2.state_dict()) and eq(o.state_dict(),o2.state_dict()),(recipe,arm,'restart')
        assert all(torch.isfinite(p).all() for p in m.parameters());rows.append(dict(arm=arm,exact_resume=True,optimizer_state_bytes=o.state_bytes(),steps=n+1,scientific_fit=False))
    return rows
def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',choices=RECIPES,required=True);p.add_argument('--rate-index',type=int,choices=range(3),required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True,exist_ok=False);rate=RATES[a.recipe][a.rate_index];lock=sources(a.recipe);manifest=recover009.corpus(a.data)
    starts=data009.order(a.data,a.seed)[:STEPS];order_hash=hashlib.sha256(starts.tobytes()).hexdigest()
    rec=dict(architecture='HH',recipe=a.recipe,rate=rate,rate_index=a.rate_index,seed=a.seed,steps=STEPS,context=T,batch=B,source=lock,data_manifest=sha(a.data/'manifest.json'),data_order=order_hash,torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),threads=a.threads,commit=os.environ.get('GITHUB_SHA'),run_id=os.environ.get('GITHUB_RUN_ID'))
    (a.out/'config.json').write_text(json.dumps(rec,indent=2))
    checks=engineering(a.recipe,rate,a.seed);(a.out/'engineering.json').write_text(json.dumps(checks,indent=2));print('ENGINEERING_PASS',a.recipe,flush=True)
    m=make('HH',a.seed);o=make_bundle(m,a.recipe,rate,STEPS);assert sum(p.numel() for p in m.parameters())==1077504
    raw=np.memmap(a.data/'train.bin',dtype=np.uint8,mode='r');idx=np.arange(T+1);curve=[];train_sec=0.;opt_sec=0.;wall0=time.monotonic();clips=0
    def save(step,terminal=False):
        o.eval();m.eval();p=a.out/'checkpoint.pt';tmp=a.out/'checkpoint.tmp';torch.save(dict(weights=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state(),step=step,config=rec,curve=curve,train_seconds=train_sec,optimizer_seconds=opt_sec),tmp);os.replace(tmp,p)
        (a.out/'receipt.json').write_text(json.dumps(dict(step=step,targets=step*B*T,terminal=terminal,checkpoint_sha256=sha(p),config=rec,partition=o.partition),indent=2))
    for i in range(STEPS):
        m.train();o.train();t=time.monotonic();x=torch.from_numpy(raw[starts[i,:,None]+idx].astype(np.int64));o.zero_grad();ls=0.
        for xx in x.split(4):
            z=m(xx[:,:-1]);l=F.cross_entropy(z.flatten(0,1),xx[:,1:].flatten())/4;assert torch.isfinite(l);l.backward();ls+=float(l.detach())
        g=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(g);clips+=int(g>1.);s=time.monotonic();o.step(i);opt_sec+=time.monotonic()-s;train_sec+=time.monotonic()-t
        if (i+1)%64==0:
            item=dict(step=i+1,targets=(i+1)*B*T,train_loss=ls,gradient_norm=float(g),train_seconds=train_sec,optimizer_seconds=opt_sec);curve.append(item);print(json.dumps(item),flush=True)
        if (i+1)%256==0:save(i+1)
        if time.monotonic()-wall0>5.2*3600 and i+1<STEPS:
            save(i+1);raise RuntimeError('partial screen retained; not a terminal fit')
    o.eval();m.eval();e=train009.evaluate(m,a.data,'dev');e.update(bpb=float(e['bpb']),accuracy=float(e['accuracy']),targets=int(e['targets']))
    assert math.isfinite(e['bpb']) and all(torch.isfinite(p).all() for p in m.parameters())
    (a.out/'development.json').write_text(json.dumps(e));save(STEPS,True)
    w={k:v.detach().cpu().contiguous().clone() for k,v in m.state_dict().items()};p=a.out/'model.safetensors';save_file(w,str(p));rr=load_file(str(p));assert all(torch.equal(v,rr[k]) for k,v in w.items())
    ck=torch.load(a.out/'checkpoint.pt',weights_only=True,map_location='cpu');m2=make('HH',a.seed);m2.load_state_dict(ck['weights']);o2=make_bundle(m2,a.recipe,rate,STEPS);o2.load_state_dict(ck['optimizer']);assert eq(o.state_dict(),o2.state_dict()) and eq(w,m2.state_dict())
    report=dict(config=rec,parameters=1077504,terminal=True,targets=STEPS*B*T,bpb=e['bpb'],eval_targets=e['targets'],train_seconds=train_sec,optimizer_seconds=opt_sec,optimizer_state_bytes=o.state_bytes(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,clipped_steps=clips,checkpoint_sha256=sha(a.out/'checkpoint.pt'),model_sha256=sha(p),partition=o.partition,reload_verified=True)
    (a.out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
