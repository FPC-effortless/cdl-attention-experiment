"""Prospective Stage A: matched value-distance and operator-composition diagnostics."""
import argparse, hashlib, itertools, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'unified-search-002'
sys.path.insert(0,str(BASE))
from integrated import WholeCOG, VOCAB
from model import D, count
from knowledge import START,MASK,QUERY,VAL,OP,END,EMPTY
import train as original
ARMS=('HH','SS','GQA','GRU','GATED_DELTA_REF')
REGIMES=('near','far')
SEEDS=(9501,9511,9529)
STEPS=384
BATCH=35
TARGET=1077488
TRAIN_PAIRS=tuple(p for p in itertools.product(range(3),repeat=2) if p not in ((1,2),(2,1)))
HELD_PAIRS=((1,2),(2,1))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrsha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def source():
    return dict(original=original.source(),diagnostic={p:sha(HERE/p) for p in ('competence.py','PROTOCOL.md')})
class GRUSeq(nn.Module):
    def __init__(self):
        super().__init__();self.rnn=nn.GRU(D,D,batch_first=True);self.out=nn.Linear(D,D)
    def forward(self,x,state=None):
        z,s=self.rnn(x,state);return self.out(z),s
class DeltaSeq(nn.Module):
    """Same reference recurrence as DISCOVERY-010; not optimized Gated DeltaNet."""
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(D,2*64+D);self.gates=nn.Linear(D,8);self.out=nn.Linear(D,D)
    def forward(self,x,state=None):
        b,t,d=x.shape
        q,k,v=self.qkv(x).split((64,64,128),-1)
        q=F.normalize(q.reshape(b,t,4,16).transpose(1,2),dim=-1)
        k=F.normalize(k.reshape(b,t,4,16).transpose(1,2),dim=-1)
        v=v.reshape(b,t,4,32).transpose(1,2)
        a,g=self.gates(x).transpose(1,2).chunk(2,1);a=.95+.05*a.sigmoid();g=g.sigmoid()
        s=x.new_zeros(b,4,16,32) if state is None else state;out=[]
        for Q,K,V,A,G in zip(q.split(16,2),k.split(16,2),v.split(16,2),a.split(16,2),g.split(16,2)):
            p=A.cumprod(-1);decay=p[..., :,None]/p[...,None,:];H=(K@K.transpose(-1,-2))*decay
            L=(G[...,None]*H).tril(-1);rhs=G[...,None]*(V-p[...,None]*(K@s))
            u=torch.linalg.solve_triangular(L,rhs,upper=False,unitriangular=True)
            z=p[...,None]*(Q@s)+((Q@K.transpose(-1,-2))*decay).tril()@u
            s=p[...,-1,None,None]*s+K.transpose(-1,-2)@(u*(p[...,-1,None]/p)[...,None]);out.append(z)
        return self.out(torch.cat(out,2).transpose(1,2).reshape(b,t,D)),s
class AltLayer(nn.Module):
    def __init__(self,old,seq):
        super().__init__();self.norm=old.norm;self.norm2=old.norm2;self.seq=seq
        diff=count(seq)-count(old.qkv)-count(old.out)
        width=old.ff[0].out_features-round(diff/(2*D+1))
        assert width>0
        self.ff=nn.Sequential(nn.Linear(D,width),nn.GELU(),nn.Linear(width,D))
        with torch.no_grad():
            self.ff[0].weight.copy_(old.ff[0].weight[:width]);self.ff[0].bias.copy_(old.ff[0].bias[:width])
            self.ff[2].weight.copy_(old.ff[2].weight[:,:width]);self.ff[2].bias.copy_(old.ff[2].bias)
    def forward(self,h,state=None,side=None):
        z,s=self.seq(self.norm(h),state)
        if side is not None:z=z+side
        h=h+z;return h+self.ff(self.norm2(h)),s
def build(arm,seed):
    if arm in ('HH','SS','GQA'):
        m=WholeCOG({'HH':'NN_native_native','SS':'SS_native_native','GQA':'GQA'}[arm],seed)
    else:
        m=WholeCOG('GQA',seed)
        for i,old in enumerate(m.parent.core.blocks):
            torch.manual_seed(seed+60013+i)
            m.parent.core.blocks[i]=AltLayer(old,GRUSeq() if arm=='GRU' else DeltaSeq())
    assert abs(count(m)-TARGET)/TARGET<.002,(arm,count(m))
    return m
def tuples(pairs):
    return np.asarray([(v,a,b) for a,b in pairs for v in range(5)],dtype=np.int64)
def encode(logical,regime,length=16):
    assert length>=16 and regime in REGIMES
    x=np.full((len(logical),length),MASK,dtype=np.int64)
    x[:,0]=START;x[:,1]=QUERY
    pos=length-4 if regime=='near' else 2
    x[:,pos]=logical[:,0]+VAL;x[:,-3:-1]=logical[:,1:]+OP;x[:,-1]=END
    v=logical[:,0];ops=logical[:,1:]
    y=np.where(v==EMPTY,EMPTY,(v+(ops==1).sum(1)-(ops==2).sum(1))%4)+VAL
    return x,y.astype(np.int64)
def interpreter(row):
    assert row[0]==START and row[1]==QUERY and row[-1]==END
    values=[int(t)-VAL for t in row[2:-3] if VAL<=t<=VAL+EMPTY]
    assert len(values)==1
    v=values[0]
    for token in row[-3:-1]:
        assert token in (OP,OP+1,OP+2)
        if v!=EMPTY:v=(v+{OP:0,OP+1:1,OP+2:-1}[int(token)])%4
    return v+VAL
def assess(pred,y,logical):
    ok=pred==y;nonempty=logical[:,0]!=EMPTY;empty=~nonempty
    pair=[(i,j) for i in range(len(y)) for j in range(i+1,len(y)) if np.array_equal(logical[i,1:],logical[j,1:]) and y[i]!=y[j]]
    return dict(n=len(y),correct=int(ok.sum()),nonempty_n=int(nonempty.sum()),nonempty_correct=int((ok&nonempty).sum()),empty_n=int(empty.sum()),empty_correct=int((ok&empty).sum()),pairs=len(pair),both_correct=int(sum(bool(ok[i] and ok[j]) for i,j in pair)),changed=int(sum(pred[i]!=pred[j] for i,j in pair)))
def panels():
    for length in (16,64):
        for regime in REGIMES:
            for label,pairs in [('familiar',TRAIN_PAIRS),('held',HELD_PAIRS)]:
                q=tuples(pairs);x,y=encode(q,regime,length)
                name=f'{regime}_{length}_{label}'
                yield name,q,x,y
def rule_audit():
    rows={}
    for name,q,x,y in panels():
        rule=np.asarray([interpreter(z) for z in x]);r=assess(rule,y,q)
        assert r['correct']==r['n'] and r['both_correct']==r['pairs']
        rows[name]={'rule':r,'constant_empty':assess(np.full(len(y),VAL+EMPTY),y,q),'constant_value':assess(np.full(len(y),VAL),y,q)}
    return rows
def data(seed,regime):
    rng=np.random.default_rng(seed+7041);q=tuples(TRAIN_PAIRS)
    order=np.stack([rng.permutation(len(q)) for _ in range(STEPS)])
    logical=q[order];x,y=encode(logical.reshape(-1,3),regime)
    return x.reshape(STEPS,BATCH,16),y.reshape(STEPS,BATCH),arrsha(logical)
def preflight(m,regime):
    m.eval();x=torch.from_numpy(encode(tuples(TRAIN_PAIRS)[:2],regime)[0])
    with torch.no_grad():
        z,s=m(x,carry=True);a,st=m(x[:,:8],carry=True);b,st=m(x[:,8:],st,True)
        err=float((z-torch.cat((a,b),1)).abs().max())
        future=x.clone();future[:,8:]=MASK
        fe=float((z[:,:8]-m(future)[:,:8]).abs().max())
    assert max(err,fe)<3e-5,(err,fe)
    return {'streaming_error':err,'future_error':fe}
def opt_for(m):
    return torch.optim.AdamW(m.parameters(),lr=.0015,betas=(.9,.95),weight_decay=.01)
def step(m,opt,x,y,i):
    m.train();opt.zero_grad(set_to_none=True)
    lr=.0015*((i+1)/20 if i<20 else .1+.45*(1+math.cos(math.pi*(i-20)/(STEPS-21))))
    opt.param_groups[0]['lr']=lr
    loss=F.cross_entropy(m(torch.from_numpy(x))[:,-1],torch.from_numpy(y))
    assert torch.isfinite(loss);loss.backward()
    gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn)
    opt.step();return float(loss.detach())
def write_json(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def preflight_all(out):
    out.mkdir(parents=True,exist_ok=False);source();r=rule_audit();write_json(out/'rule_audit.json',r)
    rec=[]
    for arm in ARMS:
        for regime in REGIMES:
            m=build(arm,SEEDS[0]);f=preflight(m,regime);o=opt_for(m);x,y,order=data(SEEDS[0],regime)
            step(m,o,x[0],y[0],0);tmp=out/'restart.pt'
            torch.save(dict(weights=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state()),tmp)
            ref=step(m,o,x[1],y[1],1)
            n=build(arm,SEEDS[0]);oo=opt_for(n);ck=torch.load(tmp,weights_only=True)
            n.load_state_dict(ck['weights']);oo.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng'])
            got=step(n,oo,x[1],y[1],1)
            assert ref==got and original.same(m.state_dict(),n.state_dict()) and original.same(o.state_dict(),oo.state_dict()),(arm,regime)
            rec.append(dict(arm=arm,regime=regime,parameters=count(m),**f,restart_exact=True));tmp.unlink()
            print(rec[-1],flush=True)
    write_json(out/'preflight.json',rec)
@torch.no_grad()
def evaluate(m):
    m.eval();results={}
    for name,q,x,y in panels():
        pred=torch.cat([m(torch.from_numpy(x[i:i+16]))[:,-1].argmax(-1) for i in range(0,len(x),16)]).numpy()
        results[name]=dict(metrics=assess(pred,y,q),predictions=pred.tolist(),targets=y.tolist(),inputs_sha256=arrsha(x))
    return results
def eligible(row):
    return row['correct']/row['n']>=.95 and row['nonempty_correct']/row['nonempty_n']>=.95 and row['both_correct']/row['pairs']>=.95
def fit(arm,regime,seed,out):
    out.mkdir(parents=True,exist_ok=False);lock=source();m=build(arm,seed);f=preflight(m,regime);o=opt_for(m)
    x,y,order=data(seed,regime)
    c=dict(arm=arm,regime=regime,seed=seed,parameters=count(m),steps=STEPS,examples=STEPS*BATCH,input_symbols=STEPS*BATCH*16,unique_tuples=35,source=lock,logical_order_sha256=order,input_sha256=arrsha(x),label_sha256=arrsha(y),torch=str(torch.__version__),numpy=np.__version__,threads=torch.get_num_threads(),run_id=os.environ.get('GITHUB_RUN_ID'),commit=os.environ.get('GITHUB_SHA'))
    write_json(out/'config.json',c);losses=[];t=time.monotonic()
    for i in range(STEPS):
        losses.append(step(m,o,x[i],y[i],i))
        if (i+1)%64==0:print(json.dumps(dict(step=i+1,last64_loss=sum(losses[-64:])/64)),flush=True)
        if (i+1)%128==0:
            tmp=out/'checkpoint.tmp';p=out/'checkpoint.pt'
            torch.save(dict(weights=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state(),step=i+1,config=c,losses=losses),tmp);os.replace(tmp,p)
            write_json(out/'receipt.json',dict(step=i+1,terminal=(i+1==STEPS),sha256=sha(p),config=c))
    w={k:v.detach().contiguous().clone() for k,v in m.state_dict().items()};save_file(w,str(out/'model.safetensors'))
    n=build(arm,seed);n.load_state_dict(load_file(str(out/'model.safetensors')),strict=True)
    ck=torch.load(out/'checkpoint.pt',weights_only=True);assert ck['step']==STEPS and original.same(n.state_dict(),ck['weights']) and original.same(w,n.state_dict())
    oo=opt_for(n);oo.load_state_dict(ck['optimizer']);assert original.same(o.state_dict(),oo.state_dict())
    result=evaluate(n);post=preflight(n,regime);write_json(out/'metrics.json',result)
    r=dict(status='TERMINAL_DIAGNOSTIC',terminal=True,reload_verified=True,config=c,preflight=f,postflight=post,checkpoint_sha256=sha(out/'checkpoint.pt'),weights_sha256=sha(out/'model.safetensors'),metrics_sha256=sha(out/'metrics.json'),seconds=time.monotonic()-t,eligible=eligible(result[f'{regime}_16_familiar']['metrics']),panels={k:v['metrics'] for k,v in result.items()})
    write_json(out/'result.json',r);print(json.dumps(r),flush=True)
def aggregate(root,out):
    lock=source();rows={}
    for p in root.rglob('result.json'):
        r=json.loads(p.read_text());c=r['config'];k=(c['arm'],c['regime'],c['seed']);assert k not in rows
        assert r['status']=='TERMINAL_DIAGNOSTIC' and r['terminal'] and r['reload_verified'] and c['source']==lock
        assert c['steps']==STEPS and c['examples']==STEPS*BATCH and c['unique_tuples']==35 and abs(c['parameters']-TARGET)/TARGET<.002
        for fn,field in [('checkpoint.pt','checkpoint_sha256'),('model.safetensors','weights_sha256'),('metrics.json','metrics_sha256')]:assert sha(p.parent/fn)==r[field]
        q=json.loads((p.parent/'receipt.json').read_text());assert q['terminal'] and q['step']==STEPS and q['config']==c and q['sha256']==r['checkpoint_sha256']
        x,y,order=data(c['seed'],c['regime']);assert (arrsha(x),arrsha(y),order)==(c['input_sha256'],c['label_sha256'],c['logical_order_sha256'])
        metrics=json.loads((p.parent/'metrics.json').read_text())
        for name,logical,xx,yy in panels():
            z=metrics[name];assert z['inputs_sha256']==arrsha(xx) and z['targets']==yy.tolist()
            assert assess(np.asarray(z['predictions']),yy,logical)==z['metrics']==r['panels'][name]
        assert r['eligible']==eligible(r['panels'][f"{c['regime']}_16_familiar"])
        rows[k]=r
    assert set(rows)==set(itertools.product(ARMS,REGIMES,SEEDS)),('incomplete',len(rows))
    for seed in SEEDS:assert len({r['config']['logical_order_sha256'] for k,r in rows.items() if k[2]==seed})==1
    summary=[]
    for arm,regime in itertools.product(ARMS,REGIMES):
        a=[rows[(arm,regime,s)] for s in SEEDS]
        summary.append(dict(arm=arm,regime=regime,passed_seeds=sum(r['eligible'] for r in a),all_seed_gate=all(r['eligible'] for r in a),parameters=a[0]['config']['parameters'],panels={name:{metric:sum(r['panels'][name][metric] for r in a) for metric in a[0]['panels'][name]} for name in a[0]['panels']}))
    out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='DEVELOPMENT_DIAGNOSTIC_ONLY',terminal_fits=len(rows),rules=rule_audit(),groups=summary,receipts=[rows[k] for k in sorted(rows)],warning='No architecture promotion or language/scale claim. Finite operator table; Stage A only.')
    write_json(out,result);print(json.dumps(dict(terminal_fits=len(rows),groups=summary),indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('preflight','train','aggregate'));p.add_argument('--arm',choices=ARMS);p.add_argument('--regime',choices=REGIMES);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--out',required=True,type=Path);p.add_argument('--input',type=Path);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if a.mode=='preflight':preflight_all(a.out)
    elif a.mode=='train':fit(a.arm,a.regime,a.seed,a.out)
    else:aggregate(a.input,a.out)
