"""Prospective complete COG training-rule crossover; all parent sources unchanged."""
import argparse,ast,csv,hashlib,itertools,json,math,os,platform,resource,sys,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'unified-search-002';BAL=HERE.parent/'unified-balanced-004'
sys.path.insert(0,str(BASE));sys.path.insert(1,str(BAL))
import train as t
from integrated import WholeCOG,count,tensor_bytes
from knowledge import START,MASK,QUERY,KEY,VAL,OP,WRITE,HOLD,RESET,TARGET,DIST,END,EMPTY,split,maps,make8,draw4,digest_split,eval4
from balanced import generate,counterfactuals
from reference import DeltaSeq,AltLayer
ARMS=('NN_native_native','SS_native_native','SS_presence_factor','BB_presence_factor','GQA','GATED_DELTA_REF')
REGIMES=('final','trace');SEEDS=(9601,9611,9629)
STEPS=1024;BATCH=16;T=768;KB=8
TRAIN_PAIRS=np.array([(a,b) for a,b in itertools.product(range(3),repeat=2) if (a,b) not in ((1,2),(2,1))],dtype=np.int64)
HELD_PAIRS=np.array([(1,2),(2,1)],dtype=np.int64)
LOCK={'space.py':'ea6de750ae5c513c1debe74f2dfd6d134247561b','balanced.py':'9027d686ce09ce125aa1f970a4bbbff8f42aff52'}
def sha(p):return t.sha(p)
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def source():
    assert all(t.blob(BAL/k)==v for k,v in LOCK.items())
    p=HERE.parent/'competence-distance-011'/'competence.py'
    if p.exists():
        assert t.blob(p)=='7ccd1fc0d0e2aedbd42b4db92dd11b605a65ed41'
        q=ast.parse(p.read_text());r=ast.parse((HERE/'reference.py').read_text())
        # Compare syntax trees, ignoring whitespace and source locations.
        a={x.name:ast.dump(x) for x in q.body if isinstance(x,ast.ClassDef)};b={x.name:ast.dump(x) for x in r.body if isinstance(x,ast.ClassDef)}
        assert all(a[n]==b[n] for n in ('DeltaSeq','AltLayer'))
    return dict(parent=t.source(),balanced={k:sha(BAL/k) for k in LOCK},stage_a_reference_blob='7ccd1fc0d0e2aedbd42b4db92dd11b605a65ed41',new={k:sha(HERE/k) for k in ('trace.py','reference.py','PROTOCOL.md')})
def build(arm,seed):
    m=WholeCOG('GQA' if arm=='GATED_DELTA_REF' else arm,seed)
    if arm=='GATED_DELTA_REF':
        for i,old in enumerate(m.parent.core.blocks):
            torch.manual_seed(seed+60013+i);m.parent.core.blocks[i]=AltLayer(old,DeltaSeq())
    assert abs(count(m)-1077488)/1077488<.002,(arm,count(m));return m

def primitive(v,o):
    assert o in (OP,OP+1,OP+2)
    return v if v==EMPTY else (v+{OP:0,OP+1:1,OP+2:-1}[o])%4

def trace4(x):
    r=[]
    for row in x.tolist():
        assert len(row)==13 and row[0]==START and row[9]==QUERY
        k=row[10];v=row[11]-VAL if VAL<=row[11]<VAL+4 else None
        if v is None:
            d={row[j]:row[j+1]-VAL for j in range(1,9,2) if VAL<=row[j+1]<VAL+4};v=d[k]
        r.append([v+VAL,primitive(v,row[12])+VAL])
    return torch.tensor(r,dtype=torch.long)

def trace8(x):
    r=[]
    for row in x.tolist():
        assert row[0]==START and row[-1]==END and (len(row)-9)%3==0
        d={};query=row[-8:]
        for j in range(1,len(row)-8,3):
            a,k,v=row[j:j+3]
            if a==MASK:assert k==MASK and v==MASK;continue
            assert KEY<=k<KEY+8
            if a==WRITE:assert VAL<=v<VAL+4;d[k]=v-VAL
            elif a==RESET:assert v==MASK;d[k]=EMPTY
            elif a==HOLD:assert v==MASK
            else:raise ValueError(a)
        key=query[query.index(TARGET)+1];assert key!=query[query.index(DIST)+1]
        v=d[key];u=primitive(v,query[-3]);w=primitive(u,query[-2]);r.append([v+VAL,u+VAL,w+VAL])
    return torch.tensor(r,dtype=torch.long)

def eight(rng,allowed,n,held=False,novel=False,oracle=False):
    _,_,m=generate(rng,allowed,n,held)
    pairs=HELD_PAIRS if novel else TRAIN_PAIRS;m['op']=pairs[rng.integers(len(pairs),size=n)]
    x,y,_=make8(m['ids'],m['q'],m['op'],m['order'],m['other'],m['syntax'],change=m['change'])
    if oracle:
        a,b,_=make8(m['ids'],m['q'],m['op'],m['order'],m['other'],m['syntax'],mode='oracle',change=m['change']);assert torch.equal(y,b);x[-n//4:]=a[-n//4:]
    z=trace8(x);assert torch.equal(z[:,-1],y)
    return x,z,m

def batch(raw,starts,i,g4,g8,s4,s8):
    x=torch.from_numpy(raw[starts[i,:,None]+np.arange(T+1)].astype(np.int64))
    a,y=draw4(g4,s4,KB,False,True);a4=trace4(a);assert torch.equal(a4[:,-1],y)
    b,b8,_=eight(g8,s8,KB,oracle=True);return x,a,a4,b,b8

def inputs(root,seed):
    raw=np.memmap(root/'train.bin',dtype=np.uint8,mode='r');starts=t.order(root,seed);g4=np.random.default_rng(seed+21201);g8=np.random.default_rng(seed+21203)
    return raw,starts,g4,g8,split(seed,4)[0],split(seed,8)[0]

def cogloss(m,a,z,b,w,regime):
    p=m(a);q=m(b);f4=F.cross_entropy(p[:,-1],z[:,-1]);f8=F.cross_entropy(q[:,-1],w[:,-1]);r4=F.cross_entropy(p[:,-2],z[:,0]);r8=F.cross_entropy(q[:,-4],w[:,0]);i8=F.cross_entropy(q[:,-3],w[:,1]);l4=f4 if regime=='final' else .5*(f4+r4);l8=f8 if regime=='final' else .5*f8+.25*(r8+i8)
    return .25*(l4+l8),[f4,f8,r4,r8,i8]

def update(m,o,d,i,regime):
    m.train();o.zero_grad(set_to_none=True);x,a,z,b,w=d;lt=0.
    for xx in x.split(4):
        p=m(xx[:,:-1]);l=F.cross_entropy(p.flatten(0,1),xx[:,1:].flatten())/4;assert torch.isfinite(l);l.backward();lt+=float(l.detach())
    l,parts=cogloss(m,a,z,b,w,regime);assert torch.isfinite(l);l.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn)
    for g in o.param_groups:g['lr']=t.lr(i)
    o.step();assert all(torch.isfinite(p).all() for p in m.parameters());return dict(text_nll=lt,final4=float(parts[0].detach()),final8=float(parts[1].detach()),read4=float(parts[2].detach()),read8=float(parts[3].detach()),first8=float(parts[4].detach()),cog_loss=float(l.detach()),grad=float(gn),lr=t.lr(i))

def causal(m,x):
    m.eval()
    with torch.no_grad():
        x=x[:2];cut=x.size(1)//2;z=m(x);a,s=m(x[:,:cut],carry=True);b,s=m(x[:,cut:],s,True);err=float((z-torch.cat((a,b),1)).abs().max());q=x.clone();q[:,cut:]=MASK;fe=float((z[:,:cut]-m(q)[:,:cut]).abs().max())
    assert max(err,fe)<3e-5,(err,fe);return dict(streaming_error=err,causal_error=fe)

def engineering(root,arm,regime,seed):
    data=inputs(root,seed);d=[batch(data[0],data[1],i,*data[2:]) for i in range(2)]
    m=build(arm,seed);o=t.optimizer(m);f=causal(m,d[0][3]);update(m,o,d[0],0,regime)
    # Serialize actual optimizer and RNG; not just cloning weights.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'checkpoint.pt';torch.save(dict(weights=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state()),p);c=torch.load(p,weights_only=True)
        a=update(m,o,d[1],1,regime);m2=build(arm,seed);o2=t.optimizer(m2);m2.load_state_dict(c['weights']);o2.load_state_dict(c['optimizer']);torch.set_rng_state(c['rng']);b=update(m2,o2,d[1],1,regime);assert a==b and t.same(m.state_dict(),m2.state_dict()) and t.same(o.state_dict(),o2.state_dict())
    z=causal(m,d[1][3]);return dict(arm=arm,regime=regime,seed=seed,parameters=count(m),initial=f,after=z,exact_restart=True,scientific_fit=False)

def panel(seed,novel):
    rng=np.random.default_rng(seed+60601);_,dev,_=split(seed,8);x,w,m=eight(rng,dev,256,True,novel);y=w[:,-1]
    ix,x2,y2,eligible=counterfactuals(x,y,m,dev);assert torch.equal(trace8(x2)[:,-1],y2)
    # Reproduce same held worlds/mutations at longer history using common HOLD events.
    hold=np.stack((np.full((len(x),64),HOLD),rng.integers(8,size=(len(x),64))+KEY,np.full((len(x),64),MASK)),-1)
    lx,ly,_=make8(m['ids'],m['q'],m['op'],m['order'],m['other'],m['syntax'],hold_events=hold,change=m['change']);assert torch.equal(trace8(lx)[:,-1],y) and torch.equal(ly,y)
    direct=torch.from_numpy(((m['change'][:,0]==WRITE)|(m['change'][:,0]==RESET))&(m['change'][:,1]==m['q']+KEY))
    # Obtain changed fact from the counterfactual raw record stream, not its computed answer.
    edit=torch.stack((torch.full((len(ix),),WRITE),torch.from_numpy(m['q'][ix])+KEY,trace8(x2)[:,0]),-1);assert ((edit[:,2]>=VAL)&(edit[:,2]<VAL+4)).all()
    # Check that the edited prefix plus original query produces exactly the counterfactual answer.
    assert torch.equal(trace8(torch.cat((x[ix,:-8],edit,x[ix,-8:]),1))[:,-1],y2)
    return x,w,ix,x2,y2,eligible,lx,direct,edit

@torch.no_grad()
def evaluate(m,seed):
    m.eval();out={'four':eval4(m,seed)}
    for label in ('familiar','held_composition'):
        x,w,ix,x2,y2,eligible,lx,direct,edit=panel(seed,label=='held_composition');y=w[:,-1]
        def preds(inp):return torch.cat([m(z)[:,[-4,-3,-1]].argmax(-1) for z in inp.split(16)])
        p=preds(x);p2=preds(x2);pl=preds(lx);rev=[];parity=0
        for a,e in zip(x[ix].split(16),edit.split(16)):
            _,s=m(a[:,:-8],carry=True);z=m(a[:,-8:],s)[:,-1].argmax(-1);parity+=int((z==m(a)[:,-1].argmax(-1)).sum());_,s=m(e,s,True);rev.append(m(a[:,-8:],s)[:,-1].argmax(-1))
        rev=torch.cat(rev);ok=p[:,-1]==y
        out[label]=dict(n=len(y),correct=int(ok.sum()),read_correct=int((p[:,0]==w[:,0]).sum()),first_correct=int((p[:,1]==w[:,1]).sum()),valid=int(((p[:,-1]>=VAL)&(p[:,-1]<=VAL+4)).sum()),prefix_n=int((~direct).sum()),prefix_correct=int((ok&~direct).sum()),direct_n=int(direct.sum()),direct_correct=int((ok&direct).sum()),long_correct=int((pl[:,-1]==y).sum()),pairs=len(ix),pair_eligible=eligible,both_correct=int((ok[ix]&(p2[:,-1]==y2)).sum()),changed=int((p[ix,-1]!=p2[:,-1]).sum()),revision_both=int((ok[ix]&(rev==y2)).sum()),revision_agrees_fresh=int((rev==p2[:,-1]).sum()),cached_parity=parity,predictions=p.tolist(),targets=w.tolist(),long_predictions=pl.tolist(),direct=direct.tolist(),paired_indices=ix.tolist(),paired_predictions=p2[:,-1].tolist(),paired_targets=y2.tolist(),revision_predictions=rev.tolist(),panel_sha256=hashlib.sha256(x.numpy().tobytes()+w.numpy().tobytes()+lx.numpy().tobytes()+x2.numpy().tobytes()).hexdigest())
        assert parity==len(ix)
    return out

def train_fit(a):
    lock=source();t.verify(a.data);a.out.mkdir(parents=True,exist_ok=False);m=build(a.arm,a.seed);o=t.optimizer(m);d=inputs(a.data,a.seed);kh=hashlib.sha256();curve=[];elapsed=0.;begin=time.monotonic()
    conf=dict(arm=a.arm,regime=a.regime,seed=a.seed,parameters=count(m),source=lock,corpus=t.CORPUS,text_order_sha256=hashlib.sha256(d[1].tobytes()).hexdigest(),knowledge_splits=digest_split(a.seed),steps=STEPS,targets=STEPS*BATCH*T,examples4=STEPS*KB,examples8=STEPS*KB,torch=str(torch.__version__),numpy=np.__version__,threads=a.threads,run_id=os.environ.get('GITHUB_RUN_ID'),commit=os.environ.get('GITHUB_SHA'))
    write(a.out/'config.json',conf)
    def save(i):
        p=a.out/'checkpoint.pt';tmp=a.out/'checkpoint.tmp';torch.save(dict(weights=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state(),rng4=d[2].bit_generator.state,rng8=d[3].bit_generator.state,step=i,config=conf,curve=curve),tmp);os.replace(tmp,p);write(a.out/'receipt.json',dict(config=conf,step=i,terminal=i==STEPS,sha256=sha(p)))
    for i in range(STEPS):
        z=batch(d[0],d[1],i,*d[2:]);
        for v in z[1:]:kh.update(v.numpy().tobytes())
        start=time.monotonic();r=update(m,o,z,i,a.regime);elapsed+=time.monotonic()-start
        if (i+1)%64==0:r.update(step=i+1,training_seconds=elapsed);curve.append(r);print(json.dumps(r),flush=True)
        if (i+1)%256==0:save(i+1)
        if time.monotonic()-begin>5*3600 and i+1<STEPS:save(i+1);raise RuntimeError('partial preserved, not terminal')
    m.eval();w={k:v.detach().contiguous().clone() for k,v in m.state_dict().items()};save_file(w,str(a.out/'model.safetensors'));ck=torch.load(a.out/'checkpoint.pt',weights_only=True);n=build(a.arm,a.seed);n.load_state_dict(load_file(str(a.out/'model.safetensors')),strict=True);oo=t.optimizer(n);oo.load_state_dict(ck['optimizer']);assert ck['step']==STEPS and t.same(w,n.state_dict()) and t.same(w,ck['weights']) and t.same(o.state_dict(),oo.state_dict())
    e=evaluate(n,a.seed);text=t.evaltext(n,a.data);assert t.same(w,n.state_dict());write(a.out/'metrics.json',dict(text=text,cognition=e));
    with torch.no_grad():_,s=n(torch.arange(36)[None,:],carry=True)
    report=dict(status='TERMINAL_DEVELOPMENT_ONLY',terminal=True,reload_verified=True,config=conf,knowledge_draw_sha256=kh.hexdigest(),checkpoint_sha256=sha(a.out/'checkpoint.pt'),model_sha256=sha(a.out/'model.safetensors'),metrics_sha256=sha(a.out/'metrics.json'),language_bpb=text['bpb'],state_bytes_36=tensor_bytes(s),training_seconds=elapsed,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,four=e['four'],panels={k:{q:v for q,v in e[k].items() if not isinstance(v,list)} for k in ('familiar','held_composition')});write(a.out/'result.json',report);print(json.dumps(report),flush=True)

def aggregate(root,out):
    lock=source();rows={}
    for p in root.rglob('result.json'):
        r=json.loads(p.read_text());c=r['config'];k=(c['arm'],c['regime'],c['seed']);assert k not in rows and r['terminal'] and r['reload_verified'] and c['source']==lock and c['steps']==STEPS and c['targets']==STEPS*BATCH*T and c['examples4']==c['examples8']==STEPS*KB
        for fn,field in [('checkpoint.pt','checkpoint_sha256'),('model.safetensors','model_sha256'),('metrics.json','metrics_sha256')]:assert sha(p.parent/fn)==r[field]
        rec=json.loads((p.parent/'receipt.json').read_text());assert rec['terminal'] and rec['step']==STEPS and rec['sha256']==r['checkpoint_sha256'] and rec['config']==c
        d=json.loads((p.parent/'metrics.json').read_text());assert d['text']['targets']==5029184 and d['text']['bpb']==r['language_bpb'] and d['cognition']['four']==r['four']
        for label in ('familiar','held_composition'):
            v=d['cognition'][label];p0=np.array(v['predictions']);y=np.array(v['targets']);ix=np.array(v['paired_indices']);p2=np.array(v['paired_predictions']);y2=np.array(v['paired_targets']);direct=np.array(v['direct']);rv=np.array(v['revision_predictions']);ok=p0[:,-1]==y[:,-1]
            assert len(y)==v['n']==256 and int(ok.sum())==v['correct'] and int((ok&~direct).sum())==v['prefix_correct'] and int((ok[ix]&(p2==y2)).sum())==v['both_correct'] and int((ok[ix]&(rv==y2)).sum())==v['revision_both'] and v['cached_parity']==len(ix) and all(v[q]==z for q,z in r['panels'][label].items())
        rows[k]=r
    assert set(rows)==set(itertools.product(ARMS,REGIMES,SEEDS)),('missing',len(rows))
    for seed in SEEDS:
        group=[r for k,r in rows.items() if k[2]==seed]
        for a in ('text_order_sha256','knowledge_splits'):assert len({json.dumps(r['config'][a],sort_keys=True) for r in group})==1
        assert len({r['knowledge_draw_sha256'] for r in group})==1
        for label in ('familiar','held_composition'):assert len({r['panels'][label]['panel_sha256'] for r in group})==1
    def met(r):
        return dict(bpb=r['language_bpb'],four_paired=r['four']['full']['both_correct']/r['four']['full']['pairs'],**{label+'_'+name:r['panels'][label][num]/r['panels'][label][den] for label in ('familiar','held_composition') for name,num,den in [('prefix','prefix_correct','prefix_n'),('paired','both_correct','pairs'),('revision','revision_both','pairs'),('read','read_correct','n'),('first','first_correct','n'),('long','long_correct','n')]})
    groups=[]
    for arm,regime in itertools.product(ARMS,REGIMES):
        z=[met(rows[arm,regime,s]) for s in SEEDS];mean={k:float(np.mean([q[k] for q in z])) for k in z[0]};ss=float(np.mean([rows['SS_native_native',regime,s]['language_bpb'] for s in SEEDS]));gate=all(min(q[k] for k in ('four_paired','familiar_prefix','held_composition_prefix','held_composition_paired','held_composition_revision'))>=.9 for q in z) and mean['bpb']<=ss+.05
        groups.append(dict(arm=arm,regime=regime,mean=mean,per_seed=z,eligible_for_confirmation=gate))
    out.parent.mkdir(parents=True,exist_ok=True);d=dict(status='DEVELOPMENT_ONLY',terminal_fits=len(rows),groups=groups,warning='No architecture promotion or general reasoning claim. Matched total cognitive loss weight, privileged intermediate labels only in training; new task distinct from prior runs.');write(out,d);print(json.dumps(d,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('preflight','train','aggregate','audit'));p.add_argument('--data',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--arm',choices=ARMS);p.add_argument('--regime',choices=REGIMES);p.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0]);p.add_argument('--threads',type=int,default=4);a=p.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source()
    if a.mode=='train':train_fit(a)
    elif a.mode=='aggregate':aggregate(a.input,a.out)
    elif a.mode=='audit':
        t.verify(a.data);rows=[]
        for seed in SEEDS:
            d=inputs(a.data,seed);h=hashlib.sha256()
            for i in range(STEPS):
                # Audit every cognitive example without materializing full text batches.
                x,y=draw4(d[2],d[4],KB,False,True);z=trace4(x);assert torch.equal(z[:,-1],y);b,w,m=eight(d[3],d[5],KB,oracle=True);assert not any(tuple(v) in [(1,2),(2,1)] for v in m['op']);
                for v in (x,z,b,w):h.update(v.numpy().tobytes())
            panels=[panel(seed,n) for n in (False,True)];rows.append(dict(seed=seed,cognitive_examples=STEPS*2*KB,draw_sha256=h.hexdigest(),held_panels=[dict(n=len(v[0]),pairs=len(v[2]),prefix=int((~v[7]).sum()),long_length=v[6].shape[1]) for v in panels]));print(rows[-1],flush=True)
        a.out.parent.mkdir(parents=True,exist_ok=True);write(a.out,rows)
    else:
        t.verify(a.data);rows=[]
        for arm,regime in itertools.product((a.arm,) if a.arm else ARMS,(a.regime,) if a.regime else REGIMES):
            v=engineering(a.data,arm,regime,a.seed);rows.append(v);print(v,flush=True)
        a.out.parent.mkdir(parents=True,exist_ok=True);write(a.out,rows)
if __name__=='__main__':main()
