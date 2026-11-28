"""Matched whole-model joint training on WikiText and two cognitive tasks."""
import argparse,hashlib,json,math,os,platform,resource,tempfile,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from integrated import WholeCOG,ARMS,count,tensor_bytes,VOCAB
from knowledge import split,draw4,draw8,digest_split,eval4,eval8
HERE=Path(__file__).resolve().parent
SEEDS=(9501,9511,9529);STEPS=1024;BATCH=16;T=768;KB=8
CORPUS='2955e5b6e772cf4539615b0585c1b522f7cf168c3c8e47c95b29dc97dda40c35'
FILES=('model.py','variants.py','integrated.py','knowledge.py','train.py','aggregate.py','PROTOCOL.md','SOURCE_LOCK.json','original014/hold_reset.py','original014/hold_grad.py','original014/hold_repair.py','original014/hierarchical.py','original014/scan.py')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def blob(p):
    b=Path(p).read_bytes();return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def verify(root):
    assert sha(root/'manifest.json')==CORPUS
    d=json.loads((root/'manifest.json').read_text());assert d['split']=='train';assert all(sha(root/k)==v for k,v in d['files'].items());return d

def source():
    lock=json.loads((HERE/'SOURCE_LOCK.json').read_text());assert all(blob(HERE/k)==v for k,v in lock['source_git_blobs'].items())
    return {k:sha(HERE/k) for k in FILES}

def order(root,seed):
    starts=np.load(root/'train_starts.npy',allow_pickle=False);return np.random.Generator(np.random.PCG64(seed+900090)).permutation(starts)[:STEPS*BATCH].reshape(STEPS,BATCH)
def optimizer(m):
    return torch.optim.AdamW([{'params':[p for p in m.parameters() if p.ndim>=2],'weight_decay':.1},{'params':[p for p in m.parameters() if p.ndim<2],'weight_decay':0.}],lr=.003,betas=(.9,.95))
def lr(i):return .003*(i+1)/20 if i<20 else .003*(.1+.45*(1+math.cos(math.pi*(i-20)/(STEPS-1-20))))

def batch(raw,starts,step,gr4,gr8,s4,s8,regime):
    x=torch.from_numpy(raw[starts[step,:,None]+np.arange(T+1)].astype(np.int64));fixed=regime=='fixed_full';mixed=regime=='random_mixed'
    a,y=draw4(gr4,s4,KB,fixed,mixed);b,z=draw8(gr8,s8,KB,fixed,mixed);return x,a,y,b,z

def update(m,o,data,i):
    m.train();o.zero_grad(set_to_none=True);x,a,y,b,z=data;loss=0.
    for t in x.split(4):
        p=m(t[:,:-1]);l=F.cross_entropy(p.flatten(0,1),t[:,1:].flatten())/4;assert torch.isfinite(l);l.backward();loss+=float(l.detach())
    l4=F.cross_entropy(m(a)[:,-1],y);l8=F.cross_entropy(m(b)[:,-1],z);assert torch.isfinite(l4)&torch.isfinite(l8);(.25*(l4+l8)).backward()
    gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn)
    for g in o.param_groups:g['lr']=lr(i)
    o.step();return dict(language_nll=loss,knowledge4_nll=float(l4.detach()),knowledge8_nll=float(l8.detach()),grad=float(gn),lr=lr(i))

def same(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

@torch.no_grad()
def evaltext(m,root):
    m.eval();raw=np.memmap(root/'dev.bin',dtype=np.uint8,mode='r');off=np.load(root/'dev_offsets.npy',allow_pickle=False);wins=[]
    for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
        for j in range(int(a),int(b)-1,T):wins.append((i,j,min(T,int(b)-1-j)))
    wins.sort(key=lambda x:x[2]);rows=np.zeros((len(off)-1,4),np.float64)
    for i in range(0,len(wins),8):
        items=wins[i:i+8];n=max(q[2] for q in items);x=np.zeros((len(items),n),np.int64);y=np.full((len(items),n),-100,np.int64)
        for k,(_,p,l) in enumerate(items):x[k,:l]=raw[p:p+l];y[k,:l]=raw[p+1:p+l+1]
        x,y=torch.from_numpy(x),torch.from_numpy(y);p=m(x);loss=F.cross_entropy(p.flatten(0,1),y.flatten(),reduction='none',ignore_index=-100).reshape(y.shape);byte=F.cross_entropy(p[...,:256].reshape(-1,256),y.flatten(),reduction='none',ignore_index=-100).reshape(y.shape);ok=(p.argmax(-1)==y)&(y!=-100)
        for k,(doc,_,l) in enumerate(items):rows[doc]+=[float(loss[k].double().sum()),float(byte[k].double().sum()),int(ok[k].sum()),l]
    v=rows.sum(0);assert int(v[3])==5029184
    return dict(bpb=float(v[0]/(v[3]*math.log(2))),byte_conditional_bpb=float(v[1]/(v[3]*math.log(2))),accuracy=float(v[2]/v[3]),targets=int(v[3]),per_doc=rows.tolist())

def init_batches(root,seed,regime):
    raw=np.memmap(root/'train.bin',dtype=np.uint8,mode='r');starts=order(root,seed);s4=split(seed,4)[0];s8=split(seed,8)[0];g4=np.random.default_rng(seed+21201);g8=np.random.default_rng(seed+21203)
    return raw,starts,g4,g8,s4,s8,regime

def engineering(root,out,arms=None):
    seed=SEEDS[0];rows=[];out.mkdir(parents=True,exist_ok=False)
    for arm in ARMS if arms is None else arms:
        m=WholeCOG(arm,seed);o=optimizer(m);raw,starts,g4,g8,s4,s8,regime=init_batches(root,seed,m.regime);x1=batch(raw,starts,0,g4,g8,s4,s8,regime);x2=batch(raw,starts,1,g4,g8,s4,s8,regime)
        v=update(m,o,x1,0)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'checkpoint.pt';torch.save(dict(weights=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state()),p);ck=torch.load(p,weights_only=True,map_location='cpu');v2=update(m,o,x2,1)
            m2=WholeCOG(arm,seed);o2=optimizer(m2);m2.load_state_dict(ck['weights']);o2.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);v3=update(m2,o2,x2,1);assert v2==v3 and same(m.state_dict(),m2.state_dict()) and same(o.state_dict(),o2.state_dict()),arm
        assert all(torch.isfinite(p).all() for p in m.parameters())
        factor_diag=[]
        if m.addressing=='factor':
            # Cold native routing can leave new addresses unused for early steps.
            # Test differentiability of the module separately, not fabricate task usage.
            eh=m.parent.core.emb(torch.cat((torch.zeros(2,4,dtype=torch.long),x2[1][:2]),1)).detach()
            for l in m.parent.core.blocks:
                assert l.slot_keys.grad is not None and torch.isfinite(l.slot_keys.grad).all()
                observed=float(l.slot_keys.grad.norm()); gg=torch.autograd.grad(l.factor_scores(eh).square().mean(),l.slot_keys)[0]
                assert torch.isfinite(gg).all() and gg.norm()>0
                factor_diag.append(dict(gain=float(l.factor_gain.detach()),actual_grad_norm=observed,isolated_gradient_norm=float(gg.norm())))
        m.eval()
        with torch.no_grad():
            x=x2[3];z,s=m(x,carry=True);a,s1=m(x[:,:17],carry=True);b,s2=m(x[:,17:],s1,True);e=float((z-torch.cat((a,b),1)).abs().max());assert e<3e-5,arm
            w=x.clone();w[:,17:]=(w[:,17:]+1)%VOCAB;ce=float((m(w)[:,:17]-z[:,:17]).abs().max());assert ce<3e-5,arm
            if m.bank=='presence':
                for st in s[0]:assert torch.all((st[2]==0)|(st[2]==1));assert torch.all(st[1][:,:8].masked_select((st[2]==0).expand(-1,-1,8))==0)
        r=dict(arm=arm,seed=seed,parameters=count(m),state_bytes=tensor_bytes(s)//len(x),first=v,second=v2,exact_restart=True,chunk_maxerror=e,causal_maxerror=ce,factor_path=factor_diag,scientific_fit=False);rows.append(r);print(json.dumps(r),flush=True);(out/'preflight.json').write_text(json.dumps(rows,indent=2))
    assert len(rows)==len(ARMS if arms is None else arms)
    print('ENGINEERING_PASSED',len(rows),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--threads',type=int,default=4);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);verify(a.data);lock=source()
    if a.preflight:engineering(a.data,a.out,None if a.arm is None else (a.arm,));return
    assert a.arm is not None and a.seed is not None;a.out.mkdir(parents=True,exist_ok=False);m=WholeCOG(a.arm,a.seed);o=optimizer(m);raw,starts,g4,g8,s4,s8,regime=init_batches(a.data,a.seed,m.regime);conf=dict(arm=a.arm,seed=a.seed,recipe=list(ARMS[a.arm]),source=lock,corpus=CORPUS,text_order_sha256=hashlib.sha256(starts.tobytes()).hexdigest(),knowledge_splits=digest_split(a.seed),steps=STEPS,targets=STEPS*BATCH*T,examples4=STEPS*KB,examples8=STEPS*KB,parameters=count(m),torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),threads=a.threads,commit=os.environ.get('GITHUB_SHA'),run_id=os.environ.get('GITHUB_RUN_ID'))
    (a.out/'config.json').write_text(json.dumps(conf,indent=2));curve=[];t=time.monotonic();trainsec=0.;kh=hashlib.sha256()
    def checkpoint(step,terminal=False):
        path=a.out/'checkpoint.pt';tmp=a.out/'checkpoint.tmp';torch.save(dict(weights=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state(),rng4=g4.bit_generator.state,rng8=g8.bit_generator.state,step=step,curve=curve,config=conf),tmp);os.replace(tmp,path);(a.out/'receipt.json').write_text(json.dumps(dict(step=step,terminal=terminal,targets=step*BATCH*T,examples4=step*KB,examples8=step*KB,sha256=sha(path),config=conf),indent=2))
    for i in range(STEPS):
        data=batch(raw,starts,i,g4,g8,s4,s8,regime)
        for x in data[1:]:kh.update(x.numpy().tobytes())
        st=time.monotonic();r=update(m,o,data,i);trainsec+=time.monotonic()-st
        if (i+1)%64==0:r.update(step=i+1,training_seconds=trainsec);curve.append(r);print(json.dumps(r),flush=True)
        if (i+1)%256==0:checkpoint(i+1)
        if time.monotonic()-t>5.0*3600 and i+1<STEPS:checkpoint(i+1);raise RuntimeError('partial preserved; no terminal fit')
    checkpoint(STEPS,True);m.eval();p=a.out/'model.safetensors';w={k:v.detach().cpu().contiguous().clone() for k,v in m.state_dict().items()};save_file(w,str(p));assert same(w,load_file(str(p)));ck=torch.load(a.out/'checkpoint.pt',weights_only=True,map_location='cpu');m2=WholeCOG(a.arm,a.seed);m2.load_state_dict(ck['weights']);o2=optimizer(m2);o2.load_state_dict(ck['optimizer']);assert ck['step']==STEPS and same(w,m2.state_dict()) and same(o.state_dict(),o2.state_dict())
    m2.eval();text=evaltext(m2,a.data);c4=eval4(m2,a.seed);c8=eval8(m2,a.seed);assert same(w,m2.state_dict());
    with torch.no_grad():_,s=m2(torch.arange(13)[None,:],carry=True);state_bytes=tensor_bytes(s)
    metrics=dict(text=text,cognition4=c4,cognition8=c8,state_bytes_13=state_bytes);(a.out/'metrics.json').write_text(json.dumps(metrics,indent=2));result=dict(status='TERMINAL_DEVELOPMENT_ONLY',terminal=True,reload_verified=True,config=conf,knowledge_draw_sha256=kh.hexdigest(),checkpoint_sha256=sha(a.out/'checkpoint.pt'),model_sha256=sha(p),metrics_sha256=sha(a.out/'metrics.json'),training_seconds=trainsec,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,language_bpb=text['bpb'],conditional_byte_bpb=text['byte_conditional_bpb'],cognition4=c4,cognition8=c8,state_bytes_13=state_bytes);(a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
