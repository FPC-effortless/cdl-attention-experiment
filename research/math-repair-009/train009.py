"""Equal-exposure, restartable FP32 CPU language pretraining. No official TEST access."""
import argparse,hashlib,importlib.util,json,math,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from model import NativeLM,count
from data009 import T,B,STEPS,STAGE,SEEDS,sha,verify,order
ARMS=('gpt','HH','A0','U','UA','AU');GPT_REV='f08abb45bd2285627d17da16daea14dda7e7253e';HERE=Path(__file__).resolve().parent

def make(arm,seed):
    torch.manual_seed(seed)
    if arm!='gpt':return NativeLM(arm)
    path=HERE/'vendor/nanogpt/model.py';actual=subprocess.check_output(['git','-C',str(path.parent),'rev-parse','HEAD'],text=True).strip();assert actual==GPT_REV,actual
    spec=importlib.util.spec_from_file_location('upstream_nanogpt',path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod.GPT(mod.GPTConfig(block_size=T,vocab_size=256,n_layer=3,n_head=5,n_embd=160,dropout=0.,bias=False))
def logits(m,x):
    if hasattr(m,'transformer'):
        # The upstream model returns all positions when targets are supplied.
        z,_=m(x,x);return z
    return m(x)
def loss(m,x,y):
    if hasattr(m,'transformer'):return m(x,y)[1]
    z=m(x);return F.cross_entropy(z.flatten(0,1),y.flatten())
def environment():return dict(torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),machine=platform.machine(),threads=torch.get_num_threads(),cpu=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),None),pip=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines())
def source_lock():return {k:sha(HERE/k) for k in ('model.py','data009.py','train009.py','PROTOCOL.md')}
@torch.no_grad()
def evaluate(m,root,split):
    m.eval();root=Path(root);raw=np.memmap(root/(split+'.bin'),dtype=np.uint8,mode='r');offs=np.load(root/(split+'_offsets.npy'),allow_pickle=False);doc_stats=np.zeros((len(offs)-1,3),dtype=np.float64);full=[];tails=[]
    for i,(a,b) in enumerate(zip(offs[:-1],offs[1:])):
        for j in range(int(a),int(b)-1,T):
            n=min(T,int(b)-1-j);(full if n==T else tails).append((i,j,n))
    # Every byte except the first of each article is scored, including tails.
    def take(items):
        n=max(x[2] for x in items);x=np.zeros((len(items),n),np.int64);y=np.full((len(items),n),-100,np.int64)
        for j,(_,start,length) in enumerate(items):x[j,:length]=raw[start:start+length];y[j,:length]=raw[start+1:start+length+1]
        x=torch.from_numpy(x);y=torch.from_numpy(y);z=logits(m,x);l=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none',ignore_index=-100).reshape_as(y);ok=(z.argmax(-1)==y)&(y!=-100)
        for i,(doc,_,nn) in enumerate(items):doc_stats[doc]+=[float(l[i].sum()),int(ok[i].sum()),nn]
    for items in (full,sorted(tails,key=lambda x:x[2])):
        for i in range(0,len(items),8):take(items[i:i+8])
    l,correct,n=doc_stats.sum(0);assert int(n)==sum(max(0,int(b-a)-1) for a,b in zip(offs[:-1],offs[1:]));return dict(bpb=l/n/math.log(2),accuracy=correct/n,targets=int(n),per_document=doc_stats.tolist())
def lr(i):
    warm=164
    return .0006*(i+1)/warm if i<warm else .00006+(.0006-.00006)*.5*(1+math.cos(math.pi*(i-warm)/(STEPS-1-warm)))
def preflight(m):
    m.eval();x=torch.arange(75)[None]%256;xx=x.clone();xx[:,39:]=(xx[:,39:]+71)%256
    with torch.no_grad():a=logits(m,x);b=logits(m,xx);err=float((a[:,:39]-b[:,:39]).abs().max());assert err<3e-5,err
    if not hasattr(m,'transformer'):
        with torch.no_grad():z,s=m(x[:,:21],carry=True);q,_=m(x[:,21:],s,True);ce=float((a-torch.cat([z,q],1)).abs().max());assert ce<3e-5,ce
    else:ce=None
    assert abs(count(m)-1077504)/1077504<.012, count(m)
    return dict(causal_error=err,chunk_error=ce,parameters=count(m))
def train(a):
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);assert a.seed in SEEDS and 1<=a.stage<=4;root=Path(a.data);mfest=verify(root);assert mfest['split']=='train';out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    starts=order(root,a.seed);order_sha=hashlib.sha256(starts.tobytes()).hexdigest();raw=np.memmap(root/'train.bin',dtype=np.uint8,mode='r');m=make(a.arm,a.seed);pf=preflight(m);opt=torch.optim.AdamW([{'params':[p for p in m.parameters() if p.ndim>=2],'weight_decay':.1},{'params':[p for p in m.parameters() if p.ndim<2],'weight_decay':0}],lr=.0006,betas=(.9,.95));lock=dict(arm=a.arm,seed=a.seed,source=source_lock(),data=sha(root/'manifest.json'),data_order=order_sha,steps=STEPS,context=T,batch=B,torch=str(torch.__version__),numpy=np.__version__,threads=a.threads,gpt_revision=GPT_REV);start=(a.stage-1)*STAGE;end=a.stage*STAGE;curve=[];training_sec=0.
    if start:
        p=Path(a.resume);assert p.is_file();receipt=json.loads(p.with_name('receipt.json').read_text());assert receipt['sha256']==sha(p);cp=torch.load(p,weights_only=True,map_location='cpu');assert cp['lock']==lock and cp['step']==start; m.load_state_dict(cp['weights']);opt.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng']);curve=cp['curve'];training_sec=cp['training_seconds']
    elif a.resume:raise ValueError('unexpected first-stage resume')
    idx=np.arange(T+1);time0=time.monotonic();env=environment();(out/'environment.json').write_text(json.dumps(env,indent=2));(out/'lock.json').write_text(json.dumps(lock,indent=2));(out/'preflight.json').write_text(json.dumps(pf,indent=2))
    def save(step,terminal=False):
        f=out/'checkpoint.pt';tmp=out/'checkpoint.tmp';torch.save(dict(weights=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),step=step,lock=lock,curve=curve,training_seconds=training_sec),tmp);os.replace(tmp,f);(out/'receipt.json').write_text(json.dumps(dict(sha256=sha(f),step=step,targets=step*B*T,terminal_stage=terminal,lock=lock,run_id=os.environ.get('GITHUB_RUN_ID'),commit=os.environ.get('GITHUB_SHA')),indent=2))
    for i in range(start,end):
        m.train();t=time.monotonic();a0=torch.from_numpy(raw[starts[i,:,None]+idx].astype(np.int64));opt.zero_grad(set_to_none=True);step_loss=0.
        for x in a0.split(4):
            l=loss(m,x[:,:-1],x[:,1:])/4;assert torch.isfinite(l);l.backward();step_loss+=float(l.detach())
        gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.param_groups[0]['lr']=opt.param_groups[1]['lr']=lr(i);opt.step();training_sec+=time.monotonic()-t
        if (i+1)%64==0:print(json.dumps(dict(step=i+1,targets=(i+1)*B*T,loss=step_loss,grad=float(gn),lr=lr(i),elapsed=time.monotonic()-time0)),flush=True)
        if (i+1)%256==0:
            curve.append(dict(step=i+1,loss=step_loss,lr=lr(i)));save(i+1)
        if time.monotonic()-time0>5.2*3600 and i+1<end:
            save(i+1);raise RuntimeError('CPU time budget exhausted: partial checkpoint retained, do not count as full stage')
    # Full development corpus evaluation; no official validation or TEST is loaded during training.
    dev=evaluate(m,root,'dev');(out/'development.json').write_text(json.dumps(dev));curve.append(dict(step=end,dev_bpb=dev['bpb'],dev_targets=dev['targets']));save(end,True)
    if end==STEPS:
        w={k:v.detach().cpu().contiguous().clone() for k,v in m.state_dict().items()};p=out/'model.safetensors';save_file(w,str(p));rr=load_file(str(p));assert all(torch.equal(v,rr[k]) for k,v in w.items());(out/'model_receipt.json').write_text(json.dumps(dict(sha256=sha(p),parameters=count(m),targets=STEPS*B*T,step=end,lock=lock,dev_bpb=dev['bpb'],training_seconds=training_sec),indent=2))
    print('TERMINAL_STAGE',a.arm,a.seed,a.stage,dev['bpb'],flush=True)
def final(a):
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);root=Path(a.data);assert verify(root)['split']=='validation';p=Path(a.resume);r=json.loads(p.with_name('model_receipt.json').read_text());assert r['step']==STEPS and r['targets']==STEPS*B*T and sha(p)==r['sha256'];assert r['lock']['arm']==a.arm and r['lock']['seed']==a.seed and r['lock']['source']==source_lock();m=make(a.arm,a.seed);m.load_state_dict(load_file(str(p)));e=evaluate(m,root,'validation');out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'validation.json').write_text(json.dumps(dict(arm=a.arm,seed=a.seed,checkpoint=r,eval=e,environment=environment(),data_manifest=sha(root/'manifest.json')),indent=2));print(a.arm,a.seed,'SEALED_VALIDATION',e['bpb'],e['targets'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','validate']);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',type=int);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--resume');p.add_argument('--threads',type=int,default=4);a=p.parse_args();train(a) if a.mode=='train' else final(a)
