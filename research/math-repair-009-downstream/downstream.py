"""Native semantic transfer. No supplied inference roles, parser or answer head."""
import argparse, json, hashlib, math, os, random, sys, time, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file, save_file
HERE=Path(__file__).resolve().parent
UP=HERE.parent/'math-repair-009'
sys.path.insert(0,str(UP))
from model import NativeLM, scan, count
from data009 import sha, verify
from train009 import evaluate as prose_eval

ARMS=('HH','U','A0','HH_random','U_random','A0_random','HH_ctx','U_ctx')
SEEDS=(9001,9011,9029)
LENGTHS=(768,2048,8192)
ENDS=(0,256,640,1024)
NAMES=('Lena','Noah','Zara','Omar','Mila','Jude','Tara','Asha')
TASKS=('recall','revision','role','scope')
IGNORE=-100

class Corpus:
    def __init__(self,p,split):
        self.raw=np.memmap(Path(p)/(split+'.bin'),dtype=np.uint8,mode='r')
        self.off=np.load(Path(p)/(split+'_offsets.npy'),allow_pickle=False)
        self.starts=np.load(Path(p)/(split+'_starts.npy'),allow_pickle=False)
        self.docs=list(zip(map(int,self.off[:-1]),map(int,self.off[1:])))
        self.cache={}
    def fill(self,r,n):
        assert n>=2
        if n not in self.cache:self.cache[n]=[(a,b) for a,b in self.docs if b-a>=n-2]
        eligible=self.cache[n];assert eligible,(n,'no eligible article')
        for _ in range(512):
            a,b=r.choice(eligible);j=r.randrange(a,b-n+3);v=bytes(self.raw[j:j+n-2])
            if not any(k.encode() in v for k in NAMES):return b'\n'+v+b'\n'
        raise RuntimeError('filler rejection limit')
    def prose(self,r):
        j=int(self.starts[r.randrange(len(self.starts))])
        return torch.tensor(self.raw[j:j+769].astype(np.int64))[None]

def record(r,i,v,held,rev=False):
    n=NAMES[i]
    if rev:
        templates=("Update: {} now has code {}. ","The new code for {} is {}. ") if not held else ("Disregard the previous identifier for {}; it has changed to {}. ","The replacement access number assigned to {} is {}. ")
    else:
        templates=("{} has code {}. ","The code for {} is {}. ") if not held else ("{} is assigned identifier {}. ","The access number associated with {} is {}. ")
    return (r.choice(templates).format(n,v).encode(),i,v)

def assemble(parts):
    out=b'';marks=[]
    for x,k,v in parts:
        out+=x
        if k>=0:marks.append((len(out)-2,1,k,v))
    return out,marks

def pair(seed,corpus,L,task,held):
    r=random.Random(seed)
    ids=r.sample(range(8),6);i,j=ids[:2]
    vals=dict(zip(ids,r.sample(range(8),6)));a,b=r.sample(range(8),2)
    q=b''; ys=[]; variants=[]
    # Freeze template choices across paired counterfactuals by restoring the RNG.
    saved=r.getstate()
    for side in (0,1):
        r.setstate(saved);v=vals.copy();mid=[]
        if task=='recall':v[i]=(a,b)[side]
        parts=[record(r,k,v[k],held) for k in ids if not(task=='scope' and k==i)]
        if task=='revision':
            mid=[record(r,i,(a,b)[side],held,True)]
            q=(f'What is the current code for {NAMES[i]}? Answer: ' if not held else f'State the replacement identifier of {NAMES[i]}: ').encode()
            y=48+(a,b)[side];key=i
        elif task=='recall':
            q=(f'What is the code for {NAMES[i]}? Answer: ' if not held else f'Give the access number assigned to {NAMES[i]}: ').encode()
            y=48+v[i];key=i
        elif task=='role':
            giver,recipient=(i,j) if side==0 else (j,i)
            if held: event=f'A parcel was received by {NAMES[recipient]} from {NAMES[giver]}. '
            else:event=f'{NAMES[giver]} gave a parcel to {NAMES[recipient]}. '
            parts.append((event.encode(),-1,-1))
            ask_recipient=bool(r.randrange(2));key=recipient if ask_recipient else giver
            q=(('The recipient has code ' if ask_recipient else 'The giver has code ') if not held else ('The party receiving the parcel has identifier ' if ask_recipient else 'The party sending the parcel has identifier ')).encode()
            y=48+v[key]
        else:
            action=('permits','forbids')[side] if not held else ('allows','denies')[side]
            parts.append((f'The manager {action} {NAMES[i]} access to the archive. '.encode(),i,side))
            q=(f'May {NAMES[i]} access the archive? Y or N: ' if not held else f'Is archive access authorized for {NAMES[i]}? Y or N: ').encode()
            y=ord(('Y','N')[side]);key=i
        p,m=assemble(parts);d,n=assemble(mid);variants.append((p,m,d,n,key,y))
        ys.append(y)
    assert ys[0]!=ys[1] and len(variants[0][0])==len(variants[1][0]) and len(variants[0][2])==len(variants[1][2])
    p,m,d,n,key,y=variants[0];f=L-len(p)-len(d)-len(q);assert f>=4
    f1=corpus.fill(r,f//2);f2=corpus.fill(r,f-f//2)
    X=[];Y=[];Labs=[]
    for p,m,d,n,key,y in variants:
        h=p+f1+d+f2+q
        assert len(h)==L
        lab=torch.full((L,3),IGNORE,dtype=torch.long)
        marks=m+[(pos+len(p)+len(f1),rr,k,v) for pos,rr,k,v in n]+[(L-1,2,key,IGNORE)]
        for pos,rr,k,v in marks:lab[pos]=torch.tensor([rr,k,v])
        avail=[z for z in r.sample(range(L),min(128,L)) if lab[z,0]==IGNORE]
        for z in avail[:16]:lab[z,0]=0
        X.append(torch.tensor(list(h)));Y.append(y);Labs.append(lab)
    assert torch.equal(X[0][-len(q):],X[1][-len(q):]) and not torch.equal(X[0],X[1])
    return torch.stack(X),torch.tensor(Y),torch.stack(Labs),len(q)

class Router(nn.Module):
    def __init__(self,arm,seed,parent):
        super().__init__();self.arm=arm;self.kind=arm.split('_')[0]
        torch.manual_seed(seed);self.base=NativeLM(self.kind)
        p=Path(parent);r=json.loads((p/'model_receipt.json').read_text())
        assert r['step']==8192 and r['targets']==100663296 and r['lock']['arm']==self.kind and r['lock']['seed']==seed
        assert sha(p/'model.safetensors')==r['sha256']
        assert sha(UP/'model.py')==r['lock']['source']['model.py'] and sha(UP/'data009.py')==r['lock']['source']['data009.py']
        self.parent_sha=r['sha256']
        if not arm.endswith('_random'):self.base.load_state_dict(load_file(str(p/'model.safetensors')))
        self.context=arm.endswith('_ctx')
        if self.context:
            self.ctx=nn.Linear(128,128)
            self.ctxout=nn.Linear(192,20)
            nn.init.zeros_(self.ctxout.weight);nn.init.zeros_(self.ctxout.bias)
            self.register_buffer('retention',torch.logit(torch.tensor([.5,.95,.99,.9999]).repeat_interleave(16)))
    def forward(self,x,state=None,labels=None,all_logits=False):
        base=self.base;st=None if state is None else state[0];cs=None if state is None else state[1]
        h,last,ax=base.local(x,base.core.emb,None if st is None else st[1])
        if self.context:
            z,a=self.ctx(h).chunk(2,-1);c,cs=scan(z.tanh(),(a+self.retention).sigmoid(),cs)
            da=self.ctxout(torch.cat((h,c),-1)).split((4,8,8),-1);ax=tuple(a+d for a,d in zip(ax,da))
        ss=[];sd=[]
        for i,l in enumerate(base.core.blocks):
            prev=None if st is None else st[0][i];extra=None
            if base.with_side:extra,side=base.sides[i](h,None if st is None else st[2][i]);sd.append(side)
            if base.kinds[i]=='H':h,s=l(h,ax,base.payload_map[i],prev,extra)
            else:h,s=l(h,prev,extra)
            ss.append(s)
        h=base.core.norm(h);z=F.linear(h if all_logits else h[:,-1],base.core.emb.weight)
        ns=(ss,last,sd) if base.with_side else (ss,last)
        aux=torch.stack([F.cross_entropy(a.flatten(0,1),labels[...,i].flatten(),ignore_index=IGNORE,reduction='sum') for i,a in enumerate(ax)]) if labels is not None else z.new_zeros(3)
        return z,(ns,cs),aux,torch.stack([a[:,-1].argmax(-1) for a in ax],-1)

def tree(v,f):
    if torch.is_tensor(v):return f(v)
    if v is None:return None
    return type(v)(tree(a,f) for a in v)
def prefix(m,x,st=None):
    for c in x.split(128,1):z,st,_,route=m(c,st)
    return z,st,route
def semantic_loss(m,x,y,labs):
    st=None;aux=torch.zeros(3)
    for c,lab in zip(x.split(128,1),labs.split(128,1)):
        z,st,a,_=checkpoint(lambda c,st,lab:m(c,st,lab),c,st,lab,use_reentrant=False)
        aux=aux+a
    den=(labs!=IGNORE).sum((0,1)).clamp_min(1)
    return F.cross_entropy(z,y)+.5*(aux/den).sum()

@torch.no_grad()
def ablate(m,st):
    ns,ctx=tree(st,lambda x:x.clone());lst=list(ns[0])
    for i,k in enumerate(m.base.kinds):
        if k=='H':
            fast,mem=lst[i];mem[:,:8]=0;lst[i]=(fast,mem)
    return ((lst,*ns[1:]),ctx)

@torch.no_grad()
def evaluate_semantic(m,corpus,seed,lengths,n):
    m.eval();res=[]
    for L in lengths:
        for held in (False,True):
            for ti,task in enumerate(TASKS):
                rows=[]
                for j in range(n):
                    es=seed+81000000+L*100000+int(held)*10000+ti*1000+j
                    x,y,lab,q=pair(es,corpus,L,task,held);_,st,_=prefix(m,x[:,:-q]);query=x[:,-q:]
                    z,_,r=prefix(m,query,st);reset,_,_=prefix(m,query)
                    sw,_,_=prefix(m,query,tree(st,lambda t:t.flip(0).contiguous()))
                    ab,_,_=prefix(m,query,ablate(m,st))
                    p=z.argmax(-1);rr=reset.argmax(-1);ss=sw.argmax(-1);aa=ab.argmax(-1)
                    rows.append(dict(seed=es,sha=[hashlib.sha256(bytes(t.tolist())).hexdigest() for t in x],target=y.tolist(),pred=p.tolist(),query_only=rr.tolist(),swapped=ss.tolist(),protected_removed=aa.tolist(),route=r.tolist(),target_key=lab[:,-1,1].tolist(),exact_both=bool((p==y).all()),swap_both=bool((ss==y.flip(0)).all()),reset_both=bool((rr==y).all()),protected_removed_both=bool((aa==y).all())))
                o=dict(length=L,held=held,task=task,pairs=n,exact_both=sum(x['exact_both'] for x in rows),exact=sum(sum(p==t for p,t in zip(x['pred'],x['target'])) for x in rows),swap_both=sum(x['swap_both'] for x in rows),reset_both=sum(x['reset_both'] for x in rows),protected_removed_both=sum(x['protected_removed_both'] for x in rows),read_endpoints=sum(sum(r[0]==2 for r in x['route']) for x in rows),correct_read_address=sum(sum(r[0]==2 and r[1]==k for r,k in zip(x['route'],x['target_key'])) for x in rows),false_permission=sum(sum(p==ord('Y') and t==ord('N') for p,t in zip(x['pred'],x['target'])) for x in rows) if task=='scope' else None,rows=rows)
                print('EVAL',json.dumps({k:v for k,v in o.items() if k!='rows'}),flush=True);res.append(o)
    return res

@torch.no_grad()
def parity(m):
    m.eval();x=torch.arange(128)[None]%256
    z,_,_,_=m(x,all_logits=True);_,st,_=prefix(m,x[:,:53]);a,_,_,_=m(x[:,53:],st,all_logits=True)
    e=float((z[:,53:]-a).abs().max());assert e<1e-4,e
    xx=x.clone();xx[:,64:]=(xx[:,64:]+17)%256;b,_,_,_=m(xx,all_logits=True)
    ce=float((z[:,:64]-b[:,:64]).abs().max());assert ce<1e-4,ce
    pe=float((z-m.base(x)).abs().max());assert pe<1e-5,pe
    return dict(chunk_error=e,causal_error=ce,parent_or_base_error=pe)

def optimizer(m):
    return torch.optim.AdamW([{'params':[p for p in m.parameters() if p.ndim>=2],'weight_decay':.01},{'params':[p for p in m.parameters() if p.ndim<2],'weight_decay':0}],lr=2e-4,betas=(.9,.95))

def save(out,m,opt,step,lock,curve,terminal,seconds):
    f=out/'checkpoint.pt';tmp=out/'checkpoint.tmp';torch.save(dict(weights=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),step=step,lock=lock,curve=curve,seconds=seconds),tmp);os.replace(tmp,f)
    (out/'receipt.json').write_text(json.dumps(dict(sha256=sha(f),step=step,terminal=terminal,lock=lock,seconds=seconds),indent=2))
def train(a,m,tr,dev,lock,out):
    opt=optimizer(m);start,end=ENDS[a.stage-1:a.stage+1];L=LENGTHS[a.stage-1];curve=[];seconds=0.
    if start:
        p=Path(a.resume)/'checkpoint.pt';r=json.loads(p.with_name('receipt.json').read_text());assert r['terminal'] and sha(p)==r['sha256'];cp=torch.load(p,weights_only=True,map_location='cpu');assert cp['step']==start and cp['lock']==lock
        m.load_state_dict(cp['weights']);opt.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng']);curve=cp['curve'];seconds=cp['seconds']
    else:
        assert not a.resume;(out/'parity.json').write_text(json.dumps(parity(m),indent=2))
    t0=time.monotonic();m.train()
    for i in range(start,end):
        t=time.monotonic();opt.zero_grad(set_to_none=True);total=0.
        for ti,task in enumerate(TASKS):
            x,y,labs,q=pair(a.seed+31000000+i*16+ti,tr,L,task,False);side=(i+ti)%2
            loss=semantic_loss(m,x[side:side+1],y[side:side+1],labs[side:side+1])/4
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
        p=tr.prose(random.Random(a.seed+53000000+i));z,_,_,_=m(p[:,:-1],all_logits=True)
        loss=F.cross_entropy(z.flatten(0,1),p[:,1:].flatten());assert torch.isfinite(loss);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step();seconds+=time.monotonic()-t
        row=dict(step=i+1,history=L,semantic_loss=total,prose_loss=float(loss.detach()),gradient=float(gn),training_seconds=seconds)
        if (i+1)%16==0:curve.append(row);print(json.dumps(row),flush=True)
        if (i+1)%32==0 or i+1==end:save(out,m,opt,i+1,lock,curve,i+1==end,seconds)
        if time.monotonic()-t0>5*3600 and i+1<end:save(out,m,opt,i+1,lock,curve,False,seconds);raise RuntimeError('partial stage; no terminal credit')
    if a.stage==3:
        weights={k:v.detach().contiguous().clone() for k,v in m.state_dict().items()};f=out/'model.safetensors';save_file(weights,str(f));rr=load_file(str(f));assert all(torch.equal(v,rr[k]) for k,v in weights.items())
        (out/'model_receipt.json').write_text(json.dumps(dict(sha256=sha(f),parameters=count(m),step=end,lock=lock),indent=2))
    print('TERMINAL_STAGE',a.arm,a.seed,a.stage,end,flush=True)

def prose(m,root):
    # Score every original development byte after first byte of each article,
    # with the same 768-byte resets as REPAIR-009. The wrapper preserves native logits.
    m.eval();cor=Corpus(root,'dev');n=0;s=0.
    with torch.no_grad():
        for a,b in cor.docs:
            for j in range(a,b-1,768):
                k=min(b-j-1,768);x=torch.tensor(cor.raw[j:j+k+1].astype(np.int64))[None];z,_,_,_=m(x[:,:-1],all_logits=True);s+=float(F.cross_entropy(z.flatten(0,1),x[:,1:].flatten(),reduction='sum'));n+=k
    assert n==sum(max(0,b-a-1) for a,b in cor.docs);return dict(bpb=s/n/math.log(2),targets=n)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('preflight','train','eval'));p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--stage',type=int,choices=(1,2,3));p.add_argument('--parent',required=True);p.add_argument('--data',required=True);p.add_argument('--resume');p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    data=Path(a.data);man=verify(data);assert man['split']=='train'
    m=Router(a.arm,a.seed,a.parent);tr=Corpus(data,'train');dev=Corpus(data,'dev');out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    lock=dict(arm=a.arm,seed=a.seed,parent_sha=m.parent_sha,parameters=count(m),source={str(x.relative_to(HERE.parent)):sha(x) for x in (HERE/'downstream.py',HERE/'PROTOCOL.md',UP/'model.py',UP/'data009.py')},corpus=sha(data/'manifest.json'),lengths=LENGTHS,ends=ENDS,torch=str(torch.__version__),numpy=np.__version__,device='cpu')
    (out/'environment.json').write_text(json.dumps(dict(lock=lock,platform=platform.platform(),cpu=next((x for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),None)),indent=2))
    if a.mode=='preflight':
        q=parity(m);m.train();x,y,labs,_=pair(a.seed+91000000,tr,8192,'revision',False);loss=semantic_loss(m,x[:1],y[:1],labs[:1]);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn) and gn>0
        opt=optimizer(m);opt.step();save(out,m,opt,1,lock,[],False,0.);cp=torch.load(out/'checkpoint.pt',weights_only=True);assert all(torch.equal(v,cp['weights'][k]) for k,v in m.state_dict().items());(out/'checkpoint.pt').unlink();(out/'receipt.json').unlink()
        (out/'preflight.json').write_text(json.dumps(dict(parity=q,finite_gradient=float(gn),loss=float(loss.detach()),classification='engineering-only',parameters=count(m)),indent=2));print('PREFLIGHT PASSED',a.arm,flush=True)
    elif a.mode=='train':
        assert a.stage is not None;train(a,m,tr,dev,lock,out)
    else:
        p=Path(a.resume);r=json.loads((p/'model_receipt.json').read_text());assert r['step']==1024 and r['lock']['parent_sha']==m.parent_sha and r['lock']['source']==lock['source'] and r['sha256']==sha(p/'model.safetensors')
        m.load_state_dict(load_file(str(p/'model.safetensors')));results=evaluate_semantic(m,dev,a.seed,LENGTHS,32);pr=prose(m,data);par=json.loads((Path(a.parent)/'model_receipt.json').read_text())['dev_bpb']
        gate=all(x['exact_both']>=math.ceil(x['pairs']*(.90 if x['held'] else .95)) and not x['false_permission'] for x in results) and (a.arm.endswith('_random') or pr['bpb']<=par+.05)
        (out/'qualification.json').write_text(json.dumps(dict(arm=a.arm,seed=a.seed,checkpoint=r,panels=results,prose=pr,parent_prose_bpb=par,pass_gate=gate),indent=2));print('QUALIFICATION',a.arm,a.seed,gate,flush=True)
if __name__=='__main__':main()
