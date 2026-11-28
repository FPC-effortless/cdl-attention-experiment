"""New prospective intelligence/knowledge integration search; not COG-COGNITION-001 reproduction.
Actual HH/U/SS/GQA source and exact frozen UNIFIED-014 presence checkpoints.
"""
import argparse,csv,hashlib,json,math,os,platform,random,sys,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'hh-u-search-001'
LOCK=json.loads((HERE/'SOURCE_LOCK.json').read_text())
SEEDS=(9401,9411,9429)
BANK_SEEDS=dict(zip(SEEDS,(173,271,907)))
STEPS=512;BATCH=32;K=8;V=4;H=17
# core, knowledge path, addressing, training regime
ARMS={
 'HH_raw':('HH','raw','none','random'), 'U_raw':('U','raw','none','random'),
 'SS_raw':('SS','raw','none','random'), 'GQA_raw':('A0','raw','none','random'),
 'HH_bank':('HH','bank','role','random'),'U_bank':('U','bank','role','random'),
 'SS_bank':('SS','bank','role','random'),
 'protected_small':('small','bank','role','random'),'protected_matched':('wide','bank','role','random'),
 'SS_factor':('SS','bank','factor','random'),'SS_pooled':('SS','bank','pooled','random'),
 'SS_raw_mixed':('SS','raw','none','mixed'),'SS_bank_mixed':('SS','bank','role','mixed'),
 'SS_raw_fixed':('SS','raw','none','fixed'),'SS_bank_fixed':('SS','bank','role','fixed')}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def blob(p):
    b=Path(p).read_bytes();return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def state_bytes(s):
    if torch.is_tensor(s):return s.numel()*s.element_size()
    if isinstance(s,(tuple,list)):return sum(state_bytes(x) for x in s)
    return 0
def init(original):
    global c,pre,fresh,NativeLM,hard,OP,END
    assert blob(BASE/'model.py')==LOCK['hh_u']['model_blob'] and blob(BASE/'variants.py')==LOCK['hh_u']['variants_blob']
    sys.path.insert(0,str(BASE));sys.path.insert(1,str(original))
    import custody014 as c,presence as pre
    from model import NativeLM,hard
    from variants import fresh
    assert c.K==K and c.H+1==H and len(c.WORDS)+3<=256
    OP=torch.tensor([len(c.WORDS),len(c.WORDS)+1]);END=len(c.WORDS)+2
    assert all(x in c.W for x in ('for','near'))
def split(seed):
    g=torch.Generator().manual_seed(940001+seed);ids=torch.randperm(V**K,generator=g);n=len(ids)
    train=ids[:3*n//4];dev=ids[3*n//4:7*n//8];confirmation=ids[7*n//8:]
    assert len(torch.unique(ids))==n
    return train,dev,confirmation
def values(ids):
    return ((ids[:,None]//(V**torch.arange(K)))%V).long()
def draws(seed,ids,worlds=None,*,held_syntax=False,fixed=False,history=8,one_name=False):
    """All model input is raw word tokens and (bank only) the supplied 014 roster.
    Gold target/answer used only by evaluator, loss and explicit oracle intervention.
    """
    g=seed; n=len(ids)
    worlds=values(ids) if worlds is None else worlds
    # Same canonical 8-key roster; record order and query distractor randomized.
    names=c.NAME[c.KNOWN[:K]].expand(n,K,2).clone()
    roster=c.KNOWN[:K].expand(n,K)
    perm=torch.rand(n,K,generator=g).argsort(-1)
    rid=roster.gather(1,perm);rv=worlds.gather(1,perm)
    tmpl=torch.randint(3,(n,K),generator=g);alias=torch.randint(2,(n,K),generator=g)
    w=c.TT[torch.ones_like(rid),tmpl,rid,rv,alias]
    target=torch.randint(K,(n,),generator=g);other=(target+torch.randint(1,K,(n,),generator=g))%K
    op=torch.randint(2,(n,),generator=g);t=names[torch.arange(n),target];d=names[torch.arange(n),other]
    o=OP[op];f=torch.full_like(o,c.W['for']);near=torch.full_like(o,c.W['near']);e=torch.full_like(o,END);pad=torch.zeros_like(o)
    # Train with target before and after distractor; held syntax reorders familiar words.
    if one_name:
        a=torch.stack((o,f,t[:,0],t[:,1],pad,pad,pad,e),-1)
        b=torch.stack((f,t[:,0],t[:,1],o,pad,pad,pad,e),-1)
    elif held_syntax:
        a=torch.stack((o,near,d[:,0],d[:,1],f,t[:,0],t[:,1],e),-1)
        b=torch.stack((f,t[:,0],t[:,1],o,near,d[:,0],d[:,1],e),-1)
    else:
        a=torch.stack((o,f,t[:,0],t[:,1],near,d[:,0],d[:,1],e),-1)
        b=torch.stack((near,d[:,0],d[:,1],o,f,t[:,0],t[:,1],e),-1)
    q=torch.where(torch.rand(n,1,generator=g)<.5,a,b)
    gold=worlds[torch.arange(n),target];answer=(gold+torch.where(op==0,1,-1))%V
    correct_row=(perm==target[:,None]).long().argmax(-1)
    if history>K:
        L=history-K;ad=torch.randint(K,(n,L),generator=g);va=torch.randint(V,(n,L),generator=g);ti=torch.randint(3,(n,L),generator=g);al=torch.randint(2,(n,L),generator=g)
        distract=c.TT[torch.zeros_like(ad),ti,roster.gather(1,ad),va,al];w=torch.cat((w,distract),1)
    return dict(w=w,q=q,names=names,target=target,answer=answer,op=op,world=worlds,correct_row=correct_row,perm=perm,tmpl=tmpl,alias=alias,ids=ids)
def pair(b,allowed,g):
    # Keep both worlds in exactly the same development partition, change one queried value.
    n=len(b['target']);idx=torch.arange(n);old=b['world'][idx,b['target']];place=V**b['target']
    candidates=b['ids'][:,None]+(((old[:,None]+torch.arange(1,V))%V)-old[:,None])*place[:,None]
    mask=torch.zeros(V**K,dtype=torch.bool);mask[allowed]=True;ok=mask[candidates]
    keep=ok.any(-1)
    # Deterministic choice among the available held-world alternatives.
    choice=ok.float().argmax(-1);ids2=candidates[idx,choice];z={k:v[keep].clone() for k,v in b.items()}
    ids2=ids2[keep];w2=values(ids2);n=len(ids2);i=torch.arange(n)
    val=w2[i,z['target']];row=z['correct_row'];r=c.KNOWN[:K][z['target']]
    record=c.TT[torch.ones(n,dtype=torch.long),z['tmpl'][i,row],r,val,z['alias'][i,row]]
    other=z['w'].clone();other[i,row]=record
    answer=(val+torch.where(z['op']==0,1,-1))%V
    assert (values(ids2)!=z['world']).sum(-1).eq(1).all() and mask[ids2].all()
    return z,dict(z,w=other,world=w2,ids=ids2,answer=answer),record[:,None,:]
class Reader(nn.Module):
    def __init__(self,bank,mode):
        super().__init__();self.mode=mode;self.e=nn.Embedding(256,32)
        with torch.no_grad():self.e.weight[:len(c.WORDS)].copy_(bank.enc.emb.weight)
        self.role=nn.Sequential(nn.Linear(96,32),nn.GELU(),nn.Linear(32,1))
        self.pooled=nn.Sequential(nn.Linear(8*32,64),nn.GELU(),nn.Linear(64,32))
    def forward(self,q,names,bank):
        # Original lexical identity metric frozen; role weighting is a new learned mechanism.
        q0=q.masked_fill(q>=len(c.WORDS),c.W['<unk>'])
        if self.mode=='pooled':
            z=self.pooled(self.e(q).flatten(1));n=self.e(names).mean(-2)
            return (F.normalize(z,dim=-1)[:,None]*F.normalize(n,dim=-1)).sum(-1)*15
        with torch.no_grad():
            x=F.normalize(bank.enc.emb(q0),dim=-1);n=F.normalize(bank.enc.emb(names),dim=-1)
            sim=15*torch.einsum('bqd,bknd->bknq',x,n)
        if self.mode=='role':
            e=self.e(q);zeros=torch.zeros_like(e[:,:1]);left=torch.cat((zeros,e[:,:-1]),1);right=torch.cat((e[:,1:],zeros),1)
            s=self.role(torch.cat((left,e,right),-1)).squeeze(-1)
            sim=sim+s[:,None,None,:]
        # Only current bounded query/roster matching, no attention over historical tokens.
        sim=sim.masked_fill((q==0)[:,None,None,:],-100.)
        return sim.max(-1).values.sum(-1)
class System(nn.Module):
    def __init__(self,arm,seed,original):
        super().__init__();self.arm=arm;self.core_kind,self.path,self.reader_kind,self.regime=ARMS[arm]
        torch.manual_seed(seed)
        if self.core_kind in ('HH','U','SS'):
            self.core=fresh({'HH':'NN','U':'BB','SS':'SS'}[self.core_kind],seed)
        elif self.core_kind=='A0':self.core=NativeLM('A0')
        else:self.core=None
        # Keep initialization independent from core construction, paired within shared modules.
        torch.manual_seed(seed+70001)
        if self.path=='bank':
            self.bank=pre.Model('presence_curriculum')
            p=Path(original)/'fits'/f'presence_curriculum_{BANK_SEEDS[seed]}.pt'
            rows={r['checkpoint_path']:r for r in csv.DictReader((Path(original)/'checkpoint_manifest.csv').open())}
            assert sha(p)==rows['fits/'+p.name]['sha256']
            ck=torch.load(p,weights_only=True,map_location='cpu');self.bank.load_state_dict(ck['weights'],strict=True);self.bank.requires_grad_(False);self.bank.eval()
            self.reader=Reader(self.bank,self.reader_kind)
            if self.core is not None:self.inject=nn.Linear(H,128,bias=False)
            else:
                width=64 if self.core_kind=='small' else 890
                self.head=nn.Sequential(nn.Linear(8*32+H,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,V))
    def encode(self,w,names,state=None):
        if self.path=='bank':
            self.bank.eval()
            with torch.no_grad():
                for z in w.split(32,1):_,state,_,_=self.bank(z,names,state,mode='hard')
            return state
        for z in w.flatten(1,2).split(256,1):
            _,state=self.core(z,state,True)
        return state
    def core_query(self,q,selected=None,state=None):
        if selected is None:
            z,_=self.core(q,state,True);return z[:,-1,:V]
        # Exact shared COG computation with a new, explicitly identified read adapter.
        core=self.core;h,last,ax=core.local(q,core.core.emb,None)
        injection=self.inject(selected);h=torch.cat((h[:,:-1],h[:,-1:]+injection[:,None]),1)
        for i,l in enumerate(core.core.blocks):
            extra=None
            if core.with_side and core.sides[i] is not None:extra,_=core.sides[i](h,None)
            h,_=l(h,ax,core.payload_map[i],None,extra)
        h=core.core.norm(h);return F.linear(h[:,-1],core.core.emb.weight[:V])
    def query(self,q,names,state,target,oracle=None,removed=False):
        if self.path=='raw':return self.core_query(q,state=state),None
        al=self.reader(q,names,self.bank);a=hard(al.softmax(-1))
        if oracle is not None:a=torch.where(oracle[:,None],F.one_hot(target,K).to(a.dtype),a)
        selected=(a[...,None]*state).sum(1)
        if removed:selected=torch.zeros_like(selected)
        z=self.core_query(q,selected) if self.core is not None else self.head(torch.cat((self.reader.e(q).flatten(1),selected),-1))
        return z,al
    def forward(self,b,oracle=None,removed=False):
        w=b['w'];n=len(w)
        if self.path=='raw':
            if oracle is not None:
                single=torch.zeros_like(w);single[:,0]=w[torch.arange(n),b['correct_row']]
                w=torch.where(oracle[:,None,None],single,w)
            if removed:w=torch.zeros_like(w)
        s=self.encode(w,b['names'])
        return self.query(b['q'],b['names'],s,b['target'],oracle if self.path=='bank' else None,removed)
def evaluate(m,dev,seed):
    m.eval();out=[];summary={}
    # Matched, independently seeded panels. All normal conditions forbid oracle address.
    with torch.no_grad():
        for panel,L,hs,one,n in [('single',8,False,True,512),('two_names',8,False,False,512),('held_syntax',8,True,False,512),('long_64',64,True,False,128),('long_256',256,True,False,128)]:
            g=torch.Generator().manual_seed(seed+950001+L*37+int(hs)*31+int(one)*11);counts=dict(n=0,learned=0,oracle=0,removed=0,address_correct=0,address_n=0,state_bytes=None,paired_n=0,paired_both=0,oracle_paired_both=0,revision_both=0,revision_non_target_exact=0,revision_non_target_n=0)
            for j in range(0,n,16):
                ids=dev[torch.randint(len(dev),(min(16,n-j),),generator=g)];b=draws(g,ids,held_syntax=hs,history=L,one_name=one);sz=len(ids);s=m.encode(b['w'],b['names'])
                z,al=m.query(b['q'],b['names'],s,b['target']);p=z.argmax(-1)
                zo,_=m(b,torch.ones(sz,dtype=torch.bool));zr,_=m(b,removed=True);o=zo.argmax(-1);r=zr.argmax(-1)
                counts['n']+=sz;counts['learned']+=int((p==b['answer']).sum());counts['oracle']+=int((o==b['answer']).sum());counts['removed']+=int((r==b['answer']).sum());counts['state_bytes']=state_bytes(s)//sz
                if al is not None:counts['address_correct']+=int((al.argmax(-1)==b['target']).sum());counts['address_n']+=sz
                for i in range(sz):out.append(dict(panel=panel,world=int(ids[i]),target=int(b['target'][i]),answer=int(b['answer'][i]),learned=int(p[i]),oracle=int(o[i]),removed=int(r[i]),address=int(al[i].argmax()) if al is not None else None))
                if L==8 and not one:
                    a,b2,record=pair(b,dev,g)
                    if len(a['target']):
                        sa=m.encode(a['w'],a['names']);sb=m.encode(b2['w'],b2['names']);p1=m.query(a['q'],a['names'],sa,a['target'])[0].argmax(-1);p2=m.query(b2['q'],b2['names'],sb,b2['target'])[0].argmax(-1)
                        oa=m(a,torch.ones(len(p1),dtype=torch.bool))[0].argmax(-1);ob=m(b2,torch.ones(len(p1),dtype=torch.bool))[0].argmax(-1)
                        revised=m.encode(record,a['names'],sa);pr=m.query(a['q'],a['names'],revised,a['target'])[0].argmax(-1)
                        both=(p1==a['answer'])&(p2==b2['answer']);counts['paired_n']+=len(p1);counts['paired_both']+=int(both.sum());counts['oracle_paired_both']+=int(((oa==a['answer'])&(ob==b2['answer'])).sum());counts['revision_both']+=int(((p1==a['answer'])&(pr==b2['answer'])).sum())
                        if m.path=='bank':
                            mask=(torch.arange(K)[None,:]!=a['target'][:,None]);aa=sa[mask].contiguous().view(torch.int32);rr=revised[mask].contiguous().view(torch.int32);counts['revision_non_target_exact']+=int((aa==rr).all(-1).sum());counts['revision_non_target_n']+=int(mask.sum())
                        for i in range(len(p1)):out.append(dict(panel=panel+'_paired',world=int(a['ids'][i]),world_after=int(b2['ids'][i]),target=int(a['target'][i]),answer=int(a['answer'][i]),answer_after=int(b2['answer'][i]),learned_before=int(p1[i]),learned_after=int(p2[i]),oracle_before=int(oa[i]),oracle_after=int(ob[i]),revised_after=int(pr[i])))
            summary[panel]=counts
    return summary,out
def fit(a,preflight=False):
    train,dev,confirmation=split(a.seed);m=System(a.arm,a.seed,a.original)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0012,betas=(.9,.95),weight_decay=.01)
    g=torch.Generator().manual_seed(a.seed+960001);mix=torch.Generator().manual_seed(a.seed+970001);order=hashlib.sha256();curve=[];t=time.monotonic()
    steps=2 if preflight else STEPS;batch=2 if preflight else BATCH
    for i in range(steps):
        ids=train[torch.randint(len(train),(batch,),generator=g)]
        if m.regime=='fixed':ids=train[0].expand_as(ids)
        b=draws(g,ids);order.update(b['w'].numpy().tobytes());order.update(b['q'].numpy().tobytes());order.update(ids.numpy().tobytes())
        oracle=torch.rand(batch,generator=mix)<.5 if m.regime=='mixed' else None
        m.train();opt.zero_grad(set_to_none=True);z,_=m(b,oracle);loss=F.cross_entropy(z,b['answer']);assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step()
        if (i+1)%64==0 or i==steps-1:curve.append(dict(step=i+1,loss=float(loss.detach()),grad=float(gn)));print(a.arm,a.seed,curve[-1],flush=True)
    m.eval()
    if preflight:
        with torch.no_grad():
            b=draws(g,dev[:2],held_syntax=True);z,_=m(b);zo,_=m(b,torch.ones(2,dtype=torch.bool));zr,_=m(b,removed=True);assert all(x.shape==(2,V) and torch.isfinite(x).all() for x in (z,zo,zr))
            s=m.encode(b['w'],b['names']);q=m.query(b['q'],b['names'],s,b['target'])[0];assert torch.equal(z,q)
        return dict(arm=a.arm,seed=a.seed,passed=True,scientific_fit=False,parameters=sum(p.numel() for p in m.parameters()),trainable=sum(p.numel() for p in m.parameters() if p.requires_grad))
    a.out.mkdir(parents=True,exist_ok=False)
    source={n:sha(HERE/n) for n in ('experiment.py','restore014.py','SOURCE_LOCK.json','PROTOCOL.md','aggregate.py')}
    conf=dict(arm=a.arm,seed=a.seed,recipe=list(ARMS[a.arm]),steps=STEPS,examples=STEPS*BATCH,train_order_sha256=order.hexdigest(),split_sha256={k:hashlib.sha256(x.numpy().tobytes()).hexdigest() for k,x in [('train',train),('development',dev),('sealed',confirmation)]},source=source,parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),torch=str(torch.__version__),python=platform.python_version(),original014_bank_seed=BANK_SEEDS[a.seed] if m.path=='bank' else None,run_id=os.environ.get('GITHUB_RUN_ID'),commit=os.environ.get('GITHUB_SHA'))
    p=a.out/'checkpoint.pt';torch.save(dict(weights=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),data_rng=g.get_state(),mix_rng=mix.get_state(),config=conf,curve=curve),p)
    elapsed=time.monotonic()-t;before=sha(p);ck=torch.load(p,weights_only=True,map_location='cpu');m2=System(a.arm,a.seed,a.original);m2.load_state_dict(ck['weights'],strict=True);m2.eval()
    assert ck['config']==conf and ck['optimizer']['state'] and all(torch.equal(v,m2.state_dict()[k]) for k,v in m.state_dict().items())
    with torch.no_grad():
        b=draws(torch.Generator().manual_seed(a.seed+2),dev[:8]);assert torch.equal(m(b)[0],m2(b)[0])
    summary,rows=evaluate(m2,dev,a.seed);assert sha(p)==before
    pred=a.out/'predictions.jsonl'
    with pred.open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    r=dict(config=conf,status='TERMINAL_DEVELOPMENT_ONLY',new_training_fit=True,reload_verified=True,checkpoint_sha256=before,predictions_sha256=sha(pred),training_seconds=elapsed,panels=summary)
    (a.out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--preflight',action='store_true');p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);init(a.original)
    receipt=json.loads((a.original/'recovery_receipt.json').read_text());assert receipt['checkpoint_count']==42 and receipt.get('closure_replay_count')==9
    if a.preflight:
        a.out.mkdir(parents=True,exist_ok=False);rows=[]
        for arm in ARMS:
            a.arm=arm;a.seed=SEEDS[0];rows.append(fit(a,True))
        (a.out/'preflight.json').write_text(json.dumps(dict(status='ENGINEERING_ONLY',checks=rows,scientific_fits=0),indent=2)+'\n');return
    assert a.arm is not None and a.seed is not None;fit(a)
if __name__=='__main__':main()
