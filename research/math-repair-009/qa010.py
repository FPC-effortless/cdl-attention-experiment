"""REPAIR-010 / SEMANTIC-TRANSFER-001: real extractive QA, no supplied spans at inference."""
import argparse, collections, hashlib, json, math, os, platform, re, string, time, urllib.request, unicodedata
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
import train009 as base
from data009 import sha, verify, order

HERE = Path(__file__).resolve().parent
ARMS = ('gpt','HH','A0','U','UA','AU')
SEEDS = (9001,9011,9029)
REV = '0a3a8b7b57e8578ec40b2d2bb4c75aca1a6d6ce1'
URL = f'https://huggingface.co/datasets/rajpurkar/squad/resolve/{REV}/plain_text/train-00000-of-00001.parquet'
PARENT_RUN = '46044686359'
PARENT_DATA = '2955e5b6e772cf4539615b0585c1b522f7cf168c3c8e47c95b29dc97dda40c35'
T,B,M,STAGE,STEPS = 768,16,4,2048,4096
REPLAY_B = 4
PREFIX_MIN,PREFIX_MAX,ANSWER_MAX = 320,704,63

def write(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def read(p): return json.loads(Path(p).read_text())
def digest(b): return hashlib.sha256(b).hexdigest()
def title(s): return ''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if c.isalnum())
def context_key(s): return digest(' '.join(unicodedata.normalize('NFKC',s).casefold().split()).encode())
def bucket(s): return int(digest(('repair010-v1:'+s).encode()),16)%100

def prompt(c,q): return ('Context:\n'+c+'\nQuestion: '+q+'\nAnswer: ').encode('utf-8')
def row_to_sample(r):
    c,q=r['context'],r['question'];ans=r['answers'];a=ans['text'][0];at=ans['answer_start'][0]
    assert c[at:at+len(a)]==a,(r['id'],'invalid annotated span')
    if not a or not norm(a) or '\n' in a or '\r' in a or not q.strip() or not c.strip(): return None
    p=prompt(c,q);ab=a.encode('utf-8')
    if not (PREFIX_MIN<=len(p)<=PREFIX_MAX and 1<=len(ab)<=ANSWER_MAX):return None
    return dict(id=r['id'],title=title(r['title']),context=c,question=q,answer=a,answer_start=at,context_hash=context_key(c))

def make_arrays(rows):
    z=np.zeros((len(rows),T+1),np.uint8);p=np.empty(len(rows),np.int16);n=np.empty(len(rows),np.int16)
    for i,r in enumerate(rows):
        x=prompt(r['context'],r['question']);y=(r['answer']+'\n').encode();b=x+y
        assert PREFIX_MIN<=len(x)<=PREFIX_MAX and len(b)<=T+1
        z[i,:len(b)]=np.frombuffer(b,np.uint8);p[i]=len(x);n[i]=len(b)
    return z,p,n

def prepare(a):
    import pyarrow.parquet as pq
    root=Path(a.corpus);assert verify(root)['split']=='train' and sha(root/'manifest.json')==PARENT_DATA
    train_titles={title(x) for x in read(root/'train_documents.json')['titles']}
    dev_titles={title(x) for x in read(root/'dev_documents.json')['titles']}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);raw=out/'source.parquet'
    urllib.request.urlretrieve(URL,raw);source_hash=sha(raw)
    rows=pq.read_table(raw).to_pylist();assert len(rows)==87599,len(rows)
    group=collections.defaultdict(set)
    for r in rows:group[context_key(r['context'])].add(title(r['title']))
    sets={k:[] for k in ('train','dev','held')};counts=collections.Counter();seen=set();ids=set()
    for r in rows:
        assert r['id'] not in ids;ids.add(r['id']);x=row_to_sample(r)
        if x is None:counts['length_or_format_excluded']+=1;continue
        t=x['title'];k=x['context_hash']
        if t in dev_titles:counts['pretraining_dev_title_excluded']+=1;continue
        if len(group[k])!=1:counts['cross_title_duplicate_context_excluded']+=1;continue
        b=bucket(t);s='train' if b<80 else 'dev' if b<90 else 'held'
        if s=='held' and t in train_titles:counts['held_pretraining_train_title_excluded']+=1;continue
        identity=(k,x['question'].casefold().strip(),x['answer'])
        if identity in seen:counts['duplicate_qa_excluded']+=1;continue
        seen.add(identity);sets[s].append(x)
    titles={k:{r['title'] for r in v} for k,v in sets.items()};contexts={k:{r['context_hash'] for r in v} for k,v in sets.items()}
    assert not titles['held'] & (train_titles|dev_titles)
    for i,k in enumerate(sets):
        for j in list(sets)[i+1:]:assert not titles[k]&titles[j] and not contexts[k]&contexts[j]
    sizes={k:dict(rows=len(v),titles=len(titles[k]),contexts=len(contexts[k])) for k,v in sets.items()}
    assert sizes['train']['rows']>=12000 and sizes['dev']['rows']>=500 and sizes['held']['rows']>=500 and sizes['held']['titles']>=20,sizes
    manifest=dict(source_revision=REV,source_url=URL,source_sha256=source_hash,source_bytes=raw.stat().st_size,raw_rows=len(rows),selection=dict(prefix_min=PREFIX_MIN,prefix_max=PREFIX_MAX,answer_max=ANSWER_MAX,full_original_context=True),parent_data=PARENT_DATA,counts=dict(counts),sizes=sizes,steps=STEPS,batch=B,replay_batch=REPLAY_B,context=T,sets={})
    for k,v in sets.items():
        v.sort(key=lambda r:r['id']);d=out/('held' if k=='held' else 'training');d.mkdir(exist_ok=True)
        path=d/(k+'.json');write(path,v);p=d/(k+'.npz');z,pl,n=make_arrays(v);np.savez_compressed(p,z=z,p=pl,n=n)
        manifest['sets'][k]={path.name:sha(path),p.name:sha(p)}
    for s in ('training','held'):write(out/s/'manifest.json',manifest)
    print(json.dumps(manifest),flush=True)

def load_panel(root,k):
    root=Path(root);m=read(root/'manifest.json');assert m['source_revision']==REV and m['parent_data']==PARENT_DATA
    assert m['steps']==STEPS and m['batch']==B and m['context']==T
    assert all(sha(root/f)==h for f,h in m['sets'][k].items());d=np.load(root/(k+'.npz'),allow_pickle=False);r=read(root/(k+'.json'))
    assert len(r)==len(d['z'])==m['sizes'][k]['rows'];return r,d['z'],d['p'],d['n'],m

def source_lock():return {n:sha(HERE/n) for n in ('model.py','data009.py','train009.py','qa010.py','TRANSFER_PROTOCOL.md','PARENT_MODELS.json')}
def make(arm,seed,init,parent):
    m=base.make(arm,seed);assert abs(base.count(m)-1077504)/1077504<.012
    if init=='pretrained':
        p=Path(parent)/'model.safetensors';r=read(Path(parent)/'model_receipt.json');expected=read(HERE/'PARENT_MODELS.json')[arm][str(seed)]
        assert sha(p)==r['sha256']==expected and r['step']==8192 and r['targets']==100663296
        assert r['lock']['arm']==arm and r['lock']['seed']==seed and r['lock']['data']==PARENT_DATA
        assert all(r['lock']['source'][k]==sha(HERE/k) for k in ('model.py','data009.py','train009.py'))
        m.load_state_dict(load_file(str(p)),strict=True)
    return m

def tensor_batch(z,p,n,idx):
    b=z[idx].astype(np.int64);x=torch.from_numpy(b[:,:-1].copy());y=torch.from_numpy(b[:,1:].copy());col=np.arange(T)[None];mask=(col>=p[idx,None]-1)&(col<n[idx,None]-1);y[~torch.from_numpy(mask)]=-1
    assert int((y!=-1).sum())==int(np.sum(n[idx]-p[idx]));return x,y

def forward(m,x,chunk=True):
    if hasattr(m,'transformer'):return m(x.contiguous(),x.contiguous())[0]
    if not chunk:return m(x)
    state=None;out=[]
    for b in x.split(256,1):
        z,state=m(b,state,carry=True);out.append(z)
    return torch.cat(out,1)

def loss_sum(m,x,y):
    if hasattr(m,'transformer'):return m(x.contiguous(),y.contiguous())[1]*(y!=-1).sum()
    z=forward(m,x);return F.cross_entropy(z.flatten(0,1),y.reshape(-1),ignore_index=-1,reduction='sum')

def tensors(s):
    if torch.is_tensor(s):return [s] if s.requires_grad else []
    if isinstance(s,(list,tuple)):return [t for v in s for t in tensors(v)]
    return []

def preflight(m,x,y):
    # Production shape; no optimizer update; actual native carried state gradient.
    m.train();x=x[:M];y=y[:M];m.zero_grad(set_to_none=True)
    z=forward(m,x,False);loss=F.cross_entropy(z.flatten(0,1),y.reshape(-1),ignore_index=-1);loss.backward();assert torch.isfinite(loss)
    g={k:p.grad.detach().clone() for k,p in m.named_parameters() if p.grad is not None};m.zero_grad(set_to_none=True)
    q=forward(m,x,True);l=F.cross_entropy(q.flatten(0,1),y.reshape(-1),ignore_index=-1);l.backward();assert torch.isfinite(l)
    err=float((z.detach()-q.detach()).abs().max());dg=max(float((p.grad-g[k]).abs().max()) for k,p in m.named_parameters() if k in g);scale=max(1.,max(float(v.abs().max()) for v in g.values()));assert err<2e-4 and dg/scale<.002,(err,dg,scale)
    carry_grad=None
    if not hasattr(m,'transformer'):
        m.zero_grad(set_to_none=True);_,s=m(x[:,:256],carry=True);a,_=m(x[:,256:],s,carry=True);yy=y[:,256:];assert (yy!=-1).any();l=F.cross_entropy(a.flatten(0,1),yy.reshape(-1),ignore_index=-1);gg=torch.autograd.grad(l,tensors(s),allow_unused=True);vals=[float(v.norm()) for v in gg if v is not None];assert vals and all(math.isfinite(v) for v in vals) and sum(vals)>0;carry_grad=sum(vals)
    m.zero_grad(set_to_none=True)
    return dict(logit_max_error=err,gradient_max_error=dg,gradient_relative_error=dg/scale,carried_state_gradient_norm_sum=carry_grad,parameters=base.count(m),classification='engineering preflight; no fit')

def schedule(n,seed):
    rng=np.random.Generator(np.random.PCG64(seed+100010));blocks=[];total=0
    while total<STEPS*B:
        p=rng.permutation(n);blocks.append(p);total+=len(p)
    return np.concatenate(blocks)[:STEPS*B].reshape(STEPS,B)
def lr(step):
    return .0003*(step+1)/128 if step<128 else .00003+.00027*.5*(1+math.cos(math.pi*(step-128)/(STEPS-1-128)))

@torch.no_grad()
def teacher(m,z,p,n):
    m.eval();tot=0.;targets=0;correct=0
    for i in range(0,len(z),8):
        x,y=tensor_batch(z,p,n,np.arange(i,min(i+8,len(z))));q=forward(m,x);tot+=float(F.cross_entropy(q.flatten(0,1),y.reshape(-1),ignore_index=-1,reduction='sum'));targets+=int((y!=-1).sum());correct+=int(((q.argmax(-1)==y)&(y!=-1)).sum())
    return dict(answer_bpb=tot/targets/math.log(2),answer_byte_accuracy=correct/targets,targets=targets,rows=len(z))

def train(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows,z,p,n,manifest=load_panel(a.data,'train');_,dz,dp,dn,_=load_panel(a.data,'dev');assert sha(Path(a.corpus)/'manifest.json')==PARENT_DATA and verify(a.corpus)['split']=='train'
    m=make(a.arm,a.seed,a.init,a.parent);seq=schedule(len(z),a.seed);replay=order(a.corpus,a.seed).ravel()[:STEPS*REPLAY_B].reshape(STEPS,REPLAY_B);raw=np.memmap(Path(a.corpus)/'train.bin',dtype=np.uint8,mode='r');lock=dict(source=source_lock(),qa_manifest=sha(Path(a.data)/'manifest.json'),corpus=PARENT_DATA,arm=a.arm,seed=a.seed,init=a.init,steps=STEPS,context=T,batch=B,replay_batch=REPLAY_B,order=digest(seq.tobytes()),replay_order=digest(replay.tobytes()),parent=read(HERE/'PARENT_MODELS.json')[a.arm][str(a.seed)] if a.init=='pretrained' else None,torch=str(torch.__version__),numpy=np.__version__)
    x,y=tensor_batch(z,p,n,seq[0]);pf=preflight(m,x,y);write(out/'preflight.json',pf)
    opt=torch.optim.AdamW([{'params':[p for p in m.parameters() if p.ndim>=2],'weight_decay':.1},{'params':[p for p in m.parameters() if p.ndim<2],'weight_decay':0.}],lr=.0003,betas=(.9,.95));start=(a.stage-1)*STAGE;end=a.stage*STAGE;curve=[];targets=0;training_seconds=0.
    if start:
        path=Path(a.resume)/'checkpoint.pt';rec=read(path.with_name('receipt.json'));assert sha(path)==rec['sha256'] and rec['terminal_stage'] is True and rec['step']==start;cp=torch.load(path,weights_only=True,map_location='cpu');assert cp['step']==start and cp['lock']==lock;m.load_state_dict(cp['weights']);opt.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng']);curve=cp['curve'];targets=cp['answer_targets'];training_seconds=cp['training_seconds']
    write(out/'lock.json',lock);write(out/'environment.json',base.environment());t0=time.monotonic()
    def save(step,terminal):
        path=out/'checkpoint.pt';tmp=out/'checkpoint.tmp';torch.save(dict(weights=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),step=step,lock=lock,curve=curve,answer_targets=targets,training_seconds=training_seconds),tmp);os.replace(tmp,path);assert not torch.serialization.get_unsafe_globals_in_checkpoint(path);write(out/'receipt.json',dict(sha256=sha(path),step=step,qa_examples=step*B,answer_targets=targets,replay_targets=step*REPLAY_B*T,terminal_stage=terminal,lock=lock,training_seconds=training_seconds,run_id=os.environ.get('GITHUB_RUN_ID'),commit=os.environ.get('GITHUB_SHA')))
    for i in range(start,end):
        t=time.monotonic();m.train();x,y=tensor_batch(z,p,n,seq[i]);num=int((y!=-1).sum());opt.zero_grad(set_to_none=True);qa=0.
        for xx,yy in zip(x.split(M),y.split(M)):
            l=loss_sum(m,xx,yy)/num;assert torch.isfinite(l);l.backward();qa+=float(l.detach())
        rr=torch.from_numpy(raw[replay[i,:,None]+np.arange(T+1)].astype(np.int64));r=loss_sum(m,rr[:,:-1],rr[:,1:])/(REPLAY_B*T);assert torch.isfinite(r);(.25*r).backward();g=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(g);opt.param_groups[0]['lr']=opt.param_groups[1]['lr']=lr(i);opt.step();targets+=num;training_seconds+=time.monotonic()-t
        if (i+1)%128==0:
            record=dict(step=i+1,qa_loss=qa,replay_loss=float(r.detach()),answer_targets=targets,elapsed=time.monotonic()-t0);curve.append(record);print(json.dumps(record),flush=True)
        if (i+1)%256==0:save(i+1,False)
        if time.monotonic()-t0>4.9*3600 and i+1<end:save(i+1,False);raise RuntimeError('Partial CPU checkpoint retained; no terminal credit')
    dev=teacher(m,dz,dp,dn);language=base.evaluate(m,a.corpus,'dev');write(out/'qa_development.json',dev);write(out/'language_development.json',language);curve.append(dict(step=end,qa_dev_bpb=float(dev['answer_bpb']),language_dev_bpb=float(language['bpb'])));save(end,True)
    if end==STEPS:
        weights={k:v.detach().cpu().contiguous().clone() for k,v in m.state_dict().items()};path=out/'model.safetensors';save_file(weights,str(path));assert all(torch.equal(v,load_file(str(path))[k]) for k,v in weights.items());write(out/'model_receipt.json',dict(sha256=sha(path),step=end,lock=lock,qa_examples=end*B,answer_targets=targets,replay_targets=end*REPLAY_B*T,language_dev_bpb=float(language['bpb']),qa_dev_bpb=float(dev['answer_bpb']),training_seconds=training_seconds))
    print('TERMINAL_TRANSFER_STAGE',a.arm,a.seed,a.init,a.stage,dev,language['bpb'],flush=True)

def norm(s):
    s=''.join(c for c in s.lower() if c not in string.punctuation);s=re.sub(r'\b(a|an|the)\b',' ',s);return ' '.join(s.split())
def score(pred,gold):
    x,y=norm(pred),norm(gold);a,b=x.split(),y.split();n=sum((collections.Counter(a)&collections.Counter(b)).values());return dict(em=int(x==y),f1=(2*n/(len(a)+len(b))) if a and b else float(a==b))

@torch.no_grad()
def generate(m,prefix):
    # No answer span or answer length supplied. Newline is the only stop token.
    a=torch.tensor(list(prefix),dtype=torch.long)[None];result=[]
    if hasattr(m,'transformer'):
        a=a[:,-PREFIX_MAX:];z=m(a)[0]
        for _ in range(64):
            v=int(z[0,-1].argmax());
            if v==10:break
            result.append(v);a=torch.cat([a,torch.tensor([[v]])],1);z=m(a[:,-T:])[0]
    else:
        state=None
        for b in a.split(256,1):z,state=m(b,state,carry=True)
        for _ in range(64):
            v=int(z[0,-1].argmax());
            if v==10:break
            result.append(v);z,state=m(torch.tensor([[v]]),state,carry=True)
    return bytes(result).decode('utf-8',errors='replace')

@torch.no_grad()
def answer_logprob(m,prefix,answer):
    ab=(answer+'\n').encode();whole=torch.tensor(list(prefix+ab),dtype=torch.long)[None];p=len(prefix)
    if hasattr(m,'transformer'):
        whole=whole[:,-(T+1):];p=whole.shape[1]-len(ab)
    z=forward(m,whole[:,:-1]);q=z[:,p-1:p+len(ab)-1];y=whole[:,p:];assert q.shape[1]==len(ab)
    return -float(F.cross_entropy(q.flatten(0,1),y.reshape(-1),reduction='sum'))/len(ab)

def length_matched_context(rows,i):
    r=rows[i];n=len(r['context'].encode());fallback=None
    for k in range(1,len(rows)):
        d=rows[(i+k)%len(rows)]
        if d['title']==r['title']:continue
        b=d['context'].encode()[:n].decode('utf-8',errors='ignore');c=b+' '*(n-len(b.encode()));collision=r['answer'].casefold() in c.casefold()
        if not collision:return c,False
        if fallback is None:fallback=c
    assert fallback is not None,'No independent distractor'
    return fallback,True

def eval_held(a):
    rows,z,p,n,manifest=load_panel(a.data,'held');out=Path(a.out);out.mkdir(parents=True,exist_ok=True);m=make(a.arm,a.seed,'fresh',None);path=Path(a.resume)/'model.safetensors';r=read(path.with_name('model_receipt.json'));assert r['step']==STEPS and r['qa_examples']==STEPS*B and r['replay_targets']==STEPS*REPLAY_B*T and sha(path)==r['sha256'];assert r['lock']['source']==source_lock() and r['lock']['qa_manifest']==sha(Path(a.data)/'manifest.json') and (r['lock']['arm'],r['lock']['seed'],r['lock']['init'])==(a.arm,a.seed,a.init);m.load_state_dict(load_file(str(path)),strict=True);m.eval();metrics=teacher(m,z,p,n);records=[]
    # Every eligible held QA, not a subset selected on performance. Wrong-source control preserves byte length.
    for i,row in enumerate(rows):
        real=prompt(row['context'],row['question']);wc,collision=length_matched_context(rows,i);wrong=prompt(wc,row['question']);assert len(real)==len(wrong)
        pred=generate(m,real);bad=generate(m,wrong);records.append(dict(id=row['id'],title=row['title'],wrong_source_answer_substring=collision,correct_source=score(pred,row['answer']),wrong_source=score(bad,row['answer']),gold_logp=answer_logprob(m,real,row['answer']),wrong_gold_logp=answer_logprob(m,wrong,row['answer']),prediction=pred,wrong_prediction=bad))
        if (i+1)%128==0:write(out/'partial_records.json',records);print('HELD_PROGRESS',i+1,len(rows),flush=True)
    summary={k:{q:float(np.mean([r[k][q] for r in records])) for q in ('em','f1')} for k in ('correct_source','wrong_source')};summary['source_logp_gain']=float(np.mean([r['gold_logp']-r['wrong_gold_logp'] for r in records]));summary['wrong_source_collision_count']=sum(r['wrong_source_answer_substring'] for r in records);write(out/'evaluation.json',dict(arm=a.arm,seed=a.seed,init=a.init,model=r,teacher=metrics,summary=summary,records=records));print('TERMINAL_HELD',a.arm,a.seed,a.init,summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','train','evaluate'));p.add_argument('--corpus');p.add_argument('--data');p.add_argument('--out',required=True);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--init',choices=('fresh','pretrained'));p.add_argument('--stage',type=int,choices=(1,2));p.add_argument('--parent');p.add_argument('--resume');p.add_argument('--threads',type=int,default=4);a=p.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);{'prepare':prepare,'train':train,'evaluate':eval_held}[a.mode](a)
