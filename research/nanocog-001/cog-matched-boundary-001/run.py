from pathlib import Path
import argparse, hashlib, json, math, random, time
import torch
from torch import nn
from torch.nn import functional as F
from data_gen import make, vocab

SEEDS = (4371,4471,4571)
ARMS = ('dot','add')
MODES = ('bind','dated','latest')
STEPS = 1024
PANELS = ('seen','novel','rewrite')
LIMIT = 24
WORDS = sorted(set(vocab()) - {'<PAD>'})
IDS = {s:i+1 for i,s in enumerate(WORDS)}

def req(ok,msg):
    if not ok: raise ValueError(msg)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def put(p,o):
    with Path(p).open('x') as f:
        json.dump(o,f,sort_keys=True,allow_nan=False)
        f.write('\n')

def read(p):
    return json.loads(Path(p).read_bytes())

def tokens(text):
    parts=text.split(' . ')
    # The segmenter knows punctuation, not grammar or reference IDs.
    req(len(parts)==4,'expected 3 sentence boundaries')
    seq=[[IDS[w] for w in p.split()] for p in parts]
    req(all(0<len(x)<=LIMIT for x in seq),'invalid length')
    return [x+[0]*(LIMIT-len(x)) for x in seq]

def rewrite(text):
    left=('in the archive ', 'according to the ledger ')
    right=(' in the archive', ' according to the ledger')
    out=[]
    for seg in text.split(' . '):
        query=seg.endswith(' ?')
        s=seg[:-2] if query else seg
        if not s.startswith(left): s=left[0]+s
        if not s.endswith(right): s=s+right[1]
        out.append(s+(' ?' if query else ''))
    return ' . '.join(out)

def build(path):
    path.mkdir()
    seen=set()
    configs={'train':(2391,512),'cal':(2392,128),
             'seen':(2393,256),'novel':(2394,256)}
    for name,(s,n) in configs.items():
        fams=make(s,name,n)
        count=dict.fromkeys(MODES,0)
        pp=[]; yy=[]
        for fam in fams:
            req(len(fam)==4 and len({r['mode'] for r in fam})==1,'bad family')
            mode=fam[0]['mode'];count[mode]+=1
            q=[r['text'] for r in fam]
            req(len(set(q))==4,'duplicate within family')
            for r in fam:
                req(r['gold'] in IDS and r['target'] in range(3) and r['mode'] in MODES,'invalid label')
                req(r['text'] not in seen,'duplicate exact prompts')
                seen.add(r['text'])
                tokens(r['text'])
            req(len({r['gold'] for r in fam})==2,'family must require distinct answers')
            pp.append(q)
            yy.append({'mode':mode,'gold':[r['gold'] for r in fam],
                       'target':[r['target'] for r in fam]})
        req(all(k==n for k in count.values()),'mode count')
        if name in ('train','cal'):
            put(path/(name+'.json'),{'text':pp,'truth':yy})
        else:
            put(path/(name+'-prompts.json'),pp)
            put(path/(name+'-truth.json'),yy)
    base=read(path/'seen-prompts.json')
    pp=[[rewrite(text) for text in fam] for fam in base]
    flat=[text for fam in pp for text in fam]
    req(not(set(flat)&seen) and len(set(flat))==len(flat),'duplicate rewrite')
    for row in flat: tokens(row)
    put(path/'rewrite-prompts.json',pp)
    put(path/'rewrite-truth.json',read(path/'seen-truth.json'))
    put(path/'lexicon.json',WORDS)
    put(path/'custody.json',{p.name:sha(p) for p in sorted(path.glob('*.json'))})

def enc(pp):
    return torch.tensor([tokens(text) for fam in pp for text in fam],dtype=torch.long)

class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm=arm
        w=40;d=48
        self.emb=nn.Embedding(len(WORDS)+1,w,padding_idx=0)
        self.p=nn.Parameter(.1*torch.randn(LIMIT,w))
        self.tmix=nn.Linear(w, w)
        self.local=nn.Conv1d(w,w,3,padding=1)
        self.phrase=nn.Sequential(nn.Linear(w*3+1,88),nn.GELU(),nn.Linear(88,96))
        self.kv=nn.Linear(96+1,2*d)
        self.q=nn.Linear(96,d)
        self.g=nn.Parameter(torch.ones(d))
        self.final=nn.Sequential(nn.Linear(2*d,72),nn.GELU(),nn.Linear(72,len(WORDS)))

    def forward(self,x):
        b=x.shape[0]
        valid=x.reshape(-1,LIMIT).ne(0)
        h=self.emb(x.reshape(-1,LIMIT))
        h=F.gelu(self.tmix(h+self.p))*valid[:,:,None]
        u=F.gelu(self.local(h.transpose(1,2)).transpose(1,2)+h)
        m=valid.sum(1,keepdim=True)
        mean=(u*valid[:,:,None]).sum(1)/m
        high=u.masked_fill(~valid[:,:,None],torch.finfo(u.dtype).min).max(1).values
        end=u[torch.arange(len(u),device=u.device),m[:,0]-1]
        z=self.phrase(torch.cat((mean,high,end,m/float(LIMIT)),dim=-1)).reshape(b,4,-1)
        p=z.new_tensor([-1.,0.,1.]).view(1,3,1).expand(b,3,1)
        k,v=self.kv(torch.cat((z[:,:3],p),dim=-1)).chunk(2,-1)
        q=self.q(z[:,3]); qx=q[:,None]
        h=(k*qx) if self.arm=='dot' else torch.tanh(k+qx)
        a=(h*self.g).sum(-1).div(math.sqrt(q.shape[-1])).softmax(-1)
        return self.final(torch.cat((q,(v*a[:,:,None]).sum(1)),dim=-1)),a

def evalx(m,x):
    with torch.no_grad():
        chunks=[m(t) for t in x.split(384)]
    return [torch.cat(t) for t in zip(*chunks)]

def y_encode(t):
    return torch.tensor([[IDS[s]-1 for s in d['gold']] for d in t]).flatten()

def score(logits, weights, t):
    req(logits.shape==(4*len(t),len(WORDS)) and weights.shape==(4*len(t),3) and torch.isfinite(logits).all() and torch.isfinite(weights).all(),'wire shape/finite')
    req(bool(torch.allclose(weights.sum(-1),torch.ones(len(weights)),atol=1e-6,rtol=0)),'normalized weights')
    y=y_encode(t)
    lmax=logits.max(-1,keepdim=True).values
    correct=((logits.argmax(-1)==y)&((logits==lmax).sum(-1)==1)).reshape(-1,4)
    selected=(weights.argmax(-1)==torch.tensor([d['target'] for d in t]).flatten()).reshape(-1,4)
    out={}
    for name in MODES:
        j=[i for i,d in enumerate(t) if d['mode']==name]
        a,b=correct[j],selected[j]
        out[name]=dict(n=len(j), correct=int(a.sum()),strict=int(a.all(-1).sum()),selected=int(b.sum()),selection_strict=int(b.all(-1).sum()))
    return out

def passes(r):
    return all(x['correct']*100>=95*4*x['n'] and x['strict']*100>=90*x['n'] for x in r.values())

def fit(data, out, arm, seed):
    torch.manual_seed(seed); r=random.Random(seed)
    tr,cal=(read(data/(n+'.json')) for n in ('train','cal'))
    x=enc(tr['text']).reshape(-1,4,4,LIMIT); y=y_encode(tr['truth']).reshape(-1,4)
    pools=[[i for i,t in enumerate(tr['truth']) if t['mode']==name] for name in MODES]
    m=Model(arm)
    start_sd=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in m.state_dict().values())).hexdigest()
    opt=torch.optim.AdamW(m.parameters(),lr=.002)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,.0002)
    out=out/f'{arm}-{seed}';out.mkdir()
    losses=[];start=time.perf_counter()
    for step in range(STEPS):
        ix=[r.choice(pools[(i+step)%3]) for i in range(16)]
        opt.zero_grad(set_to_none=True)
        z,_=m(x[ix].reshape(-1,4,LIMIT))
        loss=F.cross_entropy(z,y[ix].flatten())
        req(bool(torch.isfinite(loss)),'nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
        opt.step();sch.step();losses.append(float(loss.detach()))
    finish=time.perf_counter()
    m.eval()
    c=score(*evalx(m,enc(cal['text'])),cal['truth'])
    p=out/'weights.pt';torch.save(m.state_dict(),p)
    put(out/'fit.json',dict(arm=arm,seed=seed,steps=STEPS,params=sum(t.numel() for t in m.parameters()),common_init=start_sd,
                            train_sha=sha(data/'train.json'),cal_sha=sha(data/'cal.json'),source_sha=sha(__file__),generator_sha=sha(Path(__file__).with_name('data_gen.py')),
                            weight_sha=sha(p),fit_seconds=finish-start,cal=c,adequacy=passes(c),losses=losses))
    print(arm,seed,{k:v['strict'] for k,v in c.items()},flush=True)

def predict(data,out):
    for arm in ARMS:
        for seed in SEEDS:
            p=out/f'{arm}-{seed}'; fd=read(p/'fit.json'); w=p/'weights.pt'
            req(fd['weight_sha']==sha(w) and fd['source_sha']==sha(__file__) and fd['arm']==arm,'fit seal')
            m=Model(arm);m.load_state_dict(torch.load(w,map_location='cpu',weights_only=True));m.eval()
            ix={}
            for cohort in PANELS:
                pp=data/(cohort+'-prompts.json'); wire=p/(cohort+'.pt')
                logits,weights=evalx(m,enc(read(pp)))
                torch.save({'logits':logits,'weights':weights},wire)
                ix[cohort]={'prompts_sha':sha(pp),'wire_sha':sha(wire)}
            put(p/'predict.json',ix)

def judge(data,out):
    custody=read(data/'custody.json')
    req(all(sha(data/n)==h for n,h in custody.items()),'data custody')
    results=[]
    for arm in ARMS:
        for seed in SEEDS:
            p=out/f'{arm}-{seed}';fd=read(p/'fit.json'); manifest=read(p/'predict.json')
            req([fd['arm'],fd['seed'],fd['steps']]==[arm,seed,STEPS] and fd['train_sha']==custody['train.json'] and fd['cal_sha']==custody['cal.json'] and fd['weight_sha']==sha(p/'weights.pt'),'fit report misbound')
            r={k:v for k,v in fd.items() if k!='losses'}
            for n in PANELS:
                w=p/(n+'.pt')
                req(manifest[n]=={'prompts_sha':custody[n+'-prompts.json'],'wire_sha':sha(w)},'prediction seal')
                s=torch.load(w,map_location='cpu',weights_only=True)
                r[n]=score(s['logits'],s['weights'],read(data/(n+'-truth.json')))
            r['held_pass']=passes(r['seen']) and passes(r['novel'])
            r['qualified']=fd['adequacy'] and r['held_pass']
            results.append(r)
    for seed in SEEDS:
        a,b=[r for r in results if r['seed']==seed]
        req(a['common_init']==b['common_init'] and a['params']==b['params'],'unexpected differences in matched pair')
    put(out/'RESULTS.json',results)
    print([(a['arm'],a['seed'],a['adequacy'],a['qualified'],[[s['strict'] for s in a[k].values()] for k in ('seen','novel')]) for a in results],flush=True)

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser()
    parser.add_argument('cmd',choices=['test','build','fit','predict','judge'])
    parser.add_argument('--data',type=Path); parser.add_argument('--out',type=Path)
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--seed',type=int,choices=SEEDS)
    a=parser.parse_args()
    if a.cmd=='test':
        texts=[f['text'] for f in make(100,'train',2)[0]]
        xx=enc([texts])
        for arm in ARMS:
            z,p=Model(arm)(xx)
            req(z.shape==(4,len(WORDS)) and p.shape==(4,3) and bool(torch.allclose(p.sum(-1),torch.ones(4))),'test shapes')
            req(xx.eq(0).any(),'padding test')
    elif a.cmd=='build':build(a.data)
    elif a.cmd=='fit':fit(a.data,a.out,a.arm,a.seed)
    elif a.cmd=='predict':predict(a.data,a.out)
    else:judge(a.data,a.out)
if __name__=='__main__':main()
