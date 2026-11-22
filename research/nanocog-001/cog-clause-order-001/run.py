import argparse
from itertools import combinations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

ENT = "zavin polmek teral biafen kurni velto prisek doskan remzu xolta sanvek yorin bardun heska gotvi pelrik ivora nalven catrix sorvak fanjo dielik lafru muntek".split()
REL = "shelters carries labels tracks".split()
VAL = "dovar jezti kalto umven briska fozil nadre trilux vorku saspi mendar yunek".split()
VOCAB = ENT + REL + VAL
V = len(VOCAB)
R = len(REL)
VALUES = range(len(ENT)+R, V)
SEEDS = (3371, 3471, 3571)
ARMS = ("sym", "order", "raw")
STEPS = 1024

def check(ok, msg):
    if not ok: raise RuntimeError(msg)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, o):
    with Path(p).open("x") as file:
        json.dump(o, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")

def kind(a, b):
    if a[0] == b[0]: return 0
    if a[1] == b[1]: return 1
    return 2

def build(path):
    path.mkdir()
    rng = random.Random(160271)
    units = [(e, r) for e in range(len(ENT)) for r in range(len(ENT), len(ENT)+R)]
    groups = [[] for _ in range(3)]
    for a, b in combinations(units, 2): groups[kind(a,b)].append((a,b))
    train, cal, held = [], [], []
    for xs in groups:
        rng.shuffle(xs)
        train.append(xs[64:])
        cal.append([[pair, [rng.sample(VALUES, 3) for _ in range(2)]] for pair in xs[64:128]])
        held.append([[pair, [rng.sample(VALUES, 3) for _ in range(4)]] for pair in xs[:64]])
    # Prevent repeated base configurations within a held group, independent of assignment.
    for cohort in [cal,held]:
        for xs in cohort:
            for _, triples in xs:
                seen=set()
                for i,t in enumerate(triples):
                    # All six 2-value worlds of a triple must be distinct from
                    # every other bundle in this key group (disjoint value-pairs).
                    while seen.intersection(combinations(sorted(t),2)): t=rng.sample(VALUES,3)
                    triples[i] = t
                    seen.update(combinations(sorted(t),2))
    check(all(len(s)==64 for s in held+cal), "cohort count")
    fit=set(tuple(tuple(k) for k in g) for pool in train for g in pool)
    h=set(tuple(tuple(k) for k in g) for pool in held for g, _ in pool)
    check(len(h)==192 and not(fit & h), "separation")
    for i,xs in enumerate(train):
        check({k[0] for a in xs for k in a}==set(range(len(ENT))) and {k[1] for a in xs for k in a}==set(range(len(ENT),len(ENT)+R)), f"coverage stratum {i}")
    write(path/"train.json", {"train":train, "cal":cal})
    for j, data in enumerate(held):
        prompts=[]; truths=[]; seen=set()
        for g, vals in data:
            for v in vals:
                pp, yy = bundle(g, v)
                for row in pp:
                    key=tuple(row)
                    check(key not in seen, "duplicate exact held prompt")
                    seen.add(key)
                prompts.extend(pp); truths.extend(yy)
        write(path/f"held-{j}-prompts.json",prompts)
        write(path/f"held-{j}-truth.json",truths)
    write(path/"custody.json", {"fit_key_pairs":[len(p) for p in train], "held_key_pairs":[64]*3,
                               "files":{p.name:sha(p) for p in sorted(path.glob("*.json"))}})

def bundle(g, v):
    x,y,z=v
    pp=[]; yy=[]
    for vals in [(x,y),(z,y),(x,z),(y,x),(z,x),(y,z)]:
        for order in (0,1):
            for q in (0,1):
                pp.append([*g[order], vals[order], *g[1-order], vals[1-order], *g[q]])
                yy.append(vals[q])
    return pp,yy

class Clause(nn.Module):
    def __init__(self, symmetric):
        super().__init__()
        self.symmetric=symmetric
        self.emb=nn.Embedding(V,48)
        self.enc=nn.Sequential(nn.Linear(144,72),nn.GELU(),nn.Linear(72,96))
        self.query=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,48))
        self.pos=nn.Parameter(torch.empty(96))
        self.out=nn.Sequential(nn.Linear(96,64), nn.GELU(), nn.Linear(64,V))
        nn.init.normal_(self.pos,std=.10)
    def forward(self,x):
        h=self.emb(x)
        cs=self.enc(h[:,:6].reshape(-1,2,144))
        if self.symmetric:
            cs=cs+self.pos
        else:
            cs=cs+torch.stack((self.pos,-self.pos))
        k,v=cs.chunk(2,-1)
        q=self.query(h[:,6:].reshape(-1,96))
        a=torch.softmax((k*q[:,None,:]).sum(-1)/(48**.5),dim=1)
        return self.out(torch.cat(((v*a[:,:,None]).sum(1),q),dim=-1))

class Block(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.n1=nn.LayerNorm(w)
        self.n2=nn.LayerNorm(w)
        self.qkv=nn.Linear(w,3*w)
        self.a=nn.Linear(w,w)
        self.ff=nn.Sequential(nn.Linear(w,2*w),nn.GELU(),nn.Linear(2*w,w))
    def forward(self,x):
        b,t,w=x.shape
        q,k,v=self.qkv(self.n1(x)).reshape(b,t,3,4,w//4).permute(2,0,3,1,4).unbind(0)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+self.a(a.transpose(1,2).reshape(b,t,w))
        return x+self.ff(self.n2(x))

class Raw(nn.Module):
    def __init__(self):
        super().__init__()
        w=44
        self.emb=nn.Embedding(V,w)
        self.pos=nn.Parameter(torch.empty(8,w))
        nn.init.normal_(self.pos,std=.1)
        self.body=nn.Sequential(Block(w),Block(w),nn.LayerNorm(w),nn.Linear(w,V))
    def all(self,x):
        return self.body(self.emb(x)+self.pos[:x.shape[1]])
    def forward(self,x):
        return self.all(x)[:,-1]

def model_for(arm):
    return Raw() if arm=="raw" else Clause(arm=="sym")

def compatibility():
    rng=torch.Generator().manual_seed(19)
    x=torch.randint(0,V,(79,8),generator=rng)
    mirror=torch.cat((x[:,3:6],x[:,:3],x[:,6:]),-1)
    torch.manual_seed(32814)
    s=model_for("sym")
    torch.manual_seed(32814)
    o=model_for("order")
    check(all(torch.equal(s.state_dict()[k],o.state_dict()[k]) for k in s.state_dict()),"initialization mismatch")
    check((s(x)-s(mirror)).abs().max().item()<=1e-6,"structural identity")
    check((o(x)-o(mirror)).abs().max().item()>1e-4,"negative control")
    r=Raw()
    x2=x.clone()
    x2[:,6:]=(x2[:,6:]+1)%V
    check((r.all(x)[:,:6]-r.all(x2)[:,:6]).abs().max().item()<=1e-6,"future leakage")
    check((r.all(x)[:,:5]-r.all(x[:,:5])).abs().max().item()<=1e-6,"prefix difference")
    return {arm:sum(p.numel() for p in model_for(arm).parameters()) for arm in ARMS}

def getrows(items):
    pp, yy = [], []
    for g, vv in items:
        for v in vv:
            p, y = bundle(g,v)
            pp.extend(p); yy.extend(y)
    return torch.tensor(pp),torch.tensor(yy)

def batch_outputs(m, x):
    with torch.no_grad():
        return torch.cat([m(b) for b in x.split(384)])

def score(logits, targets):
    check(logits.shape==(len(targets),V) and len(targets)%24==0 and torch.isfinite(logits).all(),"bad score shape")
    positions=logits.reshape(-1,6,2,2,V)
    high=positions.max(-1,keepdim=True).values
    exact=(positions.argmax(-1)==targets.reshape(-1,6,2,2))&((positions==high).sum(-1)==1)
    n=len(exact)
    # world/class axes: base, edit A, edit B, swap, swapped edit A, swapped edit B.
    p=exact[:,(0,3)].reshape(n,8)
    t=(exact[:,0]&exact[:,1]&exact[:,2]) & (exact[:,3]&exact[:,4]&exact[:,5])
    first=t[:,0,0]&t[:,1,1]
    second=t[:,0,1]&t[:,1,0]
    all_order=t.reshape(n,4).all(-1)
    swap=exact[:,0]&exact[:,3]
    a=positions[:,:,0].argmax(-1)
    b=positions[:,:,1].argmax(-1)
    return dict(histories=n, individual=int(exact.sum()), ordinary=int(p.all(-1).sum()),
                first=int(first.sum()),second=int(second.sum()), both_order_triplets=int(all_order.sum()),
                swap_pairs=int(swap.sum()), joint=int(exact.reshape(n,24).all(-1).sum()),
                order_agreement=int((a==b).sum()), wrong_agreement=int(((a==b)&~exact[:,:,0]&~exact[:,:,1]).sum()),
                max_order_logit_delta=float((positions[:,:,0]-positions[:,:,1]).abs().max()))

def adequacy(x):
    return x["individual"]*100>=95*24*x["histories"] and x["ordinary"]*100>=90*x["histories"]

def gate(x):
    return (adequacy(x) and all(x[k]*100>=90*x["histories"] for k in ("first","second","both_order_triplets"))
            and x["swap_pairs"]*100>=90*4*x["histories"])

def fit(data, result, arm, seed):
    train=json.loads((data/"train.json").read_bytes())
    path=result/f"{arm}-{seed}"
    path.mkdir()
    torch.manual_seed(seed)
    rng=random.Random(seed)
    m=model_for(arm)
    opt=torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, .0002)
    start=time.perf_counter()
    losses=[]
    for step in range(STEPS):
        items=[]
        for slot in range(8):
            g=rng.choice(train["train"][(step*8+slot)%3])
            items.append((g,[rng.sample(VALUES,3)]))
        x,y=getrows(items)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(m(x),y)
        check(torch.isfinite(loss),"nonfinite train")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        opt.step();sch.step()
        losses.append(float(loss.detach()))
    duration=time.perf_counter()-start
    m.eval()
    cal=[]
    for items in train["cal"]:
        x,y=getrows(items)
        cal.append(score(batch_outputs(m,x), y))
    torch.save(m.state_dict(), path/"weights.pt")
    write(path/"fit.json", {"arm": arm,"seed":seed,"steps":STEPS,"parameters":sum(p.numel() for p in m.parameters()),
                           "cal":cal,"adequate":all(map(adequacy,cal)),"cpu_fit_seconds":duration,
                           "losses":losses,"weight_sha":sha(path/"weights.pt"),"data_sha":sha(data/"train.json"),"source_sha":sha(__file__)})
    print(arm,seed,[x["ordinary"] for x in cal],flush=True)

def predict(data, result):
    for arm in ARMS:
        for seed in SEEDS:
            p=result/f"{arm}-{seed}"
            f=json.loads((p/"fit.json").read_bytes())
            check(sha(p/"weights.pt")==f["weight_sha"] and sha(__file__)==f["source_sha"], "fit integrity")
            m=model_for(arm)
            m.load_state_dict(torch.load(p/"weights.pt",map_location="cpu",weights_only=True))
            m.eval()
            inputs=[]
            for j in range(3):
                x=data/f"held-{j}-prompts.json"
                wire=batch_outputs(m,torch.tensor(json.loads(x.read_bytes())))
                torch.save(wire,p/f"held-{j}-logits.pt")
                inputs.append({"prompt_sha":sha(x),"logit_sha":sha(p/f"held-{j}-logits.pt")})
            write(p/"predict.json",inputs)

def judge(data, result):
    reports=[]
    for arm in ARMS:
        for seed in SEEDS:
            p=result/f"{arm}-{seed}"
            f=json.loads((p/"fit.json").read_bytes())
            pred=json.loads((p/"predict.json").read_bytes())
            verify=[]
            for j, row in enumerate(pred):
                # Independent from bundle truth generation: derive label from raw input
                # using observed equality of both entity and relation, and require
                # that it equals the held truth wire.
                xs=json.loads((data/f"held-{j}-prompts.json").read_bytes())
                ys=json.loads((data/f"held-{j}-truth.json").read_bytes())
                for x,y in zip(xs,ys,strict=True):
                    check(len(x)==8 and (x[:2]==x[6:]) != (x[3:5]==x[6:]),"unidentifiable key")
                    check(y==(x[2] if x[:2]==x[6:] else x[5]),"truth mismatch")
                w=p/f"held-{j}-logits.pt"
                check(sha(w)==row["logit_sha"] and sha(data/f"held-{j}-prompts.json")==row["prompt_sha"],"actor wire integrity")
                verify.append(score(torch.load(w,weights_only=True,map_location="cpu"),torch.tensor(ys)))
            reports.append({k:v for k,v in f.items() if k!="losses"}|{"held":verify,"passed":f["adequate"] and all(map(gate,verify))})
    write(result/"RESULTS.json",reports)
    print([(r["arm"],r["seed"],r["adequate"],r["passed"],[x["joint"] for x in r["held"]]) for r in reports])

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["test","build","fit","predict","judge"])
    p.add_argument("--data",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--seed",type=int,choices=SEEDS)
    a=p.parse_args()
    if a.action=="test": print(compatibility())
    elif a.action=="build": build(a.data)
    elif a.action=="fit": fit(a.data,a.out,a.arm,a.seed)
    elif a.action=="predict": predict(a.data,a.out)
    else: judge(a.data,a.out)
if __name__=="__main__": main()
