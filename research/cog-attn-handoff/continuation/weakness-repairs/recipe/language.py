"""Inventory instructions with train/held template and lexical splits."""
import json
import random
import re
import resource
import sys
import time

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

OPS = (
    (("copy", "mirror"), ("set {d} to the amount in {s} ; leave {u} alone",
                          "{u} is fine . copy the value from {s} into {d}",
                          "take {s} 's value and use it for {d} . ignore {u}"),
     ("as for {d} , it should now match {s} , even though {u} is nearby",
      "the replacement for {d} comes from {s} . {u} stays as is")),
    (("add", "supplement"), ("put the amount from {s} on top of {d} ; ignore {u}",
                             "add {s} 's count to {d} . no change to {u}",
                             "keep {u} as it is and increase {d} by the total in {s}"),
     ("whatever {s} has , that much more belongs in {d} . not {u}",
      "{d} keeps what it has plus the amount in {s} ; disregard {u}")),
    (("subtract", "deduct"), ("reduce {d} by {s} 's amount ; leave {u} alone",
                               "subtract the count in {s} from {d} , and ignore {u}",
                               "{u} is irrelevant . take {s} 's count away from {d}"),
     ("the balance in {d} should drop by however many are in {s} . leave {u} untouched",
      "keep {u} unchanged . the amount currently in {s} comes off the count for {d}")),
    (("max", "raise"), ("use the larger of {s} and {d} as the new value for {d} ; ignore {u}",
                         "keep {u} as is . raise {d} to {s} 's amount if it is higher",
                         "{d} must contain at least the count in {s} without decreasing ; ignore {u}"),
     ("between {s} and {d} , take whichever has more and put that value into {d} . neither is {u}",
      "if {s} exceeds {d} then match it in {d} , otherwise leave it . ignore {u}")),
)
NOVEL = ("replicate","augment","debit","floor")
NAMES = "alder ash birch cedar elm fir hazel hemlock larch maple oak pine rowan spruce willow yew".split()
COHORTS = ("familiar","syntax","lexical","both","unsupported")


def samples(seed, n, cohort):
    rng=random.Random(seed)
    rows=[]
    for i in range(n):
        op=rng.randrange(4)
        role=rng.sample(NAMES,3)
        if cohort in ("lexical","both"):
            template=("{verb} {d} with the number from {s} ; don't touch {u}"
                      if cohort=="lexical" else
                      "{u} stays fixed . using the current value in {s} , {verb} the total in {d}")
            text=template.format(verb=NOVEL[op],s=role[0],d=role[1],u=role[2])
        elif cohort=="unsupported":
            text=rng.choice(("send something from {s} to {d} or maybe {u}",
                             "divide {d} by {s} and then ship {u}",
                             "explain why {s} and {d} differ before touching {u}")).format(s=role[0],d=role[1],u=role[2])
        else:
            text=rng.choice(OPS[op][2 if cohort=="syntax" else 1]).format(s=role[0],d=role[1],u=role[2])
        if rng.random()<.5:
            text=rng.choice(("quick stock check : ", "for the weekly count , ", "please note : "))+text
        tokens=re.findall(r"[a-z]+|[^\w\s]",text)
        seen=list(dict.fromkeys(word for word in tokens if word in role))
        src,dst=(seen.index(role[k]) for k in (0,1))
        values=np.array([rng.randrange(8) for _ in range(3)])
        target=values.copy()
        target[dst]=apply(op,values[src],values[dst])
        rows.append((tokens,seen,values,target,op,src,dst))
    return rows


def apply(op, source, dest):
    return (source,(source+dest)%8,(dest-source)%8,np.maximum(source,dest))[op]


def encode(rows,vocab,names):
    encoded=[]
    for tokens,entities,_,_,*rest in rows:
        sequence=[names[entities.index(t)] if t in entities else vocab.get(t,1) for t in tokens]
        encoded.append(sequence)
    x=torch.zeros((len(rows),max(map(len,encoded))),dtype=torch.long)
    for j,sequence in enumerate(encoded):
        x[j,:len(sequence)]=torch.tensor(sequence)
    states=np.stack([r[2] for r in rows])
    targets=np.stack([r[3] for r in rows])
    masks=[]
    actions=[]
    for op in range(4):
        for src in range(3):
            for dst in range(3):
                if src==dst: continue
                updated=states.copy()
                updated[:,dst]=apply(op,states[:,src],states[:,dst])
                masks.append((updated==targets).all(1))
                actions.append((op,src,dst))
    actual=[actions.index(tuple(r[4:])) for r in rows]
    return x,torch.tensor(np.stack(masks,axis=1)),torch.tensor(actual),torch.tensor(actions)


class Grounder(nn.Module):
    def __init__(self,kind,n_vocab,names):
        super().__init__()
        self.kind=kind
        self.names=names
        self.embedding=nn.Embedding(n_vocab,48,padding_idx=0)
        if kind=="pointer":
            self.encoder=nn.GRU(48,24,batch_first=True,bidirectional=True)
        else:
            self.positions=nn.Embedding(128,48)
            layer=nn.TransformerEncoderLayer(48,4,96,dropout=0.0,batch_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.op=nn.Linear(48,4)
        self.roles=nn.Linear(48,2)

    def forward(self,x,actions):
        e=self.embedding(x)
        if self.kind=="pointer":
            h=self.encoder(e)[0]
        else:
            h=self.encoder(e+self.positions(torch.arange(x.shape[1])),src_key_padding_mask=x==0)
        visible=x!=0
        weights=visible.unsqueeze(-1)
        pooled=(h*weights).sum(1)/weights.sum(1)
        mentions=torch.stack([((x==token).unsqueeze(-1)*h).sum(1)/(x==token).sum(1,keepdim=True)
                              for token in self.names],dim=1)
        roles=self.roles(mentions)
        op,src,dst=actions.T
        return self.op(pooled)[:,op]+roles[:,src,0]+roles[:,dst,1]


def fit(model,data,steps,rng,lr):
    x,allowed,_,actions=data
    optim=torch.optim.AdamW(model.parameters(),lr=lr)
    model.train()
    for _ in range(steps):
        indices=torch.randint(len(x),(48,),generator=rng)
        logits=model(x[indices],actions)
        loss=(logits.logsumexp(1)-logits.masked_fill(~allowed[indices],-1e9).logsumexp(1)).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()


def assess(model,data,cohort,has_oov):
    x,compatible,gold,actions=data
    model.eval()
    start=time.process_time_ns()
    with torch.inference_mode():
        output=torch.cat([model(block,actions).softmax(-1) for block in x.split(64)])
    elapsed=time.process_time_ns()-start
    confidence,prediction=output.max(1)
    correct=prediction==gold
    effects=compatible[torch.arange(len(x)),prediction]
    result=dict(cohort=cohort,count=len(x),action=int(correct.sum()),effects=int(effects.sum()),
                cpu_ns=elapsed,candidates=prediction.tolist(),gold=gold.tolist(),effect_ok=effects.tolist())
    if cohort=="unsupported":
        result.update(action=None,effects=None)
    for gate in (.55,.80,.95):
        ungated=confidence>=gate
        result[f"confidence{gate}"]=[int(ungated.sum()),int((ungated&correct).sum()) if cohort!="unsupported" else 0]
        accepted=ungated&((x!=1).all(1) if has_oov else torch.ones(len(x),dtype=torch.bool))
        result[f"gate{gate}"]=[int(accepted.sum()),int((accepted&correct).sum()) if cohort!="unsupported" else 0]
    return result


def run(seed,arm):
    torch.manual_seed(seed)
    rng=torch.Generator().manual_seed(seed+721)
    train=samples(seed,4096,"familiar")
    support=samples(seed+3000,96,"lexical")
    # Child model can learn novel words only from the disjoint support examples.
    vocab={t:i+2 for i,t in enumerate(sorted({t for r in train+support for t in r[0] if t not in NAMES}))}
    names=list(range(len(vocab)+2,len(vocab)+5))
    model=Grounder("pointer" if arm=="candidate" else "transformer",len(vocab)+5,names)
    tests={c:encode(samples(seed+7000+i,384,c),vocab,names) for i,c in enumerate(COHORTS)}
    results=[]
    start=time.process_time_ns()
    fit(model,encode(train,vocab,names),450,rng,.003)
    baseline=time.process_time_ns()-start
    for stage in ("base","adapted"):
        if stage=="adapted":
            start=time.process_time_ns()
            fit(model,encode(train[:288]+support,vocab,names),120,rng,.001)
        rows=[assess(model,data,c,True) for c,data in tests.items()]
        results.append(dict(stage=stage,rows=rows,fit_cpu_ns=baseline if stage=="base" else time.process_time_ns()-start,
                            weight_bytes=sum(t.numel()*t.element_size() for t in model.parameters()),
                            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        np.savez_compressed(f"{__import__('os').environ['TRAINLAB_OUTPUT_DIR']}/{stage}.npz",
                            **{k:v.detach().numpy() for k,v in model.state_dict().items()})
    return results
