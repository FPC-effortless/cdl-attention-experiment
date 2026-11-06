"""LRR-008: matched nuisance augmentation versus causal-state invariance.

No new inference parameters. The unchanged LRR-005 hard runtime is inherited.
The generator's filler mask is PRIVILEGED TRAINING AUGMENTATION, not a learned
irrelevance detector or a runtime input. The task remains synthetic.
"""
from __future__ import annotations
import hashlib, json, random, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .lrr005_register import (
    CausalRoleRegister, Example, FAMILIES, NAMES, FILLERS, VOCAB, ID,
    make_example, vectorize,
)

FORBIDDEN = frozenset({(0,3),(2,0),(1,2),(3,0),(1,3),(2,1)})
TEST_PAIR, DEV_PAIR = (1,3), (2,1)
ARMS = ("clean", "augmentation", "invariance")
MAPS = {
    "regular": {},
    "seen_permutation": dict(zip(FILLERS[:3], FILLERS[1:3]+FILLERS[:1])),
    "untrained_in_vocab": dict(zip(FILLERS[:3], FILLERS[3:])),
    "oov": dict(zip(FILLERS[:3], ("lrr8novela","lrr8novelb","lrr8novelc"))),
}

def alias_pair(ex):
    f = next(f for f in ex.facts if f[-1] == ex.answer)
    return FAMILIES[f[1]].index(f[2]), FAMILIES[f[1]].index(ex.question_alias)

def latest_records(ex):
    latest = {}
    for f in ex.facts:
        latest[f[:2]] = f
    return latest

def triple(seed, step, idx):
    base = make_example(seed, 7, "train")
    rng = random.Random(seed ^ (step << 13) ^ (idx << 7) ^ 0x8198)
    latest = latest_records(base)
    target = latest[(base.subject, base.family)]
    def allowed(f):
        words = FAMILIES[f[1]]; wi = words.index(f[2])
        return [w for j,w in enumerate(words) if (wi,j) not in FORBIDDEN]
    def variant(f, word):
        seq = list(base.words); seq[-2] = word
        return replace(base, words=tuple(seq), answer=f[-1], family=f[1], question_alias=word)
    a,p = rng.sample(allowed(target),2)
    other = rng.choice([f for (name, fam),f in latest.items()
                        if name == base.subject and fam != base.family])
    out = (variant(target,a), variant(target,p), variant(other,rng.choice(allowed(other))))
    for ex in out:
        assert alias_pair(ex) not in FORBIDDEN
        assert latest[(ex.subject,ex.family)][-1] == ex.answer
    assert out[0].answer == out[1].answer != out[2].answer
    assert out[0].question_alias != out[1].question_alias
    return out

def heldout(seed, facts, condition="regular", split="test"):
    layout = "shift" if condition == "syntax_shift" else "regular"
    base = make_example(seed,facts,"test",layout)
    pair = TEST_PAIR if split == "test" else DEV_PAIR
    gold = next(f for f in base.facts if f[-1] == base.answer)
    words = list(base.words); start=base.answer
    while start and words[start-1] != ".": start -= 1
    positions=[j for j in range(start,base.answer) if words[j] == gold[2]]
    assert len(positions)==1
    aliases=FAMILIES[base.family]
    words[positions[0]]=aliases[pair[0]]; words[-2]=aliases[pair[1]]
    mapping=MAPS.get(condition,{})
    words=tuple(mapping.get(w,w) for w in words)
    meta=tuple((f[0],f[1],aliases[pair[0]],f[3],f[4])
               if f[-1]==base.answer else f for f in base.facts)
    ex=replace(base,words=words,facts=meta,question_alias=aliases[pair[1]])
    assert alias_pair(ex)==pair
    assert latest_records(ex)[(ex.subject,ex.family)][-1]==ex.answer
    return ex

class InvariantRegister(CausalRoleRegister):
    def augmented_forward(self, x, mask, noise):
        """Identical source equations; optional TRAIN embedding substitution."""
        emb=self.emb(x)
        h=torch.where(mask[...,None],noise,emb)
        h=self.norm(h+F.gelu(self.conv(F.pad(h.transpose(1,2),(2,0))).transpose(1,2)))
        gates=self.role(h).sigmoid(); codes=self.code(h).softmax(-1)
        entity=torch.zeros(len(x),len(VOCAB),device=x.device)
        relation=torch.ones(len(x),self.codes,device=x.device)/self.codes
        entities=[]; relations=[]
        for t in range(x.shape[1]):
            entities.append(entity); relations.append(relation)
            ge,gr=gates[:,t,:1],gates[:,t,1:2]
            entity=ge*F.one_hot(x[:,t],num_classes=len(VOCAB)).float()+(1-ge)*entity
            relation=gr*codes[:,t]+(1-gr)*relation
        return torch.stack(entities,1),torch.stack(relations,1),gates

def augmented_batch(seed, step, histories=8):
    source=[]
    for i in range(histories):
        source.extend(triple(8_000_000_000+seed*1_000_000+step*1000+i,step,i))
    examples=source+source
    x,y,q=vectorize(examples); half=len(source)
    mask=torch.zeros_like(x,dtype=torch.bool)
    for i,ex in enumerate(source):
        mask[half+i,:len(ex.words)]=torch.tensor([w in FILLERS[:3] for w in ex.words])
    gen=torch.Generator().manual_seed(18_000_000_000+seed*1_000_000+step)
    # Same perturbation for the three questions over a common fact history.
    base=torch.randn(histories,x.shape[1],32,generator=gen)
    scale=.5+torch.rand(histories,x.shape[1],1,generator=gen)
    noise=torch.cat((torch.zeros(half,x.shape[1],32), (base*scale).repeat_interleave(3,0)))
    assert not mask[torch.arange(len(x)),y].any()
    for i,ex in enumerate(examples):
        for pos in mask[i].nonzero().flatten().tolist():
            assert ex.words[pos] in FILLERS[:3]
    return examples,x,y,q,mask,noise

def objective(model, x, y, q, mask, noise, arm):
    if arm not in ARMS: raise ValueError(arm)
    used=mask if arm != "clean" else torch.zeros_like(mask)
    e,r,g=model.augmented_forward(x,used,noise)
    rows=torch.arange(len(x)); qe,qr=e[rows,q],r[rows,q]
    scores=((e*qe[:,None]).sum(-1)*(r*qr[:,None]).sum(-1)*g[:,:,2]).clamp_min(1e-9)
    valid=torch.arange(x.shape[1])[None]<q[:,None]
    logits=torch.where(valid,scores.log(),torch.full_like(scores,-1e9))
    logits=torch.cat((logits,model.null_logit.expand(len(x),1)),-1)
    ce=F.cross_entropy(logits,y)
    gate=.018*(g*valid[:,:,None]).mean()+.06*F.softplus(-g[rows,y,2].clamp_min(1e-9).log()).mean()
    anchor,positive,negative=qr[0::3],qr[1::3],qr[2::3]
    same=(anchor*positive).sum(-1); diff=(anchor*negative).sum(-1)
    relational=-.2*same.clamp_min(1e-6).log().mean()-.2*torch.log1p(-diff.clamp(max=1-1e-6)).mean()
    half=len(x)//2
    d=(e[:half]-e[half:]).square().sum(-1)+(r[:half]-r[half:]).square().sum(-1)
    d=d+(g[:half]-g[half:]).square().mean(-1)
    inv=(d*valid[:half]).sum()/valid[:half].sum()
    loss=ce+gate+relational+(.5*inv if arm == "invariance" else 0)
    return loss,{"ce":ce.detach().item(),"invariance":inv.detach().item(),
                 "same":same.detach().mean().item(),"different":diff.detach().mean().item()}

def state_digest(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()

def fit(seed,arm,steps=500,histories=8,outdir=None):
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=InvariantRegister(); init_hash=state_digest(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.004)
    inp=hashlib.sha256();aug=hashlib.sha256();log=[];started=time.perf_counter()
    for step in range(steps):
        examples,x,y,q,mask,noise=augmented_batch(seed,step,histories)
        for a in (x,y,q): inp.update(a.numpy().tobytes())
        aug.update(mask.numpy().tobytes());aug.update(noise.numpy().tobytes())
        model.train();opt.zero_grad(set_to_none=True)
        loss,m=objective(model,x,y,q,mask,noise,arm)
        if not torch.isfinite(loss): raise RuntimeError(f"nonfinite {arm} {seed} {step}")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
        if step%50==0 or step==steps-1: log.append({"step":step+1,"loss":loss.item(),**m})
    elapsed=time.perf_counter()-started
    model.eval()
    assert sum(p.numel() for p in model.parameters())==4748
    counts=[int(s["step"].item()) for s in opt.state.values() if "step" in s]
    assert counts and set(counts)=={steps}
    receipt={"seed":seed,"arm":arm,"updates":steps,"parameters":4748,
             "source_histories":steps*histories,"question_view_exposures":steps*histories*6,
             "init_tensor_sha256":init_hash,"input_stream_sha256":inp.hexdigest(),
             "augmentation_sha256":aug.hexdigest(),"wall_seconds":elapsed,
             "optimizer_step_min":min(counts),"optimizer_step_max":max(counts),
             "trace":log,"final_tensor_sha256":state_digest(model)}
    if outdir:
        path=Path(outdir);path.mkdir(parents=True,exist_ok=True)
        target=path/f"lrr008_{arm}_{seed}.pt"
        torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"receipt":receipt},target)
        receipt["checkpoint_sha256"]=hashlib.sha256(target.read_bytes()).hexdigest()
    return model,receipt

@torch.no_grad()
def hard_batch(model,examples):
    """Batch only local features; execute original hard recurrence per row."""
    x,_,_=vectorize(examples);h=model.token_features(x)
    gs=model.role(h).sigmoid().tolist();cs=model.code(h).argmax(-1).tolist();out=[]
    for i,ex in enumerate(examples):
        entity=relation=None;table={};writes=nonvalue=0;goldkey=None;goldwritten=False
        valuepos={f[-1] for f in ex.facts}; written_keys={}
        for pos,word in enumerate(ex.words[:-1]):
            key=(entity,relation) if entity is not None and relation is not None else None
            if pos==ex.answer:goldkey=key;goldwritten=key is not None and gs[i][pos][2]>.5
            if gs[i][pos][2]>.5 and key is not None:
                table[key]=pos;writes+=1;nonvalue+=pos not in valuepos
                if pos in valuepos:written_keys[pos]=key
            if gs[i][pos][0]>.5:entity=word
            if gs[i][pos][1]>.5:relation=cs[i][pos]
        key=(entity,relation);pred=table.get(key)
        assert pred is None or pred<len(ex.words)-1
        failure=("correct" if pred==ex.answer else "no_candidate" if pred is None else
                 "overwrite" if goldkey==key and goldwritten else "wrong_address")
        collision_pairs=0
        by_address={}
        for name,fam,alias,value,pos in ex.facts:
            if pos in written_keys:
                group=by_address.setdefault((name,written_keys[pos]),{})
                group[fam]=group.get(fam,0)+1
        for group in by_address.values():
            counts=list(group.values())
            collision_pairs+=(sum(counts)**2-sum(c*c for c in counts))//2
        out.append({"pred":pred,"correct":pred==ex.answer,"failure":failure,
                    "collision_pairs":collision_pairs,
                    "writes":writes,"nonvalue_writes":nonvalue,"gold_written":goldwritten,
                    "gold_address_match":key==goldkey})
    return out

class SymbolicIndex:
    """Privileged grammar, answer-trained equivalence; no neural parameters."""
    def __init__(self): self.parents={}
    def root(self,x):
        self.parents.setdefault(x,x)
        while self.parents[x]!=x:
            self.parents[x]=self.parents[self.parents[x]];x=self.parents[x]
        return x
    @staticmethod
    def parse(words):
        known={w for a in FAMILIES.values() for w in a};records=[];start=0
        for j,w in enumerate(words):
            if w==".":
                segment=words[start:j];preds=[v for v in segment if v in known]
                assert len(preds)==1
                records.append((segment[0],preds[0],j-1));start=j+1
        question=words[start:];assert question[:2]==("what","does")
        return records,question[2],question[-2]
    def observe(self,ex):
        records,_,query=self.parse(ex.words)
        pred=next(r[1] for r in records if r[2]==ex.answer)
        a,b=self.root(pred),self.root(query)
        if a!=b:self.parents[b]=a
    def route(self,ex):
        records,entity,pred=self.parse(ex.words);table={}
        for name,alias,pos in records:table[(name,self.root(alias))]=pos
        return table.get((entity,self.root(pred)))
