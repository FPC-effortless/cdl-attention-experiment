"""Matched protected-memory interventions using the existing recurrent channels."""
import sys, hashlib, json, math, time, argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

here = Path(__file__).resolve().parent
for folder in ("unified-search-002","cog-balance-001","cog-adaptive-repair-002","cog-fact-diagnosis-003"):
    sys.path.insert(0,str(here.parent/folder))
from model import CogLayer, hard, scan, delta
from trace_fault import cases, ids
from scale import build_model
from repair_objective import repair_batch, repair_loss
from corpus import load_corpus
from confirm_repair import evaluate_confirmation
from profiler import profile_model

class ContextBlock(CogLayer):
    def forward(self,h,ax,payload_map,state=None,side=None):
        typ,key,val=ax
        u=self.norm(h)
        z,g=self.inp(u).chunk(2,-1)
        g=(.5+.49*g.sigmoid()).unflatten(-1,(-1,2)).mean(-1,keepdim=True).expand_as(z.unflatten(-1,(-1,2))).flatten(-2)
        recurrent,last=scan(z.tanh(),g,None if state is None else state[0])
        slot,bg=(h.new_zeros(len(h),32,8) if state is None else state[1]).split([8,24],1)
        # Residual write logits use recurrent features. The first eight address
        # projection rows are unused by the native background delta bank.
        if self.context_write:
            typ=typ+recurrent[...,:4]
            val=val+recurrent[...,4:12]
            write_key=key+self.address(recurrent)[...,:8]
        else:
            write_key=key
        role=hard(typ.softmax(-1))
        writes=role[...,1:2]+role[...,3:4]
        address=hard(write_key.softmax(-1))
        gate=writes*address
        payload=payload_map(val.softmax(-1)).tanh()*role[...,1:2]
        payload=payload.unsqueeze(-2).expand(-1,-1,8,-1)
        if self.training:
            retention=(1-gate).unsqueeze(-1).expand_as(payload)
            seq,end=scan(payload.flatten(-2),retention.flatten(-2),slot.flatten(-2))
            seq=seq.unflatten(-1,(8,8))
        else:
            ix=((torch.arange(h.size(1),device=h.device)[None,:,None]+1)*gate.long()).cummax(1).values
            seq=torch.cat([slot[:,None],payload],1).gather(1,ix[...,None].expand(-1,-1,-1,8))
            end=seq[:,-1]
        native_address=hard(key.softmax(-1))
        msg=(native_address.unsqueeze(-1)*seq).sum(-2)*(writes+role[...,2:3])
        if self.context_read:
            weights=self.readaddr(recurrent)[...,:8].softmax(-1)
            msg=msg+recurrent[...,12:13].sigmoid()*(weights.unsqueeze(-1)*seq).sum(-2)
        k=F.normalize(self.address(u)[...,8:],dim=-1)
        q=F.normalize(self.readaddr(u)[...,8:],dim=-1)
        rate=self.write(u).squeeze(-1).sigmoid().square()*role[...,0]
        read,bg=delta(k,self.payload(u).tanh(),q,rate,bg)
        msg=msg+read*role[...,0:1]
        out=recurrent+self.combine(u+recurrent).sigmoid()*self.direct(msg)
        h=h+self.out(out+side)
        h=h+self.ff(self.norm2(h))
        return h,(last,torch.cat((end.reshape_as(slot),bg),1))

@torch.inference_mode()
def grids(model,domain):
    rows=list(cases(192,957427,True,domain,independent=True))
    qs=("#a00=","#b00=") if domain=="symbol" else ("What digit does the log give for Ada?","What digit does the log give for Ben?")
    ok=[]; margins=[]; singles=[]
    for start in range(0,len(rows),8):
        batch=rows[start:start+8]
        z=model(ids([h+q for a,b,_,_ in batch for h in (a,b) for q in qs]))[:,-1].reshape(-1,2,2,288)
        for scores,(_,_,a,b) in zip(z,batch):
            d=scores[:,:,48+a]-scores[:,:,48+b]
            margins.append(float((d[0,0]-d[0,1]-d[1,0]+d[1,1])/4))
            correct=scores.argmax(-1)==48+torch.tensor([[a,b],[b,a]])
            ok.append(int(correct.all()));singles.append(float(correct.float().mean()))
    return dict(grids=sum(ok),worlds=len(ok),mean_margin=np.mean(margins),single=np.mean(singles))

def run(arm,seed,corpus,output,steps):
    torch.set_num_threads(1)
    m=build_model(arm if arm in ('GRU','GQA') else 'C',seed)
    if arm in ('write','read','both'):
        for block in m.parent.core.blocks:
            block.__class__=ContextBlock
            block.context_write=arm in ('write','both')
            block.context_read=arm in ('read','both')
    opt=torch.optim.AdamW(m.parameters(),lr=.0008,betas=(.9,.95),weight_decay=.01)
    corpus=load_corpus(Path(corpus));rng=np.random.default_rng(seed+311903);digest=hashlib.sha256()
    for step in range(steps):
        b=repair_batch(seed,step,8,balanced=True)
        opt.zero_grad()
        for t in (b.ids,b.final_targets,b.trace_targets):digest.update(t.numpy().tobytes())
        loss=repair_loss(m(b.ids),b,"trace_pair")
        loss.backward()
        if step%4==0:
            xy=torch.from_numpy(corpus.batch("train",rng,4,128))
            digest.update(xy.numpy().tobytes())
            F.cross_entropy(m(xy[:,:-1]).flatten(0,1),xy[:,1:].flatten()).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        opt.step()
    m.eval()
    with torch.inference_mode():
        rng=np.random.default_rng(781299);lm=[]
        for i in range(16):
            xy=torch.from_numpy(corpus.batch("dev",rng,8,128))
            lm.append(F.cross_entropy(m(xy[:,:-1]).flatten(0,1),xy[:,1:].flatten()).item()/math.log(2))
        report=dict(arm=arm,seed=seed,steps=steps,training_sha256=digest.hexdigest(),parameters=sum(p.numel() for p in m.parameters()),bpb=np.mean(lm),symbol=grids(m,"symbol"),english=grids(m,"english"),confirmation=evaluate_confirmation(m,worlds=64,batch_size=8),profile=profile_model(m,xy))
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("arm",choices=["C","write","read","both","GQA","GRU"])
    parser.add_argument("seed",type=int);parser.add_argument("corpus");parser.add_argument("output");parser.add_argument("--steps",type=int,default=256)
    a=parser.parse_args();run(a.arm,a.seed,a.corpus,a.output,a.steps)
