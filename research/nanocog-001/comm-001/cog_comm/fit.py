"""Prospective CPU-only FIT worker for the matched COG-COMM-001 arms.

Use one independently admitted native stage per arm/seed. Only supervised
TRAIN family rows and the frozen train-only vocabulary enter this process.
"""
import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256,file_digest
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F

from .model import build_decoder
from .objective import SPECIAL, Tokenizer, paired_loss

ANSWERS=("blue","gold","green","orange","pink","purple","red","white")
MAX_TOKENS=512
MAX_STEPS=128


@dataclass(frozen=True)
class Plan:
    arm: str
    route_mode: str
    seed: int
    steps: int=128
    worlds_per_step: int=8
    max_tokens: int=512
    width: int=32
    layers: int=2
    block: int=8
    retained_blocks: int=64
    alpha: float=1.0
    lr: float=.001
    warmup: int=8
    weight_decay: float=.01
    clip: float=1.0

    def validate(self):
        if self.arm not in ("A","B","C") or self.route_mode not in ("learned","content_blind"):
            raise ValueError("invalid enrolled arm or routing mode")
        if self.arm!="C" and self.route_mode!="learned":
            raise ValueError("content-blind is a same-shape C control only")
        if (type(self.seed) is not int or self.seed not in (171,271,371) or
            type(self.steps) is not int or not 1<=self.steps<=MAX_STEPS or
            type(self.worlds_per_step) is not int or not 2<=self.worlds_per_step<=8 or self.worlds_per_step%2 or
            type(self.max_tokens) is not int or not 2<=self.max_tokens<=MAX_TOKENS or
            (self.width,self.layers,self.block,self.retained_blocks)!=(32,2,8,64) or
            not math.isfinite(self.alpha) or not 0<=self.alpha<=1 or
            (self.lr,self.warmup,self.weight_decay,self.clip)!=(.001,8,.01,1.0)):
            raise ValueError("plan exceeds the prepared prospective scope")
        return self


def digest(path):
    with Path(path).open('rb') as f:return file_digest(f,'sha256').hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def strict_json(text):
    def hook(items):
        taken=dict(items)
        if len(taken)!=len(items):raise ValueError("duplicate JSON key")
        return taken
    def nonstrict(x):raise ValueError("invalid JSON numeric term "+x)
    return json.loads(text,object_pairs_hook=hook,parse_constant=nonstrict)


def read_train(path):
    families=[];seen=set()
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            v=strict_json(line)
            require(isinstance(v,dict) and set(v)=={"split","prompts","answers"} and v["split"]=="train",
                    "only paired TRAIN-role rows are allowed")
            p,a=v["prompts"],v["answers"]
            require(isinstance(p,list) and len(p)==2 and all(type(t) is str and t.strip() for t in p)
                    and p[0]!=p[1] and all(x not in seen for x in p),"duplicate/malformed TRAIN prompt")
            require(isinstance(a,list) and len(a)==2 and all(t in ANSWERS for t in a) and a[0]!=a[1],
                    "malformed or nonopposing TRAIN targets")
            families.append((tuple(p),tuple(a)));seen.update(p)
    require(families,"empty TRAIN input")
    return families


def derive_tokenizer(families,minimum=5):
    return Tokenizer.from_train((p for ps,_ in families for p in ps),ANSWERS,minimum=minimum)


def read_vocab(path,families):
    v=strict_json(Path(path).read_text())
    require(isinstance(v,dict) and set(v)=={"special","words","minimum","answers","shape"} and
            v["special"]==list(SPECIAL) and v["minimum"]==5 and v["answers"]==list(ANSWERS),
            "frozen vocabulary record differs from protocol")
    tok=derive_tokenizer(families,v["minimum"])
    require(list(tok.words)==v["words"] and tok.size==v["shape"],"external vocabulary is not the exact train-only fit")
    return tok


def encode(families, tok, cap):
    data=[];dupe=set()
    for prompts,answers in families:
        pair=tuple(tuple(tok.encode(p)) for p in prompts)
        require(len(pair[0])==len(pair[1]) and 2<=len(pair[0])<=cap,
                "mismatched pair length or over-cap episode; refuse instead of truncation")
        require(pair[0]!=pair[1] and not any(p in dupe for p in pair),"duplicate TRAIN encoded prompt")
        dupe.update(pair)
        data.append((pair,tuple(tok.index[v] for v in answers)))
    return data


def order_for(size,plan):
    batch=plan.worlds_per_step//2
    require(size>=batch and size%batch==0,"TRAIN family count must divide exact fixed per-step batch")
    rng=random.Random(plan.seed ^ 0xC060001)
    groups=[]
    while len(groups)<plan.steps:
        order=list(range(size));rng.shuffle(order)
        groups.extend(tuple(order[i:i+batch]) for i in range(0,size,batch))
    return tuple(groups[:plan.steps])


def batch_inputs(data,indices):
    """Suffix-pad only. Answer scoring remains at the final real prefix token.

    No target token or scorer metadata is ever an input; the padded state is
    discarded after each forward. Artificial slots never enter the loss.
    """
    rows=[(ids,yt) for k in indices for ids,yt in zip(*data[k])]
    lengths=torch.tensor([len(ids) for ids,_ in rows],dtype=torch.long)
    x=torch.zeros((len(rows),int(lengths.max())),dtype=torch.long)
    pfx=torch.full_like(x,-1)
    for j,(ids,_) in enumerate(rows):
        tok=torch.tensor(ids,dtype=torch.long)
        x[j,:len(ids)]=tok
        pfx[j,:len(ids)-1]=tok[1:]
    y=torch.tensor([yt for _,yt in rows],dtype=torch.long).reshape(-1,2)
    return x,lengths,y,pfx


def metadata(data,plan,order,tok):
    selected=[data[i] for u in order for i in u]
    lengths=[len(p) for pair,_ in selected for p in pair]
    require(len(lengths)==plan.steps*plan.worlds_per_step and
            sum(lengths)<=plan.steps*plan.worlds_per_step*plan.max_tokens,
            "realized schedule exceeds token cap")
    dense=sum(max(len(p) for i in u for p in data[i][0])*plan.worlds_per_step for u in order)
    return {"train_families":len(data),"unique_families_observed":len({i for u in order for i in u}),
            "max_encoded_tokens":max(len(p) for row,_ in data for p in row),
            "planned_unpadded_tokens":sum(lengths),"planned_dense_input_tokens":dense,
            "planned_padding_tokens":dense-sum(lengths),"planned_answer_targets":len(lengths),
            "planned_prefix_targets":sum(lengths)-len(lengths),"vocab_size":tok.size,
            "schedule_sha256":sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()}


def merge_fit_counters(pre,counters):
    """Keep a single copy of each shared field only when its values agree."""
    for key in sorted(pre.keys() & counters.keys()):
        require(type(pre[key]) is type(counters[key]) and pre[key]==counters[key],
                "planned and observed counter disagree: "+key)
    return {**pre,**counters}


def stable_tensor_hash(state,include):
    h=sha256()
    for name in sorted(state):
        if include(name):
            t=state[name].detach().cpu().contiguous()
            h.update(name.encode());h.update(str((list(t.shape),str(t.dtype))).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def checkpoint_packet(state,plan,tok,train_sha256,vocab_sha256,schedule_sha256,updates):
    return dict(format=1,plan=asdict(plan),tokenizer_words=list(tok.words),
                vocab_size=tok.size,train_sha256=train_sha256,vocab_sha256=vocab_sha256,
                schedule_sha256=schedule_sha256,updates=updates,
                state_dict={name:value.detach().cpu().clone() for name,value in state.items()})


def execute(train,vocab,out,train_sha256,vocab_sha256,plan):
    plan.validate()
    require(digest(train)==train_sha256 and digest(vocab)==vocab_sha256,"frozen TRAIN/VOCAB digest differs")
    require(sys.version_info[:2]==(3,12) and torch.__version__=="2.10.0+cpu","unsupported frozen CPU runtime")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(torch.get_num_threads()==torch.get_num_interop_threads()==1,"runtime thread cap refused")
    started=time.perf_counter(); start_cpu=time.process_time()
    families=read_train(train)
    require(len(families)==320,"wrong consented TRAIN family count")
    tok=read_vocab(vocab,families)
    rows=encode(families,tok,plan.max_tokens);order=order_for(len(rows),plan)
    pre=metadata(rows,plan,order,tok)
    out.mkdir(parents=True,exist_ok=True)
    # Signed native envelope supplies one-attempt control; this guard also
    # prevents direct accidental reuse of an existing output path.
    write(out/"attempt.json",{"plan":asdict(plan),"train_sha256":train_sha256,
        "vocab_sha256":vocab_sha256,**pre})
    m=build_decoder(tok.size,plan.arm,width=plan.width,layers=plan.layers,
                    block=plan.block,seed=plan.seed,route_mode=plan.route_mode,retained_blocks=plan.retained_blocks)
    require(all(p.device.type=="cpu" and p.dtype==torch.float32 for p in m.parameters()),"non-FP32 model")
    init=m.state_dict()
    common=stable_tensor_hash(init,lambda k:".summary." not in k and ".route." not in k)
    summary_init=stable_tensor_hash(init,lambda k:".summary." in k) if plan.arm in ("B","C") else None
    all_init=stable_tensor_hash(init,lambda _:True); del init
    named=dict(m.named_parameters())
    total=sum(p.numel() for p in named.values()); trainable=sum(p.numel() for p in named.values() if p.requires_grad)
    opt=torch.optim.AdamW(m.parameters(),lr=plan.lr,betas=(.9,.999),eps=1e-8,weight_decay=plan.weight_decay)
    m.train(); trail=[];active=set();nonzero=set()
    tokens=answers=prefix=updates=dense_tokens=0
    for step,u in enumerate(order,1):
        opt.zero_grad(set_to_none=True)
        for pg in opt.param_groups:pg["lr"]=plan.lr*min(1,step/plan.warmup)
        x,length,y,pfx=batch_inputs(rows,u)
        pred,_=m(x,state=None)
        require(pred.shape==(*x.shape,tok.size),"decoder output geometry drift")
        tokens+=int(length.sum());dense_tokens+=x.numel()
        loss,tail=paired_loss(pred,length,y,lm_targets=pfx,alpha=plan.alpha)
        require(bool(torch.isfinite(loss)),"non-finite loss; fail without skipping")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),plan.clip,error_if_nonfinite=True)
        for k,p in named.items():
            if p.grad is not None:
                active.add(k)
                if k not in nonzero and bool(p.grad.ne(0).any()):nonzero.add(k)
        opt.step()
        updates+=1
        n1=y.numel();n0=int(pfx.ne(-1).sum())
        answers+=n1;prefix+=n0
        with torch.no_grad():
            lm=float(F.cross_entropy(pred.detach().flatten(0,1),pfx.flatten(),ignore_index=-1))
            out_ce=float(F.cross_entropy(tail.detach(),y.flatten()))
            right=tail.detach().gather(1,y.flatten()[:,None])
            foil=tail.detach().gather(1,y.flip((1,)).flatten()[:,None])
            margin=float((right-foil).mean())
            pair=int(tail.detach().argmax(1).reshape_as(y).eq(y).all(1).sum())
        trail.append(dict(step=step,objective=float(loss.detach()),prefix_ce_nats_per_target=lm,
                          answer_ce_nats_per_target=out_ce,answer_foil_margin=margin,
                          prefix_targets=n0,answer_targets=n1,train_complete_pairs=pair,
                          raw_grad_norm=float(norm),lr=opt.param_groups[0]["lr"]))
    require(updates==plan.steps and tokens==pre["planned_unpadded_tokens"] and dense_tokens==pre["planned_dense_input_tokens"] and
            answers==pre["planned_answer_targets"] and prefix==pre["planned_prefix_targets"],
            "executed work differs from predeclared schedule")
    require(all(bool(torch.isfinite(p).all()) for p in named.values()),"non-finite final parameters")
    checkpoint=checkpoint_packet(m.state_dict(),plan,tok,train_sha256,vocab_sha256,
                                 pre["schedule_sha256"],updates)
    with (out/"checkpoint.pt").open('xb') as f:torch.save(checkpoint,f)
    counters=dict(committed_updates=updates,train_families=len(rows),worlds_presented=answers,
        unpadded_input_tokens=tokens,dense_input_tokens=dense_tokens,prefix_targets=prefix,answer_targets=answers,
        model_input_padding=dense_tokens-tokens,padding_targets=0,protected_test_inputs=0,dev_inputs=0,
        total_parameters=total,trainable_parameters=trainable,
        gradient_path_parameters=sum(named[k].numel() for k in active),
        nonzero_gradient_parameters=sum(named[k].numel() for k in nonzero))
    summary=dict(plan=asdict(plan),**merge_fit_counters(pre,counters),common_initial_sha256=common,
        summary_initial_sha256=summary_init,all_initial_sha256=all_init,
        train_sha256=train_sha256,vocab_sha256=vocab_sha256,checkpoint_sha256=digest(out/"checkpoint.pt"),
        torch_version=torch.__version__,threads=torch.get_num_threads(),dtype="float32",
        gradient_parameter_names=sorted(active),nonzero_gradient_parameter_names=sorted(nonzero),
        skipped_updates=0,elapsed_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-start_cpu,
        mean_prefix_ce=sum(z["prefix_ce_nats_per_target"]*z["prefix_targets"] for z in trail)/prefix,
        mean_answer_ce=sum(z["answer_ce_nats_per_target"]*z["answer_targets"] for z in trail)/answers,curve=trail)
    write(out/"fit.json",summary)
    write(out/"native-counters.json",{"observations":counters,"metrics":{"updates":updates}})


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("fit","preflight"))
    for k in ("train","vocab","out"):p.add_argument("--"+k,type=Path,required=True)
    for k in ("train-sha256","vocab-sha256"):p.add_argument("--"+k,required=True)
    p.add_argument("--arm",choices=("A","B","C"),required=True)
    p.add_argument("--route-mode",choices=("learned","content_blind"),default="learned")
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--steps",type=int,default=128)
    p.add_argument("--worlds-per-step",type=int,default=8)
    p.add_argument("--max-tokens",type=int,default=512)
    p.add_argument("--alpha",type=float,default=1.)
    a=p.parse_args()
    plan=Plan(a.arm,a.route_mode,a.seed,a.steps,a.worlds_per_step,a.max_tokens,alpha=a.alpha).validate()
    if a.phase=="preflight":
        require(digest(a.train)==a.train_sha256 and digest(a.vocab)==a.vocab_sha256,"preflight digest differs")
        families=read_train(a.train);tok=read_vocab(a.vocab,families)
        rows=encode(families,tok,plan.max_tokens);order=order_for(len(rows),plan)
        print(json.dumps({"model_initializations":0,"optimizer_steps":0,**metadata(rows,plan,order,tok)},sort_keys=True))
    else:execute(a.train,a.vocab,a.out,a.train_sha256,a.vocab_sha256,plan)


if __name__=="__main__":
    cli()
