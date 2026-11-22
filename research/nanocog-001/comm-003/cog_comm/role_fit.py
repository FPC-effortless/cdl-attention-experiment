"""Draft TRAIN-only COG-COMM-ROLE-001 baseline worker; no import-time execution."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
import inspect
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import torch
from torch.nn import functional as F

from .fit import stable_tensor_hash
from .objective import Tokenizer, paired_loss
from .role_control import CELLS, COLORS, NAMES, oracle, world
from .role_mixers import ARMS, build_arm

SCHEMA="cog-comm-role-001.train.v1"
CHECKPOINT_SCHEMA="cog-comm-role-001.checkpoint.v1"
SEEDS=(1171,1271,1371)
N_FAMILIES=320
ENDPOINTS=(128,1280)
PARAMETERS={"gated32":37414,"static32":37158,"attention32":37414,"gated48":69926}
ORDER=tuple(i for j in range(4) for i in (j,j+4))
HASH_RE=re.compile(r"[0-9a-f]{64}\Z")
MAX_TRAIN_BYTES=2*1024*1024

@dataclass(frozen=True)
class Plan:
    seed:int
    arm:str="gated32"
    width:int=32
    layers:int=2
    steps:int=1280
    saves:tuple[int,int]=ENDPOINTS
    worlds_per_update:int=8
    prompt_tokens:int=7
    lr:float=.001
    warmup:int=8
    betas:tuple[float,float]=(.9,.999)
    eps:float=1e-8
    weight_decay:float=.01
    clip:float=2.
    alpha:float=1.
    prefix_ce_weight:float=.25
    schedule_seed_xor:int=0xC060001

def require(ok,why):
    if not ok:raise ValueError(why)

def digest(path):
    with Path(path).open('rb') as f:return file_digest(f,'sha256').hexdigest()

def strict_json(data):
    def pairs(items):
        result=dict(items)
        require(len(result)==len(items),"duplicate JSON property")
        return result
    def const(name):
        raise ValueError("nonfinite JSON "+name)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=const)

def json_bytes(x):
    return (json.dumps(x,sort_keys=True,allow_nan=False,separators=(",",":"))+"\n").encode()

def write(path,x):
    with path.open('xb') as f:f.write(json_bytes(x))

def load_train(path,expected_sha):
    """Deserialize one admitted TRAIN artifact, no generator, CONTROL or held lookup."""
    require(HASH_RE.fullmatch(expected_sha),"expected an exact lowercase content digest")
    require(path.is_file() and not path.is_symlink(),"regular admitted input only")
    with path.open('rb') as f:
        data=f.read(MAX_TRAIN_BYTES+1)
    require(len(data)<=MAX_TRAIN_BYTES and sha256(data).hexdigest()==expected_sha,"TRAIN artifact bytes differ")
    doc=strict_json(data)
    require(type(doc) is dict and set(doc)=={"schema","role","families"}
            and doc["schema"]==SCHEMA and doc["role"]=="TRAIN",
            "only exact TRAIN artifact schema/role accepted")
    fs=doc["families"]
    require(type(fs) is list and len(fs)==N_FAMILIES,"exact 320 TRAIN families required")
    seen=set(); total=set(); prompts=[]
    for row in fs:
        require(type(row) is dict and set(row)=={"prompts","answers"},"malformed TRAIN family")
        pp,aa=row["prompts"],row["answers"]
        require(type(pp) is list and type(aa) is list and len(pp)==len(aa)==8,
                "exact eight ordered worlds and answers per family required")
        require(all(type(q) is str and len(q.split())==7 and q==" ".join(q.split()) for q in pp)
                and all(type(a) is str and a in COLORS for a in aa),"bad raw prompt/target")
        p=tuple(pp[0].split())
        names=tuple(sorted((p[0],p[3]),key=NAMES.index))
        colors=tuple(sorted((p[2],p[5]),key=COLORS.index))
        key=(names,colors)
        require(len(set(names))==len(set(colors))==2 and key not in seen,"duplicate or malformed family")
        seen.add(key)
        for prompt,y,cell in zip(pp,aa,CELLS,strict=True):
            obs=tuple(prompt.split())
            require(obs==world(key,cell) and y==oracle(obs),"family factorial order or answer differs")
            require(prompt not in total,"duplicate rank-expanded TRAIN prompt")
            total.add(prompt);prompts.append(prompt)
    require(len(total)==2560,"wrong TRAIN world support")
    tok=Tokenizer.from_train(prompts,COLORS,minimum=5)
    require(tok.size==287 and set(tok.words)==set(NAMES)|set(COLORS)|{"is"},"unexpected TRAIN-derived vocabulary")
    rows=[]
    for row in fs:
        ids=[]; labels=[]
        for k in ORDER:
            p,y=row["prompts"][k],row["answers"][k]
            e=tok.encode(p)
            require(e[0]==1 and e[-1]==5 and len(e)==9 and min(e[1:-1])>=262,
                    "prompt did not tokenize into exactly seven visible word IDs")
            ids.append(e[1:-1])
            labels.append(tok.index[y])
        require(all(labels[i]!=labels[i+1] for i in range(0,8,2)),"s-flip foil must differ")
        rows.append((ids,labels))
    return tok,rows

def schedule(plan):
    # One shuffled permutation of all families per epoch; no checkpoint reset.
    rng=random.Random(plan.seed ^ plan.schedule_seed_xor)
    order=[]
    for _ in range(plan.steps//N_FAMILIES):
        epoch=list(range(N_FAMILIES))
        rng.shuffle(epoch)
        order.extend(epoch)
    require(len(order)==plan.steps and tuple(ORDER[i+1]-ORDER[i] for i in range(0,8,2))==(4,4,4,4),
            "paired schedule geometry")
    return order

def tensor_batch(row):
    items,labels=row
    x=torch.tensor(items,dtype=torch.long,device='cpu')
    y=torch.tensor(labels,dtype=torch.long,device='cpu').reshape(4,2)
    length=torch.full((x.shape[0],),x.shape[1],dtype=torch.long)
    prefix=torch.full_like(x,-1)
    prefix[:,:-1]=x[:,1:]
    return x,length,y,prefix

def save_torch(path,packet):
    with path.open('xb') as f:torch.save(packet,f)

def artifacts(out,plan,model,opt,tok,ident,count,hashes,step,schedule_sha,init_sha):
    require(step in plan.saves and count['committed_updates']==step,"endpoint outside exact trajectory")
    require((count['answer_targets'],count['prefix_targets'],count['unpadded_input_tokens'])==
            (8*step,48*step,56*step),"endpoint actual token/label counts disagree")
    require(all(bool(torch.isfinite(x).all()) for x in model.parameters()),"nonfinite endpoint")
    for st in opt.state.values():
        require(int(st['step'].item())==step,"optimizer step disagrees with counter")
    model_state={k:p.detach().cpu().clone() for k,p in model.state_dict().items()}
    packet={'schema':CHECKPOINT_SCHEMA,'format':1,'plan':asdict(plan),'seed':plan.seed,
            'tokenizer_words':list(tok.words),'vocab_size':tok.size,'train_sha256':ident,
            'vocab_sha256':hashes['vocab_sha256'],'schedule_sha256':schedule_sha,
            'all_initial_sha256':init_sha,'updates':step,'counts':dict(count),'state_dict':model_state}
    # Model-only file for future actor; optimizer/RNG state separately private.
    ck=out/f"checkpoint-{step}.pt";save_torch(ck,packet)
    resume=out/f"resume-{step}.pt"
    save_torch(resume,{'schema':'cog-comm-role-001.resume.v1','updates':step,
                       'checkpoint_sha256':digest(ck),'schedule_sha256':schedule_sha,
                       'optimizer_state_dict':opt.state_dict(),'torch_cpu_rng_state':torch.random.get_rng_state()})
    return {'checkpoint':ck.name,'checkpoint_sha256':digest(ck),'resume':resume.name,
            'resume_sha256':digest(resume),'state_dict_sha256':stable_tensor_hash(model_state,lambda _:True),
            'counters':dict(count)}

def execute(train,train_sha256,seed,out,arm='gated32'):
    require(seed in SEEDS and arm=="gated32" and not sys.flags.optimize,"ROLE-003 permits only gated32 and registered seeds without optimized assertions")
    require(sys.version_info[:2]==(3,12) and torch.__version__=="2.10.0+cpu","unexpected CPU runtime")
    require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()),"fresh precreated output directory required")
    require(os.environ.get("TRAINLAB_SEED")==str(seed) and os.environ.get("TRAINLAB_OUTPUT_DIR")
            and Path(os.environ["TRAINLAB_OUTPUT_DIR"]).resolve()==out.resolve(),
            "native exact-scoped seed/output binding required")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(torch.get_num_threads()==torch.get_num_interop_threads()==1,"failed CPU thread pin")
    torch.manual_seed(seed)
    start=time.perf_counter();cpu=time.process_time()
    plan=Plan(seed,arm,width=48 if arm=="gated48" else 32)
    tok,rows=load_train(train,train_sha256)
    order=schedule(plan)
    order_packet={"family_ordinals":order,"paired_cell_ordinals":list(ORDER),
                  "mode":"four independently shuffled full epochs","rng_seed":seed ^ plan.schedule_seed_xor}
    schedule_bytes=json_bytes(order_packet); schedule_sha=sha256(schedule_bytes).hexdigest()
    vocab={"words":list(tok.words),"size":tok.size,"minimum":5}
    vocab_sha=sha256(json_bytes(vocab)).hexdigest()
    hashes={"vocab_sha256":vocab_sha,
            "worker_sha256":digest(__file__),
            "arm_source_sha256":digest(inspect.getsourcefile(build_arm)),
            "objective_source_sha256":digest(inspect.getsourcefile(paired_loss)),
            "role_control_source_sha256":digest(inspect.getsourcefile(world))}
    write(out/'attempt.json',{'status':'STARTED','claim_eligibility':'NOT_QUALIFIED','plan':asdict(plan),
                              'train_sha256':train_sha256,'schedule_sha256':schedule_sha,'sources':hashes,
                              'runtime':{'python':sys.version,'executable_sha256':digest(Path(sys.executable).resolve()),
                                         'torch':torch.__version__,'threads':1,'dtype':'float32',
                                         'deterministic':True}})
    write(out/'schedule.json',order_packet)
    write(out/'vocabulary.json',vocab)
    model=build_arm(tok.size,arm,seed).train()
    params=dict(model.named_parameters())
    require(all(p.dtype==torch.float32 and p.device.type=="cpu" for p in params.values()),
            "candidate tensor geometry changed")
    par=sum(p.numel() for p in params.values())
    require(par==PARAMETERS[arm],"candidate parameter count changed")
    init_sha=stable_tensor_hash(model.state_dict(),lambda _:True)
    trunk_sha=stable_tensor_hash(model.state_dict(),lambda name:not name.startswith("mixers."))
    opt=torch.optim.AdamW(params.values(),lr=plan.lr,betas=plan.betas,eps=plan.eps,
                          weight_decay=plan.weight_decay,foreach=False,fused=False)
    counters=Counter();res={};seen=set();gradient_path=set();nonzero=set()
    totals=Counter()
    with (out/'steps.jsonl').open('xb') as log:
        for step,family in enumerate(order,1):
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:
                group['lr']=plan.lr*min(1,step/plan.warmup)
            x,length,y,pfx=tensor_batch(rows[family])
            z,_=model(x,state=None)
            require(z.shape==(8,7,tok.size),"actual logits drifted")
            loss,tail=paired_loss(z,length,y,lm_targets=pfx,alpha=plan.alpha,
                                  prefix_ce_weight=plan.prefix_ce_weight)
            require(bool(torch.isfinite(loss)),"nonfinite objective; no update skipped or retried")
            loss.backward()
            raw_norm=torch.nn.utils.clip_grad_norm_(params.values(),plan.clip,error_if_nonfinite=True)
            live={k for k,p in params.items() if p.grad is not None}
            active={k for k in live if bool(params[k].grad.ne(0).any())}
            gradient_path|=live;nonzero|=active
            opt.step()
            counters.update(committed_updates=1,model_forwards=1,backwards=1,worlds_presented=y.numel(),
                            unpadded_input_tokens=x.numel(),dense_input_tokens=x.numel(),
                            answer_targets=y.numel(),prefix_targets=int(pfx.ne(-1).sum()))
            seen.add(family)
            with torch.no_grad():
                lm=float(F.cross_entropy(z.detach().flatten(0,1),pfx.flatten(),ignore_index=-1))
                answer=float(F.cross_entropy(tail.detach(),y.flatten()))
                right=tail.detach().gather(1,y.flatten()[:,None]).squeeze(-1)
                foil=tail.detach().gather(1,y.flip(1).flatten()[:,None]).squeeze(-1)
                contrast=float(F.softplus(foil-right).mean())
                chosen=tail.detach().clone();chosen.scatter_(1,y.flatten()[:,None],-torch.inf)
                correct=right>chosen.max(-1).values
            term={"objective":float(loss.detach()),"prefix_ce_nats_per_target":lm,
                  "answer_ce_nats_per_target":answer,"paired_foil_softplus":contrast,
                  "answer_minus_foil_mean":float((right-foil).mean())}
            require(abs(term["objective"]-(plan.prefix_ce_weight*lm+answer+plan.alpha*contrast))<3e-6,"loss accounting drift")
            totals.update(term)
            observation={**term,"step":step,"family_ordinal":family,"model_forwards":1,
                          "raw_gradient_norm":float(raw_norm),"gradient_path_tensors":len(live),
                          "nonzero_gradient_tensors":len(active),"lr":opt.param_groups[0]['lr'],
                          "worlds":int(y.numel()),"answer_targets":int(y.numel()),
                          "prefix_targets":int(pfx.ne(-1).sum()),
                          "strict_correct_worlds":int(correct.sum()),
                          "strict_both_correct_s_flip_pairs":int(correct.reshape(4,2).all(-1).sum()),
                          "strict_complete_family":int(correct.all())}
            log.write(json_bytes(observation));log.flush()
            if step in ENDPOINTS:
                ep=artifacts(out,plan,model,opt,tok,train_sha256,counters,hashes,step,schedule_sha,init_sha)
                ep['unique_train_families_observed']=len(seen)
                ep['mean_train_loss_terms']={k:v/step for k,v in totals.items()}
                res[str(step)]=ep
    require(list(res)==["128","1280"] and len(seen)==320 and Counter(order)==Counter({k:4 for k in range(320)}),
            "trajectory or complete epoch mismatch")
    c={**counters,'train_families':len(rows),'unique_train_families_observed':len(seen),
       'model_input_padding':0,'padding_targets':0,'skipped_updates':0,'total_parameters':par,
       'trainable_parameters':sum(p.numel() for p in params.values() if p.requires_grad),
       'gradient_path_parameters':sum(params[n].numel() for n in gradient_path),
       'nonzero_gradient_parameters':sum(params[n].numel() for n in nonzero),
       'protected_test_inputs':0,'control_inputs':0,'historical_split_inputs':0}
    write(out/'fit.json',{'scope':'ROLE-003 constructed TRAIN clip-threshold diagnostic only','claim_eligibility':'NOT_QUALIFIED',
                         'plan':asdict(plan),'train_sha256':train_sha256,'sources':hashes,'counters':c,
                         'schedule_sha256':schedule_sha,'trace_sha256':digest(out/'steps.jsonl'),
                         'all_initial_sha256':init_sha,'trunk_initial_sha256':trunk_sha,'endpoints':res,'gradient_path_names':sorted(gradient_path),
                         'nonzero_gradient_names':sorted(nonzero),'elapsed_seconds':time.perf_counter()-start,
                         'process_cpu_seconds':time.process_time()-cpu})
    metrics={'committed_updates':c['committed_updates']}
    write(out/'metrics.json',metrics)
    write(out/'native-counters.json',{'observations':c,'metrics':metrics})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--train-sha256',required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--arm',choices=('gated32',),default='gated32')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    out=a.out or (Path(os.environ["TRAINLAB_OUTPUT_DIR"]) if "TRAINLAB_OUTPUT_DIR" in os.environ else None)
    require(out is not None,"native output directory is required")
    execute(a.train,a.train_sha256,a.seed,out,a.arm)

if __name__=='__main__':main()
