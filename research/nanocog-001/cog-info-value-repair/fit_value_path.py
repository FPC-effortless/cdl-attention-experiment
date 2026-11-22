"""Exact native TRAIN worker, inert on import; requires separately reviewed closure."""
from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import file_digest, sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import time

BASE='517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'
OLD_TRAIN='02607c19e760e164989865a54422eb7cd78fed7d117a7fa2ff6c4df9d700f9f8'
BASE_TENSOR='1a4558037b5bfb046e8fd1a7a3c4265ce35222eda95fb737ba2e40839927d635'
SEEDS=(2071,2171,2271)
ARMS=('tap_content','literal_lag')
ENDPOINTS=(128,1280)
KEYS={'first.weight','last.weight'}

def need(ok,msg):
    if not ok:
        raise ValueError(msg)

def hash_file(p):
    with p.open("rb") as f:return file_digest(f,"sha256").hexdigest()

def wire(x):
    return (json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()

def save(p,x):
    with p.open('xb') as f: f.write(wire(x))

@dataclass(frozen=True)
class Plan:
    arm:str
    seed:int
    updates:int=1280
    endpoints:tuple=ENDPOINTS
    lr:float=.001
    warmup:int=8
    betas:tuple=(.9,.999)
    eps:float=1e-8
    weight_decay:float=.01
    clip:float=1.
    edit_weight:float=.25
    prefix_weight:float=0.

def exact(a,b):
    import torch
    if isinstance(a,torch.Tensor):
        return type(b) is type(a) and a.dtype==b.dtype and a.shape==b.shape and \
            torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    if type(a) is not type(b):return False
    if is_dataclass(a):return all(exact(getattr(a,f.name),getattr(b,f.name)) for f in fields(a))
    if type(a) is tuple:return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b,strict=True))
    if type(a) in (type(None), int):return a==b
    raise ValueError('unknown native cache type')

def main():
    from trainlab.research_adequacy import recipe_identity
    need(not sys.flags.optimize and sys.version_info[:2]==(3,12) and len(sys.argv)==3,'wrong worker runtime')
    plan=Plan(sys.argv[1],int(sys.argv[2]))
    root=Path.cwd()
    out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    need(plan.arm in ARMS and plan.seed in SEEDS and os.environ.get('TRAINLAB_SEED')==str(plan.seed)
        and out.is_dir() and not out.is_symlink() and not any(out.iterdir()),'wrong exact native invocation')
    need(all(os.environ.get(n)=='1' for n in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')),'use pinned one-thread dispatch')
    cfg=json.loads((root/'recipe.json').read_bytes())
    ad=json.loads((root/'adequacy.json').read_bytes())
    adsha=hash_file(root/'adequacy.json')
    require_files=set(ad['inputs'])|{'adequacy.json'}
    need(set(cfg['inputs'])==require_files and cfg['command']==['python','fit_value_path.py',plan.arm,str(plan.seed)]
         and cfg['seed']==plan.seed and cfg['timeout_seconds']==180 and cfg['budget']=={'model_forwards':1283,'optimizer_updates':1280}
         and cfg['research_adequacy']=={'file':'adequacy.json','sha256':adsha}
         and ad['recipe_sha256']==recipe_identity({k:v for k,v in cfg.items() if k!='research_adequacy'})
         and all((root/k).is_file() and not (root/k).is_symlink() and hash_file(root/k)==v for k,v in ad['inputs'].items()),
        'native closure integrity')
    pre=json.loads((root/'preflight.json').read_bytes())
    rt=json.loads((root/'fit_runtime.json').read_bytes())
    need(rt['arm']==plan.arm and rt['seed']==plan.seed and rt['witness_sha256']==ad['inputs']['train_witness.json']
      and pre['train_only_witness_verified']==1
      and pre['native_noop_reference_only']==1,'frozen invocation/window mismatch')
    result=execute(plan,root,out,rt)
    save(out/'observations.json',dict(schema='trainlab.execution-adequacy-observations.v1',
       recipe_sha256=ad['recipe_sha256'],inputs={**ad['inputs'],'adequacy.json':adsha},
       producer={'file':'fit_value_path.py','sha256':ad['inputs']['fit_value_path.py']},observations=result))

def execute(plan,root,out,pre):
    import torch
    import cog_comm.fit as oldfit
    import cog_comm.role_mixers as role
    import nanocog.model as native
    import frozen_value_port as port
    import value_path_residual as residual
    from value_fit_intake import parse,objective,PIN,SLOTS
    need(torch.__version__=='2.10.0+cpu' and hash_file(Path(sys.executable).resolve())==pre['python_sha256']
      and hash_file(Path(torch.__file__).parent/'lib/libtorch_cpu.so')==pre['torch_cpu_sha256'],'pinned CPU runtime')
    need(all(Path(m.__file__).is_relative_to(root) for m in (oldfit,role,native,port,residual)),'imports outside native closure')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter();cpu=time.process_time()
    raw=(root/'checkpoint.pt').read_bytes()
    need(sha256(raw).hexdigest()==BASE,'wrong frozen backbone bytes')
    ck=torch.load(BytesIO(raw),map_location='cpu',weights_only=True)
    need(ck['schema']=='cog-comm-role-001.checkpoint.v1' and ck['seed']==1371 and ck['updates']==1280
       and ck['vocab_size']==287 and ck['train_sha256']==OLD_TRAIN and ck['plan']['arm']=='gated32'
       and ck['plan']['clip']==1. and ck['plan']['prefix_ce_weight']==.25,'checkpoint provenance')
    words=tuple(ck['tokenizer_words']); schedule,counts=parse(plan.seed,words,root)
    bb=role.build_arm(287,'gated32',1371).eval().requires_grad_(False)
    bb.load_state_dict(ck['state_dict'],strict=True)
    del ck,raw
    state_sha=lambda m:oldfit.stable_tensor_hash(m.state_dict(),lambda _:True)
    base_state=state_sha(bb)
    need(base_state==BASE_TENSOR and sum(p.numel() for p in bb.parameters())==37414 and
        all(p.dtype==torch.float32 and bool(torch.isfinite(p).all()) for p in bb.parameters()),'frozen tensor lineage')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan.seed)
        adap=residual.ValuePathResidual(arm=plan.arm)
    model=port.FrozenValueRepairPort(bb,adap).train()
    a0=state_sha(adap); params=dict(adap.named_parameters())
    need(set(params)==KEYS and {k for k,p in model.named_parameters() if p.requires_grad}==
         {f'value_residual.{k}' for k in KEYS} and sum(p.numel() for p in params.values())==800
         and bool(adap.last.weight.eq(0).all()),'matched trainable geometry')
    opt=torch.optim.AdamW(params.values(),lr=plan.lr,betas=plan.betas,eps=plan.eps,
                          weight_decay=plan.weight_decay,foreach=False,fused=False)
    meta=dict(plan=asdict(plan),base_checkpoint_sha256=BASE,base_state_sha256=base_state,
              train_manifest_sha256=PIN['train_manifest.json'],schedule_sha256=SLOTS[plan.seed],
              initial_adapter_state_sha256=a0,vocab_size=287,torch=torch.__version__,python=sys.version,
              torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
              claim_eligibility='NOT_QUALIFIED',witness_sha256=pre['witness_sha256'])
    save(out/'attempt.json',dict(meta,status='STARTED'))
    fixture=torch.tensor(schedule[0][2],dtype=torch.long)
    with torch.no_grad(): original,cache=model(fixture,enabled=False)
    need(original.shape==(8,7,287) and bool(torch.isfinite(original).all()),'native fixture shape')
    c=Counter(model_forwards=1,integrity_forwards=1); e_seen=Counter();o_seen=Counter(); ever=set()
    endpoints={}
    with (out/'steps.jsonl').open('xb') as log:
        for i,(ed,ord_,xx,yy) in enumerate(schedule,1):
            opt.zero_grad(set_to_none=True)
            lr=plan.lr*min(1.,i/plan.warmup)
            for group in opt.param_groups:group['lr']=lr
            x=torch.tensor(xx,dtype=torch.long); y=torch.tensor(yy,dtype=torch.long)
            z,state=model(x,enabled=True)
            need(z.shape==(8,7,287) and z.dtype==torch.float32 and bool(torch.isfinite(z).all()),'finite trained logits')
            if i==1:need(exact(z,original) and exact(state,cache),'exact native warm-start logits/cache')
            loss,ce,aux=objective(z[:,-1],y,len(ed));loss.backward()
            grads={k:p.grad for k,p in params.items()}
            need(all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
              and all(p.grad is None and not p.requires_grad for p in bb.parameters()),'adapter-only finite backward')
            nonzero={k:int(g.ne(0).sum()) for k,g in grads.items()}
            if i==1:need(nonzero['first.weight']==0 and nonzero['last.weight']>0,'zero-start gradient geometry')
            ever.update(k for k,v in nonzero.items() if v)
            norm=float(torch.nn.utils.clip_grad_norm_(params.values(),plan.clip,error_if_nonfinite=True))
            opt.step()
            actual,ce_,aux_=(float(v.detach()) for v in (loss,ce,aux))
            need(all(math.isfinite(v) for v in (actual,ce_,aux_,norm)) and aux_>=0 and (bool(ed) or aux_==0)
                 and abs(actual-ce_-.25*aux_)<3e-6,'objective trace')
            e_seen.update(ed);o_seen.update(ord_)
            c.update(committed_updates=1,training_forwards=1,model_forwards=1,backwards=1,answer_targets=8,
               train_worlds=8,train_input_tokens=56,mixed_batches=int(bool(ed)),ordinary_only_batches=int(not ed),
               edit_triple_presentations=len(ed),edit_rows=3*len(ed),ordinary_rows=len(ord_),edit_loss_terms=2*len(ed))
            log.write(wire(dict(step=i,lr=lr,edit_slots=ed,ordinary_slots=ord_,objective=actual,
                answer_ce_nats=ce_,edit_margin_softplus=aux_,raw_gradient_norm=norm,clip_active=norm>plan.clip,
                nonzero_gradient_elements=nonzero)))
            if i in ENDPOINTS:
                need(state_sha(bb)==base_state and all(int(opt.state[p]['step'].item())==i for p in params.values()),'endpoint unchanged base/optimizer')
                with torch.no_grad(): zs,ss=model(fixture,enabled=True)
                c.update(model_forwards=1,integrity_forwards=1)
                need(bool(torch.isfinite(zs).all()) and exact(zs[:,:6],original[:,:6]) and exact(ss,cache),
                     'earlier logits/cache changed')
                for k,v in counts[str(i)].items():
                    need(v==c['train_worlds' if k=='total_world_rows' else k],'endpoint actual exposure')
                weights={k:v.detach().cpu().clone() for k,v in adap.state_dict().items()}
                hs=state_sha(adap)
                need(set(weights)==KEYS and hs!=a0 and all(bool(torch.isfinite(w).all()) for w in weights.values()),'adapter-only finite endpoint')
                path=out/f'adapter-{i}.pt'
                with path.open('xb') as f:
                    torch.save(dict(meta,schema='cog-info-value-repair.adapter.v1',arm=plan.arm,
                      seed=plan.seed,updates=i,adapter_state_sha256=hs,tokenizer_words=list(words),state_dict=weights),f)
                endpoints[str(i)]=dict(sha256=hash_file(path),adapter_state_sha256=hs,
                     counters=dict(c),distinct_edit_slots=len(e_seen),distinct_ordinary_slots=len(o_seen))
    need(e_seen==Counter({k:5 for k in range(256)}) and o_seen==Counter({k:10 for k in range(640)})
      and c['model_forwards']==1283 and c['integrity_forwards']==3 and ever==KEYS and len(endpoints)==2,'exact finished exposure')
    final=dict(**c,unique_edit_triples=len(e_seen),unique_ordinary_rows=len(o_seen),
      trainable_parameters=800,frozen_parameters=37414,endpoint_artifacts=2,unchanged_base_checks=2,
      initialized_native_logits_cache_parity=1,earlier_logits_cache_endpoint_checks=2,nonzero_gradient_tensors=len(ever),
      skipped_updates=0,protected_inputs=0,prefix_targets=0,selected_checkpoints=0,trace_rows=1280)
    save(out/'fit.json',dict(meta,status='COMPLETE',checkpoints=endpoints,counters=final,
      trace_sha256=hash_file(out/'steps.jsonl'),elapsed_seconds=time.perf_counter()-start,
      process_cpu_seconds=time.process_time()-cpu))
    save(out/'metrics.json',{'committed_updates':1280,'model_forwards':1283})
    return final

if __name__=='__main__':main()
