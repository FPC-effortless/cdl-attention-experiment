"""Draft exact NAME-CONSIST-001 TRAIN-only adapter native fit; no import-time execution."""
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import math,os,sys,time
from pathlib import Path
import torch
from cog_comm.fit import stable_tensor_hash
from cog_comm.role_fit import load_train,digest,json_bytes,strict_json,write
from consistency_objective import answer_foil_consistency
from schedule_intake import parse_schedule,PINS,COEFF
from cog_comm.role_mixers import build_arm
from frozen_gate_adapter import FrozenGateAdapter
from tap_gate_residual import TapGateResidual

TRAIN='02607c19e760e164989865a54422eb7cd78fed7d117a7fa2ff6c4df9d700f9f8'
BASE='517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'
BASE_SEED=1371
SEEDS=tuple(PINS)
ENDPOINTS=(128,1280)
ARMS=tuple(COEFF)
STATE_KEYS={'first.weight','last.weight'}
def need(ok,msg):
    if not ok:raise ValueError(msg)

@dataclass(frozen=True)
class Plan:
    seed:int
    arm:str
    backbone_seed:int=BASE_SEED
    steps:int=1280
    saves:tuple=ENDPOINTS
    schedule_seed_xor:int=0xC060001
    lr:float=.001
    warmup:int=8
    betas:tuple=(.9,.999)
    eps:float=1e-8
    weight_decay:float=.01
    clip:float=1.
    alpha:float=1.
    prefix_ce_weight:float=0.
    @property
    def coefficient(self):return COEFF[self.arm]

def base_model(path):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==BASE,'unregistered frozen checkpoint')
    ck=torch.load(BytesIO(raw),map_location='cpu',weights_only=True)
    need(ck['schema']=='cog-comm-role-001.checkpoint.v1' and ck['seed']==BASE_SEED and ck['updates']==1280
        and ck['vocab_size']==287 and ck['train_sha256']==TRAIN and ck['plan']['arm']=='gated32'
        and ck['plan']['clip']==1. and ck['plan']['prefix_ce_weight']==.25,'frozen base provenance')
    m=build_arm(287,'gated32',BASE_SEED).eval().requires_grad_(False)
    m.load_state_dict(ck['state_dict'],strict=True)
    need(sum(p.numel() for p in m.parameters())==37414 and all(p.dtype==torch.float32 and p.device.type=='cpu'
         and bool(torch.isfinite(p).all()) for p in m.parameters()),'frozen backbone geometry')
    return m,ck

def exact_logits(a,b):
    return a.shape==b.shape and torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))

def state_hash(m):
    return stable_tensor_hash(m.state_dict(),lambda _:True)

def execute(plan,out):
    need(not sys.flags.optimize and sys.version_info[:2]==(3,12) and torch.__version__=='2.10.0+cpu',
         'wrong pinned runtime')
    need(plan.seed in SEEDS and plan.arm in ARMS and os.environ.get('TRAINLAB_SEED')==str(plan.seed)
         and str(out.resolve())==str(Path(os.environ['TRAINLAB_OUTPUT_DIR']).resolve()),'wrong exact execution')
    need(out.is_dir() and not out.is_symlink() and not any(out.iterdir()),'fresh native output required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter();cpu=time.process_time()
    tok,rows=load_train(Path('train.json'),TRAIN)
    backbone,ck=base_model(Path('checkpoint.pt'))
    need(tuple(ck['tokenizer_words'])==tok.words,'vocabulary lineage')
    schedule,exposure=parse_schedule(plan.seed,rows)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan.seed)
        adapter=TapGateResidual(arm='tap_compare')
    model=FrozenGateAdapter(backbone,adapter).train()
    params=dict(adapter.named_parameters())
    need(set(params)==STATE_KEYS and all(p.requires_grad for p in params.values())
         and sum(p.numel() for p in params.values())==552 and bool(adapter.last.weight.eq(0).all()),
         'adapter parameter/no-op geometry')
    b0=state_hash(backbone); a0=state_hash(adapter)
    need(b0==stable_tensor_hash(ck['state_dict'],lambda _:True),'checkpoint tensor mismatch')
    opt=torch.optim.AdamW(params.values(),lr=plan.lr,betas=plan.betas,eps=plan.eps,weight_decay=plan.weight_decay,
                          foreach=False,fused=False)
    sched_pin=exposure['source_sha256']
    write(out/'exposure.json',exposure)
    info=dict(status='STARTED',claim_eligibility='NOT_QUALIFIED',plan=asdict(plan),consistency_weight=plan.coefficient,backbone_sha256=BASE,
         backbone_state_sha256=b0,adapter_initial_sha256=a0,train_sha256=TRAIN,schedule_sha256=sched_pin,
         torch=torch.__version__,runtime_python=sys.version,executable_sha256=digest(Path(sys.executable).resolve()),
         intraop_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),dtype='float32')
    write(out/'attempt.json',info)
    fixture=torch.tensor(schedule[0][2],dtype=torch.long)
    with torch.no_grad():original=model(fixture,enabled=False)
    need(original.shape==(8,7,287) and bool(torch.isfinite(original).all()),'invalid frozen reference')
    counters=Counter(model_forwards=1,training_forwards=0,integrity_forwards=1)
    seen=set(); sums=Counter(); endpoints={}; ever_nonzero=set()
    with (out/'steps.jsonl').open('xb') as log:
        for i,(f,paired,xx,yy) in enumerate(schedule,1):
            opt.zero_grad(set_to_none=True)
            for gr in opt.param_groups:gr['lr']=plan.lr*min(1,i/plan.warmup)
            x=torch.tensor(xx,dtype=torch.long); y=torch.tensor(yy,dtype=torch.long)
            z=model(x); counters.update(model_forwards=1,training_forwards=1)
            need(z.shape==(8,7,287) and z.dtype==torch.float32 and bool(torch.isfinite(z).all()),'logit geometry')
            if i==1:need(exact_logits(z,original),'initial enabled parity')
            loss,terms=answer_foil_consistency(z[:,-1],y,paired=paired,coefficient=plan.coefficient)
            loss.backward()
            grad={n:p.grad for n,p in params.items()}
            need(all(g is not None and bool(torch.isfinite(g).all()) for g in grad.values())
              and all(p.grad is None and not p.requires_grad for p in backbone.parameters()),'gradient isolation')
            nonzero={n:int(g.ne(0).sum()) for n,g in grad.items()}
            if i==1:need(nonzero['last.weight']>0 and nonzero['first.weight']==0,'initial gradient path')
            ever_nonzero|={n for n,c in nonzero.items() if c}
            norm=float(torch.nn.utils.clip_grad_norm_(params.values(),plan.clip,error_if_nonfinite=True))
            need(math.isfinite(norm) and norm>0,'invalid raw grad norm')
            opt.step()
            a=float(terms['answer_ce'].detach())
            foil=float(terms['assignment_foil'].detach())
            match=float(terms['consistency'].detach())
            margin=terms['signed_margin'].detach()
            objective=float(loss.detach())
            need(math.isfinite(objective) and 0<=match<=1 and (paired or match==0)
              and abs(objective-a-foil-plan.coefficient*match)<3e-6,'objective or bound drift')
            sums.update(objective=objective,answer_ce=a,foil=foil,consistency=match,raw_gradient_norm=norm)
            counters.update(committed_updates=1,backwards=1,train_worlds=8,train_input_tokens=56,answer_targets=8,paired_foil_pairs=4,
                     paired_updates=int(paired),ordinary_updates=int(not paired),consistency_pairs=4*int(paired))
            seen.add(f)
            log.write(json_bytes(dict(step=i,family_ordinal=f,lr=opt.param_groups[0]['lr'],objective=objective,
               answer_ce_nats=a,paired_foil_softplus=foil,remap_consistency=match,consistency_weight=plan.coefficient,
               is_paired=int(paired),answer_minus_foil_mean=float(margin.mean()),
               raw_gradient_norm=norm,above_clip_threshold=norm>plan.clip,adapter_nonzero_gradient_counts=nonzero,
               train_worlds=8,train_input_tokens=56,answer_targets=8,paired_foil_pairs=4,prefix_targets=0)))
            if i in ENDPOINTS:
                require_frozen=state_hash(backbone)
                need(require_frozen==b0,'frozen base changed')
                with torch.no_grad():trace=model(fixture)
                counters.update(model_forwards=1,integrity_forwards=1)
                need(bool(torch.isfinite(trace).all()) and exact_logits(trace[:,:6],original[:,:6]),'earlier logits changed')
                need(all(int(t['step'].item())==i for t in opt.state.values()),'wrong optimizer step')
                ah=state_hash(adapter)
                need(ah!=a0 and all(bool(torch.isfinite(p).all()) for p in params.values()),'invalid learned adapter endpoint')
                path=out/f'adapter-{i}.pt'
                with path.open('xb') as f_obj:
                    torch.save(dict(schema='cog-role-name-consist-001.adapter.v1',format=1,plan=asdict(plan),arm=plan.arm,seed=plan.seed,
                      model_arm='tap_compare',consistency_weight=plan.coefficient,
                      updates=i,backbone_seed=BASE_SEED,backbone_sha256=BASE,backbone_state_sha256=b0,
                      adapter_initial_sha256=a0,adapter_state_sha256=ah,train_sha256=TRAIN,schedule_sha256=sched_pin,
                      tokenizer_words=list(tok.words),vocab_size=tok.size,
                      state_dict={n:p.detach().cpu().clone() for n,p in adapter.state_dict().items()}),f_obj)
                endpoints[str(i)]=dict(artifact=path.name,sha256=digest(path),adapter_state_sha256=ah,state_parameters=552,
                     unique_families=len(seen),counters=dict(counters),mean_trace={k:v/i for k,v in sums.items()})
    need(set(ENDPOINTS)=={int(k) for k in endpoints} and ever_nonzero==STATE_KEYS and len(seen)==320 and
      Counter(j for j,_,_,_ in schedule)==Counter({j:4 for j in range(320)}) and
      (counters['paired_updates'],counters['ordinary_updates'],counters['consistency_pairs'])==(1188,92,4752) and
      (counters['model_forwards'],counters['training_forwards'],counters['integrity_forwards'])==(1283,1280,3),
      'exact training geometry')
    final=dict(**counters,trainable_parameters=552,frozen_parameters=37414,prefix_targets=0,skipped_updates=0,
      coefficient=plan.coefficient,covered_train_edges=exposure['edge_count'],unique_training_worlds=exposure['served_original_worlds'],
      unique_train_families=len(seen),unique_train_at_128=endpoints['128']['unique_families'],endpoint_artifacts=2,
      initialized_exact_parity=1,earlier_logits_endpoint_parity_checks=2,frozen_base_integrity_checks=2,
      train_foil_labels_only=1,control_inputs=0,protected_test_inputs=0,selected_best_endpoint=0,
      nonzero_gradient_parameter_tensors=len(ever_nonzero),observed_trace_steps=1280)
    write(out/'fit.json',dict(**{k:v for k,v in info.items() if k!='status'},status='COMPLETE',checkpoints=endpoints,
      trace_sha256=digest(out/'steps.jsonl'),counters=final,elapsed_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu))
    write(out/'metrics.json',{'committed_updates':1280,'model_forwards':1283})

def main():
    arm,seed=sys.argv[1:]; out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    case=strict_json(Path('adequacy.json').read_bytes())
    recipe=strict_json(Path('recipe.json').read_bytes())
    need(recipe['research_adequacy']['sha256']==digest(Path('adequacy.json')) and
      all(digest(Path(x))==h for x,h in case['inputs'].items()),'native exact byte closure')
    execute(Plan(int(seed),arm),out)
    inputs={**case['inputs'],'adequacy.json':digest(Path('adequacy.json'))}
    write(out/'observations.json',dict(schema='trainlab.execution-adequacy-observations.v1',
      recipe_sha256=case['recipe_sha256'],inputs=inputs,
      producer={'file':'fit_name_consistency.py','sha256':inputs['fit_name_consistency.py']},
      observations=strict_json((out/'fit.json').read_bytes())['counters']))
if __name__=='__main__':main()
