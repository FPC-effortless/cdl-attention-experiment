"""One exact jointly trainable ROLE experiment. No work on import."""
from collections import Counter
import json,math,os,sys,time
from pathlib import Path
from joint_local_objective import ARMS,SEEDS,ENDPOINTS,need,objective
from joint_local_intake import batches,digest,gather
from joint_local_schedule import encoded

PARAMS=37414
TRAIN_AUDITS=(1152,1280)
COUNTS=dict(committed_updates=1280,training_forwards=1280,train_audit_forwards=352,model_forwards=1632,
  backwards=1280,answer_targets=10240,prefix_targets=61440,input_tokens=71680,mixed_steps=640,
  ordinary_steps=640,margin_terms=2560,distinct_history_slots=128,distinct_ordinary_rows=640,
  selected_endpoints=0,saved_endpoints=2,protected_inputs=0,trainable_parameters=37414,
  edit_rows_presented=3840,ordinary_rows_presented=6400)

def save(p,x):
    with p.open('xb') as f:f.write(encoded(x))

def audit_train(model,edit,ordinary,torch):
    from torch.nn.functional import cross_entropy
    model.eval()
    allr=edit+ordinary
    xx=[p for x,_,_ in allr for p in x]; yy=[p for _,y,_ in allr for p in y]
    scores=[]; correct=[]; fw=0
    with torch.no_grad():
        for i in range(0,1408,8):
            x=torch.tensor(xx[i:i+8],dtype=torch.long); y=torch.tensor(yy[i:i+8],dtype=torch.long)
            z=model(x,state=None)[0][:,6,:]
            need(z.shape==(8,287) and bool(torch.isfinite(z).all()),'fixed finite train audit logits')
            strict=z.gather(1,y[:,None])>z.scatter(1,y[:,None],-torch.inf).amax(1,keepdim=True)
            scores.extend(cross_entropy(z,y,reduction='none').tolist());correct.extend(strict[:,0].tolist());fw+=1
    model.train()
    need(fw==176 and len(scores)==len(correct)==1408,'fixed TRAIN audit forwards')
    first=sum(a and c and e for a,b,c,d,e,f in zip(*(correct[j:768:6] for j in range(6)),strict=True))
    second=sum(b and d and f for a,b,c,d,e,f in zip(*(correct[j:768:6] for j in range(6)),strict=True))
    return dict(forwards=fw,strict_first_triplets=first,strict_second_triplets=second,
       strict_ordinary_worlds=sum(correct[768:]),strict_ordinary_families=sum(all(correct[i:i+8]) for i in range(768,1408,8)),
       answer_ce_by_role_condition=[math.fsum(scores[j:768:6])/128 for j in range(6)])

def mixer_operation_budget(feature, b=8, t=7):
    """Actual scoped forward geometry of pinned source; partial analytic count."""
    c,w,l=32,4,2
    if feature=='gated32':
        mac=b*t*(c*w+c*c)
        elem=b*t*c*(2*w+w-1) + b*t*w
        terms=dict(offset_gate_mac=b*t*c*w,pointwise_mac=b*t*c*c,
            depthwise_gating_mul=b*t*c*2*w,lag_sum_add=b*t*c*(w-1),offset_bias_add=b*t*w,
            softmax_elements=b*t*w)
    elif feature=='attention32':
        d=10
        mac=b*t*(4*c*d)+2*b*t*t*d
        elem=2*b*t*t
        terms=dict(qkv_mac=3*b*t*c*d,project_mac=b*t*d*c,full_qk_mac=b*t*t*d,
            full_weighted_v_mac=b*t*t*d,score_scaling_mul=b*t*t,pos_bias_add=b*t*t,
            softmax_elements=b*t*t)
    else:raise ValueError('unregistered source geometry')
    return dict(batch=b,tokens=t,width=c,layers=l,per_layer_terms=terms,per_layer_mac=mac,
        per_layer_scalar_add_mul=elem,two_layer_flops_2_per_mac=l*(2*mac+elem),
        scope='Partial forward mixer only; excludes softmax FLOPs, masks, padding, trunk, backward, optimizer, and data traffic.')

def execute(arm,seed,out,root,pins,run,runtime):
    import torch
    import cog_comm.role_mixers as mix
    import cog_comm.fit as common
    import nanocog.model as core
    need(torch.__version__=='2.10.0+cpu' and not sys.flags.optimize and sys.version_info[:2]==(3,12)
      and digest(Path(sys.executable).resolve())==runtime['python_sha256']
      and digest(Path(torch.__file__).parent/'lib/libtorch_cpu.so')==runtime['torch_cpu_sha256']
      and all(Path(m.__file__).is_relative_to(root) for m in (mix,common,core)),'one exact local runtime')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    feature,tail=ARMS[arm]
    words=tuple(run['tokenizer_words']); edits,ordinary,slots=batches(root,pins,seed,words)
    model=mix.build_arm(287,feature,seed).train()
    params=tuple(model.parameters())
    need(sum(t.numel() for t in params)==PARAMS and all(t.requires_grad and t.dtype==torch.float32 and
      t.device.type=='cpu' for t in params),'joint 37,414 trainable fp32 CPU weights')
    full=lambda:common.stable_tensor_hash(model.state_dict(),lambda _:True)
    trunk=common.stable_tensor_hash(model.state_dict(),lambda n:not n.startswith('mixers.'))
    init=full(); plan=dict(arm=arm,seed=seed,mixer=feature,tail=tail,lr=.001,warmup=8,weight_decay=.01,
        clip=1.,betas=[.9,.999],eps=1e-8,prefix=.25,margin=.25,tail_weight=.5 if tail else 0.,temp=1.)
    info=dict(plan=plan,initial_state_sha256=init,initial_shared_trunk_sha256=trunk,slots_sha256=pins['slots.json'],
      train_data_pins={s:pins[s] for s in ('edit_bundles.json','ordinary_rows.json')},
      tokenizer_words=list(words),vocab_size=287,runtime={**runtime,'torch_version':str(torch.__version__)},
      status='NOT_QUALIFIED')
    save(out/'attempt.json',info)
    opt=torch.optim.AdamW(params,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
    c=Counter();ec=Counter();oc=Counter();audits={};points={};gradient_seen=set()
    start=time.perf_counter();cpu=time.process_time(); audit_wall=audit_cpu=0.
    with (out/'steps.jsonl').open('xb') as log:
        for i,s in enumerate(slots,1):
            x,y,fo=gather(edits,ordinary,s)
            x=torch.tensor(x,dtype=torch.long);y=torch.tensor(y,dtype=torch.long)
            f=None if fo is None else torch.tensor(fo,dtype=torch.long)
            rate=.001*min(i/8.,1.)
            for q in opt.param_groups:q['lr']=rate
            opt.zero_grad(set_to_none=True)
            logits,_=model(x,state=None)
            loss,parts=objective(logits,x,y,f,tail=tail)
            need(bool(torch.isfinite(loss)),'loss finite')
            loss.backward()
            need(all(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in params),'finite gradient routes in entire model')
            gradient_seen.update(j for j,p in enumerate(params) if bool((p.grad!=0).any()))
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            opt.step()
            vals={k:float(v.detach()) for k,v in parts.items()}
            actual=float(loss.detach());mixed=f is not None
            need(abs(actual-vals['answer_aggregation']-.25*(vals['prefix_ce']+vals['edit_margin_softplus']))<=3e-6,
              'exact objective accounting')
            if mixed:ec.update([s['bundle_slot']])
            oc.update(s['ordinary_slots'])
            c.update(committed_updates=1,training_forwards=1,model_forwards=1,backwards=1,
                answer_targets=8,prefix_targets=48,input_tokens=56,mixed_steps=int(mixed),
                ordinary_steps=int(not mixed),margin_terms=4*int(mixed),edit_rows_presented=6*int(mixed),
                ordinary_rows_presented=2 if mixed else 8)
            log.write(encoded(dict(step=i,lr=rate,raw_gradient_norm=norm,clipped=norm>1.,
                bundle_slot=s['bundle_slot'],ordinary_slots=s['ordinary_slots'],objective=actual,**vals)))
            if i in TRAIN_AUDITS:
                w0=time.perf_counter(); c0=time.process_time()
                a=audit_train(model,edits,ordinary,torch);audits[str(i)]=a
                audit_wall+=time.perf_counter()-w0; audit_cpu+=time.process_time()-c0
                c.update(model_forwards=a['forwards'],train_audit_forwards=a['forwards'])
            if i in ENDPOINTS:
                ep=out/f'checkpoint-{i}.pt';d=full()
                need(d!=init and all(int(opt.state[q]['step'].item())==i and bool(torch.isfinite(q).all()) for q in params),
                    'exact learned endpoint')
                if i==128:
                    need((c['mixed_steps'],c['ordinary_steps'],c['edit_rows_presented'],c['ordinary_rows_presented'])==
                         (64,64,384,640),'early schedule receipt')
                with ep.open('xb') as f:
                    torch.save(dict(info,schema='cog-joint-local-001.checkpoint.v1',updates=i,arm=arm,seed=seed,
                              state_sha256=d,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),f)
                points[str(i)]=dict(sha256=digest(ep),state_sha256=d,counters=dict(c))
    need(ec==Counter({i:5 for i in range(128)}) and oc==Counter({i:10 for i in range(640)})
         and len(gradient_seen)==len(params),'exact fit exposure and active gradients')
    final=dict(c)|dict(distinct_history_slots=len(ec),distinct_ordinary_rows=len(oc),selected_endpoints=0,
        saved_endpoints=len(points),protected_inputs=0,trainable_parameters=PARAMS)
    need(final==COUNTS,'exact native fit counters')
    a,b=map(audits.get,('1152','1280'))
    drift=[abs(x-y) for x,y in zip(a['answer_ce_by_role_condition'],b['answer_ce_by_role_condition'],strict=True)]
    fit_ok=(min(b['strict_first_triplets'],b['strict_second_triplets'])>=122 and
        b['strict_ordinary_worlds']>=608 and b['strict_ordinary_families']>=72 and max(drift)<=.02)
    wall=time.perf_counter()-start; proc=time.process_time()-cpu
    save(out/'fit.json',dict(info,terminal_status='COMPLETE',points=points,train_audits=audits,
        train_column_ce_absolute_drift=drift,train_max_role_ce_drift=max(drift),train_fit_adequacy=fit_ok,
        counts=final,training_and_audit_wall_seconds=wall,training_and_audit_process_cpu_seconds=proc,
        train_audit_wall_seconds=audit_wall,train_audit_process_cpu_seconds=audit_cpu,
        fit_loop_wall_excluding_train_audits_seconds=wall-audit_wall,
        fit_loop_cpu_excluding_train_audits_seconds=proc-audit_cpu,
        analytic_forward_mixer_per_batch=mixer_operation_budget(feature),
        trace_sha256=digest(out/'steps.jsonl'),profiled_arithmetic_status='SEPARATE_STAGE_REQUIRED'))
    save(out/'metrics.json',{k:final[k] for k in ('committed_updates','model_forwards')})
    return final

def main():
    from trainlab.research_adequacy import recipe_identity
    need(len(sys.argv)==3 and sys.argv[1] in ARMS and int(sys.argv[2]) in SEEDS,'registered invocation')
    arm,seed=sys.argv[1],int(sys.argv[2])
    root=Path.cwd(); out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    need(out.is_dir() and not out.is_symlink() and not any(out.iterdir()) and os.environ.get('TRAINLAB_SEED')==str(seed)
        and all(os.getenv(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
        'fresh native one-thread closure')
    cfg=json.loads((root/'recipe.json').read_bytes()); ad=json.loads((root/'adequacy.json').read_bytes())
    pins=ad['inputs']; ah=digest(root/'adequacy.json')
    need(cfg['command']==['python','fit_joint_local_001.py',arm,str(seed)] and cfg['seed']==seed and cfg['timeout_seconds']==180
      and cfg['budget']==dict(model_forwards=1632,optimizer_updates=1280) and cfg['research_adequacy']==dict(file='adequacy.json',sha256=ah)
      and ad['recipe_sha256']==recipe_identity(cfg) and set(cfg['inputs'])==set(pins)|{'adequacy.json'}
      and all((root/k).is_file() and not (root/k).is_symlink() and digest(root/k)==v for k,v in pins.items()),
      'exact native recipe/contract bytes')
    c=json.loads((root/'run_case.json').read_bytes())
    need(c['arm']==arm and c['seed']==seed and c['data_custody_stage']=='INDEPENDENT_TRAIN_ONLY','fixed case')
    result=execute(arm,seed,out,root,pins,c,c['runtime'])
    save(out/'observations.json',dict(schema='trainlab.execution-adequacy-observations.v1',inputs=pins|{'adequacy.json':ah},
       recipe_sha256=ad['recipe_sha256'],producer=dict(file='fit_joint_local_001.py',sha256=pins['fit_joint_local_001.py']),observations=result))

if __name__=='__main__':main()
