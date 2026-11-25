"""Stage exact TRAIN-only candidate closures. No model import or worker dispatch."""
from hashlib import sha256
import json,shutil
from pathlib import Path
from trainlab.runner import load_config,verify_run
from trainlab.research_adequacy import assess,recipe_identity

R=Path(__file__).resolve().parent
OLD=R.parent/'cog-role-002'
REPAIR=R.parent/'cog-role-repair-004'
SEEDS=(1771,1871,1971)
ARMS=('consistency000','consistency025')
COEFF={'consistency000':0.,'consistency025':.25}
SCHED='bb6d38ae584bf4421742bc063cc275044646bd8c5e61fc2571447201276b065c'
BASE='517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'
TRAIN='02607c19e760e164989865a54422eb7cd78fed7d117a7fa2ff6c4df9d700f9f8'
OLDFIT='92d9bbf3b1b4643236be235fa7c9292dcceb29de7aeb65eca5a00ce3e2cf880c'
RANK='e26ece9ce000621235138f338dd29adcdbed26ab5018142e7065b2698c9966c4'
RANK_SEAL='3878393d89de5f85491f05f06416534741f400bd2fd32971fbbf78ca5f3b61e0'
LOCK='b0ba8b65677ccb8a29c4fb2992321247b0099a72acad350709a8a03a4ffa2889'
def req(ok,m):
    if not ok:raise ValueError(m)
def sha(p):return sha256(p.read_bytes()).hexdigest()
def read(p,expected=None):
    b=p.read_bytes()
    if expected:req(sha256(b).hexdigest()==expected,'input SHA '+str(p))
    return json.loads(b)
def write(p,value):
    with p.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def main():
    old=read(OLD/'ROLE002_FIT_INDEPENDENT_AUDIT.json',OLDFIT)['runs']['1371']
    run=OLD/'fits'/old['run_id']; rr=read(run/'run.json',old['run_sha256'])
    ck=run/'output/checkpoint-1280.pt'
    src=OLD/'admissions/gated32-1371'
    req(verify_run(run)['valid'] and rr['status']=='succeeded'
        and rr['artifacts']['checkpoint-1280.pt']==old['endpoints']['1280']['checkpoint_sha256']==sha(ck)==BASE,'base lineage')
    rk=read(REPAIR/'INDEPENDENT_RANK_AUDIT.json',RANK)
    req(rk['seal']==RANK_SEAL and sha(REPAIR/'RANK_SEAL.json')==RANK_SEAL,'rank audit lineage')
    weak=rk['results']['rank-1371']
    req(weak['common']['rank']=={'1e-06':19,'0.0001':19} and
        all(weak['arms'][a]['full']['rank']=={'1e-06':51,'0.0001':51} for a in ('tap_compare','literal_control')),'native rank prerequisites')
    files={x:src/x for x in rr['inputs'] if x.startswith(('nanocog/','cog_comm/')) or x in {'train.json','audit.json','forge-receipt.json'}}
    req(sha(src/'train.json')==TRAIN and all(sha(v)==rr['inputs'][k] for k,v in files.items()),'TRAIN/import bytes')
    for name in ('frozen_gate_adapter.py','tap_gate_residual.py','INDEPENDENT_RANK_AUDIT.json'):
        files[name]=REPAIR/name
    for name in ('fit_name_consistency.py','consistency_objective.py','schedule_intake.py'):
        files[name]=R/name
    req(sha(R/'NAME_CONSIST_EVAL_LOCK.md')==LOCK,'prospective protocol lock changed')
    files['NAME_CONSIST_EVAL_LOCK.md']=R/'NAME_CONSIST_EVAL_LOCK.md'
    m=read(R/'train_schedule/manifest.json',SCHED)
    req(m['train_sha256']==TRAIN and m['candidate_pairs']==444 and m['covered_families']==297,'TRAIN schedules')
    files['train_schedule_manifest.json']=R/'train_schedule/manifest.json'
    files['checkpoint.pt']=ck
    train=read(src/'train.json',TRAIN)
    req(train['role']=='TRAIN' and len(train['families'])==320 and all(len(x['prompts'])==len(x['answers'])==8 for x in train['families']),'TRAIN geometry')
    pre=dict(nominal_adapter_parameters=552,frozen_backbone_parameters=37414,base_checkpoint_verified=1,
      base_backbone_seed=1371,vocab_size=287,common_feature_rank=19,joint_feature_rank_both_arms=51,
      planned_updates=1280,planned_endpoints=2,planned_integrity_forwards=3,train_families=320,
      train_distinct_worlds=len(set(x for f in train['families'] for x in f['prompts'])),max_visible_tokens=7,
      worlds_per_update=8,foil_pairs_per_update=4,prefix_ce_weight=0,protected_inputs=0,train_pair_edges=444)
    groups={
     'model_size': (dict(nominal_adapter_parameters=552,frozen_backbone_parameters=37414,base_checkpoint_verified=1,common_feature_rank=19,joint_feature_rank_both_arms=51),
         dict(trainable_parameters=552,frozen_parameters=37414,frozen_base_integrity_checks=2,nonzero_gradient_parameter_tensors=2),'Only adapter weights train; rank checked but not capacity equivalence.'),
     'training_duration': (dict(planned_updates=1280,planned_endpoints=2,planned_integrity_forwards=3),
         dict(committed_updates=1280,model_forwards=1283,training_forwards=1280,integrity_forwards=3,observed_trace_steps=1280,endpoint_artifacts=2,unique_train_at_128=128),
         'Uninterrupted fixed trajectory, no early/model selection.'),
     'datasets': (dict(train_families=320,train_distinct_worlds=2560,max_visible_tokens=7,protected_inputs=0),
         dict(train_worlds=10240,train_input_tokens=71680,control_inputs=0,protected_test_inputs=0,unique_train_families=320,unique_training_worlds=2560,covered_train_edges=444),
         'Exact TRAIN only; seven-token constructed grammar.'),
     'training_regime': (dict(base_backbone_seed=1371,worlds_per_update=8,foil_pairs_per_update=4,prefix_ce_weight=0),
         dict(initialized_exact_parity=1,earlier_logits_endpoint_parity_checks=2,answer_targets=10240,paired_foil_pairs=5120,
              prefix_targets=0,train_foil_labels_only=1,skipped_updates=0,selected_best_endpoint=0,paired_updates=1188,ordinary_updates=92,consistency_pairs=4752),
         'Answer cross-entropy + fixed paired foil plus registered bounded margin consistency coefficient. No labels entering inference.')}
    out=R/'fit_admissions';out.mkdir()
    ledger={}
    for arm in ARMS:
        for seed in SEEDS:
            d=out/f'{arm}-{seed}';d.mkdir()
            st=R/'train_schedule'/f'schedule-{seed}.json'
            stats=m['seeds'][str(seed)]
            req(sha(st)==stats['sha256'] and stats['summary']['distinct_worlds']==2560
                and stats['summary']['paired_consistency_pairs']==4752,'seed schedule witness')
            copies={**files,'schedule.json':st}
            for n,p in copies.items():
                j=d/n;j.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,j)
            write(d/'preflight.json',{**pre,'planned_coefficient':COEFF[arm]})
            decl=sorted([*copies,'preflight.json'])
            cfg=dict(schema='trainlab.experiment.v1',name=d.name,execution_purpose='research',
               description='Known-weak frozen tap adapter, exact matched TRAIN name-remap schedule and coefficient only.',
               command=['python','fit_name_consistency.py',arm,str(seed)],seed=seed,timeout_seconds=180,
               source_kind='CONSTRUCTED',split_identity='ROLE-001-public-synthetic-train-v1',
               inputs=sorted([*decl,'adequacy.json']),
               artifacts=['adapter-128.pt','adapter-1280.pt','fit.json','attempt.json','exposure.json','steps.jsonl','observations.json'],
               metrics={'committed_updates':'max','model_forwards':'min'},budget={'optimizer_updates':1280,'model_forwards':1283})
            dims={}; execution={}
            for n,(p,q,basis) in groups.items():
                if n=='training_regime':
                    p={**p,'planned_coefficient':COEFF[arm],'train_pair_edges':444}
                    q={**q,'coefficient':COEFF[arm]}
                def cs(f):return [dict(field=k,operator='eq',threshold=v,rationale=basis) for k,v in f.items()]
                dims[n]=dict(configuration=basis,justification=basis,limitations='Necessary conditions only, not generalization or qualification.',
                  criteria=[dict(c,evidence='preflight.json',id=c['field']) for c in cs(p)])
                execution[n]=cs(q)
            contract=dict(schema='trainlab.research-adequacy.v1',claim='Record exact known-backbone residual-only TRAIN trajectories.',
               claim_kind='bounded_diagnostic',scope=f'TRAIN {arm} seed {seed}; frozen ROLE-002 seed1371 backbone, exact paired schedule.',
               excluded_claims='No held results, speed, natural language, independent backbone seeds or model promotion.',
               recipe_sha256=recipe_identity(cfg),inputs={n:sha(d/n) for n in decl},
               review=dict(author='Tony',reviewer='independent adequacy reviewer',
                 basis='Draft exact candidate awaits separate explicit exact FIT admission; frozen backbone plus matched residual.',
                 criteria_frozen_before_evaluation=True),dimensions=dims,execution={'artifact':'observations.json','dimensions':execution})
            write(d/'adequacy.json',contract)
            cfg['research_adequacy']=dict(file='adequacy.json',sha256=sha(d/'adequacy.json'))
            write(d/'recipe.json',cfg)
            inputs={n:sha(d/n) for n in cfg['inputs']}
            req(load_config(d/'recipe.json')==cfg and assess(cfg,d,inputs)['status']=='DECLARED_CRITERIA_MET','native declarations failed')
            ledger[d.name]=dict(seed=seed,arm=arm,recipe_sha256=sha(d/'recipe.json'),recipe_identity=recipe_identity(cfg),inputs=inputs,
              frozen_base_sha256=BASE,train_sha256=TRAIN,frozen_base_run_id=old['run_id'],rank_audit_sha256=RANK,
              prospective_lock_sha256=LOCK,schedule_sha256=sha(st),coefficient=COEFF[arm],status='DECLARED_CRITERIA_MET')
    write(out/'ledger.json',ledger)
    print(sha(out/'ledger.json'))
if __name__=='__main__':main()
