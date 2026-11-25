"""Build exact TrainLab closures, never start a fit."""
import argparse
import collections
from hashlib import sha256
import json
from pathlib import Path
import shutil
from trainlab.research_adequacy import recipe_identity

ROOT=Path(__file__).resolve().parent
NANO=ROOT/'smoke'
PARAM={'gated32':37414,'static32':37158,'attention32':37414,'gated48':69926}
def dump(p,d):
    p.write_text(json.dumps(d,sort_keys=True,allow_nan=False,indent=2)+'\n')
def hashfile(p):
    return sha256(p.read_bytes()).hexdigest()
def prep(dest,arm,seed,review):
    if not review.strip():
        raise ValueError('independent review basis required')
    dest.mkdir(parents=True)
    relative=['cog_comm/'+p+'.py' for p in ('__init__','fit','model','mix','router','objective','role_fit','role_mixers','role_control','role_panel')]
    relative+=['nanocog/__init__.py','nanocog/model.py']
    for p in relative:
        ty=dest/p
        ty.parent.mkdir(exist_ok=True)
        shutil.copyfile(NANO/p,ty)
    audit=ROOT/'role-data-audit'
    for name,expected in {
       'audit.json':'674e5125633ed84a4d252763916f24893036e1a956fb85f4a3193731f00c583b',
       'forge-receipt.json':'0488fbfdbdad9bdf6956ae6d19a9c2d1013792391361f391f749bea1f9d19467',
    }.items():
        if hashfile(audit/name)!=expected:
            raise ValueError('native data audit identity changed')
        shutil.copyfile(audit/name,dest/name)
    for p in ('observe_fit.py','ROLE_002_PROTOCOL.md'):
        shutil.copyfile(ROOT/p,dest/p)
    train = ROOT/'panels'/'train.json'
    sha=hashfile(train)
    if sha!='02607c19e760e164989865a54422eb7cd78fed7d117a7fa2ff6c4df9d700f9f8':
        raise ValueError('TRAIN byte identity changed')
    shutil.copyfile(train,dest/'train.json')
    spec=json.loads(train.read_bytes())['families']
    prompts=[s for f in spec for s in f['prompts']]
    from cog_comm.role_fit import load_train, Plan, schedule
    from cog_comm.role_mixers import build_arm
    tok,ids=load_train(train,sha)
    model=build_arm(tok.size,arm,seed)
    count=sum(p.numel() for p in model.parameters())
    order=schedule(Plan(seed,arm,width=48 if arm=='gated48' else 32))
    evidence=dict(parameters=count,two_local_layers=len(model.mixers),vocab_size=tok.size,
                  planned_updates=len(order),planned_unique_at_128=len(set(order[:128])),
                  planned_unique_at_1280=len(set(order)),
                  train_families=len(spec), unique_worlds=len(set(prompts)),
                  max_prompt_tokens=max(len(x.split()) for x in prompts),
                  worlds_per_step=len(ids[0][0]),paired_foils_per_update=4)
    dump(dest/'preflight.json',evidence)
    files=sorted([*relative,'observe_fit.py','ROLE_002_PROTOCOL.md','train.json','preflight.json','audit.json','forge-receipt.json'])
    artifacts=['checkpoint-128.pt','checkpoint-1280.pt','resume-128.pt','resume-1280.pt','attempt.json',
               'steps.jsonl','vocabulary.json','schedule.json','fit.json','native-counters.json','observations.json']
    recipe=dict(schema='trainlab.experiment.v1',execution_purpose='research',
        name=f'ROLE-002 quarter-prefix {arm} seed {seed}',description='Fixed-budget seven-token role-binding trajectory, TRAIN only.',
        command=['python','observe_fit.py',arm,sha],seed=seed,timeout_seconds=180,
        source_kind='CONSTRUCTED',split_identity='ROLE-001-public-synthetic-train-v1',
        inputs=files+['adequacy.json'],artifacts=artifacts,
        metrics={'committed_updates':'max'},budget={'optimizer_updates':1280,'presented_worlds':10240})
    groups={
      'model_size':({'parameters':PARAM[arm],'two_local_layers':2,'vocab_size':287},
                   {'total_parameters':PARAM[arm],'trainable_parameters':PARAM[arm]},'Small two-block arm at this exact capacity, not a scale or architecture-limit claim'),
      'training_duration':({'planned_updates':1280,'planned_unique_at_128':128,'planned_unique_at_1280':320},
                   {'committed_updates':1280,'observed_trace_steps':1280,'unique_train_at_128':128,'endpoint_artifacts':2},'One uninterrupted trajectory with two frozen exposure endpoints; no convergence claim'),
      'datasets':({'train_families':320,'unique_worlds':2560,'max_prompt_tokens':7},
                   {'train_families':320,'protected_test_inputs':0,'control_inputs':0,'historical_split_inputs':0,
                    'worlds_presented':10240,'dense_input_tokens':71680},'Exact accepted constructed role families; observation limited to familiar tokens and grammar'),
      'training_regime':({'worlds_per_step':8,'paired_foils_per_update':4},
                   {'prefix_ce_weight':.25,'observed_trace_answer_targets':10240,'observed_trace_prefix_targets':61440,'skipped_updates':0},'Quarter prefix CE with unchanged answer CE, sibling foil and minibatch geometry; fixed AdamW and no tuning')}
    dims={}; limits={}
    for dim,(pre,post,basis) in groups.items():
        def criteria(prepost):
            return [dict(field=k,operator='eq',threshold=v,rationale=basis) for k,v in prepost.items()]
        dims[dim]=dict(configuration=basis,justification=basis,
                       limitations='Necessary conditions only. Not benchmark qualification.',
                       criteria=[dict(x,evidence='preflight.json',id=x['field']) for x in criteria(pre)])
        limits[dim]=criteria(post)
    contract=dict(schema='trainlab.research-adequacy.v1',claim='Record deterministic local-role trajectory and the fixed 128/1280 checkpoints.',
                  claim_kind='bounded_diagnostic',scope=f'TRAIN only, one {arm} trajectory, CPU FP32, seed {seed}',
                  excluded_claims='No language transfer, no causal architectural incapacity or inference speed claim.',
                  recipe_sha256=recipe_identity(recipe),inputs={x:hashfile(dest/x) for x in files},
                  review=dict(author='Tony',reviewer='independent adequacy reviewer',basis=review,
                              criteria_frozen_before_evaluation=True),
                  dimensions=dims,execution={'artifact':'observations.json','dimensions':limits})
    dump(dest/'adequacy.json',contract)
    recipe['research_adequacy']=dict(file='adequacy.json',sha256=hashfile(dest/'adequacy.json'))
    dump(dest/'recipe.json',recipe)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    p.add_argument('--arm',choices=PARAM,default='gated32')
    p.add_argument('--basis',required=True)
    a=p.parse_args()
    for s in (1171,1271,1371):
        prep(a.folder/f'{a.arm}-{s}',a.arm,s,a.basis)
