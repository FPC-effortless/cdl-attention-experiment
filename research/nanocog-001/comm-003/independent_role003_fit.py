"""Verify retained native TRAIN receipts + safely decode endpoints; never call a model."""
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys
from statistics import median

import torch
from trainlab.runner import verify_run
from trainlab.research_adequacy import recipe_identity

def require(ok,reason):
    if not ok:raise ValueError(reason)
def read(p):return json.loads(p.read_bytes())
def stamp(p):return sha256(p.read_bytes()).hexdigest()
def hash_state(state):
    h=sha256()
    for n in sorted(state):
        t=state[n]
        require(torch.is_tensor(t) and t.dtype==torch.float32 and torch.isfinite(t).all().item(),"bad state tensor")
        h.update(n.encode());h.update(str((list(t.shape),str(t.dtype))).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def run(root,admissions):
    out={}
    original_root=admissions.resolve().parent.parent/'cog-role-002'
    orig_audit=original_root/'ROLE002_FIT_INDEPENDENT_AUDIT.json'
    require(stamp(orig_audit)=='92d9bbf3b1b4643236be235fa7c9292dcceb29de7aeb65eca5a00ce3e2cf880c',
            "untouched original audit")
    original=read(orig_audit)['runs']
    candidate=admissions.parent/'ROLE003_STAGE_CANDIDATE.json'
    require(stamp(candidate)=='9291eadbdb548d63d779baac8fd4a47d7a9baf1d6af9ccdc44126047da926a50',"candidate drift")
    seals=read(candidate)['fits']
    for d in sorted(root.iterdir()):
        r=read(d/'run.json');seed=r['seed'];p=d/'output';a=admissions/f'gated32-{seed}'
        require(seed in (1171,1271,1371) and seed not in out,"unexpected seed roster")
        require(r['status']=='succeeded' and r['exit_code']==0 and r['execution_purpose']=='research'
             and r['qualification']=='NOT_QUALIFIED' and r['research_adequacy']=='EXECUTED_CRITERIA_MET'
             and verify_run(d)['valid'],"native run integrity/qualification")
        exp=read(d/'experiment.json');observ=read(p/'observations.json')
        seal=seals[a.name]
        require(stamp(a/'recipe.json')==seal['recipe_sha256'] and r['inputs']==seal['inputs'],"candidate closure changed")
        require(exp==read(a/'recipe.json') and r==read(a/'cli-output.json'),"root recipe/receipt drift")
        require(set(exp['inputs'])==set(r['inputs']) and all(stamp(a/n)==h for n,h in r['inputs'].items()),
                "sealed recipe input changed")
        require(not any('actor' in n or 'judge' in n or 'control_' in n for n in exp['inputs']),"inference data input")
        oldrun=original_root/'fits'/original[str(seed)]['run_id']
        oldreceipt=read(oldrun/'run.json'); oldp=oldrun/'output'; oldfit=read(oldp/'fit.json')
        oldattempt=read(oldp/'attempt.json')
        require(verify_run(oldrun)['valid'] and stamp(oldrun/'run.json')==original[str(seed)]['run_sha256'],
                "original run identity")
        unchanged={n for n in oldreceipt['inputs'] if n in r['inputs']}
        changed={n for n in unchanged if oldreceipt['inputs'][n]!=r['inputs'][n]}
        require(changed=={'adequacy.json','preflight.json','observe_fit.py','cog_comm/role_fit.py'}
                and r['inputs'].keys()-oldreceipt['inputs'].keys()=={'ROLE_003_PROTOCOL.md'}
                and oldreceipt['inputs'].keys()-r['inputs'].keys()=={'ROLE_002_PROTOCOL.md'},
                "unanticipated changed worker input bytes")
        require(r['inputs']['cog_comm/role_fit.py']=='154798ce7128282f2ec5d2ca6b8dfe83f078db26471c96220ba3ddf185a3ed4d'
                and r['inputs']['cog_comm/objective.py']=='b8ddabcd50054664c0811615f5c2ec9a53c0740fbf73f002d4e451daa1e2e481',
                "ROLE-003 implementation bytes mismatch")
        require(set(r['artifacts'])==set(exp['artifacts']) and all(stamp(p/n)==h for n,h in r['artifacts'].items())
                and all(stamp(d/n)==h for n,h in r['files'].items()),"run artifact drift")
        require(stamp(a/'adequacy.json')==exp['research_adequacy']['sha256'],"adequacy pin")
        execution=read(d/'execution_adequacy.json')
        require(execution["status"]=="MET" and all(t['status']=='MET' for t in execution['checks'])
                and observ['recipe_sha256']==recipe_identity(exp) and observ['inputs']==r['inputs']
                and r['inputs'][observ['producer']['file']]==observ['producer']['sha256'],
                "execution observation not bound")
        fit=read(p/'fit.json');attempt=read(p/'attempt.json');plan=fit['plan']
        require(plan==attempt['plan']==oldfit['plan']|{'clip':2.0}
            and observ['observations']['prefix_ce_weight']==.25 and observ['observations']['clip_threshold']==2.0
            and oldfit['plan']['clip']==1.0,
            "plan drift")
        require(attempt['runtime']==oldattempt['runtime']
                and attempt['runtime']['threads']==1 and attempt['runtime']['deterministic'] is True
                and attempt['runtime']['torch']==torch.__version__=='2.10.0+cpu'
                and attempt['runtime']['executable_sha256']==stamp(Path(sys.executable).resolve())
                and r['runtime']==oldreceipt['runtime']
                and 0<r['elapsed_seconds']<=180,"runtime changed or cap exceeded")
        require(fit['all_initial_sha256']==oldfit['all_initial_sha256']
                and fit['trunk_initial_sha256']==oldfit['trunk_initial_sha256']
                and fit['schedule_sha256']==oldfit['schedule_sha256']
                and stamp(p/'vocabulary.json')==stamp(oldp/'vocabulary.json')
                and fit['train_sha256']==oldfit['train_sha256']==r['inputs']['train.json'],
                "initial weights, tokenizer, schedule, or exposure changed")
        schedule=read(p/'schedule.json')
        expected=[]
        rng=random.Random(seed ^ 0xC060001)
        for _ in range(4):
            order=list(range(320));rng.shuffle(order);expected+=order
        require(schedule['family_ordinals']==expected and schedule['paired_cell_ordinals']==[0,4,1,5,2,6,3,7]
                and stamp(p/'schedule.json')==fit['schedule_sha256']==attempt['schedule_sha256']
                and Counter(expected)==Counter({i:4 for i in range(320)}),
                "independent schedule mismatch")
        notes=[json.loads(line) for line in (p/'steps.jsonl').read_bytes().splitlines()]
        require(len(notes)==1280 and stamp(p/'steps.jsonl')==fit['trace_sha256'],"trace count/hash")
        for i,t in enumerate(notes,1):
            require(t['step']==i and t['family_ordinal']==expected[i-1] and t['model_forwards']==1
                and (t['worlds'],t['prefix_targets'],t['answer_targets'])==(8,48,8)
                and math.isclose(t['lr'],.001*min(1,i/8),abs_tol=1e-20)
                and math.isfinite(t['raw_gradient_norm']) and t['raw_gradient_norm']>0
                and abs(t['objective']-.25*t['prefix_ce_nats_per_target']-t['answer_ce_nats_per_target']-t['paired_foil_softplus'])<3e-6,
                f"trace trajectory mismatch step {i}")
        old_notes=[json.loads(line) for line in (oldp/'steps.jsonl').read_bytes().splitlines()]
        require(len(old_notes)==len(notes)==1280 and stamp(oldp/'steps.jsonl')==oldfit['trace_sha256']
                and all((a['step'],a['family_ordinal'],a['worlds'],a['prefix_targets'],a['answer_targets'],a['lr'])
                     ==(b['step'],b['family_ordinal'],b['worlds'],b['prefix_targets'],b['answer_targets'],b['lr'])
                     for a,b in zip(notes,old_notes,strict=True)),"paired old/new exposure mismatch")
        first_keys=('objective','raw_gradient_norm','prefix_ce_nats_per_target','answer_ce_nats_per_target',
                    'paired_foil_softplus','answer_minus_foil_mean')
        require(all(notes[0][k]==old_notes[0][k] for k in first_keys),"first preclip terms/norm differ")
        def norm_bands(trace):
            xs=[t['raw_gradient_norm'] for t in trace]
            require(all(math.isfinite(x) and x>0 for x in xs),"invalid gradient norm in paired trace")
            return {'steps':len(xs),'lte_1':sum(x<=1.0 for x in xs),'gt_1_lte_2':sum(1.0<x<=2.0 for x in xs),
                    'gt_2':sum(x>2.0 for x in xs),'min':min(xs),'median':median(xs),'max':max(xs)}
        bands={label:{str(k):norm_bands(trace[:k]) for k in (128,1280)}
               for label,trace in (('clip1',old_notes),('clip2',notes))}
        b=bands['clip2']['1280']
        require(observ['observations']['raw_norm_above_1']==b['gt_1_lte_2']+b['gt_2']
                and observ['observations']['raw_norm_above_2']==b['gt_2'],"native clipping input norm counts")
        max_error=max(abs(t['objective']-.25*t['prefix_ce_nats_per_target']-t['answer_ce_nats_per_target']-t['paired_foil_softplus']) for t in notes)
        states={}
        for n in (128,1280):
            ep=fit['endpoints'][str(n)]
            cp=torch.load(p/ep['checkpoint'],map_location='cpu',weights_only=True)
            ws=torch.load(p/ep['resume'],map_location='cpu',weights_only=True)
            state=cp['state_dict'];h=hash_state(state)
            require(cp['schema']=='cog-comm-role-001.checkpoint.v1' and cp['updates']==ws['updates']==n
                 and cp['plan']==plan | {'betas':tuple(plan['betas']),'saves':tuple(plan['saves'])}
                 and cp['train_sha256']==r['inputs']['train.json']
                 and cp['all_initial_sha256']==fit['all_initial_sha256']
                 and cp['seed']==seed and cp['vocab_size']==287
                 and cp['vocab_sha256']==stamp(p/'vocabulary.json')
                 and cp['schedule_sha256']==ws['schedule_sha256']==fit['schedule_sha256']
                 and sum(t.numel() for t in state.values())==37414
                 and h==ep['state_dict_sha256'] and ep['checkpoint_sha256']==stamp(p/ep['checkpoint'])==ws['checkpoint_sha256']
                 and ep['resume_sha256']==stamp(p/ep['resume']),"endpoint checkpoint closure mismatch")
            for st in ws['optimizer_state_dict']['state'].values():
                require(int(st['step'].item())==n,"optimizer step differs")
            counts=ep['counters']
            require(counts==cp['counts'] and counts['committed_updates']==n
                 and counts['answer_targets']==8*n and counts['prefix_targets']==48*n
                 and counts['unpadded_input_tokens']==56*n
                 and ep['unique_train_families_observed']==len(set(expected[:n])),"endpoint derived traffic")
            for k,x in ep['mean_train_loss_terms'].items():
                require(math.isclose(x,sum(t[k] for t in notes[:n])/n,abs_tol=1e-12),"endpoint loss average mismatch")
            oldep=original[str(seed)]['endpoints'][str(n)]
            require(h!=oldep['state_sha256'] and ep['checkpoint_sha256']!=oldep['checkpoint_sha256'],
                    "intervened checkpoint unexpectedly matches old weights")
            states[str(n)]={'optimizer_step':n,'checkpoint_sha256':ep['checkpoint_sha256'],'state_sha256':h,
                            'paired_original_checkpoint_sha256':oldep['checkpoint_sha256']}
        require(states['128']['state_sha256']!=states['1280']['state_sha256'] and
                r['metrics']==read(p/'metrics.json')=={'committed_updates':1280} and
                all(fit['counters'][k]==0 for k in ('protected_test_inputs','historical_split_inputs','control_inputs','skipped_updates')),
                "distinct endpoint/final scope mismatch")
        out[seed]={'max_weighted_objective_error_nats':max_error,'first_step_preclip':{k:notes[0][k] for k in first_keys},'exact_first_step_raw_term_parity':True,'raw_gradient_norm_bands':bands,'matched_initial_state_sha256':fit['all_initial_sha256'],'paired_original_run_id':oldrun.name,'run_id':d.name,'run_sha256':stamp(d/'run.json'),'verification':'valid',
                   'qualification':r['qualification'],'research_adequacy':r['research_adequacy'],
                   'trace_steps':len(notes),'train_sha256':r['inputs']['train.json'],
                   'endpoints':states,'schedule_sha256':fit['schedule_sha256'],
                   'subprocess_elapsed_seconds':r['elapsed_seconds']}
    require(set(out)=={1171,1271,1371},"missing seed")
    return {'schema':'cog-comm-role-003.independent-fit-audit.v1','outcome':'FIT_INTEGRITY_CHECKED_ONLY',
            'model_forwards_by_auditor':0,'optimizer_updates_by_auditor':0,'prediction_outcomes_inspected':False,'runs':out}

if __name__=='__main__':
    root,admissions,result=map(Path,sys.argv[1:])
    report=run(root,admissions)
    with result.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(stamp(result))
