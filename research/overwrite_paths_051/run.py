"""Frozen native old/new factorial assay; no optimizer or infrastructure mutation."""
from pathlib import Path
import sys,json,hashlib,os,shutil,types,itertools
from datetime import datetime
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'lifecycle_learning_001'
sys.path.insert(0,str(PARENT))
import torch
import numpy as np
from data import encode,history_map,VOCAB,TOKENS,HEADS,CUTOFF
from models import build,tensors
from candidate_hooks import final_set_interventions,categorical_only
from grid import generate,validate
from factorial_summary import summarize
from forge_data.core import canonical,release
from forge_data.dataset import Dataset
from trainlab.runner import digest,read_json,write_json,run_experiment,verify_run
from research_lab.ecosystem import _runtime
from research_lab.core import Controller
from research_lab.adapters import import_forge,import_trainlab
from research_lab.evidence import put_node,link,retrieve
from investigation_world.benchmark.lifecycle_observability import replay
from investigation_world.evidence import EvidenceRecord,EvidenceOutcome,EvidenceVisibility,EvidenceSubjectRef,EvidenceProducerRef,EvidenceArtifactRef
CONDITIONS=tuple(itertools.product(('normal','categorical'),(False,True),(False,True)))

def require(x,message):
    if not x:raise ValueError(message)

def runtime():return {**_runtime(),'torch':torch.__version__}

def parent_pins():
    require(digest(PARENT/'freeze.json')=='449e0aa06076e99547ad78f70a8a9206c6887cd2346fde1a25d7a085bf309d33','parent freeze')
    require(digest(PARENT/'training_closed.json')=='a181cb5c3fca3d34a761c98cf2ca03f282618fccc9cef0fca73b634dc49deaac','parent closure')
    approved=read_json(PARENT/'freeze.json')
    require(approved['runtime']=={**runtime(),'torch_file':str(Path(torch.__file__).resolve())},'parent runtime')
    pins={n:digest(PARENT/n) for n in ('freeze.json','training_closed.json','data.py','models.py','native_csm.py')}
    for n in ('data.py','models.py','native_csm.py'):require(pins[n]==approved['files'][n],'parent source '+n)
    for r in read_json(PARENT/'training_closed.json')['runs']:
        p=f"runs/{r['name']}/{r['run_id']}"
        require(verify_run(PARENT/p)['valid'],'parent receipt')
        require(digest(PARENT/p/'output/checkpoint.pt')==r['checkpoint_sha256'],'parent checkpoint')
        pins[p+'/output/checkpoint.pt']=r['checkpoint_sha256'];pins[p+'/run.json']=digest(PARENT/p/'run.json')
    return pins

def check():
    frozen=read_json(ROOT/'freeze.json')
    require(frozen['runtime']==runtime(),'runtime drift');require(frozen['parent_pins']==parent_pins(),'parent drift')
    for n,h in frozen['files'].items():require(digest(ROOT/n)==h,'input drift '+n)
    return frozen

def weights_pin(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def make_batch(cases):
    require(0<len(cases)<=32,'batch size')
    finals={c['final_set_token'] for c in cases};require(len(finals)==1 and finals<={2,5},'mixed final tokens')
    rows=[]
    for c in cases:
        x=encode(c['memory'],c['key']);p=c['final_set_token']
        require(len(x['ids'])==27 and x['ids'][p-2:p+1]==[VOCAB['set'],VOCAB[c['key']],VOCAB[c['new']]],'commit mismatch')
        require(p==len(c['memory']['events'])*3-1,'last commit mismatch')
        require(replay(c['memory'])==history_map(c['memory']),'oracle mismatch')
        require(replay(c['memory'])[c['key']]==c['new'],'wrong label')
        rows.append({**x,'answer':VOCAB[c['new']]})
    return tensors(rows),next(iter(finals))

def assay_batch(model,cases,heads,zero_reads,full_gain):
    require(heads in ('normal','categorical'),'head regime')
    require(all(p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters()),'CPU float32 only')
    batch,final=make_batch(cases);indices=torch.tensor([int(c['key'][1:]) for c in cases]);bi=torch.arange(len(cases))
    saved={'steps':0};original_step=model.step;handles=[]
    def tracked_step(token,state):
        saved['steps']+=1
        p=model.position
        if p==final:
            saved['before']=state.values[:,:4].clone();saved['before_links']=state.links[:,:4].clone()
        if p==24:saved['before_query']=state.values[bi,indices].clone()
        logits,after=original_step(token,state)
        if p==final:
            saved['after_set']=after.values[bi,indices].clone()
            mask=torch.arange(4)[None]!=indices[:,None]
            saved['untouched']=(((after.values[:,:4]==saved['before']).all(-1)&(after.links[:,:4]==saved['before_links']).all(-1))|~mask).all(-1)
        if p==26:saved['query_state']=after.values[:,7].clone()
        return logits,after
    def effective_gain(module,args,out):
        if model.position==final:saved['gain']=out.sigmoid().squeeze(-1).clone()
    def effective_edit(module,args,out):
        if model.position==final:saved['edit']=out.argmax(-1).clone()
    def effective_reads(module,args):
        if model.position==final:
            d=model.config.width
            require(args[0].shape==(len(cases),model.transform[0].in_features),'transform shape')
            require(args[0].dtype==torch.float32 and args[0].device.type=='cpu','transform dtype/device')
            saved['reads']=args[0][:,d:3*d].clone()
    model.step=tracked_step
    try:
        with final_set_interventions(model,final,zero_reads=zero_reads,full_gain=full_gain) as calls:
            handles=[model.write_gate.register_forward_hook(effective_gain),model.value_edit.register_forward_hook(effective_edit),model.transform.register_forward_pre_hook(effective_reads)]
            logits,cap=model.sequence(batch['ids'],collect=True,supplied=categorical_only(batch) if heads=='categorical' else None)
        require(calls=={'zero_reads':int(zero_reads),'full_gain':int(full_gain)},'hook count')
    finally:
        for h in handles:h.remove()
        model.step=original_step
    require(saved['steps']==27,'one sequence required')
    require(bool(torch.isfinite(logits).all()),'nonfinite logits')
    if zero_reads:require(bool((saved['reads']==0).all()),'zero reads failed')
    if full_gain:require(bool((saved['gain']==1).all()),'full gain failed')
    records=[]
    for i,c in enumerate(cases):
        pred=int(logits[i].argmax());a=VOCAB[c['new']];other=logits[i].clone();other[a]=-torch.inf
        records.append({'case_id':c['case_id'],'key':c['key'],'old':c['old'],'new':c['new'],'panel':c['panel'],
            'heads':heads,'zero_reads':zero_reads,'full_gain':full_gain,'prediction':TOKENS[pred],'correct':pred==a,
            'logits':logits[i].tolist(),'margin':float(logits[i,a]-other.max()),
            'top_two_margin':float(logits[i].topk(2).values[0]-logits[i].topk(2).values[1]),
            'raw_head_errors':{h:int(((cap[h][i].argmax(-1)!=batch[h][i])&(batch[h][i]>=0)).sum()) for h in HEADS},
            'effective_final_gate':float(saved['gain'][i]),'effective_final_edit':int(saved['edit'][i]),
            'query_preserved':bool(model.last_preservation[i]),'untouched_set_preserved':bool(saved['untouched'][i]),
            'target_retained_until_query':bool((saved['after_set'][i]==saved['before_query'][i]).all())})
    states=torch.stack([saved[n] for n in ('after_set','before_query','query_state')],1).cpu().numpy()
    require(bool(np.isfinite(states).all()),'nonfinite states')
    return records,states,{'steps':saved['steps'],'zero_hook_calls':calls['zero_reads'],'gain_hook_calls':calls['full_gain']}

def prepare():
    require(not (ROOT/'freeze.json').exists(),'already frozen')
    pins=parent_pins();cases=generate();validate(cases)
    (ROOT/'checkpoints').mkdir()
    for r in read_json(PARENT/'training_closed.json')['runs']:
        shutil.copyfile(PARENT/'runs'/r['name']/r['run_id']/'output/checkpoint.pt',ROOT/'checkpoints'/(r['name']+'.pt'))
    rows=[]
    for c in cases:
        make_batch([c])
        rows.append({'record_id':c['case_id'],'episode_id':c['case_id'],'subject_id':'one-correlated-old-new-grid',
            'observed_at':CUTOFF,'available_at':CUTOFF,'observation':{'case':c},'action':None,'outcome':{'status':'unknown','verified':False}})
    (ROOT/'source.jsonl').write_bytes(b''.join(canonical(r) for r in rows))
    write_json(ROOT/'source_spec.json',{'dataset_id':'overwrite-paths-051','source_id':'generated/overwrite050-grid','source_version':digest(ROOT/'grid.py'),
        'adapter':'operations-jsonl','license_id':'GENERATED_RESEARCH_DEV','provenance_uri':'urn:woyengi:overwrite-paths-051','ai_use_allowed':True,
        'redistribution_allowed':True,'privacy_reviewed':True,'split_salt':'overwrite-paths-051-v1'})
    release(ROOT/'source.jsonl',ROOT/'source_spec.json',ROOT/'dataset');manifest=digest(ROOT/'dataset/manifest.json')
    files=[p for p in ROOT.iterdir() if p.suffix in ('.py','.md','.json','.jsonl')]+list((ROOT/'dataset').iterdir())+list((ROOT/'checkpoints').iterdir())
    write_json(ROOT/'freeze.json',{'files':{p.relative_to(ROOT).as_posix():digest(p) for p in files},'parent_pins':pins,'runtime':runtime(),'manifest_sha256':manifest,'model_fits':0})
    write_json(ROOT/'recipe.json',{'schema':'trainlab.experiment.v2','name':'overwrite-paths-051','description':'Frozen native old/new pathway factorial',
        'command':[sys.executable,'run.py','infer'],'seed':0,'timeout_seconds':180,'source_kind':'CONSTRUCTED','split_identity':manifest,
        'inputs':[p.relative_to(ROOT).as_posix() for p in files]+['freeze.json'],
        'artifacts':['predictions.jsonl','state_traces.npz','summary.json','weight_receipt.json','batch_receipts.json'],'metrics':{'predictions':'max'},
        'budget':{'model_fits':0,'cases':288,'checkpoints':6,'conditions':8},'workload':{'kind':'test','training_method':None,'execution':'single_process'},'checks':{}})
    write_json(ROOT/'dispatch.json',{'freeze':digest(ROOT/'freeze.json'),'recipe':digest(ROOT/'recipe.json')})
    return {'dispatch_sha256':digest(ROOT/'dispatch.json'),'dataset':Dataset(ROOT/'dataset',manifest_sha256=manifest).summary()}

def infer():
    f=check();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ds=Dataset(ROOT/'dataset',manifest_sha256=f['manifest_sha256'])
    cases=[r['features']['case'] for r in ds.examples('train',cutoff=CUTOFF,feature_fields=['case'])];validate(cases)
    records=[];states=[];receipts=[];batches=[]
    with torch.inference_mode():
        for r in read_json(PARENT/'training_closed.json')['runs']:
            p=ROOT/'checkpoints'/(r['name']+'.pt');require(digest(p)==r['checkpoint_sha256'],'checkpoint drift')
            model=build(r['arm']);model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['model']);model.eval();before=weights_pin(model)
            for heads,zero,full in CONDITIONS:
                for panel in ('baseline','overwrite'):
                    group=[c for c in cases if c['panel']==panel]
                    for start in range(0,len(group),32):
                        part,traces,observed=assay_batch(model,group[start:start+32],heads,zero,full)
                        for rec in part:rec.update(arm=r['arm'],seed=r['seed'],trace_index=len(records));records.append(rec)
                        states.append(traces)
                        batches.append({'arm':r['arm'],'seed':r['seed'],'heads':heads,'zero_reads':zero,'full_gain':full,'panel':panel,'rows':len(part),'final_set_token':2 if panel=='baseline' else 5,**observed})
            after=weights_pin(model);require(before==after,'weights changed');receipts.append({'arm':r['arm'],'seed':r['seed'],'checkpoint_sha256':digest(p),'before':before,'after':after})
    require(len(records)==13824 and len(batches)==432,'prediction/batch count');check();out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    np.savez_compressed(out/'state_traces.npz',states=np.concatenate(states),labels=np.array(['after_set','before_query','query_state']))
    write_json(out/'batch_receipts.json',batches)
    write_json(out/'summary.json',summarize(records));write_json(out/'weight_receipt.json',receipts);write_json(out/'metrics.json',{'predictions':len(records)})

def finalize():
    f=check();paths=list((ROOT/'runs').iterdir());require(len(paths)==1,'unexpected run count');path=paths[0];receipt=read_json(path/'run.json')
    require(receipt['status']=='succeeded' and verify_run(path)['valid'],'failed receipt')
    predictions=[json.loads(line) for line in (path/'output/predictions.jsonl').open()];result=summarize(predictions)
    require(result==json.loads((path/'output/summary.json').read_text()),'summary disagreement');write_json(ROOT/'RESULT.json',result)
    lab=Controller(ROOT/'research_lab');manifest=f['manifest_sha256'];import_forge(lab,ROOT/'dataset',manifest);import_trainlab(lab,path)
    ev=EvidenceRecord(evidence_type='research.overwrite_pathways',outcome=EvidenceOutcome.OBSERVED,visibility=EvidenceVisibility.OPERATOR_PRIVATE,
        claim='Frozen checkpoint pathway intervention diagnostic, not learned architectural success.',subjects=(EvidenceSubjectRef(kind='dataset',subject_id='overwrite-old-new-grid',content_sha256=manifest),),
        producer=EvidenceProducerRef(producer_id='veritas-lifecycle-semantics',producer_version='1',content_sha256=_runtime()['source_trees']['investigation_world']),
        artifacts=(EvidenceArtifactRef(artifact_id='overwrite-predictions',content_sha256=digest(path/'output/predictions.jsonl')),),
        observed_at=datetime.fromisoformat(receipt['finished_at']),provenance={'scope':'FROZEN_FACTORIAL_DIAGNOSTIC','model_fits':0})
    write_json(ROOT/'evidence.json',ev.model_dump(mode='json'));put_node(lab,'overwrite-paths-051','finding',result);put_node(lab,'overwrite-paths-evidence-051','evidence',ev.model_dump(mode='json'))
    for target in ('overwrite-paths-evidence-051','forge.'+manifest[:24],'trainlab.'+receipt['id']):link(lab,'overwrite-paths-051','derived_from',target)
    write_json(ROOT/'knowledge.json',retrieve(lab));return {'run_id':receipt['id'],'predictions':len(predictions),'receipt_sha256':digest(path/'run.json')}

def execute(pin):
    require(digest(ROOT/'dispatch.json')==pin,'dispatch pin');d=read_json(ROOT/'dispatch.json')
    require(d=={'freeze':digest(ROOT/'freeze.json'),'recipe':digest(ROOT/'recipe.json')},'dispatch drift');check();require(not (ROOT/'runs').exists(),'no silent retry')
    r=run_experiment(ROOT/'recipe.json',ROOT/'runs');require(r['status']=='succeeded','failed inference; retain attempt');return finalize()

if __name__=='__main__':
    if sys.argv[1]=='prepare':print(json.dumps(prepare(),indent=2))
    elif sys.argv[1]=='infer':infer()
    elif sys.argv[1]=='run':print(json.dumps(execute(sys.argv[2]),indent=2))
    elif sys.argv[1]=='finalize':print(json.dumps(finalize(),indent=2))
    else:raise ValueError('unknown mode')
