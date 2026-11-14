"""Frozen learning diagnostic using the existing four-component ecosystem."""
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import hashlib,json,os,random,sys,time
import torch
from forge_data.core import canonical,release
from forge_data.dataset import Dataset
from trainlab.runner import digest,read_json,write_json,run_experiment,verify_run
from research_lab.ecosystem import _runtime
from research_lab.core import Controller
from research_lab.adapters import import_forge,import_trainlab
from research_lab.evidence import put_node,link,retrieve
from investigation_world.benchmark.lifecycle_observability import replay
from investigation_world.evidence import (EvidenceRecord,EvidenceOutcome,EvidenceVisibility,EvidenceSubjectRef,
                                          EvidenceProducerRef,EvidenceArtifactRef)
from data import generate,examples,CUTOFF,HEADS,TEMPLATES
from models import build,tensors,objective

ROOT=Path(__file__).resolve().parent
ARMS=('csm','pointer_slot');SEEDS=(101,202,303)

def require(condition,message):
    if not condition:raise ValueError(message)


def runtime():
    return {**_runtime(),'torch':torch.__version__,'torch_file':str(Path(torch.__file__).resolve())}


def verify():
    freeze=read_json(ROOT/'freeze.json')
    require(freeze['runtime']==runtime(),'runtime changed')
    for name,pin in freeze['files'].items():require(digest(ROOT/name)==pin,'source changed: '+name)
    return freeze


def dataset():
    frozen=verify()
    return Dataset(ROOT/'dataset',manifest_sha256=frozen['manifest_sha256'])


def prepare():
    require(not (ROOT/'freeze.json').exists(),'already frozen')
    rows=generate()
    for row in rows:
        from data import history_map
        require(history_map(row['observation']['memory'])==replay(row['observation']['memory']),'oracle discrepancy')
    (ROOT/'source.jsonl').write_bytes(b''.join(canonical(r) for r in rows))
    spec={'dataset_id':'lifecycle-learning-001-dev','source_id':'generated/lifecycle-learning-001',
        'source_version':digest(ROOT/'data.py'),'adapter':'operations-jsonl','license_id':'GENERATED_RESEARCH_DEV',
        'provenance_uri':'urn:woyengi:lifecycle-learning-001','ai_use_allowed':True,
        'redistribution_allowed':True,'privacy_reviewed':True,'split_salt':'lifecycle-learning-001-fixed-v1'}
    write_json(ROOT/'source_spec.json',spec)
    release(ROOT/'source.jsonl',ROOT/'source_spec.json',ROOT/'dataset')
    manifest=digest(ROOT/'dataset/manifest.json')
    files=[p for p in ROOT.iterdir() if p.suffix in ('.py','.md','.json','.jsonl')]
    files += sorted((ROOT/'dataset').iterdir())
    write_json(ROOT/'freeze.json',{'schema':'woyengi.lifecycle-learning.freeze.v1',
        'files':{p.relative_to(ROOT).as_posix():digest(p) for p in files},
        'runtime':runtime(),'manifest_sha256':manifest,'seeds':list(SEEDS),'arms':list(ARMS),
        'updates':512,'batch':32,'scope':'KNOWN_TEMPLATE_SUPERVISED_FEASIBILITY_NOT_QUALIFIED_COMPARISON'})
    recipes={}
    for seed in SEEDS:
        for arm in ARMS:
            name=f'{arm}-{seed}';recipe={'schema':'trainlab.experiment.v2','name':name,
                'description':'Fresh fixed-budget supervised mechanism-learning diagnostic',
                'command':[sys.executable,'experiment.py','train',arm,str(seed)],'seed':seed,
                'timeout_seconds':180,'source_kind':'CONSTRUCTED','split_identity':manifest,
                'inputs':sorted([p.relative_to(ROOT).as_posix() for p in files]+['freeze.json']),
                'artifacts':['checkpoint.pt','updates.jsonl','complete.json'],
                'metrics':{'final_training_loss':'min','updates':'max'},
                'budget':{'updates':512,'batch_size':32,'answer_exposures':16384,'cpu_threads':1},
                'workload':{'kind':'train','training_method':'supervised','execution':'single_process'},'checks':{}}
            write_json(ROOT/(name+'.json'),recipe);recipes[name+'.json']=digest(ROOT/(name+'.json'))
    write_json(ROOT/'dispatch.json',{'freeze_sha256':digest(ROOT/'freeze.json'),'recipes':recipes})
    return {'dispatch_sha256':digest(ROOT/'dispatch.json'),
            'dataset':Dataset(ROOT/'dataset',manifest_sha256=manifest).summary()}


def train(arm,seed):
    frozen=verify();require(arm in ARMS and seed in SEEDS,'unregistered fit')
    require(not (ROOT/'training_closed.json').exists(),'training closed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    random.seed(seed);torch.manual_seed(seed)
    rows=examples(dataset().examples('train',cutoff=CUTOFF,feature_fields=['memory','condition']))
    bank=tensors(rows);schedule=torch.randint(len(rows),(512,32),generator=torch.Generator().manual_seed(seed+900000))
    model=build(arm);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);started=time.perf_counter()
    with (out/'updates.jsonl').open('x') as log:
        for step,index in enumerate(schedule,1):
            batch={k:v[index] for k,v in bank.items()};optimizer.zero_grad(set_to_none=True)
            logits,cap=model.sequence(batch['ids'],collect=True);loss,parts=objective(logits,cap,batch)
            require(bool(torch.isfinite(loss)),'nonfinite objective');loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            require(bool(torch.isfinite(norm)),'nonfinite gradient');optimizer.step()
            entry={'step':step,'loss':float(loss.detach()),'parts':parts,'gradient_norm':float(norm),
                'batch_answer_accuracy':float((logits.argmax(-1)==batch['answer']).float().mean())}
            log.write(json.dumps(entry)+'\n');log.flush()
            if step%128==0:print(arm,seed,step,entry['batch_answer_accuracy'],flush=True)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),
                'arm':arm,'seed':seed,'updates':512},out/'checkpoint.pt')
    write_json(out/'complete.json',{'arm':arm,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),
        'updates':512,'answer_exposures':16384,'train_queries':len(rows),
        'schedule_sha256':hashlib.sha256(schedule.numpy().tobytes()).hexdigest(),
        'freeze_sha256':digest(ROOT/'freeze.json'),'checkpoint_sha256':digest(out/'checkpoint.pt'),
        'updates_sha256':digest(out/'updates.jsonl'),'seconds':time.perf_counter()-started,
        'final_batch_accuracy':entry['batch_answer_accuracy']})
    write_json(out/'metrics.json',{'final_training_loss':entry['loss'],'updates':512})


def dispatch_check(pin):
    require(digest(ROOT/'dispatch.json')==pin,'dispatch drift');dispatch=read_json(ROOT/'dispatch.json')
    require(digest(ROOT/'freeze.json')==dispatch['freeze_sha256'],'freeze drift')
    for name,expected in dispatch['recipes'].items():require(digest(ROOT/name)==expected,'recipe drift')
    verify();return dispatch


def fit_all(pin):
    dispatch_check(pin);receipts=[]
    for index,seed in enumerate(SEEDS):
        for arm in ARMS if index%2==0 else tuple(reversed(ARMS)):
            dispatch_check(pin);name=f'{arm}-{seed}';dest=ROOT/'runs'/name
            require(not dest.exists(),'existing attempt: no silent retries')
            result=run_experiment(ROOT/(name+'.json'),dest)
            require(result['status']=='succeeded','fit failed; retain evidence and stop')
            path=dest/result['id'];require(verify_run(path)['valid'],'receipt integrity')
            complete=read_json(path/'output/complete.json')
            receipts.append({'name':name,'run_id':result['id'],'receipt_sha256':digest(path/'run.json'),**complete})
    for seed in SEEDS:require(len({r['schedule_sha256'] for r in receipts if r['seed']==seed})==1,'unmatched schedules')
    write_json(ROOT/'training_closed.json',{'runs':receipts,'dispatch_sha256':pin,'fits_closed_before_validation':True})
    return {'closed_fits':len(receipts),'model_fits':6}


def aggregate(records):
    groups=defaultdict(list)
    for row in records:groups[row['record_id']].append(row['correct'])
    return {'queries':len(records),'correct':sum(r['correct'] for r in records),
        'accuracy':sum(r['correct'] for r in records)/len(records),
        'maps':len(groups),'complete_maps':sum(len(v)==4 and all(v) for v in groups.values()),
        'complete_map_accuracy':sum(len(v)==4 and all(v) for v in groups.values())/len(groups)}


def evaluate():
    verify();closure=read_json(ROOT/'training_closed.json');dispatch_check(closure['dispatch_sha256'])
    require(len(closure['runs'])==6 and closure['fits_closed_before_validation'] is True,'all fits must close')
    out=ROOT/'results';out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    data=dataset();forge=data.examples('validation',cutoff=CUTOFF,feature_fields=['memory','condition'])
    rows=examples(forge);bank=tensors(rows);summaries=[]
    for row in forge:require(replay(row['features']['memory'])==__import__('data').history_map(row['features']['memory']),'Veritas label disagreement')
    with (out/'predictions.jsonl').open('x') as log,torch.inference_mode():
        for receipt in closure['runs']:
            path=ROOT/'runs'/receipt['name']/receipt['run_id']
            require(digest(path/'run.json')==receipt['receipt_sha256'] and verify_run(path)['valid'],'run drift')
            checkpoint=path/'output/checkpoint.pt';require(digest(checkpoint)==receipt['checkpoint_sha256'],'checkpoint drift')
            model=build(receipt['arm']);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
            for condition in ('normal','erase_values','identity_links','supplied_control'):
                records=[];heads={h:[0,0] for h in HEADS}
                for start in range(0,len(rows),64):
                    batch={k:v[start:start+64] for k,v in bank.items()}
                    logits,cap=model.sequence(batch['ids'],collect=condition=='normal',
                        mutation=condition if condition in ('erase_values','identity_links') else None,
                        supplied=batch if condition=='supplied_control' else None)
                    require(bool(torch.isfinite(logits).all()),'nonfinite inference')
                    for source,pred,preserved in zip(rows[start:start+64],logits.argmax(-1).tolist(),model.last_preservation.tolist()):
                        result={k:source[k] for k in ('record_id','subject_id','query','template','focal_op','role','answer')}
                        result.update(arm=receipt['arm'],seed=receipt['seed'],condition=condition,
                                      prediction=pred,correct=pred==source['answer'],query_preserved=preserved)
                        records.append(result);log.write(json.dumps(result,sort_keys=True)+'\n')
                    for name in cap:
                        if name not in heads:continue
                        target=batch[name];mask=target>=0
                        heads[name][0]+=int(((cap[name].argmax(-1)==target)&mask).sum())
                        heads[name][1]+=int(mask.sum())
                summary={'arm':receipt['arm'],'seed':receipt['seed'],'condition':condition,
                    'overall':aggregate(records),'query_preservation':{'correct':sum(r['query_preserved'] for r in records),'total':len(records)},
                    'template_focal':{t:{op:aggregate([r for r in records if r['template']==t and r['focal_op']==op]) for op in ('copy','link')} for t in TEMPLATES},
                    'templates':{t:aggregate([r for r in records if r['template']==t]) for t in TEMPLATES},
                    'roles':{role:{'correct':sum(r['correct'] for r in records if r['role']==role),
                        'total':sum(r['role']==role for r in records)} for role in ('source','alias','snapshot','unrelated')},
                    'heads':heads}
                summaries.append(summary);print(receipt['name'],condition,summary['overall']['accuracy'],flush=True)
    report={'schema':'woyengi.lifecycle-learning-001.v1','results':summaries,
        'validation_worlds':len(forge),'validation_groups':len({r['subject_id'] for r in forge}),
        'known_templates':list(TEMPLATES),'closure_sha256':digest(ROOT/'training_closed.json'),
        'predictions_sha256':digest(out/'predictions.jsonl'),'model_fits':6,
        'scientific_status':'DEVELOPMENT_FEASIBILITY_ONLY_NOT_QUALIFIED','inference_cost':'NOT_MEASURED'}
    write_json(out/'summary.json',report)
    lab=Controller(ROOT/'research_lab');manifest=digest(ROOT/'dataset/manifest.json')
    import_forge(lab,ROOT/'dataset',manifest)
    for r in closure['runs']:import_trainlab(lab,ROOT/'runs'/r['name']/r['run_id'])
    env=EvidenceRecord(evidence_type='research.lifecycle_learning',outcome=EvidenceOutcome.OBSERVED,
        visibility=EvidenceVisibility.OPERATOR_PRIVATE,claim='Supervised one-hop known-template learning feasibility; no architecture qualification.',
        subjects=(EvidenceSubjectRef(kind='dataset',subject_id='lifecycle-learning-001-dev',content_sha256=manifest),),
        producer=EvidenceProducerRef(producer_id='veritas-lifecycle-semantics',producer_version='1',
            content_sha256=runtime()['source_trees']['investigation_world']),
        artifacts=(EvidenceArtifactRef(artifact_id='learning-predictions',role='raw_predictions',content_sha256=report['predictions_sha256']),),
        observed_at=datetime.now().astimezone(),provenance={'scope':'GENERATED_DEVELOPMENT','qualification':'NOT_QUALIFIED'})
    write_json(out/'evidence.json',env.model_dump(mode='json'))
    put_node(lab,'lifecycle-learning-001','finding',report)
    put_node(lab,'lifecycle-learning-evidence-001','evidence',env.model_dump(mode='json'))
    link(lab,'lifecycle-learning-001','derived_from','lifecycle-learning-evidence-001')
    link(lab,'lifecycle-learning-001','derived_from','forge.'+manifest[:24])
    for r in closure['runs']:link(lab,'lifecycle-learning-001','derived_from','trainlab.'+r['run_id'])
    write_json(out/'knowledge.json',retrieve(lab))
    return {'summary':str(out/'summary.json'),'predictions':report['predictions_sha256']}

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':result=prepare()
    elif mode=='train':result=train(sys.argv[2],int(sys.argv[3]))
    elif mode=='fit':result=fit_all(sys.argv[2])
    elif mode=='evaluate':result=evaluate()
    else:raise ValueError('unknown mode')
    if result is not None:print(json.dumps(result,indent=2))
