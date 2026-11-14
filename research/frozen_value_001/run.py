"""Frozen-checkpoint research assay; uses existing ecosystem without modifying it."""
from pathlib import Path
import sys,json,hashlib,os,shutil
from collections import defaultdict
from datetime import datetime
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'lifecycle_learning_001'
sys.path.insert(0,str(PARENT))
import torch
from data import encode,history_map,VOCAB,TOKENS,HEADS,CUTOFF
from models import build,tensors
from forge_data.core import canonical,release
from forge_data.dataset import Dataset
from trainlab.runner import digest,read_json,write_json,run_experiment,verify_run
from research_lab.ecosystem import _runtime
from research_lab.core import Controller
from research_lab.adapters import import_forge,import_trainlab
from research_lab.evidence import put_node,link,retrieve
from investigation_world.benchmark.lifecycle_observability import replay
from investigation_world.evidence import EvidenceRecord,EvidenceOutcome,EvidenceVisibility,EvidenceSubjectRef,EvidenceProducerRef,EvidenceArtifactRef
PANELS=('single','filled_last','filled_first','overwrite')

def require(x,message):
    if not x:raise ValueError(message)


def generate():
    rows=[]
    for key in range(4):
        target=f'r{key}'
        for value in range(8):
            new=f'v{value}';old=f'v{(value+1)%8}'
            background=[{'op':'set','key':f'r{j}','value':f'v{(value+i+2)%8}'}
                for i,j in enumerate(j for j in range(4) if j!=key)]
            target_event={'op':'set','key':target,'value':new}
            streams={'single':[target_event],'filled_last':background+[target_event],
                'filled_first':[target_event]+background,
                'overwrite':background+[{'op':'set','key':target,'value':old},target_event]}
            for panel in PANELS:
                identity=f'{panel}-r{key}-v{value}'
                memory={'schema':'woyengi.lifecycle.v1','keys':[f'r{i}' for i in range(4)],
                    'events':streams[panel],'composed_keys':[target]}
                rows.append({'record_id':identity,'episode_id':identity,'subject_id':'fixed-correlated-probe-grid',
                    'observed_at':CUTOFF,'available_at':CUTOFF,'observation':{'memory':memory,
                    'probe':{'panel':panel,'query':target,'expected':new,'old':old}},
                    'action':None,'outcome':{'status':'unknown','verified':False}})
    return rows


def parent_pins():
    require(digest(PARENT/'freeze.json')=='449e0aa06076e99547ad78f70a8a9206c6887cd2346fde1a25d7a085bf309d33','parent freeze changed')
    require(digest(PARENT/'training_closed.json')=='a181cb5c3fca3d34a761c98cf2ca03f282618fccc9cef0fca73b634dc49deaac','parent closure changed')
    approved=read_json(PARENT/'freeze.json')
    require(approved['runtime']=={**runtime(),'torch_file':str(Path(torch.__file__).resolve())},'parent runtime changed')
    for name in ('data.py','models.py','native_csm.py'):
        require(digest(PARENT/name)==approved['files'][name],'parent source changed: '+name)
    closure=read_json(PARENT/'training_closed.json')
    pins={'freeze.json':digest(PARENT/'freeze.json'),'training_closed.json':digest(PARENT/'training_closed.json')}
    for name in ('data.py','models.py','native_csm.py'):pins[name]=digest(PARENT/name)
    for r in closure['runs']:
        path=f"runs/{r['name']}/{r['run_id']}"
        require(verify_run(PARENT/path)['valid'],'parent receipt failed')
        require(digest(PARENT/path/'output/checkpoint.pt')==r['checkpoint_sha256'],'checkpoint changed')
        pins[path+'/output/checkpoint.pt']=r['checkpoint_sha256']
        pins[path+'/run.json']=digest(PARENT/path/'run.json')
    return pins


def runtime():return {**_runtime(),'torch':torch.__version__}


def check():
    frozen=read_json(ROOT/'freeze.json')
    require(frozen['runtime']==runtime(),'runtime drift')
    require(frozen['parent_pins']==parent_pins(),'parent drift')
    for name,pin in frozen['files'].items():require(digest(ROOT/name)==pin,'input drift: '+name)
    return frozen


def prepare():
    require(not (ROOT/'freeze.json').exists(),'already prepared')
    pins=parent_pins()
    (ROOT/'checkpoints').mkdir()
    for r in read_json(PARENT/'training_closed.json')['runs']:
        source=PARENT/'runs'/r['name']/r['run_id']/'output/checkpoint.pt'
        destination=ROOT/'checkpoints'/(r['name']+'.pt')
        shutil.copyfile(source,destination)
        require(digest(destination)==r['checkpoint_sha256'],'checkpoint copy changed')
    rows=generate()
    for row in rows:
        m=row['observation']['memory'];p=row['observation']['probe']
        require(replay(m)==history_map(m),'semantic disagreement')
        require(replay(m)[p['query']]==p['expected'],'wrong target')
    (ROOT/'source.jsonl').write_bytes(b''.join(canonical(r) for r in rows))
    write_json(ROOT/'source_spec.json',{'dataset_id':'frozen-value-001','source_id':'generated/frozen-value-001',
        'source_version':digest(ROOT/'run.py'),'adapter':'operations-jsonl','license_id':'GENERATED_RESEARCH_DEV',
        'provenance_uri':'urn:woyengi:frozen-value-001','ai_use_allowed':True,'redistribution_allowed':True,
        'privacy_reviewed':True,'split_salt':'frozen-value-001-v1'})
    release(ROOT/'source.jsonl',ROOT/'source_spec.json',ROOT/'dataset')
    manifest=digest(ROOT/'dataset/manifest.json')
    files=[p for p in ROOT.iterdir() if p.suffix in ('.py','.md','.json','.jsonl')]+list((ROOT/'dataset').iterdir())+list((ROOT/'checkpoints').iterdir())
    write_json(ROOT/'freeze.json',{'files':{p.relative_to(ROOT).as_posix():digest(p) for p in files},
        'parent_pins':parent_pins(),'runtime':runtime(),'manifest_sha256':manifest,'model_fits':0})
    write_json(ROOT/'recipe.json',{'schema':'trainlab.experiment.v2','name':'frozen-value-001',
        'description':'Frozen32key/value grid in four contexts; no optimizer',
        'command':[sys.executable,'run.py','infer'],'seed':0,'timeout_seconds':120,
        'source_kind':'CONSTRUCTED','split_identity':manifest,
        'inputs':[p.relative_to(ROOT).as_posix() for p in files]+['freeze.json'],
        'artifacts':['predictions.json','summary.json','checkpoint_receipt.json'],
        'metrics':{'predictions':'max'},'budget':{'model_fits':0,'cases':128,'checkpoints':6,'conditions':2},
        'workload':{'kind':'test','training_method':None,'execution':'single_process'},'checks':{}})
    write_json(ROOT/'dispatch.json',{'freeze':digest(ROOT/'freeze.json'),'recipe':digest(ROOT/'recipe.json')})
    return {'dispatch_sha256':digest(ROOT/'dispatch.json'),'dataset':Dataset(ROOT/'dataset',manifest_sha256=manifest).summary()}


def weights_pin(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def summarize(records):
    totals=[]
    for arm in ('csm','pointer_slot'):
        for seed in (101,202,303):
            for condition in ('normal','supplied_control'):
                for panel in PANELS:
                    part=[r for r in records if (r['arm'],r['seed'],r['condition'],r['panel'])==(arm,seed,condition,panel)]
                    require(len(part)==32,'incomplete panel')
                    totals.append({'arm':arm,'seed':seed,'condition':condition,'panel':panel,
                        'correct':sum(r['correct'] for r in part),'total':32,
                        'query_preserved':sum(r['query_preserved'] for r in part),
                        'raw_control_error_cases':sum(sum(r['raw_head_errors'].values())>0 for r in part),
                        'old_value_answers':sum(r['prediction']==r['old_value'] for r in part) if panel=='overwrite' else None,
                        'by_value':{f'v{i}':sum(r['correct'] for r in part if r['expected']==f'v{i}') for i in range(8)},
                        'by_key':{f'r{i}':sum(r['correct'] for r in part if r['query']==f'r{i}') for i in range(4)}})
    paired=[]
    for arm in ('csm','pointer_slot'):
        for seed in (101,202,303):
            for condition in ('normal','supplied_control'):
                lookup={(r['panel'],r['query'],r['expected']):r for r in records if (r['arm'],r['seed'],r['condition'])==(arm,seed,condition)}
                for a,b in (('single','filled_last'),('filled_first','filled_last'),('filled_last','overwrite')):
                    pairs=[(lookup[a,f'r{k}',f'v{v}'],lookup[b,f'r{k}',f'v{v}']) for k in range(4) for v in range(8)]
                    paired.append({'arm':arm,'seed':seed,'condition':condition,'from':a,'to':b,
                        'changed':sum(x['prediction']!=y['prediction'] for x,y in pairs),
                        'rescued':sum(not x['correct'] and y['correct'] for x,y in pairs),
                        'harmed':sum(x['correct'] and not y['correct'] for x,y in pairs),
                        'maximum_absolute_logit_difference':max(max(abs(u-v) for u,v in zip(x['logits'],y['logits'])) for x,y in pairs)})
    return {'experiment':'FROZEN-VALUE-001','totals':totals,'paired':paired,'predictions':len(records),
        'model_fits':0,'scope':'POST_HOC_FROZEN_CHECKPOINT_FAILURE_LOCALIZATION; NO_QUALIFICATION_OR_COST_CLAIM'}


def infer():
    frozen=check();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ds=Dataset(ROOT/'dataset',manifest_sha256=frozen['manifest_sha256'])
    rows=ds.examples('train',cutoff=CUTOFF,feature_fields=['memory','probe'])
    require(len(rows)==128,'grid incomplete')
    records=[];weight_receipts=[]
    with torch.inference_mode():
        for receipt in read_json(PARENT/'training_closed.json')['runs']:
            checkpoint=ROOT/'checkpoints'/(receipt['name']+'.pt')
            model=build(receipt['arm']);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
            before=weights_pin(model)
            for condition in ('normal','supplied_control'):
                for panel in PANELS:
                    group=[r for r in rows if r['features']['probe']['panel']==panel]
                    examples=[]
                    for row in group:
                        memory=row['features']['memory'];probe=row['features']['probe'];truth=replay(memory)
                        require(truth==history_map(memory),'independent oracle mismatch')
                        examples.append({**encode(memory,probe['query']),'answer':VOCAB[truth[probe['query']]]})
                    batch=tensors(examples)
                    logits,cap=model.sequence(batch['ids'],collect=True,supplied=batch if condition=='supplied_control' else None)
                    require(bool(torch.isfinite(logits).all()),'nonfinite inference')
                    for i,row in enumerate(group):
                        p=row['features']['probe'];pred=int(logits[i].argmax());answer=examples[i]['answer']
                        others=logits[i].clone();others[answer]=-torch.inf
                        errors={h:int(((cap[h][i].argmax(-1)!=batch[h][i])&(batch[h][i]>=0)).sum()) for h in HEADS}
                        records.append({'arm':receipt['arm'],'seed':receipt['seed'],'condition':condition,
                            'record_id':row['record_id'],'panel':panel,'query':p['query'],'expected':p['expected'],
                            'old_value':p['old'],'prediction':TOKENS[pred],'correct':pred==answer,
                            'query_preserved':bool(model.last_preservation[i]),'raw_head_errors':errors,
                            'logits':logits[i].tolist(),'correct_logit_margin':float(logits[i,answer]-others.max()),
                            'raw_query_gain':float(cap['write_gate'][i,-1].sigmoid().item())})
            after=weights_pin(model);require(before==after,'weights changed')
            weight_receipts.append({'arm':receipt['arm'],'seed':receipt['seed'],'checkpoint_sha256':digest(checkpoint),
                'weights_before':before,'weights_after':after})
    check();out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    write_json(out/'predictions.json',records);write_json(out/'summary.json',summarize(records))
    write_json(out/'checkpoint_receipt.json',weight_receipts);write_json(out/'metrics.json',{'predictions':len(records)})


def execute(pin):
    require(digest(ROOT/'dispatch.json')==pin,'dispatch pin mismatch');d=read_json(ROOT/'dispatch.json')
    require(digest(ROOT/'freeze.json')==d['freeze'] and digest(ROOT/'recipe.json')==d['recipe'],'dispatch drift');check()
    require(not (ROOT/'runs').exists(),'do not retry silently')
    receipt=run_experiment(ROOT/'recipe.json',ROOT/'runs')
    require(receipt['status']=='succeeded','failed diagnostic: retain attempt')
    path=ROOT/'runs'/receipt['id'];require(verify_run(path)['valid'],'run integrity')
    predictions=read_json(path/'output/predictions.json')
    result=summarize(predictions);require(result==read_json(path/'output/summary.json'),'aggregate mismatch')
    write_json(ROOT/'RESULT.json',result)
    manifest=check()['manifest_sha256'];lab=Controller(ROOT/'research_lab');import_forge(lab,ROOT/'dataset',manifest);import_trainlab(lab,path)
    ev=EvidenceRecord(evidence_type='research.frozen_value_probe',outcome=EvidenceOutcome.OBSERVED,visibility=EvidenceVisibility.OPERATOR_PRIVATE,
        claim='Frozen-checkpoint value discrimination across controlled history contexts, not architecture qualification.',
        subjects=(EvidenceSubjectRef(kind='dataset',subject_id='frozen-value-grid',content_sha256=manifest),),
        producer=EvidenceProducerRef(producer_id='veritas-lifecycle-semantics',producer_version='1',content_sha256=_runtime()['source_trees']['investigation_world']),
        artifacts=(EvidenceArtifactRef(artifact_id='frozen-value-predictions',content_sha256=digest(path/'output/predictions.json')),),
        observed_at=datetime.fromisoformat(receipt['finished_at']),provenance={'scope':'GENERATED_FROZEN_MODEL_DIAGNOSTIC','model_fits':0})
    write_json(ROOT/'evidence.json',ev.model_dump(mode='json'));put_node(lab,'frozen-value-001','finding',result)
    put_node(lab,'frozen-value-evidence-001','evidence',ev.model_dump(mode='json'))
    link(lab,'frozen-value-001','derived_from','frozen-value-evidence-001');link(lab,'frozen-value-001','derived_from','forge.'+manifest[:24]);link(lab,'frozen-value-001','derived_from','trainlab.'+receipt['id'])
    write_json(ROOT/'knowledge.json',retrieve(lab))
    return {'predictions':len(predictions),'model_fits':0,'run_id':receipt['id'],'receipt_sha256':digest(path/'run.json')}

if __name__=='__main__':
    if sys.argv[1]=='prepare':print(json.dumps(prepare(),indent=2))
    elif sys.argv[1]=='infer':infer()
    elif sys.argv[1]=='run':print(json.dumps(execute(sys.argv[2]),indent=2))
    else:raise ValueError('unknown mode')
