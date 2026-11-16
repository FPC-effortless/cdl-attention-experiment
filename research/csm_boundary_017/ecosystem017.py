"""Native ecosystem glue for NEW 017. Preparation never launches training.
Forge is a provenance container: its automatic partitions are NOT the study split.
The separately hashed data.json and explicit study map remain authoritative.
"""
import hashlib,json,os,subprocess,sys
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
def freeze_forge(root, *, redistribution_allowed=False):
 from forge_data.core import release,canonical
 root=Path(root).resolve();data=json.loads((root/'data.json').read_text());dst=root/'forge_017'
 if dst.exists():raise ValueError('immutable Forge release already exists')
 rows=[];mapping={}
 for split,panels in data.items():
  for panel,cases in panels.items():
   for c in cases:
    identity=c['case_id'];mapping[identity]={'split':split,'panel':panel,'case_sha256':hashlib.sha256(canonical(c)).hexdigest()}
    rows.append({'episode_id':identity,'record_id':identity,'subject_id':identity,'observed_at':'2026-11-16T00:00:00Z','available_at':'2026-11-16T00:00:00Z','observation':{'study_case':c},'action':None,'outcome':{'status':'unknown','verified':False}})
 if len(mapping)!=len(rows):raise ValueError('duplicate study case ids')
 spec={'dataset_id':'csm-boundary-017-provenance','source_id':'constructed/csm-boundary-017','source_version':sha(root/'data.json'),'adapter':'operations-jsonl','license_id':'PRIVATE_RESEARCH','provenance_uri':'local:constructed-017','ai_use_allowed':True,'redistribution_allowed':redistribution_allowed,'privacy_reviewed':True,'split_salt':'017-provenance-only-not-study-splits'}
 write(root/'forge_source.json',spec);(root/'forge_source.jsonl').write_bytes(b''.join(canonical(r) for r in rows));release(root/'forge_source.jsonl',root/'forge_source.json',dst)
 write(root/'study_split_map.json',{'data_sha256':sha(root/'data.json'),'cases':mapping,'scope':'Explicit TRAIN/DEV study map; Forge automatic partitions are provenance-only, not evaluated or training-selected.'})
 from research_lab.core import Controller
 from research_lab.adapters import import_forge
 import_forge(Controller(root/'research_lab_017'),dst,sha(dst/'manifest.json'))
 return sha(dst/'manifest.json')

def freeze_study_private(root):
 """Requires separately reviewed/installed Forge private_snapshot addition.
 Unlike temporal release, no automatic redistribution or repartition is implied.
 """
 from forge_data.private_snapshot import freeze_private,verify_private
 from research_lab.core import Controller
 from research_lab.evidence import put_node
 root=Path(root).resolve();data=json.loads((root/'data.json').read_text());mapping={}
 for split,panels in data.items():
  for panel,cases in panels.items():
   for c in cases:
    if c['case_id'] in mapping:raise ValueError('duplicate case ID')
    mapping[c['case_id']]={'split':split,'panel':panel}
 write(root/'study_split_map.json',{'data_sha256':sha(root/'data.json'),'cases':mapping,'scope':'Constructed explicit017 TRAIN/DEV; no protected TEST or automatic resplitting'})
 files=['data.json','EVENT_COUNTS.json','study_split_map.json']
 spec={'snapshot_id':'csm-boundary-017-private-inputs','source_version':sha(root/'data.json'),'ai_use_allowed':True,'redistribution_allowed':False,'privacy_reviewed':True,'visibility':'operator_private','purpose':'User-authorized local constructed017 research with exact explicit TRAIN/DEV map','excluded_claims':'No redistribution, native temporal-release equivalence, independently attested rights, split independence or scientific qualification'}
 freeze_private(root,files,spec,root/'forge_017');pin=sha(root/'forge_017/manifest.json');report=verify_private(root/'forge_017',pin)
 put_node(Controller(root/'research_lab_017'),'forge.private.'+pin[:24],'evidence',report)
 return pin

def verify_freeze(root):
 root=Path(root);path=root/'forge_017';manifest=json.loads((path/'manifest.json').read_text());pin=sha(path/'manifest.json')
 if manifest.get('schema')=='forge.private-snapshot.v1':
  from forge_data.private_snapshot import verify_private
  verify_private(path,pin)
  for name in ('data.json','EVENT_COUNTS.json','study_split_map.json'):
   if sha(root/name)!=sha(path/name):raise ValueError('frozen private input mismatch')
 else:
  from forge_data.dataset import Dataset
  Dataset(path,manifest_sha256=pin)
  if manifest['source_version']!=sha(root/'data.json'):raise ValueError('provenance data hash mismatch')
 return pin

def prepare(root,fit_path,review_path):
 """Build pinned native adequacy recipe; requires real external review evidence.
 The reviewer approves criteria for bounded fixed-budget study, not success claims.
 """
 from trainlab.runner import load_config
 from trainlab.research_adequacy import recipe_identity,assess,verify_report
 from trainlab.execution_adequacy import validate_execution
 from model import Model,ARMS
 from prepare017 import source_hashes,STUDY_SPEC
 root=Path(root).resolve();fit_path=Path(fit_path);review_path=Path(review_path)
 if fit_path.parent.resolve()!=root or review_path.parent.resolve()!=root:raise ValueError('inputs must be directly within candidate root')
 fit=json.loads(fit_path.read_text());review=json.loads(review_path.read_text())
 if review.get('approved') is not True or review.get('author')==review.get('reviewer') or not review.get('reviewer') or not review.get('basis'):raise ValueError('actual distinct independent review required')
 if fit['arm'] not in ARMS or fit['seed'] not in (101,202,303) or fit['updates']!=1024:raise ValueError('unapproved fit')
 stop=fit.get('stop_update',1024);start=json.loads((root/fit['resume_checkpoint']).read_text())['completed_updates'] if fit.get('resume_checkpoint') else 0
 if not(0<=start<stop<=1024 and start%64==stop%64==0 and stop-start==64):raise ValueError('exact64update segments required')
 data=json.loads((root/'data.json').read_text());manifest=root/'forge_017/manifest.json';mapping=json.loads((root/'study_split_map.json').read_text())
 if mapping['data_sha256']!=sha(root/'data.json'):raise ValueError('study data changed')
 verify_freeze(root)
 counts={s:sum(map(len,p.values())) for s,p in data.items()}
 if counts!={'train':512,'dev':576}:raise ValueError('unexpected study roster')
 params=sum(p.numel() for p in Model(fit['arm']).parameters());obs={'parameters':params,'total_planned_updates':1024,'segment_updates':64,'batch_size':16,'train_cases':512,'dev_cases':576,'protected_test_cases':0,'seeds':3}
 stem=f"recipe_{fit['arm']}_{fit['seed']}_{stop:04d}";obsname=stem+'_plan.json';contractname=stem+'_adequacy.json';recipe_name=stem+'.json';write(root/obsname,obs)
 inputs=sorted(p.name for p in root.glob('*.py'))+['data.json','EVENT_COUNTS.json','PROTOCOL.md','TIMING_PLAN.json','ADMISSION_017.json','study_split_map.json','forge_017/manifest.json',fit_path.name,review_path.name,obsname]
 # Required exact prior-generation bytes are retained inputs; review record binds them.
 if fit.get('resume_checkpoint'):
  cp=Path(fit['resume_checkpoint'])
  if cp.is_absolute() or '..' in cp.parts:raise ValueError('resume input escapes root')
  rec=json.loads((root/cp).read_text());gen=rec['generation'];inputs.extend([str(cp),str(cp.parent/(cp.name+'.generations')/(gen+'.pt')),str(cp.parent/(cp.name+'.generations')/(gen+'.json')),fit['durable_resume_receipt']])
 inputs=sorted(set(inputs));pins={n:sha(root/n) for n in inputs}
 if review.get('source_sha256')!=source_hashes(root) or review.get('protocol')!=STUDY_SPEC:raise ValueError('review source/protocol closure changed')
 recipe={'schema':'trainlab.experiment.v2','name':stem,'description':'New017 bounded64update segment; not016 recovery or scientific qualification','command':[sys.executable,'ecosystem017.py','worker',recipe_name,fit_path.name],'seed':fit['seed'],'timeout_seconds':1800,'source_kind':'CONSTRUCTED','split_identity':'explicit017TRAINDEV:'+sha(root/'study_split_map.json'),'inputs':inputs+[contractname],'artifacts':['progress.jsonl','result.json','schedule.json','resume.pt','resume_payload.pt','resume_generation.json','resume_bundle.zip','execution_observations.json'],'metrics':{'completed_updates':'max','new_updates':'max','token_positions':'max'},'budget':{'updates':64,'total_fit_updates':1024,'batch_size':16},'execution_purpose':'research','workload':{'kind':'train','training_method':'supervised','execution':'single_process'},'checks':{'completed_updates':{'min':stop,'max':stop},'new_updates':{'min':64,'max':64}}}
 if stop==1024:recipe['artifacts'].append('endpoint.pt')
 dims={'model_size':('parameters',params,'Exact arm parameter count, not capacity sufficiency or FLOP equality.'),'training_duration':('total_planned_updates',1024,'Fixed bounded budget; not convergence.'),'datasets':('train_cases',512,'Constructed explicit TRAIN roster; no broad generalization claim.'),'training_regime':('batch_size',16,'Matched sampling/exposure declarations; schedule replay separately checked.')}
 dimensions={k:{'configuration':f'{field}={v}','justification':lim,'limitations':lim,'criteria':[{'id':field,'rationale':lim,'evidence':obsname,'field':field,'operator':'eq','threshold':v}]} for k,(field,v,lim) in dims.items()}
 execution={'model_size':[('parameters',params)],'training_duration':[('completed_updates',stop),('new_updates',64)],'datasets':[('train_cases',512),('dev_cases',576),('protected_test_cases',0)],'training_regime':[('batch_size',16),('schedule_rows',64)]}
 contract={'schema':'trainlab.research-adequacy.v1','claim':'Bounded fixed-budget comparison of learned parsers with supplied typed executor','claim_kind':'fixed_budget_comparison','scope':'New017 constructed four-record TRAIN/DEV; all failed seeds retained','excluded_claims':'No convergence, nativeCSM superiority, real-language or broad generalization, equalFLOPs, host recovery or independent execution attestation','recipe_sha256':recipe_identity(recipe),'inputs':pins,'review':{'author':review['author'],'reviewer':review['reviewer'],'basis':review['basis'],'criteria_frozen_before_evaluation':True},'dimensions':dimensions,'execution':{'artifact':'execution_observations.json','dimensions':{k:[{'field':f,'operator':'eq','threshold':v,'rationale':'Reconcile declared actual bounded segment observations; not independent attestation'} for f,v in rows] for k,rows in execution.items()}}}
 write(root/contractname,contract);recipe['research_adequacy']={'file':contractname,'sha256':sha(root/contractname)};write(root/recipe_name,recipe);config=load_config(root/recipe_name);allpins={n:sha(root/n) for n in config['inputs']};report=assess(config,root,allpins);verify_report(report,config,allpins);validate_execution(contract,config);write(root/(stem+'_preflight.json'),report);return root/recipe_name

def worker(recipe_path,fit_path):
 from trainlab.runner import load_config
 from trainlab.research_adequacy import recipe_identity
 from model import Model
 root=Path.cwd();recipe=load_config(Path(recipe_path));fit=json.loads(Path(fit_path).read_text());pins={n:sha(root/n) for n in recipe['inputs']};out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
 env=dict(os.environ,CSM_FIT_CONFIG=str(Path(fit_path).resolve()));subprocess.run([sys.executable,'worker.py'],check=True,env=env)
 result=json.loads((out/'result.json').read_text());schedule=json.loads((out/'schedule.json').read_text());data=json.loads((root/'data.json').read_text())
 observations={'parameters':sum(p.numel() for p in Model(fit['arm']).parameters()),'completed_updates':result['finished_update'],'new_updates':result['finished_update']-result['started_update'],'schedule_rows':len(schedule),'batch_size':16,'train_cases':sum(map(len,data['train'].values())),'dev_cases':sum(map(len,data['dev'].values())),'protected_test_cases':0}
 write(out/'execution_observations.json',{'schema':'trainlab.execution-adequacy-observations.v1','recipe_sha256':recipe_identity(recipe),'inputs':pins,'producer':{'file':'ecosystem017.py','sha256':pins['ecosystem017.py']},'observations':observations})

def record_run(root,run_dir):
 from research_lab.core import Controller
 from research_lab.adapters import import_trainlab
 from research_lab.evidence import put_node,link
 from trainlab.runner import verify_run
 from investigation_world.evidence import EvidenceRecord,EvidenceOutcome,EvidenceVisibility,EvidenceProducerRef,EvidenceSubjectRef,EvidenceArtifactRef
 from datetime import datetime
 run_dir=Path(run_dir);record=json.loads((run_dir/'run.json').read_text());check=verify_run(run_dir)
 if not check['valid']:raise ValueError('TrainLab record integrity failed; preserve quarantined run')
 lab=Controller(Path(root)/'research_lab_017');import_trainlab(lab,run_dir)
 ev=EvidenceRecord(evidence_type='csm017.segment_execution',outcome=EvidenceOutcome.OBSERVED,visibility=EvidenceVisibility.OPERATOR_PRIVATE,claim='Recorded bounded segment, not model qualification or independently attested compute',subjects=(EvidenceSubjectRef(kind='run',subject_id=record['id'],content_sha256=sha(run_dir/'run.json')),),producer=EvidenceProducerRef(producer_id='csm017-ecosystem',producer_version='draft1',content_sha256=sha(__file__)),artifacts=(EvidenceArtifactRef(artifact_id='trainlab_receipt',role='execution',content_sha256=sha(run_dir/'run.json'),media_type='application/json'),),observed_at=datetime.fromisoformat(record['finished_at']),provenance={'scope':'CONSTRUCTED_PRIVATE','status':record['status']})
 write(run_dir/'veritas_evidence.json',ev.model_dump(mode='json'));node='csm017.'+record['id'];put_node(lab,node,'finding',ev.model_dump(mode='json'));link(lab,node,'derived_from','trainlab.'+record['id']);return check

if __name__=='__main__':
 if len(sys.argv)==4 and sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3])
 else:raise SystemExit('Preparation APIs only; worker requires native TrainLab invocation and admission')
