"""Draft native Forge/TrainLab/Veritas/Research Lab adapter; not run admission.
Private exact-source/data/runtime evidence, distinct from model qualification.
"""
from pathlib import Path
import hashlib,json,os,platform,subprocess,sys,time

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def relative(root,name):
 root=Path(root).resolve();raw=root/name;p=raw.resolve()
 if any(q.is_symlink() for q in [raw,*raw.parents] if q.is_relative_to(root)):raise ValueError('symlink input prohibited')
 if Path(name).is_absolute() or not p.is_relative_to(root) or not p.is_file():raise ValueError('missing or escaping input '+str(name))
 return p

def runtime():
 import torch
 deps=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
 dependency_sha=hashlib.sha256(('\n'.join(sorted(x.strip() for x in deps.splitlines() if x.strip()))+'\n').encode()).hexdigest()
 return {'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':'cuda' if torch.cuda.is_available() else 'cpu','gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'deterministic':torch.are_deterministic_algorithms_enabled()},'environment':{'torch_num_threads':torch.get_num_threads(),'pip_freeze_sha256':dependency_sha,'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'float32_matmul_precision':torch.get_float32_matmul_precision(),'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'python':platform.python_version(),'cudnn':torch.backends.cudnn.version(),'gpu_count':torch.cuda.device_count(),'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}}
def check_runtime(root):
 import torch
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 observed=runtime();expected=read(Path(root)/'RUNTIME_020.json')
 if observed!=expected:raise ValueError('actual execution runtime differs from admitted GPU runtime')
 if observed['runtime']['device']!='cuda':raise ValueError('GPU study must not silently fall back to CPU')
 return observed

def freeze_inputs(root,names):
 """Explicit allowlist only; callers must exclude credentials and protected TEST."""
 from forge_data.private_snapshot import freeze_private,verify_private
 from research_lab.core import Controller
 from research_lab.evidence import put_node
 root=Path(root).resolve();pins={n:sha(relative(root,n)) for n in names}
 if any('test' in Path(n).stem.lower() and Path(n).suffix in ('.parquet','.jsonl') for n in names):raise ValueError('protected TEST-like data name rejected')
 write(root/'INPUT_MANIFEST_020.json',{'files':pins,'scope':'Explicit source TRAIN/validation and constructed TRAIN/DEV only; no protected TEST. Per-source licences/provenance are separate retained inputs.'})
 spec={'snapshot_id':'woyengi-endtoend-020-private','source_version':sha(root/'INPUT_MANIFEST_020.json'),'ai_use_allowed':True,'redistribution_allowed':False,'privacy_reviewed':True,'visibility':'operator_private','purpose':'User-authorized end-to-end language and constructed memory comparison','excluded_claims':'No public redistribution rights, contamination-free certification or scientific qualification'}
 freeze_private(root,sorted(pins)+['INPUT_MANIFEST_020.json'],spec,root/'forge_020')
 h=sha(root/'forge_020/manifest.json');report=verify_private(root/'forge_020',h)
 put_node(Controller(root/'research_lab_020'),'forge.private.'+h[:24],'evidence',report);return h

def validate_config(c):
 if c['arm'] not in ('native','separated','memory_off','transformer') or c['seed'] not in (101,202,303):raise ValueError('unadmitted arm/seed')
 start,stop,total,segment=[c[x] for x in ('start_update','stop_update','total_updates','segment')]
 if not all(type(x) is int for x in (start,stop,total,segment)) or not 0<=start<stop<=total or segment<=0 or start%segment!=0 or stop!=min(start+segment,total):raise ValueError('invalid segment boundary')
 if bool(start)!=bool(c.get('resume_checkpoint')) or bool(start)!=bool(c.get('durable_resume_receipt')):raise ValueError('resume requires checkpoint and verified durable receipt')
 if any(type(c[k]) is not int or c[k]<=0 for k in ('width','slots','prose_context','prose_batch','task_batch','declared_parameters','planned_targets')):raise ValueError('invalid declared dimensions')
 if c['lr']<=0 or c['answer_weight']<=0:raise ValueError('invalid training scalar')
 return c

def verify_admission(root,config):
 root=Path(root);c=validate_config(config);a=read(root/'ADMISSION_020.json');review=read(root/'REVIEW_020.json')
 if a.get('approved') is not True or a.get('public_source_verified') is not True or a.get('private_backup_verified') is not True:raise ValueError('real review and durable backups required')
 if review.get('approved') is not True or not review.get('reviewer') or review.get('reviewer')==review.get('author'):raise ValueError('distinct independent reviewer required')
 if sha(root/'REVIEW_020.json')!=a.get('review_sha256'):raise ValueError('review drift')
 for n in (c['prose'],c['tasks'],c['exposure_plan']):
  if n not in a['input_sha256']:raise ValueError('training data/plan missing from frozen closure')
 plans=read(relative(root,c['exposure_plan']));planned=[p for p in plans if p['seed']==c['seed']]
 if len(planned)!=1 or planned[0]['updates']!=c['total_updates'] or planned[0]['targets']!=c['planned_targets']:raise ValueError('prospective exposure plan mismatch')
 for section in ('source_sha256','input_sha256'):
  if review[section]!=a[section]:raise ValueError('admission closure mismatch')
  for n,h in a[section].items():
   if sha(relative(root,n))!=h:raise ValueError('frozen input drift '+n)
 vendor_manifest=root/'ecosystem_vendor/VENDOR_MANIFEST.json'
 if sha(vendor_manifest)!=a.get('vendor_manifest_sha256') or a.get('vendor_manifest_sha256')!=review.get('vendor_manifest_sha256'):raise ValueError('ecosystem vendor review binding mismatch')
 vendor=read(vendor_manifest)
 for n,h in vendor['files'].items():
  if sha(relative(root/'ecosystem_vendor',n))!=h:raise ValueError('ecosystem vendor source drift '+n)
 if c.get('runtime_sha256')!=sha(root/'RUNTIME_020.json'):raise ValueError('runtime pin drift')
 # Fit-specific configuration, including budgets, must occur in reviewed roster.
 core={k:v for k,v in c.items() if k not in ('start_update','stop_update','resume_checkpoint','durable_resume_receipt')}
 if core not in review['fit_roster']:raise ValueError('fit not in independently reviewed roster')
 if c.get('resume_checkpoint'):
  cp=relative(root,c['resume_checkpoint']);d=read(relative(root,c['durable_resume_receipt']))
  if d.get('roundtrip_verified') is not True or not d.get('library_file_id','').startswith('libfile_') or d.get('checkpoint_sha256')!=sha(cp) or d.get('completed_updates')!=c['start_update']:raise ValueError('durable predecessor binding mismatch')
 return a,review

def prepare(root,config_name):
 from trainlab.runner import load_config
 from trainlab.research_adequacy import recipe_identity,assess,verify_report
 from trainlab.execution_adequacy import validate_execution
 from forge_data.private_snapshot import verify_private
 root=Path(root).resolve();c=read(relative(root,config_name));a,review=verify_admission(root,c)
 verify_private(root/'forge_020',sha(root/'forge_020/manifest.json'))
 stem=f"recipe_{c['arm']}_{c['seed']}_{c['stop_update']:06d}";plan_name=stem+'_plan.json';contract_name=stem+'_adequacy.json'
 docs=[json.loads(x)['text'].encode('utf-8') for x in relative(root,c['prose']).read_text().splitlines()]
 docs=[x for x in docs if len(x)>c['prose_context']+1];tasks=[json.loads(x) for x in relative(root,c['tasks']).read_text().splitlines()]
 if not docs or not tasks:raise ValueError('empty admitted training data')
 observations={'train_prose_documents':len(docs),'train_prose_bytes':sum(map(len,docs)),'train_task_sessions':len(tasks),'planned_targets':c['planned_targets'],'parameters':c['declared_parameters'],'total_planned_updates':c['total_updates'],'segment_updates':c['stop_update']-c['start_update'],'prose_batch':c['prose_batch'],'task_batch':c['task_batch'],'prose_context':c['prose_context'],'protected_test_cases':0}
 write(root/plan_name,observations)
 inputs=sorted(set(a['source_sha256'])|set(a['input_sha256'])|{'ADMISSION_020.json','REVIEW_020.json','RUNTIME_020.json','forge_020/manifest.json','ecosystem_vendor/VENDOR_MANIFEST.json','BUNDLE_MANIFEST.json',config_name,plan_name})
 if c.get('resume_checkpoint'):inputs+= [c['resume_checkpoint'],c['durable_resume_receipt']]
 pins={n:sha(relative(root,n)) for n in inputs}
 recipe={'schema':'trainlab.experiment.v2','name':stem,'description':'Mixed authentic-language and constructed-state research; native receipt does not establish capability','command':[sys.executable,'ecosystem020.py','worker',stem+'.json',config_name],'seed':c['seed'],'timeout_seconds':c['timeout_seconds'],'source_kind':'CONSTRUCTED' if c.get('software_fixture') else 'AUTHENTIC_CANDIDATE','split_identity':'020explicitTRAINDEV:'+sha(root/'INPUT_MANIFEST_020.json'),'inputs':inputs+[contract_name],'artifacts':['checkpoint.pt','updates.jsonl','result.json','schedule.json','segment_summary.json','execution_observations.json','runtime_before.json','runtime_after.json'],'metrics':{'completed_updates':'max','new_updates':'max','new_targets':'max'},'budget':{'updates':observations['segment_updates'],'total_fit_updates':c['total_updates'],'prose_batch':c['prose_batch'],'task_batch':c['task_batch']},'execution_purpose':'engineering_diagnostic' if c.get('software_fixture') else 'research','workload':{'kind':'train','training_method':'supervised','execution':'single_process'},'checks':{'completed_updates':{'min':c['stop_update'],'max':c['stop_update']},'new_updates':{'min':observations['segment_updates'],'max':observations['segment_updates']}}}
 dims={'model_size':('parameters',c['declared_parameters'],'Measured exact parameter count; capacity adequacy requires independent protocol review.'),'training_duration':('total_planned_updates',c['total_updates'],'Prospective full budget; not convergence.'),'datasets':('train_task_sessions',len(tasks),'Pinned declared prose and constructed task counts plus per-source manifests; semantic adequacy independently reviewed, no contamination certification.'),'training_regime':('task_batch',c['task_batch'],'Mixed next-byte/task training with fixed sampling; no oracle action execution.')}
 dimensions={k:{'configuration':f'{field}={v}','justification':scope,'limitations':scope,'criteria':[{'id':field,'rationale':scope,'evidence':plan_name,'field':field,'operator':'eq','threshold':v}]} for k,(field,v,scope) in dims.items()}
 ex={'model_size':[('parameters',c['declared_parameters'])],'training_duration':[('completed_updates',c['stop_update']),('new_updates',observations['segment_updates'])],'datasets':[('protected_test_cases',0),('train_prose_documents',len(docs)),('train_task_sessions',len(tasks))],'training_regime':[('task_batch',c['task_batch']),('prose_batch',c['prose_batch']),('schedule_rows',observations['segment_updates'])]}
 dimensions['training_duration']['criteria'].append({'id':'planned_targets','rationale':'Prospective exact seeded exposure, not convergence','evidence':plan_name,'field':'planned_targets','operator':'eq','threshold':c['planned_targets']})
 dimensions['datasets']['criteria'].append({'id':'train_prose_documents','rationale':'Eligible distinct prose articles after context-length filter','evidence':plan_name,'field':'train_prose_documents','operator':'eq','threshold':len(docs)})
 contract={'schema':'trainlab.research-adequacy.v1','claim':'Infrastructure and tiny trainer software fixture; no model capability claim' if c.get('software_fixture') else 'Bounded end-to-end language and state task feasibility comparison','claim_kind':'fixed_budget_comparison','scope':'Pinned TRAIN/validation plus constructed memory sessions; all failed attempts retained','excluded_claims':'No foundation-model readiness, broad transfer, equalFLOPs or convergence implied by receipt','recipe_sha256':recipe_identity(recipe),'inputs':pins,'review':{'author':review['author'],'reviewer':review['reviewer'],'basis':review['basis'],'criteria_frozen_before_evaluation':True},'dimensions':dimensions,'execution':{'artifact':'execution_observations.json','dimensions':{k:[{'field':f,'operator':'eq','threshold':v,'rationale':'Reconcile actual execution against declared segment'} for f,v in rows] for k,rows in ex.items()}}}
 write(root/contract_name,contract);recipe['research_adequacy']={'file':contract_name,'sha256':sha(root/contract_name)};path=root/(stem+'.json');write(path,recipe)
 config=load_config(path);allpins={n:sha(root/n) for n in config['inputs']};report=assess(config,root,allpins);verify_report(report,config,allpins);validate_execution(contract,config);write(root/(stem+'_preflight.json'),report);return path

def worker(recipe_name,config_name):
 import torch
 from models import make
 from trainlab.runner import load_config
 from trainlab.research_adequacy import recipe_identity
 root=Path.cwd();c=read(root/config_name);verify_admission(root,c);recipe=load_config(root/recipe_name);pins={n:sha(root/n) for n in recipe['inputs']};out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
 before=check_runtime(root);write(out/'runtime_before.json',before)
 actual_parameters=sum(p.numel() for p in make(c['arm'],width=c['width'],slots=c['slots']).parameters())
 if actual_parameters!=c['declared_parameters']:raise ValueError('actual model size differs before training')
 planned=next(x for x in read(root/c['exposure_plan']) if x['seed']==c['seed'])
 if c.get('resume_checkpoint'):
  predecessor=torch.load(root/c['resume_checkpoint'],map_location='cpu',weights_only=False)
  if predecessor['updates']!=c['start_update']:raise ValueError('checkpoint start-update mismatch before training')
  if predecessor['targets']!=planned['per_update'][c['start_update']-1]['cumulative_targets']:raise ValueError('checkpoint cumulative exposure mismatch before training')
 cmd=[sys.executable,'train.py','--arm',c['arm'],'--output',str(out),'--prose',c['prose'],'--tasks',c['tasks'],'--seed',str(c['seed']),'--width',str(c['width']),'--slots',str(c['slots']),'--runtime-manifest','RUNTIME_020.json','--updates',str(c['total_updates']),'--segment',str(c['segment']),'--prose-context',str(c['prose_context']),'--prose-batch',str(c['prose_batch']),'--task-batch',str(c['task_batch']),'--answer-weight',str(c['answer_weight']),'--lr',str(c['lr'])]
 if c.get('resume_checkpoint'):cmd+=['--resume',c['resume_checkpoint']]
 try:subprocess.run(cmd,check=True)
 finally:write(out/'runtime_after.json',runtime())
 if read(out/'runtime_after.json')!=before:raise ValueError('runtime changed during execution')
 cp=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);rows=[json.loads(x) for x in (out/'updates.jsonl').read_text().splitlines()]
 if [x['update'] for x in rows]!=list(range(c['start_update']+1,c['stop_update']+1)) or cp['updates']!=c['stop_update']:raise ValueError('actual update sequence mismatch')
 start_targets=0
 if c.get('resume_checkpoint'):start_targets=torch.load(root/c['resume_checkpoint'],map_location='cpu',weights_only=False)['targets']
 actual=read(out/'result.json');schedule=read(out/'schedule.json')
 if actual['started_update']!=c['start_update'] or actual['finished_update']!=c['stop_update'] or actual['schedule_sha256']!=hashlib.sha256(json.dumps(schedule,sort_keys=True).encode()).hexdigest() or len(schedule)!=len(rows):raise ValueError('trainer schedule/result binding mismatch')
 for row,expected in zip(rows,planned['per_update'][c['start_update']:c['stop_update']]):
  if row['update']!=expected['update'] or row['actual_targets']!=expected['targets'] or row['padded_token_positions']!=expected['padded_positions'] or row['targets']!=expected['cumulative_targets'] or row['sequences']!=expected['sequences']:raise ValueError('per-update exposure differs from prospective plan')
 if [x['samples'] for x in rows]!=schedule or sum(x['actual_targets'] for x in rows)!=actual['segment_targets'] or sum(x['sequences'] for x in rows)!=actual['sequences']:raise ValueError('perstep exposure observations mismatch')
 if actual['segment_targets']!=cp['targets']-start_targets or actual['cumulative_targets']!=cp['targets']:raise ValueError('trainer target accounting mismatch')
 docs=[json.loads(x)['text'].encode('utf-8') for x in (root/c['prose']).read_text().splitlines()];docs=[x for x in docs if len(x)>c['prose_context']+1]
 task_count=sum(bool(x.strip()) for x in (root/c['tasks']).read_text().splitlines())
 if c['stop_update']==c['total_updates'] and cp['targets']!=c['planned_targets']:raise ValueError('final exact exposure differs from prospective plan')
 summary={'train_prose_documents':len(docs),'train_task_sessions':task_count,'arm':c['arm'],'seed':c['seed'],'started_update':c['start_update'],'completed_updates':cp['updates'],'new_updates':len(rows),'new_targets':cp['targets']-start_targets,'cumulative_targets':cp['targets'],'checkpoint_sha256':sha(out/'checkpoint.pt'),'parameters':actual['parameters'],'task_batch':c['task_batch'],'prose_batch':c['prose_batch'],'schedule_rows':len(rows),'protected_test_cases':0,'scope':'Actual checkpoint/update/target accounting; sampling provenance separately audited, not independent compute attestation'}
 write(out/'segment_summary.json',summary);write(out/'metrics.json',{k:summary[k] for k in ('completed_updates','new_updates','new_targets')})
 write(out/'execution_observations.json',{'schema':'trainlab.execution-adequacy-observations.v1','recipe_sha256':recipe_identity(recipe),'inputs':pins,'producer':{'file':'ecosystem020.py','sha256':pins['ecosystem020.py']},'observations':summary})

if __name__=='__main__':
 if len(sys.argv)==4 and sys.argv[1]=='worker':worker(*sys.argv[2:])
 else:raise SystemExit('Preparation API only; native worker invocation required')
