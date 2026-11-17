"""DRAFT separate scientific seed-stage adapter. Requires new protocol/review/backups; never calibration admission."""
from pathlib import Path
import hashlib,json,sys,os,subprocess,time,traceback,zipfile,math

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,a):Path(p).write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
def path(root,n):
 q=Path(n)
 if q.is_absolute() or not q.parts or '..' in q.parts:raise ValueError('unsafe input path')
 p=root/q
 if root.is_symlink() or any((root/Path(*q.parts[:i])).is_symlink() for i in range(1,len(q.parts)+1)):raise ValueError('symlink input')
 if not p.resolve().is_relative_to(root.resolve()):raise ValueError('input escaped')
 return p

def validate(root):
 root=Path(root);m=read(root/'INPUTS.json');c=read(root/'STAGE.json');r=read(root/'REVIEW.json');a=read(root/'ADMISSION.json')
 if c['stage']!='scientific_seed_stage':raise ValueError('only reviewed per-seed scientific stages admitted')
 if r.get('approved') is not True or r.get('scope')!=c['stage'] or not r.get('reviewer') or r['reviewer']==r.get('author'):raise ValueError('genuine independent scoped review required')
 if r.get('inputs_sha256')!=sha(root/'INPUTS.json') or a.get('inputs_sha256')!=sha(root/'INPUTS.json') or a.get('review_sha256')!=sha(root/'REVIEW.json'):raise ValueError('admission drift')
 if a.get('public_source_verified') is not True or a.get('private_roundtrip_verified') is not True:raise ValueError('durable source/input backup required')
 for n,h in m['files'].items():
  p=path(root,n)
  if not p.is_file() or sha(p)!=h:raise ValueError('full input closure mismatch '+n)
 if not isinstance(c.get('command'),list) or not c['command'] or not all(isinstance(x,str) for x in c['command']):raise ValueError('invalid command list')
 for n in ('native_scientific021.py','STAGE.json',c['command'][0],c['runtime_manifest']):
  if n not in m['files']:raise ValueError('unbound execution source '+n)
 if not isinstance(c['command'],list) or not c['command'] or not all(isinstance(x,str) for x in c['command']):raise ValueError('invalid command list')
 if not c['command'][0].endswith('.py') or c['timeout_seconds']<=0 or c['timeout_seconds']>15000:raise ValueError('invalid bounded command')
 if c['expected_metrics'].get('scientific_fits')!=4 or c['expected_metrics'].get('protected_test_examples')!=0:raise ValueError('invalid seed-stage scope')
 if c.get('seed') not in (11,23,37) or c.get('head_updates')!=4096:raise ValueError('unreviewed scientific seed/budget')
 for flag,value in (('--seed',str(c['seed'])),('--updates',str(c['head_updates']))):
  if c['command'].count(flag)!=1 or c['command'][c['command'].index(flag)+1]!=value:raise ValueError('command budget mismatch '+flag)
 if c['expected_metrics'].get('optimizer_updates')!=4*c['head_updates']:raise ValueError('update budget mismatch')
 if c.get('resume_receipt'):
  receipt=read(path(root,c['resume_receipt']))
  if receipt.get('roundtrip_verified') is not True or not receipt.get('library_file_id'):raise ValueError('durable predecessor missing')
  for n,h in receipt['files'].items():
   if sha(path(root,n))!=h:raise ValueError('predecessor checkpoint drift')
 if not c['artifacts'] or len(c['artifacts'])>32 or len(set(c['artifacts']))!=len(c['artifacts']):raise ValueError('invalid artifact roster')
 for n in c['artifacts']:path(root,n)
 if c['result'] not in c['artifacts']:raise ValueError('result must be retained artifact')
 if any(type(v) is not int or v<0 for v in c['expected_metrics'].values()):raise ValueError('nonnegative exact integer counts required')
 vendor=path(root,c['vendor'])
 if not vendor.is_dir():raise ValueError('vendor directory missing')
 actual={str(p.relative_to(root)) for p in vendor.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 declared={n for n in m['files'] if n.startswith(c['vendor']+'/')}
 if actual!=declared:raise ValueError('vendor closure mismatch')
 return m,c

def validate_result(result,c):
 metrics=result['metrics']
 if set(metrics)!=set(c['expected_metrics']):raise ValueError('metric roster mismatch')
 for k,v in c['expected_metrics'].items():
  if type(metrics[k]) not in (float,int) or not math.isfinite(metrics[k]) or metrics[k]!=v:raise ValueError('metric mismatch '+k)
 rt=result['runtime']
 if not rt.get('torch') or not rt.get('cuda') or 'T4' not in rt.get('gpu','') or rt.get('deterministic') is not True or rt.get('threads')!=1:raise ValueError('actual GPU runtime missing or unexpected')
 for k,v in {'cublas':':4096:8','matmul_tf32':False,'cudnn_tf32':False,'cudnn_deterministic':True,'matmul_precision':'highest'}.items():
  if rt.get(k)!=v:raise ValueError('runtime precision mismatch '+k)
 return metrics

def worker(root):
 root=Path(root).resolve();out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);start=time.perf_counter()
 try:
  _,c=validate(root);sys.path.insert(0,str(path(root,c['vendor'])))
  from train_heads import configure,runtime
  configure()
  before={'python':sys.version,'pip_freeze':subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout.splitlines(),'runtime':runtime()}
  expected=read(path(root,c['runtime_manifest']))
  if any(before[k]!=expected[k] for k in ('python','pip_freeze','runtime')):raise ValueError('actual pretraining runtime differs from admitted calibration')
  write(out/'runtime_before.json',before)
  argv=[sys.executable]+[str(out/'stage') if x=='{OUTPUT}' else x for x in c['command']]
  proc=subprocess.run(argv,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=c['timeout_seconds'])
  (out/'stage.stdout').write_bytes(proc.stdout);(out/'stage.stderr').write_bytes(proc.stderr)
  write(out/'process.json',{'command':argv,'elapsed_seconds':time.perf_counter()-start,'returncode':proc.returncode,'scope':c['stage']})
  if proc.returncode:raise RuntimeError('seed-stage subprocess failed')
  for n in c['artifacts']:
   if not path(out,n).is_file():raise ValueError('missing artifact '+n)
  result=read(out/c['result']);metrics=validate_result(result,c)
  validate(root) # Detect source/asset mutation during execution.
  deps=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
  after={'python':sys.version,'pip_freeze':deps.splitlines(),'pip_freeze_sha256':hashlib.sha256(deps.encode()).hexdigest(),'runtime':result['runtime']}
  if any(after[k]!=expected[k] for k in ('python','pip_freeze','runtime')):raise ValueError('postrun runtime drift')
  write(out/'runtime_dependencies.json',after)
  write(out/'metrics.json',metrics)
  from trainlab.research_adequacy import recipe_identity
  rec=read(root/'recipe.json');pins={n:sha(path(root,n)) for n in rec['inputs']}
  observations=result['adequacy_observations']
  if observations!=c['adequacy_observations']:raise ValueError('actual research exposure differs from frozen declaration')
  write(out/'execution_observations.json',{'schema':'trainlab.execution-adequacy-observations.v1','recipe_sha256':recipe_identity(rec),'inputs':pins,'producer':{'file':'native_scientific021.py','sha256':pins['native_scientific021.py']},'observations':observations})
 except BaseException as e:
  if isinstance(e,subprocess.TimeoutExpired):
   (out/'stage.stdout').write_bytes(e.stdout or b'');(out/'stage.stderr').write_bytes(e.stderr or b'')
  write(out/'failure.json',{'error':repr(e),'elapsed_seconds':time.perf_counter()-start,'traceback':traceback.format_exc()});raise

def recipe(root,c):
 return {'schema':'trainlab.experiment.v2','name':'output021-'+c['stage'],'description':'Declared seed-stage training and evaluation; execution does not imply qualification','command':[sys.executable,'native_scientific021.py','worker',str(root)],'seed':c['seed'],'timeout_seconds':c['timeout_seconds']+120,'source_kind':'CONSTRUCTED','split_identity':'021-declared-TRAIN-exposed-DEV:'+sha(root/'INPUTS.json'),'inputs':['INPUTS.json','REVIEW.json','ADMISSION.json','STAGE.json','native_scientific021.py','FORGE_BINDING.json'],'artifacts':c['artifacts']+['stage.stdout','stage.stderr','process.json','runtime_dependencies.json','runtime_before.json','execution_observations.json'],'metrics':{k:'max' for k in c['expected_metrics']},'budget':c['expected_metrics'],'execution_purpose':'research','workload':{'kind':'train','training_method':'supervised','execution':'single_process'},'checks':{k:{'min':v,'max':v} for k,v in c['expected_metrics'].items()}}

def prepare_recipe(root,c):
 from trainlab.research_adequacy import recipe_identity,assess,verify_report
 from trainlab.execution_adequacy import validate_execution
 from trainlab.runner import load_config
 rec=recipe(root,c);review=read(root/'REVIEW.json');observations=c['adequacy_observations']
 write(root/'DECLARED_OBSERVATIONS.json',observations);rec['inputs'].append('DECLARED_OBSERVATIONS.json')
 pins={n:sha(path(root,n)) for n in rec['inputs']};rec['inputs'].append('ADEQUACY.json')
 dimensions={};execution={}
 for dim,field in c['adequacy_dimensions'].items():
  if dim not in ('model_size','training_duration','datasets','training_regime') or field not in observations:raise ValueError('incomplete adequacy mapping')
  dimensions[dim]={'configuration':str(observations[field]),'justification':review['basis'],'limitations':'Fixed budget and declared task exposure do not establish convergence or broad generalization','criteria':[{'id':field,'rationale':'Exact prospectively declared count with independent adequacy review','evidence':'DECLARED_OBSERVATIONS.json','field':field,'operator':'eq','threshold':observations[field]}]}
  execution[dim]=[{'field':field,'operator':'eq','threshold':observations[field],'rationale':'Reconcile actual checkpoint/data accounting'}]
 if set(dimensions)!=set(('model_size','training_duration','datasets','training_regime')):raise ValueError('missing adequacy dimension')
 contract={'schema':'trainlab.research-adequacy.v1','claim':'Matched frozen-condition AR/diffusion head study on declared constructed development panels','claim_kind':'fixed_budget_comparison','scope':'One independently reviewed seed stage; no protected TEST','excluded_claims':'No general CSM improvement, convergence, matched compute or frontier qualification','recipe_sha256':recipe_identity(rec),'inputs':pins,'review':{'author':review['author'],'reviewer':review['reviewer'],'basis':review['basis'],'criteria_frozen_before_evaluation':True},'dimensions':dimensions,'execution':{'artifact':'execution_observations.json','dimensions':execution}}
 write(root/'ADEQUACY.json',contract);rec['research_adequacy']={'file':'ADEQUACY.json','sha256':sha(root/'ADEQUACY.json')};write(root/'recipe.json',rec)
 config=load_config(root/'recipe.json');allpins={n:sha(root/n) for n in config['inputs']};report=assess(config,root,allpins);verify_report(report,config,allpins);validate_execution(contract,config);write(root/'ADEQUACY_PREFLIGHT.json',report)

def run(root,output):
 root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 try:
  m,c=validate(root);sys.path.insert(0,str(path(root,c['vendor'])))
  from forge_data.private_snapshot import freeze_private,verify_private
  from trainlab.runner import run_experiment,verify_run
  from research_lab.core import Controller
  from research_lab.adapters import import_trainlab
  from research_lab.evidence import put_node
  from investigation_world.evidence import EvidenceRecord,EvidenceOutcome,EvidenceVisibility,EvidenceProducerRef,EvidenceSubjectRef,EvidenceArtifactRef
  from datetime import datetime
  small=['INPUTS.json','REVIEW.json','ADMISSION.json','STAGE.json','native_scientific021.py',c['command'][0]]
  spec={'snapshot_id':'output021-scientific-seed','source_version':sha(root/'INPUTS.json'),'ai_use_allowed':True,'redistribution_allowed':False,'privacy_reviewed':True,'visibility':'operator_private','purpose':'Reviewed constructed head training and evaluation','excluded_claims':'No protected TEST, foundation-model promotion or superiority without independent scoring'}
  freeze_private(root,small,spec,output/'forge');fv=verify_private(output/'forge',sha(output/'forge/manifest.json'))
  write(root/'FORGE_BINDING.json',{'manifest_sha256':sha(output/'forge/manifest.json'),'full_input_closure_sha256':sha(root/'INPUTS.json'),'verification':fv})
  prepare_recipe(root,c);record=run_experiment(root/'recipe.json',output/'runs');rd=output/'runs'/record['id'];rv=verify_run(rd)
  lab=Controller(output/'lab');import_trainlab(lab,rd)
  if record['status']!='succeeded' or not rv['valid']:raise RuntimeError('native seed-stage failed; retained')
  ev=EvidenceRecord(evidence_type='output021.scientific_seed_stage',outcome=EvidenceOutcome.OBSERVED,visibility=EvidenceVisibility.OPERATOR_PRIVATE,claim='Bounded seed-stage execution observed; scientific interpretation requires independent replay and scoring',subjects=(EvidenceSubjectRef(kind='run',subject_id=record['id'],content_sha256=sha(rd/'run.json')),),producer=EvidenceProducerRef(producer_id='output021-native',producer_version='1',content_sha256=sha(root/'native_scientific021.py')),observed_at=datetime.fromisoformat(record['finished_at']),artifacts=(EvidenceArtifactRef(artifact_id='receipt',role='execution',content_sha256=sha(rd/'run.json'),media_type='application/json'),),provenance={'scope':c['stage']})
  write(output/'VERITAS.json',ev.model_dump(mode='json'));put_node(lab,'output021.scientific_seed','finding',ev.model_dump(mode='json'));write(output/'RESULT.json',{'native':rv,'forge':fv,'status':record['status'],'elapsed_seconds':time.perf_counter()-start,'scope':c['stage']})
 except BaseException as e:
  write(output/'FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-start});raise
 finally:
  files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()};write(output/'ARCHIVE_MANIFEST.json',{'files':files,'scope':'Bounded Forge code/hash manifest and unique outputs; restore external input assets using admitted full input manifest'})
  with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
   for n in sorted(list(files)+['ARCHIVE_MANIFEST.json']):z.write(output/n,n)
if __name__=='__main__':
 if sys.argv[1]=='worker':worker(sys.argv[2])
 elif sys.argv[1]=='run':run(sys.argv[2],sys.argv[3])
 else:raise SystemExit('worker ROOT | run ROOT OUTPUT')
