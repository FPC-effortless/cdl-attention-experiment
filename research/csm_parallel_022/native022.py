"""Reviewed native CSM-PARALLEL-022 execution. Current admission: software calibration only."""
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
 if c['stage']!='software_calibration':raise ValueError('scientific fits not admitted by this adapter version')
 if r.get('approved') is not True or r.get('scope')!=c['stage'] or not r.get('reviewer') or r['reviewer']==r.get('author'):raise ValueError('genuine independent scoped review required')
 if r.get('inputs_sha256')!=sha(root/'INPUTS.json') or a.get('inputs_sha256')!=sha(root/'INPUTS.json') or a.get('review_sha256')!=sha(root/'REVIEW.json'):raise ValueError('admission drift')
 if a.get('public_source_verified') is not True or a.get('private_roundtrip_verified') is not True:raise ValueError('durable source/input backup required')
 for n,h in m['files'].items():
  p=path(root,n)
  if not p.is_file() or sha(p)!=h:raise ValueError('full input closure mismatch '+n)
 if not isinstance(c.get('command'),list) or not c['command'] or not all(isinstance(x,str) for x in c['command']):raise ValueError('invalid command list')
 for n in ('native022.py','runtime022.py','STAGE.json',c['command'][0]):
  if n not in m['files']:raise ValueError('unbound execution source '+n)
 if not isinstance(c['command'],list) or not c['command'] or not all(isinstance(x,str) for x in c['command']):raise ValueError('invalid command list')
 if not c['command'][0].endswith('.py') or c['timeout_seconds']<=0 or c['timeout_seconds']>3600:raise ValueError('invalid bounded command')
 if c['expected_metrics'].get('scientific_fits')!=0 or c['expected_metrics'].get('protected_test_examples')!=0:raise ValueError('invalid calibration scope')
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
 if rt!=c['expected_runtime']:raise ValueError('actual CPU calibration runtime differs from declaration')
 return metrics

def worker(root):
 root=Path(root).resolve();out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);start=time.perf_counter()
 try:
  _,c=validate(root)
  from runtime022 import fingerprint
  before=fingerprint()
  if before!=c['expected_runtime']:raise ValueError('pre-execution CPU runtime drift')
  write(out/'runtime_before.json',before)
  argv=[sys.executable]+[str(out/'stage') if x=='{OUTPUT}' else x for x in c['command']]
  proc=subprocess.run(argv,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=c['timeout_seconds'])
  (out/'stage.stdout').write_bytes(proc.stdout);(out/'stage.stderr').write_bytes(proc.stderr)
  write(out/'process.json',{'command':argv,'elapsed_seconds':time.perf_counter()-start,'returncode':proc.returncode,'scope':c['stage']})
  if proc.returncode:raise RuntimeError('calibration subprocess failed')
  for n in c['artifacts']:
   if not path(out,n).is_file():raise ValueError('missing artifact '+n)
  result=read(out/c['result']);metrics=validate_result(result,c)
  validate(root) # Detect source/asset mutation during execution.
  if fingerprint()!=before:raise ValueError('post-execution CPU runtime drift')
  deps=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
  write(out/'runtime_dependencies.json',{'python':sys.version,'pip_freeze':deps.splitlines(),'pip_freeze_sha256':hashlib.sha256(deps.encode()).hexdigest(),'runtime':result['runtime']})
  write(out/'metrics.json',metrics)
 except BaseException as e:
  if isinstance(e,subprocess.TimeoutExpired):
   (out/'stage.stdout').write_bytes(e.stdout or b'');(out/'stage.stderr').write_bytes(e.stderr or b'')
  write(out/'failure.json',{'error':repr(e),'elapsed_seconds':time.perf_counter()-start,'traceback':traceback.format_exc()});raise

def recipe(root,c):
 return {'schema':'trainlab.experiment.v2','name':'csmparallel022-'+c['stage'],'description':'Software calibration, no scientific fit or model qualification','command':[sys.executable,'native022.py','worker',str(root)],'seed':c['seed'],'timeout_seconds':c['timeout_seconds']+120,'source_kind':'CONSTRUCTED','split_identity':'fixed constructed software fixture only','inputs':['INPUTS.json','REVIEW.json','ADMISSION.json','STAGE.json','native022.py','FORGE_BINDING.json'],'artifacts':c['artifacts']+['stage.stdout','stage.stderr','process.json','runtime_dependencies.json','runtime_before.json'],'metrics':{k:'max' for k in c['expected_metrics']},'budget':c['expected_metrics'],'execution_purpose':'engineering_diagnostic','workload':{'kind':'test','training_method':None,'execution':'single_process'},'checks':{k:{'min':v,'max':v} for k,v in c['expected_metrics'].items()}}

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
  small=['INPUTS.json','REVIEW.json','ADMISSION.json','STAGE.json','native022.py',c['command'][0]]
  spec={'snapshot_id':'csmparallel022-software','source_version':sha(root/'INPUTS.json'),'ai_use_allowed':True,'redistribution_allowed':False,'privacy_reviewed':True,'visibility':'operator_private','purpose':'Bounded constructed software calibration','excluded_claims':'No scientific head fits, held-out performance, protected TEST or model promotion'}
  freeze_private(root,small,spec,output/'forge');fv=verify_private(output/'forge',sha(output/'forge/manifest.json'))
  write(root/'FORGE_BINDING.json',{'manifest_sha256':sha(output/'forge/manifest.json'),'full_input_closure_sha256':sha(root/'INPUTS.json'),'verification':fv})
  write(root/'recipe.json',recipe(root,c));record=run_experiment(root/'recipe.json',output/'runs');rd=output/'runs'/record['id'];rv=verify_run(rd)
  lab=Controller(output/'lab');import_trainlab(lab,rd)
  if record['status']!='succeeded' or not rv['valid']:raise RuntimeError('native calibration failed; retained')
  ev=EvidenceRecord(evidence_type='csmparallel022.software_calibration',outcome=EvidenceOutcome.OBSERVED,visibility=EvidenceVisibility.OPERATOR_PRIVATE,claim='Bounded software calibration executed; no scientific capability claim',subjects=(EvidenceSubjectRef(kind='run',subject_id=record['id'],content_sha256=sha(rd/'run.json')),),producer=EvidenceProducerRef(producer_id='csmparallel022-native',producer_version='1',content_sha256=sha(root/'native022.py')),observed_at=datetime.fromisoformat(record['finished_at']),artifacts=(EvidenceArtifactRef(artifact_id='receipt',role='execution',content_sha256=sha(rd/'run.json'),media_type='application/json'),),provenance={'scope':c['stage']})
  write(output/'VERITAS.json',ev.model_dump(mode='json'));put_node(lab,'csmparallel022.software','finding',ev.model_dump(mode='json'));write(output/'RESULT.json',{'native':rv,'forge':fv,'status':record['status'],'elapsed_seconds':time.perf_counter()-start,'scope':c['stage']})
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
