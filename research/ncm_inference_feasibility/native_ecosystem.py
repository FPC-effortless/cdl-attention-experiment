"""Native NCM software inference receipt. No training or protected TEST access."""
from pathlib import Path
import hashlib,json,os,sys,time,subprocess,traceback,zipfile,math

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,a):Path(p).write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
def rel(root,n):
 p=Path(n)
 if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe relative path')
 q=root/p
 if any((root/Path(*p.parts[:i])).is_symlink() for i in range(1,len(p.parts)+1)) or not q.resolve().is_relative_to(root.resolve()):raise ValueError('unsafe input')
 return q

def validate(root):
 manifest=read(root/'INPUTS.json');review=read(root/'REVIEW.json');a=read(root/'ADMISSION.json')
 if review.get('approved') is not True or review.get('scope')!='software_inference_only' or not review.get('reviewer') or review.get('reviewer')==review.get('author'):raise ValueError('genuine scoped independent review required')
 if a.get('review_sha256')!=sha(root/'REVIEW.json') or a.get('inputs_sha256')!=sha(root/'INPUTS.json'):raise ValueError('admission drift')
 if not a.get('public_source_verified') or not a.get('private_roundtrip_verified'):raise ValueError('durable backup missing')
 if review.get('inputs_sha256')!=sha(root/'INPUTS.json'):raise ValueError('review closure drift')
 for n,h in manifest['files'].items():
  if not rel(root,n).is_file() or sha(rel(root,n))!=h:raise ValueError('input drift '+n)
 cfg=read(root/'CONFIG.json')
 if cfg.get('repeats')!=3 or cfg.get('device')!='cuda' or cfg.get('training') is not False or cfg.get('protected_test') is not False:raise ValueError('invalid software fixture scope')
 for n in ('CONFIG.json','native_ecosystem.py'):
  if n not in manifest['files']:raise ValueError('unbound execution source')
 loader=cfg['loader'];source=cfg['source'];ck=cfg['checkpoints'];tokenizer=cfg['text_tokenizer']
 if loader not in manifest['files']:raise ValueError('unbound loader')
 for prefix in (source,ck,tokenizer,cfg['vendor']):
  tree=rel(root,prefix)
  if not tree.is_dir():raise ValueError('missing asset tree '+prefix)
  actual={str(p.relative_to(root)) for p in tree.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
  declared={n for n in manifest['files'] if n.startswith(prefix+'/')}
  if actual!=declared:raise ValueError('incomplete asset closure '+prefix)
 return manifest,cfg

def worker(root):
 root=Path(root).resolve();_,c=validate(root);out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);start=time.perf_counter()
 command=[sys.executable,c['loader'],'--source',c['source'],'--checkpoints',c['checkpoints'],'--text-tokenizer',c['text_tokenizer'],'--device','cuda','--repeats','3','--out',str(out/'generation.json')]
 try:
  p=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=c['timeout_seconds'])
  (out/'loader.stdout').write_text(p.stdout);(out/'loader.stderr').write_text(p.stderr)
  write(out/'process.json',{'elapsed_seconds':time.perf_counter()-start,'returncode':p.returncode,'command':command,'training_updates':0,'protected_test_examples':0})
  if p.returncode:raise RuntimeError('loader failed; stderr retained')
  r=read(out/'generation.json')
  if len(r.get('repeat_seconds',[]))!=3 or r.get('weights_only') is not True or r.get('strict_state_dict') is not True:raise ValueError('incomplete software inference receipt')
  for v in [r['cold_generation_seconds']]+r['repeat_seconds']:
   if not isinstance(v,(float,int)) or not math.isfinite(v) or v<0:raise ValueError('invalid timing')
  if not r.get('runtime') or not r.get('device','').startswith('cuda'):raise ValueError('actual GPU fingerprint missing')
  counts={'benchmark_generation_calls':4,'additional_diagnostic_generation_calls':3,'vq_roundtrip_calls':3}
  if r.get('execution_counts')!=counts or len(r.get('diagnostic_panel',[]))!=3:raise ValueError('wrong execution roster')
  if 'T4' not in r['runtime'].get('gpu',{}).get('name',''):raise ValueError('unadmitted GPU')
  def finite_times(obj):
   if isinstance(obj,dict):
    for k,v in obj.items():
     if 'seconds' in k and isinstance(v,(int,float)) and (not math.isfinite(v) or v<0):raise ValueError('invalid raw timing '+k)
     finite_times(v)
   elif isinstance(obj,list):
    for v in obj:finite_times(v)
  finite_times(r)
  if not isinstance(r['cost_breakdown']['full_startup_seconds'],(int,float)) or not isinstance(r['gpu_memory']['peak_allocated_bytes'],int):raise ValueError('missing full cost')
  write(out/'metrics.json',dict(counts,training_updates=0))
 except BaseException as e:
  write(out/'failure.json',{'error':repr(e),'elapsed_seconds':time.perf_counter()-start,'scope':'software_inference_only'})
  if isinstance(e,subprocess.TimeoutExpired):
   (out/'loader.stdout').write_bytes(e.stdout or b'');(out/'loader.stderr').write_bytes(e.stderr or b'')
  raise

def run(root,output):
 root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 try:
  m,c=validate(root)
  sys.path.insert(0,str(rel(root,c['vendor'])))
  from forge_data.private_snapshot import freeze_private,verify_private
  from trainlab.runner import run_experiment,verify_run
  from research_lab.core import Controller
  from research_lab.adapters import import_trainlab
  from research_lab.evidence import put_node
  from investigation_world.evidence import EvidenceRecord,EvidenceOutcome,EvidenceVisibility,EvidenceProducerRef,EvidenceSubjectRef,EvidenceArtifactRef
  from datetime import datetime
  spec={'snapshot_id':'ncm-software-inputs','source_version':sha(root/'INPUTS.json'),'ai_use_allowed':True,'redistribution_allowed':False,'privacy_reviewed':True,'visibility':'operator_private','purpose':'Own fictional article, released model software feasibility only','excluded_claims':'No training, protected TEST, CSM capability or research promotion'}
  # Native Forge has a bounded snapshot contract. Freeze the exact full-closure manifest and small execution sources; rehash all referenced assets in the worker.
  small=['INPUTS.json','REVIEW.json','ADMISSION.json','CONFIG.json','PROTOCOL.md','native_ecosystem.py','safe_ncm_loader.py','assemble_private_inputs.py']
  freeze_private(root,small,spec,output/'forge')
  verified_forge=verify_private(output/'forge',sha(output/'forge/manifest.json'))
  write(root/'FORGE_BINDING.json',{'manifest_sha256':sha(output/'forge/manifest.json'),'input_closure_sha256':sha(root/'INPUTS.json'),'verification':verified_forge,'scope':'Every full input hash rechecked by native worker before model invocation'})
  recipe={'schema':'trainlab.experiment.v2','name':'ncm-software-inference','description':'Own fictional article inference fixture; no quality qualification','command':[sys.executable,'native_ecosystem.py','worker',str(root)],'seed':1729,'timeout_seconds':c['timeout_seconds']+120,'source_kind':'CONSTRUCTED','split_identity':'own-fictional-article-software-only','inputs':['INPUTS.json','REVIEW.json','ADMISSION.json','CONFIG.json','native_ecosystem.py','FORGE_BINDING.json'],'artifacts':['generation.json','process.json','loader.stdout','loader.stderr'],'metrics':{'benchmark_generation_calls':'max','additional_diagnostic_generation_calls':'max','vq_roundtrip_calls':'max','training_updates':'min'},'budget':{'benchmark_generation_calls':4,'additional_diagnostic_generation_calls':3,'vq_roundtrip_calls':3,'training_updates':0},'execution_purpose':'engineering_diagnostic','workload':{'kind':'test','training_method':None,'execution':'single_process'},'checks':{'benchmark_generation_calls':{'min':4,'max':4},'additional_diagnostic_generation_calls':{'min':3,'max':3},'vq_roundtrip_calls':{'min':3,'max':3},'training_updates':{'min':0,'max':0}}}
  write(root/'recipe.json',recipe);record=run_experiment(root/'recipe.json',output/'runs');rd=output/'runs'/record['id'];verification=verify_run(rd)
  lab=Controller(output/'lab');import_trainlab(lab,rd)
  if record['status']!='succeeded' or not verification['valid']:raise RuntimeError('native run failed; retained')
  ev=EvidenceRecord(evidence_type='ncm.software_inference',outcome=EvidenceOutcome.OBSERVED,visibility=EvidenceVisibility.OPERATOR_PRIVATE,claim='Pinned released NCM executes seven generation calls and three VQ reconstructions on own fictional text; not CSM or benchmark qualification',subjects=(EvidenceSubjectRef(kind='run',subject_id=record['id'],content_sha256=sha(rd/'run.json')),),producer=EvidenceProducerRef(producer_id='ncm-native-adapter',producer_version='1',content_sha256=sha(root/'native_ecosystem.py')),observed_at=datetime.fromisoformat(record['finished_at']),artifacts=(EvidenceArtifactRef(artifact_id='receipt',role='execution',content_sha256=sha(rd/'run.json'),media_type='application/json'),),provenance={'scope':'software_inference_only'})
  write(output/'VERITAS.json',ev.model_dump(mode='json'));put_node(lab,'ncm.software.inference','finding',ev.model_dump(mode='json'))
  write(output/'RESULT.json',{'forge':verified_forge,'native':verification,'status':record['status'],'total_elapsed_seconds':time.perf_counter()-start,'scope':'software_inference_only'})
 except BaseException as e:
  write(output/'FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'total_elapsed_seconds':time.perf_counter()-start});raise
 finally:
  omitted={}
  for name in ('generator.pt','tokenizer.pt'):
   p=root/'checkpoints'/name
   if p.is_file():
    omitted['checkpoints/'+name]={'bytes':p.stat().st_size,'sha256':sha(p),'restore_from':'hazyresearch/ncm-checkpoints@d342ce8aaea58830eef3b8ad473664673e41f692 or verified private mirror recorded in ADMISSION.json'}
  write(output/'EXTERNAL_ASSETS.json',{'referenced_not_snapshotted':omitted,'scope':'Forge freezes the full input hash manifest and bounded execution code, not weight bytes. Worker rehashes the entire closure. Inference replay requires exact external assets; portable evidence is not self-contained.'})
  files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()};write(output/'ARCHIVE_MANIFEST.json',{'files':files})
  with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
   for n in sorted(list(files)+['ARCHIVE_MANIFEST.json']):z.write(output/n,n)

if __name__=='__main__':
 if sys.argv[1]=='worker':worker(sys.argv[2])
 elif sys.argv[1]=='run':run(sys.argv[2],sys.argv[3])
 else:raise SystemExit('run root output | worker root')
