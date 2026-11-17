"""One bounded segment per TrainLab invocation. Admission and durable resume required."""
import os,json,random,time,hashlib
from pathlib import Path
import torch
from model import Model,ARMS,tensor
from training import targets,loss
from evaluate import score_dataset
from recovery import save,restore
from prepare018 import source_hashes,STUDY_SPEC


def main():
 root=Path(__file__).resolve().parent
 from runtime018 import verify
 verify(root)
 admission=json.loads((root/'ADMISSION_018.json').read_text())
 if not admission.get('independent_review_approved') or not admission.get('durable_source_backup_verified'):
  raise RuntimeError('pre-execution review and durable source backup required')
 review_path=root/'review018.json'
 if admission.get('reviewer_record_sha256')!=hashlib.sha256(review_path.read_bytes()).hexdigest() or admission.get('source_sha256')!=source_hashes(root):raise RuntimeError('complete source/data roster and reviewer record required')
 review=json.loads(review_path.read_text())
 if review.get('approved') is not True or review.get('author')==review.get('reviewer') or not review.get('reviewer') or review.get('source_sha256')!=admission['source_sha256'] or review.get('protocol')!=STUDY_SPEC:raise RuntimeError('review must bind full source/data and exact study protocol')
 for name,h in admission['source_sha256'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise RuntimeError('frozen source changed')
 cfg=json.loads(Path(os.environ['CSM_FIT_CONFIG']).read_text())
 if set(cfg) not in ({'arm','seed','updates','stop_update'},{'arm','seed','updates','stop_update','resume_checkpoint','durable_resume_receipt'}):raise ValueError('unexpected config fields')
 arm=cfg['arm'];seed=cfg['seed'];stop=cfg['stop_update']
 if arm not in ARMS or seed not in [101,202,303] or cfg['updates']!=1024 or type(stop)!=int or not 64<=stop<=1024 or stop%64:raise ValueError('fit outside declared protocol')
 out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed);random.seed(seed)
 m=Model(arm);opt=torch.optim.AdamW(m.parameters(),lr=.003);rng=random.Random(seed)
 data=json.loads((root/'data.json').read_text());keys=sorted(data['train']);weights=torch.tensor(json.loads((root/'EVENT_COUNTS.json').read_text())['weights'])
 core={k:v for k,v in cfg.items() if k not in ('resume_checkpoint','stop_update','durable_resume_receipt')}
 identity=hashlib.sha256(json.dumps({'cfg':core,'inputs':admission['source_sha256']},sort_keys=True).encode()).hexdigest();start=0
 if cfg.get('resume_checkpoint'):
  receipt=json.loads(Path(cfg['durable_resume_receipt']).read_text())
  # Parent supplies this only after Library upload completion; declaration is not independent attestation.
  if not receipt.get('library_file_id') or receipt.get('checkpoint_index_sha256')!=hashlib.sha256(Path(cfg['resume_checkpoint']).read_bytes()).hexdigest():raise ValueError('durable resume receipt mismatch')
  start=restore(cfg['resume_checkpoint'],m,opt,rng,identity)
 if stop-start!=64:raise ValueError('exactly64 new updates per durable segment required')
 schedule=[];gate_target_schedule=[];curve=[];positions=0;begin=time.perf_counter()
 for u in range(start,stop):
  k=rng.choice(keys);ix=[rng.randrange(len(data['train'][k])) for _ in range(16)];schedule.append([u,k,ix]);cases=[data['train'][k][i] for i in ix];x=tensor(cases);positions+=x.numel();y=targets(cases)
  opt.zero_grad();r=m.segment(x,m.init(16),execute=True,return_raw=True,return_details=True,gold_events=[{e['pos']:e for e in c['events']} for c in cases]);raw=r[0];v=loss(raw,y,weights,r[-1]);gate_target_schedule.append({'update':u,'owned_state_targets':[[int(z) for z in row[row!=-100]] for row in r[-1]],'gold_history_targets':[[int(e['gate']) for e in c['events'] if e['op']==6] for c in cases]})
  if not bool(torch.isfinite(v)):raise RuntimeError('nonfinite loss')
  v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
  if any(not bool(torch.isfinite(p).all()) for p in m.parameters()):raise RuntimeError('nonfinite updated parameter')
  curve.append([u,float(v.detach())])
  with (out/'progress.jsonl').open('a') as log:
   log.write(json.dumps({'update_observed':u+1,'loss':float(v.detach()),'token_positions_in_segment':positions,'elapsed_seconds':time.perf_counter()-begin,'checkpoint_committed':False})+'\n');log.flush();os.fsync(log.fileno())
 elapsed=time.perf_counter()-begin
 save(out/'resume.pt',m,opt,rng,stop,identity)
 # Fixed portable generation names for native TrainLab artifact hashing; immutable originals retained.
 index=json.loads((out/'resume.pt').read_text());gen=out/'resume.pt.generations'/index['generation']
 import shutil
 shutil.copyfile(str(gen)+'.pt',out/'resume_payload.pt');shutil.copyfile(str(gen)+'.json',out/'resume_generation.json')
 result={'arm':arm,'seed':seed,'started_update':start,'finished_update':stop,'train_seconds':elapsed,'curve':curve,'token_positions':positions,'sampled_sequences':64*16,'identity':identity,'gate_target_schedule':gate_target_schedule}
 if stop==1024:
  m.eval();torch.save(m.state_dict(),out/'endpoint.pt');result.update(dev=score_dataset(m,data['dev']),train_probe=score_dataset(m,data['train']))
 (out/'result.json').write_text(json.dumps(result));(out/'schedule.json').write_text(json.dumps(schedule));(out/'metrics.json').write_text(json.dumps({'completed_updates':stop,'new_updates':stop-start,'token_positions':positions}))
 import zipfile
 with zipfile.ZipFile(out/'resume_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in [out/'resume.pt',Path(str(gen)+'.pt'),Path(str(gen)+'.json'),out/'result.json',out/'schedule.json',out/'metrics.json',out/'progress.jsonl']:
   z.write(p,p.relative_to(out))
 print('segment retained; durable backup required before continuation; independent endpoint replay required')
if __name__=='__main__':main()
