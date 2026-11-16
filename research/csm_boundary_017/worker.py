"""One fit per TrainLab invocation; not executable until reviewed admission exists."""
import os,json,random,time,hashlib
from pathlib import Path
import torch
from model import Model,ARMS,tensor
from training import targets,loss
from evaluate import score_dataset
from recovery import save,restore

def main():
 root=Path(__file__).resolve().parent; admission=json.loads((root/'ADMISSION_017.json').read_text())
 if not admission.get('independent_review_approved') or not admission.get('durable_source_backup_verified'):raise RuntimeError('pre-execution review and durable source backup required')
 for name,h in admission['source_sha256'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise RuntimeError('frozen source changed')
 cfg=json.loads(Path(os.environ['CSM_FIT_CONFIG']).read_text());arm=cfg['arm'];seed=cfg['seed']
 if arm not in ARMS or seed not in [101,202,303] or cfg['updates']!=1024:raise ValueError('fit outside declared protocol')
 out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
 m=Model(arm);opt=torch.optim.AdamW(m.parameters(),lr=.003);rng=random.Random(seed);data=json.loads((root/'data.json').read_text());keys=sorted(data['train']);weights=torch.tensor(json.loads((root/'EVENT_COUNTS.json').read_text())['weights']);identity=hashlib.sha256(json.dumps({'cfg':{k:v for k,v in cfg.items() if k!='resume_checkpoint'},'inputs':admission['source_sha256']},sort_keys=True).encode()).hexdigest();start=0
 # Resume source is explicit, separate original evidence remains untouched.
 if cfg.get('resume_checkpoint'):start=restore(cfg['resume_checkpoint'],m,opt,rng,identity)
 schedule=[];curve=[];begin=time.perf_counter()
 for u in range(start,1024):
  k=rng.choice(keys);ix=[rng.randrange(len(data['train'][k])) for _ in range(16)];schedule.append([u,k,ix]);cases=[data['train'][k][i] for i in ix];x=tensor(cases);y=targets(cases);opt.zero_grad();raw,_,_,_=m.segment(x,m.init(16),execute=m.conditioned,return_raw=True);v=loss(raw,y,weights)
  if not bool(torch.isfinite(v)):raise RuntimeError('nonfinite loss')
  v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if u%128==0 or u==1023:curve.append([u,float(v.detach())])
  if (u+1)%64==0:save(out/'resume.pt',m,opt,rng,u+1,identity)
 elapsed=time.perf_counter()-begin;m.eval();torch.save(m.state_dict(),out/'endpoint.pt');result={'arm':arm,'seed':seed,'started_update':start,'finished_update':1024,'train_seconds':elapsed,'curve':curve,'dev':score_dataset(m,data['dev']),'train_probe':score_dataset(m,data['train'])};(out/'result.json').write_text(json.dumps(result));(out/'schedule.json').write_text(json.dumps(schedule));(out/'metrics.json').write_text(json.dumps({'completed_updates':1024,'new_updates':1024-start}));print('fit endpoint retained; independent replay required')
if __name__=='__main__':main()
