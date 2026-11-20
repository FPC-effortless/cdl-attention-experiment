"""Prospective one-seed 021 scientific stage. Requires separate native admission."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, time
from pathlib import Path
import torch
from conditioner import load_conditioner
from cache_scientific import cache
from heads import OutputHead
from train_heads import train, configure, runtime
from evaluate_scientific import run as evaluate
from score_sessions import score

AXES=('id','names','wording','structure','long','joint')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return [json.loads(line) for line in Path(p).read_text().splitlines() if line]
def write(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def restore(path,cache_sha,seed,mode,condition,updates):
 c=torch.load(path,map_location='cpu',weights_only=True);b=c['bindings'];cfg=b['config']
 expected={'seed':seed,'mode':mode,'condition':condition,'updates':updates,'width':64,'layers':2,'batch':128,'length':64,'condition_dim':392,'lr':0.0003,'time_min':0.0001,'padding_loss':'all64 positions, including EOS/PAD','weight_decay':0.01,'betas':[0.9,0.999],'eps':1e-8,'gradient_clip_norm':1.0}
 if cfg!=expected or c['updates']!=updates or b['cache_sha256']!=cache_sha:raise ValueError('checkpoint identity mismatch')
 for name,digest in b['source'].items():
  if sha(Path(__file__).parent/name)!=digest:raise ValueError('checkpoint source mismatch')
 if b['runtime']!={**runtime(),'requested_device':'cuda'}:raise ValueError('checkpoint runtime mismatch')
 h=OutputHead(392,64,2,max_length=64).cuda();h.load_state_dict(c['model'],strict=True);h.eval();return h

def stable_rows(path):
 return [row for batch in read(path) for row in batch['rows']]
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--source',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,choices=[11,23,37],required=True);p.add_argument('--updates',type=int,default=4096);a=p.parse_args()
 configure()
 if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('admitted T4 required')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise ValueError('scientific output must be empty')
 started=time.perf_counter();status={'scope':'frozen native CSM versus supplied resolved state, matched AR/diffusion heads','seed':a.seed,'status':'running','training':{},'evaluations':{},'runtime':runtime()};write(out/'progress.json',status)
 try:
  data=Path(a.data);training=read(data/'train.jsonl');panels={axis:read(data/f'dev_{axis}.jsonl') for axis in AXES}
  if len(training)!=8192 or any(len(v)!=256 for v in panels.values()):raise ValueError('fixed data roster mismatch')
  model,binding=load_conditioner(a.checkpoint,a.source,'cuda')
  if binding['seed']!=a.seed:raise ValueError('conditioner seed mismatch')
  selected=training[:4096]
  selected_pairs={}
  for record in selected:selected_pairs[record['pair_id']]=selected_pairs.get(record['pair_id'],0)+1
  if len(selected_pairs)!=2048 or set(selected_pairs.values())!={2}:raise ValueError('TRAIN subset must contain first2048 complete pairs')
  subset_path=out/'selected_train.jsonl';subset_path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in selected))
  selection=json.loads((Path(__file__).parent/'TRAIN_SELECTION_021.json').read_text())
  if sha(data/'train.jsonl')!=selection['original_sha256'] or sha(subset_path)!=selection['selected_sha256'] or [r['id'] for r in selected]!=selection['selected_session_ids']:raise ValueError('frozen TRAIN selection mismatch')
  status['training_subset']={'original_sha256':sha(data/'train.jsonl'),'selected_sha256':sha(subset_path),'selection':'first2048 complete pairs before outcomes','sessions':4096,'query_features':24576}
  status['cache']=cache(model,selected,out/'train_cache.pt',conditioner_receipt=binding,data_sha256=sha(subset_path),batch_size=32)
  cache_sha=sha(out/'train_cache.pt');write(out/'progress.json',status)
  for condition in ('learned','exact'):
   for mode in ('ar','diffusion'):
    arm=f'{condition}_{mode}';status['training'][arm]=train(out/'train_cache.pt',out/arm,mode,condition,a.seed,a.updates,device='cuda');write(out/'progress.json',status)
  schedules={arm:[x['sampled_ids_sha256'] for x in read(out/arm/'updates.jsonl')] for arm in status['training']}
  if any(v!=next(iter(schedules.values())) for v in schedules.values()):raise ValueError('paired exposure schedule mismatch')
  for arm in status['training']:
   condition,mode=arm.split('_');cp=out/arm/'checkpoint.pt';head=restore(cp,cache_sha,a.seed,mode,condition,a.updates)
   status['evaluations'][arm]={}
   for axis,records in panels.items():
    primary=out/f'{arm}_{axis}_primary.jsonl';kw={'kind':mode,'steps':8,'condition':'native' if condition=='learned' else 'exact','seed':a.seed}
    evaluate(model,head,records,primary,**kw);metrics=score(records,read(primary))
    replay_head=restore(cp,cache_sha,a.seed,mode,condition,a.updates);replay=out/f'{arm}_{axis}_replay.jsonl';evaluate(model,replay_head,records,replay,**kw)
    if stable_rows(primary)!=stable_rows(replay):raise ValueError('exact checkpoint replay output mismatch')
    replay_metrics=score(records,read(replay));del replay_head
    entry={'checkpoint_sha256':sha(cp),'primary':metrics,'replay':replay_metrics,'raw_replay_equal':True,'primary_sha256':sha(primary),'replay_sha256':sha(replay),'verification_scope':'same-host checkpoint execution replay plus separate scoring; independent audit still required','diagnostics':{}}
    subset=records[:32]
    pair_counts={}
    for record in subset:pair_counts[record['pair_id']]=pair_counts.get(record['pair_id'],0)+1
    if len(pair_counts)!=16 or set(pair_counts.values())!={2}:raise ValueError('diagnostic pair roster mismatch')
    for intervention,k in [('zero',8),('swapped',8)]+([('full',4),('full',16)] if mode=='diffusion' else []):
     dest=out/f'{arm}_{axis}_{intervention}_k{k}.jsonl';evaluate(model,head,subset,dest,**{**kw,'steps':k,'intervention':intervention});entry['diagnostics'][f'{intervention}_k{k}']=score(subset,read(dest))
    status['evaluations'][arm][axis]=entry;write(out/'progress.json',status)
   del head
  observed_parameters=set();observed_updates=0;observed_exposures=0
  for arm in status['training']:
   checkpoint=torch.load(out/arm/'checkpoint.pt',map_location='cpu',weights_only=True)
   observed_parameters.add(sum(x.numel() for x in checkpoint['model'].values()))
   trace=read(out/arm/'updates.jsonl');observed_updates+=len(trace);observed_exposures+=sum(x['answer_exposures'] for x in trace)
  if len(observed_parameters)!=1:raise ValueError('head size mismatch')
  status['adequacy_observations']={'head_parameters':observed_parameters.pop(),'optimizer_updates':observed_updates,'train_cache_examples':status['cache']['examples'],'answer_exposures':observed_exposures}
  status['status']='completed';status['elapsed_seconds']=time.perf_counter()-started;status['metrics']={'scientific_fits':4,'optimizer_updates':4*a.updates,'answer_exposures':4*a.updates*128,'primary_sessions':4*6*256,'replay_sessions':4*6*256,'diagnostic_sessions':12*6*32,'protected_test_examples':0};write(out/'result.json',status)
 except Exception as e:
  status.update(status='failed',error_type=type(e).__name__,error=str(e),elapsed_seconds=time.perf_counter()-started);write(out/'progress.json',status);raise
if __name__=='__main__':main()
