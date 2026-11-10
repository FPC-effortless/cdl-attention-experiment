"""No-fit replay and optional isolated CPU timing for completed registered fits."""
import argparse, hashlib, json, platform, time
import numpy as np
import torch
from experiment import HERE,ARMS,corpus,panels,model,evaluate

def run(timing=False):
 torch.set_num_threads(1);_,d=corpus();data,_=panels(d);rows=[]
 for seed in (11,23,37):
  for arm in ARMS:
   folder=HERE/f'seed-{seed}';r=json.loads((folder/f'{arm}-result.json').read_text());p=folder/f'{arm}-latest.pt'
   assert hashlib.sha256(p.read_bytes()).hexdigest()==r['checkpoint_sha256']
   ck=torch.load(p,weights_only=True);assert ck['update']==256
   m=model(arm);m.load_state_dict(ck['model']);m.eval()
   score=evaluate(m,data,details=False);expected=r['evaluation']['none']
   delta=abs(score['nll']-expected['nll']);per=max(abs(a-b) for a,b in zip(score['per_excerpt_nll'],expected['per_excerpt_nll']))
   row=dict(seed=seed,arm=arm,expected_nll=expected['nll'],replay_nll=score['nll'],absolute_error=delta,max_excerpt_error=per)
   print(json.dumps(row),flush=True)
   assert delta<=1e-6 and per<=1e-5,row
   if timing:
    with torch.inference_mode():
     s=m.init_state(1,'cpu')
     for t in range(100):_,s,_=m.step(data[:1,t],s)
     times=[]
     for repeat in range(5):
      for t in range(256):
       tick=time.perf_counter_ns();_,s,_=m.step(data[:1,t],s);times.append((time.perf_counter_ns()-tick)/1000)
     c=s.core
     row.update(step_median_us=float(np.median(times)),step_p95_us=float(np.quantile(times,.95)),retained_tensor_bytes=sum(v.numel()*v.element_size() for v in (c.values,c.links,c.cursor,c.history,s.regulation)),inference_calls=len(times))
   rows.append(row)
 result=dict(status='PASS',replays=len(rows),rows=rows,torch=torch.__version__,platform=platform.platform(),timing_scope='single-thread eager batch1 supplied-token steps; five256-token blocks after100 warmups; not optimized deployment or generated-text benchmark' if timing else None)
 name='replay_benchmark.json' if timing else 'cold_replay.json';(HERE/name).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--timing',action='store_true');run(ap.parse_args().timing)
