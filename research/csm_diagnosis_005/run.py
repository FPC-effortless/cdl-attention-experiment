import sys,json,hashlib,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'csm_collapse_004'))
from variants import make,ARMS
from run import corpus
# The local module is __main__; imported run resolves the earlier package path.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
CONDITIONS=['intact','values','regulation','history','links','cursor','permute','all']
@torch.no_grad()
def evaluate(m,x,condition):
 s=m.init_state(1,'cpu');ys=[];tr=[]
 for token in x[:-1]:
  fresh=m.init_state(1,'cpu')
  if condition=='all':s=fresh
  elif condition=='regulation':s.regulation=fresh.regulation
  elif condition=='permute':s.core.values=s.core.values.roll(1,1)
  elif condition in ('values','history','links','cursor'):setattr(s.core,condition,getattr(fresh.core,condition))
  y,s,d=m.step(token.reshape(1),s);ys.append(y[0]);tr.append(d)
 y=torch.stack(ys);pred=y.argmax(-1);prob=y.softmax(-1)
 r=dict(nll=F.cross_entropy(y,x[1:]).item(),accuracy=(pred==x[1:]).float().mean().item(),space_share=(pred==32).float().mean().item(),confidence=prob.max(-1).values.mean().item())
 if condition=='intact':
  r['routes']={k:torch.cat([d[k].flatten() for d in tr]).bincount(minlength=8 if k!='operator' else 4).tolist() for k in ['first','second','destination','operator']}
  r['gates']={k:torch.cat([d[k].flatten() for d in tr]).float().mean().item() for k in ['gate','edits_value','uses_link','edits_link']}
 return r,y

def main():
 train,dev=corpus();rows=[]
 for seed in [11,23,37]:
  for arm in ARMS:
   ck=P.parent/'csm_collapse_004'/f'seed-{seed}'/f'{arm}.pt'
   receipt=json.loads(ck.with_suffix('.json').read_text());sha=hashlib.sha256(ck.read_bytes()).hexdigest();assert sha==receipt['checkpoint_sha256']
   m=make(arm);m.load_state_dict(torch.load(ck,weights_only=True)['model']);m.eval()
   for panel,x in [('fit',train[:129]),('dev',dev[:513])]:
    ref,logits=evaluate(m,x,'intact');again,y=evaluate(m,x,'intact');assert torch.equal(logits,y)
    for condition in CONDITIONS:
     r,y=(ref,logits) if condition=='intact' else evaluate(m,x,condition)
     r.update(seed=seed,arm=arm,panel=panel,condition=condition,targets=len(x)-1,checkpoint_sha256=sha,nll_delta=r['nll']-ref['nll'],prediction_change=(y.argmax(-1)!=logits.argmax(-1)).float().mean().item(),kl_from_intact=F.kl_div(y.log_softmax(-1),logits.softmax(-1),reduction='batchmean').item())
     rows.append(r)
   print(seed,arm,flush=True)
   (P/'results.json').write_text(json.dumps(dict(rows=rows,complete=len(rows)==192),indent=2)+'\n')
 lines=['# Frozen intervention diagnosis','','192 paired evaluations; 12 frozen checkpoints; 24 exact no-op replays. Positive NLL delta means ablation hurts. Negative means it helps. DEV is exposed development data, never protected TEST.','','| Arm | Panel | Intervention | Mean NLL change | Prediction changes |','|---|---|---|---:|---:|']
 for arm in ARMS:
  for panel in ['fit','dev']:
   for c in CONDITIONS[1:]:
    rr=[r for r in rows if (r['arm'],r['panel'],r['condition'])==(arm,panel,c)]
    lines.append(f"| {arm} | {panel} | {c} | {statistics.mean(r['nll_delta'] for r in rr):+.4f} | {statistics.mean(r['prediction_change'] for r in rr):.1%} |")
 lines+=['','Persistent ablations alter state distributions and cannot establish semantic correctness or identify the cause of the previous long-run collapse. Routing histograms, confidence, gates, per-seed results and KL are in results.json. No weights were updated.']
 (P/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
