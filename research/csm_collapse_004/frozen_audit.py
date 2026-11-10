"""No-fit activation/gradient audit of the six specified long-run checkpoints."""
import json,hashlib,torch
from run import HERE,corpus,score,audit
from variants import make

def main():
 torch.set_num_threads(1);tr,_=corpus();x=tr[:129].unsqueeze(0);rows=[]
 for seed in (11,23,37):
  for arm in ('instant','multi_tb64'):
   folder=HERE.parent/'csm_longbench_003'/f'seed-{seed}';p=folder/f'{arm}-latest.pt';r=json.loads((folder/f'{arm}-result.json').read_text())
   assert hashlib.sha256(p.read_bytes()).hexdigest()==r['checkpoint_sha256']
   m=make('hard');m.load_state_dict(torch.load(p,weights_only=True)['model']);rows.append(dict(seed=seed,arm=arm,checkpoint_sha256=r['checkpoint_sha256'],score=score(m,x),audit=audit(m,x)))
 result=dict(scope='Frozen model; authentic first128 training targets; zero optimizer updates',rows=rows)
 (HERE/'frozen_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 for r in rows:
  a=r['audit']['activations'];print(r['seed'],r['arm'],r['score']['nll'],{k:(round(v['abs_above_3'],3),round(v['mean_coordinate_std'],4)) for k,v in a.items()})
if __name__=='__main__':main()
