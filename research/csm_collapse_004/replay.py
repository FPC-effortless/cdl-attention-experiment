"""Hash-verified cold replay of all final diagnostic scores; no fitting."""
import hashlib,json,torch
from variants import make,ARMS
from run import HERE,corpus,score

def main():
 torch.set_num_threads(1);tr,d=corpus();panels={'fit':tr[:129].unsqueeze(0),'adjacent_train':tr[129:258].unsqueeze(0),'exposed_dev':d[:1025].unsqueeze(0)};rows=[]
 for seed in (11,23,37):
  for arm in ARMS:
   p=HERE/f'seed-{seed}'/f'{arm}.pt';r=json.loads(p.with_suffix('.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==r['checkpoint_sha256']
   ck=torch.load(p,weights_only=True);assert ck['update']==256
   m=make(arm);m.load_state_dict(ck['model'])
   for name,x in panels.items():
    got=score(m,x);error=abs(got['nll']-r[name]['nll']);assert error<=1e-6,(seed,arm,name,error)
    assert got['accuracy']==r[name]['accuracy'] and got['distinct_top1']==r[name]['distinct_top1']
    rows.append(dict(seed=seed,arm=arm,panel=name,absolute_nll_error=error))
 print(json.dumps(dict(status='PASS',checkpoint_replays=12,panel_scores=len(rows),max_error=max(r['absolute_nll_error'] for r in rows),rows=rows),indent=2))
if __name__=='__main__':main()
