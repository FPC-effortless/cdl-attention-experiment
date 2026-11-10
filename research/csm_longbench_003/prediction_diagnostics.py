"""Post-fit exploratory prediction-collapse diagnostics on registered DEV windows."""
import json
import torch
from experiment import HERE,ARMS,corpus,panels,model

def run():
 torch.set_num_threads(1);_,dev=corpus();data,_=panels(dev);rows=[]
 target_counts=torch.bincount(data[:,1:].flatten(),minlength=256)
 for seed in (11,23,37):
  for arm in ARMS:
   ck=torch.load(HERE/f'seed-{seed}'/f'{arm}-latest.pt',weights_only=True)
   m=model(arm);m.load_state_dict(ck['model']);counts=torch.zeros(256,dtype=torch.long);entropy=0.
   margins={};handles=[]
   def hook(name):
    margins[name]=dict(count=0,gap_sum=0.,near_tie_count=0,min_gap=float('inf'))
    def collect(module,args,y):
     gap=y.topk(2,dim=-1).values;gap=gap[:,0]-gap[:,1];r=margins[name]
     r['count']+=len(gap);r['gap_sum']+=gap.sum().item();r['near_tie_count']+=int((gap<.01).sum());r['min_gap']=min(r['min_gap'],gap.min().item())
    return collect
   for name in ('first_address','second_address','destination','operator','value_edit','link_edit','use_link'):
    handles.append(getattr(m.core,name).register_forward_hook(hook(name)))
   with torch.inference_mode():
    s=m.init_state(16,'cpu')
    for t in range(2048):
     y,s,_=m.step(data[:,t],s);p=y.softmax(-1);counts+=torch.bincount(y.argmax(-1),minlength=256)
     entropy+=(-(p*p.clamp_min(1e-30).log()).sum(-1)).sum().item()
   for h in handles:h.remove()
   r=json.loads((HERE/f'seed-{seed}'/f'{arm}-result.json').read_text());d=r['evaluation']['none']['diagnostics']
   def effective(values):
    p=torch.tensor(values,dtype=torch.float64);p=p/p.sum();return float((-(p*p.clamp_min(1e-30).log()).sum()).exp())
   rows.append(dict(seed=seed,arm=arm,routing_margins=margins,distinct_predicted_bytes=int((counts>0).sum()),most_predicted_byte=int(counts.argmax()),most_predicted_share=float(counts.max()/counts.sum()),predicted_counts=counts.tolist(),target_counts=target_counts.tolist(),mean_predictive_entropy=entropy/32768,effective_destinations=effective(d['destination_counts']),effective_operators=effective(d['operator_counts'])))
 (HERE/'prediction_diagnostics.json').write_text(json.dumps(dict(scope='Post-fit exploratory; same exposed DEV; not pre-registered',rows=rows),indent=2)+'\n')
 print('Completed prediction diagnostics:',len(rows))
if __name__=='__main__':run()
