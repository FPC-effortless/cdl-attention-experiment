import json,sys
from pathlib import Path
import numpy as np
import torch
from regulated import Regulated,base
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'csm_regulation_math_001'))
import reference as ref
mapping={'embedding':'embedding.weight','operation_embedding':'operator_embedding','first':'first_address','second':'second_address','gain':'write_gate','transform1':'transform.0','transform2':'transform.2'}
torch.set_num_threads(1)
rows=[]
for variant in ('instant','single','multi'):
 for mode in ('rewrite','ring_write','no_links'):
  for seed in range(3):
   c=ref.Config();p=ref.parameters(c,seed);m=Regulated(base.Config(vocab_size=19,mode=mode),variant).double()
   weights={}
   for k,v in p.items():
    if k=='trace_to_controller': target='injection.weight'
    elif k.startswith('trace_driver.'): target='driver.'+('weight' if k.endswith('.W') else 'bias')
    elif k in ('embedding','operation_embedding'): target='core.'+mapping[k]
    else:
     stem,suffix=k.rsplit('.',1);target='core.'+mapping.get(stem,stem)+('.weight' if suffix=='W' else '.bias')
    weights[target]=torch.tensor(v,dtype=torch.float64)
   weights['decay']=torch.tensor(ref.decay(c,variant));m.load_state_dict(weights)
   s=m.init_state(1,'cpu');n=ref.initialize(c);error=0
   with torch.no_grad():
    for t,tok in enumerate(np.random.default_rng(seed+1000).integers(0,19,96)):
     reset=t==47
     n,y,rt=ref.step(c,p,n,int(tok),ref.decay(c,variant),mode,reset=reset)
     ty,s,tt=m.step(torch.tensor([tok]),s,reset=reset)
     for k in ('first','second','destination','operator'): assert rt[k]==tt[k].item()
     assert rt['admitted']==bool(tt['edits_value'].item())
     assert rt['link_edit']==bool(tt['edits_link'].item())
     np.testing.assert_array_equal(n.links,s.core.links[0].argmax(-1).numpy())
     assert n.cursor==s.core.cursor.argmax(-1).item() and n.position==s.core.position
     for a,b in ((y,ty[0].numpy()),(n.values,s.core.values[0].numpy()),(n.regulation,s.regulation[0].numpy())):
      np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-11);error=max(error,float(np.abs(a-b).max()))
     _,padded,_=m.step(torch.tensor([tok]),s,valid=False);assert padded is s
   rows.append(dict(variant=variant,mode=mode,seed=seed,max_abs_error=error,steps=96))
# Training smoke diagnostic: gradient connectivity, not finite-difference validation
# of the biased hard-choice estimator.
torch.manual_seed(42)
m=Regulated(base.Config(vocab_size=19),'multi').double()
x=torch.tensor([[1,2,3,4,5,6,7,8]])
y=m(x);torch.nn.functional.cross_entropy(y.flatten(0,1),x.roll(-1,1).flatten()).backward()
gradients={k:float(v.grad.norm()) for k,v in m.named_parameters() if k.startswith(('driver.','injection.'))}
assert all(np.isfinite(v) and v>0 for v in gradients.values()),gradients
m.zero_grad();m.injection.weight.data.zero_();m(x).square().sum().backward()
assert torch.count_nonzero(m.driver.weight.grad)==0 and torch.count_nonzero(m.driver.bias.grad)==0
print(json.dumps(dict(status='PASS',rows=rows,gradient_norms=gradients,zero_injection_driver_gradient_zero=True,scope='FP64 regulated forward fixtures; connectivity only, no estimator correctness or quality claim'),indent=2))
