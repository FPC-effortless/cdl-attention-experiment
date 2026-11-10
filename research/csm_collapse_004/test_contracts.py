import torch
from variants import make,ARMS,Regulated,base
from run import sequence

def main():
 torch.set_num_threads(1);x=torch.tensor([[1,2,3,4,5,6,7,8,9]])
 torch.manual_seed(11);original=Regulated(base.Config(vocab_size=256),'multi').float();ys=sequence(original,x);ys.square().sum().backward()
 initials=[]
 for arm in ARMS:
  torch.manual_seed(11);m=make(arm);initials.append({k:v.clone() for k,v in m.state_dict().items()})
  assert sum(p.numel() for p in m.parameters())==34479
  y=sequence(m,x);assert torch.equal(y[:,:4],sequence(m,x[:,:5]))
  y.square().sum().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
  if arm=='hard':
   assert torch.equal(y,ys)
   for p,q in zip(m.parameters(),original.parameters()):assert torch.equal(p.grad,q.grad)
  # Independently scheduled streams equal separate runs.
  with torch.no_grad():
   bat=sequence(m,x.repeat(2,1));assert torch.allclose(bat[0],sequence(m,x)[0],atol=1e-6)
 for item in initials[1:]:assert all(torch.equal(v,item[k]) for k,v in initials[0].items())
 print('PASS: exact hard forward/backward parity; matched arrays/counts; causal prefixes; finite gradients; batch independence')
if __name__=='__main__':main()
