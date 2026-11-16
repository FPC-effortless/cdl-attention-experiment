import unittest,torch
from model import Model,ARMS

def equal_state(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return all(equal_state(a[k],b[k]) for k in a)
 return all(equal_state(i,j) for i,j in zip(a,b))
class ModelTests(unittest.TestCase):
 def test_split_and_raw_gradient(self):
  torch.set_num_threads(1)
  x=torch.tensor([list(b'Set A to a.\nCopy B from A.\nRead B.\n')])
  for arm in ARMS:
   torch.manual_seed(17);m=Model(arm);a=m.segment(x,m.init(1),trace=True)
   for cut in (1,12,13,20,len(x[0])-1):
    b=m.segment(x[:,:cut],m.init(1));c=m.segment(x[:,cut:],b[2])
    self.assertTrue(torch.equal(a[0],torch.cat([b[0],c[0]],1)),(arm,cut));self.assertTrue(equal_state(a[2],c[2]))
   r=m.segment(x,m.init(1),return_raw=True)[0];self.assertTrue(bool(torch.isfinite(r[:,:,:6]).all()));torch.where(torch.isfinite(r),r,torch.zeros_like(r)).sum().backward();self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
 def test_guards(self):
  x=torch.tensor([list(b'Set A to a.\nRead A?')])
  for arm in ARMS[1:]:
   m=Model(arm);l,a,s,tr=m.segment(x,m.init(1),trace=True);op=l[:,:,:6].argmax(-1)
   self.assertTrue(bool((op[(x!=46)&(x!=63)]==0).all()))
if __name__=='__main__':unittest.main()
