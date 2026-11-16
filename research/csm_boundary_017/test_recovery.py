import unittest,tempfile,random,torch
from pathlib import Path
from recovery import save,restore
class TestRecovery(unittest.TestCase):
 def test_scalar_resume(self):
  torch.manual_seed(7);m=torch.nn.Linear(1,1);o=torch.optim.AdamW(m.parameters(),lr=.01);rng=random.Random(9)
  def step():
   x=torch.tensor([[rng.random()]])+torch.rand(1,1);o.zero_grad();loss=m(x).square().sum();loss.backward();o.step();return loss.detach().clone()
  step()
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'state.pt';save(p,m,o,rng,1,'fixture');v=step();expected=[x.detach().clone() for x in m.parameters()];restore(p,m,o,rng,'fixture');w=step();self.assertTrue(torch.equal(v,w));self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,m.parameters())))
   with self.assertRaises(ValueError):restore(p,m,o,rng,'wrong')
   p.write_bytes(p.read_bytes()+b'x')
   with self.assertRaises(ValueError):restore(p,m,o,rng,'fixture')
if __name__=='__main__':unittest.main()
