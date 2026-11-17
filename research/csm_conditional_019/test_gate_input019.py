"""Boundary instrumentation, not evidence of learned role accuracy."""
import unittest,types,torch
from model import Model
import audit019
class GateInput(unittest.TestCase):
 def test_gate_receives_predicted_read_and_expected(self):
  torch.set_num_threads(1)
  # A=a, B=b, C->A, D absent; resolved source must differ from raw row encoding.
  memory=(torch.tensor([[1,1,2,0]]),torch.tensor([[1,2,0,0]]))
  for arm in ('local_read','state_read'):
   m=Model(arm);seen=[]
   hook=m.gate_network.register_forward_pre_hook(lambda mod,args:seen.append(args[0].detach().clone()))
   for src in range(4):
    for expected in range(1,5):
     def decoded(self,h,candidates):
      r=torch.full((1,25),-10.);r[:,0]=10;r[:,11+src]=10;r[:,19+expected-1]=10
      return r
     m.decode=types.MethodType(decoded,m)
     st=m.init(1);st=(memory,*st[1:])
     gold=[{0:{'op':6,'src':(src+1)%4,'expected':expected%4+1}}]
     result=m.segment(torch.tensor([[46]]),st,gold_events=gold,return_details=True)
     effective=(1,2,1,0)[src]
     wanted=torch.zeros(1,9);wanted[0,effective]=1;wanted[0,5+expected-1]=1
     self.assertTrue(torch.equal(seen[-1],wanted))
     self.assertEqual(result[-1].item(),int(effective==expected))
   hook.remove()
 def test_isolated_table_diagnostic(self):
  from gate_diagnostic019 import probe
  m=Model('local_read')
  with torch.no_grad():
   for p in m.gate_network.parameters():p.zero_()
  r=probe(m);self.assertEqual((r['correct'],r['total']),(16,20))
  self.assertEqual(len({(x['source'],x['expected']) for x in r['rows']}),20)
  r=probe(Model('local_exact'));self.assertEqual(r['correct'],20)
 def test_gate_boundary_supports_former_counterexample(self):
  # Gate input for A=a versus B=b with expected a is no longer invariant.
  m=Model('local_read');a=torch.tensor([[0.,1,0,0,0,1,0,0,0]])
  b=torch.tensor([[0.,0,1,0,0,1,0,0,0]])
  with torch.no_grad():
   for p in m.gate_network.parameters():p.zero_()
   m.gate_network[0].weight[0,1]=1
   m.gate_network[2].weight[1,0]=1
  self.assertFalse(torch.equal(m.gate_network(a),m.gate_network(b)))
if __name__=='__main__':unittest.main()
