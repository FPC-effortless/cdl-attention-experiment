"""Independent bounded causality/streaming tests; not learned capability."""
import unittest,torch
from data018 import build
from model import Model,ARMS,tensor
class Tests(unittest.TestCase):
 def assert_nested(self,a,b):
  if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
  elif isinstance(a,dict):
   self.assertEqual(set(a),set(b))
   for k in a:self.assert_nested(a[k],b[k])
  elif isinstance(a,(list,tuple)):
   self.assertEqual(len(a),len(b))
   for x,y in zip(a,b):self.assert_nested(x,y)
  else:self.assertEqual(a,b)
 def test_prefix_and_segment_parity(self):
  torch.set_num_threads(1);rows=build()['dev']['chain_form4'][:2];x=tensor(rows)
  cuts=sorted({1,rows[0]['text'].index('A')+1,rows[0]['text'].index('\n')+1,next(e['pos'] for e in rows[0]['events'] if e['op']==6),x.shape[1]-2})
  for arm in ARMS:
   torch.manual_seed(188);m=Model(arm).eval()
   with torch.inference_mode():
    whole=m.segment(x,m.init(2),trace=True)
    for cut in cuts:
     prefix=m.segment(x[:,:cut],m.init(2),trace=True)
     suffix=m.segment(x[:,cut:],prefix[2],trace=True)
     self.assert_nested(prefix[0],whole[0][:,:cut]);self.assert_nested(prefix[1],whole[1][:,:cut])
     self.assert_nested(suffix[0],whole[0][:,cut:]);self.assert_nested(suffix[1],whole[1][:,cut:]);self.assert_nested(suffix[2],whole[2])
     changed=x.clone();changed[:,cut:]=32
     other=m.segment(changed,m.init(2),trace=True)
     self.assert_nested(prefix[0],other[0][:,:cut]);self.assert_nested(prefix[1],other[1][:,:cut])
     self.assert_nested(tuple(t[:,:cut] for t in whole[3]),tuple(t[:,:cut] for t in other[3]))
if __name__=='__main__':unittest.main()
