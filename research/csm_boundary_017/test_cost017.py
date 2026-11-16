import unittest,torch,tempfile,json
from pathlib import Path
from model import Model
from cost017 import warm,cold,selected
class CostTests(unittest.TestCase):
 def test_warm_and_cold(self):
  torch.set_num_threads(1);m=Model('guard_clause_mlp');cases=[{'case_id':'fixture','text':'Set A to a.\nRead A.\n'}]
  w=warm(m,cases,repeats=2,warmups=0);self.assertEqual(len(w['rows']),2);self.assertTrue(m.training);self.assertTrue(all(r['seconds']>0 for r in w['rows']))
  with tempfile.TemporaryDirectory() as d:
   cp=Path(d)/'model.pt';dp=Path(d)/'data.json';torch.save(m.state_dict(),cp);dp.write_text(json.dumps({'train':{'fixture':cases},'dev':{}}))
   c=cold(cp,m.arm,dp,'train',['fixture'],repeats=2);self.assertEqual(len(c['rows']),2);self.assertTrue(c['timing_complete'])
   failed=cold(cp,m.arm,dp,'train',['fixture'],repeats=2,timeout=.001);self.assertFalse(failed['timing_complete']);self.assertIsNone(failed['median_seconds']);self.assertGreater(failed['total_attempt_seconds'],0)
  with self.assertRaises(ValueError):selected({},'test',['fixture'])
if __name__=='__main__':unittest.main()
