import unittest,copy
from audit017 import rescore,paired_uncertainty
from test_evaluate import oracle_fixture
import json
from pathlib import Path
class AuditTests(unittest.TestCase):
 def fixture(self):
  c=json.loads((Path(__file__).parent/'data.json').read_text())['train']['delete_seen_form0'][0];r=oracle_fixture();r.update(case_id=c['case_id'],family=c['family'],arm='fixture',raw_nonnoop_positions=[e[0] for e in r['pred_nonnoop']],raw_interior_commits=[],suppressed_commits=[]);return c,r
 def test_gold_and_roster(self):
  c,r=self.fixture();self.assertTrue(rescore([r],[c])['qualified'])
  for kind in ('gold','truncated','raw','duplicate','rawmissing','family'):
   a=copy.deepcopy(r)
   if kind=='gold':a['gold_final_state'][0]=[1,4] if a['gold_final_state'][0]!=[1,4] else [0,0]
   if kind=='truncated':a['query_predictions']=[]
   if kind=='raw':a['raw_interior_commits']=[0]
   if kind=='duplicate':a['case_id']='unknown'
   if kind=='rawmissing':a['raw_nonnoop_positions'].pop()
   if kind=='family':a['family']='wrong'
   with self.assertRaises(ValueError):rescore([a],[c])
 def test_errors(self):
  c,r=self.fixture();r['query_predictions'][-1]=-1;r['final_query_predictions'][-1]=-1
  self.assertFalse(rescore([r],[c])['qualified'])
 def test_grouping(self):
  a={s:[dict(case_id=str(i),family='f'+str(i//2),complete_session=1) for i in range(4)] for s in (101,202,303)};b=copy.deepcopy(a)
  for rows in b.values():
   for r in rows:r['complete_session']=0
  v=paired_uncertainty(a,b,repeats=100);self.assertEqual(v['difference'],1);self.assertEqual(v['percentile_95'],[1,1]);self.assertEqual(len(v['families']),2)
  with self.assertRaises(ValueError):paired_uncertainty({101:a[101]},b)
if __name__=='__main__':unittest.main()
