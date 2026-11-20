"""Malformed-evidence regression tests; no model fitting or checkpoint change."""
import copy,json,unittest
from pathlib import Path
from evaluate import metrics_from_records,canonical_tuple

def oracle_fixture():
 d=json.loads((Path(__file__).resolve().parent/'data.json').read_text())
 c=d['train']['delete_seen_form0'][0]; ev=c['events']; non=[canonical_tuple(e['pos'],e['op'],e['dst'],e['src'],e['value']) for e in ev if e['op']]
 return dict(length=len(c['text']),gold_nonnoop=non,pred_nonnoop=copy.deepcopy(non),query_positions=c['query_positions'],query_targets=c['query_targets'],query_predictions=c['query_targets'][:],final_query_positions=c['final_query_positions'],final_query_targets=c['query_targets'][-4:],final_query_predictions=c['query_targets'][-4:],terminal_positions=[e['pos'] for e in ev],gold_terminal_states=[e['state_after'] for e in ev],pred_terminal_states=[e['state_after'] for e in ev],gold_final_state=c['final_state'],pred_final_state=c['final_state'])

class EvidenceTests(unittest.TestCase):
 def test_valid(self):
  m=metrics_from_records([oracle_fixture()]);self.assertEqual(m['complete_session_accuracy'],1);self.assertEqual(m['present_query_accuracy'],1);self.assertEqual(m['absent_query_accuracy'],1)
 def test_truncations(self):
  for key in ['query_predictions','query_targets','final_query_predictions','final_query_targets','pred_terminal_states','gold_terminal_states','pred_final_state','gold_final_state']:
   with self.subTest(key=key):
    r=oracle_fixture();r[key]=[]
    with self.assertRaises(ValueError):metrics_from_records([r])
 def test_bad_positions(self):
  for key in ['query_positions','final_query_positions','terminal_positions']:
   for mode in ['duplicate','bounds']:
    with self.subTest(key=key,mode=mode):
     r=oracle_fixture();r[key]=r[key][:];r[key][0]=r[key][1] if mode=='duplicate' else r['length']
     with self.assertRaises(ValueError):metrics_from_records([r])
 def test_actions(self):
  for key in ['pred_nonnoop','gold_nonnoop']:
   for mode in ['duplicate','bounds','badfield']:
    with self.subTest(key=key,mode=mode):
     r=oracle_fixture();r[key]=copy.deepcopy(r[key])
     if mode=='duplicate':r[key].append(r[key][-1])
     elif mode=='bounds':r[key][-1][0]=r['length']
     else:r[key][0][2]=4
     with self.assertRaises(ValueError):metrics_from_records([r])
 def test_false_query_fails(self):
  r=oracle_fixture();r['pred_nonnoop']=[[0,5,0,0,0]]+r['pred_nonnoop'];m=metrics_from_records([r]);self.assertEqual(m['complete_session_accuracy'],0);self.assertEqual(m['counts']['false_queries'],1)
if __name__=='__main__':unittest.main()
