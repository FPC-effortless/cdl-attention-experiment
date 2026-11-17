import unittest,copy
import audit018
from data018 import build

def golden(c):
 ge=[audit018.gold_event(e) for e in c['events']];states=[e['state_after'] for e in c['events']];answers=[e['answer'] for e in c['events'] if e['op']==5]
 return dict(case_id=c['case_id'],pair_id=c['pair_id'],gold_events=ge,pred_events=copy.deepcopy(ge),gold_answers=answers,pred_answers=list(answers),gold_states=states,pred_states=copy.deepcopy(states),gold_final=c['final_state'],pred_final=copy.deepcopy(c['final_state']))
def result(data):
 out=dict(arm='local_exact',seed=101,finished_update=1024)
 for output,split in [('train_probe','train'),('dev','dev')]:
  out[output]={}
  for name,cases in data[split].items():
   rows=[golden(c) for c in cases]
   out[output][name]=dict(records=rows,metrics=audit018.recompute(rows),truth_metrics={str(t):audit018.recompute([r for r,c in zip(rows,cases) if int(c['truth'])==t]) for t in (0,1)})
 return out
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.data=build();cls.result=result(cls.data)
 def test_golden_full(self):self.assertTrue(audit018.audit(self.data,self.result)['valid'])
 def test_tampered_prediction(self):
  r=copy.deepcopy(self.result);row=next(iter(r['dev'].values()))['records'][0];row['pred_answers'][0]=(row['pred_answers'][0]+1)%5
  with self.assertRaisesRegex(ValueError,'execution consistency'):audit018.audit(self.data,r)
 def test_wrong_metric(self):
  r=copy.deepcopy(self.result);next(iter(r['dev'].values()))['metrics']['complete_map']=.5
  with self.assertRaisesRegex(ValueError,'metric mismatch'):audit018.audit(self.data,r)
 def test_missing_case(self):
  r=copy.deepcopy(self.result);next(iter(r['dev'].values()))['records'].pop()
  with self.assertRaisesRegex(ValueError,'roster'):audit018.audit(self.data,r)
 def test_missing_truth(self):
  r=copy.deepcopy(self.result);next(iter(r['dev'].values()))['truth_metrics'].pop('0')
  with self.assertRaisesRegex(ValueError,'truth roster'):audit018.audit(self.data,r)
 def test_duplicate_event(self):
  r=copy.deepcopy(self.result);row=next(iter(r['dev'].values()))['records'][0];row['pred_events'].append(row['pred_events'][-1])
  with self.assertRaisesRegex(ValueError,'order/duplicates'):audit018.audit(self.data,r)
 def test_wrong_gold(self):
  r=copy.deepcopy(self.result);next(iter(r['dev'].values()))['records'][0]['gold_answers'][0]=99
  with self.assertRaisesRegex(ValueError,'gold answers'):audit018.audit(self.data,r)
 def test_random_model_records(self):
  import torch
  from model import Model,ARMS
  from evaluate import score_dataset
  torch.set_num_threads(1)
  cases=self.data['dev']['chain_form0'][:2]
  for arm in ARMS:
   torch.manual_seed(18);m=Model(arm).eval();p=score_dataset(m,{'mini':cases})['mini']
   for r,c in zip(p['records'],cases):audit018.validate_record(r,c,arm)
   audit018.compare(audit018.recompute(p['records']),p['metrics'])
if __name__=='__main__':unittest.main()
