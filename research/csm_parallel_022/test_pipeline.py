import unittest,tempfile,json,copy
from pathlib import Path
import torch
from data import episode_pair,OPS,external_events
from evaluate import commands,score
from executor import Executor
from train import train

class PipelineTests(unittest.TestCase):
 def test_gold_controller_exact_pipeline(self):
  records=episode_pair('software',0);rows=[]
  for r in records:
   actions=[{'op':OPS[e['labels']['op']],'valid':True,'spans':e['labels']['spans']} for e in r['events']]
   ops,n,banks,errors,registry=commands(external_events(r),actions);x=Executor().execute(ops,n,bank_ids=banks)
   from executor import reference
   from data import payload_text
   state=None;guards=[]
   for op in ops:
    guards.append(reference([[5,op[2],0,0,0]],n,state)['answers'][0]==op[4] if op[0]==6 else None)
    state=reference([op],n,state)['state']
   rendered=[None if a==-2 else 'No code is currently assigned.' if a is None else 'The current code is '+payload_text(a)+'.' for a in x['answers']]
   rows.append({'id':r['id'],'pair_id':r['pair_id'],'variant':r['variant'],'actions':actions,'answers':x['answers'],'errors':errors,'typed_commands':ops,'registry':registry,'predicted_guards':guards,'rendered_answers':rendered,'final_state':[list(v) for v in x['state']]})
  s=score(records,rows);self.assertEqual(s['complete_sessions'],2);self.assertEqual(s['paired_first_correct'],1)
 def test_invalid_identity_no_registration(self):
  rows,n,banks,errors,ids=commands(['Set west/A to ABC.'],[{'op':'SET','valid':True,'spans':{'dst':[0,3],'value':[14,17]}}]);self.assertEqual(ids,{});self.assertEqual(rows[0][0],0);self.assertIsNotNone(errors[0])
 def test_exact_restart(self):
  records=episode_pair('software',1);examples=[{'id':f'{r["id"]}-{i}','text':e['text'],'labels':e['labels']} for r in records for i,e in enumerate(r['events'])]
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);source=p/'examples.jsonl';source.write_text(''.join(json.dumps(x)+'\n' for x in examples))
   for arm in ['parallel','gru']:
    train(source,p/(arm+'full'),arm,11,4,batch=2)
    train(source,p/(arm+'half'),arm,11,4,batch=2,stop=2)
    train(source,p/(arm+'resume'),arm,11,4,batch=2,resume=p/(arm+'half')/'checkpoint.pt')
    a=torch.load(p/(arm+'full')/'checkpoint.pt',weights_only=True);b=torch.load(p/(arm+'resume')/'checkpoint.pt',weights_only=True)
    def eq(a,b):
     if isinstance(a,torch.Tensor):return torch.equal(a,b)
     if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
     if isinstance(a,(list,tuple)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
     return a==b
    self.assertTrue(eq(a,b))
if __name__=='__main__':unittest.main()
