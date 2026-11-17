import unittest,tempfile
from pathlib import Path
import torch
from train_heads import train
class TrainingTests(unittest.TestCase):
 def test_exact_resume_and_matched_schedule(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);g=torch.Generator().manual_seed(55);y=torch.full((12,64),257,dtype=torch.long);y[:,:3]=torch.randint(65,90,(12,3),generator=g);y[:,3]=256
   cache=root/'cache.pt';torch.save({'features':torch.randn(12,392,generator=g),'exact_features':torch.randn(12,392,generator=g),'targets':y,'ids':[str(i) for i in range(12)],'bindings':{'scope':'random software fixture'}},cache)
   schedules=[]
   for mode in ['ar','diffusion']:
    train(cache,root/(mode+'full'),mode,'learned',11,4,batch=4,width=16,layers=1)
    train(cache,root/(mode+'part'),mode,'learned',11,4,batch=4,width=16,layers=1,stop=2)
    train(cache,root/(mode+'resume'),mode,'learned',11,4,batch=4,width=16,layers=1,resume=root/(mode+'part')/'checkpoint.pt')
    a=torch.load(root/(mode+'full')/'checkpoint.pt',weights_only=True);b=torch.load(root/(mode+'resume')/'checkpoint.pt',weights_only=True)
    for k in a['model']:self.assertTrue(torch.equal(a['model'][k],b['model'][k]),k)
    self.assertTrue(torch.equal(a['sampler'],b['sampler']));self.assertTrue(torch.equal(a['noise_rng'],b['noise_rng']))
    import json
    schedules.append([json.loads(l)['sampled_ids_sha256'] for l in (root/(mode+'full')/'updates.jsonl').read_text().splitlines()])
   self.assertEqual(*schedules)
if __name__=='__main__':unittest.main()
