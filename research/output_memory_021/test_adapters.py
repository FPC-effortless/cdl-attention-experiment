import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from data_adapter import ExactState,turns,render_target,extract_payload
from evaluate_sessions import evaluate_pair
from cache_train import cache
from score_sessions import score

spec=importlib.util.spec_from_file_location('fixture_generator',Path(__file__).parents[2]/'goal-campaign/campaign/build_sessions.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
class FakeSession:
    instances=[]
    def __init__(self,model):self.text=b'';self.instances.append(self)
    def consume(self,data):self.text+=data
    def feature(self):
        x=torch.zeros(1,392);x[0,0]=sum(self.text)%10007;return x
class FakeBatch:
    def __init__(self,model,size):self.sessions=[FakeSession(model) for _ in range(size)]
    def consume(self,chunks):
        for s,c in zip(self.sessions,chunks):s.consume(c)
    def feature(self):return torch.cat([s.feature() for s in self.sessions])
class Head(nn.Module):
    def __init__(self):super().__init__();self.dummy=nn.Parameter(torch.zeros(1))
def wrong_generate(head,features,length,ids,seed):
    data=list(b'The current code is ZZZ.')+[256];return torch.tensor([data+[257]*(length-len(data))]*len(ids)),{'head_calls':length,'position_evaluations':length*len(ids)}
class AdapterTests(unittest.TestCase):
    def test_exact_all_semantics(self):
        for names in ['train','dev']:
          for wording in ['train','dev']:
            for structure in ['base','chain']:
              for r in g.episode_pair('fixture',1,256,wording,names,structure):
                state=ExactState()
                for _,chunk,a in turns(r):
                    state.consume(chunk);v=state.read(state.query)
                    self.assertEqual(v.decode() if v else 'unknown',a['target']);self.assertEqual(state.feature().shape,(1,392))
    def test_no_current_future_target_feature(self):
        rows=g.episode_pair('train',0,0)
        with tempfile.TemporaryDirectory() as d,patch('cache_train.BatchedFrozenSession',FakeBatch):
            cache(None,rows,Path(d)/'one.pt');a=torch.load(Path(d)/'one.pt',weights_only=True)
            rows[0]['answers'][-1]['target']='ZZZ'
            cache(None,rows,Path(d)/'two.pt');b=torch.load(Path(d)/'two.pt',weights_only=True)
            self.assertTrue(torch.equal(a['features'],b['features']));self.assertTrue(torch.equal(a['exact_features'],b['exact_features']))
            self.assertFalse(torch.equal(a['targets'],b['targets']))
    def test_free_running_and_rescore(self):
        rows=g.episode_pair('dev_id',0,0);FakeSession.instances=[]
        with patch('evaluate_sessions.BatchedFrozenSession',FakeBatch),patch('evaluate_sessions.generate_ar',wrong_generate):out=evaluate_pair(None,Head(),rows)
        for r,session in zip(rows,FakeSession.instances):
            expected=b''.join(chunk+b'The current code is ZZZ.\n' for _,chunk,_ in turns(r))
            self.assertEqual(session.text,expected)
        self.assertFalse(score(rows,[out])['gate_pass'])
        with self.assertRaises(ValueError):score(rows,[out,out])
        out['rows'][0]['answers'][0]['payload_correct']=True
        with self.assertRaises(ValueError):score(rows,[out])
    def test_train_only(self):
        with self.assertRaises(ValueError):cache(None,g.episode_pair('dev_id',0,0),'never.pt')
    def test_literal_field_is_not_semantics(self):
        self.assertEqual(extract_payload('The code is not ABC.'),'ABC')
        self.assertIsNone(extract_payload('ABC ABC'))
        self.assertIsNone(extract_payload('ABC DEF'))
    def test_render(self):
        for value in ['ABC','unknown']:
            self.assertEqual(extract_payload(render_target({'target':value})),value)
class RealBatchTests(unittest.TestCase):
    def test_ragged_feature_parity(self):
        from conditioner import load_conditioner,FrozenSession
        from batch_conditioner import BatchedFrozenSession
        base=Path(__file__).parents[2]/'goal-campaign/restored/LANGUAGE_BASELINE_001/replay_sources/csm-11'
        model,_=load_conditioner(base/'checkpoint.pt',base/'native_csm.py')
        batch=BatchedFrozenSession(model,3);serial=[FrozenSession(model) for _ in range(3)]
        for chunks in [[b'abc',b'',b'longer prefix'],[b'x'*71,b'next',b''],[b'',b'answer',b'z']]:
            batch.consume(chunks)
            for s,c in zip(serial,chunks):s.consume(c)
            self.assertTrue(torch.allclose(batch.feature(),torch.cat([s.feature() for s in serial]),atol=1e-6,rtol=1e-5))
    def test_cached_roster_batch_parity(self):
        from conditioner import load_conditioner
        base=Path(__file__).parents[2]/'goal-campaign/restored/LANGUAGE_BASELINE_001/replay_sources/csm-11'
        model,_=load_conditioner(base/'checkpoint.pt',base/'native_csm.py')
        rows=g.episode_pair('train',9,0)
        with tempfile.TemporaryDirectory() as d:
            cache(model,rows,Path(d)/'batch.pt',batch_size=2)
            cache(model,rows,Path(d)/'serial.pt',batch_size=1,serial_reference=True)
            a=torch.load(Path(d)/'batch.pt',weights_only=True);b=torch.load(Path(d)/'serial.pt',weights_only=True)
            self.assertEqual(a['ids'],b['ids']);self.assertTrue(torch.equal(a['targets'],b['targets']))
            self.assertTrue(torch.equal(a['exact_features'],b['exact_features']))
            self.assertTrue(torch.allclose(a['features'],b['features'],atol=1e-6,rtol=1e-5))
    def test_real_free_running_batch_parity(self):
        from conditioner import load_conditioner
        from heads import OutputHead
        from evaluate_sessions import evaluate_batch
        base=Path(__file__).parents[2]/'goal-campaign/restored/LANGUAGE_BASELINE_001/replay_sources/csm-11'
        model,_=load_conditioner(base/'checkpoint.pt',base/'native_csm.py')
        torch.manual_seed(31);head=OutputHead(392,width=16,layers=1,max_length=64).eval()
        records=g.episode_pair('software',91,0)
        for kind in ['ar','diffusion']:
            batch=evaluate_batch(model,head,records,kind=kind,steps=4,length=8)
            single=[evaluate_batch(model,head,[r],kind=kind,steps=4,length=8,software_single=True)['rows'][0] for r in records]
            self.assertEqual(batch['rows'],single)
if __name__=='__main__':unittest.main()

