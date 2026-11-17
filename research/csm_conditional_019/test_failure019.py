"""Failure-path fixtures, not review approvals or trained-model results."""
import unittest,tempfile,json,hashlib,subprocess
from pathlib import Path
from unittest.mock import patch
import cost019,runtime019,replay019,torch
class Failures(unittest.TestCase):
 def test_runtime_rejects(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'RUNTIME_LOCK.json').write_text('{}')
   with self.assertRaises(RuntimeError):runtime019.verify(p)
 def test_replay_rejects_bad_hash(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'checkpoint';p.write_bytes(b'fixture')
   with self.assertRaises(ValueError):replay019.replay(Path(replay019.__file__).parent,p,p,p,'bad','bad')
 def test_cold_retains_timeout(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'TIMING_PLAN.json').write_text('{}');cp=p/'cp';cp.write_bytes(b'test');out=p/'cost.json'
   with patch.object(cost019,'verify_inputs',return_value={}),patch('subprocess.run',side_effect=subprocess.TimeoutExpired(['fixture'],1800)):
    with self.assertRaises(subprocess.TimeoutExpired):cost019.cold(p,{a:cp for a in cost019.ARMS},out)
   r=json.loads(out.read_text());self.assertFalse(r['complete']);self.assertEqual(r['records'][0]['status'],'failed');self.assertGreater(r['records'][0]['elapsed_ns'],0)
 def test_warm_retains_parity_failure(self):
  class Dummy:
   def __init__(self,*a):pass
   def load_state_dict(self,*a):pass
   def eval(self):pass
   def init(self,*a):return None
   def segment(self,*a):return None
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);data={'dev':{'panel':[{'case_id':'one','text':'a'}]}};(p/'data.json').write_text(json.dumps(data));(p/'TIMING_PLAN.json').write_text(json.dumps({'data_sha256':hashlib.sha256((p/'data.json').read_bytes()).hexdigest(),'case_ids':['one']}));out=p/'cost.json'
   with patch.object(cost019,'verify_inputs',return_value={}),patch.object(cost019,'Model',Dummy),patch.object(torch,'load',return_value={}),patch.object(cost019,'check_output',side_effect=ValueError('fixture parity failure')):
    with self.assertRaises(ValueError):cost019.warm(p,{a:p/'unused' for a in cost019.ARMS},out)
   r=json.loads(out.read_text());self.assertFalse(r['complete']);self.assertEqual(r['records'][0]['status'],'failed')
if __name__=='__main__':unittest.main()
