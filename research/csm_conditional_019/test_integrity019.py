"""Independent mutation tests for replay/runtime/cost admission boundaries."""
import unittest,json,tempfile,hashlib
from pathlib import Path
from unittest.mock import patch
import cost019,replay019,runtime019
from model import ARMS
class Tests(unittest.TestCase):
 def test_runtime_rejects_drift(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'RUNTIME_LOCK.json').write_text(json.dumps({'a':1}))
   with patch.object(runtime019,'fingerprint',return_value={'a':1}):self.assertEqual(runtime019.verify(root),{'a':1})
   with patch.object(runtime019,'fingerprint',return_value={'a':2}):
    with self.assertRaisesRegex(RuntimeError,'drift'):runtime019.verify(root)
 def test_replay_wrong_hash(self):
  root=Path(replay019.__file__).parent
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'payload';p.write_text('bad')
   with self.assertRaisesRegex(ValueError,'artifact drift'):replay019.replay(root,p,p,Path(td)/'out','0'*64,'0'*64)
 def test_timing_admission_hashes(self):
  import prepare019
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);endpoints={}
   for arm in ARMS:
    p=root/arm;p.write_text(arm);endpoints[arm]=p
   a=dict(approved=True,author='a',reviewer='b',seed=101,source_sha256={'src':'pin'},endpoints={arm:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for arm,p in endpoints.items()})
   def write(): (root/'TIMING_ADMISSION.json').write_text(json.dumps(a))
   with patch.object(runtime019,'verify',return_value={}),patch.object(prepare019,'source_hashes',return_value={'src':'pin'}):
    write();self.assertEqual(cost019.verify_inputs(root,endpoints),a)
    a['source_sha256']={'src':'changed'};write()
    with self.assertRaisesRegex(ValueError,'source drift'):cost019.verify_inputs(root,endpoints)
    a['source_sha256']={'src':'pin'};a['endpoints'][ARMS[0]]['sha256']='0'*64;write()
    with self.assertRaisesRegex(ValueError,'checkpoint mismatch'):cost019.verify_inputs(root,endpoints)
 def test_preservation_identical(self):
  from test_audit019 import result
  from data019 import build
  from audit019 import preservation,audit
  d=build();r=result(d);self.assertTrue(audit(d,r)['competence_gate']['passed']);self.assertTrue(preservation(d,r,r)['passed'])
if __name__=='__main__':unittest.main()
