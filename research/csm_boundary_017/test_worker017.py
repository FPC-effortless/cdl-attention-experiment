"""Admission failure tests only: all reject before any optimizer step."""
import json,tempfile,unittest,hashlib,shutil
from pathlib import Path
from unittest.mock import patch
import worker
from prepare017 import source_hashes,STUDY_SPEC
class WorkerTests(unittest.TestCase):
 def test_reject_unbound_admission(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);origin=Path(worker.__file__).parent
   for p in origin.glob('*.py'):shutil.copyfile(p,root/p.name)
   for n in ('data.json','EVENT_COUNTS.json','PROTOCOL.md','TIMING_PLAN.json'):shutil.copyfile(origin/n,root/n)
   review={'approved':True,'author':'fixture-author','reviewer':'fixture-reviewer','protocol':STUDY_SPEC,'source_sha256':source_hashes(root)}
   rp=root/'review017.json';rp.write_text(json.dumps(review));ad={'independent_review_approved':True,'durable_source_backup_verified':True,'reviewer_record_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'source_sha256':source_hashes(root)}
   for key,value in [('independent_review_approved',False),('durable_source_backup_verified',False),('reviewer_record_sha256','0'*64),('source_sha256',{})]:
    bad=dict(ad);bad[key]=value;(root/'ADMISSION_017.json').write_text(json.dumps(bad))
    with patch.object(worker,'__file__',str(root/'worker.py')),patch.object(worker,'Model',side_effect=AssertionError('must not initialize')):
     with self.assertRaises(RuntimeError):worker.main()
   review['protocol']=dict(STUDY_SPEC,updates=2048);rp.write_text(json.dumps(review));ad['reviewer_record_sha256']=hashlib.sha256(rp.read_bytes()).hexdigest();(root/'ADMISSION_017.json').write_text(json.dumps(ad))
   with patch.object(worker,'__file__',str(root/'worker.py')),patch.object(worker,'Model',side_effect=AssertionError('must not initialize')):
    with self.assertRaises(RuntimeError):worker.main()
if __name__=='__main__':unittest.main()
