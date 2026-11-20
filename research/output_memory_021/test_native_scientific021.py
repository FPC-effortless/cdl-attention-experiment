import unittest,tempfile,pathlib,json,importlib.util
from unittest.mock import patch
s=importlib.util.spec_from_file_location('native',pathlib.Path(__file__).with_name('native_scientific021.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class NativeTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.r=pathlib.Path(self.t.name)
  self.c={'stage':'scientific_seed_stage','runtime_manifest':'RUNTIME.json','command':['calibration.py','--seed','11','--updates','4096','--out','{OUTPUT}'],'artifacts':['result.json'],'result':'result.json','expected_metrics':{'scientific_fits':4,'optimizer_updates':16384,'protected_test_examples':0,'resume_checks':2},'timeout_seconds':60,'seed':11,'head_updates':4096,'vendor':'vendor'}
  m.write(self.r/'STAGE.json',self.c)
  for n in ('native_scientific021.py','calibration.py','vendor/module.py','RUNTIME.json'):
   p=self.r/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('UNIT FIXTURE ONLY')
  m.write(self.r/'INPUTS.json',{'files':{str(p.relative_to(self.r)):m.sha(p) for p in self.r.rglob('*') if p.is_file()}})
  m.write(self.r/'REVIEW.json',{'approved':True,'scope':'scientific_seed_stage','author':'unitmock','reviewer':'unitfixture','inputs_sha256':m.sha(self.r/'INPUTS.json')})
  m.write(self.r/'ADMISSION.json',{'inputs_sha256':m.sha(self.r/'INPUTS.json'),'review_sha256':m.sha(self.r/'REVIEW.json'),'public_source_verified':True,'private_roundtrip_verified':True})
  self.result={'metrics':self.c['expected_metrics'].copy(),'runtime':{'torch':'fixture','cuda':'fixture','gpu':'Tesla T4','deterministic':True,'threads':1,'cublas':':4096:8','matmul_tf32':False,'cudnn_tf32':False,'cudnn_deterministic':True,'matmul_precision':'highest'}}
 def tearDown(self):self.t.cleanup()
 def test_valid(self):m.validate(self.r);m.validate_result(self.result,self.c)
 def test_source_drift(self):
  (self.r/'calibration.py').write_text('changed')
  with self.assertRaises(ValueError):m.validate(self.r)
 def test_vendor_extra(self):
  (self.r/'vendor/unlisted.py').write_text('changed')
  with self.assertRaises(ValueError):m.validate(self.r)
 def test_count_wrong(self):
  self.result['metrics']['resume_checks']=1
  with self.assertRaises(ValueError):m.validate_result(self.result,self.c)
 def test_nan(self):
  self.result['metrics']['resume_checks']=float('nan')
  with self.assertRaises(ValueError):m.validate_result(self.result,self.c)
 def test_cpu_rejected(self):
  self.result['runtime']['gpu']='CPU'
  with self.assertRaises(ValueError):m.validate_result(self.result,self.c)
 def test_parent_symlink(self):
  (self.r/'linked').symlink_to(self.r/'vendor',target_is_directory=True)
  with self.assertRaises(ValueError):m.path(self.r,'linked/module.py')
 def test_admission_failure_archived(self):
  (self.r/'ADMISSION.json').write_text('{}');out=self.r/'failed'
  with self.assertRaises(ValueError):m.run(self.r,out)
  self.assertTrue((out/'FAILURE.json').is_file());self.assertTrue(out.with_suffix('.zip').is_file())
if __name__=='__main__':unittest.main()
