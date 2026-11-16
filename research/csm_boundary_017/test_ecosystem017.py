"""Integration preparation tests only; never invoke worker or fit."""
import json,shutil,tempfile,unittest
from pathlib import Path
import ecosystem017 as e
class EcosystemTests(unittest.TestCase):
 def test_native_preflight(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=Path(__file__).parent
   for p in source.glob('*.py'):shutil.copyfile(p,root/p.name)
   for n in ['data.json','EVENT_COUNTS.json','PROTOCOL.md','TIMING_PLAN.json']:shutil.copyfile(source/n,root/n)
   e.write(root/'ADMISSION_017.json',{'fixture_only':True})
   e.freeze_forge(root,redistribution_allowed=True)
   e.write(root/'fit.json',{'arm':'context_mlp','seed':101,'updates':1024,'stop_update':64})
   from prepare017 import source_hashes,STUDY_SPEC
   review={'approved':True,'author':'unit-fixture-author','reviewer':'unit-fixture-reviewer','basis':'Synthetic API fixture, NOT actual review or training admission','source_sha256':source_hashes(root),'protocol':STUDY_SPEC}
   e.write(root/'review.json',review)
   recipe=e.prepare(root,root/'fit.json',root/'review.json')
   config=json.loads(recipe.read_text());self.assertEqual(config['execution_purpose'],'research');self.assertIn('research_adequacy',config)
   report=json.loads(recipe.with_name(recipe.stem+'_preflight.json').read_text());self.assertEqual(report['status'],'DECLARED_CRITERIA_MET');self.assertEqual(report['qualification'],'NOT_QUALIFIED')
   from trainlab.execution_adequacy import assess_execution
   contract=report['contract'];pins=report['input_hashes'];values={}
   for criteria in contract['execution']['dimensions'].values():
    for c in criteria:values[c['field']]=c['threshold']
   receipt={'schema':'trainlab.execution-adequacy-observations.v1','recipe_sha256':contract['recipe_sha256'],'inputs':pins,'producer':{'file':'ecosystem017.py','sha256':pins['ecosystem017.py']},'observations':values}
   self.assertEqual(assess_execution(contract,config,pins,receipt)['status'],'MET')
   receipt['observations']['new_updates']=63
   self.assertEqual(assess_execution(contract,config,pins,receipt)['status'],'INADEQUATE')
   (root/'model.py').write_text('changed')
   with self.assertRaises(ValueError):e.prepare(root,root/'fit.json',root/'review.json')
 def test_no_cli_training_default(self):
  import subprocess,sys
  r=subprocess.run([sys.executable,str(Path(e.__file__))],capture_output=True);self.assertNotEqual(r.returncode,0)
if __name__=='__main__':unittest.main()
