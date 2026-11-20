import tempfile,json,unittest
from pathlib import Path
from assemble_private_inputs import assemble,digest
class AssemblyTests(unittest.TestCase):
 def fixture(self,root):
  code=root/'code';asset=root/'asset';code.mkdir();asset.mkdir();(code/'main.py').write_text('pass\n');(asset/'weight.bin').write_bytes(b'weights')
  for n in ('REVIEW.json','ADMISSION.json'):(code/n).write_text('{}')
  (code/'INPUTS.json').write_text(json.dumps({'files':{'main.py':digest(code/'main.py'),'checkpoints/weight.bin':digest(asset/'weight.bin')}}))
  return code,asset
 def test_exact_copy_and_private_permissions(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);c,a=self.fixture(r);out=r/'out';receipt=assemble(c,a,out,digest(c/'INPUTS.json'));self.assertEqual(receipt['files'],5)
   for p in out.rglob('*'):self.assertEqual(p.stat().st_mode&0o777,0o700 if p.is_dir() else 0o600)
 def test_wrong_manifest_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);c,a=self.fixture(r)
   with self.assertRaises(ValueError):assemble(c,a,r/'out','bad')
   self.assertFalse((r/'out').exists())
 def test_wrong_asset_rejected_before_copy(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);c,a=self.fixture(r);(a/'weight.bin').write_bytes(b'changed')
   with self.assertRaises(ValueError):assemble(c,a,r/'out',digest(c/'INPUTS.json'))
   self.assertFalse((r/'out').exists())
 def test_path_escape_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);c,a=self.fixture(r);(c/'INPUTS.json').write_text(json.dumps({'files':{'../escape':digest(a/'weight.bin')}}))
   with self.assertRaises(ValueError):assemble(c,a,r/'out',digest(c/'INPUTS.json'))
 def test_asset_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);c,a=self.fixture(r);(a/'weight.bin').rename(r/'weight.bin');(a/'weight.bin').symlink_to(r/'weight.bin')
   with self.assertRaises(ValueError):assemble(c,a,r/'out',digest(c/'INPUTS.json'))
if __name__=='__main__':unittest.main()
