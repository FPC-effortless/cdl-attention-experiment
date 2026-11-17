import unittest
from native022 import validate_result,path
from pathlib import Path
class Native(unittest.TestCase):
 def test_result_contract(self):
  c={'expected_metrics':{'scientific_fits':0,'protected_test_examples':0},'expected_runtime':{'device':'cpu','torch':'pinned'}}
  r={'metrics':dict(c['expected_metrics']),'runtime':dict(c['expected_runtime'])}
  self.assertEqual(validate_result(r,c),c['expected_metrics'])
  for bad in [{'metrics':{'scientific_fits':1,'protected_test_examples':0},'runtime':r['runtime']},{'metrics':r['metrics'],'runtime':{'device':'cpu','torch':'other'}}]:
   with self.assertRaises(ValueError):validate_result(bad,c)
 def test_paths(self):
  for n in ['../escape','/absolute']:
   with self.assertRaises(ValueError):path(Path('.'),n)
if __name__=='__main__':unittest.main()
