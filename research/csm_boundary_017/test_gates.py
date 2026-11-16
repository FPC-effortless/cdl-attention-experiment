import unittest,copy
from gates import gates,FIELDS
class GatesTests(unittest.TestCase):
 def test_strict(self):
  r={'seed':101,**{s:{'p':{'records':[{'case_id':'case1'}],'metrics':{'counts':{'cases':1},**{f:1. for f in FIELDS}}}} for s in ('dev','train_probe')}};roster={s:{'p':['case1']} for s in ('dev','train_probe')}
  self.assertTrue(gates(r,roster,101)['qualified'])
  for mut in ('empty','nan','none','seed','count'):
   q=copy.deepcopy(r)
   if mut=='empty':q['dev']={}
   elif mut=='seed':q['seed']=202
   elif mut=='count':q['dev']['p']['metrics']['counts']['cases']=0
   else:q['dev']['p']['metrics'][FIELDS[0]]=float('nan') if mut=='nan' else None
   with self.assertRaises(ValueError):gates(q,roster,101)
if __name__=='__main__':unittest.main()
