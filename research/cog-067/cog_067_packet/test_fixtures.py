"""Small handwritten/no-run unit checks; never fit or score study cases."""
import unittest
from . import world as w
from . import policies as p
from .pipeline import oracle,scoped,one_group

C={'early':[[3]*8,[1]*8],'caps':[1,1],'effect':'strong','fault':'clean','workload':[[3]*12,[1]*12]}

class Fixtures(unittest.TestCase):
    def test_truth_fixture(self):
        self.assertEqual(w.truth(C),oracle(C))
        self.assertGreater(oracle(C)[1],oracle(C)[0])
        for e in ('null','reverse'):
            self.assertEqual(max(oracle({**C,'effect':e})),oracle({**C,'effect':e})[0])
    def test_fault_lineage(self):
        missing={**C,'fault':'journal_missing'}
        self.assertIsNone(w.replay(missing));self.assertEqual(w.direct(C),w.direct(missing))
        self.assertEqual(w.observe(C,'mirror'),w.passive(C))
    def test_scope(self):
        sub=scoped([('hand',[C])])[0][1][0]
        self.assertEqual(set(sub),{'early','caps','fault'})
    def test_opportunity_proof(self):
        rows=scoped([('hand',[C]*12)])[0][1]
        zero={'table':{},'hist':{},'tau':.55,'ask_floor':.06,'ask_threshold':.8,
              'fixed':{'0':'direct','1':'direct','2':'replay'}}
        for arm in p.MATCHED:
            log=one_group(zero,'hand',rows,arm)
            self.assertEqual(sum(r['status']=='DISPLACED' for r in log),4)
            self.assertEqual(log[-1]['clock'],12)
        free=one_group(zero,'hand',rows,'passive')
        self.assertFalse(any(r['status']=='DISPLACED' for r in free))
