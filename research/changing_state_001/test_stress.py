import unittest
from stress_data import generate, validate, encode, digest, canonical, DEPTHS, REGIMES
from run_frozen import summarize
class StressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.worlds=generate(set())
    def test_balance_shape_and_reproducibility(self):
        self.assertEqual(validate(self.worlds,set())['queries'],2560)
        self.assertEqual(self.worlds,generate(set()))
    def test_excludes_train_history(self):
        forbidden={digest(canonical(self.worlds[0]['writes']))}
        validate(generate(forbidden),forbidden)
    def test_one_changed_preservation(self):
        for w in self.worlds:
            rows=[encode(w,q) for q in range(4)]
            self.assertEqual(len(rows[0]['input_ids']),24+4*w['updates'])
            if w['regime']=='one_changed': self.assertEqual(sum(r['changed'] for r in rows),1)
            self.assertTrue(all(r['prior']!=r['answer'] for r in rows if r['changed']))
    def test_metric_denominators(self):
        rows=[{'family':'a','correct':True,'changed':False,'prediction':3,'prior':None},
              {'family':'a','correct':False,'changed':True,'prediction':2,'prior':2}]
        s=summarize(rows)
        self.assertEqual((s['accuracy'],s['complete_world_accuracy'],s['stale_rate'],s['untouched_accuracy']),(.5,0,1,1))
if __name__=='__main__': unittest.main()
