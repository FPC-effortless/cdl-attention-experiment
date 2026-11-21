import json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent.parent))
from maze_transfer_091 import (feature,explore,collect_training,experience_model,preregistered_budget)
from score_ladder import environment

class TestOnlineMaze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=[json.loads(x) for x in (HERE.parent/'results'/'cases.jsonl').read_text().splitlines()]
    def test_observation_only(self):
        c=next(c for c in self.cases if c['domain']=='maze');m=environment(c)
        r=explore(m,budget=2*m.size*m.size)
        self.assertTrue(r['solved'])
        self.assertTrue(m.apply(r['walk']))
        self.assertEqual(r['illegal_actions'],0)
    def test_split(self):
        traces,model,options,_=collect_training(self.cases)
        self.assertEqual(len(traces),10)
        train={x['source_family'] for x in traces}
        test={c['input_sha256'] for c in self.cases if c['split']=='reserved' and c['domain']=='maze'}
        self.assertFalse(train & test)
        self.assertTrue(all(t['verified'] for t in traces))
    def test_corruption_does_not_change_test_reference(self):
        c=next(c for c in self.cases if c['split']=='reserved' and c['domain']=='maze')
        m=environment(c)
        a=explore(m,'heuristic',budget=preregistered_budget(m.size))
        b=explore(m,'learned',model={},alpha=0.0,budget=preregistered_budget(m.size))
        self.assertEqual(a['walk'],b['walk'])
    def test_support_requires_distinct_roots(self):
        fake=[{'verified':True,'source_family':'A','observations':[((1,1,'DR'),'D')]*20}]
        self.assertFalse(experience_model(fake))
        fake.append({'verified':True,'source_family':'B','observations':[((1,1,'DR'),'D')]})
        self.assertEqual(experience_model(fake)[(1,1,'DR')]['D'],2)
    def test_budget_enforced(self):
        c=next(c for c in self.cases if c['split']=='reserved' and c['level']==10 and c['domain']=='maze')
        m=environment(c)
        r=explore(m,'heuristic',budget=5)
        self.assertFalse(r['solved']);self.assertLessEqual(r['actions'],5)

if __name__=='__main__':unittest.main()
