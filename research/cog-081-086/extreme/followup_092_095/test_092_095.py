"""Invariants for source-root learning, heldout isolation and honest scoring."""
import json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent),str(HERE.parents[1])]
from score_ladder import environment,score_chess
from maze_092 import navigate,load_cases,collect_training,calibrate,preregistered_budget
from sliding_093 import search,learn
from sliding_fast_093b import fast
from sudoku_094 import solve,fit
from chess_095 import search as chess_search
from verify_stable import run as verify

class HiddenMaze:
    """Test that navigation cannot read underlying full graph or reference path."""
    def __init__(self,m):self.m=m
    @property
    def size(self):return self.m.size
    @property
    def start(self):return self.m.start
    @property
    def target(self):return self.m.target
    def neighbors(self,state):return self.m.neighbors(state)
    def apply(self,actions):return self.m.apply(actions)
    def shortest(self):raise AssertionError('reference shortest path forbidden')
    @property
    def open_cells(self):raise AssertionError('global wall map forbidden')

class Qualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=load_cases()
        cls.r={f:json.loads((HERE/'results'/f).read_text()) for f in (
            'maze-092.json','sliding-093.json','sliding-093b.json','sliding-093c.json',
            'sudoku-094.json','chess-095.json')}

    def test_training_vs_reserved_disjoint(self):
        for domain in ('maze','sudoku','chess','sliding3'):
            training={x['input_sha256'] for x in self.cases if x['domain']==domain and x['split']=='development'}
            reserve={x['input_sha256'] for x in self.cases if x['domain']==domain and x['split']=='reserved'}
            self.assertTrue(training.isdisjoint(reserve),domain)

    def test_fresh_sliding_disjoint(self):
        source={x['input_sha256'] for x in self.cases if x['domain']=='sliding3'}
        cohort={x['sha'] for x in self.r['sliding-093c.json']['rows']}
        self.assertEqual(len(cohort),32)
        self.assertTrue(cohort.isdisjoint(source))

    def test_sliding_optimality_and_parity(self):
        data=self.r['sliding-093c.json']
        for row in data['rows']:
            self.assertTrue(row['solved'])
            self.assertEqual(row['actions'],row['depth'])
        previous=self.r['sliding-093.json']['summary']
        optimized=self.r['sliding-093b.json']['summary']
        for arm in previous:
            for key in ('n','solved','attempts','expanded'):
                self.assertEqual(previous[arm][key],optimized[arm][key])

    def test_fallback_on_empty_sliding_memory(self):
        test=next(x for x in self.cases if x['domain']=='sliding3' and x['split']=='reserved')
        p=environment(test)
        a=fast(p,{},'primitive')
        b=fast(p,{},'learned')
        for key in ('solved','actions','expanded','attempts'):
            self.assertEqual(a[key],b[key])

    def test_maze_local_only(self):
        case=next(x for x in self.cases if x['domain']=='maze' and x['level']==2)
        maze=HiddenMaze(environment(case))
        result=navigate(maze,{},0,2,1,0,preregistered_budget(maze.size)*20)
        self.assertTrue(result['solved']);self.assertEqual(result['illegal_actions'],0)

    def test_maze_correction_did_not_hide_failures(self):
        r=self.r['maze-092.json']
        self.assertEqual(r['admission']['correction'],0.0)
        for factor in ('1','2'):
            self.assertEqual(r['summary'][factor]['gated']['solved'],r['summary'][factor]['corrected']['solved'])

    def test_sudoku_matches_selected_reference_primitive(self):
        r=self.r['sudoku-094.json']
        self.assertEqual(r['frozen_n_policies'],{'4':'mrv','9':'mrv'})
        for row in r['rows']:
            if row['arm']=='learned':self.assertTrue(row['solved'])
        self.assertEqual(r['summary']['learned']['nodes'],r['summary']['mrv']['nodes'])

    def test_chess_forced_strategy_replayed_against_all_responses(self):
        case=next(x for x in self.cases if x['domain']=='chess' and x['split']=='reserved' and x['level']==4)
        p=environment(case);depth=case['rules']['maximum_required_mate_in']
        for arm in ('primitive','learned'):
            r=chess_search(p,depth,arm,{},2500)
            self.assertTrue(r['solved']);self.assertFalse(r['exhausted'])
        self.assertEqual(self.r['chess-095.json']['summary']['primitive']['solved'],20)

    def test_stable_receipts(self):self.assertEqual(len(verify()),6)

if __name__=='__main__':unittest.main()
