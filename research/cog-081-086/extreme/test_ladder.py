"""Fast invariant checks. Full dataset regeneration is a separate reproducibility run."""
import json, tempfile, unittest
from pathlib import Path
from difficulty import (LEVELS,MAZE_SIZES,SUDOKU_SPEC,HANOI_DISCS,
 sliding_distance_buckets,hanoi_min_moves,hanoi_moves,validate_hanoi_prefix,
 make_sliding4,force_mate,exact_mate_depth,ChessEndgame,
 make_sudoku_challenge,count_sudoku_solutions_capped,make_maze_challenge,maze_changed)
from score_ladder import environment,score_case,score

ROOT=Path(__file__).resolve().parent

class LadderTests(unittest.TestCase):
  def test_ten_levels_and_growth(self):
    self.assertEqual(len(LEVELS),10)
    self.assertEqual(MAZE_SIZES[-1],151)
    self.assertEqual(HANOI_DISCS[-1],22)
    self.assertTrue(all(x<y for x,y in zip(MAZE_SIZES,MAZE_SIZES[1:])))

  def test_entire_eight_puzzle_space(self):
    buckets=sliding_distance_buckets()
    self.assertEqual(sum(map(len,buckets.values())),181440)
    self.assertEqual(len(buckets[31]),2)

  def test_hanoi_symbolic_bounds(self):
    self.assertEqual(hanoi_min_moves(22),4194303)
    p=list(hanoi_moves(7))
    self.assertEqual(len(p),127)
    self.assertEqual(validate_hanoi_prefix(7,p)[1],(2,)*7)

  def test_15puzzle_reversible_but_not_optimality_claim(self):
    p,rev=make_sliding4(12345,96)
    self.assertTrue(p.apply(rev))
    self.assertEqual(len(rev),96)
    self.assertEqual(sorted(p.start),list(range(16)))

  def test_chess_minimax(self):
    cases=[json.loads(x) for x in (ROOT/'results'/'cases.jsonl').read_text().splitlines() if x]
    for depth in (1,2,3,4):
        c=next(c for c in cases if c['domain']=='chess' and c['rules']['maximum_required_mate_in']==depth)
        p=environment(c)
        self.assertEqual(exact_mate_depth(p),depth)
        self.assertTrue(force_mate(p.wk,p.heavy,p.bk,p.kind,depth))
        self.assertFalse(force_mate(p.wk,p.heavy,p.bk,p.kind,depth-1))
    c=next(c for c in cases if c['domain']=='chess' and c['level']==1)
    p=environment(c)
    move=p.mates()[0]
    self.assertTrue(score_case(c,{'actions':[list(move)]})['success'])
    self.assertFalse(force_mate(0,0,0,'Q',0))

  def test_bounded_unique_sudoku(self):
    p,profile=make_sudoku_challenge(25003,9)
    solutions,_=count_sudoku_solutions_capped(p)
    self.assertEqual(solutions,1)
    self.assertEqual(profile['clues']+profile['blanks'],81)

  def test_maze_replay(self):
    p,r=make_maze_challenge(902,8)
    self.assertTrue(p.apply(r.path))
    q,change=maze_changed(p,r.path)
    if q:
        self.assertFalse(q.apply(r.path))
        self.assertTrue(q.apply(change.path))

  def test_reference_validation_and_split(self):
    cases=[json.loads(x) for x in (ROOT/'results'/'cases.jsonl').read_text().splitlines() if x]
    evidence=[json.loads(x) for x in (ROOT/'results'/'reference_metrics.jsonl').read_text().splitlines() if x]
    self.assertEqual(len(cases),210)
    self.assertEqual(len(cases),len(evidence))
    ids=[c['id'] for c in cases]
    self.assertEqual(len(ids),len(set(ids)))
    dedupe=[c['input_sha256'] for c in cases if c['domain']!='hanoi']
    self.assertEqual(len(dedupe),len(set(dedupe)))
    for c in cases:
      if c['domain']=='chess':
        p=environment(c)
        self.assertEqual(exact_mate_depth(p),c['rules']['maximum_required_mate_in'])
      elif c['domain']=='sliding4':
        self.assertFalse(c['rules']['exact_optimal_depth'])

  def test_scorer_rejects_bad_actions(self):
    cases=[json.loads(x) for x in (ROOT/'results'/'cases.jsonl').read_text().splitlines() if x]
    for d in ('sliding3','maze','sudoku','hanoi'):
        c=next(c for c in cases if c['domain']==d)
        out=score_case(c,{'actions':['?']})
        self.assertFalse(out['success'])
    chess=next(c for c in cases if c['domain']=='chess')
    self.assertFalse(score_case(chess,{'actions':[['Q',1000,999]]})['success'])
    big=next(c for c in cases if c['domain']=='hanoi' and c['level']==10)
    self.assertEqual(score_case(big,{'actions':[]})['status'],'requires_streaming_evaluator')

  def test_scorer_no_implicit_success(self):
    cases=[json.loads(x) for x in (ROOT/'results'/'cases.jsonl').read_text().splitlines() if x]
    self.assertFalse(score_case(cases[0],None)['success'])

if __name__=='__main__':unittest.main()
