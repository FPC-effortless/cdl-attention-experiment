"""Explicit dependency hazards, added prospectively before scientific freeze."""
import unittest
from executor import Executor,reference
class CrossBankHazards(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.e=Executor()
 def parity(self,ops,state=None):
  expected=reference(ops,4,state)
  serial=self.e.execute(ops,4,state,[0,1,2,3],4,False)
  parallel=self.e.execute(ops,4,state,[0,1,2,3],4,True)
  for got in (serial,parallel):
   self.assertEqual(got['answers'],expected['answers']);self.assertEqual(got['state'],expected['state'])
  self.assertTrue(parallel['stats']['cross_bank_fallback'])
  self.assertEqual(parallel['stats']['actual_team_threads'],0)
  return parallel
 def test_live_query_and_snapshot_before_later_write(self):
  ops=[(3,0,1,0,0),(1,1,0,1,0),(5,0,0,0,0),(2,2,0,0,0),(1,1,0,2,0),(5,0,0,0,0),(5,2,0,0,0)]
  got=self.parity(ops);self.assertEqual([got['answers'][i] for i in (2,5,6)],[1,2,1])
 def test_transitive_initial_links_delete_recreate(self):
  state=([2,2,1,0],[1,2,1,0]) # A->B->C, C=1
  ops=[(5,0,0,0,0),(2,3,0,0,0),(1,2,0,2,0),(5,0,0,0,0),(4,2,0,0,0),(5,0,0,0,0),(1,2,0,3,0),(5,0,0,0,0),(5,3,0,0,0)]
  got=self.parity(ops,state);self.assertEqual([got['answers'][i] for i in (0,3,5,7,8)],[1,2,None,3,1])
 def test_cross_bank_cycle_then_repair_and_selfcopy(self):
  ops=[(3,0,1,0,0),(3,1,2,0,0),(3,2,0,0,0),(5,0,0,0,0),(1,2,0,7,0),(5,0,0,0,0),(2,0,0,0,0),(1,2,0,8,0),(5,0,0,0,0),(5,1,0,0,0)]
  got=self.parity(ops);self.assertEqual([got['answers'][i] for i in (3,5,8,9)],[None,7,7,8])
 def test_transitive_conditional_uses_prestate(self):
  state=([2,2,1,0],[1,2,1,0])
  ops=[(6,3,0,9,1),(1,2,0,2,0),(6,3,0,8,1),(5,3,0,0,0)]
  got=self.parity(ops,state);self.assertEqual(got['answers'][-1],9)
if __name__=='__main__':unittest.main()
