import unittest,random
from executor import Executor,reference
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.e=Executor()
 def check(self,ops,n=4,banks=None,state=None):
  r=reference(ops,n,state)
  for par in (False,True):
   a=self.e.execute(ops,n,state,banks,2,par);self.assertEqual(a['state'],r['state']);self.assertEqual(a['answers'],r['answers'])
 def test_snapshot_live_recreate_cycle(self):
  self.check([(1,0,0,0,0),(2,1,0,0,0),(3,2,0,0,0),(1,0,0,4294967295,0),(5,1,0,0,0),(5,2,0,0,0),(4,0,0,0,0),(5,2,0,0,0),(1,0,0,99,0),(5,2,0,0,0),(3,0,2,0,0),(5,0,0,0,0),(2,2,2,0,0)],banks=[0,0,0,0])
 def test_parallel_and_cross(self):
  ops=[(1,i,0,i,0) for i in range(4)]+[(5,i,0,0,0) for i in range(4)];self.check(ops)
  self.assertEqual(self.e.execute(ops)['stats']['actual_team_threads'],2)
  ops+=[(2,0,1,0,0)];self.check(ops);self.assertTrue(self.e.execute(ops)['stats']['cross_bank_fallback'])
 def test_high_payload(self):self.check([(1,0,0,0x01616263,0),(6,1,0,0xffffffff,0x01616263),(5,1,0,0,0)],banks=[0]*4)
 def test_conditions(self):self.check([(1,0,0,0,0),(6,1,0,22,0),(6,1,2,33,0),(5,1,0,0,0)],banks=[0]*4)
 def test_existing_cross_link(self):
  state=([2,1,0,0],[1,7,0,0]);ops=[(5,0,0,0,0)];self.check(ops,state=state);self.assertTrue(self.e.execute(ops,state=state)['stats']['cross_bank_fallback'])
 def test_random(self):
  rng=random.Random(2201)
  for trial in range(150):
   n=8;ops=[(rng.randrange(7),rng.randrange(n),rng.randrange(n),rng.randrange(1<<24),rng.randrange(1<<24)) for _ in range(100)];self.check(ops,n,[i//2 for i in range(n)])
 def test_random_parallel_local_dependencies(self):
  rng=random.Random(2203)
  for trial in range(60):
   ops=[]
   for t in range(240):
    b=rng.randrange(4);ops.append((rng.randrange(7),b*4+rng.randrange(4),b*4+rng.randrange(4),rng.randrange(8),rng.randrange(8)))
   self.check(ops,16,[i//4 for i in range(16)]);self.assertEqual(self.e.execute(ops,16,bank_ids=[i//4 for i in range(16)])['stats']['actual_team_threads'],2)
 def test_invalid_false_branch(self):
  for bad in [[(6,0,99,1,2)],[(6,0,0,-1,2)],[(6,0,0,1,-1)],[(9,0,0,0,0)]]:
   with self.assertRaises(ValueError):self.e.execute(bad)
 def test_selfcopy_detaches(self):self.check([(1,0,0,77,0),(3,1,0,0,0),(2,1,1,0,0),(4,0,0,0,0),(5,1,0,0,0)])
if __name__=='__main__':unittest.main()

class Pinned019Tests(unittest.TestCase):
 def test_pinned_reference(self):
  from reference019 import Memory
  e=Executor();rng=random.Random(2202);names=['NOOP','SET','COPY','LINK','DELETE','QUERY','IFSET']
  for trial in range(80):
   m=Memory(8);ops=[];answers=[]
   for t in range(120):
    op=rng.randrange(7);i=rng.randrange(8);j=rng.randrange(8);v=rng.randrange(256);g=rng.randrange(256);ops.append((op,i,j,v,g));a=-2
    if op==6:
     if m.read(j)==g:m.apply('SET',i,v=v)
    elif op==5:a=m.apply('QUERY',i)
    elif op:m.apply(names[op],i,j,v)
    answers.append(a)
   got=e.execute(ops,8,bank_ids=[i//2 for i in range(8)])
   kinds={'absent':0,'value':1,'ref':2};expected=([kinds[c.kind] for c in m.cells],[0 if c.kind=='absent' else c.payload for c in m.cells])
   self.assertEqual(got['state'],expected);self.assertEqual(got['answers'],answers)
