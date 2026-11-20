"""Hand-built invariants only. No fit, corpus execution, or panel scoring."""
from unittest import TestCase
from .domain import World, initial, step
from .planning import plan, acquire
from .episodes import cases, execute, flat_fields
from .reporting import template
from .verify import exact_cost, observe, score, verify_acquisition

class Fixtures(TestCase):
    def setUp(self):
        self.w=World(((0,1,1),(1,0,0),(2,2,1)),(2,2,2),('u','v','w'))
    def test_exact_option_invariance(self):
        s=initial(self.w);a=plan(self.w,s);b=plan(self.w,s,options=((4,1,4),(1,5,0,5)))
        self.assertEqual(a.cost,b.cost)
        self.assertIsNotNone(a.path)
    def test_oracle_and_full_receipts(self):
        w=self.w; r=execute(w,0)
        self.assertEqual(exact_cost(w),r['spent'])
        data=[((0,0,0),'success',0,w)]
        trace={'source_family':(0,0,0),'verified':True,'execution':r,
               'path':[x[1] for x in r['receipts']]}
        self.assertEqual(verify_acquisition(data,[trace]),1)
        with self.assertRaises(AssertionError):
            verify_acquisition(data,[{**trace,'path':[]}])
    def test_initial_hold(self):
        w=World(self.w.bays,self.w.wanted,self.w.names,blocked=1)
        r=execute(w,1)
        self.assertEqual(observe(w,r)[1],'blocked')
        self.assertEqual(r['receipts'],[])
    def test_revision_fails_closed(self):
        w=World(self.w.bays,self.w.wanted,self.w.names,revise_after=1)
        r=execute(w,1)
        self.assertEqual(observe(w,r)[1:3],('blocked',1))
        self.assertEqual(len(r['receipts']),1)
    def test_receipt_mutation_rejected(self):
        r=execute(self.w,0)
        lines=r['receipts']
        ep,a,s,c=lines[0];lines[0]=(ep+1,a,s,c)
        with self.assertRaises(AssertionError):observe(self.w,r)
    def test_communication_rejects_false_claim(self):
        r=execute(self.w,0);truth=observe(self.w,r)
        self.assertTrue(score(template(flat_fields(self.w,r)),truth)['strict'])
        fake=(truth[0],'blocked',*truth[2:])
        self.assertTrue(score(template(flat_fields(self.w,r)),fake)['false_completed'])
    def test_mine_from_multiple_families(self):
        src=[{'source_family':(i,),'verified':True,'path':[0,4,1,4,0,4,1,4]} for i in range(8)]
        found=acquire(src)
        self.assertTrue(found)
        self.assertTrue(all(2<=len(x)<=4 for x in found))
    def test_nonoverlap_metadata(self):
        data=cases()
        keys=[{x[0] for x in sub} for sub in data.values()]
        self.assertEqual(sum(map(len,keys)),512)
        self.assertFalse(keys[0]&keys[1] or keys[1]&keys[2] or keys[0]&keys[2])

