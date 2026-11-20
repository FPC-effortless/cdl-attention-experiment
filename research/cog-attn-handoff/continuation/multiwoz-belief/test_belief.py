"""Isolation, carry, clearing and dialogue lifecycle regressions."""
import unittest
import numpy as np
from study import DialogueState, EMPTY, HOLD

class Isolation(unittest.TestCase):
    def test_commit_boundary_and_lifecycle(self):
        slots = dict(area=[EMPTY,HOLD,'north','south'], price=[EMPTY,HOLD,'cheap'])
        thresholds = dict(area=.95, price=.95)
        state = DialogueState(slots,thresholds,'soft')
        def observe(turn,a,p):
            return state.observe(turn,[np.array(a),np.array(p)])
        try:
            self.assertEqual(observe(0,[0,.04,.96,0],[0,.3,.7]), {'area':'north'})
            self.assertEqual(state.verified(), {'area':'north'})
            self.assertEqual(state.estimates()['price'][:2],('cheap',True))
            self.assertEqual(observe(2,[0,.15,0,.85],[0,1,0]),{})
            self.assertEqual(state.estimates()['area'][:2],('south',True))
            with self.assertRaises(ValueError):
                observe(2,[0,0,1,0],[0,0,1])
            self.assertEqual(state.verified(),{'area':'north'})
            self.assertEqual(observe(4,[.98,.02,0,0],[0,.02,.98]),{'area':None,'price':'cheap'})
            self.assertEqual(state.verified(),{'price':'cheap'})
            self.assertEqual(state.estimates()['area'],(EMPTY,False,1.))
            self.assertEqual(state.estimates()['price'],('cheap',False,1.))
            self.assertEqual(state._store.db.execute('select count(*) from journal').fetchone()[0],2)
        finally:
            state.close()
        fresh = DialogueState(slots,thresholds,'soft')
        try:
            self.assertEqual(fresh.verified(),{})
            self.assertTrue(all(r==(EMPTY,False,1.) for r in fresh.estimates().values()))
        finally:
            fresh.close()

if __name__=='__main__':
    unittest.main()
