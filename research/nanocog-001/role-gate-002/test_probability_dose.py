from collections import Counter
from hashlib import sha256
import math
import random
import unittest

from probability_dose import match_scores,match_probabilities

class Geometry(unittest.TestCase):
    def test_balanced_and_noncollinear(self):
        d=match_probabilities([.25]*4,[.28,.22,.27,.23])
        self.assertEqual(d.status,'active')
        self.assertAlmostEqual(d.radius,d.requested)
        self.assertLessEqual(d.abs_cosine,.5)
        self.assertAlmostEqual(d.toward_actual,d.sham_actual,places=14)

    def test_border_joint_caps(self):
        d=match_probabilities([.001,.019,.705,.275],[.05,.1,.1,.75])
        self.assertTrue(d.capped)
        self.assertGreater(min(d.toward),0)
        self.assertGreater(min(d.sham),0)
        self.assertAlmostEqual(d.toward_actual,d.sham_actual,places=14)
        self.assertLess(d.radius,min(d.toward_cap,d.sham_cap))

    def test_zero_and_tiny(self):
        d=match_probabilities([.25]*4,[.25]*4);self.assertEqual(d.status,'tiny_native_donor')
        d=match_scores([0,-1000,0,0],[2,1,0,-1]);self.assertEqual(d.status,'native_support_zero')
        d=match_probabilities([.5,.5-1e-11,5e-12,5e-12],[.01,.8,.1,.09])
        self.assertIn(d.status,('active','tiny_common_feasible'))
        if d.status!='active':
            for y in (d.toward,d.sham):self.assertEqual(y,d.native)

    def test_feasibility_maximization(self):
        from probability_dose import PERMS, radius_to_boundary
        p=(.001,.019,.705,.275);q=(.05,.1,.1,.75)
        d=match_probabilities(p,q)
        raw=tuple(b-a for a,b in zip(p,q))
        m=math.fsum(raw)/4; u=tuple((x-m)/d.requested for x in raw)
        other=[]
        for pm in PERMS:
            v=tuple(u[i] for i in pm);cos=abs(math.fsum(a*b for a,b in zip(u,v)))
            if cos<=.5:
                cap=min(d.requested,.99*min(radius_to_boundary(p,u),radius_to_boundary(p,v)))
                other.append((-cap,cos,pm))
        radius,_,pm=min(other)
        self.assertEqual(pm,d.sham_permutation)
        self.assertAlmostEqual(d.radius,-radius,places=15)

    def test_reject_bad_geometry(self):
        for p in ([-.1,.3,.3,.5],[.25]*3,[.25]*5,[.3]*4,[.3,.3,.4,math.nan]):
            with self.assertRaises(ValueError):match_probabilities(p,[.25]*4)

    def test_f64_sweep(self):
        rng=random.Random(20261125)
        status=Counter()
        for _ in range(5000):
            p=[rng.gauss(0,10) for _ in range(4)];q=[rng.gauss(0,10) for _ in range(4)]
            a=match_scores(p,q);b=match_scores(p,q)
            self.assertEqual(a,b)
            status[a.status]+=1
            if a.status=='active':
                for pr in (a.toward,a.sham):
                    self.assertAlmostEqual(math.fsum(pr),1,places=13)
                    self.assertGreater(min(pr),0)
                self.assertLessEqual(a.abs_cosine,.5)
                self.assertTrue(math.isclose(a.toward_actual,a.sham_actual,rel_tol=3e-7,abs_tol=4e-16))
        self.assertGreater(status['active'],100)
        self.assertGreater(status['tiny_common_feasible'],10)
