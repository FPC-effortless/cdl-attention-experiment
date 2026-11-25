"""Inert tensor fixtures only: no model/checkpoint or optimization."""
import unittest
import torch
from tap_gate_residual import TapGateResidual

class Geometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def make(self,arm):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(113)
            return TapGateResidual(arm=arm)
    def fixture(self):
        g=torch.Generator().manual_seed(42)
        q=torch.randn((3,7,32),generator=g,requires_grad=True)
        t=torch.randn((3,7,4,32),generator=g,requires_grad=True)
        v=torch.arange(7)[:,None]+torch.arange(4)[None,:]>=3
        x=torch.randn((3,7,4),generator=g,requires_grad=True)
        with torch.no_grad():x[0,0,0]=-0.
        return x,q,t,v
    def test_counts_and_identical_initialization(self):
        a,b=[self.make(arm) for arm in TapGateResidual.ARMS]
        self.assertEqual(sum(w.numel() for w in a.parameters()),552)
        self.assertEqual(sum(w.numel() for w in b.parameters()),552)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        for k in a.state_dict():self.assertTrue(torch.equal(a.state_dict()[k],b.state_dict()[k]))
    def test_zero_init_and_gradient(self):
        for arm in TapGateResidual.ARMS:
            m=self.make(arm);x,q,t,v=self.fixture()
            y=m.apply_to(x,q,t,v)
            self.assertTrue(torch.equal(y.view(torch.int32),x.view(torch.int32)))
            self.assertIs(m.apply_to(x,None,None,None,enabled=False),x)
            g=torch.arange(y.numel()).reshape(y.shape).float()
            (y*g).sum().backward()
            self.assertTrue(torch.equal(x.grad,g))
            self.assertIsNone(q.grad)
            self.assertIsNone(t.grad)
            self.assertTrue((m.last.weight.grad.abs()>0).all())
            self.assertTrue((m.first.weight.grad==0).all())
    def test_trainable_after_initialization_and_control_inert_taps(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            for arm in TapGateResidual.ARMS:
                m=self.make(arm);x,q,t,v=self.fixture()
                with torch.no_grad():m.last.weight.normal_()
                r=m(q,t,v)
                changed=t.detach().clone()
                changed[:,3,1]=changed[:,3,1]*1.35+1.25
                alt=m(q,changed,v)
                if arm=='tap_compare':
                    self.assertFalse(torch.equal(r[:,3],alt[:,3]))
                    self.assertTrue(torch.equal(r[:,:3],alt[:,:3]))
                else:self.assertTrue(torch.equal(r,alt))
                coef=torch.randn(r.shape)
                (r*coef).sum().backward()
                self.assertIsNone(q.grad)
                self.assertIsNone(t.grad)
                for p in m.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertTrue((p.grad!=0).all())
    def test_lag_maps_and_feature_rank(self):
        from feature_rank import ranks
        a,b=[self.make(arm) for arm in TapGateResidual.ARMS]
        g=torch.Generator().manual_seed(723)
        unique_q=torch.randn((16,32),generator=g)
        many_q=torch.randn((64,32),generator=g)
        t=torch.randn((1,64,4,32),generator=g)
        v=torch.ones((64,4),dtype=torch.bool)
        for q in (many_q,unique_q.repeat((4,1))):
            y=b.features(q.unsqueeze(0),t)
            for lag in range(4):
                self.assertEqual(len(set(b.literal_permutation[lag].tolist())),32)
            self.assertTrue(torch.equal(y[:,:,0,32:64],-y[:,:,1,32:64]))
            self.assertFalse(torch.equal(y[:,:,2,32:64],y[:,:,3,32:64]))
            ra=ranks(a,q.unsqueeze(0),t,v)
            rb=ranks(b,q.unsqueeze(0),t,v)
            self.assertEqual(ra['all_affine_input_width'],68)
            self.assertEqual(rb['all_affine_input_width'],68)
            if q is many_q:
                self.assertEqual(ra['all_active_feature_rank'],68)
                self.assertEqual(rb['all_active_feature_rank'],68)
            else:
                self.assertEqual(ra['literal_query_rank'],16)
                self.assertEqual(rb['literal_query_rank'],16)
                self.assertLessEqual(ra['all_active_feature_rank'],52)
                self.assertLessEqual(rb['all_active_feature_rank'],52)
    def test_mask_center_bounds_and_no_leak(self):
        m=self.make('tap_compare');x,q,t,v=self.fixture()
        with torch.no_grad():m.last.weight.fill_(100)
        alt=t.detach().clone(); alt[:,~v]=float('nan')
        one=m(q,t,v); two=m(q,alt,v)
        self.assertTrue(torch.equal(one,two))
        self.assertTrue(torch.isfinite(one).all())
        self.assertTrue(torch.all(one.abs()<=4))
        self.assertTrue(torch.equal(one[:,~v],torch.zeros_like(one[:,~v])))
        torch.testing.assert_close(one.sum(-1),torch.zeros((3,7)),atol=1e-6,rtol=0)
        active=torch.arange(7)==6
        lone=m(q,t,v,active=active)
        self.assertTrue(torch.equal(lone[:,:6],torch.zeros_like(lone[:,:6])))
        torch.testing.assert_close(lone[:,-1],one[:,-1],atol=3e-7,rtol=3e-7)
    def test_empty_and_contract(self):
        m=self.make('tap_compare');x,q,t,v=self.fixture()
        self.assertEqual(m(q[:,:0],t[:,:0],v[:0]).shape,(3,0,4))
        with self.assertRaises(ValueError):m(q,t[:,:,:3],v)
        with self.assertRaises(ValueError):m(q,t,v.float())
