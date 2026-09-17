import unittest

import torch

from casm_v01.phase1_dag.casm_s import (
    CASMExecutor,
    FactorizedRouter,
    StaticMask,
    assert_integrity,
    batch_forward,
    gate_0,
    structural_tensor,
)
from casm_v01.phase1_dag.generator import generate_episode


class Phase1CASMSTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.episodes = [generate_episode(seed=s, n_inputs=2, n_ops=2) for s in range(8)]

    def test_gate_zero_initialization(self):
        router = FactorizedRouter()
        d = gate_0(router, self.episodes)
        self.assertAlmostEqual(d["gate_mean"], 0.5, delta=0.05)

    def test_alpha_is_relation_indexed_and_unit_initialized(self):
        ex = CASMExecutor()
        self.assertEqual(tuple(ex.eta.shape), (2,))
        self.assertTrue(torch.allclose(ex.alpha(), torch.ones(2), atol=1e-6))

    def test_static_mask_uses_physical_edge_identity(self):
        model = StaticMask()
        e0 = self.episodes[0]
        e1 = self.episodes[1]
        g0 = model.gates(structural_tensor(e0), e0.candidate_edges)
        g1 = model.gates(structural_tensor(e1), e1.candidate_edges)
        self.assertEqual(len(g0), len(e0.candidate_edges))
        self.assertEqual(len(g1), len(e1.candidate_edges))
        self.assertEqual(len(model.logits), 10)

    def test_router_has_gradient_path(self):
        router = FactorizedRouter()
        executor = CASMExecutor()
        loss, _, _ = batch_forward(self.episodes[:2], router, executor)
        loss.backward()
        grads = [p.grad for p in router.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))

    def test_value_independence_is_structural(self):
        router = FactorizedRouter()
        e = self.episodes[0]
        s = structural_tensor(e)
        g1 = router.gates(s, e.candidate_edges)
        g2 = router.gates(s, e.candidate_edges)
        self.assertTrue(torch.equal(g1, g2))

    def test_integrity_gate(self):
        assert_integrity(FactorizedRouter(), CASMExecutor(), self.episodes)


if __name__ == "__main__":
    unittest.main()
