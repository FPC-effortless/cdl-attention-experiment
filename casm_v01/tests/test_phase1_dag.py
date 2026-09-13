import unittest

from casm_v01.phase1_dag.diagnostics import copy_mask_report, preflight
from casm_v01.phase1_dag.generator import generate_episode
from casm_v01.phase1_dag.oracle import evaluate_episode, exhaustive_minimality


class Phase1DAGTests(unittest.TestCase):
    def test_candidate_is_strict_superset(self):
        e = generate_episode(seed=7, n_inputs=2, n_ops=2)
        self.assertGreater(len(e.candidate_edges), len(e.true_edges))
        self.assertTrue(e.true_edge_set.issubset({(x.src, x.dst, x.port) for x in e.candidate_edges}))

    def test_oracle_matches_truth_table(self):
        e = generate_episode(seed=11, n_inputs=2, n_ops=2)
        self.assertEqual(evaluate_episode(e), e.truth_table)
        self.assertEqual(len(e.truth_table), 4)

    def test_copy_mask_is_not_the_oracle(self):
        e = generate_episode(seed=13, n_inputs=2, n_ops=2)
        r = copy_mask_report(e)
        self.assertGreater(r["distractor_edges"], 0)
        self.assertFalse(r["copy_mask_equals_true"])
        self.assertFalse(r["copy_mask_executable"])
        self.assertFalse(r["copy_mask_exact"])

    def test_exhaustive_check_has_global_status(self):
        e = generate_episode(seed=17, n_inputs=2, n_ops=1)
        r = exhaustive_minimality(e, max_edges=8)
        self.assertTrue(r["checked"])
        self.assertIn("unique_minimal", r)

    def test_preflight_requires_structural_discrimination(self):
        e = generate_episode(seed=19, n_inputs=2, n_ops=2)
        r = preflight(e, max_edges=12)
        self.assertGreater(r["distractor_edges"], 0)
        self.assertIn("minimality", r)


if __name__ == "__main__":
    unittest.main()
