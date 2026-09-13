import unittest

from casm.diagnostics import copy_mask_report, phase1_preflight
from casm.generator import generate_episode
from casm.oracle import evaluate_episode, exhaustive_minimality


class TestCASMPhase1(unittest.TestCase):
    def test_candidate_substrate_is_strict_superset(self):
        episode = generate_episode(seed=7, n_inputs=2, n_ops=2)
        self.assertGreater(len(episode.candidate_edges), len(episode.true_edges))
        self.assertTrue(
            episode.true_edge_set.issubset(
                {(e.src, e.dst, e.port) for e in episode.candidate_edges}
            )
        )

    def test_oracle_reproduces_stored_truth_table(self):
        episode = generate_episode(seed=11, n_inputs=2, n_ops=2)
        self.assertEqual(evaluate_episode(episode), episode.truth_table)
        self.assertEqual(len(episode.truth_table), 4)

    def test_copy_mask_cannot_be_silent_oracle(self):
        episode = generate_episode(seed=13, n_inputs=2, n_ops=2)
        report = copy_mask_report(episode)
        self.assertGreater(report.distractor_edges, 0)
        self.assertFalse(report.copy_mask_equals_true)
        self.assertFalse(report.copy_mask_exact)
        self.assertFalse(report.copy_mask_executable)

    def test_exhaustive_check_is_explicit_about_small_domain(self):
        episode = generate_episode(seed=17, n_inputs=2, n_ops=1)
        report = exhaustive_minimality(episode, max_edges=8)
        self.assertTrue(report["checked"])
        self.assertIn("unique_minimal", report)

    def test_preflight_blocks_router_when_oracle_is_unproven(self):
        report = phase1_preflight(seed=19, n_inputs=2, n_ops=2)
        self.assertGreater(report["distractor_edge_count"], 0)
        self.assertIn("global_minimality_check", report)


if __name__ == "__main__":
    unittest.main()
