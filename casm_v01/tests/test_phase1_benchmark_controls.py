import unittest

from casm_v01.phase1_dag.benchmark_preflight import run_suite
from casm_v01.phase1_dag.controls import nonoracle_valid_control, valid_wirings
from casm_v01.phase1_dag.generator import generate_episode


class Phase1BenchmarkControlTests(unittest.TestCase):
    def test_nonoracle_valid_wiring_exists_and_is_functionally_different(self):
        episode = generate_episode(seed=7, n_inputs=2, n_ops=2)
        result = nonoracle_valid_control(episode)
        self.assertTrue(result["found"])
        self.assertGreater(result["checked"], 0)

    def test_valid_wirings_preserve_arity_and_topology(self):
        episode = generate_episode(seed=11, n_inputs=2, n_ops=2)
        wirings = valid_wirings(episode)
        self.assertGreater(len(wirings), 1)
        for wiring in wirings:
            self.assertEqual(len(wiring), len(episode.true_edges))
            for edge in wiring:
                self.assertLess(edge.src, edge.dst)

    def test_preflight_suite_requires_real_selection_problem(self):
        report = run_suite(seeds=range(8), n_inputs=2, n_ops=2, max_edges=12)
        self.assertEqual(report["oracle_exact"], 8)
        self.assertEqual(report["copy_mask_equals_true"], 0)
        self.assertEqual(report["copy_mask_executable"], 0)
        self.assertEqual(report["copy_mask_exact"], 0)
        self.assertEqual(report["minimality_checked"], 8)
        self.assertEqual(report["nonoracle_valid_found"], 8)
        self.assertTrue(report["ready_for_router"])


if __name__ == "__main__":
    unittest.main()
