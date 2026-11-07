import math
import unittest

import torch

from relational_composition.design_selection import select_qualified
from relational_composition.hierarchical_index import bits_page
from relational_composition.math_spec import schedule_cost
from relational_composition.train_hierarchical import focused_positions


class IntegrationTests(unittest.TestCase):
    def test_focus_tracks_each_rows_answer_window(self):
        x = torch.ones((2, 12), dtype=torch.long)
        x[0, 7:] = 0
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[0, 4:6] = True
        mask[1, 10:12] = True
        readers = focused_positions(x, mask, writer=False)
        self.assertEqual(readers, [(0, t) for t in range(2, 6)] + [(1, t) for t in range(8, 12)])
        writers = focused_positions(x, mask, writer=True)
        self.assertEqual(writers, [(0, t) for t in range(4)] + [(1, t) for t in range(10)])
        reversed_readers = focused_positions(x.flip(0), mask.flip(0), writer=False)
        self.assertEqual(set(readers), {(1-r, t) for r, t in reversed_readers})

    def test_unlabeled_rows_are_not_used_for_focused_credit(self):
        x = torch.ones((2, 8), dtype=torch.long)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[1, 6] = True
        self.assertTrue(all(r == 1 for r, _ in focused_positions(x, mask, writer=True)))
        self.assertTrue(all(r == 1 for r, _ in focused_positions(x, mask, writer=False)))

    def test_selection_rejects_nonfinite_reference_costs(self):
        for key in ('reference_inference_ms', 'reference_training_ms', 'memory_limit_mib', 'inference_weight'):
            for bad in (math.nan, math.inf, -math.inf):
                options = dict(reference_inference_ms=1., reference_training_ms=1.)
                options[key] = bad
                with self.assertRaises(ValueError):
                    select_qualified([], **options)

    def test_cost_rejects_nonfinite_inputs(self):
        base = dict(inference_ms=1., baseline_inference_ms=1., training_ms=1.,
                    baseline_training_ms=1., inference_weight=12.)
        for key in base:
            for bad in (math.nan, math.inf):
                with self.assertRaises(ValueError):
                    schedule_cost(**{**base, key: bad})

    def test_fractional_page_digits_are_not_silently_truncated(self):
        for bits in ([0.5, 1], [1.8, 0], ["1", 0]):
            with self.assertRaises(ValueError):
                bits_page(bits)


if __name__ == '__main__':
    unittest.main()
