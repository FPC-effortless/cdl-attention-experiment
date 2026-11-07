import math
import unittest

import torch

from .data import batch, encode
from .information import example, forward_options, generate_answers, oracle_routes
from .model import Config, RelationalLM
from .train import counterfactual_step


class InformationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)
        self.model = RelationalLM(Config(width=8))
        self.examples = [example(400), example(401)]
        self.x, self.y, self.mask = batch(self.examples)

    def test_random_label_does_not_change_prefix_or_oracle_routes(self):
        for seed in range(100):
            clean, noise = example(seed), example(seed, random_answer=True)
            self.assertEqual(clean.text[:clean.answer_start], noise.text[:noise.answer_start])
            self.assertEqual(clean.fact_ends, noise.fact_ends)
            self.assertEqual(clean.answer_start, noise.answer_start)
            self.assertIn(noise.answer[1], "abcdefghijklmnopqrst")
        self.assertAlmostEqual(math.log2(20), 4.321928094887363)

    def test_oracle_has_no_future_or_answer_record(self):
        routes = oracle_routes(self.examples, self.x, 2)
        for row, ex in enumerate(self.examples):
            self.assertTrue((routes[:, row, :ex.answer_start - 1] == -1).all())
            self.assertTrue((routes[:, row] < ex.answer_start - 1).all())
        out = self.model(self.x, oracle_operands=routes)
        self.assertGreater(out.metrics["oracle_operand_reads"], 0)
        for row, ex in enumerate(self.examples):
            self.assertEqual(out.operands[-1][row, ex.answer_start - 1].tolist(), list(ex.fact_ends))

    def test_bad_oracle_routes_rejected(self):
        routes = oracle_routes(self.examples, self.x, 2)
        routes[0, 0, 0, 0] = 0
        with self.assertRaises(ValueError):
            self.model(self.x, oracle_operands=routes)
        routes[0, 0, 0, 0] = -2
        with self.assertRaises(ValueError):
            self.model(self.x, oracle_operands=routes)
        with self.assertRaises(ValueError):
            self.model(self.x, oracle_operands=routes.float())

    def test_masked_objective_is_explicit_and_finite(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002)
        receipt = counterfactual_step(self.model, optimizer, self.x, self.y, seed=71,
            forward_kwargs=forward_options("oracle_answer", self.examples, self.x, 2), loss_mask=self.mask)
        self.assertTrue(math.isfinite(receipt["nll"]))
        self.assertGreater(receipt["cost"]["oracle_operand_reads"], 0)
        self.assertEqual(receipt["forward_passes"], 2)

    def test_local_control_and_corruption(self):
        self.assertEqual(forward_options("local_lm", self.examples, self.x, 2), {"memory": "disabled"})
        clean = oracle_routes(self.examples, self.x, 2)
        corrupt = forward_options("oracle_lm", self.examples, self.x, 2, corrupt=True)["oracle_operands"]
        self.assertFalse(torch.equal(clean, corrupt))
        self.model(self.x, oracle_operands=corrupt)

    def test_free_generation_never_receives_gold_answer(self):
        seen = []
        handle = self.model.register_forward_pre_hook(lambda model, args: seen.append(args[0].clone()))
        generate_answers(self.model, self.examples, "local_lm")
        handle.remove()
        self.assertEqual(len(seen), 2)
        for row, ex in enumerate(self.examples):
            self.assertEqual(seen[0][row, :ex.answer_start].tolist(), encode(ex.text[:ex.answer_start]))
            self.assertTrue((seen[0][row, ex.answer_start:] == 0).all())
            self.assertTrue((seen[1][row, ex.answer_start + 1:] == 0).all())


if __name__ == "__main__":
    unittest.main()
