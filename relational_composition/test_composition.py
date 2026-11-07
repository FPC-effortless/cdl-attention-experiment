import io
import unittest

import torch

from .data import batch, encode, make_example
from .index import CausalIndex
from .model import Config, RelationalLM, dereference, language_loss
from .train import counterfactual_step


class IndexTests(unittest.TestCase):
    def test_collisions_preserve_all_occurrences_and_cap_reads(self):
        index = CausalIndex(3)
        for t in range(10_000):
            index.append(t, 0, 0)
        self.assertEqual(len(index.postings[(0, 0)]), 10_000)
        self.assertEqual(index.lookup(0, 0, 9_000).candidates, (8999, 8998, 8997))
        self.assertEqual(index.lookup(0, 0, 9_000).omitted, 8997)
        self.assertEqual(index.lookup(0, 0, 1).candidates, (0,))
        self.assertEqual(index.lookup(0, 0, 0).candidates, ())

    def test_relation_address_isolation_and_empty(self):
        index = CausalIndex(2)
        index.append(0, 1, 2)
        index.append(1, 2, 2)
        self.assertEqual(index.lookup(1, 2, 2).candidates, (0,))
        self.assertEqual(index.lookup(1, 3, 2).candidates, ())

    def test_invalid_writes(self):
        with self.assertRaises(ValueError):
            CausalIndex(0)
        index = CausalIndex(1)
        index.append(1, 0, 0)
        with self.assertRaises(ValueError):
            index.append(1, 0, 0)
        with self.assertRaises(ValueError):
            index.append(0, 0, 0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.config = Config(width=8, layers=2, local_kernel=3)
        self.model = RelationalLM(self.config)
        self.x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    def forced_routes(self, model=None):
        output = (model or self.model)(self.x)
        choices = {k: torch.zeros_like(v) for k, v in output.decisions.items()}
        for k in choices:
            if k[1] == "continuation":
                choices[k].fill_(2)
        return choices

    def test_future_token_changes_do_not_affect_prefix(self):
        original = self.model(self.x)
        altered = self.x.clone()
        altered[:, 4:] += 20
        changed = self.model(altered)
        torch.testing.assert_close(original.logits[:, :4], changed.logits[:, :4], rtol=0, atol=0)
        for a, b in zip(original.operands, changed.operands):
            self.assertTrue(torch.equal(a[:, :4], b[:, :4]))

    def test_prefix_recomputation_agrees_with_full_pass(self):
        full = self.model(self.x)
        for t in range(1, self.x.shape[1] + 1):
            part = self.model(self.x[:, :t])
            torch.testing.assert_close(part.logits[:, -1], full.logits[:, t - 1], atol=1e-6, rtol=1e-6)
            for a, b in zip(part.references, full.references):
                self.assertTrue(torch.equal(a[:, -1], b[:, t - 1]))

    def test_actual_reads_and_references_strictly_earlier(self):
        out = self.model(self.x, overrides=self.forced_routes())
        positions = torch.arange(self.x.shape[1])[None, :]
        for refs, operands in zip(out.references, out.operands):
            self.assertTrue((refs < positions).all())
            self.assertTrue((operands < positions.unsqueeze(-1)).all())
        self.assertEqual(out.metrics["writes"], 16)
        self.assertEqual(out.metrics["reads"], 32)
        self.assertLessEqual(out.metrics["candidates"], 32 * self.config.candidates)
        self.assertGreater(out.metrics["omitted"], 0)

    def test_documents_and_padding_are_isolated(self):
        alone = self.model(self.x)
        joined = self.model(torch.cat((self.x, self.x + 30)))
        torch.testing.assert_close(alone.logits[0], joined.logits[0], atol=1e-6, rtol=1e-6)
        padded = self.model(torch.cat((self.x, torch.zeros(1, 3, dtype=torch.long)), 1))
        torch.testing.assert_close(alone.logits, padded.logits[:, :8], atol=1e-6, rtol=1e-6)
        self.assertEqual(alone.metrics["writes"], padded.metrics["writes"])
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[1, 0, 2]]))

    def test_null_does_not_read_last_record(self):
        h = torch.arange(24).reshape(1, 3, 8).float()
        read = dereference(h, torch.tensor([[-1, 0, 1]]))
        self.assertEqual(int(read[:, 0].count_nonzero()), 0)
        torch.testing.assert_close(read[:, 1], h[:, 0])

    def test_fixed_replay_is_exact(self):
        base = self.model(self.x, sample=True)
        replay = self.model(self.x, replay=base.decisions)
        torch.testing.assert_close(base.logits, replay.logits, rtol=0, atol=0)
        self.assertEqual(base.metrics, replay.metrics)

    def test_common_noise_recomputes_same_trajectory(self):
        base = self.model(self.x, sample=True)
        counter = self.model(self.x, sample=True, noise=base.noise)
        torch.testing.assert_close(base.logits, counter.logits, rtol=0, atol=0)
        for key in base.decisions:
            self.assertTrue(torch.equal(base.decisions[key], counter.decisions[key]))

    def test_writer_intervention_changes_later_reads(self):
        choices = self.forced_routes()
        before = self.model(self.x, overrides=choices)
        changed = {k: v.clone() for k, v in choices.items()}
        changed[(0, "write_address")][0, 0] = 1
        after = self.model(self.x, replay=choices, overrides={(0, "write_address"): changed[(0, "write_address")]})
        self.assertEqual(int(before.operands[0][0, 1, 0]), 0)
        self.assertEqual(int(after.operands[0][0, 1, 0]), -1)
        self.assertFalse(torch.equal(before.logits[:, 1], after.logits[:, 1]))

    def test_continuation_affects_next_router(self):
        choices = self.forced_routes()
        carried = self.model(self.x, overrides=choices)
        zero = torch.zeros_like(choices[(0, "continuation")])
        dropped = self.model(self.x, replay=choices, overrides={(0, "continuation"): zero})
        self.assertFalse(torch.equal(carried.policies[(1, "query")], dropped.policies[(1, "query")]))
        # Fixing every next-layer action removes the route-mediated output effect.
        torch.testing.assert_close(carried.logits, dropped.logits, rtol=0, atol=0)

    def test_operand_order_matters(self):
        choices = self.forced_routes()
        choices[(0, "rank")][:, :, 1] = 1
        first = self.model(self.x, overrides=choices)
        choices[(0, "rank")] = choices[(0, "rank")].flip(-1)
        swapped = self.model(self.x, overrides=choices)
        self.assertFalse(torch.equal(first.logits[:, 3:], swapped.logits[:, 3:]))

    def test_variants_have_equal_parameters_and_live_controls(self):
        counts = []
        for variant in ("coupled", "independent", "no_reference"):
            model = RelationalLM(Config(width=8, layers=2, local_kernel=3, variant=variant))
            model.load_state_dict(self.model.state_dict())
            counts.append(sum(p.numel() for p in model.parameters()))
            out = model(self.x, overrides=self.forced_routes(model))
            if variant == "no_reference":
                self.assertTrue(all((r == -1).all() for r in out.references))
                self.assertEqual(out.metrics["reference_dereferences"], 0)
            disabled = model(self.x, memory="disabled")
            self.assertTrue(all((p == -1).all() for p in disabled.operands))
        self.assertEqual(len(set(counts)), 1)

    def test_shuffled_control_cannot_leak_future(self):
        choices = self.forced_routes()
        normal = self.model(self.x, overrides=choices)
        shuffled = self.model(self.x, overrides=choices, memory="shuffled")
        self.assertFalse(torch.equal(normal.operands[0], shuffled.operands[0]))
        for ids in shuffled.operands:
            self.assertTrue((ids < torch.arange(8)[None, :, None]).all())
        self.assertEqual(normal.metrics["candidates"], shuffled.metrics["candidates"])

    def test_bad_override_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.x, overrides={(0, "rank"): torch.full((1, 8, 2), 100)})

    def test_training_cost_and_finite_gradients(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002)
        result = counterfactual_step(self.model, optimizer, self.x, self.x, seed=1,
                                     decision_key=(0, "write_address"))
        self.assertEqual(result["forward_passes"], 2)
        self.assertEqual(result["cost"]["writes"], 32)
        self.assertEqual(result["cost"]["reads"], 64)
        self.assertTrue(torch.isfinite(torch.tensor(result["nll"])))
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))

    def test_exact_optimizer_and_rng_resume(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002)
        counterfactual_step(self.model, optimizer, self.x, self.x, seed=8)
        buffer = io.BytesIO()
        torch.save(dict(model=self.model.state_dict(), optimizer=optimizer.state_dict(),
                        rng=torch.get_rng_state()), buffer)
        receipt_a = counterfactual_step(self.model, optimizer, self.x, self.x, seed=9)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        other = RelationalLM(self.config)
        other.load_state_dict(saved["model"])
        opt = torch.optim.AdamW(other.parameters(), lr=0.002)
        opt.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        receipt_b = counterfactual_step(other, opt, self.x, self.x, seed=9)
        self.assertEqual(receipt_a, receipt_b)
        for a, b in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


class DataTests(unittest.TestCase):
    def test_raw_bytes_and_answer_shift(self):
        example = make_example(11)
        x, y, mask = batch([example])
        self.assertEqual(bytes((y[mask] - 1).tolist()).decode(), example.answer)
        self.assertEqual(x[0, 1:].tolist(), y[0, :-1].tolist())
        self.assertNotEqual(encode("Nora"), encode("Niko"))

    def test_training_excludes_reserved_order(self):
        for i in range(100):
            self.assertNotIn("via visits then follows?", make_example(i).text)
            self.assertIn("via visits then follows?", make_example(i, "dev", "composition").text)
        with self.assertRaises(ValueError):
            make_example(1, "train", "names")

    def test_diagnostics_have_explicit_version_semantics(self):
        for condition in ("iid", "names", "composition", "paraphrase", "updates", "older"):
            example = make_example(40, "dev", condition)
            self.assertTrue(all(p < example.answer_start for p in example.fact_ends))
            self.assertTrue(example.text.endswith(example.answer + "\n"))
        self.assertIn("Before the update", make_example(40, "dev", "older").text)


if __name__ == "__main__":
    unittest.main()
