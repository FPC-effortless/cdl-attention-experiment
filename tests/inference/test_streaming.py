"""No learned competence claimed: tests cached greedy equivalence and causality."""
import unittest
import torch
from relational_composition.model import Config, RelationalLM
from relational_composition.fast_inference import StreamingInference

class StreamingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_full_prefix_equivalence_across_variants_shapes_and_seeds(self):
        for variant in ('coupled', 'independent', 'no_reference'):
            for seed in (2, 7, 12):
                torch.manual_seed(seed)
                config = Config(width=12, layers=3, local_kernel=5, variant=variant)
                model = RelationalLM(config).eval()
                x = torch.randint(1, 97, (2, 17))
                fast = StreamingInference(model)
                for t in range(1, x.shape[1] + 1):
                    cache_logits = fast.step(x[:, t - 1])
                    full = model(x[:, :t])
                    torch.testing.assert_close(cache_logits, full.logits[:, -1],
                                               rtol=2e-5, atol=2e-5,
                                               msg=f'variant={variant} seed={seed} t={t}')
                self.assertEqual(fast.cache.position, x.shape[1])
                self.assertEqual(fast.cache.count, full.metrics)

    def test_generation_parity(self):
        torch.manual_seed(19)
        model = RelationalLM(Config(width=16, layers=2)).eval()
        prompt = torch.randint(1, 100, (1, 24))
        fast = StreamingInference(model)
        generated = fast.greedy_generate(prompt, 5)
        seq = prompt.clone()
        for j in range(5):
            ref = model(seq).logits[:, -1].argmax(-1)
            torch.testing.assert_close(generated[:, j], ref, rtol=0, atol=0)
            if j + 1 < 5:
                seq = torch.cat((seq, ref.unsqueeze(1)), dim=1)

    def test_prefix_conditioning_and_reset(self):
        torch.manual_seed(9)
        model = RelationalLM(Config(width=8, local_kernel=1, layers=1)).eval()
        fast = StreamingInference(model)
        x = torch.randint(1, 200, (3, 11))
        a = fast.prefill(x)
        b = model(x).logits[:, -1]
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        with self.assertRaises(ValueError):
            fast.step(torch.tensor([0, 1, 2]))
        with self.assertRaises(ValueError):
            fast.step(torch.tensor([1, 2]))
        torch.testing.assert_close(fast.prefill(x), a, rtol=0, atol=0)

if __name__ == '__main__':
    unittest.main()
