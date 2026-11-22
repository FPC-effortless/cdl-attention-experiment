import unittest
import torch
from .model import Config, Decoder


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = Decoder(Config(31, 32)).eval()
        with torch.no_grad():
            self.model.smear_gain.fill_(0.7)
        self.x = torch.randint(31, (2, 9))

    def assertClose(self, a, b):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_chunk_and_token_parity_with_active_smear(self):
        with torch.inference_mode():
            full, sf = self.model(self.x)
            state, outputs = None, []
            for x in torch.split(self.x, [3, 1, 5], dim=1):
                y, state = self.model(x, state)
                outputs.append(y)
        self.assertClose(full, torch.cat(outputs, dim=1))
        self.assertEqual(sf.tokens_seen, state.tokens_seen)
        self.assertClose(sf.previous_embedding, state.previous_embedding)
        for u, v in zip(sf.mixer_states, state.mixer_states):
            self.assertClose(u, v)

    def test_suffix_cannot_change_prefix(self):
        with torch.inference_mode():
            y, _ = self.model(self.x)
            x = self.x.clone()
            x[:, 5:] = (x[:, 5:] + 1) % 31
            ys, _ = self.model(x)
        self.assertClose(y[:, :5], ys[:, :5])

    def test_fork_reuses_prefix_without_mutation(self):
        with torch.inference_mode():
            _, state = self.model(self.x[:, :3])
            a, _ = self.model(self.x[:, 3:], state)
            alternate = (self.x[:, 3:] + 1) % 31
            b, _ = self.model(alternate, state)
            af, _ = self.model(self.x)
            bf, _ = self.model(torch.cat((self.x[:, :3], alternate), dim=1))
        self.assertClose(a, af[:, 3:])
        self.assertClose(b, bf[:, 3:])
        self.assertEqual(state.tokens_seen, 3)

    def test_rejects_padded_token_ids_and_empty_mean_loss(self):
        invalid = self.x.clone()
        invalid[0, 0] = 50  # below alignment but greater than actual vocabulary
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.model(invalid)
        with self.assertRaisesRegex(ValueError, "observed target"):
            self.model(self.x, targets=torch.full_like(self.x, -1))
        loss, _ = self.model(self.x, targets=torch.full_like(self.x, -1), loss_reduction="sum")
        self.assertEqual(loss.item(), 0.0)

    def test_loss_gradient_and_reset(self):
        loss, state = self.model(self.x[:, :-1], targets=self.x[:, 1:])
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(self.model.mixers[0].conv.weight.grad)
        with torch.no_grad():
            x = self.x[:1]
            z1, _ = self.model(x)
            z2, _ = self.model(x, None)
            self.assertClose(z1, z2)
            with self.assertRaisesRegex(ValueError, "geometry"):
                self.model(x, state)


if __name__ == "__main__":
    unittest.main()
