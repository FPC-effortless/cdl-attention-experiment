import unittest
import torch
from .model import Config, Decoder
from .interaction import PacketMixer, PacketAndTypedProbe


class CausalIntegration(unittest.TestCase):
    def check_mixer(self, mixer):
        torch.manual_seed(347)
        model = Decoder(Config(41, 32), mixer).eval()
        ids = torch.randint(41, (2, 11))
        with torch.inference_mode():
            full, _ = model(ids)
            prefix, state = model(ids[:, :4])
            first, _ = model(ids[:, 4:], state)
            new_suffix = (ids[:, 4:] + 13) % 41
            second, _ = model(new_suffix, state)
            changed, _ = model(torch.cat((ids[:, :4], new_suffix), 1))
            reset, _ = model(new_suffix)
        torch.testing.assert_close(full[:, :4], prefix, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(full[:, 4:], first, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(changed[:, :4], prefix, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(changed[:, 4:], second, atol=3e-6, rtol=3e-5)
        self.assertFalse(torch.allclose(reset, second))
        self.assertEqual(state.tokens_seen, 4)

    def test_packet_decoder(self):
        self.check_mixer(PacketMixer)

    def test_packet_and_typed_probe_decoder(self):
        self.check_mixer(PacketAndTypedProbe)


if __name__ == '__main__':
    unittest.main()
