import copy
import io

import torch
import pytest
from architecture import Config, LM, Readout, RevisionBank
from runtime import Runtime


@pytest.mark.parametrize('arm', ['local','mamba2'])
@torch.inference_mode()
def test_continuation_parity_and_replay(arm):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    m = LM(Config(arm)).eval()
    h = Readout(m.c.width, arm=='local').eval()
    s = Runtime(m,h)
    x = torch.randint(0, m.c.vocab, (2, 93))
    # Include varied literal successors to test the stateful mixture.
    x[:, 60:66] = x[:, 3:9]
    full, _ = s.advance(x)
    first, state = s.advance(x[:, :37])
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    restored = torch.load(buf, weights_only=False)
    # Internal trusted in-memory roundtrip of Python tables.
    tail, _ = s.advance(x[:, 37:], restored)
    torch.testing.assert_close(torch.cat([first,tail],1),full,rtol=4e-5,atol=4e-5)
    one, state = s.advance(x[:,:1])
    outs = [one]
    for i in range(1,x.size(1)):
        part, state = s.advance(x[:,i:i+1],state)
        outs.append(part)
    torch.testing.assert_close(torch.cat(outs,1), full,rtol=4e-5,atol=4e-5)
    if arm == 'mamba2':
        from architecture import features
        native, _ = m(x)
        hidden, _ = features(m, x[:, :64])
        logits = torch.log_softmax(hidden @ m.emb.weight.T, dim=-1)
        torch.testing.assert_close(logits, native[:,:64],rtol=4e-5,atol=4e-5)


def test_revision_competes_and_gets_gradients():
    torch.manual_seed(33)
    bank = RevisionBank(16, slots=3, dim=6, period=4)
    x = torch.randn(2, 12, 16, requires_grad=True)
    pred, state = bank(x)
    torch.testing.assert_close(pred[:,:4], torch.zeros_like(pred[:,:4]))
    assert state[0].shape == (2,3,6)
    assert state[1].abs().sum() > 0
    pred[:,-4:].square().sum().backward()
    for name in ('key','query','value','salience','strength','readout'):
        assert getattr(bank,name).weight.grad.abs().sum() > 0
    other = x.detach().clone()
    other[:,7:] *= -1
    changed, _ = bank(other)
    torch.testing.assert_close(pred[:,:7],changed[:,:7])
