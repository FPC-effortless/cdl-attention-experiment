"""Causal and numerical preflight for a BALANCE training arm."""
import argparse
import copy
import json
import math

import torch
from torch.nn import functional as F

from scale import build_model


ARMS = tuple("ABCDEFGH") + ("GQA", "GRU", "MAMBA2", "GATED_DELTA")
# Parallel reductions and one-row GEMMs in fp32 sum terms in a different
# order. The reference mixer tests use float64 and tighter tolerances.
ATOL = 5e-5
RTOL = 1e-5


def compare_state(a, b):
    if a is None:
        assert b is None
        return
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL)
        return
    assert type(a) is type(b) and len(a) == len(b)
    for left, right in zip(a, b):
        compare_state(left, right)


def direct_mixer(mixer, arm):
    """Validate the batched solver against a causal single-step implementation.

    The convolution uses an explicit dot product across its four taps, and
    both latent recurrences update and read one time step at a time.
    """
    from model import D

    # Clone to avoid changing the dtype or weights of the language model.
    m = copy.deepcopy(mixer).double().eval()
    x = torch.randn(2, 35, D, dtype=torch.float64)
    if arm == "MAMBA2":
        key_dim, val_dim = m.state_dim, m.head_dim
    else:
        key_dim, val_dim = m.key_dim, m.value_dim
    memory = torch.randn(len(x), m.heads, key_dim, val_dim, dtype=x.dtype)
    hist = torch.randn(len(x), m.conv.in_channels, 3, dtype=x.dtype)
    output, (expected_state, expected_hist) = m(x, (memory.clone(), hist.clone()))
    values = []
    for token in x.unbind(1):
        if arm == "MAMBA2":
            inner = m.heads * val_dim
            gate, raw, dt = m.in_proj(token).split((inner, inner + 2 * key_dim, m.heads), -1)
        else:
            raw = m.qkv(token)
        taps = torch.cat((hist, raw.unsqueeze(-1)), dim=-1)
        conv = (taps * m.conv.weight[:, 0]).sum(-1)
        if m.conv.bias is not None:
            conv = conv + m.conv.bias
        features = F.silu(conv)
        hist = taps[..., 1:]
        if arm == "MAMBA2":
            v, b, c = features.split((inner, key_dim, key_dim), -1)
            v = v.reshape(len(x), m.heads, val_dim)
            dt = F.softplus(dt + m.dt_bias)
            keep = (-m.log_a.exp() * dt).exp()
            memory = keep[..., None, None] * memory + (
                dt[..., None, None] * b[:, None, :, None] * v[:, :, None, :]
            )
            read = torch.einsum("bk,bhkv->bhv", c, memory) + m.skip[None, :, None] * v
            y = m.out(m.norm(read.flatten(1) * F.silu(gate)))
        else:
            q, k, v = features.split((m.heads * key_dim, m.heads * key_dim, m.heads * val_dim), -1)
            q = F.normalize(q.reshape(len(x), m.heads, key_dim), dim=-1) / math.sqrt(key_dim)
            k = F.normalize(k.reshape(len(x), m.heads, key_dim), dim=-1)
            v = v.reshape(len(x), m.heads, val_dim)
            dt, beta = m.rates(token).chunk(2, -1)
            decay = (-m.log_a.exp() * F.softplus(dt + m.dt_bias)).exp()
            prior = memory * decay[..., None, None]
            error = v - torch.einsum("bhk,bhkv->bhv", k, prior)
            memory = prior + beta.sigmoid()[..., None, None] * k.unsqueeze(-1) * error.unsqueeze(-2)
            read = torch.einsum("bhk,bhkv->bhv", q, memory)
            y = m.out(m.norm(read).flatten(1) * F.silu(m.gate(token)))
        values.append(y)
    direct_output = torch.stack(values, dim=1)
    for a, b in ((direct_output, output), (memory, expected_state), (hist, expected_hist)):
        torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-9)
    return float((direct_output - output).abs().max())


@torch.no_grad()
def verify_eval(model, x):
    model.eval()
    output, final = model(x, carry=True)
    for width in (1, 7):
        state, chunks = None, []
        for part in x.split(width, 1):
            y, state = model(part, state=state, carry=True)
            chunks.append(y)
        reconstructed = torch.cat(chunks, dim=1)
        torch.testing.assert_close(output, reconstructed, atol=ATOL, rtol=RTOL)
        compare_state(final, state)
        if width == 1:
            error = float((output - reconstructed).abs().max())
    cut = 17
    changed = x.clone()
    changed[:, cut:] = (changed[:, cut:] + 1) % 288
    torch.testing.assert_close(model(changed)[:, :cut], output[:, :cut], atol=ATOL, rtol=RTOL)
    # Reset explicitly after other windows have run on the same instance.
    reset, reset_state = model(x, state=None, carry=True)
    torch.testing.assert_close(reset, output, atol=0.0, rtol=0.0)
    compare_state(reset_state, final)
    return error


@torch.no_grad()
def force_dispatch(side, arm):
    """Exercise routes even if an untrained or collapsed router never selects them."""
    side = copy.deepcopy(side).eval()
    states = [torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8, 8)]
    states.append(torch.randn(2, 8) if arm == "H" else None)
    x = torch.randn(2, 9, side.context.in_features)
    destinations = range(side.router.out_features)
    errors = []
    for route in destinations:
        for write in (False, True):
            side.router.weight.zero_()
            side.router.bias.fill_(-4)
            side.router.bias[route] = 4
            side.event.weight.zero_()
            side.event.bias.fill_(4 if write else -4)
            if arm == "H":
                side.axis_choice.weight.zero_()
                side.axis_choice.bias.zero_()
                side.axis_choice.bias[0] = 4
            all_y, carry = side(x, tuple(states))
            single = tuple(states)
            ys = []
            for token in x.split(1, dim=1):
                y, single = side(token, single)
                ys.append(y)
            decoded = torch.cat(ys, dim=1)
            torch.testing.assert_close(all_y, decoded, atol=ATOL, rtol=RTOL)
            compare_state(carry, single)
            errors.append((all_y - decoded).abs().max().item())
            for index in (1, 2):
                if route == index and write:
                    assert not torch.equal(states[index], carry[index])
                else:
                    assert torch.equal(states[index], carry[index])
            if route == 0:
                assert torch.count_nonzero(all_y) == 0
            if arm == "H" and route == 3:
                assert not torch.equal(states[3], carry[3])
                torch.testing.assert_close(states[3].norm(dim=-1), carry[3].norm(dim=-1),
                                           atol=ATOL, rtol=RTOL)
            elif arm == "H":
                assert torch.equal(states[3], carry[3])
    return {"forced_configurations": len(errors), "forced_side_max_error": max(errors)}


def check(arm, scale):
    torch.set_num_threads(1)
    model = build_model(arm, 9701, scale)
    generator = torch.Generator().manual_seed(329243)
    x = torch.randint(288, (2, 37), generator=generator)
    targets = torch.randint(288, x.shape, generator=generator)
    first = verify_eval(model, x)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0005)
    logits = model(x)
    assert logits.shape == (*x.shape, 288)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    assert norm > 0
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with torch.no_grad():
        after = F.cross_entropy(model(x).flatten(0, 1), targets.flatten())
        assert torch.isfinite(after)
    last = verify_eval(model, x)
    row = dict(arm=arm, scale=scale, stream_max_error_before=first, stream_max_error_after=last,
               initial_loss=loss.item(), updated_loss=after.item(), initial_grad_norm=norm.item())
    with torch.no_grad():
        if arm in ("GATED_DELTA", "MAMBA2"):
            row["direct_mixer_max_error"] = direct_mixer(model.parent.core.blocks[0].seq, arm)
        if arm in ("F", "G", "H"):
            row.update(force_dispatch(model.parent.sides[0], arm))
        if hasattr(model, "measure_routes"):
            model.measure_routes(True)
            model(x)
            stats = [{k: int(v) for k, v in layer.items()} for layer in model.routing_metrics()]
            for layer in stats:
                assert layer["positions"] == x.numel()
                assert 0 <= layer["memory_writes"] <= layer["memory_selected"] <= x.numel()
                assert 0 <= layer["slow_updates"] <= layer["slow_selected"] <= x.numel()
                if arm in "FGH":
                    assert sum(layer[k] for k in ("slow_selected", "memory_selected", "geometric_selected")) <= x.numel()
            row["routes"] = stats
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--scale", choices=("1m", "10m"), default="1m")
    args = parser.parse_args()
    print(json.dumps(check(args.arm, args.scale), allow_nan=False))
