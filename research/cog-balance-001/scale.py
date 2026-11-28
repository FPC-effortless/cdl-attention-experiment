"""Build matched 1.08M and 10M BALANCE models.

The 10M design changes the number of mixing layers from two to 22 at fixed
embedding dimension. It tests a depth-scaling recipe. Its result combines
depth, parameter count, optimization and compute effects. Embedding, local
front end, final norm and tied decoder are shared across depth. The first
layer pair uses the 1M arm's seed and each additional pair is initialized by
the same factory with its own seed. All arms keep their original head/state
sizes. No layer weights are shared between pairs.
"""
import torch
from torch import nn

from architecture import build_cog
from controls import build_control
from model import D, count


def build_pair(arm, seed):
    if arm in tuple("ABCDEFGH"):
        return build_cog(arm, seed)
    return build_control(arm, seed)


def build_model(arm, seed, scale="1m"):
    if scale not in ("1m", "10m"):
        raise ValueError(f"Unknown scale {scale!r}")
    model = build_pair(arm, seed)
    if scale == "1m":
        return model
    parent = model.parent
    kinds = parent.kinds
    for pair in range(1, 11):
        donor = build_pair(arm, seed + pair * 104729).parent
        parent.core.blocks.extend(donor.core.blocks)
        parent.kinds += kinds
        if parent.with_side:
            parent.sides.extend(donor.sides)
        if "H" in kinds:
            parent.payload_map.extend(donor.payload_map)
            parent.payload_probe.extend(donor.payload_probe)

    # One FF unit consumes D + 1 input and D output parameters. Distribute
    # the nearest integer number of units across all mixing layers.
    target = 10_000_000
    blocks = parent.core.blocks
    trim = round((count(model) - target) / (2 * D + 1))
    if trim < 0:
        raise ValueError("The 22-layer starting point must exceed the target")
    units, remainder = divmod(trim, len(blocks))
    widths = []
    for index, block in enumerate(blocks):
        old = block.ff
        width = old[0].out_features - units - int(index < remainder)
        if width <= 0:
            raise ValueError("Invalid residual FF width")
        ff = nn.Sequential(nn.Linear(D, width), nn.GELU(), nn.Linear(width, D))
        with torch.no_grad():
            ff[0].weight.copy_(old[0].weight[:width])
            ff[0].bias.copy_(old[0].bias[:width])
            ff[2].weight.copy_(old[2].weight[:, :width])
            ff[2].bias.copy_(old[2].bias)
        block.ff = ff
        widths.append(width)
    parent.ff = tuple(widths)
    if abs(count(model) - target) > 129 or len(blocks) != 22:
        raise RuntimeError(f"10M target mismatch, {len(blocks)} layers, {count(model)} weights")
    return model
