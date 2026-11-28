"""Probe fp32 state and distant-logit sensitivity of a trained checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from profiler import _storage_sizes


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _tensors(value)


@torch.inference_mode()
def longstate(model, seed=81917, chunk=64):
    """Stream 4096 bytes and compare two worlds differing only at byte zero.

    Storage is the total tensor backing of the two carried rows on this CPU.
    Probe activation magnitudes and sensitivity convey no semantic score.
    """
    if chunk < 1 or 64 % chunk:
        raise ValueError("chunk must be a positive divisor of 64")
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    data = rng.integers(32, 127, size=4096, dtype=np.uint8)
    stream = torch.from_numpy(data.astype("int64")).to(device).repeat(2, 1)
    stream[1, 0] = 32 + (stream[1, 0] - 32 + 1) % 95
    previous_training = model.training
    model.eval()
    try:
        # A distinct, irregular microbatch schedule tests framing at boundaries.
        prefix = stream[:, :128]
        prefill = model(prefix)
        pieces, carry = [], None
        for x in prefix.split(13, dim=1):
            y, carry = model(x, state=carry, carry=True)
            pieces.append(y)
        framed = torch.cat(pieces, dim=1)
        error = (prefill - framed).abs()
        short_ok = torch.allclose(prefill, framed, atol=2e-3, rtol=1e-4)
        if not short_ok:
            raise AssertionError(f"128-byte prefill vs 13-byte chunks: max error {error.max().item()}")

        state = None
        sections = {}
        for start in range(0, stream.shape[1], chunk):
            output, state = model(stream[:, start:start + chunk], state=state, carry=True)
            tensors = [t for t in _tensors(state) if t.is_floating_point() and t.numel()]
            if not torch.isfinite(output).all() or any(not t.isfinite().all() for t in tensors):
                raise ValueError(f"nonfinite state or logits at chunk starting {start}")
            if start == 0:
                immediate = float((output[0, 0] - output[1, 0]).abs().max())
            steps = start + chunk
            if steps in (64, 512, 2048, 4096):
                difference = output[0, -1] - output[1, -1]
                sections[steps] = {
                    **_storage_sizes(state),
                    "distance_from_revised_byte": steps - 1,
                    "state_max_abs": max((t.abs().max().item() for t in tensors), default=0.0),
                    "final_max_abs_logit_difference": float(difference.abs().max()),
                    "final_rms_logit_difference": float(difference.square().mean().sqrt()),
                    "final_argmax_changed": bool(output[0, -1].argmax() != output[1, -1].argmax()),
                }
        return {
            "stream_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
            "paired_stream_batch": 2, "chunk": chunk, "prefix_bytes": 128,
            "prefix_max_abs_error": float(error.max()), "prefix_rms_error": float(error.square().mean().sqrt()),
            "revision_byte_index": 0, "immediate_max_abs_logit_difference": immediate,
            "state_and_logits_finite_all_chunks": True, "records": sections,
            "backing_growth_2048_to_4096_bytes":
                sections[4096]["backing_tensor_bytes"] - sections[2048]["backing_tensor_bytes"],
        }
    finally:
        model.train(previous_training)


if __name__ == "__main__":
    from scale import build_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--scale", default="1m", choices=("1m", "10m"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = build_model(args.arm, args.seed, args.scale)
    with args.checkpoint.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        handle.seek(0)
        model.load_state_dict(torch.load(handle, weights_only=True, map_location="cpu"), strict=True)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "scale": args.scale,
                      "checkpoint_sha256": digest, **longstate(model)}, allow_nan=False))
