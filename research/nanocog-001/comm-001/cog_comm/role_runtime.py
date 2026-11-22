"""Zero-update decoder boundary check; not a fitted-model result."""
import json
import os
from pathlib import Path
import sys

import torch

from .model import build_decoder
from .objective import Tokenizer
from .role_control import CELLS, COLORS, families, world

def require(test, why):
    if not test:
        raise ValueError(why)

def main():
    require(sys.version_info[:2] == (3, 12) and torch.__version__ == "2.10.0+cpu",
            "unexpected runtime")
    seed = int(os.environ["TRAINLAB_SEED"])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    fit = families()["fit"]
    text = [" ".join(world(key, c)) for key in fit for c in CELLS]
    tok = Tokenizer.from_train(text, COLORS)
    x = torch.tensor([tok.encode(p)[1:-1] for p in text[:8]])
    require(tuple(x.shape) == (8, 7) and x.min() >= 262, "word-ID geometry changed")
    for c in (1,2,4):
        dims = int((x[0] != x[c]).sum())
        require(dims in {1: (1,), 2: (4,), 4: (2,)}[c], "encoded factor changed")
    model = build_decoder(tok.size, "A", width=32, layers=2, seed=seed).eval()
    require(model.smear_gain.item() == 0.0, "expected zero initial smear")
    emb = []
    hook = model.embedding.register_forward_hook(lambda m, args, y: emb.append(y))
    z, _ = model(x, state=None)
    hook.remove()
    require(tuple(z.shape) == (8, 7, tok.size), "full vocabulary geometry")
    # Probe a logit independently of task truth, at the prediction token only.
    dz, = torch.autograd.grad(z[:, -1, tok.index[COLORS[0]]].sum(), emb[0])
    norms = dz.norm(dim=-1)
    require(torch.isfinite(norms).all() and (norms > 0).all(),
            "at least one position did not reach the readout")
    with torch.inference_mode():
        prefix = torch.zeros((8,1), dtype=x.dtype)
        earlier, _ = model(torch.cat((prefix, x), dim=1))
        prefix.fill_(tok.index[COLORS[-1]])
        latter, _ = model(torch.cat((prefix, x), dim=1))
        require(torch.equal(earlier[:, -1], latter[:, -1]),
                "the initialized local decoder responded beyond seven tokens")
    out = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    metrics = dict(positions_with_gradient=int((norms > 0).sum().item()),
                   optimizer_updates=0, vocab_size=tok.size,
                   parameters=sum(p.numel() for p in model.parameters()))
    (out / "metrics.json").write_text(json.dumps(metrics, sort_keys=True)+"\n")
    (out / "geometry.json").write_text(json.dumps({
        **metrics, "prefix_collision_pairs":8, "min_gradient_by_position":norms.min(0).values.tolist(),
        "seed":seed, "scope":"initialized model only, one FIT family, no optimizer",
    },sort_keys=True)+"\n")
if __name__ == "__main__":
    main()
