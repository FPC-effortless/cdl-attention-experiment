"""Re-profile saved matched-data models in randomized sequential order."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from corpus import load_corpus
from profiler import profile_model
from scale import build_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    files = sorted(args.input.glob("*/metrics.json"))
    np.random.default_rng(90107).shuffle(files)
    tokens = torch.from_numpy(load_corpus(args.corpus).batch("dev", np.random.default_rng(6198), 1, 128))
    rows = []
    for file in files:
        run = json.loads(file.read_text())
        assert "matched_data" in run["readouts"]
        model = build_model(run["arm"], run["seed"], run["scale"])
        model.load_state_dict(torch.load(file.with_name("matched_data_weights.pt"), weights_only=True))
        rows.append(dict(arm=run["arm"],seed=run["seed"],scale=run["scale"],
                         profile=profile_model(model,tokens)))
    args.out.write_text(json.dumps(rows,separators=(",",":"),allow_nan=False)+"\n")
