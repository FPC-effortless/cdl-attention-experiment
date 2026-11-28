"""Frozen-weight mechanism interventions and long-stream diagnostics."""
import argparse
import json
from pathlib import Path

import torch

from interventions import assess_interventions
from longstate import longstate
from scale import build_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scale", choices=("1m", "10m"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = build_model(args.arm, args.seed, args.scale)
    model.load_state_dict(torch.load(args.out / "matched_data_weights.pt", weights_only=True))
    model.eval()
    record = {"long_stream": longstate(model)}
    if args.arm in ("D", "E", "F", "G", "H"):
        record["last_component_intervention"] = assess_interventions(model, worlds=64)
    (args.out / "postfit.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
