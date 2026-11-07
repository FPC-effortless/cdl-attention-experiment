"""Post-fit integrity and exact evaluation replay; not independent replication."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from relational_composition.information import example, measure
from relational_composition.model import Config, RelationalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((args.directory / "manifest.json").read_text())
    assert manifest["phase"] == "complete"
    assert len(manifest["results"]) == 18
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((root / "relational_composition" / name).read_bytes()).hexdigest() == digest
    protocol = root / "relational_composition/EXPERIMENT_LRC002.md"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == manifest["protocol_sha256"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cells = examples_checked = 0
    for result in manifest["results"]:
        path = args.directory / result["checkpoint"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == result["checkpoint_sha256"]
        saved = torch.load(path, weights_only=True)
        assert len(saved["trace"]) == saved["steps"] == 400
        assert saved["seed"] == result["seed"] and saved["arm"] == result["arm"]
        assert saved["source_sha256"] == manifest["source_sha256"]
        assert all(torch.isfinite(value).all() for value in saved["model"].values())
        assert all(math.isfinite(t["nll"]) and math.isfinite(t["gradient_norm"]) for t in saved["trace"])
        assert sum(t["forward_passes"] for t in saved["trace"]) == result["forward_passes"] == 800
        assert result["train_cost"] == {k: sum(t["cost"][k] for t in saved["trace"])
                                        for k in result["train_cost"]}
        model = RelationalLM(Config(**saved["config"]))
        model.load_state_dict(saved["model"])
        for condition, expected in result["held_out"].items():
            examples = [example(90_000_000 + i, condition=condition,
                        random_answer=result["arm"] == "oracle_random_answer") for i in range(64)]
            actual = measure(model, examples, result["arm"])
            assert actual == expected, (result["seed"], result["arm"], condition)
            assert expected["generated_accuracy"] == sum(r["generated_correct"] for r in expected["records"]) / 64
            cells += 1
            examples_checked += 64
    print(json.dumps(dict(status="PASS", checkpoints=18, replayed_cells=cells,
                          replayed_examples=examples_checked, training_receipts=7200)))


if __name__ == "__main__":
    main()
