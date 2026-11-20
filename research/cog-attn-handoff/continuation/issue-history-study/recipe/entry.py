"""Execute the frozen candidate or control through TrainLab."""
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np

from evaluation import ROOT, digest, evaluate, replay
from models import run_models


def main():
    start = perf_counter()
    seed = int(os.environ["TRAINLAB_SEED"])
    out = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    arm = sys.argv[1]
    rows = json.loads((ROOT / "dataset.json").read_text())
    semantic = np.load(ROOT / "semantic.npy", allow_pickle=False)
    splits = ("train", "validation", "test", "external")
    ix = {split: np.asarray([i for i, r in enumerate(rows) if r["split"] == split and r["strict"]])
          for split in splits}
    assert len(semantic) == len(rows) and all(len(indices) for indices in ix.values())
    y = np.asarray([r["target"] for r in rows])
    assert all(len(np.unique(y[ix[split]])) == 2 for split in ("train", "validation"))
    pred, cues, found, fitted = run_models(rows, semantic, ix["train"], ix["validation"], arm, seed)
    scores, details = evaluate(rows, semantic, pred, cues, found, ix, arm)
    # Evaluate server-ordered timestamp ties separately using already trained models and gates.
    tied = {}
    from evaluation import eligible, measure_admissions, summarize
    for name, probability in pred.items():
        gate = scores[name]
        admitted = eligible(probability, cues, semantic, gate["threshold"], gate["corroboration"])
        tied[name] = {}
        for split in ("test", "external"):
            idx = np.asarray([i for i, r in enumerate(rows) if r["split"] == split and not r["strict"]],
                             dtype=int)
            result = summarize(y[idx], probability[idx])
            result.update(measure_admissions(
                y[idx], admitted[idx], np.asarray([rows[i]["issue"] for i in idx])))
            tied[name][split] = result
    systems = replay() if arm == "candidate" else None
    report = {"seed": seed, "arm": arm, "heads": scores, "tied_only": tied,
              "systems": systems, "fit_and_score_seconds": perf_counter() - start}
    name = "modular" if arm == "candidate" else "flat_all_features"
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out / "predictions.json").write_text(json.dumps(details, separators=(",", ":"), allow_nan=False))
    (out / "predictions.sha256").write_text(digest(details))
    (out / "metrics.json").write_text(json.dumps({"ap": scores[name]["cohorts"]["test"]["ap"]}))
    joblib.dump(fitted, out / "checkpoint.joblib", compress=3)


if __name__ == "__main__":
    main()
