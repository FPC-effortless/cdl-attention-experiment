"""Replay BALANCE-002 from a verified 96-update parent and collect the artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "cog-balance-001"
SEEDS = (9731, 9743, 9769)
ROSTER = {
    "C": [(arm, regime, data)
          for arm, data, regimes in (
              ("C", "original", ("final", "trace_pair")),
              ("C", "balanced", ("final", "trace", "pair", "trace_pair")),
              ("R", "balanced", ("final", "trace_pair")),
          )
          for regime in regimes],
    "H": [("H", "trace_pair", "balanced")],
    "GQA": [("GQA", regime, "balanced") for regime in ("final", "trace_pair")],
    "GRU": [("GRU", regime, "balanced") for regime in ("final", "trace_pair")],
}
PARENT_TRAIN = {
    9731: "bcc9bd256c92e551ce14a1650e1c12fe776bd2be829401c3fea517c6a20baf0d",
    9743: "1a5bca24027ce9787dda9a4e93f77081c7480785cb58e5b8525e12e7c1718f3f",
    9769: "dade27e013ab4ce889c3a6d4350bd85bb9a775910b9646f925855571f7f740fe",
}
PARENT_DEV = "6f27261fa931b545056f82030d9f4b1c7383d4206852d8f163e691a234dadb8d"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit(parent_arm, seed, corpus, out):
    corpus, out = corpus.resolve(), out.resolve()
    parent = out / "parent"
    subprocess.run([
        sys.executable, str(PARENT / "run_balance.py"), "--arm", parent_arm,
        "--seed", str(seed), "--scale", "1m", "--corpus", str(corpus),
        "--out", str(parent), "--steps", "96", "--cpu-seconds", "8",
        "--threads", "1", "--batch", "4", "--price", "0.01", "--worlds", "64",
        "--dev-batches", "16",
    ], check=True)
    record = json.loads((parent / "metrics.json").read_text())["readouts"]["matched_data"]
    if (record["step"], record["training_sha256"], record["dev_sha256"]) != (
        96, PARENT_TRAIN[seed], PARENT_DEV
    ):
        raise ValueError(f"Parent data differed from the paired pilot for seed {seed}")
    weights = parent / "matched_data_weights.pt"
    receipt = {
        "parent_arm": parent_arm, "seed": seed, "commit": os.environ.get("SOURCE_SHA"),
        "parent_weights_sha256": sha(weights), "parent_training_sha256": PARENT_TRAIN[seed],
        "parent_dev_sha256": PARENT_DEV, "fits": {},
    }
    for arm, regime, data in ROSTER[parent_arm]:
        name = f"{arm}-{data}-{regime}-{seed}"
        dest = out / name
        subprocess.run([
            sys.executable, str(HERE / "run_repair.py"), "--arm", arm, "--regime", regime,
            "--seed", str(seed), "--data", data, "--corpus", str(corpus), "--parent",
            str(weights), "--out", str(dest),
        ], check=True)
        receipt["fits"][name] = sha(dest / "metrics.json")
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def collect(root, out):
    expected = {f"{arm}-{data}-{regime}-{seed}" for seed in SEEDS
                for configs in ROSTER.values() for arm, regime, data in configs}
    out.mkdir(parents=True, exist_ok=True)
    receipts, fits = [], {}
    training_digests = {}
    panel_digests = {}
    for path in sorted(root.glob("*/receipt.json")):
        receipt = json.loads(path.read_text())
        receipts.append(receipt)
        for name, fingerprint in receipt["fits"].items():
            if name in fits or name not in expected:
                raise ValueError(f"Duplicate or unexpected fit {name}")
            source = path.parent / name / "metrics.json"
            if sha(source) != fingerprint:
                raise ValueError(f"Fingerprint mismatch for {name}")
            row = json.loads(source.read_text())
            identity = f"{row['arm']}-{row['data']}-{row['regime']}-{row['seed']}"
            if identity != name or row["parent_sha256"] != receipt["parent_weights_sha256"]:
                raise ValueError(f"Run identity or parent hash mismatch for {name}")
            if sha(source.with_name("weights.pt")) != row["trained_weights_sha256"]:
                raise ValueError(f"Trained weight hash mismatch for {name}")
            if (row["steps"], row["question_streams"], row["real_text_replay_targets"]) != (256, 2048, 32768):
                raise ValueError(f"Training budget mismatch for {name}")
            key = (row["data"], row["seed"])
            if training_digests.setdefault(key, row["training_sha256"]) != row["training_sha256"]:
                raise ValueError(f"Training examples disagree within a matched condition for {name}")
            for panel, measures in row["assessment"]["panels"].items():
                sig = (measures["worlds"], measures["sha256"])
                if panel_digests.setdefault(panel, sig) != sig:
                    raise ValueError(f"Evaluation worlds disagree for {name} panel {panel}")
            fits[name] = row
    missing = sorted(expected - fits.keys())
    (out / "replay.json").write_text(json.dumps(
        {"expected": len(expected), "complete": len(fits), "missing": missing,
         "parents": receipts, "fits": fits}, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(fits)}/{len(expected)} repair fits.")
    if missing:
        print("Missing cells: " + ", ".join(missing))
    return not missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    train = commands.add_parser("fit")
    train.add_argument("--parent-arm", required=True, choices=ROSTER)
    train.add_argument("--seed", required=True, type=int, choices=SEEDS)
    train.add_argument("--corpus", required=True, type=Path)
    train.add_argument("--out", required=True, type=Path)
    report = commands.add_parser("collect")
    report.add_argument("--input", required=True, type=Path)
    report.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "fit":
        fit(args.parent_arm, args.seed, args.corpus, args.out)
    else:
        sys.exit(0 if collect(args.input, args.out) else 1)
