"""Prospective confirmation runner: close every fit before making fresh TEST."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from lrc_vb001_comparison.source import study as vb
from .bridge_data import dataset as bridge_dataset, validate as validate_bridge
from .train import ARMS, build, fit


HERE = Path(__file__).resolve().parent
SEEDS = (901, 902, 903)
SOURCE_NAMES = ("model.py", "index.py", "train.py", "run.py", "benchmark.py", "key_learning_probe.py",
                "bridge_data.py", "PROTOCOL.md")
TEST_SPECS = {"fresh_iid_a": (85002, False), "fresh_withheld_a": (85003, True),
              "fresh_iid_b": (85005, False), "fresh_withheld_b": (85006, True)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    hashes = {name: digest(HERE / name) for name in SOURCE_NAMES}
    original_root = Path(vb.__file__).resolve().parents[1]
    hashes.update({"original/" + str(path.relative_to(original_root)): digest(path)
                   for path in sorted(original_root.rglob("*.py"))})
    return hashes


def write_json(path, data):
    with Path(path).open("x") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    train = vb.dataset(71001, 2048)
    torch.save(train, output / "train.pt")
    manifest = dict(experiment="LRC-ROUTE-REPAIR-002", arms=ARMS, seeds=SEEDS,
                    source_sha256=source_hashes(),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
                    train_sha256=vb.digest_data(train), train_examples=len(train["answer"]),
                    test_specs=TEST_SPECS, bridge_seed=85004, test_groups=256,
                    steps=2000, batch=32, key_steps=1600, threads=1,
                    fit_execution="sequential isolated CPU fits, no concurrent fit workers",
                    python=platform.python_version(), torch=str(torch.__version__),
                    platform=platform.platform(), created_unix=time.time(),
                    parameters={arm: sum(p.numel() for p in build(arm).parameters()) for arm in ARMS})
    write_json(output / "prefit_manifest.json", manifest)
    return manifest


def check_frozen(output):
    manifest = json.loads((output / "prefit_manifest.json").read_text())
    if manifest["source_sha256"] != source_hashes():
        raise RuntimeError("frozen fitting/evaluation source changed")
    return manifest


def fit_all(output):
    manifest = check_frozen(output)
    if (output / "fits_closed.json").exists() or (output / "evaluation_started.json").exists():
        raise RuntimeError("fitting already closed")
    train = torch.load(output / "train.pt", weights_only=True)
    if vb.digest_data(train) != manifest["train_sha256"]:
        raise RuntimeError("training data changed")
    receipts = []
    # Sequential execution gives interpretable complete-fit wall times. Each
    # seed rotates arm order to reduce a fixed order effect; no best run chosen.
    for seed_index, seed in enumerate(SEEDS):
        order = ARMS[seed_index:] + ARMS[:seed_index]
        for arm in order:
            check_frozen(output)
            _, receipt = fit(arm, seed, train, output)
            receipts.append(receipt)
    for seed in SEEDS:
        orders = {r["minibatch_sha256"] for r in receipts if r["seed"] == seed}
        if len(orders) != 1:
            raise AssertionError("unmatched minibatch order")
    write_json(output / "fits_closed.json", dict(status="all_fits_closed", receipts=receipts,
               closed_unix=time.time(), source_sha256=source_hashes()))
    print(json.dumps({"status": "all_fits_closed", "count": len(receipts)}), flush=True)


def load_model(output, arm, seed):
    checkpoint = torch.load(output / f"{arm}_{seed}.pt", weights_only=True)
    model = build(arm).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"])
    return model


@torch.no_grad()
def infer(model, arm, data, *, mode="normal", batch_size=128):
    gathered = {}
    for start in range(0, len(data["answer"]), batch_size):
        b = vb.take(data, slice(start, start + batch_size))
        if arm.startswith("repair"):
            divisor = 2 if mode in ("paired_collision", "paired_collision_ordinal") else 1
            memory = model.prepare_memory(b["records"], code_divisor=divisor)
            o = model.query_memory(memory, b["query"],
                resolver="ordinal" if mode == "paired_collision_ordinal" else "content",
                probe_count=1 if mode == "one_probe" else None)
            logits, selected = o.logits, o.selected
            extra = {name: getattr(o, name) for name in ("writer_codes", "read_codes", "inspected", "returned",
                                                        "overflow", "bucket_size", "probes", "ties")}
        elif arm == "original_lrc":
            o = model(b["records"], b["query"])
            logits, selected = o.logits[:, -1], o.operands[0][:, -1]
            extra = {"writer_codes": o.decisions[0, "write_relation"][:, :16] * 8
                      + o.decisions[0, "write_address"][:, :16]}
        else:
            logits, reads = model(b["records"], b["query"], hard=mode == "hard")
            selected = reads.argmax(-1)
            extra = {}
        result = dict(pred=logits.argmax(-1), loss=F.cross_entropy(logits, b["answer"], reduction="none"),
                      selected=selected, **extra)
        for name, value in result.items():
            gathered.setdefault(name, []).append(value.cpu())
    return {name: torch.cat(values) for name, values in gathered.items()}


def summarize(p, data):
    correct = p["pred"].eq(data["answer"])
    hits = p["selected"].eq(data["support"])
    result = dict(n=len(correct), accuracy=float(correct.float().mean()), nll=float(p["loss"].mean()),
                  quartet_accuracy=float(correct.reshape(-1, 4).all(-1).float().mean()),
                  hop1=float(hits[:, 0].float().mean()), hop2=float(hits[:, 1].float().mean()),
                  both=float(hits.all(-1).float().mean()),
                  distinct_nonnull_hops=float((p["selected"][:, 0].ne(p["selected"][:, 1])
                                              & p["selected"].ge(0).all(-1)).float().mean()))
    if "inspected" in p:
        result.update(inspected_per_hop=float(p["inspected"].float().mean()),
                      maximum_inspected_per_hop=int(p["inspected"].max()),
                      overflow_probes=int(p["overflow"].sum()), ties=int(p["ties"].sum()),
                      null_read_fraction=float(p["selected"].lt(0).float().mean()),
                      occupied_codes_mean=sum(len(row.unique()) for row in p["writer_codes"]) / len(correct))
    return result


def evaluate(output):
    torch.set_num_threads(1)
    check_frozen(output)
    closed = json.loads((output / "fits_closed.json").read_text())
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed = [(receipt["arm"], receipt["seed"]) for receipt in closed["receipts"]]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("incomplete fitting barrier")
    for receipt in closed["receipts"]:
        if digest(output / f'{receipt["arm"]}_{receipt["seed"]}.pt') != receipt["checkpoint_sha256"]:
            raise RuntimeError("checkpoint changed")
    write_json(output / "evaluation_started.json", dict(started_unix=time.time(),
               fits_closed_sha256=digest(output / "fits_closed.json")))
    datasets = {name: vb.dataset(seed, 256, heldout) for name, (seed, heldout) in TEST_SPECS.items()}
    datasets["fresh_bridge"] = bridge_dataset(85004, 256)
    bridge_validation = validate_bridge(datasets["fresh_bridge"])
    torch.save(datasets, output / "evaluation_data.pt")
    results, predictions, permutations = [], {}, {}
    reversal = torch.cat((torch.arange(7, -1, -1), torch.arange(15, 7, -1)))
    for seed in SEEDS:
        for arm in ARMS:
            model = load_model(output, arm, seed)
            for name, data in datasets.items():
                modes = ("normal", "hard") if arm == "memnet" else ("normal",)
                for mode in modes:
                    p = infer(model, arm, data, mode=mode)
                    key = f"{arm}/{seed}/{name}/{mode}"
                    predictions[key] = p
                    result = dict(arm=arm, seed=seed, dataset=name, mode=mode, **summarize(p, data))
                    results.append(result)
                    print(json.dumps(result), flush=True)
            if arm == "repair":
                data = datasets["fresh_bridge"]
                for mode in ("paired_collision", "paired_collision_ordinal", "one_probe"):
                    p = infer(model, arm, data, mode=mode)
                    predictions[f"{arm}/{seed}/fresh_bridge/{mode}"] = p
                    results.append(dict(arm=arm, seed=seed, dataset="fresh_bridge", mode=mode, **summarize(p, data)))
                for mode in ("normal", "paired_collision"):
                    permuted = {**data, "records": data["records"][:, reversal],
                                "support": reversal[data["support"]]}
                    p = infer(model, arm, permuted, mode=mode)
                    permutations[f"{arm}/{seed}/{mode}"] = p
                    reference = predictions[f"{arm}/{seed}/fresh_bridge/{mode}"]
                    mapped = torch.where(p["selected"] >= 0, reversal[p["selected"].clamp_min(0)], -1)
                    results.append(dict(arm=arm, seed=seed, dataset="fresh_bridge_reversed", mode=mode,
                        **summarize(p, permuted), identical_answers=bool(torch.equal(p["pred"], reference["pred"])),
                        identical_mapped_supports=bool(torch.equal(mapped, reference["selected"]))))
    torch.save(predictions, output / "predictions.pt")
    torch.save(permutations, output / "permutation_predictions.pt")
    # The archived quality screen is retained on two fresh IID/withheld pairs;
    # causal/support screens are additional requirements for this repair.
    primary = [r for r in results if r["arm"] == "repair" and r["mode"] == "normal"
               and r["dataset"] in datasets]
    gates = dict(answer_and_quartet=all(r["accuracy"] >= .95 and r["quartet_accuracy"] >= .90 for r in primary),
                 both_supports=all(r["both"] >= .95 for r in primary),
                 no_overflow=all(r["overflow_probes"] == 0 for r in primary),
                 bounded_candidates=all(r["maximum_inspected_per_hop"] <= 4 for r in primary),
                 permutation=all(r["identical_answers"] and r["identical_mapped_supports"]
                                 for r in results if r["dataset"] == "fresh_bridge_reversed"))
    repair_withheld = [r["accuracy"] for r in results if r["arm"] == "repair" and r["mode"] == "normal"
                       and r["dataset"].startswith("fresh_withheld")]
    shuffled_withheld = [r["accuracy"] for r in results if r["arm"] == "repair_shuffled" and r["mode"] == "normal"
                         and r["dataset"].startswith("fresh_withheld")]
    label_gain = sum(repair_withheld)/len(repair_withheld) - sum(shuffled_withheld)/len(shuffled_withheld)
    gates["label_specific_benefit"] = label_gain >= .10
    report = dict(results=results, gates=gates, all_gates=all(gates.values()),
                  repair_minus_shuffled_withheld=label_gain,
                  bridge_validation=bridge_validation,
                  data_hashes={name: vb.digest_data(data) for name, data in datasets.items()})
    write_json(output / "results.json", report)
    write_json(output / "complete.json", dict(status="complete", completed_unix=time.time(),
               files={name: digest(output/name) for name in ("results.json", "predictions.pt", "permutation_predictions.pt", "evaluation_data.pt")},
               source_sha256=source_hashes(), gates=gates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "fit", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"prepare": prepare, "fit": fit_all, "evaluate": evaluate}[args.operation](args.output)


if __name__ == "__main__":
    main()
