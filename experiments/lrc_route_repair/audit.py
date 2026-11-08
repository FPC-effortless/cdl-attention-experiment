"""Independent postfit audit of the sentence LRC route-repair experiment.

The semantic parser in this file is an auditor, never a model input or inference
implementation. No training or evaluation-set generation belongs in this file.
The final audit entrypoint must enforce fit and scoring closure before opening
any fresh evaluation tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch


NAMES = ("Alice", "Boris", "Cora", "Dion")
PLACES = ("Rome", "Lima", "Oslo", "Accra")
ERAS = ("first", "second")
WORDS = ("<pad>", "During", "the", "era", "partnered", "with", "lived", "in", ".",
         "where", "did", "'s", "partner", "live", "?") + NAMES + PLACES + ERAS
VOCAB = {word: i for i, word in enumerate(WORDS)}
DATA_KEYS = {"records", "query", "answer", "support", "era", "entity", "person"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_digest(data: dict[str, torch.Tensor]) -> str:
    """Reproduce the frozen VB tensor digest without calling its implementation."""
    digest = hashlib.sha256()
    for key in sorted(data):
        digest.update(key.encode())
        digest.update(data[key].cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def input_signatures(data: dict[str, torch.Tensor]) -> set[bytes]:
    return {torch.cat((record.flatten(), query)).cpu().contiguous().numpy().tobytes()
            for record, query in zip(data["records"], data["query"])}


def _encode(words: list[str]) -> list[int]:
    result = [VOCAB[word] for word in words]
    if len(result) > 12:
        raise AssertionError("Auditor attempted an oversized sentence.")
    return result + [0] * (12 - len(result))


def parse_record(tokens: torch.Tensor) -> tuple[int, int, int, int]:
    """Validate the entire template, including padding, then return its facts."""
    values = tokens.tolist()
    assert len(values) == 12 and all(0 <= v < len(WORDS) for v in values)
    era = tuple(VOCAB[word] for word in ERAS).index(values[2])
    person = tuple(VOCAB[word] for word in NAMES).index(values[4])
    if values[5] == VOCAB["partnered"]:
        relation, labels, phrase = 0, NAMES, ["partnered", "with"]
    elif values[5] == VOCAB["lived"]:
        relation, labels, phrase = 1, PLACES, ["lived", "in"]
    else:
        raise AssertionError("Unknown record relation.")
    value = tuple(VOCAB[word] for word in labels).index(values[7])
    expected = _encode(["During", "the", ERAS[era], "era", NAMES[person]]
                       + phrase + [labels[value], "."])
    assert values == expected, "Malformed record template or padding."
    return era, person, relation, value


def validate_task(data: dict[str, torch.Tensor], *, kind: str) -> dict:
    """Check fresh IID/held-out/bridge labels and edits independently.

    kind is one of iid, heldout, bridge. This validator is deliberately stronger
    than recomputing accuracy: malformed labels, duplicate facts and accidental
    counterfactual changes are rejected.
    """
    assert kind in {"iid", "heldout", "bridge"}
    assert set(data) == DATA_KEYS
    assert all(isinstance(value, torch.Tensor) and value.dtype == torch.long
               for value in data.values())
    n = len(data["answer"])
    assert n > 0 and n % 4 == 0
    shapes = {"records": (n, 16, 12), "query": (n, 12), "support": (n, 2),
              **{key: (n,) for key in ("answer", "era", "entity", "person")}}
    assert all(tuple(data[key].shape) == shape for key, shape in shapes.items())
    expected_keys = {(e, person, relation) for e in range(2)
                     for person in range(4) for relation in range(2)}
    for row in range(n):
        era, entity = int(data["era"][row]), int(data["entity"][row])
        assert 0 <= era < 2 and 0 <= entity < 4
        question = _encode(["During", "the", ERAS[era], "era", "where", "did",
                            NAMES[entity], "'s", "partner", "live", "?"])
        assert data["query"][row].tolist() == question
        facts = [parse_record(record) for record in data["records"][row]]
        assert {fact[:3] for fact in facts} == expected_keys
        assert len({fact[:3] for fact in facts}) == 16
        assert all(fact[0] == pos // 8 for pos, fact in enumerate(facts))
        lookup = {fact[:3]: (pos, fact[3]) for pos, fact in enumerate(facts)}
        first, person = lookup[era, entity, 0]
        second, answer = lookup[era, person, 1]
        assert first != second
        assert data["support"][row].tolist() == [first, second]
        assert int(data["person"][row]) == person
        assert int(data["answer"][row]) == answer
        if kind == "heldout":
            assert person == (entity + 1) % 4
        elif kind == "iid":
            assert person != (entity + 1) % 4
        else:
            assert len({lookup[era, p, 1][1] for p in range(4)}) == 4
    changed_tokens = 0
    for start in range(0, n, 4):
        rows = slice(start, start + 4)
        records, support = data["records"][rows], data["support"][rows]
        assert bool(data["query"][rows].eq(data["query"][start]).all())
        assert bool(data["era"][rows].eq(data["era"][start]).all())
        assert bool(data["entity"][rows].eq(data["entity"][start]).all())
        assert sorted(data["answer"][rows].tolist()) == [0, 1, 2, 3]
        assert bool(support[:, 0].eq(support[0, 0]).all())
        if kind == "bridge":
            assert sorted(data["person"][rows].tolist()) == [0, 1, 2, 3]
            assert len(support[:, 1].unique()) == 4
            changed_record = int(support[0, 0])
        else:
            assert bool(support.eq(support[0]).all())
            assert bool(data["person"][rows].eq(data["person"][start]).all())
            changed_record = int(support[0, 1])
        changed = records.ne(records[0])
        allowed = torch.zeros_like(changed)
        allowed[1:, changed_record, 7] = True
        assert torch.equal(changed, allowed), "Counterfactual modifies undeclared tokens."
        changed_tokens += int(changed.sum())
    return {"examples": n, "quartets": n // 4, "kind": kind,
            "exact_templates": True, "unique_distinct_supports": True,
            "answers_recomputed": True, "counterfactual_edits_exact": True,
            "changed_tokens_vs_first_variant": changed_tokens,
            "data_sha256": data_digest(data)}


def prediction_metrics(prediction: dict, data: dict) -> dict:
    """Independently recompute scores from saved per-example tensors."""
    pred, loss = prediction["pred"], prediction["loss"]
    assert pred.shape == loss.shape == data["answer"].shape
    assert pred.dtype == torch.long
    assert bool(pred.ge(0).all()) and bool(pred.lt(4).all())
    assert bool(torch.isfinite(loss).all()) and bool(loss.ge(0).all())
    correct = pred.eq(data["answer"])
    metrics = {"n": len(pred), "accuracy": float(correct.float().mean()),
               "nll": float(loss.mean()),
               "quartet_accuracy": float(correct.reshape(-1, 4).all(-1).float().mean())}
    selected = prediction.get("selected")
    if selected is not None:
        assert selected.shape == data["support"].shape
        assert selected.dtype == torch.long
        assert bool(selected.ge(-1).all()) and bool(selected.lt(data["records"].shape[1]).all())
        matched = selected.eq(data["support"])
        metrics.update(hop1=float(matched[:, 0].float().mean()),
                       hop2=float(matched[:, 1].float().mean()),
                       both=float(matched.all(-1).float().mean()),
                       distinct_nonnull_hops=float((selected[:, 0].ne(selected[:, 1])
                           & selected.ge(0).all(-1)).float().mean()))
    return metrics


def key_metrics(codes: torch.Tensor, data: dict, *, kind: str) -> dict:
    """Semantic key diagnostics are computed only by the independent auditor."""
    assert codes.shape == data["records"].shape[:2]
    assert codes.dtype == torch.long
    counts = {"partner": [], "location": []}
    person_pair_collisions = {"partner": 0, "location": 0}
    for row in range(len(codes)):
        facts = [parse_record(record) for record in data["records"][row]]
        for era in range(2):
            for relation, label in enumerate(("partner", "location")):
                positions = [i for i, fact in enumerate(facts)
                             if fact[0] == era and fact[2] == relation]
                assert len(positions) == 4
                keys = codes[row, positions].tolist()
                counts[label].append(len(set(keys)))
                person_pair_collisions[label] += sum(keys[a] == keys[b]
                                                     for a in range(4) for b in range(a + 1, 4))
    hop = 0 if kind == "bridge" else 1
    rows = torch.arange(len(codes))
    gold_codes = codes[rows, data["support"][:, hop]].reshape(-1, 4)
    result = {"occupied_codes_mean": sum(len(row.unique()) for row in codes) / len(codes),
              "edited_record_key_stable_quartet_rate": float(
                  gold_codes.eq(gold_codes[:, :1]).all(-1).float().mean())}
    for label, values in counts.items():
        result[f"{label}_codes_per_era_mean"] = sum(values) / len(values)
        result[f"{label}_four_distinct_rate"] = sum(value == 4 for value in values) / len(values)
        result[f"{label}_person_pair_collision_rate"] = person_pair_collisions[label] / (6 * len(values))
    return result


def require_completed_run(output: Path) -> tuple[dict, dict]:
    """Enforce the boundary before any checkpoint or fresh TEST tensor is read."""
    closed_path, complete_path = output / "fits_closed.json", output / "complete.json"
    assert closed_path.is_file(), "Fits are not closed; refusing to open TEST data."
    assert complete_path.is_file(), "Scoring is not complete; refusing to open TEST data."
    closed = json.loads(closed_path.read_text())
    complete = json.loads(complete_path.read_text())
    assert closed.get("status") == "all_fits_closed", "Fitting barrier is not closed."
    assert complete.get("status") == "complete", "Scoring is not complete."
    return closed, complete


def check_route_telemetry(prediction: dict, *, mode: str, capacity: int = 2) -> dict:
    """Rebuild buckets independently to audit every saved bounded read.

    This checks the accounting against discrete keys, not only against the
    model's aggregate counters. Actual neural gather/score behavior also needs
    the separate structural instrumentation tests and native replay.
    """
    selected, codes = prediction["selected"], prediction["writer_codes"]
    read_codes = prediction["read_codes"]
    n = len(selected)
    assert codes.shape == (n, 16)
    assert read_codes.shape[:2] == (n, 2)
    probe_budget = 1 if mode == "one_probe" else 2
    assert read_codes.shape[2] == probe_budget
    for key in ("inspected", "returned", "overflow", "bucket_size", "probes", "ties"):
        assert prediction[key].shape == (n, 2)
        assert prediction[key].dtype == torch.long
        assert bool(prediction[key].ge(0).all())
    dropped = 0
    for row in range(n):
        buckets = {}
        for position, code in enumerate(codes[row].tolist()):
            assert 0 <= code < (8 if mode.startswith("paired_collision") else 16)
            buckets.setdefault(code, []).append(position)
        for hop in range(2):
            requests = [code for code in read_codes[row, hop].tolist() if code >= 0]
            assert len(requests) == len(set(requests)) <= probe_budget
            if hop == 1 and int(selected[row, 0]) < 0:
                assert not requests, "Null first hop must abort the chain."
            candidate_ids, sizes, overflows = [], [], 0
            for code in requests:
                bucket = buckets.get(code, [])
                sizes.append(len(bucket))
                if len(bucket) > capacity:
                    overflows += 1
                    dropped += len(bucket)
                else:
                    candidate_ids.extend(bucket)
            assert len(candidate_ids) == len(set(candidate_ids))
            assert len(candidate_ids) <= capacity * probe_budget
            assert int(prediction["returned"][row, hop]) == len(candidate_ids)
            assert int(prediction["probes"][row, hop]) == len(requests)
            assert int(prediction["overflow"][row, hop]) == overflows
            assert int(prediction["bucket_size"][row, hop]) == sum(sizes)
            expected_inspections = 0 if mode == "paired_collision_ordinal" else len(candidate_ids)
            assert int(prediction["inspected"][row, hop]) == expected_inspections
            chosen, tied = int(selected[row, hop]), int(prediction["ties"][row, hop])
            assert tied in (0, 1)
            if chosen >= 0:
                assert chosen in candidate_ids and not tied
            else:
                assert not candidate_ids or tied
            if mode == "paired_collision_ordinal" and candidate_ids:
                assert chosen == min(candidate_ids)
    return {"examples": n, "probes": int(prediction["probes"].sum()),
            "candidate_scores": int(prediction["inspected"].sum()),
            "maximum_scored_per_hop": int(prediction["inspected"].max()),
            "overflow_probes": int(prediction["overflow"].sum()),
            "dropped_occurrences": dropped, "ties": int(prediction["ties"].sum()),
            "reconstructed_bucket_accounting": True}


def _assert_metrics_equal(recomputed: dict, stored: dict, *, label: str) -> None:
    for key, expected in recomputed.items():
        assert key in stored, (label, "missing metric", key)
        actual = stored[key]
        if isinstance(expected, float):
            assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-7), (
                label, key, expected, actual)
        else:
            assert actual == expected, (label, key, expected, actual)


def _check_optimizer(state: dict, *, maximum_step: int, exact: bool) -> dict:
    steps = []
    assert state["state"], "Empty optimizer state."
    for parameter_state in state["state"].values():
        for value in parameter_state.values():
            if isinstance(value, torch.Tensor):
                assert bool(torch.isfinite(value).all()), "Nonfinite optimizer tensor."
        steps.append(int(parameter_state["step"]))
    assert max(steps) == maximum_step and all(0 < value <= maximum_step for value in steps)
    if exact:
        assert set(steps) == {maximum_step}
    return {"parameter_states": len(steps), "minimum_step": min(steps),
            "maximum_step": max(steps), "finite": True}


def _replay_rows(example_count: int, requested: int) -> torch.Tensor:
    assert requested > 0 and requested % 4 == 0
    groups = min(requested // 4, example_count // 4)
    group_ids = torch.linspace(0, example_count // 4 - 1, groups).round().long().unique()
    return (group_ids[:, None] * 4 + torch.arange(4)[None]).flatten()


@torch.inference_mode()
def _native_prediction(model, arm: str, data: dict, mode: str) -> dict:
    """Direct native calls; never invoke the experiment evaluator/summarizer."""
    from torch.nn import functional as F

    records, query = data["records"], data["query"]
    if arm.startswith("repair"):
        if mode == "normal":
            result = model(records, query)
        else:
            memory = model.prepare_memory(records, code_divisor=2 if mode.startswith("paired_collision") else 1)
            result = model.query_memory(memory, query,
                resolver="ordinal" if mode == "paired_collision_ordinal" else "content",
                probe_count=1 if mode == "one_probe" else None)
        logits, selected = result.logits, result.selected
        extras = {key: getattr(result, key) for key in ("writer_codes", "read_codes", "inspected",
                   "returned", "overflow", "bucket_size", "probes", "ties")}
    elif arm == "original_lrc":
        result = model(records, query)
        logits, selected = result.logits[:, -1], result.operands[0][:, -1]
        extras = {"writer_codes": result.decisions[0, "write_relation"][:, :16] * 8
                                   + result.decisions[0, "write_address"][:, :16]}
    else:
        logits, reads = model(records, query, hard=mode == "hard")
        selected, extras = reads.argmax(-1), {}
    return {"pred": logits.argmax(-1), "selected": selected,
            "loss": F.cross_entropy(logits, data["answer"], reduction="none"), **extras}


def run_audit(output: Path, *, source: Path, replay_examples: int = 16) -> dict:
    """Audit the complete frozen cohort, preserving failure outcomes as evidence."""
    here = Path(__file__).resolve().parent
    output, source = output.resolve(), source.resolve()
    closed, complete = require_completed_run(output)
    # No checkpoint or TEST tensor is opened before both closure guards above.
    manifest = json.loads((output / "prefit_manifest.json").read_text())
    evaluation_started = json.loads((output / "evaluation_started.json").read_text())
    assert manifest["experiment"] == "LRC-ROUTE-REPAIR-002"
    assert evaluation_started["fits_closed_sha256"] == sha256(output / "fits_closed.json")
    assert manifest["created_unix"] <= closed["closed_unix"] <= evaluation_started["started_unix"] <= complete["completed_unix"]
    assert closed["source_sha256"] == complete["source_sha256"] == manifest["source_sha256"]
    for name, expected in manifest["source_sha256"].items():
        path = source / name.removeprefix("original/") if name.startswith("original/") else here / name
        assert sha256(path) == expected, ("frozen source changed", name)
    assert any(name.startswith("original/") for name in manifest["source_sha256"])
    for name, expected in complete["files"].items():
        assert sha256(output / name) == expected, ("completed output changed", name)

    sys.path.insert(0, str(source.parent))
    sys.path.insert(0, str(here.parents[1]))
    from experiments.lrc_route_repair.train import build

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    arms, seeds = tuple(manifest["arms"]), tuple(manifest["seeds"])
    assert arms == ("repair", "repair_shuffled", "original_lrc", "memnet")
    assert seeds == (901, 902, 903)
    assert manifest["steps"] == 2000 and manifest["key_steps"] == 1600 and manifest["batch"] == 32
    expected_fits = {(arm, seed) for arm in arms for seed in seeds}
    assert {(receipt["arm"], receipt["seed"]) for receipt in closed["receipts"]} == expected_fits
    assert len(closed["receipts"]) == len(expected_fits) == 12
    train = torch.load(output / "train.pt", map_location="cpu", weights_only=True)
    assert data_digest(train) == manifest["train_sha256"] == "8cd4f0dc7ed824954e0b342c17d50e1d014654a0b84902852353e03498ce69c2"
    assert len(train["answer"]) == manifest["train_examples"] == 8192
    order_digests, order_states = {}, {}
    for seed in seeds:
        generator, digest = torch.Generator().manual_seed(100000 + seed), hashlib.sha256()
        for _ in range(manifest["steps"]):
            indices = torch.randint(len(train["answer"]), (manifest["batch"],), generator=generator)
            digest.update(indices.numpy().tobytes())
        order_digests[seed], order_states[seed] = digest.hexdigest(), generator.get_state()
    checkpoints, fit_audits = {}, []
    for receipt in closed["receipts"]:
        arm, seed = receipt["arm"], receipt["seed"]
        checkpoint_path = output / f"{arm}_{seed}.pt"
        external = json.loads((output / f"{arm}_{seed}.receipt.json").read_text())
        assert external == receipt
        assert sha256(checkpoint_path) == receipt["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert all(checkpoint["receipt"][key] == value for key, value in receipt.items()
                   if key != "checkpoint_sha256")
        assert receipt["steps"] == 2000 and receipt["batch"] == 32 and receipt["examples_seen"] == 64000
        assert receipt["development_only"] is False and receipt["threads"] == 1
        assert receipt["train_sha256"] == manifest["train_sha256"]
        assert receipt["minibatch_sha256"] == order_digests[seed]
        assert torch.equal(checkpoint["minibatch_rng"], order_states[seed])
        assert receipt["finished_unix"] <= closed["closed_unix"]
        assert math.isfinite(receipt["fit_seconds"]) and receipt["fit_seconds"] > 0
        assert checkpoint["rng"].dtype == checkpoint["shuffle_rng"].dtype == torch.uint8
        assert all(bool(torch.isfinite(value).all()) for value in checkpoint["model"].values())
        model = build(arm)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        assert parameters == receipt["parameters"] == manifest["parameters"][arm]
        model.load_state_dict(checkpoint["model"])
        is_repair = arm.startswith("repair")
        assert receipt["training_forwards"] == (4000 if arm == "original_lrc" else 2000)
        assert receipt["optimizer_step_calls"] == (4000 if is_repair else 2000)
        if is_repair:
            optimizer_audits = {
                "stage1": _check_optimizer(checkpoint["key_optimizer_stage1"], maximum_step=1600, exact=True),
                "stage2": _check_optimizer(checkpoint["key_optimizer"], maximum_step=400, exact=True),
                "answer": _check_optimizer(checkpoint["answer_optimizer"], maximum_step=2000, exact=True)}
            codebook = receipt["codebook"]
            assert codebook["unique_training_strings"] == len(torch.unique(train["records"].flatten(0, 1), dim=0)) == 64
            assert codebook["restarts"] == 5 and codebook["iterations_max"] == 100
            assert math.isfinite(codebook["inertia"]) and codebook["seconds"] > 0
            assert checkpoint["model"]["router.codebook"].shape == (16, 32)
        else:
            optimizer_audits = {"main": _check_optimizer(checkpoint["optimizer"], maximum_step=2000,
                                                         exact=arm == "memnet")}
            assert receipt["codebook"] is None
        log_path = output / f"{arm}_{seed}.jsonl"
        assert sha256(log_path) == receipt["log_sha256"]
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(records) == 2000 and [row["step"] for row in records] == list(range(1, 2001))
        assert all(math.isfinite(value) for row in records for value in row.values()
                   if isinstance(value, (int, float)))
        shuffle_generator = torch.Generator().manual_seed(200000 + seed)
        if arm == "repair_shuffled":
            for _ in range(2000):
                torch.randperm(32, generator=shuffle_generator)
        assert torch.equal(checkpoint["shuffle_rng"], shuffle_generator.get_state())
        checkpoints[arm, seed] = checkpoint["model"]
        fit_audits.append({"arm": arm, "seed": seed, "parameters": parameters,
                           "training_forwards": receipt["training_forwards"],
                           "optimizer_step_calls": receipt["optimizer_step_calls"],
                           "optimizers": optimizer_audits, "finite_log_rows": len(records),
                           "checkpoint_sha256": sha256(checkpoint_path)})

    datasets = torch.load(output / "evaluation_data.pt", map_location="cpu", weights_only=True)
    predictions = torch.load(output / "predictions.pt", map_location="cpu", weights_only=True)
    permutations = torch.load(output / "permutation_predictions.pt", map_location="cpu", weights_only=True)
    report = json.loads((output / "results.json").read_text())
    names = ("fresh_iid_a", "fresh_withheld_a", "fresh_iid_b", "fresh_withheld_b", "fresh_bridge")
    assert set(datasets) == set(names)
    assert manifest["test_specs"] == {"fresh_iid_a": [85002, False], "fresh_withheld_a": [85003, True],
                                       "fresh_iid_b": [85005, False], "fresh_withheld_b": [85006, True]}
    assert manifest["bridge_seed"] == 85004 and manifest["test_groups"] == 256
    train_inputs, test_inputs, task_audits = input_signatures(train), {}, {}
    for name, data in datasets.items():
        assert len(data["answer"]) == 1024
        assert data_digest(data) == report["data_hashes"][name]
        kind = "bridge" if name == "fresh_bridge" else "heldout" if "withheld" in name else "iid"
        task_audits[name] = validate_task(data, kind=kind)
        test_inputs[name] = input_signatures(data)
        assert not (train_inputs & test_inputs[name]), ("exact training input overlap", name)
        for previous in task_audits:
            if previous != name:
                assert not (test_inputs[previous] & test_inputs[name]), ("exact cross-TEST input overlap", previous, name)
    expected_prediction_keys = {f"{arm}/{seed}/{name}/{mode}"
        for arm in arms for seed in seeds for name in names
        for mode in (("normal", "hard") if arm == "memnet" else ("normal",))}
    expected_prediction_keys |= {f"repair/{seed}/fresh_bridge/{mode}" for seed in seeds
                                 for mode in ("paired_collision", "paired_collision_ordinal", "one_probe")}
    expected_permutation_keys = {f"repair/{seed}/{mode}" for seed in seeds
                                 for mode in ("normal", "paired_collision")}
    assert set(predictions) == expected_prediction_keys and len(predictions) == 84
    assert set(permutations) == expected_permutation_keys and len(permutations) == 6
    result_rows = {(row["arm"], row["seed"], row["dataset"], row["mode"]): row for row in report["results"]}
    assert len(result_rows) == len(report["results"]) == 90
    expected_rows = {(arm, int(seed), name, mode) for arm, seed, name, mode
                     in (key.split("/") for key in expected_prediction_keys)}
    expected_rows |= {("repair", seed, "fresh_bridge_reversed", mode)
                      for seed in seeds for mode in ("normal", "paired_collision")}
    assert set(result_rows) == expected_rows
    telemetry, key_diagnostics = {}, {}
    for key, prediction in predictions.items():
        arm, seed, name, mode = key.split("/")
        data = datasets[name]
        metrics = prediction_metrics(prediction, data)
        if "inspected" in prediction:
            telemetry[key] = check_route_telemetry(prediction, mode=mode)
            metrics.update(inspected_per_hop=float(prediction["inspected"].float().mean()),
                maximum_inspected_per_hop=int(prediction["inspected"].max()),
                overflow_probes=int(prediction["overflow"].sum()), ties=int(prediction["ties"].sum()),
                null_read_fraction=float(prediction["selected"].lt(0).float().mean()),
                occupied_codes_mean=sum(len(row.unique()) for row in prediction["writer_codes"]) / len(data["answer"]))
        _assert_metrics_equal(metrics, result_rows[arm, int(seed), name, mode], label=key)
        if mode == "normal" and "writer_codes" in prediction:
            kind = "bridge" if name == "fresh_bridge" else "heldout" if "withheld" in name else "iid"
            key_diagnostics[key] = key_metrics(prediction["writer_codes"], data, kind=kind)
    reversal = torch.cat((torch.arange(7, -1, -1), torch.arange(15, 7, -1)))
    permutation_audits = {}
    permuted_data = {**datasets["fresh_bridge"], "records": datasets["fresh_bridge"]["records"][:, reversal],
                     "support": reversal[datasets["fresh_bridge"]["support"]]}
    validate_task(permuted_data, kind="bridge")
    for key, prediction in permutations.items():
        arm, seed, mode = key.split("/")
        metrics = prediction_metrics(prediction, permuted_data)
        telemetry["permutation/" + key] = check_route_telemetry(prediction, mode=mode)
        reference = predictions[f"repair/{seed}/fresh_bridge/{mode}"]
        mapped = torch.where(prediction["selected"] >= 0,
                             reversal[prediction["selected"].clamp_min(0)], -1)
        metrics.update(inspected_per_hop=float(prediction["inspected"].float().mean()),
            maximum_inspected_per_hop=int(prediction["inspected"].max()),
            overflow_probes=int(prediction["overflow"].sum()), ties=int(prediction["ties"].sum()),
            null_read_fraction=float(prediction["selected"].lt(0).float().mean()),
            occupied_codes_mean=sum(len(row.unique()) for row in prediction["writer_codes"]) / len(permuted_data["answer"]),
            identical_answers=bool(torch.equal(prediction["pred"], reference["pred"])),
            identical_mapped_supports=bool(torch.equal(mapped, reference["selected"])))
        _assert_metrics_equal(metrics, result_rows[arm, int(seed), "fresh_bridge_reversed", mode], label=key)
        assert torch.equal(prediction["writer_codes"], reference["writer_codes"][:, reversal])
        permutation_audits[key] = {"answers_unchanged": metrics["identical_answers"],
                                   "mapped_supports_unchanged": metrics["identical_mapped_supports"],
                                   "writer_codes_follow_record_content": True}
    primary = [result_rows["repair", seed, name, "normal"] for seed in seeds for name in names]
    repair_withheld = [result_rows["repair", seed, name, "normal"]["accuracy"]
                       for seed in seeds for name in names if "withheld" in name]
    shuffled_withheld = [result_rows["repair_shuffled", seed, name, "normal"]["accuracy"]
                         for seed in seeds for name in names if "withheld" in name]
    label_gain = sum(repair_withheld) / len(repair_withheld) - sum(shuffled_withheld) / len(shuffled_withheld)
    gates = {"answer_and_quartet": all(row["accuracy"] >= .95 and row["quartet_accuracy"] >= .90 for row in primary),
             "both_supports": all(row["both"] >= .95 for row in primary),
             "no_overflow": all(row["overflow_probes"] == 0 for row in primary),
             "bounded_candidates": all(row["maximum_inspected_per_hop"] <= 4 for row in primary),
             "permutation": all(value["answers_unchanged"] and value["mapped_supports_unchanged"]
                                for value in permutation_audits.values()),
             "label_specific_benefit": label_gain >= .10}
    assert gates == report["gates"] == complete["gates"]
    assert all(gates.values()) == report["all_gates"]
    assert math.isclose(label_gain, report["repair_minus_shuffled_withheld"], rel_tol=0, abs_tol=1e-12)

    replay_audits, maximum_loss_error = [], 0.0
    for arm in arms:
        for seed in seeds:
            model = build(arm).eval().requires_grad_(False)
            model.load_state_dict(checkpoints[arm, seed])
            replay_items = [(key, prediction, datasets[key.split("/")[2]])
                            for key, prediction in predictions.items() if key.startswith(f"{arm}/{seed}/")]
            if arm == "repair":
                replay_items += [(f"{arm}/{seed}/fresh_bridge_reversed/{mode}",
                                  permutations[f"{arm}/{seed}/{mode}"], permuted_data)
                                 for mode in ("normal", "paired_collision")]
            for key, expected, data in replay_items:
                indices = _replay_rows(len(data["answer"]), replay_examples)
                batch = {name: values[indices] for name, values in data.items()}
                actual = _native_prediction(model, arm, batch, key.split("/")[-1])
                assert set(actual) == set(expected), (key, set(actual), set(expected))
                error = float((actual["loss"] - expected["loss"][indices]).abs().max())
                maximum_loss_error = max(maximum_loss_error, error)
                assert error <= 2e-6, ("native replay loss mismatch", key, error)
                for field in actual:
                    if field != "loss":
                        assert torch.equal(actual[field].cpu(), expected[field][indices]), ("native replay mismatch", key, field)
                replay_audits.append({"cell": key, "indices": indices.tolist(), "examples": len(indices),
                                      "maximum_loss_error": error, "all_discrete_fields_exact": True})
    return {"status": "passed", "meaning": "Integrity and replay passed; scientific gates are reported separately.",
            "audit_source_sha256": sha256(Path(__file__)),
            "prefit_manifest_sha256": sha256(output / "prefit_manifest.json"),
            "fits_closed_sha256": sha256(output / "fits_closed.json"),
            "complete_sha256": sha256(output / "complete.json"),
            "frozen_source_files": len(manifest["source_sha256"]),
            "fits": fit_audits, "tasks": task_audits,
            "no_exact_training_or_cross_test_overlap": True,
            "score_cells": len(result_rows), "saved_predictions": sum(len(p["pred"]) for p in predictions.values())
                + sum(len(p["pred"]) for p in permutations.values()),
            "route_telemetry": telemetry, "key_diagnostics": key_diagnostics,
            "permutations": permutation_audits, "gates": gates,
            "repair_minus_shuffled_withheld": label_gain,
            "native_replay_cells": len(replay_audits),
            "native_replay_examples": sum(row["examples"] for row in replay_audits),
            "native_replay_maximum_loss_error": maximum_loss_error,
            "replay": replay_audits,
            "limits": ["Replay is a deterministic bounded subset, not all saved forward passes.",
                       "Structural bounded-score tests complement saved counter reconstruction.",
                       "No independent fit rerun or ingredient-level causal attribution is performed."]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="Extracted lrc_vb001_comparison package directory.")
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--replay-examples", type=int, default=16)
    args = parser.parse_args()
    destination = args.audit_output or args.output / "independent_audit.json"
    assert not destination.exists(), "Refusing to overwrite an audit result."
    result = run_audit(args.output, source=args.source, replay_examples=args.replay_examples)
    with destination.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps({"status": result["status"], "fits": len(result["fits"]),
                      "score_cells": result["score_cells"], "predictions": result["saved_predictions"],
                      "native_replay_cells": result["native_replay_cells"],
                      "native_replay_examples": result["native_replay_examples"],
                      "native_replay_maximum_loss_error": result["native_replay_maximum_loss_error"],
                      "gates": result["gates"], "audit": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
