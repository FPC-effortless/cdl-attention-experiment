"""Frozen LRC refinement diagnostic. No fitting or model mutation is performed."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from lrc_vb001_comparison.source import study as vb
from experiments.lrc_route_repair import run as original
from experiments.lrc_route_repair.bridge_data import (
    _parse_record, bridge_dataset, validate_bridge_dataset,
)
from .backend import RefinedIndex, choose_probes, prefix_bits
from .validation import validate_location_quartets, split_overlap

HERE = Path(__file__).resolve().parent
SEEDS = (901, 902, 903)
HASH_SEEDS = (94101, 94102, 94103, 94104, 94105)
DEPTHS = (0, 1, 2, 4, 8, 16, 32)
GROUPS = 128
DATASETS = {
    "fresh_iid_a": (94201, False), "fresh_withheld_a": (94202, True),
    "fresh_iid_b": (94203, False), "fresh_withheld_b": (94204, True),
    "fresh_bridge": (94205, None),
}
COUNTERS = ("scores", "probes", "empty", "overflow", "bucketrefs", "ties")
REFERENCE_COUNTERS = ("probes", "empty", "overflow", "bucketrefs")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_hashes():
    paths = sorted(HERE.glob("*.py")) + [HERE / "PROTOCOL.md"]
    return {path.name: digest(path) for path in paths}


def checkpoint_hashes(repair):
    result = {}
    for seed in SEEDS:
        path = repair / f"repair_{seed}.pt"
        receipt = json.loads((repair / f"repair_{seed}.receipt.json").read_text())
        result[path.name] = digest(path)
        if result[path.name] != receipt["checkpoint_sha256"]:
            raise RuntimeError("Original checkpoint differs from its fitting receipt")
    return result


def environment():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "thread_environment": {name: os.environ.get(name) for name in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def configure():
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    expected = ("3.12.14", "2.5.1+cpu", "2.3.5")
    if (platform.python_version(), str(torch.__version__), np.__version__) != expected:
        raise RuntimeError("Use the recorded Python/Torch/NumPy environment")
    if any(os.environ.get(name) != "1" for name in
           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise RuntimeError("Set all three CPU/BLAS thread environment variables to 1")


def unique_tokens(train):
    return torch.unique(train["records"].reshape(-1, vb.MAX_WORDS), dim=0), \
        torch.unique(train["query"], dim=0)


class FrozenCache:
    """Exact raw-token lookup for repeated offline neural inputs; no labels."""

    def __init__(self, model, record_tokens, query_tokens):
        self.model = model
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Model must be frozen and in evaluation mode")
        self.record_tokens = record_tokens
        self.query_tokens = query_tokens
        self.record_lookup = {tuple(row): i for i, row in enumerate(record_tokens.tolist())}
        self.query_lookup = {tuple(row): i for i, row in enumerate(query_tokens.tolist())}
        self.values, self.keys = model.router.write(record_tokens)
        self.codes = (self.keys @ model.router.codebook.T).argmax(-1).numpy()
        self.states, first = model.router.query_first(query_tokens)
        self.logits = model.answer(self.values).numpy()
        self.null_logits = model.answer(torch.zeros_like(self.values[0])).numpy()
        self.readers = {(i, -1): key for i, key in enumerate(first)}
        self.orders = {}
        self.projected_readers = {}
        self.probe_cache = {}
        self.dense_audit_cache = {}
        self.dense_audit_new_scores = 0

    def inputs(self, data):
        records = np.array([[self.record_lookup[tuple(row)] for row in story]
                            for story in data["records"].tolist()], dtype=np.int16)
        queries = np.array([self.query_lookup[tuple(row)] for row in data["query"].tolist()],
                           dtype=np.int16)
        return records, queries

    def reader(self, query_id, first_id=-1):
        key = (int(query_id), int(first_id))
        if key not in self.readers:
            self.readers[key] = self.model.router.query_second(
                self.states[key[0]], self.values[key[1]])
        return self.readers[key]

    def coarse_order(self, query_id, first_id=-1):
        key = (int(query_id), int(first_id))
        if key not in self.orders:
            scores = self.reader(*key) @ self.model.router.codebook.T
            self.orders[key] = tuple(scores.argsort(descending=True, stable=True)[:2].tolist())
        return self.orders[key]

    def requests(self, config, descriptor, query_id, first_id=-1):
        reader_id = (int(query_id), int(first_id))
        cache_id = (config["id"], *reader_id)
        if cache_id not in self.probe_cache:
            depth = config["depth"]
            if depth == 0:
                projected = np.zeros(32, dtype=np.float64)
                thresholds = np.zeros((16, 32), dtype=np.float64)
            else:
                projected_id = (config["hash_seed"], *reader_id)
                if projected_id not in self.projected_readers:
                    self.projected_readers[projected_id] = (
                        descriptor["normals"] @ self.reader(*reader_id).double().numpy())
                projected = self.projected_readers[projected_id]
                thresholds = descriptor["thresholds"]
            self.probe_cache[cache_id] = choose_probes(
                self.coarse_order(*reader_id), projected, thresholds, depth, config["policy"])
        return self.probe_cache[cache_id]

    def score(self, record_ids, read_key):
        # Identical reduction for each record regardless of candidate-set size.
        return (self.keys[torch.as_tensor(record_ids, dtype=torch.long)] * read_key).sum(-1)

    def dense_audit(self, record_ids, query_id, first_id=-1):
        cache_id = (int(query_id), int(first_id), *map(int, record_ids))
        if cache_id not in self.dense_audit_cache:
            scores = self.score(record_ids, self.reader(query_id, first_id))
            maximum = scores.max()
            winners = scores.eq(maximum).nonzero().flatten()
            chosen = int(winners[0]) if len(winners) == 1 else -1
            ranks = (scores[None, :] > scores[:, None]).sum(-1).numpy() + 1
            self.dense_audit_cache[cache_id] = (chosen, scores.numpy(), ranks)
            self.dense_audit_new_scores += len(record_ids)
        return self.dense_audit_cache[cache_id]


def configurations():
    yield {"id": "coarse", "family": "coarse", "depth": 0,
           "policy": "two_coarse", "hash_seed": None}
    yield {"id": "dense", "family": "dense", "depth": 0,
           "policy": "dense", "hash_seed": None}
    yield {"id": "coarse_sibling", "family": "coarse", "depth": 0,
           "policy": "sibling", "hash_seed": None}
    for family in ("origin", "train_median"):
        for hash_seed in HASH_SEEDS:
            for depth in DEPTHS[1:]:
                for policy in ("two_coarse", "sibling"):
                    yield {"id": f"{family}_{hash_seed}_d{depth}_{policy}",
                           "family": family, "hash_seed": hash_seed,
                           "depth": depth, "policy": policy}


def descriptor_for(arrays, config):
    if config["depth"] == 0:
        return {"normals": np.zeros((32, 32)), "thresholds": np.zeros((16, 32))}
    h = config["hash_seed"]
    return {"normals": arrays[f"normals_{h}"],
            "thresholds": arrays[f"{config['family']}_{h}"]}


def prepare(repair, output):
    original_manifest = original.check_frozen(repair)
    checkpoints = checkpoint_hashes(repair)
    train = torch.load(repair / "train.pt", weights_only=True)
    if vb.digest_data(train) != original_manifest["train_sha256"]:
        raise RuntimeError("Original TRAIN tensor digest changed")
    output.mkdir(parents=True, exist_ok=False)
    records, queries = unique_tokens(train)
    descriptor_files, descriptor_receipts = {}, {}
    for seed in SEEDS:
        cache = FrozenCache(original.load_model(repair, "repair", seed), records, queries)
        arrays = {"record_tokens": records.numpy(), "query_tokens": queries.numpy(),
                  "writer_keys": cache.keys.numpy(), "writer_codes": cache.codes}
        counts = np.bincount(cache.codes, minlength=16)
        for h in HASH_SEEDS:
            generator = torch.Generator().manual_seed(h)
            normals = torch.randn((32, 32), dtype=torch.float64, generator=generator)
            normals = (normals / normals.norm(dim=-1, keepdim=True)).numpy()
            projections = cache.keys.double().numpy() @ normals.T
            medians = np.zeros((16, 32), dtype=np.float64)
            for code in range(16):
                if counts[code]:
                    medians[code] = np.median(projections[cache.codes == code], axis=0)
            arrays[f"normals_{h}"] = normals
            arrays[f"origin_{h}"] = np.zeros_like(medians)
            arrays[f"train_median_{h}"] = medians
        path = output / f"descriptors_{seed}.npz"
        np.savez_compressed(path, **arrays)
        descriptor_files[path.name] = digest(path)
        descriptor_receipts[str(seed)] = {
            "unique_train_record_strings": len(records), "unique_train_queries": len(queries),
            "record_tokens_digest": array_digest(records.numpy()),
            "coarse_counts": counts.tolist(), "empty_code_fallbacks": np.flatnonzero(counts == 0).tolist(),
            "arrays": {name: array_digest(value) for name, value in arrays.items()},
        }
    manifest = {
        "experiment": "LRC-REFINEMENT-001", "created_unix": time.time(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                                  text=True).strip(),
        "source_tree": subprocess.check_output(["git", "write-tree"], cwd=HERE, text=True).strip(),
        "source_sha256": source_hashes(), "original_source_sha256": original_manifest["source_sha256"],
        "original_checkpoints_sha256": checkpoints, "train_sha256": vb.digest_data(train),
        "environment": environment(), "model_seeds": SEEDS, "hash_seeds": HASH_SEEDS,
        "depths": DEPTHS, "datasets": DATASETS, "quartets_per_dataset": GROUPS,
        "configs": list(configurations()), "descriptor_files": descriptor_files,
        "descriptor_receipts": descriptor_receipts, "training_performed": False,
        "permutation_seed": 94501, "permutation_examples_per_cohort": 16,
    }
    write_json(output / "prescoring_manifest.json", manifest)
    print(json.dumps({"status": "frozen", "manifest_sha256": digest(output / "prescoring_manifest.json"),
                      "model_seeds": SEEDS, "configs_per_seed": len(manifest["configs"])}), flush=True)


def check_frozen(repair, output):
    manifest = json.loads((output / "prescoring_manifest.json").read_text())
    if source_hashes() != manifest["source_sha256"]:
        raise RuntimeError("New diagnostic source changed after pre-scoring freeze")
    if original.check_frozen(repair)["source_sha256"] != manifest["original_source_sha256"]:
        raise RuntimeError("Original frozen source differs")
    if checkpoint_hashes(repair) != manifest["original_checkpoints_sha256"]:
        raise RuntimeError("Original checkpoints changed")
    train = torch.load(repair / "train.pt", weights_only=True)
    if vb.digest_data(train) != manifest["train_sha256"]:
        raise RuntimeError("Original TRAIN data changed after descriptor preparation")
    for name, expected in manifest["descriptor_files"].items():
        if digest(output / name) != expected:
            raise RuntimeError("Frozen descriptors changed")
    return manifest


def empty_trace(n):
    return {
        "pred": np.zeros(n, dtype=np.int16), "selected": np.full((n, 2), -1, dtype=np.int16),
        "active": np.zeros((n, 2), dtype=np.uint8),
        "gold_capture": np.zeros((n, 2), dtype=np.uint8),
        "dense_capture": np.zeros((n, 2), dtype=np.uint8),
        "dense_selected": np.full((n, 2), -1, dtype=np.int16),
        "gold_rank": np.zeros((n, 2), dtype=np.int16),
        "gold_score": np.zeros((n, 2), dtype=np.float32),
        "counts": np.zeros((n, 2, len(COUNTERS)), dtype=np.int16),
        "codes": np.full((n, 2, 2), -1, dtype=np.int16),
        "prefixes": np.zeros((n, 2, 2), dtype=np.uint64),
        "candidates": np.full((n, 2, 16), -1, dtype=np.int16),
        "ref_gold_capture": np.zeros((n, 2), dtype=np.uint8),
        "ref_dense_capture": np.zeros((n, 2), dtype=np.uint8),
        "reference_audit_counts": np.zeros((n, 2, len(REFERENCE_COUNTERS)), dtype=np.int16),
        "logits": np.zeros((n, 4), dtype=np.float32),
    }


def writer_prefixes(cache, descriptor, depth):
    projected = cache.keys.double().numpy() @ descriptor["normals"].T
    return prefix_bits(projected, descriptor["thresholds"][cache.codes], depth)


def execute(cache, data, config, descriptor, reference=None):
    """Actual bounded reads; gold labels are consulted only after lookup/selection."""
    record_ids, query_ids = cache.inputs(data)
    labels = data["support"].numpy()
    n = len(query_ids)
    trace = empty_trace(n)
    dense = config["family"] == "dense"
    prefixes = writer_prefixes(cache, descriptor, config["depth"])
    for row in range(n):
        ids = record_ids[row]
        query_id = int(query_ids[row])
        index = None if dense else RefinedIndex(cache.codes[ids], prefixes[ids], capacity=2)
        first_id, final_id = -1, -1
        for hop in range(2):
            trace["active"][row, hop] = 1
            read_key = cache.reader(query_id, first_id)
            if dense:
                candidates = tuple(range(len(ids)))
                trace["counts"][row, hop, 0] = len(candidates)
            else:
                requests = cache.requests(config, descriptor, query_id, first_id)
                hit = index.gather(requests)
                candidates = hit.candidates
                trace["counts"][row, hop, :5] = (
                    len(candidates), hit.probes, hit.empty, hit.overflow, hit.bucketrefs)
                for slot, (code, prefix) in enumerate(requests):
                    trace["codes"][row, hop, slot] = code
                    trace["prefixes"][row, hop, slot] = prefix
            trace["candidates"][row, hop, :len(candidates)] = candidates
            chosen = -1
            if candidates:
                scores = cache.score(ids[list(candidates)], read_key)
                winners = scores.eq(scores.max()).nonzero().flatten()
                if len(winners) == 1:
                    chosen = candidates[int(winners[0])]
                else:
                    trace["counts"][row, hop, 5] = 1
            trace["selected"][row, hop] = chosen
            # The following full-key comparison is explicitly offline audit work.
            winner, full_scores, ranks = cache.dense_audit(ids, query_id, first_id)
            trace["dense_selected"][row, hop] = winner
            trace["dense_capture"][row, hop] = winner >= 0 and winner in candidates
            gold = int(labels[row, hop])
            trace["gold_capture"][row, hop] = gold in candidates
            trace["gold_rank"][row, hop] = ranks[gold]
            trace["gold_score"][row, hop] = full_scores[gold]
            if winner >= 0 and winner in candidates and chosen != winner:
                raise AssertionError("Unique dense winner captured but not selected")
            if hop == 0:
                if chosen < 0:
                    break
                first_id = int(ids[chosen])
            else:
                final_id = int(ids[chosen]) if chosen >= 0 else -1
        logits = cache.logits[final_id] if final_id >= 0 else cache.null_logits
        trace["logits"][row] = logits
        trace["pred"][row] = logits.argmax()
        if reference is not None:
            reference_first = int(reference["selected"][row, 0])
            for hop in range(2):
                if hop == 1 and reference_first < 0:
                    continue
                reader_first = -1 if hop == 0 else int(ids[reference_first])
                if dense:
                    candidates = tuple(range(len(ids)))
                else:
                    requests = cache.requests(config, descriptor, query_id, reader_first)
                    ref_hit = index.gather(requests)
                    candidates = ref_hit.candidates
                    trace["reference_audit_counts"][row, hop] = (
                        ref_hit.probes, ref_hit.empty, ref_hit.overflow, ref_hit.bucketrefs)
                winner, _, _ = cache.dense_audit(ids, query_id, reader_first)
                trace["ref_gold_capture"][row, hop] = int(labels[row, hop]) in candidates
                trace["ref_dense_capture"][row, hop] = winner >= 0 and winner in candidates
    if reference is None:
        trace["ref_gold_capture"][:] = trace["gold_capture"]
        trace["ref_dense_capture"][:] = trace["dense_capture"]
    return trace


def summarize(trace, data, config):
    labels = data["support"].numpy()
    correct = trace["pred"] == data["answer"].numpy()
    selected = trace["selected"]
    hits = selected == labels
    captured = trace["gold_capture"].astype(bool)
    dense_valid = trace["dense_selected"] >= 0
    dense_agreement = selected == trace["dense_selected"]
    active = trace["active"].astype(bool)
    counts = trace["counts"].astype(np.int64)
    result = {
        "examples": len(correct), "quartets": len(correct) // 4,
        "answer_correct": int(correct.sum()), "answer_accuracy": float(correct.mean()),
        "perfect_quartets": int(correct.reshape(-1, 4).all(-1).sum()),
        "quartet_accuracy": float(correct.reshape(-1, 4).all(-1).mean()),
        "both_correct": int(hits.all(-1).sum()), "both_support_accuracy": float(hits.all(-1).mean()),
        "hop_support_accuracy": hits.mean(0).tolist(), "gold_capture": captured.mean(0).tolist(),
        "reference_gold_capture": trace["ref_gold_capture"].mean(0).tolist(),
        "reference_dense_capture": trace["ref_dense_capture"].mean(0).tolist(),
        "active_hops": active.sum(0).tolist(), "captured_gold_hops": captured.sum(0).tolist(),
        "selected_gold_given_capture": [float(hits[:, h][captured[:, h]].mean())
                                         if captured[:, h].any() else None for h in range(2)],
        "dense_unique_hops": int(dense_valid.sum()),
        "dense_winner_capture": float(trace["dense_capture"].sum() / dense_valid.sum())
                                if dense_valid.any() else None,
        "dense_agreement": float(dense_agreement[dense_valid].mean()) if dense_valid.any() else None,
        "dense_capture_selection_violations": int((trace["dense_capture"].astype(bool)
                                                    & ~dense_agreement).sum()),
        "filtering_rescues": int((hits & dense_valid & (trace["dense_selected"] != labels)).sum()),
        "aborted_chains": int((selected[:, 0] < 0).sum()),
        "null_reads": int(((selected < 0) & active).sum()),
        "max_candidate_scores": int(counts[:, :, 0].max()),
        "candidate_scores_per_example": float(counts[:, :, 0].sum() / len(correct)),
        "candidate_scores_per_active_read": float(counts[:, :, 0].sum() / active.sum()),
        "counters": {name: int(counts[:, :, i].sum()) for i, name in enumerate(COUNTERS)},
        "reference_audit_counters": {name: int(trace["reference_audit_counts"][:, :, i].sum())
                                     for i, name in enumerate(REFERENCE_COUNTERS)},
        "gold_rank_mean_on_active_reads": float(trace["gold_rank"][active].mean()),
        "gold_score_mean_on_active_reads": float(trace["gold_score"][active].mean()),
        "declared_unamortized_codeword_scores": 0 if config["family"] == "dense" else int(active.sum()) * 16,
        "declared_unamortized_hash_projections": int(active.sum()) * config["depth"],
        "declared_unamortized_threshold_checks": int(active.sum()) * config["depth"]
                                                 * (2 if config["policy"] == "two_coarse" else 1),
        "declared_sibling_margin_comparisons": int(active.sum()) * max(0, config["depth"] - 1)
                                               if config["policy"] == "sibling" else 0,
    }
    if config["family"] != "dense" and result["max_candidate_scores"] > 4:
        raise AssertionError("Bounded record-score allowance violated")
    return result


def native_parity(cache, data, trace):
    model = cache.model
    fields = {"selected": [], "pred": [], "logits": [], "codes": [], "counts": [],
              "writer_codes": [], "returned": []}
    for start in range(0, len(data["answer"]), 128):
        records, query = data["records"][start:start + 128], data["query"][start:start + 128]
        out = model(records, query)
        fields["selected"].append(out.selected.numpy())
        fields["pred"].append(out.logits.argmax(-1).numpy())
        fields["logits"].append(out.logits.numpy())
        fields["codes"].append(out.read_codes.numpy())
        fields["writer_codes"].append(out.writer_codes.numpy())
        fields["returned"].append(out.returned.numpy())
        native_bucket_sizes = (out.writer_codes[:, None, None, :]
                               == out.read_codes[:, :, :, None]).sum(-1)
        native_empty = ((native_bucket_sizes == 0) & (out.read_codes >= 0)).sum(-1)
        fields["counts"].append(torch.stack((out.inspected, out.probes,
            native_empty, out.overflow, out.bucket_size, out.ties), -1).numpy())
    fields = {name: np.concatenate(values) for name, values in fields.items()}
    for name in ("selected", "pred", "codes"):
        if not np.array_equal(fields[name], trace[name]):
            raise AssertionError(f"Native depth-zero {name} parity failed")
    for index in range(len(COUNTERS)):
        if not np.array_equal(fields["counts"][:, :, index], trace["counts"][:, :, index]):
            raise AssertionError(f"Native depth-zero {COUNTERS[index]} parity failed")
    if not np.array_equal(fields["returned"], trace["counts"][:, :, 0]):
        raise AssertionError("Native depth-zero returned-candidate parity failed")
    ids, _ = cache.inputs(data)
    if not np.array_equal(fields["writer_codes"], cache.codes[ids]):
        raise AssertionError("Native writer code parity failed")
    delta = float(np.abs(fields["logits"] - trace["logits"]).max())
    if delta > 1e-6:
        raise AssertionError(f"Native logit parity failed: {delta}")
    return {"status": "passed", "examples": len(data["answer"]), "max_logit_abs_difference": delta,
            "selected_codes_counters_predictions_equal": True}


def reference_invariance(cache, data, reference, bridge):
    ids, query_ids = cache.inputs(data)
    hop = 0 if bridge else 1
    groups = []
    for start in range(0, len(ids), 4):
        reader_keys, input_keys = [], []
        active = True
        for row in range(start, start + 4):
            position = int(reference["selected"][row, 0])
            if hop == 1 and position < 0:
                active = False
                break
            first_id = -1 if hop == 0 else int(ids[row, position])
            input_keys.append((int(query_ids[row]), first_id))
            reader_keys.append(cache.reader(query_ids[row], first_id))
        equal_inputs = active and len(set(input_keys)) == 1
        equal_keys = active and all(torch.equal(reader_keys[0], x) for x in reader_keys[1:])
        groups.append(bool(equal_inputs and equal_keys))
    bridge_changed = None
    if bridge:
        bridge_changed = []
        for start in range(0, len(ids), 4):
            keys = []
            for row in range(start, start + 4):
                first = int(reference["selected"][row, 0])
                if first >= 0:
                    keys.append(cache.reader(query_ids[row], int(ids[row, first])).numpy().tobytes())
            bridge_changed.append(len(set(keys)))
    return {"hop": hop, "eligible": np.array(groups, dtype=np.uint8),
            "eligible_quartets": sum(groups), "total_quartets": len(groups),
            "bridge_hop2_distinct_read_keys": bridge_changed}


def coverage_bounds(cache, data, descriptor, depth, invariance):
    ids, _ = cache.inputs(data)
    hop = invariance["hop"]
    support = data["support"][:, hop].numpy()
    target_ids = ids[np.arange(len(ids)), support]
    prefixes = writer_prefixes(cache, descriptor, depth)
    diversity, ceilings, coarse_diversity = [], [], []
    for start in range(0, len(ids), 4):
        targets = target_ids[start:start + 4]
        addresses = [(int(cache.codes[i]), int(prefixes[i])) for i in targets]
        frequencies = sorted(Counter(addresses).values(), reverse=True)
        diversity.append(len(frequencies))
        coarse_diversity.append(len(set(int(cache.codes[i]) for i in targets)))
        ceilings.append([sum(frequencies[:k]) / 4 for k in (1, 2, 4)])
    eligible = invariance["eligible"].astype(bool)
    ceilings = np.array(ceilings, dtype=np.float64)
    diversity = np.array(diversity, dtype=np.int16)
    return {
        "hop": hop, "eligible_quartets": int(eligible.sum()), "total_quartets": len(eligible),
        "coarse_stable_fraction": float((np.array(coarse_diversity) == 1).mean()),
        "target_prefix_diversity_mean": float(diversity[eligible].mean()) if eligible.any() else None,
        "target_prefix_diversity_max": int(diversity[eligible].max()) if eligible.any() else None,
        "U1": float(ceilings[eligible, 0].mean()) if eligible.any() else None,
        "U2": float(ceilings[eligible, 1].mean()) if eligible.any() else None,
        "U4": float(ceilings[eligible, 2].mean()) if eligible.any() else None,
        "U2_below_one_quartets": int((ceilings[eligible, 1] < 1).sum()),
    }, {"eligible": eligible.astype(np.uint8), "diversity": diversity,
        "ceilings": ceilings, "target_ids": target_ids.reshape(-1, 4),
        "target_codes": cache.codes[target_ids].reshape(-1, 4),
        "target_prefixes": prefixes[target_ids].reshape(-1, 4)}


def geometry(cache):
    groups = {}
    for record_id, tokens in enumerate(cache.record_tokens):
        era, person, relation, payload = _parse_record(tokens)
        groups.setdefault((era, person, relation), {})[payload] = record_id
    if len(groups) != 16 or any(set(value) != set(range(4)) for value in groups.values()):
        raise AssertionError("The TRAIN template universe does not cover all 64 valid strings")
    result = []
    for address, payloads in sorted(groups.items()):
        ids = [payloads[i] for i in range(4)]
        keys = cache.keys[ids].double()
        unit = keys / keys.norm(dim=-1, keepdim=True)
        angles = torch.acos((unit @ unit.T).clamp(-1, 1)) * (180 / np.pi)
        diameter = torch.cdist(keys, keys).max()
        result.append({"era": address[0], "subject": address[1], "relation": address[2],
                       "record_ids": ids, "coarse_codes": cache.codes[ids].tolist(),
                       "unique_fine_keys": len(torch.unique(keys, dim=0)),
                       "payload_l2_diameter": float(diameter),
                       "payload_angle_diameter_degrees": float(angles.max())})
    separation = []
    for payload in range(4):
        ids = [value[payload] for _, value in sorted(groups.items())]
        keys = cache.keys[ids].double()
        distances = torch.cdist(keys, keys)
        distances.fill_diagonal_(float("inf"))
        separation.append({"payload_index": payload, "distinct_coarse_codes": len(set(cache.codes[ids])),
                           "distinct_logical_addresses": 16,
                           "min_between_address_l2": float(distances.min())})
    return {"payload_groups": result, "address_separation": separation,
            "record_tokens": cache.record_tokens.tolist()}


def permutation_check(cache, data, config, descriptor, reference, permutation, trace):
    count = min(16, len(data["answer"]))
    inverse = np.argsort(permutation)
    small = {name: value[:count].clone() for name, value in data.items()}
    small["records"] = small["records"][:, torch.tensor(permutation)]
    small["support"] = torch.tensor(inverse)[small["support"]]
    permuted = execute(cache, small, config, descriptor)
    selected = permuted["selected"]
    mapped = np.where(selected >= 0, permutation[np.maximum(selected, 0)], -1)
    if not np.array_equal(mapped, trace["selected"][:count]):
        raise AssertionError("Record permutation changed selected occurrences")
    if not np.array_equal(permuted["pred"], trace["pred"][:count]):
        raise AssertionError("Record permutation changed answer predictions")
    if not np.array_equal(permuted["counts"], trace["counts"][:count]):
        raise AssertionError("Record permutation changed read counters")
    if not np.array_equal(permuted["codes"], trace["codes"][:count]):
        raise AssertionError("Record permutation changed requested coarse codes")
    if not np.array_equal(permuted["prefixes"], trace["prefixes"][:count]):
        raise AssertionError("Record permutation changed requested fine prefixes")
    candidates = permuted["candidates"]
    mapped_candidates = np.where(candidates >= 0, permutation[np.maximum(candidates, 0)], -1)
    if not np.array_equal(np.sort(mapped_candidates, axis=-1),
                          np.sort(trace["candidates"][:count], axis=-1)):
        raise AssertionError("Record permutation changed the mapped candidate sets")
    if config["id"] == "coarse":
        native_parity(cache, small, permuted)
    return count


def evaluate(repair, output):
    manifest = check_frozen(repair, output)
    write_json(output / "evaluation_started.json", {"started_unix": time.time(),
               "prescoring_manifest_sha256": digest(output / "prescoring_manifest.json")})
    data = {name: bridge_dataset(seed, GROUPS) if withheld is None else vb.dataset(seed, GROUPS, withheld)
            for name, (seed, withheld) in DATASETS.items()}
    validations = {name: validate_bridge_dataset(data[name]) if withheld is None
                   else validate_location_quartets(data[name], withheld)
                   for name, (_, withheld) in DATASETS.items()}
    train = torch.load(repair / "train.pt", weights_only=True)
    overlaps = split_overlap(train, data)
    data_hashes_before = {name: vb.digest_data(value) for name, value in data.items()}
    torch.save(data, output / "evaluation_data.pt")
    write_json(output / "data_validation.json", {"validations": validations, "overlap": overlaps,
               "data_hashes": data_hashes_before})
    if not overlaps["all_disjoint"]:
        raise RuntimeError("Fresh input panels overlap; preserve this failed qualification without reseeding")
    rng = np.random.default_rng(94501)
    permutation = np.concatenate((rng.permutation(8), rng.permutation(8) + 8))
    if np.array_equal(permutation[permutation], np.arange(16)):
        raise AssertionError("Declared permutation is accidentally self-inverse")
    all_rows, bounds_rows, parities, invariances, geometries, work = [], [], {}, {}, {}, {}
    for seed in SEEDS:
        check_frozen(repair, output)
        model = original.load_model(repair, "repair", seed)
        cache = FrozenCache(model, *unique_tokens(train))
        arrays = dict(np.load(output / f"descriptors_{seed}.npz", allow_pickle=False))
        if not np.array_equal(arrays["writer_keys"], cache.keys.numpy()):
            raise AssertionError("Writer encoding differs from frozen descriptor preparation")
        geometries[str(seed)] = geometry(cache)
        references, local_invariance = {}, {}
        parities[str(seed)], invariances[str(seed)] = {}, {}
        for name, dataset in data.items():
            config = manifest["configs"][0]
            reference = execute(cache, dataset, config, descriptor_for(arrays, config))
            parities[str(seed)][name] = native_parity(cache, dataset, reference)
            references[name] = reference
            inv = reference_invariance(cache, dataset, reference, name == "fresh_bridge")
            local_invariance[name] = inv
            invariances[str(seed)][name] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                           for key, value in inv.items()}
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir()
        prior_ceiling = {}
        permutation_worlds = 0
        permutation_probes, permutation_scores = 0, 0
        for config in manifest["configs"]:
            desc = descriptor_for(arrays, config)
            trace_arrays = {}
            for name, dataset in data.items():
                trace = references[name] if config["id"] == "coarse" else execute(
                    cache, dataset, config, desc, references[name])
                row = {"seed": seed, "cohort": name, **config, **summarize(trace, dataset, config)}
                all_rows.append(row)
                permutation_worlds += permutation_check(
                    cache, dataset, config, desc, references[name], permutation, trace)
                permutation_probes += int(trace["counts"][:16, :, 1].sum())
                permutation_scores += int(trace["counts"][:16, :, 0].sum())
                for field, value in trace.items():
                    trace_arrays[f"{name}__{field}"] = value
                if config["family"] != "dense":
                    bound, bound_trace = coverage_bounds(cache, dataset, desc, config["depth"], local_invariance[name])
                    eligible = bound_trace["eligible"].astype(bool)
                    group_capture = trace["ref_gold_capture"][:, bound["hop"]].reshape(-1, 4).mean(-1)
                    budget_index = 0 if config["policy"] == "sibling" and config["depth"] == 0 else 1
                    budget_ceiling = bound_trace["ceilings"][:, budget_index]
                    if np.any(group_capture[eligible] > budget_ceiling[eligible]):
                        raise AssertionError("Fixed-state candidate capture exceeds its prefix budget ceiling")
                    stable_code = np.all(bound_trace["target_codes"] == bound_trace["target_codes"][:, :1], axis=1)
                    allocation_ceiling = np.where(stable_code, bound_trace["ceilings"][:, 0], budget_ceiling) \
                        if config["policy"] == "two_coarse" else budget_ceiling
                    if np.any(group_capture[eligible] > allocation_ceiling[eligible]):
                        raise AssertionError("Fixed-state capture exceeds the policy allocation ceiling")
                    bound.update(eligible_worlds=int(eligible.sum()) * 4,
                                 eligible_reference_capture=float(group_capture[eligible].mean()) if eligible.any() else None,
                                 policy_allocation_ceiling=float(allocation_ceiling[eligible].mean()) if eligible.any() else None)
                    bound_trace["reference_group_capture"] = group_capture
                    bound_trace["policy_allocation_ceiling"] = allocation_ceiling
                    bounds_rows.append({"seed": seed, "cohort": name, **config, **bound})
                    if config["policy"] == "two_coarse":
                        key = (name, config["family"], config["hash_seed"])
                        last = prior_ceiling.get(key)
                        now = bound_trace["ceilings"]
                        if last is not None and np.any(now > last):
                            raise AssertionError("Nested-prefix coverage ceiling increased")
                        prior_ceiling[key] = now
                    for field, value in bound_trace.items():
                        trace_arrays[f"{name}__bound_{field}"] = value
            np.savez_compressed(seed_dir / f"{config['id']}.npz", **trace_arrays)
            if config["family"] in ("coarse", "dense") or config["depth"] == 32:
                group_rows = all_rows[-len(data):]
                print(json.dumps({"seed": seed, "config": config["id"],
                    "answer_mean": sum(x["answer_accuracy"] for x in group_rows) / len(group_rows),
                    "both_mean": sum(x["both_support_accuracy"] for x in group_rows) / len(group_rows)}), flush=True)
        work[str(seed)] = {"dense_audit_new_score_evaluations": cache.dense_audit_new_scores,
                          "cached_reader_inputs": len(cache.readers),
                          "cached_reader_projections": len(cache.projected_readers),
                          "permutation_worlds": permutation_worlds,
                          "permutation_additional_probes": permutation_probes,
                          "permutation_additional_scores": permutation_scores,
                          "reference_audit_probes": sum(row["reference_audit_counters"]["probes"]
                                                       for row in all_rows if row["seed"] == seed)}
        # Rebuild and compare all state tensors, including the frozen codebook.
        disk = original.load_model(repair, "repair", seed).state_dict()
        if any(not torch.equal(tensor, disk[name]) for name, tensor in model.state_dict().items()):
            raise AssertionError("Model state mutated during diagnostic")
        expected_arrays = manifest["descriptor_receipts"][str(seed)]["arrays"]
        if {name: array_digest(value) for name, value in arrays.items()} != expected_arrays:
            raise AssertionError("In-memory descriptors changed during diagnostic")
    if {name: vb.digest_data(value) for name, value in data.items()} != data_hashes_before:
        raise AssertionError("In-memory evaluation data changed during diagnostic")
    write_json(output / "results.json", {"rows": all_rows, "counter_names": COUNTERS})
    write_json(output / "bounds.json", {"rows": bounds_rows})
    write_json(output / "geometry.json", geometries)
    write_json(output / "integrity.json", {"native_parity": parities, "reference_invariance": invariances,
               "permutation": permutation.tolist(), "inverse_permutation": np.argsort(permutation).tolist(),
               "offline_work": work, "model_states_unchanged": True,
               "evaluation_data_unchanged": True, "descriptors_unchanged": True})
    check_frozen(repair, output)
    files = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*"))
             if path.is_file() and path.name != "complete.json" and path.suffix not in (".log", ".txt")}
    write_json(output / "complete.json", {"status": "complete", "completed_unix": time.time(),
               "source_sha256": source_hashes(), "original_checkpoints_sha256": checkpoint_hashes(repair),
               "files": files, "cells": len(all_rows), "training_performed": False})
    print(json.dumps({"status": "complete", "cells": len(all_rows), "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "evaluate"))
    parser.add_argument("--repair-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    {"prepare": prepare, "evaluate": evaluate}[args.operation](
        args.repair_output.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
