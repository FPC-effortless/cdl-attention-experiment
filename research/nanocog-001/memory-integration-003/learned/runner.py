"""Pinned local CPU experiment and strict loader for the actual native adapter."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
NANO = HERE.parents[1]
if str(NANO) not in sys.path:
    sys.path.insert(0, str(NANO))
from nanocog.memory.learned_archive import FactualAdapter, UNKNOWN, execute_operation, read_ledger
from dataset import canonical, sha

NATIVE_SHA = "d8bab3d1d99a43dfb6cca2257f2caae4d3a7282d62b079c92874ce736a44ec3c"
SOURCE_SHAS = {
    "model.py": "f15e7d5430bd1df83a1264ebbec63576b818733386d6fb0f1d8c1e2a930370f9",
    "base_model.py": "f96bd967b4bc4ece9d2750a5cf1afdf1cc819a6b007e2d638af633376c7eccaa",
    "scripted_core.py": "6c00b2c4425a5309927fb3aba4e53add670e79c88a7b051fbe963b6b060f99e3",
}
SEEDS = (17, 29, 41)
ARMS = ("control", "candidate")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_adapter(kit_path: str | Path, checkpoint_path: str | Path | None = None,
                 seed: int = 17) -> FactualAdapter:
    """Admit the native kit and optionally strict-load a full fitted adapter.

    The native kit manifest supplies source hashes, itself bound by the selected
    checkpoint's embedded source_sha256. Arbitrary state dictionaries and partial
    native checkpoints are rejected. No prediction labels enter this loader.
    """
    kit_path = Path(kit_path)
    native_path = kit_path / "COG_CONNECTED_1M.pt"
    if file_sha(native_path) != NATIVE_SHA:
        raise ValueError("native checkpoint digest is not the selected trained model")
    payload = torch.load(native_path, map_location="cpu", weights_only=True)
    observed_sources = {name: file_sha(kit_path / name) for name in SOURCE_SHAS}
    if observed_sources != SOURCE_SHAS:
        raise ValueError("native source differs from the prospectively admitted code")
    embedded = payload["source_sha256"]
    if not isinstance(embedded, dict):
        raise ValueError("native checkpoint does not bind source files")
    for name, digest in observed_sources.items():
        matches = [v for k, v in embedded.items() if Path(k).name == name]
        if matches != [digest]:
            raise ValueError("native source differs from checkpoint-bound source: " + name)
    module_name = "memory003_admitted_native"
    if module_name not in sys.modules:
        # The distributed kit has no package initializer. A synthetic package
        # permits relative imports without replacing any application's model.py.
        import types
        package = types.ModuleType(module_name); package.__path__ = [str(kit_path)]
        sys.modules[module_name] = package
        source_spec = importlib.util.spec_from_file_location(module_name + ".model", kit_path / "model.py")
        module = importlib.util.module_from_spec(source_spec)
        sys.modules[source_spec.name] = module
        source_spec.loader.exec_module(module)
    module = sys.modules[module_name + ".model"]
    native = module.ConnectedCOGTinyLM(module.ConnectedConfig(**payload["model_config"]))
    native.load_state_dict(payload["model"], strict=True)
    if sum(p.numel() for p in native.parameters()) != 993574:
        raise ValueError("unexpected native parameter count")
    torch.manual_seed(seed)
    admission = {"native_checkpoint_sha256": NATIVE_SHA, "native_source_sha256": observed_sources,
                 "native_seed": payload["seed"], "native_update": payload["step"],
                 "native_config": payload["model_config"], "tokenizer": payload["tokenizer"]}
    adapter = FactualAdapter(native, payload["tokenizer"]["chars"], admission=admission)
    if checkpoint_path is not None:
        fitted = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (fitted["schema"] != "memory003-factual-checkpoint-v1" or
                fitted["admission"] != admission or fitted["model_config"] != payload["model_config"] or
                fitted["chars"] != list(adapter.chars)):
            raise ValueError("factual checkpoint admission mismatch")
        adapter.load_state_dict(fitted["state_dict"], strict=True)
        for name, value in native.state_dict().items():
            if not torch.equal(value, payload["model"][name]):
                raise ValueError("factual checkpoint changed native language weights")
        if adapter.model_identity() != fitted["model_identity"]:
            raise ValueError("factual checkpoint model identity mismatch")
    return adapter.eval()


def source_lock() -> dict:
    paths = [HERE / "PROTOCOL.md", HERE / "DATA_MANIFEST.json", HERE / "dataset.py",
             HERE / "runner.py", NANO / "nanocog/memory/learned_archive.py"]
    return {str(p.relative_to(NANO)): file_sha(p) for p in paths}


def load_corpus(private: Path) -> dict:
    raw = (private / "corpus.json").read_bytes()
    manifest = json.loads((HERE / "DATA_MANIFEST.json").read_text())
    if hashlib.sha256(raw).hexdigest() != manifest["corpus_sha256"]:
        raise ValueError("corpus changed after prospective data lock")
    return json.loads(raw)["data"]


def texts_in(pairs: list[dict]) -> set[str]:
    return ({e["text"] for p in pairs for w in p["worlds"] for e in w["events"]} |
            {w["query"] for p in pairs for w in p["worlds"]})


def timing(start: tuple[float, float]) -> dict:
    return {"wall_seconds": time.perf_counter() - start[0], "cpu_seconds": time.process_time() - start[1]}


def clock() -> tuple[float, float]:
    return time.perf_counter(), time.process_time()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cache_features(adapter: FactualAdapter, texts: set[str], path: Path,
                   reuse_path: Path | None = None, already_cpu: float = 0.0) -> dict:
    started = clock()
    ordered = sorted(texts, key=lambda s: (len(s), s))
    lengths = torch.tensor([len(t) for t in ordered], dtype=torch.long)
    features = torch.zeros(len(ordered), int(lengths.max()), 122)
    existing = {}
    if reuse_path is not None and reuse_path.exists():
        reuse = torch.load(reuse_path, weights_only=True)
        if reuse["native_sha"] != NATIVE_SHA:
            raise ValueError("feature cache native identity mismatch")
        existing = {t: i for i, t in enumerate(reuse["texts"])}
    missing = []
    for i, text in enumerate(ordered):
        if text in existing:
            j = existing[text]
            features[i, :len(text)] = reuse["features"][j, :len(text)]
        else:
            missing.append(i)
    for start in range(0, len(missing), 64):
        idx = missing[start:start + 64]
        extracted, lens = adapter.native_features([ordered[i] for i in idx])
        for j, i in enumerate(idx):
            features[i, :int(lens[j])] = extracted[j, :int(lens[j])]
        if already_cpu + timing(started)["cpu_seconds"] > 900:
            raise RuntimeError("native extraction CPU cap exceeded; study incomplete")
    data = {"native_sha": NATIVE_SHA, "texts": ordered, "texts_sha256": sha(ordered),
            "features": features, "lengths": lengths}
    torch.save(data, path)
    stats = {**timing(started), "unique_texts": len(ordered), "new_texts": len(missing),
             "characters": sum(map(len, ordered)), "file_bytes": path.stat().st_size,
             "sha256": file_sha(path)}
    save_json(path.with_suffix(".json"), stats)
    return stats


class Batches:
    def __init__(self, data: dict, cache: dict, seed: int, arm: str):
        self.families = data[arm + "_train"]
        self.rng = random.Random(99571 + seed)
        self.views = random.Random(31199 + seed)
        self.arm = arm
        self.lookup = {t: i for i, t in enumerate(cache["texts"])}
        self.cache = cache

    def pair(self, family: int):
        return self.families[family] if self.arm == "control" else self.families[family * 3 + self.views.randrange(3)]

    def next(self):
        indices, event_labels, query_labels = [], [], []
        for _ in range(96):
            family = self.rng.randrange(512)
            world = self.rng.randrange(2)
            pair = self.pair(family)
            event = pair["worlds"][world]["events"][self.rng.randrange(pair["length"])]
            indices.append(self.lookup[event["text"]]); event_labels.append(event["operation"])
        for _ in range(32):
            pair = self.pair(self.rng.randrange(512))
            world = pair["worlds"][self.rng.randrange(2)]
            indices.append(self.lookup[world["query"]]); query_labels.append(world["route"])
        indices = torch.tensor(indices)
        lengths = self.cache["lengths"][indices]
        return (self.cache["features"][indices, :int(lengths.max())], lengths,
                torch.tensor(event_labels), torch.tensor(query_labels))


def update(adapter: FactualAdapter, optimizer, batch):
    features, lengths, operations, queries = batch
    writer_logits, query_logits = adapter(features, lengths)
    dictionary = torch.arange(13)
    labels = torch.where(dictionary < 8, dictionary, 8)
    loss = (F.cross_entropy(writer_logits[:96], operations) +
            F.cross_entropy(query_logits[96:], queries) +
            F.cross_entropy(adapter.answer_logits(dictionary), labels))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    return float(loss.detach())


def optimizer_for(adapter):
    return torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad], lr=.003,
                            betas=(.9, .999), eps=1e-8, weight_decay=0)


def calibrate(kit: Path, private: Path):
    started = clock()
    if (private / "CALIBRATION.json").exists():
        raise ValueError("a calibration already exists; do not recalibrate after results")
    lock = {"sources": source_lock(), "native_sha256": NATIVE_SHA,
            "corpus_sha256": json.loads((HERE / "DATA_MANIFEST.json").read_text())["corpus_sha256"],
            "pinned_unix_time": time.time()}
    save_json(private / "LOCK.json", lock); save_json(HERE / "LOCK.json", lock)
    data = load_corpus(private)
    adapter = load_adapter(kit, seed=1009)
    extraction = cache_features(adapter, texts_in(data["candidate_train"]), private / "train_features.pt")
    cache = torch.load(private / "train_features.pt", weights_only=True)
    batches = Batches(data, cache, 1009, "candidate")
    optimizer = optimizer_for(adapter)
    adapter.train()
    for _ in range(32):
        update(adapter, optimizer, batches.next())
    timed = clock()
    for _ in range(64):
        update(adapter, optimizer, batches.next())
    measure = timing(timed)
    per_update = measure["cpu_seconds"] / 64
    updates = min(2048, int(540 / (6 * per_update) / 128) * 128)
    result = {"warmup_updates": 32, "timed_updates": 64, "timing": measure,
              "cpu_seconds_per_update": per_update, "shared_scored_updates": updates,
              "predicted_fit_cpu_seconds": 6 * updates * per_update,
              "extraction": extraction, "total": timing(started),
              "accuracy_used_for_budget": False, "calibration_checkpoint_discarded": True,
              "lock_sha256": file_sha(private / "LOCK.json")}
    save_json(private / "CALIBRATION.json", result); save_json(HERE / "CALIBRATION.json", result)
    print(json.dumps(result), flush=True)
    if updates < 128:
        raise RuntimeError("minimum balanced fit does not fit CPU budget")


def fit(kit: Path, private: Path):
    started = clock()
    lock = json.loads((private / "LOCK.json").read_text())
    if lock["sources"] != source_lock():
        raise ValueError("experiment sources changed after calibration lock")
    calibration = json.loads((private / "CALIBRATION.json").read_text())
    updates = calibration["shared_scored_updates"]
    if not 128 <= updates <= 2048:
        raise ValueError("invalid calibrated shared update count")
    data = load_corpus(private)
    cache = torch.load(private / "train_features.pt", weights_only=True)
    (private / "checkpoints").mkdir(exist_ok=True)
    records, fit_cpu = [], 0.0
    for seed in SEEDS:
        for arm in ARMS:
            adapter = load_adapter(kit, seed=seed)
            initial_identity = adapter.model_identity()
            native_before = {n: v.clone() for n, v in adapter.native.state_dict().items()}
            optimizer, batches = optimizer_for(adapter), Batches(data, cache, seed, arm)
            adapter.train()
            run_start = clock()
            for step in range(updates):
                loss = update(adapter, optimizer, batches.next())
                if fit_cpu + timing(run_start)["cpu_seconds"] > 600:
                    raise RuntimeError("combined scored fitting cap exceeded; study incomplete")
            run_time = timing(run_start); fit_cpu += run_time["cpu_seconds"]
            adapter.eval()
            if any(not torch.equal(v, native_before[n]) for n, v in adapter.native.state_dict().items()):
                raise RuntimeError("fitting mutated the frozen native model")
            path = private / "checkpoints" / f"{arm}-{seed}.pt"
            payload = {"schema": "memory003-factual-checkpoint-v1", "arm": arm, "seed": seed,
                       "updates": updates, "state_dict": adapter.state_dict(),
                       "model_config": asdict(adapter.native.config), "chars": list(adapter.chars),
                       "admission": adapter.admission, "model_identity": adapter.model_identity(),
                       "initial_identity": initial_identity, "training_timing": run_time,
                       "source_lock": lock, "calibration_sha256": file_sha(private / "CALIBRATION.json"),
                       "native_unchanged": True}
            torch.save(payload, path)
            record = {"arm": arm, "seed": seed, "updates": updates, "timing": run_time,
                      "final_training_loss": loss, "file_sha256": file_sha(path),
                      "model_identity": payload["model_identity"], "initial_identity": initial_identity,
                      "native_unchanged": True}
            records.append(record)
            print(json.dumps(record), flush=True)
    result = {"runs": records, "scored_fit_cpu_seconds": fit_cpu, "total": timing(started)}
    save_json(private / "FITS.json", result); save_json(HERE / "FITS.json", result)


@torch.inference_mode()
def predictions(adapter: FactualAdapter, cache: dict, texts: set[str], mode="normal") -> dict:
    lookup = {t: i for i, t in enumerate(cache["texts"])}
    chosen = sorted(texts)
    shuffle = list(range(len(cache["texts"]))); random.Random(60013).shuffle(shuffle)
    result = {}
    for offset in range(0, len(chosen), 128):
        part = chosen[offset:offset + 128]
        idx = torch.tensor([lookup[t] for t in part])
        if mode == "shuffled_features":
            idx = torch.tensor([shuffle[int(i)] for i in idx])
        lengths = cache["lengths"][idx]
        features = cache["features"][idx, :int(lengths.max())]
        if mode == "zero_features":
            features = torch.zeros_like(features)
        writes, queries = adapter(features, lengths)
        for text, write, query in zip(part, writes.argmax(-1).tolist(), queries.argmax(-1).tolist()):
            result[text] = (write, query)
    result["__answer_dictionary__"] = adapter.predict_answers(list(range(13)))
    return result


def evaluate_pairs(pairs: list[dict], predicted: dict, archive_mode="normal") -> list[dict]:
    rows = []
    for pair in pairs:
        operations_ok = ledgers_ok = final_ok = answers_ok = query_ok = True
        wrong_operations = state_errors = invalid = damages = 0
        first_error = None
        answers, routes = [], []
        for world_i, world in enumerate(pair["worlds"]):
            ledger, previous_gold = [UNKNOWN] * 8, [UNKNOWN] * 8
            for event_i, event in enumerate(world["events"]):
                op = predicted[event["text"]][0]
                old = list(ledger)
                status = execute_operation(ledger, op)
                exact = op == event["operation"]
                if not exact:
                    wrong_operations += 1
                    if first_error is None:
                        first_error = {"world": world_i, "event": event_i, "predicted": op,
                                       "required": event["operation"]}
                operations_ok &= exact
                state_equal = ledger == event["ledger"]
                ledgers_ok &= state_equal
                state_errors += not state_equal
                invalid += status == "rejected_invalid"
                damages += sum(old[r] == previous_gold[r] != UNKNOWN and
                               previous_gold[r] == event["ledger"][r] and ledger[r] != event["ledger"][r]
                               for r in range(8))
                previous_gold = event["ledger"]
            final_ok &= ledger == world["events"][-1]["ledger"]
            route = predicted[world["query"]][1]
            if archive_mode == "empty_archive":
                ledger = [UNKNOWN] * 8
            elif archive_mode == "shuffled_values":
                ledger = [((v + 1) % 8) if v < 8 else ((v - 8 + 1) % 4 + 8) if v < 12 else v for v in ledger]
            value = read_ledger(ledger, route)
            answer = predicted["__answer_dictionary__"][value]
            query_ok &= route == world["route"]
            answers_ok &= answer == world["answer"]
            answers.append(answer); routes.append(route)
        rows.append({"lineage": pair["lineage"], "index": pair["index"], "length": pair["length"],
                     "syntax": pair["syntax"], "all_operations": bool(operations_ok),
                     "all_post_event_ledgers": bool(ledgers_ok), "both_final_ledgers": bool(final_ok),
                     "both_answers": bool(answers_ok), "both_query_routes": bool(query_ok),
                     "strict_pair": bool(operations_ok and ledgers_ok and answers_ok),
                     "wrong_operations": wrong_operations, "post_event_ledger_errors": state_errors,
                     "invalid_operations": invalid, "damaged_previously_correct_facts": damages,
                     "operations_count": 2 * pair["length"], "first_error": first_error,
                     "answers": answers, "routes": routes})
    return rows


def wilson(successes: int, n: int) -> list[float]:
    z = 1.959963984540054
    p = successes / n; d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [center - margin, center + margin]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    result = {"pairs": n, "histories": 2 * n,
              "operations": sum(r["operations_count"] for r in rows)}
    for metric in ("all_operations", "all_post_event_ledgers", "both_final_ledgers", "both_answers",
                   "both_query_routes", "strict_pair"):
        success = sum(r[metric] for r in rows)
        result[metric] = {"correct": success, "fraction": success / n,
                          "wilson95": wilson(success, n)}
    for metric in ("wrong_operations", "post_event_ledger_errors", "invalid_operations",
                   "damaged_previously_correct_facts"):
        result[metric] = sum(r[metric] for r in rows)
    result["operation_accuracy"] = 1 - result["wrong_operations"] / result["operations"]
    result["post_event_ledger_accuracy"] = 1 - result["post_event_ledger_errors"] / result["operations"]
    result["first_error_event_counts"] = dict(Counter(str(r["first_error"]["event"])
                                               for r in rows if r["first_error"] is not None))
    return result


def evaluate(kit: Path, private: Path):
    started = clock()
    data = load_corpus(private)
    calibration = json.loads((private / "CALIBRATION.json").read_text())
    all_texts = set().union(*(texts_in(v) for v in data.values()))
    adapter = load_adapter(kit, private / "checkpoints/candidate-17.pt")
    extraction = cache_features(adapter, all_texts, private / "all_features.pt",
                                private / "train_features.pt", calibration["extraction"]["cpu_seconds"])
    cache = torch.load(private / "all_features.pt", weights_only=True)
    reports, raw_rows = [], {}
    for seed in SEEDS:
        for arm in ARMS:
            adapter = load_adapter(kit, private / "checkpoints" / f"{arm}-{seed}.pt")
            run_start = clock()
            normal = predictions(adapter, cache, all_texts)
            record = {"arm": arm, "seed": seed, "panels": {}, "negative_controls": {}}
            for panel in ("control_train", "dev", "familiar_test", "fresh_test"):
                rows = evaluate_pairs(data[panel], normal)
                raw_rows[f"{arm}-{seed}-{panel}"] = rows
                summary = summarize(rows)
                summary["by_length"] = {str(length): summarize([r for r in rows if r["length"] == length])
                                         for length in (8, 24, 64)}
                summary["by_syntax"] = {syntax: summarize([r for r in rows if r["syntax"] == syntax])
                                         for syntax in sorted({r["syntax"] for r in rows})}
                record["panels"][panel] = summary
            for mode in ("zero_features", "shuffled_features"):
                pred = predictions(adapter, cache, texts_in(data["familiar_test"]), mode)
                record["negative_controls"][mode] = summarize(evaluate_pairs(data["familiar_test"], pred))
            for mode in ("empty_archive", "shuffled_values"):
                record["negative_controls"][mode] = summarize(evaluate_pairs(data["familiar_test"], normal, mode))
            record["answer_dictionary"] = normal["__answer_dictionary__"]
            record["timing"] = timing(run_start)
            reports.append(record)
            print(json.dumps({"arm": arm, "seed": seed,
                              "strict_pairs": {k: v["strict_pair"]["correct"] for k, v in record["panels"].items()},
                              "timing": record["timing"]}), flush=True)
    comparisons = []
    for seed in SEEDS:
        entry = {"seed": seed}
        for panel in ("dev", "familiar_test", "fresh_test"):
            before, after = (raw_rows[f"{arm}-{seed}-{panel}"] for arm in ARMS)
            assert [r["lineage"] for r in before] == [r["lineage"] for r in after]
            entry[panel] = {"pairs": len(before),
                            "lost_control_correct": sum(a["strict_pair"] and not b["strict_pair"] for a, b in zip(before, after)),
                            "gained_control_wrong": sum(not a["strict_pair"] and b["strict_pair"] for a, b in zip(before, after)),
                            "both_correct": sum(a["strict_pair"] and b["strict_pair"] for a, b in zip(before, after))}
        comparisons.append(entry)
    result = {"runs": reports, "paired_comparisons": comparisons, "extraction": extraction,
              "total": timing(started), "train_evaluation": "all 512 canonical TRAIN families, not a sampled estimate",
              "native_feature_cache_only": True,
              "durable_inference_evidence": "reported separately; this scorer replays a typed in-memory ledger"}
    save_json(private / "PAIR_ROWS.json", raw_rows)
    save_json(private / "RESULTS.json", result); save_json(HERE / "RESULTS.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("calibrate", "fit", "evaluate"))
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    {"calibrate": calibrate, "fit": fit, "evaluate": evaluate}[args.stage](args.kit, args.private)


if __name__ == "__main__":
    main()
