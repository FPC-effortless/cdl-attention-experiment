"""Audit one completed run and sample its saved best checkpoint; never train.

The optional memory intervention uses validation only, on a matched CPU pair.
It is an inference intervention, not a retrained ablation or attention baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path, required=True):
    path = Path(path)
    if not path.exists() and not required:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_discarded_histories(run, interruptions, effective_targets):
    """Verify archived abandoned branches without running the neural model.

    Canonical metrics may legitimately reuse update numbers after recovery. The
    exact retained prefix and explicit recovery marker identify the surviving
    branch; original and discarded files preserve the abandoned observations.
    """
    run = Path(run).resolve()
    canonical_raw = (run / "metrics.jsonl").read_bytes()
    canonical = [(line, json.loads(line)) for line in canonical_raw.splitlines(keepends=True) if line.strip()]
    entries = []
    archive_names = []

    def safe_file(name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe archived-metrics path")
        path = run / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(run):
            raise ValueError("Archived-metrics file missing or outside run")
        return path

    for event in interruptions:
        if event.get("event") != "intentional_runtime_stop":
            continue
        archive_name, tail_name = event["full_original_metrics_archive"], event["discarded_tail_file"]
        archived_path, tail_path = safe_file(archive_name), safe_file(tail_name)
        archive_names.extend((archive_name, tail_name))
        original = [(line, json.loads(line)) for line in archived_path.read_bytes().splitlines(keepends=True) if line.strip()]
        retained_step = event["last_retained_checkpoint_step"]
        discarded = [(line, row) for line, row in original
                     if type(row.get("step")) is int and row["step"] > retained_step]
        retained = [(line, row) for line, row in original
                    if not (type(row.get("step")) is int and row["step"] > retained_step)]
        tail_bytes = b"".join(line for line, _ in discarded)
        tail_records = [row for _, row in discarded if row.get("event") == "train"]
        last_step = max((row["step"] for row in tail_records), default=retained_step)
        minimum_updates = max(0, last_step - retained_step)
        markers = [i for i, (_, row) in enumerate(canonical)
                   if row.get("event") == "trajectory_recovery"
                   and row.get("interruption_session") == event["session_id"]
                   and row.get("resume_step") == retained_step
                   and row.get("discarded_tail_file") == tail_name]
        identities_match = (sha256(archived_path) == event["full_original_metrics_sha256"]
                            and sha256(tail_path) == event["discarded_tail_sha256"])
        retained_prefix_matches = (len(markers) == 1
                                   and b"".join(line for line, _ in canonical[:markers[0]])
                                   == b"".join(line for line, _ in retained))
        tail_matches = tail_path.read_bytes() == tail_bytes
        declared_counts_match = (len(tail_records) == event["discarded_logged_train_records"]
                                 and last_step == event["last_logged_discarded_step"]
                                 and minimum_updates == event["minimum_discarded_completed_updates"])
        retained_train = [row for _, row in retained
                          if row.get("event") == "train" and row.get("step") == retained_step]
        recorded_seconds = None
        if retained_train and tail_records:
            recorded_seconds = tail_records[-1]["training_seconds"] - retained_train[-1]["training_seconds"]
        entries.append({
            "interruption_session": event["session_id"],
            "archive": archive_name, "archive_sha256": sha256(archived_path),
            "discarded_tail": tail_name, "discarded_tail_sha256": sha256(tail_path),
            "retained_checkpoint_step": retained_step, "last_logged_discarded_step": last_step,
            "discarded_logged_train_records": len(tail_records),
            "minimum_documented_discarded_completed_updates": minimum_updates,
            "minimum_documented_discarded_target_presentations": minimum_updates * effective_targets,
            "recorded_optimizer_seconds_between_retained_and_last_discarded_log": recorded_seconds,
            "hashes_match_interruption_receipt": identities_match,
            "discarded_tail_matches_exact_original_records": tail_matches,
            "canonical_retained_prefix_and_recovery_marker_match": retained_prefix_matches,
            "declared_discarded_counts_match": declared_counts_match,
            "passed": identities_match and tail_matches and retained_prefix_matches and declared_counts_match,
            "additional_uncheckpointed_work": "unknown",
        })
    return {
        "passed": all(entry["passed"] for entry in entries),
        "entries": entries, "archive_names": sorted(set(archive_names)),
        "minimum_documented_discarded_completed_updates": sum(e["minimum_documented_discarded_completed_updates"] for e in entries),
        "minimum_documented_discarded_target_presentations": sum(e["minimum_documented_discarded_target_presentations"] for e in entries),
        "unlogged_work_at_other_interruptions": "unknown",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--memory-intervention", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--sample-characters", type=int, default=384)
    args = parser.parse_args()
    run, data = args.run_dir.resolve(), args.data_dir.resolve()
    output = (args.output_dir or run / "postfit_audit").resolve()
    if not (run / "result.json").exists():
        raise SystemExit("Audit not started: result.json does not exist; wait for completed training.")
    result = read_json(run / "result.json")
    if result.get("status") != "completed" or not result.get("test_evaluated"):
        raise SystemExit("Audit not started: run is not completed with its selected-checkpoint final test.")
    if args.eval_batch_size < 1 or args.sample_characters < 1:
        raise SystemExit("Evaluation batch size and sample length must be positive.")

    # No Torch import or expensive inference occurs before the completed guard.
    import torch
    from data import CharTokenizer, load_split
    from sample import generate, load_model
    from train import evaluate

    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(run / "run_config.json")
    dataset = read_json(data / "dataset.json")
    metrics = read_jsonl(run / "metrics.jsonl")
    sessions = read_jsonl(run / "sessions.jsonl")
    interruptions = read_jsonl(run / "interruption_events.jsonl", required=False)
    cfg = manifest["training_config"]
    source_dir = Path(__file__).resolve().parent
    checks = []
    effective = cfg["batch_size"] * cfg["gradient_accumulation"] * cfg["context_length"]
    discarded_history = audit_discarded_histories(run, interruptions, effective)

    def check(name, condition, **evidence):
        checks.append({"name": name, "passed": bool(condition), **evidence})

    receipt = {
        "scope": "post-fit artifact and trajectory audit; no additional training or model-ranking claim",
        "run_directory": str(run), "environment": {"torch": str(torch.__version__),
        "device": "cpu", "threads": args.threads},
        "audit_script_sha256": sha256(__file__),
        "checks": checks, "sessions": sessions, "interruption_events": interruptions,
        "discarded_history_audit": discarded_history,
        "accounting": {
            "reported_presentations_scope": "restored training trajectory",
            "reported_training_seconds_scope": "recorded optimizer-attempt time on restored trajectory; excludes evaluation and other overhead",
            "discarded_uncheckpointed_work": "unknown" if interruptions else "no interruption receipt recorded; not independently observed",
            "total_physical_compute_across_interruptions": "not established" if interruptions else "not independently metered",
            "session_start_records": len(sessions),
            "minimum_documented_discarded_completed_updates": discarded_history["minimum_documented_discarded_completed_updates"],
            "minimum_documented_discarded_target_presentations": discarded_history["minimum_documented_discarded_target_presentations"],
            "total_physical_target_presentations_lower_bound": result["training_targets_seen"] + discarded_history["minimum_documented_discarded_target_presentations"],
            "note": "Restarting a saved model/optimizer/RNG trajectory is not an independent fit. No exact total physical work is inferred across uncheckpointed interruption.",
        },
    }

    def save_receipt():
        receipt.update(passed=sum(c["passed"] for c in checks), total=len(checks),
                       status="PASS" if all(c["passed"] for c in checks) else "FAIL",
                       elapsed_seconds=time.perf_counter() - started)
        (output / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    input_names = ["run_config.json", "result.json", "metrics.jsonl", "sessions.jsonl",
                   "tokenizer.json", "initial_evaluation.json", "best.pt", "resume.pt"]
    input_names += [name for name in ("interruption_events.jsonl", "allocator_recovery_receipt.json",
                                      "launcher_sessions.jsonl", "launcher_model_config.json")
                    if (run / name).exists()]
    input_names += discarded_history["archive_names"]
    receipt["input_sha256"] = {name: sha256(run / name) for name in sorted(set(input_names))}
    check("discarded_trajectory_archives_and_recovery_verified", discarded_history["passed"],
          interruption_archives=len(discarded_history["entries"]),
          minimum_discarded_updates=discarded_history["minimum_documented_discarded_completed_updates"],
          minimum_discarded_presentations=discarded_history["minimum_documented_discarded_target_presentations"])
    check("source_identity_matches_frozen_run", all(sha256(source_dir / k) == v for k, v in manifest["source_sha256"].items()),
          source_sha256=manifest["source_sha256"])
    check("dataset_and_tokenizer_identity", manifest["dataset"] == dataset
          and sha256(data / "input.txt") == dataset["input_sha256"]
          and sha256(data / "tokenizer.json") == dataset["tokenizer_sha256"]
          and sha256(run / "tokenizer.json") == dataset["tokenizer_sha256"])
    split_ok = dataset["split_policy"] == "contiguous 90% train / 5% validation / 5% test"
    previous_end = 0
    for name in ("train", "val", "test"):
        split = dataset["splits"][name]
        split_ok &= split["start_character_inclusive"] == previous_end
        split_ok &= split["end_character_exclusive"] - previous_end == split["characters"]
        split_ok &= split["available_next_character_targets"] == split["characters"] - 1
        split_ok &= sha256(data / (name + ".bin")) == split["binary_sha256"]
        previous_end = split["end_character_exclusive"]
    tokenizer_data = read_json(data / "tokenizer.json")
    raw_text = (data / "input.txt").read_text(encoding="utf-8")
    split_ok &= previous_end == len(raw_text) == dataset["characters"]
    split_ok &= tokenizer_data["chars"] == sorted(set(raw_text[:dataset["splits"]["train"]["end_character_exclusive"]]))
    check("training_only_vocabulary_and_exact_splits", split_ok and dataset["vocab_size"] == len(tokenizer_data["chars"]),
          split_metadata=dataset["splits"], vocab_size=dataset["vocab_size"])

    best_path = run / "best.pt"
    best_hash = sha256(best_path)
    model, tokenizer, best = load_model(best_path, device="cpu", threads=args.threads)
    resume = torch.load(run / "resume.pt", map_location="cpu", weights_only=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    check("actual_parameters_and_configuration", 9_000_000 <= parameter_count <= 11_000_000
          and parameter_count == manifest["parameters"] == manifest["trainable_parameters"] == result["parameters"]
          and best["model_config"] == manifest["model_config"] == resume["model_config"], actual_parameters=parameter_count)
    check("checkpoint_parameters_are_finite", all(bool(torch.isfinite(p).all()) for p in model.parameters()))
    check("best_checkpoint_hash_and_metadata", best_hash == result["best_checkpoint_sha256"] == resume["best_checkpoint_sha256"]
          and best["source_sha256"] == manifest["source_sha256"] == resume["source_sha256"]
          and best["dataset_sha256"] == dataset["input_sha256"] == resume["dataset_sha256"]
          and best["tokenizer_sha256"] == dataset["tokenizer_sha256"] == resume["tokenizer_sha256"]
          and best["tokenizer"] == tokenizer_data, best_checkpoint_sha256=best_hash)
    check("completed_trajectory_matches_resume", resume["checkpoint_status"] == "resumable"
          and resume["training_config"] == cfg
          and all(resume[k] == result[k] for k in ("step", "training_targets_seen", "attempted_updates", "skipped_updates", "training_seconds", "best_step", "best_val_nll")))

    validations = [m for m in metrics if m.get("event") == "validation"]
    trains = [m for m in metrics if m.get("event") == "train"]
    skips = [m for m in metrics if m.get("event") == "fp16_overflow_skipped"]
    lowest = min(validations, key=lambda m: m["mean_nll"])
    check("selected_checkpoint_matches_strict_validation_best", best["step"] == result["best_step"] == lowest["step"]
          and best["val"]["mean_nll"] == result["best_val_nll"] == lowest["mean_nll"]
          and best["step"] <= result["step"] and best["training_targets_seen"] == result["best_checkpoint_training_targets"] == lowest["targets_seen"],
          selected_step=best["step"], selected_validation_nll=best["val"]["mean_nll"], validation_events=len(validations))
    stop_ok = result["planned_max_steps"] == cfg["max_steps"]
    if result["stop_reason"] == "max_steps":
        stop_ok &= result["step"] == cfg["max_steps"]
    elif result["stop_reason"] == "validation_early_stopping":
        stop_ok &= cfg["patience"] > 0 and result["step"] >= cfg["minimum_steps"] and resume["bad_evals"] >= cfg["patience"]
    else:
        stop_ok = False
    check("stopping_reason_matches_declared_budget", stop_ok, stop_reason=result["stop_reason"], completed_updates=result["step"])
    exposure_ok = effective == cfg["effective_targets_per_step"]
    exposure_ok &= result["attempted_updates"] == result["step"] + result["skipped_updates"]
    exposure_ok &= result["training_targets_seen"] == result["attempted_updates"] * effective
    exposure_ok &= math.isclose(result["train_target_equivalent_passes"], result["training_targets_seen"] / dataset["splits"]["train"]["available_next_character_targets"], rel_tol=1e-12)
    exposure_ok &= all(m["targets_seen"] == m["attempted_updates"] * effective for m in trains)
    check("restored_trajectory_presentations_are_consistent", exposure_ok,
          effective_targets_per_update=effective, restored_trajectory_targets=result["training_targets_seen"],
          selected_checkpoint_targets=best["training_targets_seen"], equivalent_training_passes=result["train_target_equivalent_passes"])
    no_skips = result["skipped_updates"] == 0 and len(skips) == 0
    check("skip_accounting_matches_precision_and_claim", (cfg["precision"] != "fp32" or no_skips)
          and result["skipped_updates"] == len(skips), precision=cfg["precision"], skipped_updates=result["skipped_updates"], skip_events=len(skips))
    check("logged_training_steps_are_ordered", all(a["step"] < b["step"] for a, b in zip(trains, trains[1:]))
          and all(m["step"] <= result["step"] for m in trains))

    summary_ok = True
    for split, values in (("val", best["val"]), ("test", result["test"])):
        summary_ok &= values["targets"] == dataset["splits"][split]["available_next_character_targets"]
        summary_ok &= math.isfinite(values["mean_nll"]) and values["mean_nll"] >= 0
        summary_ok &= math.isclose(values["character_perplexity"], math.exp(values["mean_nll"]), rel_tol=1e-12)
        summary_ok &= math.isclose(values["bits_per_character"], values["mean_nll"] / math.log(2), rel_tol=1e-12)
        summary_ok &= 0 <= values["next_character_accuracy"] <= 1
    check("heldout_summaries_and_denominators_are_consistent", summary_ok,
          selected_validation=best["val"], recorded_final_test=result["test"])
    check("initial_evaluation_receipt_matches_result", result["initial_evaluation"] == read_json(run / "initial_evaluation.json"))
    receipt["quality_observations"] = {
        "initial_validation_nll": result["initial_evaluation"]["val"]["mean_nll"],
        "selected_validation_nll": best["val"]["mean_nll"],
        "unigram_validation_nll": result["unigram_validation"]["mean_nll"],
        "recorded_final_test": result["test"],
        "test_recomputed_by_audit": False,
        "selection_uses_validation_only": True,
    }
    del resume
    save_receipt()
    if not all(c["passed"] for c in checks):
        raise SystemExit("Audit metadata checks failed; see audit.json. No sample or intervention was run.")

    samples = []
    for prompt, seed in (("ROMEO:\n", 42), ("First Citizen:\n", 43)):
        text, stats = generate(model, tokenizer, prompt, args.sample_characters, 0.8, 0, seed, "cpu")
        samples.append({"prompt": prompt, "seed": seed, "temperature": 0.8, "top_k": 0,
                        "text": text, "stats": stats})
    short_n = min(64, args.sample_characters)
    repeated, _ = generate(model, tokenizer, samples[0]["prompt"], short_n, 0.8, 0, 42, "cpu")
    check("saved_best_checkpoint_generates_reproducibly", repeated == samples[0]["text"][:len(samples[0]["prompt"]) + short_n]
          and all(len(s["text"]) == len(s["prompt"]) + args.sample_characters for s in samples))
    (output / "samples.json").write_text(json.dumps({"checkpoint_sha256": best_hash, "checkpoint_step": best["step"], "samples": samples}, indent=2) + "\n")
    (output / "samples.txt").write_text("\n\n".join(f"=== Prompt {i+1}; seed {s['seed']}; temperature 0.8 ===\n{s['text']}" for i, s in enumerate(samples)) + "\n")
    receipt["sampling"] = {"checkpoint_step": best["step"], "checkpoint_sha256": best_hash,
                           "prompts_fixed_in_audit_source_before_completion": True,
                           "samples_json_sha256": sha256(output / "samples.json"),
                           "samples_text_sha256": sha256(output / "samples.txt")}

    if args.memory_intervention:
        class MemoryDisabled(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, tokens, state=None):
                return self.base(tokens, state, memory_enabled=False)
        validation_tokens = load_split(data, "val")
        native = evaluate(model, validation_tokens, cfg["context_length"], args.eval_batch_size, torch.device("cpu"), "fp32")
        disabled = evaluate(MemoryDisabled(model).eval(), validation_tokens, cfg["context_length"], args.eval_batch_size, torch.device("cpu"), "fp32")
        diagnostic = {
            "scope": "one validation-only matched CPU inference intervention on selected checkpoint; no retraining or attention comparator",
            "checkpoint_step": best["step"], "checkpoint_sha256": best_hash,
            "recorded_native_validation": best["val"], "native_cpu_replay": native,
            "memory_disabled_cpu": disabled,
            "disabled_minus_native_nll": disabled["mean_nll"] - native["mean_nll"],
            "disabled_minus_native_bits_per_character": disabled["bits_per_character"] - native["bits_per_character"],
            "native_replay_minus_recorded_nll": native["mean_nll"] - best["val"]["mean_nll"],
            "interpretation": "A positive disabled-minus-native NLL means this checkpoint loses validation prediction quality when its trained board contribution is removed. This is checkpoint dependence, not evidence that the architecture beats a separately trained local-only model or attention.",
            "device": "cpu", "precision": "fp32", "threads": args.threads,
            "test_set_evaluated": False,
        }
        (output / "memory_intervention.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        check("selected_native_validation_replays_within_tolerance", abs(diagnostic["native_replay_minus_recorded_nll"]) <= 1e-4,
              nll_difference=diagnostic["native_replay_minus_recorded_nll"], tolerance=1e-4)
        receipt["memory_intervention"] = diagnostic
    save_receipt()
    print(json.dumps({"status": receipt["status"], "passed": receipt["passed"], "total": receipt["total"],
                      "selected_step": best["step"], "output_directory": str(output)}, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
