"""Independent TRAIN-only draw, accounting, and final-checkpoint audit.

The six completed endpoints are checked before Torch is imported. No candidate
is instantiated and no forward pass is performed. Raw parent, final, and
optimizer payloads are inspected sequentially with one Torch thread.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import PARENT, SEEDS, ARMS, parent_checkpoint, sha256

STEPS = 1200
PREFIX = "\nAnswer: "
BOARD_PREFIX = "native.record_board."
DETERMINISTIC_FIELDS = (
    "input_chars", "input_physical_positions", "teacher_decode_chars", "teacher_decode_physical_positions",
    "answer_targets", "generation_prefix_chars", "rehearsal_targets", "rehearsal_physical_positions",
    "write_total", "post_answer_write_total", "first_post_answer_write_total", "query_total",
    "generated_answer_total", "teacher_decoder_calls", "semantic_encoder_calls", "rehearsal_calls",
)


def read(path):
    return json.loads(Path(path).read_text())


def create(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(HERE.parent.resolve()))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_trace(path, *, updates=None, allow_partial=False):
    rows, partial_bytes = [], 0
    lines = Path(path).read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            require(allow_partial and index == len(lines) - 1 and not line.endswith(b"\n"), "malformed training trace")
            partial_bytes = len(line)
            break
        require(row.get("step") == len(rows) + 1, "noncontiguous training trace")
        require(all(type(row.get(key)) in (int, float) for key in ("loss", "gradient_norm", "cpu_seconds", "wall_seconds")),
                "training trace is missing numerical loss/gradient/clock fields")
        require(all(math.isfinite(value) for value in row.values() if type(value) in (int, float)), "nonfinite training trace")
        require(0 <= row["cpu_seconds"] <= 3600 and row["wall_seconds"] >= 0, "invalid training trace clocks")
        if allow_partial:
            require(row["cpu_seconds"] < 3600, "interrupted attempt reached the CPU cap")
        if rows:
            require(row["cpu_seconds"] >= rows[-1]["cpu_seconds"] and row["wall_seconds"] >= rows[-1]["wall_seconds"],
                    "nonmonotonic training trace clocks")
        rows.append(row)
    require(bool(rows) and (updates is None or len(rows) == updates), "training trace update count differs")
    return rows, partial_bytes


def admit_completed_endpoints():
    """Independent lightweight admission; no Torch import or model load."""
    evidence = HERE / "evidence"
    lock_path = evidence / "TRAINING_LOCK.json"
    lock = read(lock_path)
    required = {"schema": "cog-fusion-002-lock-v1", "seeds": list(SEEDS), "arms": list(ARMS),
                "updates": 1200, "teacher_updates": 100, "batch_families": 4,
                "threads_per_fit": 1, "cpu_cap_seconds_per_fit": 3600,
                "selected_checkpoint": "fixed_final_update"}
    require(all(lock.get(key) == value for key, value in required.items()), "unexpected training lock")
    for name, expected in lock["source_sha256"].items():
        require(sha256(HERE.parent / name) == expected, "locked training input changed: " + name)
    paths = [lock_path, evidence / "CAMPAIGN_EXIT.json"]
    exits = read(paths[-1])
    expected_jobs = {(arm, seed) for arm in ARMS for seed in SEEDS}
    require(len(exits) == 6 and {(row["arm"], row["seed"]) for row in exits} == expected_jobs,
            "original campaign exit inventory is incomplete")
    failed = {(row["arm"], row["seed"]): row for row in exits if row["returncode"] != 0}
    recovery = None
    recovered = {}
    if failed:
        require(all(type(row["returncode"]) is int and row["returncode"] < 0 and row["returncode"] != -int(signal.SIGXCPU)
                    for row in failed.values()),
                "a process-reported scientific failure is not a recovered signal interruption")
        paths.append(evidence / "RECOVERY_RESULT.json")
        recovery = read(paths[-1])
        protocol_path = evidence / "RECOVERY_PROTOCOL.json"
        protocol = read(protocol_path)
        require(recovery.get("original_campaign_exit_sha256") == sha256(evidence / "CAMPAIGN_EXIT.json")
                and recovery.get("recovery_protocol_sha256") == sha256(protocol_path), "recovery receipt hashes differ")
        require(protocol.get("training_lock_sha256") == sha256(lock_path)
                and protocol.get("source_sha256") == sha256(HERE / "recover_interrupted.py")
                and protocol.get("original_campaign_exit_present_at_declaration") is False,
                "prospective recovery declaration or source differs")
        require(recovery.get("held_model_scores_inspected") is False, "recovery does not retain its pre-scoring declaration")
        paths.extend([protocol_path, HERE / "recover_interrupted.py"])
        recovered = {(row["arm"], row["seed"]): row for row in recovery.get("recovered", [])}
        require(recovery.get("status") == "completed" and set(recovered) == set(failed)
                and len(recovery["recovered"]) == len(failed), "recovery is incomplete")
        for row in recovered.values():
            require(row.get("status") == "completed", "recovery endpoint is incomplete")
            code = failed[(row["arm"], row["seed"])]["returncode"]
            require(row.get("original_returncode") == code and row.get("original_signal") == signal.Signals(-code).name,
                    "recovery return code differs from the original campaign")
            archive = (HERE / row["failed_attempt"]).resolve()
            archive.relative_to(HERE.resolve())
            require(archive.is_dir(), "failed attempt is not preserved")
            manifest_path = archive / "ARCHIVE_MANIFEST.json"
            require(row.get("archive_manifest_sha256") == sha256(manifest_path), "recovery archive manifest hash differs")
            manifest = read(manifest_path)
            require(manifest.get("original_returncode") == code and manifest.get("original_trace") == row.get("original_trace"),
                    "recovery original-attempt receipt differs from the archived manifest")
            require({"IDENTITY.json", "train.jsonl"} <= set(manifest["original_run_manifest"]), "archive manifest omits original training artifacts")
            for name, item in manifest["original_run_manifest"].items():
                require(sha256(archive / name) == item["sha256"], "archived original training file changed: " + name)
            require(sha256(archive / "ORIGINAL_CAMPAIGN_EXIT.json") == recovery["original_campaign_exit_sha256"]
                    and sha256(archive / "ORIGINAL_TRAINING_LOCK.json") == protocol["training_lock_sha256"]
                    and sha256(archive / "ORIGINAL_CAMPAIGN_JOBS.json") == protocol["campaign_jobs_sha256"]
                    and sha256(archive / "ORIGINAL_STDOUT.log") == manifest["original_log_sha256"],
                    "preserved original campaign receipts differ")
            paths.extend(sorted(path for path in archive.rglob("*") if path.is_file()))
    elif (evidence / "RECOVERY_RESULT.json").exists():
        raise ValueError("unexpected recovery receipt without an interrupted original attempt")
    runs = []
    for seed in SEEDS:
        parent = parent_checkpoint(seed)
        parent_hash = sha256(parent)
        require(parent_hash == lock["parent_sha256"][str(seed)], "parent checkpoint changed")
        paths.append(parent)
        for arm in ARMS:
            folder = evidence / "runs" / f"{arm}_seed{seed}"
            result, identity = read(folder / "RESULT.json"), read(folder / "IDENTITY.json")
            expected = {"status": "completed", "study": "COG-FUSION-002", "arm": arm, "seed": seed,
                        "steps": 1200, "teacher_steps": 100, "batch_families": 4, "threads": 1,
                        "cpu_cap": 3600, "parent_sha256": parent_hash,
                        "native_lr": lock["native_lr"], "interface_lr": lock["interface_lr"]}
            require(all(result.get(key) == value for key, value in expected.items()), "endpoint is not the fixed completed fit")
            require(all(result.get(key) == value for key, value in identity.items()), "identity and result differ")
            require(identity.get("source_sha256") and all(lock["source_sha256"].get(name) == value
                    for name, value in identity["source_sha256"].items()), "fit source differs from the lock")
            require(result.get("checkpoint_sha256") == sha256(folder / "final.pt"), "final checkpoint hash differs")
            for leaf in ("RESULT.json", "IDENTITY.json", "train.jsonl", "final.pt", "optimizer.pt"):
                path = folder / leaf
                require(path.is_file(), "canonical endpoint artifact missing: " + str(path))
                paths.append(path)
            trace, _ = checked_trace(folder / "train.jsonl", updates=1200)
            require(all(row.get("teacher_typed_execution") is (row["step"] <= 100) for row in trace), "typed warmup trace differs")
            require(all(type(result.get(key)) in (int, float) and math.isfinite(result[key])
                        for key in ("cpu_seconds", "wall_seconds")), "completed receipt has invalid clocks")
            require(result["cpu_seconds"] >= trace[-1]["cpu_seconds"] and result["wall_seconds"] >= trace[-1]["wall_seconds"],
                    "completed receipt clocks precede the final update")
            if (arm, seed) in recovered:
                row = recovered[(arm, seed)]
                require(row["checkpoint_sha256"] == result["checkpoint_sha256"], "recovered checkpoint hash differs")
                require(read(HERE / row["failed_attempt"] / "IDENTITY.json") == identity,
                        "recovery initialization or configuration changed")
            runs.append({"arm": arm, "seed": seed, "folder": folder, "result": result})
    return lock, runs, recovery, {relative(path): sha256(path) for path in paths}


def reconstruct_draws(seed, family_count, lm_length, *, updates=1200, batch_families=4, teacher_updates=100):
    require(family_count > 0 and lm_length > 129 and updates > 0 and batch_families > 0, "invalid replay geometry")
    rng = random.Random(20090017 + seed)
    trace, family_trace, lm_trace = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    counts, predicted_counts = Counter(), Counter()
    draws = []
    for step in range(1, updates + 1):
        indices = [rng.randrange(family_count) for _ in range(batch_families)]
        starts = [rng.randrange(lm_length - 129) for _ in range(4)]
        # Match the locked trainer's default JSON separators and exact draw order.
        trace.update(json.dumps([indices, starts]).encode())
        family_trace.update(json.dumps(indices).encode())
        lm_trace.update(json.dumps(starts).encode())
        counts.update(indices)
        if step > teacher_updates:
            predicted_counts.update(indices)
        draws.append({"step": step, "family_indices": indices, "lm_window_starts": starts})
    probe_size = min(128, family_count)
    report = {"seed": seed, "python_rng_seed": 20090017 + seed, "updates": updates,
              "batch_families": batch_families, "lm_windows_per_update": 4,
              "lm_targets_per_window": 128, "lm_start_range_inclusive": [0, lm_length - 130],
              "batch_trace_sha256": trace.hexdigest(), "family_draw_sha256": family_trace.hexdigest(),
              "lm_start_sha256": lm_trace.hexdigest(), "distinct_training_families_seen": len(counts),
              "training_families_available": family_count, "total_family_draws": sum(counts.values()),
              "train_probe_families": probe_size, "train_probe_families_sampled": sum(index in counts for index in range(probe_size)),
              "train_probe_families_sampled_during_predicted_execution": sum(index in predicted_counts for index in range(probe_size)),
              "train_probe_unsampled_indices": [index for index in range(probe_size) if index not in counts],
              "train_probe_sample_counts": [counts[index] for index in range(probe_size)],
              "full_family_sample_counts": [counts[index] for index in range(family_count)], "draws": draws}
    return report, rng.getstate()


def expected_batch_counts(families):
    """Count supplied text, teacher targets and padding without any model code."""
    worlds = [world for family in families for world in family["worlds"]]
    require(bool(worlds), "empty audit batch")
    schedule = [step["type"] for step in worlds[0]["steps"]]
    require(all([step["type"] for step in world["steps"]] == schedule for world in worlds), "different batch schedules")
    batch = len(worlds)
    counts = {key: 0 for key in DETERMINISTIC_FIELDS}
    seen_answer = immediately_after_answer = False
    for position, kind in enumerate(schedule):
        steps = [world["steps"][position] for world in worlds]
        lengths = [len(step["text"]) + 1 for step in steps]
        counts["input_chars"] += sum(lengths)
        counts["input_physical_positions"] += batch * max(lengths)
        counts["semantic_encoder_calls"] += 1
        if kind == "observe":
            counts["write_total"] += batch
            if seen_answer:
                counts["post_answer_write_total"] += batch
            if immediately_after_answer:
                counts["first_post_answer_write_total"] += batch
            immediately_after_answer = False
        elif kind == "ask":
            target_lengths = [len(step["answer_text"]) for step in steps]
            teacher_lengths = [len(PREFIX) + length - 1 for length in target_lengths]
            counts["teacher_decode_chars"] += sum(teacher_lengths)
            counts["teacher_decode_physical_positions"] += batch * max(teacher_lengths)
            counts["answer_targets"] += sum(target_lengths)
            counts["generation_prefix_chars"] += batch * len(PREFIX)
            counts["query_total"] += batch
            counts["generated_answer_total"] += batch
            counts["teacher_decoder_calls"] += 1
            seen_answer = immediately_after_answer = True
        else:
            raise ValueError("unsupported TRAIN boundary")
    counts.update(rehearsal_targets=4 * 128, rehearsal_physical_positions=4 * 128, rehearsal_calls=1)
    return counts


def audit_trace(rows, result, replay, expected_rows):
    failures = []
    totals, expected_totals = Counter(), Counter()
    require(len(rows) == replay["updates"], "canonical trace does not contain exactly 1200 updates")
    previous_cpu = previous_wall = 0.0
    for index, (row, expected) in enumerate(zip(rows, expected_rows), 1):
        require(row.get("step") == index, "training trace is not contiguous")
        for key, value in expected.items():
            require(type(row.get(key)) is int and row[key] >= 0, "invalid integer accounting field: " + key)
            if row.get(key) != value:
                failures.append({"step": index, "field": key, "expected": value, "observed": row.get(key)})
        reset_count = expected["generated_answer_total"] if result["arm"] == "reset_training" else 0
        require(row.get("answer_boundaries_reset") == reset_count, "reset policy counter differs")
        require(row.get("teacher_typed_execution") is (index <= result["teacher_steps"]), "typed warmup differs")
        for key, value in row.items():
            if type(value) in (int, float):
                require(math.isfinite(value), "nonfinite trace metric: " + key)
            if type(value) is int and key != "step":
                totals[key] += value
        expected_totals.update(expected)
        generated, decoder_calls = row["generated_chars"], row["generation_decoder_calls"]
        require(type(generated) is int and type(decoder_calls) is int, "generated accounting fields must be integers")
        questions = expected["teacher_decoder_calls"]
        batch = expected["generated_answer_total"] // questions
        maxima = decoder_calls - questions
        require(questions <= maxima <= 12 * questions and maxima + questions * (batch - 1) <= generated <= batch * maxima,
                "generated length/decoder call counts violate the unrestricted 12-character rollout bounds")
        physical = sum(expected[key] for key in ("input_physical_positions", "teacher_decode_physical_positions",
                                                 "generation_prefix_chars", "rehearsal_physical_positions")) + generated
        require(row.get("physical_token_positions") == physical, "physical position accounting is inconsistent")
        for exact, total in (("write_exact", "write_total"), ("post_answer_write_exact", "post_answer_write_total"),
                             ("first_post_answer_write_exact", "first_post_answer_write_total"),
                             ("query_exact", "query_total"), ("generated_answer_exact", "generated_answer_total")):
            require(type(row.get(exact)) is int and 0 <= row[exact] <= row[total], "invalid reported accuracy count")
        require(previous_cpu <= row["cpu_seconds"] <= result["cpu_cap"] and previous_wall <= row["wall_seconds"],
                "training clock is nonmonotonic or exceeds the process CPU cap")
        previous_cpu, previous_wall = row["cpu_seconds"], row["wall_seconds"]
    for key, total in totals.items():
        if result.get(key) != total:
            failures.append({"field": key, "expected_log_sum": total, "observed_result": result.get(key)})
    for key in ("batch_trace_sha256", "distinct_training_families_seen", "training_families_available"):
        if result.get(key) != replay[key]:
            failures.append({"field": key, "expected_replay": replay[key], "observed_result": result.get(key)})
    require(result["cpu_seconds"] >= previous_cpu and result["wall_seconds"] >= previous_wall, "final clocks precede final update")
    return {"pass": not failures, "updates_checked": len(rows), "deterministic_fields_checked": list(DETERMINISTIC_FIELDS),
            "mismatch_count": len(failures), "first_mismatches": failures[:20],
            "expected_supplied_counts": dict(expected_totals), "reported_integer_counter_sums": dict(totals),
            "generated_counts_check": "reported characters/calls satisfy rollout bounds and position identities; strings were not saved and are not replayed",
            "training_loop_cpu_seconds": result["cpu_seconds"], "training_loop_wall_seconds": result["wall_seconds"]}


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def state_digest(state, prefix=""):
    output = hashlib.sha256()
    for name, tensor in state.items():
        if name.startswith(prefix):
            output.update(name.encode())
            output.update(tensor_bytes(tensor))
    return output.hexdigest()


def checkpoint_identity(payload):
    config = {"schema": payload["schema"], "config": payload["config"], "native": payload["native_config"],
              "chars": payload["chars"], "memory_enabled": payload["memory_enabled"]}
    output = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name, tensor in payload["model"].items():
        output.update(name.encode()); output.update(tensor_bytes(tensor))
    return output.hexdigest()


def audit_checkpoint_payloads(torch, parent, final, optimizer, result, replay, rng_state, counters):
    for key in ("schema", "config", "native_config", "chars", "memory_enabled", "admission"):
        require(final[key] == parent[key], "inherited checkpoint configuration changed: " + key)
    require(final["config"]["native_board_enabled"] is False and final["memory_enabled"] is True,
            "the admitted disabled-native-board memory configuration differs")
    require(len(final["chars"]) == 65 and len(set(final["chars"])) == 65, "inherited tokenizer differs")
    require(state_digest(parent["model"]) == result["initial_tensor_sha256"], "initial tensors differ from the parent")
    require(checkpoint_identity(final) == result["final_model_identity"], "final tensor identity differs from the receipt")
    expected = {"study": "COG-FUSION-002", "step": 1200, "seed": result["seed"], "arm": result["arm"],
                "parent_sha256": result["parent_sha256"], "batch_trace_sha256": replay["batch_trace_sha256"],
                "cpu_seconds": result["cpu_seconds"], "wall_seconds": result["wall_seconds"], **counters}
    require(all(final["metadata"].get(key) == value for key, value in expected.items()), "final metadata or counters differ")
    require(optimizer.get("step") == 1200, "optimizer endpoint step differs")
    require(optimizer.get("python_rng") == rng_state, "saved Python RNG differs from all 1200 replayed draws")
    before, after = parent["model"], final["model"]
    require(list(before) == list(after), "parameter inventory or order differs")
    changed, unchanged, inactive = [], [], []
    changed_elements = 0
    for name in before:
        left, right = before[name], after[name]
        require(left.shape == right.shape and left.dtype == right.dtype, "tensor shape/dtype differs: " + name)
        require(bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()), "nonfinite checkpoint tensor: " + name)
        equal = tensor_bytes(left) == tensor_bytes(right)
        if name.startswith(BOARD_PREFIX):
            inactive.append(name)
            require(equal, "inactive native-board tensor changed: " + name)
        elif equal:
            unchanged.append(name)
        else:
            changed.append(name)
            changed_elements += int(torch.count_nonzero(left != right))
    total = sum(t.numel() for t in after.values())
    inactive_values = sum(after[name].numel() for name in inactive)
    require((total, inactive_values, total - inactive_values) == (1077834, 80160, 997674), "parameter budget differs")
    groups = optimizer["optimizer"]["param_groups"]
    require(len(groups) == 2, "optimizer group inventory differs")
    names = [[name for name in after if name.startswith("native.")], [name for name in after if not name.startswith("native.")]]
    require(all(len(group["params"]) == len(group_names) for group, group_names in zip(groups, names)),
            "optimizer parameter groups do not match the native/interface tensor inventory")
    mapping = {index: name for group, group_names in zip(groups, names) for index, name in zip(group["params"], group_names)}
    require(len(mapping) == len(after), "optimizer contains duplicate parameter IDs")
    state = optimizer["optimizer"]["state"]
    expected_active_ids = {index for index, name in mapping.items() if not name.startswith(BOARD_PREFIX)}
    require(set(state) == expected_active_ids, "optimizer state must cover every active parameter and no inactive board parameter")
    optimizer_steps = []
    for index, values in state.items():
        require(not mapping[index].startswith(BOARD_PREFIX), "inactive native board acquired optimizer state")
        require(set(values) == {"step", "exp_avg", "exp_avg_sq"}, "AdamW state fields differ")
        step = float(values["step"])
        require(step == 1200, "every active parameter must retain 1200 AdamW steps")
        optimizer_steps.append(int(step))
        for key in ("exp_avg", "exp_avg_sq"):
            require(torch.is_tensor(values[key]) and values[key].shape == after[mapping[index]].shape
                    and values[key].dtype == after[mapping[index]].dtype, "AdamW moment shape/dtype differs")
        require(all(bool(torch.isfinite(value).all()) for value in values.values() if torch.is_tensor(value)),
                "nonfinite optimizer state")
    for group, base in zip(groups, (result["native_lr"], result["interface_lr"])):
        expected_lr = base * (.2 + .8 * (1 + math.cos(math.pi * 1199 / 1200)) / 2)
        require(group["lr"] == expected_lr and group["weight_decay"] == .01, "final optimizer configuration differs")
        require(tuple(group.get("betas", ())) == (.9, .999) and group.get("eps") == 1e-8
                and group.get("amsgrad") is False and group.get("maximize") is False, "AdamW declared defaults differ")
    return {"pass": True, "final_step": final["metadata"]["step"], "optimizer_step": optimizer["step"],
            "saved_python_rng_matches_full_replay": True, "parent_tensor_sha256": state_digest(before),
            "final_tensor_sha256": state_digest(after), "final_model_identity": checkpoint_identity(final),
            "optimizer_torch_rng_sha256": hashlib.sha256(tensor_bytes(optimizer["torch_rng"])).hexdigest(),
            "total_parameters": total, "inactive_board_parameters": inactive_values, "active_parameters": total - inactive_values,
            "inactive_board_tensors": len(inactive), "inactive_board_bitwise_unchanged": True,
            "inactive_board_parent_sha256": state_digest(before, BOARD_PREFIX),
            "inactive_board_final_sha256": state_digest(after, BOARD_PREFIX), "inactive_board_optimizer_states": 0,
            "active_tensors": len(changed) + len(unchanged), "changed_active_tensors": len(changed),
            "numerically_changed_active_parameter_values": changed_elements, "changed_active_tensor_names": changed,
            "unchanged_active_tensor_names": unchanged, "active_optimizer_states": len(state),
            "optimizer_parameter_step_range": [min(optimizer_steps), max(optimizer_steps)],
            "optimizer_name_mapping": "state_dict order filtered into native/interface groups; pinned sources have no buffers, tied parameters, hooks or aliases",
            "interpretation": "parameter changes establish numerical updates only; they do not establish meaningful learning or task qualification"}


def interrupted_costs(recovery):
    output = []
    for entry in (recovery or {}).get("recovered", []):
        path = HERE / entry["failed_attempt"] / "train.jsonl"
        rows, partial_bytes = checked_trace(path, allow_partial=True)
        require(len(rows) <= 1200, "archived trace exceeds the fixed attempt budget")
        counters = Counter()
        for row in rows:
            counters.update({key: value for key, value in row.items() if type(value) is int
                             and key not in ("step", "loss", "gradient_norm", "cpu_seconds", "wall_seconds")})
        expected_summary = {"completed_updates": len(rows), "trailing_partial_line_bytes": partial_bytes,
                            "first_reported": {key: rows[0][key] for key in ("step", "cpu_seconds", "wall_seconds")},
                            "last_reported": {key: rows[-1][key] for key in ("step", "cpu_seconds", "wall_seconds")},
                            "training_loop_cpu_seconds_lower_bound": rows[-1]["cpu_seconds"],
                            "reported_counters_lower_bound": dict(counters)}
        require(all(entry["original_trace"].get(key) == value for key, value in expected_summary.items()),
                "archived trace clocks/counters disagree with the recovery receipt")
        replacement_cpu = entry.get("process", {}).get("child_cpu_seconds_including_initialization")
        require(type(replacement_cpu) in (int, float) and math.isfinite(replacement_cpu) and replacement_cpu >= 0,
                "invalid replacement process CPU")
        output.append({"arm": entry["arm"], "seed": entry["seed"], "failed_attempt": entry["failed_attempt"],
                       "original_returncode": entry["original_returncode"], "completed_updates": len(rows),
                       "training_loop_cpu_seconds_lower_bound": rows[-1]["cpu_seconds"],
                       "training_loop_wall_seconds_lower_bound": rows[-1]["wall_seconds"], "trailing_partial_line_bytes": partial_bytes,
                       "reported_counters_lower_bound": dict(counters),
                       "physical_token_positions_lower_bound": counters.get("physical_token_positions", 0),
                       "replacement_child_process_cpu_seconds_including_initialization": replacement_cpu,
                       "scope": "original CPU is a logged lower bound; unreported interrupted work and initialization are not measured"})
    return output


def run(output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("training audit output already exists")
    lock, runs, recovery, input_hashes = admit_completed_endpoints()
    data_path = HERE / "evidence/data/train.json"
    lm_path = PARENT / "evidence/reproduction_inputs/train.bin"
    families = read(data_path)["families"]
    require(len(families) == 2048 and lm_path.stat().st_size % 2 == 0, "TRAIN geometry differs")
    data_hash, lm_hash = sha256(data_path), sha256(lm_path)
    require(all(run["result"]["train_data_sha256"] == data_hash and run["result"]["train_families_sha256"] == digest(families)
                and run["result"]["shakespeare_train_sha256"] == lm_hash for run in runs), "TRAIN input hashes differ")
    input_hashes.update({relative(data_path): data_hash, relative(lm_path): lm_hash})
    audit = {"schema": "cog-fusion-002-training-audit-v1", "pass": False, "status": "running",
             "source_sha256": {relative(Path(__file__)): sha256(Path(__file__)), relative(HERE / "common.py"): sha256(HERE / "common.py")},
             "input_sha256": input_hashes, "training_source_sha256": lock["source_sha256"],
             "threads": 1, "model_inference_calls": 0, "held_data_files_read": [],
             "train_probe_interpretation": "the fixed first 128 TRAIN-pool families remain the complete training-distribution adequacy panel; sampled/unsampled counts are descriptive and never filter or replace that gate",
             "replays": {}, "runs": [], "matched_pairs": [], "interrupted_attempts": interrupted_costs(recovery)}
    started, cpu_started = time.perf_counter(), time.process_time()
    # Admission of every endpoint is complete. Tensor payloads are now safe to inspect.
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    audit["torch_version"] = torch.__version__
    try:
        for seed in SEEDS:
            replay, rng_state = reconstruct_draws(seed, len(families), lm_path.stat().st_size // 2)
            replay["train_probe_unsampled_family_ids"] = [families[index]["family_id"] for index in replay["train_probe_unsampled_indices"]]
            audit["replays"][str(seed)] = replay
            expected_rows = [expected_batch_counts([families[index] for index in draw["family_indices"]]) for draw in replay["draws"]]
            parent = torch.load(parent_checkpoint(seed), map_location="cpu", weights_only=True)
            pair = []
            for run_entry in (entry for entry in runs if entry["seed"] == seed):
                folder, receipt = run_entry["folder"], run_entry["result"]
                rows = [json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()]
                counted = audit_trace(rows, receipt, replay, expected_rows)
                final = torch.load(folder / "final.pt", map_location="cpu", weights_only=True)
                optimizer = torch.load(folder / "optimizer.pt", map_location="cpu", weights_only=True)
                tensor_checks = audit_checkpoint_payloads(torch, parent, final, optimizer, receipt, replay, rng_state,
                                                          counted["reported_integer_counter_sums"])
                report = {"arm": receipt["arm"], "seed": seed, "checkpoint_sha256": receipt["checkpoint_sha256"],
                          "pass": counted["pass"] and tensor_checks["pass"], "accounting": counted, "checkpoints": tensor_checks}
                audit["runs"].append(report); pair.append(report)
                del final, optimizer
                gc.collect()
            audit["matched_pairs"].append({"seed": seed,
                "provided_input_target_rehearsal_counts_identical": all(pair[0]["accounting"]["reported_integer_counter_sums"][key]
                    == pair[1]["accounting"]["reported_integer_counter_sums"][key] for key in DETERMINISTIC_FIELDS),
                "both_arms_match_replayed_counts": all(row["accounting"]["pass"] for row in pair),
                "parent_tensors_identical": pair[0]["checkpoints"]["parent_tensor_sha256"] == pair[1]["checkpoints"]["parent_tensor_sha256"],
                "generated_characters_by_arm": {row["arm"]: row["accounting"]["reported_integer_counter_sums"]["generated_chars"] for row in pair},
                "physical_positions_by_arm": {row["arm"]: row["accounting"]["reported_integer_counter_sums"]["physical_token_positions"] for row in pair},
                "torch_rng_states_identical": pair[0]["checkpoints"]["optimizer_torch_rng_sha256"] == pair[1]["checkpoints"]["optimizer_torch_rng_sha256"],
                "torch_rng_comparison_status": "observed diagnostic; exact replay and saved-state admission use the Python RNG that selects families and rehearsal windows"})
            del parent
            gc.collect()
        for name, expected in input_hashes.items():
            require(sha256(HERE.parent / name) == expected, "audit input changed during inspection: " + name)
        audit["pass"] = len(audit["runs"]) == 6 and all(row["pass"] for row in audit["runs"])
        audit["status"] = "completed" if audit["pass"] else "failed"
    except Exception as exc:
        audit["status"] = "failed"
        audit["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        audit["audit_wall_seconds"] = time.perf_counter() - started
        audit["audit_cpu_seconds"] = time.process_time() - cpu_started
        audit["canonical_training_loop_cpu_seconds"] = sum(entry["result"]["cpu_seconds"] for entry in runs)
        audit["interrupted_training_loop_cpu_seconds_lower_bound"] = sum(row["training_loop_cpu_seconds_lower_bound"] for row in audit["interrupted_attempts"])
        audit["training_cpu_total_lower_bound"] = audit["canonical_training_loop_cpu_seconds"] + audit["interrupted_training_loop_cpu_seconds_lower_bound"]
        audit["cpu_scope"] = "canonical fit training-loop CPU plus archived interrupted-loop lower bounds; replacement whole-process CPU reported separately and not double-counted"
        create(output, audit)
    if not audit["pass"]:
        raise RuntimeError("training audit found mismatches; failed evidence retained")
    print(json.dumps({"output": str(output), "pass": audit["pass"],
                      "sampled_train_probe_families": {seed: row["train_probe_families_sampled"] for seed, row in audit["replays"].items()}}), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "evidence/TRAINING_AUDIT.json")
    run(parser.parse_args().output)
