#!/usr/bin/env python3
"""Constructed, no-Torch checks for discarded-history and delivery metadata.

This script does not train, run the completed-model audit/finalizer, or read test
data/results. It writes temporary metadata fixtures and a separate review receipt.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
FROZEN = {
    "model.py": "f96bd967b4bc4ece9d2750a5cf1afdf1cc819a6b007e2d638af633376c7eccaa",
    "train.py": "52751e1a7d383a8bb809835d1d528ee02fa4abbbe8439a5ef0918186ce3bedf6",
    "data.py": "5c53755333460b7fdb35e0f9a5f88b2b3c7e3f37674ef92df506164728dad69d",
    "sample.py": "b8188e737856dacc67cbc948c2fc8a884007421397075d38f973e9b37e6207ba",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(rows):
    return b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)


def fixture(directory):
    def train(step):
        return {"event": "train", "step": step, "training_seconds": step * 2.0}

    old_prefix = [train(0), train(419)]
    old_tail = [train(420), train(430), train(440)]
    marker1 = {"event": "trajectory_recovery", "interruption_session": 21959,
               "resume_step": 419, "discarded_tail_file": "discarded_metrics_21959.jsonl"}
    new_prefix = old_prefix + [marker1, train(1000), train(1800)]
    new_tail = [train(step) for step in (1810, 1820, 1830, 1840, 1850)]
    marker2 = {"event": "trajectory_recovery", "interruption_session": 48409,
               "resume_step": 1800, "discarded_tail_file": "discarded_metrics_48409.jsonl"}
    canonical = new_prefix + [marker2, train(1810), train(2000)]
    for suffix, prefix, tail in (("21959", old_prefix, old_tail), ("48409", new_prefix, new_tail)):
        (directory / f"metrics_before_interruption_{suffix}.jsonl").write_bytes(encode(prefix + tail))
        (directory / f"discarded_metrics_{suffix}.jsonl").write_bytes(encode(tail))
    (directory / "metrics.jsonl").write_bytes(encode(canonical))
    events = []
    for sid, kind, retained, last, count in (
        (21959, "intentional_runtime_stop", 419, 440, 3),
        (48409, "unobserved_runtime_stop", 1800, 1850, 5),
    ):
        archive, tail = f"metrics_before_interruption_{sid}.jsonl", f"discarded_metrics_{sid}.jsonl"
        event = {
            "event": kind, "session_id": sid, "last_retained_checkpoint_step": retained,
            "last_logged_discarded_step": last, "discarded_logged_train_records": count,
            "minimum_discarded_completed_updates": last - retained,
            "full_original_metrics_archive": archive, "full_original_metrics_sha256": digest(directory / archive),
            "discarded_tail_file": tail, "discarded_tail_sha256": digest(directory / tail),
            "tool_exit_code": 130 if sid == 21959 else None, "reason": "Synthetic metadata fixture",
        }
        if sid == 48409:
            event["launcher_session_token"] = "100-2"
        events.append(event)
    launchers = [
        {"event": "launcher_start", "session_token": "100-2", "start_unix": 100.0,
         "log": "launcher_100-2.log"},
        {"event": "launcher_start", "session_token": "200-2", "start_unix": 200.0,
         "log": "launcher_200-2.log"},
        {"event": "launcher_end", "session_token": "200-2", "start_unix": 200.0,
         "end_unix": 201.0, "returncode": 0, "log": "launcher_200-2.log"},
    ]
    (directory / "launcher_sessions.jsonl").write_bytes(encode(launchers))
    return events, launchers


def main():
    audit = load_module("recovery_audit", PROJECT / "postfit_audit.py")
    finalizer = load_module("recovery_finalizer", PROJECT / "tools/finalize_delivery.py")
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    def rejects(call):
        try:
            call()
        except (ValueError, RuntimeError):
            return True
        return False

    with tempfile.TemporaryDirectory(prefix="tiny_recovery_metadata_") as tmp:
        directory = Path(tmp)
        events, launchers = fixture(directory)
        observed = audit.audit_discarded_histories(directory, events, 8192)
        check("two_recoveries_preserve_prefixes_and_sum_lower_bounds",
              observed["passed"] and len(observed["entries"]) == 2
              and observed["minimum_documented_discarded_completed_updates"] == 71
              and observed["minimum_documented_discarded_target_presentations"] == 581632
              and len(observed["archive_names"]) == 4)
        check("duplicate_interruption_rejected",
              rejects(lambda: audit.audit_discarded_histories(directory, events + [events[-1]], 8192)))
        reused = copy.deepcopy(events)
        reused[-1]["full_original_metrics_archive"] = events[0]["full_original_metrics_archive"]
        check("reused_archive_identity_rejected",
              rejects(lambda: audit.audit_discarded_histories(directory, reused, 8192)))

        for name, mutation in (
            ("tampered_tail_fails", lambda e, l: (directory / e[-1]["discarded_tail_file"]).write_bytes(b"{}\n")),
            ("missing_recovery_marker_fails", lambda e, l: (directory / "metrics.jsonl").write_bytes(
                b"".join(line for line in (directory / "metrics.jsonl").read_bytes().splitlines(keepends=True)
                         if json.loads(line).get("interruption_session") != 48409))),
            ("changed_retained_prefix_fails", lambda e, l: (directory / "metrics.jsonl").write_bytes(
                (directory / "metrics.jsonl").read_bytes().replace(b'"step": 419', b'"step": 418', 1))),
            ("wrong_discarded_count_fails", lambda e, l: e[-1].update(minimum_discarded_completed_updates=49)),
            ("invented_exit_code_fails", lambda e, l: e[-1].update(tool_exit_code=0)),
            ("missing_unobserved_launcher_fails", lambda e, l: e[-1].update(launcher_session_token="missing")),
            ("observed_exit_disallows_unobserved_claim", lambda e, l: (directory / "launcher_sessions.jsonl").write_bytes(
                encode(l + [{"event": "launcher_end", "session_token": "100-2", "returncode": 0}]))),
        ):
            events, launchers = fixture(directory)
            mutation(events, launchers)
            check(name, not audit.audit_discarded_histories(directory, events, 8192)["passed"])

        events, launchers = fixture(directory)
        unmatched = finalizer.unmatched_launcher_receipts(launchers, events)
        check("unmatched_launcher_keeps_unknown_resources",
              len(unmatched) == 1 and unmatched[0]["session_token"] == "100-2"
              and unmatched[0]["exit_status"] == "unknown"
              and all(unmatched[0][key] is None for key in (
                  "observed_end_unix", "observed_wall_duration_seconds", "max_child_rss_kib")))
        check("unreceipted_start_rejected",
              rejects(lambda: finalizer.unmatched_launcher_receipts(launchers, events[:1])))
        check("completed_old_launcher_rejected_as_unobserved",
              rejects(lambda: finalizer.unmatched_launcher_receipts(
                  launchers + [{"event": "launcher_end", "session_token": "100-2"}], events)))
        check("duplicate_launcher_identity_rejected",
              rejects(lambda: finalizer.unmatched_launcher_receipts(launchers + [launchers[0]], events)))
        bad_events = copy.deepcopy(events)
        bad_events[-1]["exit_code"] = 137
        check("finalizer_rejects_fabricated_exit_code",
              rejects(lambda: finalizer.unmatched_launcher_receipts(launchers, bad_events)))

        new_archives = {"metrics_before_interruption_48409.jsonl", "discarded_metrics_48409.jsonl"}
        check("new_archives_are_required_and_hash_bound",
              new_archives <= set(finalizer.RUN_REQUIRED) and new_archives <= finalizer.AUDITED_INPUTS_REQUIRED)
        check("source_evidence_and_checker_are_whitelisted",
              set(finalizer.RECOVERY_EVIDENCE) <= set(finalizer.SOURCE_REQUIRED)
              and "tools/verify_interruption_recovery.py" in finalizer.SOURCE_REQUIRED
              and "tools/finalize_delivery.py" in finalizer.SOURCE_REQUIRED
              and "interruption_recovery_review.json" in finalizer.SOURCE_OPTIONAL)

        # Only configuration/identity metadata are read. All scores below are
        # synthetic fixtures rendered in memory; no actual result/test is read.
        manifest = json.loads((PROJECT / "runs" / finalizer.RUN_NAME / "run_config.json").read_text())
        metric = {"mean_nll": 1.0, "bits_per_character": 1.0 / math.log(2),
                  "character_perplexity": math.e, "next_character_accuracy": 0.1, "targets": 55769}
        result = {
            "status": "completed", "stop_reason": "synthetic_fixture", "step": 2000, "best_step": 1000,
            "parameters": 9725682, "test": metric, "initial_evaluation": {"val": metric},
            "unigram_validation": metric, "unigram_test": metric,
            "best_checkpoint_training_targets": 8192000, "attempted_updates": 2000, "skipped_updates": 0,
            "training_targets_seen": 16384000, "train_target_equivalent_passes": 1.0,
            "training_seconds": 1.0, "session_wall_seconds": 1.0,
        }
        audit_fixture = {"passed": 1, "total": 1, "sampling": {"samples_json_sha256": "synthetic"},
                         "accounting": {"minimum_documented_discarded_completed_updates": 71,
                                        "minimum_documented_discarded_target_presentations": 581632,
                                        "total_physical_target_presentations_lower_bound": 16965632}}
        sessions = [{"threads": 2, "resume_step": 1800, "max_wall_seconds": 900,
                     "python": "synthetic", "torch": "not imported"}]
        wrapper = {"start_unix": 200.0, "end_unix": 201.0, "return_code": 0,
                   "max_child_rss_kib": 1, "allocator_env": {"MALLOC_ARENA_MAX": "2"},
                   "argv": ["python", "--max-wall-seconds", "900"], "kind": "synthetic completed receipt"}
        _, report, _ = finalizer.render_reports(
            manifest, result, audit_fixture, {"val": metric}, sessions, events, [wrapper], wrapper,
            {"best.pt": "synthetic", "resume.pt": "synthetic"},
            {"commit": "synthetic", "repository": "fixture"}, unmatched, True)
        report = report.decode()
        check("report_separates_intentional_unobserved_and_completed_receipts",
              "**1 separately recorded intentional runtime stops**" in report
              and "**1 separately recorded runtime stops without an observed exit status**" in report
              and "no observed end time, exit code, complete wall duration, or peak child RSS" in report
              and "**running at capture**" in report and "581,632" in report and "16,965,632" in report
              and "optional scalar-replay receipt" in report
              and "does not establish full weight/RNG tensor identity" in report)

    check("frozen_training_sources_unchanged", all(digest(PROJECT / key) == val for key, val in FROZEN.items()))
    launcher_copy = PROJECT.parent / "tiny_launch/finalize_delivery.py"
    mirror_checked = launcher_copy.is_file()
    if mirror_checked:
        check("finalizer_copies_identical", (PROJECT / "tools/finalize_delivery.py").read_bytes() == launcher_copy.read_bytes())
    check("torch_not_imported", "torch" not in sys.modules)

    run = PROJECT / "runs" / finalizer.RUN_NAME
    actual = audit.audit_discarded_histories(run, audit.read_jsonl(run / "interruption_events.jsonl"), 8192)
    check("actual_archived_prefixes_and_counts_bind_without_inference",
          actual["passed"] and actual["minimum_documented_discarded_completed_updates"] == 71
          and actual["minimum_documented_discarded_target_presentations"] == 581632)
    receipt = {
        "scope": "Constructed no-Torch metadata/report checks and read-only archived-history binding; no training or model inference",
        "passed": sum(row["passed"] for row in checks), "total": len(checks), "checks": checks,
        "actual_archived_history": actual, "frozen_source_sha256": FROZEN,
        "postfit_audit_sha256": digest(PROJECT / "postfit_audit.py"),
        "finalizer_sha256": digest(PROJECT / "tools/finalize_delivery.py"),
        "check_script_sha256": digest(__file__),
        "recovery_evidence_sha256": {rel: digest(PROJECT / rel) for rel in finalizer.RECOVERY_EVIDENCE},
        "synthetic_report_scores_are_not_training_results": True,
        "torch_imported": False, "final_audit_or_finalizer_executed": False,
        "external_finalizer_mirror_checked": mirror_checked,
    }
    (PROJECT / "interruption_recovery_review.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": receipt["passed"], "total": receipt["total"],
                      "actual_discarded_updates_lower_bound": 71,
                      "actual_discarded_presentations_lower_bound": 581632}))


if __name__ == "__main__":
    main()
