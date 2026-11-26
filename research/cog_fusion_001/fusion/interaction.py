"""Predeclared multi-turn integration scenario on each fitted memory checkpoint.

This sequence extrapolates beyond one-answer training histories. It records
failures without repairing the model, substituting gold operations or spelling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .model import FusionModel, state_bytes
from .session import Session


def run_checkpoint(checkpoint: Path, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "session.sqlite"
    if database.exists():
        raise FileExistsError("interaction fixture already exists")
    model = FusionModel.from_checkpoint(checkpoint).eval()
    session = Session(model, database, seed=991)
    transcript = []

    def observe(text, rid, clear=()):
        result = session.observe(text, rid, clear_pending=clear)
        transcript.append({"type": "observe", "input": text, "result": result})

    def ask(text, rid, expected):
        result = session.ask(text, rid)
        transcript.append({"type": "ask", "input": text, "expected": expected,
                           "correct": result["text"] == expected, "result": result})
        return result

    observe("Amara owns the chest.", "owner-initial")
    observe("The coin is in the chest.", "link-initial")
    ask("Who owns the container holding the coin?", "first-question", "Amara.")
    observe("Cora owns the crate.", "irrelevant")
    ask("Who owns the container holding the coin?", "after-irrelevant", "Amara.")
    observe("The chest belongs to Bela.", "correction")
    session.close()
    session = Session(FusionModel.from_checkpoint(checkpoint).eval(), database, seed=12345)
    ask("Who owns the container holding the coin?", "after-restart", "Bela.")
    ask("Name the owner of whatever holds the coin.", "unfamiliar-question", "Bela.")
    session.store.mark_pending("pending-owner", key=("world", "chest", "owner"))
    session.close()
    session = Session(FusionModel.from_checkpoint(checkpoint).eval(), database, seed=12345)
    ask("Who owns the container holding the coin?", "pending-question", "pending")
    ask("Who owns the crate?", "unrelated-during-pending", "Cora.")
    observe("Hadi owns the chest.", "resolve-pending", clear=("pending-owner",))
    ask("Who owns the container holding the coin?", "after-pending-retry", "Hadi.")
    before_replay = dict(session.counters)
    replay = session.ask("Who owns the container holding the coin?", "first-question")
    transcript.append({"type": "historical_receipt_replay", "result": replay,
                       "tokens_unchanged": before_replay == session.counters,
                       "expected_historical_text": transcript[2]["result"]["text"]})
    observe("The owner of the chest is unknown.", "delete-owner")
    ask("Who owns the container holding the coin?", "after-delete", "unknown.")
    snapshot = session.store.snapshot_state()
    result = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "transcript": transcript, "counters": session.counters,
              "neural_tensor_bytes": state_bytes(session.state), "snapshot_blob_bytes": len(snapshot["neural_state"]),
              "database_bytes": database.stat().st_size,
              "learned_question_count": sum(r["type"] == "ask" and r["expected"] != "pending" for r in transcript),
              "learned_questions_correct": sum(r["type"] == "ask" and r["expected"] != "pending" and r["correct"] for r in transcript),
              "pending_policy_correct": all(r["correct"] for r in transcript if r["type"] == "ask" and r["expected"] == "pending"),
              "scope": "predeclared multi-turn extrapolation, with held-wording question; no training selection"}
    session.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parents[1] / "evidence")
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = {}
    for seed in (17, 29, 41):
        checkpoint = args.evidence / "runs" / f"memory_seed{seed}" / "final.pt"
        result = run_checkpoint(checkpoint, args.evidence / "interactions" / f"seed{seed}")
        results[str(seed)] = result
        print(json.dumps({"seed": seed, "learned_correct": result["learned_questions_correct"],
                          "learned_total": result["learned_question_count"],
                          "pending_policy": result["pending_policy_correct"]}), flush=True)
    (args.evidence / "INTERACTION_RESULTS.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
