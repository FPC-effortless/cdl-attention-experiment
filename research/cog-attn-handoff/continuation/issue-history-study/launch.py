"""Pin the public-source diagnostic and execute it under Research Lab OS."""
import json
from pathlib import Path
import shutil
import sys

from research_lab.core import Controller
from research_lab.evidence import put_node
from research_lab.execution import run_worker
from trainlab.runner import run_experiment, verify_run

ROOT = Path(__file__).resolve().parent
RECIPES = ROOT / "recipe"
SEEDS = [49103, 49109, 49121]
SOURCE = "eec439492407b0aa26922c23b5ae40d95289f4eb"
PROGRAM = "authentic-issue-history"
SPLIT = "github-public-issues-five-repos-2024-2026-v1"


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def experiment(name, command, inputs, artifacts, metrics, seed=SEEDS[0], timeout=900):
    return {
        "schema": "trainlab.experiment.v1", "execution_purpose": "engineering_diagnostic",
        "name": name, "description": "Chronological real GitHub issue history diagnostic",
        "command": [sys.executable, *command], "seed": seed, "timeout_seconds": timeout,
        "source_kind": "AUTHENTIC_CANDIDATE", "split_identity": SPLIT,
        "inputs": inputs, "artifacts": artifacts, "metrics": metrics,
        "budget": {"human_comment_corpus": 1, "cpu": 1},
    }


def pin_semantic():
    runfile = ROOT / "semantic-receipt.json"
    if runfile.exists():
        run = json.loads(runfile.read_text())
    else:
        inputs = ["semantic.py", "dataset.json", "design.json", "source-manifest.json"]
        inputs += [str(path.relative_to(RECIPES)) for path in sorted((RECIPES / "nli").iterdir())]
        config = experiment("semantic-evidence", ["semantic.py"], inputs,
                            ["semantic.npy", "semantic-timing.json"], {"rows": "max"})
        config_path = RECIPES / "semantic-recipe.json"
        write(config_path, config)
        run = run_experiment(config_path, ROOT / "semantic-runs")
        if run["status"] != "succeeded":
            raise RuntimeError(f"Semantic run {run['id']} failed")
        write(runfile, run)
    runpath = ROOT / "semantic-runs" / run["id"]
    assert verify_run(runpath)["valid"]
    for name in ("semantic.npy", "semantic-timing.json"):
        shutil.copyfile(runpath / "output" / name, RECIPES / name)


def main():
    if sys.argv[1:] == ["semantic"]:
        pin_semantic()
        return
    lab = Controller(ROOT / "ecosystem")
    lab.program(PROGRAM, "Real comment-conditioned issue transitions and event replay", 14400, 0)
    lab.resource("cpu-one", "cpu", 1, 0)
    files = ["entry.py", "design.json", "evaluation.py", "models.py", "wr006.py",
             "state_store.py", "dataset.json", "streams.json", "cohort-counts.json",
             "source-manifest.json", "semantic.npy", "semantic-timing.json", "frozen-plan.sha256"]
    for arm in ("control", "candidate"):
        write(RECIPES / f"{arm}.json", experiment(
            f"issues-{arm}", ["entry.py", arm], files,
            ["results.json", "predictions.json", "predictions.sha256", "checkpoint.joblib"],
            {"ap": "max"}))
    protocol = {
        "schema": "research_lab.protocol.v1", "id": PROGRAM, "program": PROGRAM, "author": "codex",
        "question": "Does state-scoped modular grounding beat an input-matched flat model on real future issue changes?",
        "hypothesis": "Separate language, state and semantic evidence yields commit precision at useful coverage.",
        "alternatives": ["Recorded behavior is poorly identified from a comment.", "A flat model is sufficient."],
        "predictions": ["Modular test AP exceeds the flat control.", "Recorded state replay is exact."],
        "controls": ["Identical source, split, safe history and semantic checkpoint."],
        "ablations": ["Remove NLI, add normalized binding reuse, compare history-only and character/word text models."],
        "metric": "ap", "direction": "max", "minimum_effect": 0,
        "seeds": SEEDS, "resource": "cpu-one", "slots": 1, "max_attempts": 1,
        "stopping_rule": "Run each paired seed once and replay all predictions in clean directories.",
        "scope": "AUTHENTIC_CANDIDATE", "priority": 1, "source_commit": SOURCE,
        "next_test": "Independent prospective recording with human-labeled intent and authority.",
        "limitations": ["Historical behavior proxy, no independently authenticated comment authority.",
                        "Fixed single-host retrospective data, no independent external review."],
        "candidate": "candidate.json", "control": "control.json", "replication_tolerance": 0,
        "replay_artifacts": ["predictions.sha256"],
    }
    write(ROOT / "lab-protocol.json", protocol)
    lab.register(protocol, RECIPES)
    put_node(lab, "issue-history.protocol", "experiment", protocol)
    for role in ("experiment", "replication"):
        worker = "issue-" + role
        lab.worker(worker, role, "cpu-one")
        run_worker(lab, worker)
        write(ROOT / "lab-status.json", lab.status())


if __name__ == "__main__":
    main()
