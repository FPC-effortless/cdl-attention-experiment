"""Recompute behavior labels from source timelines and verify frozen scored runs."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import sys

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from trainlab.runner import verify_run

from investigation_world.evidence.models import (
    EvidenceArtifactRef, EvidenceOutcome, EvidenceProducerRef, EvidenceRecord,
    EvidenceSubjectRef, EvidenceVisibility,
)

ROOT = Path(__file__).resolve().parent
RECIPE = ROOT / "recipe"
DESIGN = json.loads((RECIPE / "design.json").read_text())
HORIZON = timedelta(seconds=DESIGN["horizon_seconds"])


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def date(value):
    return datetime.fromisoformat(value)


def source_rows():
    m = read(RECIPE / "source-manifest.json")
    expected = {}
    counts = Counter()
    issue_sets = defaultdict(set)
    for cohort in m["cohorts"]:
        repo, split = cohort["repo"], cohort["split"]
        cutoff = date(DESIGN["splits"]["test" if split == "external" else split][2])
        for source in cohort["issues"]:
            number = source["number"]
            issue = f"{repo}#{number}"
            assert not any(issue in ids for ids in issue_sets.values())
            issue_sets[split].add(issue)
            name = f"{repo.replace('/', '--')}--{number}.json"
            path = ROOT / "data" / name
            assert digest(path) == m["sha256"][name]
            raw = read(path)["timeline"]
            action_rows = [e for e in raw if e["event"] in ("closed", "reopened")]
            state_names = [e["event"] for e in action_rows]
            counts["raw_snapshots_match"] += int(
                ("closed" if state_names and state_names[-1] == "closed" else "open") == source["state"])
            counts["issues"] += 1
            if state_names != ["closed" if i % 2 == 0 else "reopened" for i in range(len(state_names))]:
                counts["invalid_state_sequence"] += 1
                continue
            times = [date(e["created_at"]) for e in raw]
            if any(a > b for a, b in zip(times, times[1:])):
                counts["out_of_order"] += 1
                continue
            prefix = []
            for i, (event, stamp) in enumerate(zip(raw, times)):
                is_action = event["event"] in ("closed", "reopened")
                is_person = event["actor_type"] == "User" and event["actor"] is not None and not event["actor_bot"]
                if stamp <= cutoff and (is_action or is_person):
                    prefix.append((i, event, stamp))
            timestamps = Counter(t for _, _, t in prefix)
            for j, (i, event, stamp) in enumerate(prefix):
                if event["event"] != "commented" or event["updated_at"] != event["created_at"]:
                    continue
                if stamp + HORIZON > cutoff or not event["body"].strip():
                    continue
                earlier = [(e, t) for k, e, t in prefix[:j]]
                state = next((int(e["event"] == "closed") for e, _ in reversed(earlier)
                              if e["event"] != "commented"), 0)
                next_event = None
                if j + 1 < len(prefix):
                    _, follow, follow_time = prefix[j + 1]
                    if (follow["event"] != "commented" and follow["actor"] == event["actor"]
                            and follow_time - stamp <= HORIZON):
                        next_event = follow["id"]
                key = (issue, event["id"])
                assert key not in expected
                expected[key] = (split, i, timestamps[stamp] == 1, state, int(next_event is not None), next_event)
    return expected, counts, issue_sets


def check_data():
    for named in (RECIPE / "frozen-plan.sha256").read_text().splitlines():
        checksum, relative = named.split("  ", 1)
        assert digest(ROOT / relative) == checksum, relative
    expected, counts, issue_sets = source_rows()
    rows = read(RECIPE / "dataset.json")
    assert len(rows) == len(expected)
    for row in rows:
        value = expected.pop((row["issue"], row["comment"]))
        assert value == tuple(row[k] for k in ("split", "ordinal", "strict", "closed", "target", "event"))
    assert not expected
    counts["labels_independently_reconstructed"] = len(rows)
    counts["strict_labels"] = sum(r["strict"] for r in rows)
    counts["issues_by_split"] = {key: len(value) for key, value in issue_sets.items()}
    return rows, counts


def run_records(rows):
    lab = read(ROOT / "lab-status.json")
    strict = {(r["issue"], r["comment"]): r for r in rows if r["strict"] and r["split"] != "train"}
    reports = defaultdict(dict)
    predictions = defaultdict(dict)
    primary = {}
    replication = []
    for task in lab["tasks"]:
        if task["role"] == "review":
            continue
        assert task["status"] == "succeeded", task
        if task["role"] == "replication":
            replication.append(task)
            continue
        runpath = ROOT / "ecosystem" / "runs" / task["result"]["run_id"]
        assert verify_run(runpath)["valid"], runpath
        report = read(runpath / "output/results.json")
        preds = read(runpath / "output/predictions.json")
        checksum = hashlib.sha256(json.dumps(preds, sort_keys=True, allow_nan=False,
                                            separators=(",", ":")).encode()).hexdigest()
        assert checksum == (runpath / "output/predictions.sha256").read_text()
        seed = report["seed"]
        primary[task["id"]] = checksum
        reports[seed].update(report["heads"])
        predictions[seed].update(preds)
        for name, entries in preds.items():
            assert len(entries) == len(strict)
            assert {(p["issue"], p["comment"]) for p in entries} == strict.keys()
            for p in entries:
                r = strict[p["issue"], p["comment"]]
                assert (p["target"], p["closed"], p["split"]) == (r["target"], r["closed"], r["split"])
            for split in ("test", "external", "validation"):
                ps = [p for p in entries if p["split"] == split]
                y = np.array([p["target"] for p in ps])
                score_dtype = np.float32 if name == "zero_shot_nli" else np.float64
                s = np.array([p["score"] for p in ps], dtype=score_dtype)
                a = np.array([p["admitted"] for p in ps])
                result = report["heads"][name]["cohorts"][split]
                assert result["positives"] == int(y.sum())
                assert result["tp"] == int(y[a].sum())
                assert result["fp"] == int(a.sum() - y[a].sum())
                assert result["brier"] == brier_score_loss(y, s), (name, split, result["brier"], brier_score_loss(y, s))
                if y.any():
                    assert result["ap"] == average_precision_score(y, s)
                    assert result["auc"] == roc_auc_score(y, s)
    assert len(replication) == 1
    matches = replication[0]["result"]
    assert matches["all_match"]
    assert set(matches["replays"]) == set(primary)
    for task_id, receipt in matches["replays"].items():
        path = ROOT / "ecosystem" / "runs" / receipt["run_id"]
        assert verify_run(path)["valid"]
        assert (path / "output/predictions.sha256").read_text() == primary[task_id]
    return reports, predictions, len(primary), len(matches["replays"])


def diagnostics(predictions):
    seed = DESIGN["seed"]
    out = {}
    heads = predictions[seed]
    for split in ("test", "external"):
        groups = {}
        for name, entries in heads.items():
            groups[name] = [p for p in entries if p["split"] == split]
        anchors = [(p["issue"], p["comment"]) for p in groups["modular"]]
        assert all(anchors == [(p["issue"], p["comment"]) for p in entries]
                   for entries in groups.values())
        issue = np.asarray([key for key, _ in anchors])
        unique, inv = np.unique(issue, return_inverse=True)
        y = np.asarray([p["target"] for p in groups["modular"]])
        scores = {name: np.asarray([p["score"] for p in rows]) for name, rows in groups.items()}
        rnd = np.random.default_rng(seed)
        differences = defaultdict(list)
        for _ in range(DESIGN["bootstrap_draws"]):
            picked = rnd.integers(0, len(unique), len(unique))
            weight = np.bincount(picked, minlength=len(unique))[inv]
            keep = weight > 0
            if not np.any(y[keep]):
                continue
            aps = {name: average_precision_score(y[keep], scores[name][keep], sample_weight=weight[keep])
                   for name in ("modular", "flat_all_features", "lexical_history", "modular_no_nli")}
            for baseline in ("flat_all_features", "lexical_history", "modular_no_nli"):
                differences[baseline].append(aps["modular"] - aps[baseline])
        by_repo = {}
        repos = np.asarray([key.split("#", 1)[0] for key in issue])
        for repo in np.unique(repos):
            mask = repos == repo
            by_repo[repo] = {"rows": int(mask.sum()), "positives": int(y[mask].sum()),
                             "ap": {name: float(average_precision_score(y[mask], s[mask]))
                                    for name, s in scores.items()} if y[mask].any() else {}}
        out[split] = {
            "bootstrap_primary_seed": seed,
            "ap_difference_95_intervals": {
                key: [float(x) for x in np.quantile(values, [0.025, 0.975])]
                for key, values in differences.items()
            },
            "by_repository": by_repo,
        }
    return out



def corrected_ablation(rows, predictions):
    """Apply the same gate search without querying the NLI corroborator."""
    sys.path.insert(0, str(RECIPE))
    from evaluation import THRESHOLDS, measure_admissions
    from models import cues

    lookup = {(r["issue"], r["comment"]): r for r in rows}
    results = {}
    for seed, heads in predictions.items():
        entries = heads["modular_no_nli"]
        values = np.asarray([p["score"] for p in entries])
        labels = np.asarray([p["target"] for p in entries])
        cohorts = np.asarray([p["split"] for p in entries])
        issues = np.asarray([p["issue"] for p in entries])
        cue = np.asarray([cues(lookup[p["issue"], p["comment"]]) for p in entries], dtype=bool)
        corroborated = cue[:, 4] & ~cue[:, 2]
        validation = cohorts == "validation"
        choices = []
        for t in THRESHOLDS:
            for corroborate in (False, True):
                admitted = (values >= t) & (corroborated | (not corroborate))
                chosen = validation & admitted
                n = int(chosen.sum())
                found = int(labels[chosen].sum())
                if found and found / n >= DESIGN["precision_gate"]:
                    choices.append((found, found / n, t, corroborate))
        threshold, use_cue = max(choices)[2:] if choices else (None, False)
        admit = ((values >= threshold) & (corroborated | (not use_cue))) if choices else np.zeros(
            len(entries), dtype=bool)
        results[seed] = {"threshold": threshold, "use_explicit_cue": use_cue, "cohorts": {}}
        for split in ("validation", "test", "external"):
            mask = cohorts == split
            results[seed]["cohorts"][split] = measure_admissions(labels[mask], admit[mask], issues[mask])
    return results


def main():
    rows, counts = check_data()
    if not (ROOT / "lab-status.json").exists():
        print(json.dumps(counts, indent=2))
        return
    heads, preds, primary, replicas = run_records(rows)
    semantic = read(ROOT / "semantic-receipt.json")
    assert verify_run(ROOT / "semantic-runs" / semantic["id"])["valid"]
    summary = {"corpus_audit": counts, "trainlab_primary_verified": primary,
               "trainlab_replay_verified": replicas, "heads": heads,
               "bootstrap_and_repository": diagnostics(preds),
               "no_nli_admission_gate_correction": corrected_ablation(rows, preds)}
    out = ROOT / "audit.json"
    out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    record = EvidenceRecord(
        evidence_type="runtime.authentic_issue_history",
        outcome=EvidenceOutcome.OBSERVED, visibility=EvidenceVisibility.OPERATOR_PRIVATE,
        claim="Retrospective behavior diagnostic with exact artifact verification. See recorded gate results.",
        subjects=(
            EvidenceSubjectRef(kind="dataset", subject_id="github-issue-history-five-repos",
                               content_sha256=digest(RECIPE / "source-manifest.json")),
            EvidenceSubjectRef(kind="protocol", subject_id="authentic-issue-history",
                               content_sha256=digest(ROOT / "lab-protocol.json")),
        ),
        producer=EvidenceProducerRef(producer_id="local-issue-history-audit", producer_version="1",
                                    content_sha256=digest(Path(__file__))),
        artifacts=(EvidenceArtifactRef(artifact_id="audit-summary", content_sha256=digest(out)),),
        observed_at=datetime.now(timezone.utc),
        provenance={"source": "GitHub REST public timelines", "scope": "RETROSPECTIVE_AUTHENTIC_DIAGNOSTIC",
                    "review": "SAME_SESSION_SEPARATE_IMPLEMENTATION"},
    )
    (ROOT / "veritas-evidence.json").write_text(record.model_dump_json(indent=2) + "\n")
    print(json.dumps({**counts, "verified_primary": primary, "verified_replays": replicas}, indent=2))


if __name__ == "__main__":
    main()
