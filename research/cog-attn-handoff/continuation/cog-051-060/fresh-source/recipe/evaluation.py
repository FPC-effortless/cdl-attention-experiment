"""Temporal holdout scoring, threshold freezing and exact state replay."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from state_store import World
from wr006 import History

ROOT = Path(__file__).resolve().parent
DESIGN = json.loads((ROOT / "design.json").read_text())
THRESHOLDS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3,
              0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
              0.85, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999)
CORROBORATION = (0.0, 0.25, 0.5, 0.75, 0.9)


def measure_admissions(target, admitted, issue):
    count = int(admitted.sum())
    tp = int(target[admitted].sum())
    fp = count - tp
    positive = int(target.sum())
    matched = np.unique(issue[admitted])
    failed = np.unique(issue[admitted & (target == 0)])
    ok = len(matched) - len(failed)
    lower = float(beta.ppf(0.05, ok, len(failed) + 1)) if ok else 0.0
    return {
        "tp": tp, "fp": fp, "fn": positive - tp, "admitted": count,
        "precision": tp / count if count else None,
        "positive_recall": tp / positive if positive else None,
        "comment_coverage": count / len(target) if len(target) else None,
        "admitted_issues": len(matched), "issues_with_false_admissions": len(failed),
        "issue_success_lower_95": lower,
        "passes": bool(count and positive and tp / count >= DESIGN["precision_gate"]
                       and tp / positive >= DESIGN["minimum_positive_coverage"]
                       and lower >= DESIGN["precision_gate"]),
    }


def eligible(scores, cue, semantic, threshold, support):
    if threshold is None:
        return np.zeros(len(scores), dtype=bool)
    corroborated = (semantic[:, 1] >= support) | (cue[:, 4].astype(bool) & ~cue[:, 2].astype(bool))
    return (scores >= threshold) & corroborated


def threshold_selection(scores, cue, semantic, target, validation, two_factor):
    s, c, e, y = scores[validation], cue[validation], semantic[validation], target[validation]
    candidates = []
    for threshold in THRESHOLDS:
        for support in CORROBORATION if two_factor else (0.0,):
            admitted = eligible(s, c, e, threshold, support)
            tp = int(y[admitted].sum())
            n = int(admitted.sum())
            if tp and tp / n >= DESIGN["precision_gate"]:
                candidates.append((tp, tp / n, threshold, support))
    if not candidates:
        return None, 0.0
    return max(candidates)[2:]


def summarize(y, scores):
    if not len(y):
        return {"rows": 0, "positives": 0, "ap": None, "auc": None, "brier": None}
    return {
        "rows": len(y), "positives": int(y.sum()),
        "ap": float(average_precision_score(y, scores)) if y.any() else None,
        "auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, scores)),
    }


def evaluate(rows, semantic, predictions, cue, found, ix, arm):
    target = np.asarray([r["target"] for r in rows])
    issue = np.asarray([r["issue"] for r in rows])
    comment = np.asarray([r["comment"] for r in rows])
    state = np.asarray([r["closed"] for r in rows])
    texts = {rows[i]["text"] for i in ix["train"]}
    seen = np.asarray([r["text"] in texts for r in rows])
    heads = {}
    details = {}
    for head, scores in predictions.items():
        threshold, support = threshold_selection(
            scores, cue, semantic, target, ix["validation"], head.startswith("modular"))
        admit = eligible(scores, cue, semantic, threshold, support)
        heads[head] = {"threshold": threshold, "corroboration": support, "cohorts": {}}
        details[head] = []
        for split in ("validation", "test", "external"):
            idx = ix[split]
            y, p, selected = target[idx], scores[idx], admit[idx]
            ranking = summarize(y, p)
            ranking.update(measure_admissions(y, selected, issue[idx]))
            ranking["slices"] = {}
            for group, mask in (("close", state[idx] == 0), ("reopen", state[idx] == 1),
                                ("unseen_text", ~seen[idx]), ("seen_text", seen[idx])):
                summary = summarize(y[mask], p[mask])
                summary.update(measure_admissions(y[mask], selected[mask], issue[idx][mask]))
                ranking["slices"][group] = summary
            ranking["frontier_diagnostic"] = [
                {"threshold": t, "tp": int(y[p >= t].sum()), "admitted": int((p >= t).sum())}
                for t in THRESHOLDS
            ]
            if found is not None:
                ranking["historical_cache_matches"] = int(found[idx].sum())
                ranking["historical_cache_positive_matches"] = int(y[found[idx]].sum())
            heads[head]["cohorts"][split] = ranking
            details[head].extend(
                {"split": split, "issue": issue[i], "comment": int(comment[i]), "closed": int(state[i]),
                 "target": int(target[i]), "score": float(scores[i]), "admitted": bool(admit[i])}
                for i in idx
            )
    return heads, details


def replay():
    with TemporaryDirectory() as temporary:
        return replay_database(Path(temporary) / "events.sqlite")


def replay_database(path):
    streams = json.loads((ROOT / "streams.json").read_text())
    world = World(path)
    assert world.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    horizon = 1 + max(ordinal for stream in streams if stream["usable"]
                      for ordinal, _ in stream["queries"])
    horizon = max(horizon, 1 + max(ordinal for stream in streams if stream["usable"]
                                   for ordinal, *_ in stream["transitions"]))
    index = History(horizon)
    results = {name: Counter() for name in ("train", "validation", "test", "external")}
    mark = perf_counter()
    for stream in streams:
        stats = results[stream["split"]]
        if not stream["usable"]:
            stats["excluded_issues"] += 1
            continue
        key = stream["id"]
        stats["replayed_issues"] += 1
        transitions = stream["transitions"]
        scheduled = sorted((i, 0, e, v) for i, v, e in transitions)
        scheduled += [(i, 1, 0, v) for i, v in stream["queries"]]
        for ordinal, is_query, event, expected in sorted(scheduled):
            if not is_query:
                world.advance()
                status = world.commit(f"{key}/{event}", world.epoch(), {key: expected})
                stats["writes"] += 1
                stats["write_failures"] += int(status != "committed")
                index.insert(key, ordinal, ordinal, expected)
                continue
            stats["queries"] += 1
            # Independent reference scans only strictly earlier state actions.
            oracle = 0
            for event_ordinal, value, _ in transitions:
                if event_ordinal < ordinal:
                    oracle = value
            sql = world.db.execute("SELECT value FROM state WHERE k=?", (key,)).fetchone()
            current = json.loads(sql[0]) if sql else 0
            indexed, _ = index.lookup(key, ordinal - 1, ordinal - 1)
            stats["stream_oracle_mismatch"] += int(expected != oracle)
            stats["sqlite_mismatches"] += int(current != oracle)
            stats["index_mismatches"] += int(max(indexed, 0) != oracle)
        for ordinal, _ in stream["queries"]:
            # Query older prefixes after loading the entire eligible issue history.
            prior = [value for when, value, _ in transitions if when < ordinal]
            oracle = prior[-1] if prior else 0
            indexed, _ = index.lookup(key, ordinal - 1, ordinal - 1)
            stats["post_replay_history_mismatches"] += int(max(indexed, 0) != oracle)
    payload = (world.db.execute("PRAGMA page_count").fetchone()[0]
               * world.db.execute("PRAGMA page_size").fetchone()[0])
    wal_bytes = path.with_name(path.name + "-wal").stat().st_size
    world.db.close()
    return {"scopes": results, "index_array_bytes": index.payload(),
            "sqlite_page_payload_bytes": payload, "sqlite_wal_bytes": wal_bytes,
            "elapsed_seconds": perf_counter() - mark}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()
