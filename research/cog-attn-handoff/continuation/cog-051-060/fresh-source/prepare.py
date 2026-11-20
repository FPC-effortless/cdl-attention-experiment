"""Freeze fresh language rows and separate provenance records before scoring."""
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DESIGN = json.loads((ROOT / "recipe/design.json").read_text())
CACHE = ROOT / "data"
STATE_EVENTS = {"closed", "reopened"}
URL = re.compile(r"https?://\S+", re.I)
HANDLE = re.compile(r"(?<!\w)@[\w-]+")
NUMBER = re.compile(r"\b\d+\b")


def stamp(value):
    return int(datetime.fromisoformat(value).timestamp())


def normalize(body):
    body = re.sub(r"```.*?```", " <code> ", body, flags=re.S)
    body = URL.sub(" <url> ", body)
    body = HANDLE.sub(" <mention> ", body)
    body = NUMBER.sub(" <number> ", body)
    return " ".join(body.lower().split())


def human(event):
    return event["actor_type"] == "User" and event["actor"] is not None and not event["actor_bot"]


def build(issue, split, counts):
    source = issue["issue"]
    created = stamp(source["created_at"])
    cutoff = stamp(DESIGN["cutoff"])
    events = issue["timeline"]
    times = [stamp(e["created_at"]) for e in events]
    key = f"{issue['repo']}#{source['number']}"
    state = 0
    invalid = False
    state_history = []
    for i, event in enumerate(events):
        if event["event"] in STATE_EVENTS:
            value = int(event["event"] == "closed")
            invalid |= state == value
            state = value
    counts["latest_snapshot_agrees"] += int(state == int(source["state"] == "closed"))
    if times != sorted(times):
        counts["timeline_order_conflicts"] += 1
        invalid = True
    if invalid:
        counts["unusable_issues"] += 1
        return [], {"id": key, "usable": False}
    keep = bisect_right(times, cutoff)
    events, times = events[:keep], times[:keep]
    ties = Counter(times[i] for i, event in enumerate(events)
                   if event["event"] in STATE_EVENTS or human(event))
    upcoming = {}
    following = None
    for i in range(len(events) - 1, -1, -1):
        event = events[i]
        upcoming[i] = following
        if event["event"] in STATE_EVENTS or human(event):
            following = i
    state = 0
    state_time = created
    action_counts = Counter()
    comments = Counter()
    rows = []
    for i, event in enumerate(events):
        now = times[i]
        actor = event["actor"]
        if event["event"] in STATE_EVENTS:
            state = int(event["event"] == "closed")
            state_time = now
            action_counts[actor] += int(human(event))
            counts[event["event"]] += 1
            state_history.append([i, state, event["id"]])
            continue
        if not human(event):
            counts["nonhuman_comments"] += 1
            continue
        counts["human_comments"] += 1
        text = normalize(event["body"])
        edited = event["updated_at"] != event["created_at"]
        complete = now + DESIGN["horizon_seconds"] <= cutoff
        counts["edited_comments"] += int(edited)
        counts["incomplete_horizon"] += int(not complete)
        if text and not edited and complete:
            next_index = upcoming[i]
            delta = None if next_index is None else times[next_index] - now
            link = (
                next_index is not None
                and delta <= DESIGN["horizon_seconds"]
                and events[next_index]["event"] in STATE_EVENTS
                and events[next_index]["actor"] == actor
            )
            strict = ties[now] == 1
            counts["timestamp_ties_excluded"] += int(not strict)
            counts["ordered_positive_ties"] += int(bool(link) and not strict)
            row = {
                "issue": key, "split": split, "comment": event["id"],
                "ordinal": i, "stamp": now, "text": text, "strict": strict,
                "closed": state, "reporter": int(actor == source["reporter"]),
                "past_actor_actions": action_counts[actor],
                "past_actor_comments": comments[actor],
                "past_comments": comments.total(),
                "age": now - created, "since_action": now - state_time,
                "target": int(link), "event": events[next_index]["id"] if link else None,
                "delta_seconds": delta if link else None,
            }
            rows.append(row)
        comments[actor] += 1
    return rows, {"id": key, "usable": True, "transitions": state_history,
                  "queries": [[row["ordinal"], row["closed"]] for row in rows]}


def snapshot(value):
    return {k: value[k] for k in ("state", "updated_at", "closed_at")}


def main():
    rows = []
    snapshots = []
    counts = defaultdict(Counter)
    identity = {}
    counts_by_repository = defaultdict(Counter)
    cohorts = json.loads((CACHE / "cohorts.json").read_text())
    assert {c["repo"] for c in cohorts} == {*DESIGN["familiar"], *DESIGN["external"]}
    for cohort in cohorts:
        repo = cohort["repo"]
        split = "test" if cohort["split"] == "familiar" else "external"
        for issue in cohort["issues"]:
            name = f"{repo.replace('/', '--')}--{issue['number']}.json"
            assert name not in identity
            raw = (CACHE / name).read_bytes()
            identity[name] = hashlib.sha256(raw).hexdigest()
            record = json.loads(raw)
            counts[split]["issues"] += 1
            normalized, stream = build(record, split, counts[split])
            rows.extend(normalized)
            counts_by_repository[repo]["issues"] += 1
            counts_by_repository[repo]["strict"] += sum(r["strict"] for r in normalized)
            counts_by_repository[repo]["positive"] += sum(r["strict"] and r["target"] for r in normalized)
            actions = [[stamp(e["created_at"]), int(e["event"] == "closed"), e["id"]]
                       for e in record["timeline"] if e["event"] in STATE_EVENTS]
            snapshots.append({
                "issue": stream["id"], "split": split, "created": stamp(issue["created_at"]),
                "search_at": cohort["search_at"], "replay_at": record["timeline_at"],
                "direct_at": record["direct_at"], "search": snapshot(issue), "direct": snapshot(record["direct"]),
                "actions": actions, "comment_times": [r["stamp"] for r in normalized],
                "valid_sequence": stream["usable"],
            })
    saved = {"dataset": rows, "snapshots": snapshots, "cohort-counts": counts,
             "repository-counts": counts_by_repository, "source-manifest": {"sha256": identity, "cohorts": cohorts}}
    for name, artifact in saved.items():
        (ROOT / "recipe" / f"{name}.json").write_text(json.dumps(artifact, separators=(",", ":")))


if __name__ == "__main__":
    main()
