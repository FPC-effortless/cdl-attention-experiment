"""Cross-check disagreeing search snapshots using two independent REST paths."""
from datetime import datetime, timezone
import json
from pathlib import Path

from acquire import API, CACHE, fetch

ROOT = Path(__file__).resolve().parent
STATES = {"closed", "reopened"}


def latest(events):
    return next(("closed" if e["event"] == "closed" else "open"
                 for e in reversed(events) if e["event"] in STATES), "open")


def main():
    results = []
    cohorts = json.loads((CACHE / "cohorts.json").read_text())
    for cohort in cohorts:
        repo = cohort["repo"]
        for issue in cohort["issues"]:
            name = f"{repo.replace('/', '--')}--{issue['number']}.json"
            timeline = json.loads((CACHE / name).read_text())["timeline"]
            timeline_state = latest(timeline)
            if timeline_state == issue["state"]:
                continue
            url = f"{API}/repos/{repo}/issues/{issue['number']}"
            snapshot = fetch(url).json()
            page = fetch(url + "/events", {"per_page": 100})
            events = page.json()
            while "next" in page.links:
                page = fetch(page.links["next"]["url"])
                events.extend(page.json())
            results.append({
                "issue": f"{repo}#{issue['number']}",
                "search_state": issue["state"], "timeline_state": timeline_state,
                "issue_endpoint_state": snapshot["state"], "events_endpoint_state": latest(events),
                "snapshot_closed_at": snapshot["closed_at"],
                "search_updated_at": issue["updated_at"], "issue_endpoint_updated_at": snapshot["updated_at"],
                "only_in_issue_events": sorted(
                    {e["id"] for e in events if e["event"] in STATES}
                    - {e["id"] for e in timeline if e["event"] in STATES}),
                "only_in_timeline": sorted(
                    {e["id"] for e in timeline if e["event"] in STATES}
                    - {e["id"] for e in events if e["event"] in STATES}),
            })
    path = ROOT / "snapshot-crosscheck.json"
    path.write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(),
                                "disagreements": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
