"""Capture a new cohort with chronological issue history and current snapshots."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

from github_source import API, fetch, save

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DESIGN = json.loads((ROOT / "recipe/design.json").read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def compact_issue(item):
    return {**{k: item[k] for k in ("number", "created_at", "updated_at", "state", "closed_at", "comments")},
            "reporter": (item["user"] or {}).get("id"), "node_id": item["node_id"]}


def collect_cohorts(existing=()):
    start, end = DESIGN["created"]
    limit = DESIGN["issue_limit_per_source"]
    cohorts = []
    for split in ("familiar", "external"):
        for repo in DESIGN[split]:
            if repo in existing:
                continue
            result = fetch(f"{API}/search/issues", {
                "q": f"repo:{repo} is:issue comments:>=1 created:{start}..{end}",
                "sort": "created", "order": "asc", "per_page": limit,
            }).json()
            assert not result["incomplete_results"], repo
            cohorts.append({"split": split, "repo": repo, "search_at": now(),
                            "total": result["total_count"],
                            "issues": [compact_issue(issue) for issue in result["items"]]})
            print(f"{split} {repo}: {len(result['items'])}/{result['total_count']}", flush=True)
    return cohorts


def timeline_and_snapshot(repo, issue):
    path = DATA / f"{repo.replace('/', '--')}--{issue['number']}.json"
    if path.exists():
        return
    uri = f"{API}/repos/{repo}/issues/{issue['number']}"
    page = fetch(uri + "/timeline", {"per_page": 100})
    timeline = []
    kinds = {}
    pages = 0
    while True:
        pages += 1
        for event in page.json():
            kind = event["event"]
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind not in ("commented", "closed", "reopened"):
                continue
            user = (event.get("user") if kind == "commented" else event.get("actor")) or {}
            row = {"event": kind, "id": event["id"], "created_at": event["created_at"],
                   "actor": user.get("id"), "actor_type": user.get("type"),
                   "actor_bot": (user.get("login") or "").endswith("[bot]")}
            if kind == "commented":
                row.update(updated_at=event["updated_at"], body=event.get("body") or "")
            timeline.append(row)
        if "next" not in page.links:
            break
        page = fetch(page.links["next"]["url"])
    timeline_at = now()
    direct = compact_issue(fetch(uri).json())
    assert direct["node_id"] == issue["node_id"], uri
    save(path, {"repo": repo, "issue": issue, "timeline": timeline, "pages": pages,
                "timeline_at": timeline_at, "direct_at": now(), "direct": direct, "event_counts": kinds})


def main():
    DATA.mkdir(exist_ok=True)
    cohort_file = DATA / "cohorts.json"
    cohorts = json.loads(cohort_file.read_text()) if cohort_file.exists() else []
    new_cohorts = collect_cohorts({c["repo"] for c in cohorts})
    if new_cohorts:
        cohorts += new_cohorts
        save(cohort_file, cohorts)
    failures = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        jobs = {pool.submit(timeline_and_snapshot, c["repo"], i): (c["repo"], i["number"])
                for c in cohorts for i in c["issues"]}
        for count, future in enumerate(as_completed(jobs), 1):
            try:
                future.result()
            except Exception as error:
                failures.append([*jobs[future], str(error)])
            if count % 100 == 0:
                print(f"{count}/{len(jobs)} issues collected, {len(failures)} failed", flush=True)
    save(DATA / "acquisition.json", {"issues": len(jobs), "failures": failures, "completed_at": now()})
    if failures:
        raise SystemExit("Collection incomplete. Retry with the cached cohort.")


if __name__ == "__main__":
    main()
