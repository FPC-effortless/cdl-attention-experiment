"""Collect issue timelines through the public GitHub API with a resumable cache."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

import requests

ROOT = Path(__file__).resolve().parent
DESIGN = json.loads((ROOT / "recipe/design.json").read_text())
API = "https://api.github.com"
CACHE = ROOT / "data"
RATE_LOCK = threading.Lock()
LAST = 0.0
LOCAL = threading.local()
TOKEN = os.environ.get("GH_TOKEN") or os.environ["GITHUB_TOKEN"]


def fetch(url, params=None):
    global LAST
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers.update({
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
    for attempt in range(6):
        with RATE_LOCK:
            time.sleep(max(0, LAST + 0.15 - time.monotonic()))
            LAST = time.monotonic()
        try:
            response = LOCAL.session.get(url, params=params, timeout=35)
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.ok:
            return response
        if response.status_code == 429 or (
            response.status_code == 403 and "rate limit" in response.text.lower()
        ):
            time.sleep(max(10, int(response.headers.get("Retry-After", "15"))))
            continue
        if response.status_code < 500:
            response.raise_for_status()
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Retry budget exhausted for {url}")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(path)


def search(repo, start, end):
    limit = DESIGN["issue_limit_per_source_and_split"]
    response = fetch(f"{API}/search/issues", {
        "q": f"repo:{repo} is:issue comments:>=1 created:{start}..{end}",
        "sort": "created", "order": "asc", "per_page": 100,
    })
    matches = response.json()
    if matches["incomplete_results"]:
        raise RuntimeError(f"Incomplete search results for {repo} {start}")
    issues = matches["items"]
    while len(issues) < limit and "next" in response.links:
        response = fetch(response.links["next"]["url"])
        page = response.json()
        if page["incomplete_results"]:
            raise RuntimeError(f"Incomplete search results for {repo} {start}")
        issues.extend(page["items"])
    unique = {i["number"]: i for i in issues}
    if len(unique) != len(issues):
        raise RuntimeError(f"Repeated search result for {repo} {start}")
    return matches["total_count"], [{
        "number": i["number"], "created_at": i["created_at"],
        "updated_at": i["updated_at"], "state": i["state"],
        "reporter": (i["user"] or {}).get("id"), "comments": i["comments"],
    } for i in issues[:limit]]


def timeline(repo, issue):
    path = CACHE / f"{repo.replace('/', '--')}--{issue['number']}.json"
    if path.exists():
        return
    response = fetch(f"{API}/repos/{repo}/issues/{issue['number']}/timeline", {"per_page": 100})
    reduced = []
    counts = {}
    pages = 0
    while True:
        pages += 1
        for event in response.json():
            kind = event.get("event")
            counts[kind] = counts.get(kind, 0) + 1
            if kind not in {"commented", "closed", "reopened"}:
                continue
            user = (event.get("user") if kind == "commented" else event.get("actor")) or {}
            row = {"event": kind, "id": event["id"], "created_at": event["created_at"],
                   "actor": user.get("id"), "actor_type": user.get("type"),
                   "actor_bot": user.get("login", "").endswith("[bot]")}
            if kind == "commented":
                row.update(updated_at=event["updated_at"], body=event.get("body") or "")
            reduced.append(row)
        if "next" not in response.links:
            break
        response = fetch(response.links["next"]["url"])
    save(path, {"repo": repo, "issue": issue, "pages": pages,
                "event_counts": counts, "timeline": reduced})


def main():
    path = CACHE / "cohorts.json"
    if path.exists():
        cohorts = json.loads(path.read_text())
    else:
        cohorts = []
        for split, (start, end, _) in DESIGN["splits"].items():
            repos = DESIGN["sources"] + ([DESIGN["external_source"]] if split == "test" else [])
            for repo in repos:
                total, issues = search(repo, start, end)
                print(f"{split} {repo}: selected {len(issues)} of {total}", flush=True)
                cohorts.append({"split": "external" if repo == DESIGN["external_source"] else split,
                                "repo": repo, "total": total, "issues": issues})
        save(path, cohorts)
    with ThreadPoolExecutor(max_workers=12) as pool:
        tasks = {pool.submit(timeline, c["repo"], i): (c["repo"], i["number"])
                 for c in cohorts for i in c["issues"]}
        failures = []
        for count, future in enumerate(as_completed(tasks), 1):
            try:
                future.result()
            except Exception as error:
                failures.append((*tasks[future], str(error)))
            if count % 200 == 0:
                print(f"{count}/{len(tasks)} requests processed, {len(failures)} failures", flush=True)
    save(CACHE / "acquisition.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
                                      "issues_requested": len(tasks), "failures": failures})
    if failures:
        raise SystemExit(f"Retry incomplete acquisition ({len(failures)} failed)")


if __name__ == "__main__":
    main()
