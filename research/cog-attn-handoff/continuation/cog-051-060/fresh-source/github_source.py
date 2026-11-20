"""Rate-limited, resumable GitHub REST source operations."""
import json
import os
import threading
import time

import requests

API = "https://api.github.com"
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


