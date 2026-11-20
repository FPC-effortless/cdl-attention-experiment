"""Warm public-payload microbenchmark, excluding fixture generation and verifier."""
import json
import random
import statistics
import time
from pathlib import Path
from recipe.cache_trial import fingerprint, fixture_pair
from investigation_world.benchmark.policies import PublicEvidenceReferencePolicy

policy = PublicEvidenceReferencePolicy()
rng = random.Random(401)
rows = []
for case in range(24):
    episode,_ = fixture_pair(120300+case*3+case%2,1,"unchanged")
    payload = episode.public_payload()
    cached = policy(payload)
    for count in (0,32,256):
        enlarged = dict(payload,records=[dict(payload["records"][1],object_id=f"OTHER-{i}") for i in range(count)]+payload["records"])
        initial = fingerprint(enlarged,cached,"candidate")
        def guarded():
            return cached if fingerprint(enlarged,cached,"candidate")==initial else policy(enlarged)
        jobs = [("fresh",lambda:policy(enlarged)),("guard",guarded)]
        for rep in range(5):
            rng.shuffle(jobs)
            for label,fn in jobs:
                start = time.perf_counter_ns()
                for _ in range(100):
                    fn()
                rows.append(dict(case=case,extra=count,mode=label,ns=(time.perf_counter_ns()-start)/100))
Path("timing.json").write_text(json.dumps(rows))
for count in (0,32,256):
    print(count,[round(statistics.median(r["ns"] for r in rows if r["extra"]==count and r["mode"]==m)/1000,2) for m in ("guard","fresh")])
