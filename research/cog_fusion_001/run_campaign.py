"""Run the preregistered six local CPU fits without holding a terminal open."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
NATIVE = Path("/workspace/scratch/2a5ba884844e/deliverables/COG_CONNECTED_1M.pt")
SHAKESPEARE = Path("/workspace/scratch/2a5ba884844e/cog_connected_1m_003/data/train.bin")


def main():
    paths = [HERE / "PROTOCOL.md", EVIDENCE / "DATA_MANIFEST.json", EVIDENCE / "corpus.json"]
    paths += [HERE / "fusion" / name for name in ("model.py", "data.py", "train.py", "engine.py")]
    paths += sorted((HERE / "fusion/native").glob("*.py"))
    lock = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_sha256": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "seeds": [17, 29, 41], "arms": ["memory", "no_memory"], "updates": 1000,
            "teacher_updates": 250, "cpu_cap_seconds_per_fit": 1800, "parallel_fits": 6}
    lock_path = EVIDENCE / "TRAINING_LOCK.json"
    if lock_path.exists():
        raise FileExistsError("training lock already exists; inspect existing run rather than overwrite")
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    jobs = []
    for seed in lock["seeds"]:
        for arm in lock["arms"]:
            output = EVIDENCE / "runs" / f"{arm}_seed{seed}"
            output.mkdir(parents=True, exist_ok=False)
            log = (output / "console.log").open("w")
            cmd = [sys.executable, "-m", "fusion.train", "--native", str(NATIVE), "--corpus", str(EVIDENCE / "corpus.json"),
                   "--shakespeare", str(SHAKESPEARE), "--output", str(output), "--arm", arm, "--seed", str(seed),
                   "--steps", "1000", "--teacher-steps", "250", "--pairs", "4", "--cpu-cap", "1800"]
            process = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
            jobs.append({"seed": seed, "arm": arm, "pid": process.pid, "process": process, "log": log, "command": cmd})
    public = [{k: v for k, v in job.items() if k not in ("process", "log")} for job in jobs]
    (EVIDENCE / "CAMPAIGN_JOBS.json").write_text(json.dumps(public, indent=2) + "\n")
    print(json.dumps(public), flush=True)
    # Short ticks only. The tool session may yield while jobs run independently.
    while any(job["process"].poll() is None for job in jobs):
        time.sleep(5)
    results = []
    for job in jobs:
        job["log"].close()
        results.append({"seed": job["seed"], "arm": job["arm"], "returncode": job["process"].returncode})
    (EVIDENCE / "CAMPAIGN_EXIT.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)
    if any(result["returncode"] for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
