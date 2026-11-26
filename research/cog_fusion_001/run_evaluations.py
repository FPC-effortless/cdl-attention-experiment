"""Wait for the fixed training campaign, then evaluate each final checkpoint once."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"


def main():
    while not (EVIDENCE / "CAMPAIGN_EXIT.json").exists():
        time.sleep(5)
    status = json.loads((EVIDENCE / "CAMPAIGN_EXIT.json").read_text())
    if any(row["returncode"] for row in status):
        raise RuntimeError("a training fit is incomplete; inspect receipts")
    lock = json.loads((EVIDENCE / "TRAINING_LOCK.json").read_text())
    for name, expected in lock["source_sha256"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError("locked training source changed: " + name)
    paths = [HERE / "fusion" / name for name in ("evaluate.py", "interaction.py", "session.py", "store.py")]
    (EVIDENCE / "EVALUATION_LOCK.json").write_text(json.dumps({
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    }, indent=2) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    jobs = []
    for row in status:
        name = f"{row['arm']}_seed{row['seed']}"
        output = EVIDENCE / "evaluation" / name
        output.mkdir(parents=True, exist_ok=False)
        log = (output / "console.log").open("w")
        command = [sys.executable, "-m", "fusion.evaluate", "--checkpoint", str(EVIDENCE / "runs" / name / "final.pt"),
                   "--corpus", str(EVIDENCE / "corpus.json"), "--output", str(output)]
        job = subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
        jobs.append((name, job, log))
    print("Started six final-checkpoint evaluations", flush=True)
    while any(job.poll() is None for _, job, _ in jobs):
        time.sleep(5)
    results = []
    for name, job, log in jobs:
        log.close()
        results.append({"name": name, "returncode": job.returncode})
    (EVIDENCE / "EVALUATION_EXIT.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)
    if any(row["returncode"] for row in results):
        sys.exit(1)
    for module in ("fusion.preservation", "fusion.interaction"):
        code = subprocess.call([sys.executable, "-m", module], cwd=HERE, env=env)
        if code:
            raise RuntimeError(module + " failed")


if __name__ == "__main__":
    main()
