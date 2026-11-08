"""Run the frozen cohort with bounded CPU concurrency, then isolated profiling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ARMS = ("v1_coupled", "v1_independent", "v1_no_reference", "v3", "v4_p16", "v4",
        "lrc_local", "lrc_cache", "v4_lexical", "v4_lexical_slot", "v4_shuffled",
        "gru", "gru_cache", "transformer")
GATES = ("utility", "utility_shuffled", "utility_random")
SEEDS = (2201, 2202, 2203)
OUT = P / "artifacts"


def pool(jobs, workers, stage):
    active = []
    journal = []
    directory = OUT / "worker_console"
    directory.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    while active or pending:
        while pending and len(active) < workers:
            args = pending.pop(0)
            name = "_".join(args).replace("--", "")
            output = (directory / (name + ".log")).open("w")
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            proc = subprocess.Popen([sys.executable, str(P / "run.py")] + args,
                                    cwd=P, stdout=output, stderr=subprocess.STDOUT, env=env)
            active.append((args, proc, output, time.time()))
            print("START", stage, name, "pid", proc.pid, flush=True)
        finished = []
        for entry in active:
            args, proc, output, start = entry
            code = proc.poll()
            if code is not None:
                output.close()
                journal.append({"args": args, "pid": proc.pid, "start_epoch": start,
                                "end_epoch": time.time(), "exit_code": code})
                (OUT / f"journal_{stage}.json").write_text(json.dumps(journal, indent=2))
                print("END", stage, " ".join(args), "exit", code, flush=True)
                if code != 0:
                    for _, child, _, _ in active:
                        if child.poll() is None:
                            child.terminate()
                    raise RuntimeError(f"{stage} failed: {args}; inspect worker log")
                finished.append(entry)
        active = [e for e in active if e not in finished]
        if active:
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("worker count outside8-core runtime")
    start = time.time()
    # Start the common V4 bases early, but close every backbone before fitting gates.
    order = ("v4",) + tuple(a for a in ARMS if a != "v4")
    jobs = [["train", "--arm", a, "--seed", str(s)] for a in order for s in SEEDS]
    pool(jobs, args.workers, "backbones")
    pool([["gates", "--seed", str(s)] for s in SEEDS], min(3, args.workers), "gates")
    subprocess.run([sys.executable, str(P / "run.py"), "close"], check=True)
    pool([["evaluate", "--arm", a, "--seed", str(s)] for a in ARMS + GATES for s in SEEDS],
         args.workers, "evaluation")
    # Every concurrent trainer/evaluator has exited before the serial timing pass.
    subprocess.run([sys.executable, str(P / "run.py"), "profile"], check=True)
    subprocess.run([sys.executable, str(P / "run.py"), "summarize"], check=True)
    (OUT / "complete.json").write_text(json.dumps({"start_epoch": start,
        "end_epoch": time.time(), "workers": args.workers, "status": "complete"}, indent=2))
    print("COHORT_COMPLETE", time.time() - start, flush=True)


if __name__ == "__main__":
    main()
