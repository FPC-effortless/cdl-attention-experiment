"""Historical CSM replay recipe, executed by TrainLab."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys

# Working directory contains the pinned Git inputs for one study.
STEPS = {
    "study041": [("replay.py", "scores.csv", None), ("scope_checks.py", "scope.txt", None)],
    "skill001": [("pilot.py", "pilot-results.json", None)],
    "skill002": [("run.py", "scores.csv", None)],
    "skill003": [("experiment.py --time", "scores.csv", "timing.csv")],
    "skill004": [("experiment.py", "scores.csv", "model.txt")],
    "skill005": [("study.py fit", "models.json", None), ("study.py evaluate", "scores.csv", None)],
}

if __name__ == "__main__":
    name = sys.argv[1]
    output = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    random.seed(int(os.environ["TRAINLAB_SEED"]))
    for command, result, secondary in STEPS[name]:
        with (output / result).open("wb") as stdout:
            completed = subprocess.run([sys.executable, *command.split()], cwd=name, stdout=stdout,
                                       stderr=subprocess.PIPE if secondary else None, check=True,
                                       env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
        if secondary:
            (output / secondary).write_bytes(completed.stderr)
    (output / "metrics.json").write_text(json.dumps({"steps_completed": len(STEPS[name])}))
