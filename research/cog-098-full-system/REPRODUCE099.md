# Reproduce the mixed-domain experiment

Use the PR 83 source stack and these files. The evidence ZIP contains the full results099 directory, 12 frozen checkpoints, action traces and the experiment overlay. Run on Linux. Data are read from external Git checkouts, never from a network connection during an episode.

The run used Python 3.12.14, torch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0, pytest 9.0.2, gymnasium 1.1.1, minigrid 3.0.0 and pygame 2.6.1. Install the CPU torch wheel from its CPU index. Create a directory containing these clean clones, using these exact names and commits.

| Directory | Source | Commit |
|---|---|---|
| quix | https://github.com/jkoppel/QuixBugs | 4257f44b0ff1181dedaedee6a447e133219fcebf |
| mw | https://github.com/budzianowski/multiwoz | fe0c8e65cfcd8462bd33c86e35f21addc84ca82b |
| bugs | https://github.com/soarsmu/BugsInPy | 11c5f1eea954a42132cfd06bf257766a7963e0fd |
| snooper | https://github.com/cool-RR/PySnooper | Full clone containing the buggy and fixed revisions in BugsInPy project metadata |

The MultiWOZ checkout only needs dialogues_001.json from the official 2.2 train and test directories. The BugsInPy checkout only needs projects/PySnooper. The measured external files and split members are in data_manifest.json. The content hashes include runtime tiling/environment source.

Run from research/cog-098-full-system in the review branch:

```bash
python -m pytest -q test_gate.py test_agent099.py ../cog-097e/test_experiment.py
python manifest099.py /path/to/data /tmp/manifest.json
cmp data_manifest.json /tmp/manifest.json
bash reproduce099.sh /path/to/data /path/to/new-results
```

To replay an archived policy, copy its .pt file to an empty output directory (or link it), and run the same frozen command. The decision digest includes results and actions and excludes timings. The unaided checkpoint requires the matching flag.

```bash
python evaluate.py --data /path/to/data --out /path/to/output --arm cog --seed 1701 --unaided
```

For real-project transfer install pytest 7.4.4, python-toolbox 1.3.2, six 1.17.0, future 0.18.3 and decorator 4.4.2 in an additional Python 3.8.20 environment. Its interpreter is used solely for upstream project tests. The harness and learning runtime still run on 3.12. The first two commands reproduce the assisted controls, the third uses the frozen unassisted weights. The last two commands check every generated edit in a separate evaluator diagnostic.

```bash
python bugsinpy_eval.py --data /path/to/data --out /path/to/new-results --python /path/to/py38
python bugsinpy_eval.py --data /path/to/data --out /path/to/new-results --python /path/to/py38 --compat
python bugsinpy_eval.py --data /path/to/data --out /path/to/new-results/unaided --python /path/to/py38 --compat
python bugsinpy_eval.py --data /path/to/data --out /path/to/new-results --python /path/to/py38 --compat --diagnose
python bugsinpy_eval.py --data /path/to/data --out /path/to/new-results --python /path/to/py38 --compat --diagnose --tail
```

Extract the evidence ZIP at the repository root to recover the original files by relative path. Compare the checkpoint SHA-256, trace SHA-256 and decision digests in the run records. A new run may produce different elapsed times or watchdog classifications on different hardware. The admission command remains `python gate.py --strict`. The original internal evaluation supplies no independently held test receipt.
