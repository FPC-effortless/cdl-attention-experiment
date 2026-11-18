# Resume CSM-BDH-002 research

[Research Lab OS historical replay](pipeline/README.md) contains the six TrainLab runs, Forge inventory, Veritas evidence records and portable lab database.

## Resume from a new machine

```sh
git clone --branch research/csm-skill-first-pilot https://github.com/FPC-effortless/cdl-attention-experiment.git
cd cdl-attention-experiment/research/csm-bdh002-handoff
```

Follow the experiment-specific commands below. The checkout contains the source, frozen inputs, protocols, and expected results for the current replay. See [ARCHIVES.md](ARCHIVES.md) for historical evidence that requires separate access.

- [Study 041](study041/README.md) replays the Study 040 bottlenecks with a bounded scope grammar.
- [CSM-SKILL-001](skill001/README.md) records the skill-first proposal and a language-free procedural-memory pilot.
- [CSM-SKILL-002](skill002/README.md) compares a goal-directed planner with and without the acquired SWAP across every four-record assignment.
- [CSM-SKILL-003](skill003/README.md) evaluates when to defer SWAP and records controller-plus-executor timing.
- [CSM-SKILL-004](skill004/README.md) tests learned controller selection across record counts and documents the remaining transfer error.
- [CSM-SKILL-005](skill005/README.md) freezes the independent-pair selector and evaluates every six-record assignment.
- [CSM-SKILL-006 development panel](skill006/RESULTS.md) measures headroom for routing between acquired SWAP and ROTATE3.

Start with [study 040](study040/README.md). Its best deployable methods passed three of five held-out synthetic layouts across three seeds. Every deployable method failed on contrastive subjects and subordinate denials. The supplied-occurrence oracle passed all five. The 92/96 per-layout gate prevented a qualified timing comparison. The open research question is whether a compact parser can identify governing subjects and predicate scope before a reference is bound to a record.

## Replay the latest study

Use Python 3.12.14 in an isolated environment. Run these commands from this directory.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cd study040
OPENBLAS_NUM_THREADS=1 python run_study.py
python audit.py
python check_targets.py
```

The runner overwrites generated results in its own directory. Replay in a disposable checkout to preserve the committed evidence. Compare the generated `scores.csv` with the committed file. Gzip container hashes can differ on regeneration. The archived input includes frozen NumPy checkpoints and supplied expression boundaries and bindings. These experiments concern governance over typed COPY and LINK in synthetic instructions.

## Recent experiments

| Study | Question | Replay from the study folder |
| --- | --- | --- |
| [036](study036/README.md) | Does wider lexical context identify the governing clause? | `python study.py && python audit.py && python boundaries.py` |
| [037](study037/README.md) | Does scoring expression–approval pairs resolve reversed references? | `python experiment.py && python audit.py && python collision.py` |
| [038](study038/README.md) | Can a learned reference mapping improve on a dictionary? | `python experiment.py && python audit.py && python interventions.py` |
| [039](study039/README.md) | Do clause boundaries resolve competing mentions? | `python run_study.py && python audit.py && python intervention.py` |
| [040](study040/README.md) | Do argument cues resolve contrast and subordinate scope? | `python run_study.py && python audit.py && python check_targets.py` |

Each study folder contains its protocol, source, required frozen checkpoints and recorded score table. Replay regenerates fitted weights, raw predictions and audit outputs. The complete evidence package, CSM_BDH002_RESEARCH_HANDOFF.zip, includes these outputs, checksums and the 15 earlier archives inventoried in [ARCHIVES.md](ARCHIVES.md). Ask the repository owner for access to that package if you need the archived bytes. Older archives can require additional dependencies such as PyTorch or Lean.

## Earlier GitHub continuation points

- The [recovery guide in draft PR #65](https://github.com/FPC-effortless/cdl-attention-experiment/pull/65) indexes 008–019 and documents the unavailable 016 workspace and private dependencies.
- The [CONDITIONAL-019 draft PR #64](https://github.com/FPC-effortless/cdl-attention-experiment/pull/64) records its negative bounded result.
- [OUTPUT-MEMORY-021](https://github.com/FPC-effortless/cdl-attention-experiment/pull/68) and [PARALLEL-022](https://github.com/FPC-effortless/cdl-attention-experiment/pull/69) preserve their respective recovery and result records.
- Reproduction of private historical runs requires access from the repository owner to the materials identified in those recovery notes. The private BDH-001 and 021 evidence packages and the unavailable 016 workspace are outside this public handoff.
