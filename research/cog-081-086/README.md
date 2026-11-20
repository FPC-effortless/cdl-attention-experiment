# COG-ATTN-081 puzzle benchmark · reference-control pilot

A deterministic local benchmark for **chess endgame mate-in-one, Sudoku, mazes, Tower of Hanoi and sliding puzzles**. It was created after inspecting COG-ATTN-071–080 in GitHub PR #74 (`FPC-effortless/cdl-attention-experiment`).

**This repository does not yet load or test COG-ATTN.** It tests puzzle generators, validators, exact reference solvers and a deliberate stale-plan regression. In the next phase, wire the COG-ATTN agent to the JSONL observation/action contract without giving it access to reference algorithms.

## Run

Python 3.10+; runtime depends only on the standard library. Pytest is optional for invariant checks.

```bash
python run_pilot.py --out results.json
python -m pytest -q test_puzzles.py
python evaluate_submissions.py puzzle_cases.jsonl my_cog_attn_actions.jsonl --out my_scores.json
```

`results.json` has per-instance work, solution lengths, wall timings and shifted-maze traces. `puzzle_cases.jsonl` contains puzzle observations and no reference actions. `PROTOCOL.md` contains the 081–090 experiment plan and honest restrictions. The reproducible input path deliberately uses a fixed seed protocol; benchmark sources are shipped for auditability, not to grant the test-time agent solver access.

## Pilot findings (not model results)

- 396/396 reference paths replayed to valid goals across 11 domains/conditions.
- 258/396 were solved by the heterogeneous shallow heuristic controls.
- All 396 exactly repeated tasks could replay their own verified cached plans (per-episode cache, not learned generalization).
- On 36 specially constructed maze shifts, a stale plan failed in every case, version-sensitive lookup rejected it in every case, and BFS replanning recovered a valid route in every case.
- 326 unique input states. The 70 duplicates are **intentional canonical Hanoi repetitions** (3-disc and 5-disc, one state for each). Accidental non-Hanoi overlaps were eliminated before scoring. No independent generalization is claimed from these repetitions.
- No COG-ATTN inference accuracy, training gain, speedup, skill discovery or cross-domain transfer was measured. The shallow heuristics are not a calibrated model baseline.

## Known limitations and next research task

1. Mate-in-one only covers K+Q/R versus K; full legal chess and authentic Lichess suites remain necessary.
2. Maze observations are completely visible; moving obstacles in the pilot appear as an explicit state change. Partial observation and probe acquisition are not implemented.
3. A 100% reference score is generator/validator certification, not evidence that a neural model has acquired those skills.
4. Exact replay improvement is on identical states only. No candidate retrieval generalization is demonstrated.
5. Unlike model evaluation, test and reference code share a file tree in this transparent pilot. Qualification requires process-level separation or sealed access.
6. Timings are local CPU observations with no formal CPU benchmarking standard; do not compare them with COG-ATTN or GPU measurements.
7. The 3- and 5-disc Hanoi repetition cases must be recast as a **3→5 disc skill-transfer study** with novel conditions; repeated fixtures must not be counted as independent evidence.

**Integration priority:** create a generic `observe → choose valid action → execute → receive feedback → update memory` interface on the actual COG-ATTN controller, then compare a no-skill arm, exact replay, verified composed skills and reference planning under matched competence and fully charged inference latency.

## Executed COG-ATTN-086a/b subsystem bridge (2026-11-21)

`bridge/experiment_086.py` evaluates the **actual COG-066 verified-option acquirer** on previously unseen mazes, sliding puzzles and Hanoi tasks using a new exact A* adapter. The acquirer is isolated in `bridge/cog066_source.py` and comes from the pinned upstream research code. This is a genuine *subsystem* transfer experiment, not a full COG-ATTN neural/controller evaluation.

Training (84 supervised demonstration paths) uses primary seeds 8101/8107; validation (42 tasks) uses primary seed 8113; final test uses 126 instances on replication seeds 9101/9107/9113. Options were rejected for all seven domains by a strict pre-heldout efficiency gate. The **ungated** option arm took 88,206 primitive evaluations versus 20,560 for the exact baseline, with no reduction in expanded states, and matched success at 126/126. See `bridge/REPORT.md` and `bridge/086-results.json` for exact counts, per-task traces and qualifications.

```bash
python bridge/experiment_086.py
python -m pytest -q test_puzzles.py bridge/test_086.py
```
