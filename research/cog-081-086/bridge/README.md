# COG-ATTN-086a/b · Puzzle option transfer

Runs the **actual COG-066 `acquire` implementation** extracted from the pinned upstream COG-ATTN repository, then applies the acquired verified action macros to new generic A* puzzle adapters. The learned macros are not a neural controller and cannot solve puzzles without exact search. This is a subsystem transfer and negative-control study, **not full COG-ATTN inference**.

The other puzzle cases (restricted chess mate-in-one and Sudoku) remain benchmark-only. Chess, Sudoku and Hanoi procedure discovery by a COG policy were not demonstrated.

## Reproduce

From the extracted package root with Python 3.10+ and no third-party runtime dependencies (pytest optional):

```bash
python run_pilot.py --out results.json
python -m pytest -q test_puzzles.py bridge/test_086.py
python bridge/experiment_086.py --out bridge/086-results.json
```

The `run_pilot.py` call creates deterministic JSONL instances and logs. The experiment uses primary seeds 8101/8107 as demonstration acquisition, held primary seed 8113 for cost-based admission calibration, and **replication** seeds 9101/9107/9113 strictly for final evaluation. The exact same puzzle cases and randomized arm orders are used for primitive, COG options, option-shuffled, and calibration-admitted arms. Training demonstrations are supplied by the reference solver; the test-time search never calls the reference solver.

The acquired options are copied from `research/cog-071-080/vendor/packet/planning.py` at `975732f1081d4e5c98c460df1cfa067c48af213d` (blob SHA `47257edde1e7700f14ea63f46fe541a1b5d999ad`). `acquire` itself is unchanged; the generic puzzle search and safety checks are additions in this study. Keep both mechanisms separate in descriptions.

## Gates

- Demonstrated plan is admitted to acquisition only after reference success and >=3 independent input-source families.
- Search options are replayed one primitive at a time; any illegal submove rejects the entire candidate without committing it.
- Source state changes invalidate plan hashes. Exact search remains available even if all options fail.
- An economic option gate is calibrated **before** replication: allow a domain's options only if the dev cohort has equal competence, strictly fewer primitive evaluations and strictly less measured time. Include training and calibration times in total-cost accounting, not just test-time execution.
- No COG language policy, control head, RL learner or neural model was run. Exact A* remains an oracle-capable solver constructed by the evaluator. Therefore, success cannot be attributed to autonomous COG problem-solving.

## Key negative result

With 12 distinct primary solution traces per domain (other than the repeated Hanoi setup), maze and sliding macros were discovered. None passed development cost admission. Across 126 held puzzle instances, the ungated learned-option arm solved every test, but expanded exactly as many states as exact search while attempting substantially more primitive actions. The guarded arm correctly selected exact search in all tested domains. Report raw per-instance timing variation; the primary reliable economic effect is on deterministic operation counts, not small wall-time differences.

Main findings and exact totals: `bridge/086-results.json`, plus `bridge/REPORT.md`. See `PROTOCOL.md` for the broader 081–090 roadmap.
