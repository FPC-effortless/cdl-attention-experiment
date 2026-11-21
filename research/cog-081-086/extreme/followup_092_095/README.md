# COG-ATTN-092–095 — Difficulty-aware skill selection and genuine consequence feedback

**Qualification:** Benchmarked *partial agent subsystems*, not a full trained neural COG-ATTN replacement for attention. No supplied correct moves are used during training or evaluation, but foundational algorithms remain explicitly engineered. The exact distinction is part of the scientific result.

## Experimental lineage

* Based on PR #77, which extends #76's ten-level 210-task puzzle benchmark and #75's COG-066 macro acquisition controls.
* Puzzle inputs: `../results/cases.jsonl`, materialized from `../artifacts/puzzle-087-090-cases-evidence.zip` in the hosted workflow.
* Local module imports: `../learning/maze_transfer_091.py` (train from self-explored paths) and `../score_ladder.py` (validation); no hidden test reference paths are accessed by agent policies.
* The original `cog066_source.acquire()` runs in the 091 training/control acquisition path. COG-092 routes its self-explored source-root support table, not an end-to-end neural system. COG-093–095 are newly engineered, controlled follow-ups, **not original COG-066 itself**.

## Completed evidence

| Phase | Experience / test | Primary result | Inference decision |
| --- | --- | --- | --- |
| **092** | Admit local maze preferences conditionally, calibrate on development levels 6–7, evaluate four entirely new seed families (40 roots + 12 correlated shifts, budgets ×1 and ×2) | On 52 tests ×1: heuristic 36, learned 38, gated 38, corrected 38. At ×2: 44, 46, 46, 46 respectively. The development-selected correction coefficient was 0. | **Fail**: no gain from gate or correction; extra CPU overhead |
| **093** | Learn state/action tie preferences from 19 A*-self-solved development 8-puzzle episodes; exact A* same in both arms | On 25 held puzzle submissions: primitive/learned/shuffled all 22 solve (19/19 8-puzzle; 3/6 15-puzzle). Learned expands 49,927 vs 74,094 primitive; first implementation is slower. | **Fail** initial implementation's measured wall-time gate |
| **093b** | Fuse parent Manhattan/blank-state feature computation; three randomized evaluation repetitions | Same 22/25 success and identical search decisions; median complete batch **535.009 → 422.472 ms**; 74,094 → 49,927 states. Shuffled 575.108 ms / 73,723 states. | **Provisional scoped pass** for inference-only CPU batch; train & compile must be amortized |
| **093c** | 32 novel, nonoverlapping, exactly distance-certified 8-puzzles, depths 4–28; three matched repeats; model frozen | Primitive / learned / shuffled all **32/32**; states 38,950 / **18,699** / 37,944; medians 213.840 / **110.241** / 223.670 ms. Learned expands fewer states on 22 problems, ties 10, worsens 0. | **Scoped positive replication** within 8-puzzle, not evidence of novel search acquisition |
| **094** | Sudoku meta-selection from its own full puzzle completion results; training 20 development 4×4/9×9 puzzles | Held 20: first-empty 14, MRV 20, most-open 6, learned selector 20; 1,112 search nodes for learned and MRV. | **No distinct inference advantage**; selected supplied MRV search rule |
| **095** | Chess self-search terminal mate feedback; legal AND/OR adversarial search; training 12 development KQK/KRK positions | Held 20: all 20 solved for primitive, learned, shuffled. White move checks 142,712 → 140,523, but CPU ~544 → 1,531 ms. | **Fail** inference cost; not full chess |

## Cost and amortization

The fused sliding agent was trained using 19 self-solved development puzzles; the measured training search plus table build took **238.894 ms** on this local CPU. On the disjoint 32-puzzle cohort, its median advantage is **103.599 ms per 32-instance batch**, excluding training, generation and compilation. At an unchanged workload and host, the observed saving would cover that one-time training charge after roughly **74 puzzle evaluations**. This is a rough illustrative amortization calculation, **not** an externally validated throughput law. Report preparation/cold starts, compilation and cache expiry separately in deployed implementations.

Two key interpretation boundaries: the 32 puzzles come from the *same 8-puzzle domain* as development (new states, not new game rules); and training used **engineered A*** to self-discover successful episodes. The proposed cognitive controller learned tie preferences, **not A***, Sudoku constraints, or chess minimax.

## Severe difficulty and failure modes

* Level 10 mazes are **not reliably solved**. In 092, 4/8 Level 10 original/shift evaluations succeeded at the standard budget, 6/8 at 2× for the learned policy, and source/shift scores are correlated.
* 15-puzzle exact optimal difficulty is unknown. At the fixed 10,000-search-state cap, three of six reserved 15-puzzles remained unsolved in 093; no claim of Frontier 15-puzzle mastery.
* Chess is only K+queen/rook vs K, forced mates in 1–4 White turns; complete games, rules discovery and opponent modeling remain untested.
* The online failure-credit component was **rejected by calibration**. A zero correction coefficient is not evidence that online repair works.
* All measured timings are local single-host CPU observations; the hosted workflow verifies **decisions**, not timing numerics.

## Frozen evidence

`results/*.json` records complete per-task success, work, and CPU timing; `STABLE_SHA256.json` excludes the host-dependent timing fields only. `verify_stable.py` fails on any non-timing replay mismatch. The raw CSV/JSON and source are bundled in the companion evidence ZIP.

Run commands and dependency layout are in [REPRODUCE.md](REPRODUCE.md). Experiment and admission criteria are in [PROTOCOL.md](PROTOCOL.md). A concise interpretation is in [REPORT.md](REPORT.md).
