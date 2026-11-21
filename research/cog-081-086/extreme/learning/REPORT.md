# COG-ATTN-091 — Learning to navigate mazes from verified experience

**Status:** Tested narrow local-controller prototype; **not** the complete COG-ATTN architecture. The real pinned COG-066 `acquire()` algorithm is imported unchanged and discovers 16 reusable 2–4 action fragments; those fragments were intentionally **not executed** by this policy. The action preference learner is new and interpretable, not a trained neural network.

## Self-experience and admission

Ten development mazes at levels 1–5 were explored by a local wall-sensing DFS controller with episodic visited-cell memory. Every admitted training path was confirmed by the environment's final-goal validator. No test-time reference search or per-instance oracle path was available to the policy. These episodes required 564 executed actions. A source-root quorum admitted 29 feature-action associations across 21 local sensor contexts. Calibration on development levels 6–7 chose alpha **0.25**.

## First reserved evaluation: 20 unseen mazes + 6 correlated shifted worlds

| Local-control arm | Solved | Total physical actions | Total policy CPU ms |
| --- | ---: | ---: | ---: |
| Heuristic goal-directed DFS | 22/26 | 9,414 | 24.283 |
| Acquired local-action preferences | 22/26 | 9,442 | 28.539 |
| Shuffled preferences | 22/26 | 11,128 | 32.145 |
| Random exploration | 11/26 | 12,582 | 25.115 |

**Interpretation:** learned preference counts did not beat the stronger untrained heuristic at fixed budget in the original reserved run, and took more CPU time. It is weaker evidence than an improvement on fully matched tasks. The shuffled control incurs larger traversal cost, showing the learned features contain some directional information, but that alone is insufficient for promotion.

## Fresh, frozen-policy replication: 40 distinct mazes + 12 correlated shift replays

| Arm | Solved | Actions | Policy CPU ms |
| --- | ---: | ---: | ---: |
| Heuristic | 39/52 | 20,064 | 49.311 |
| Learned | 43/52 | 19,340 | 54.787 |
| Shuffled | 41/52 | 19,812 | 61.957 |
| Random | 20/52 | 25,572 | 53.980 |

On this post-initial fresh-seed panel the learned controller rescued **two distinct level-9 maze roots**, each in both its original and shifted form, producing a four-decision gain (43 vs 39) but only two independent root gains. Because this was a small deterministic same-generator replication, it is not a statistically established improvement. Crucially, learned was **~11% slower in aggregate policy CPU time** than untrained, even before training was charged, so it fails inference-economic promotion.

**Frontier failure remains:** at level 10, the fresh panel yielded **0/8** completions for learned and heuristic at the fixed budget; the shuffled control happened to solve 2/8. At level 9, learned solved 8/8 vs 4/8 for the heuristic. Difficulty transfer is uneven.

## After-the-fact budget sensitivity, tier 8–10 only (24 correlated tasks)

| Per-case action budget multiplier | Heuristic solved | Learned solved | Heuristic actions | Learned actions |
| --- | ---: | ---: | ---: | ---: |
| 1× | 12/24 | 16/24 | 17,610 | 16,830 |
| 2× | 20/24 | 18/24 | 25,062 | 26,190 |
| 4× | 22/24 | 18/24 | 31,982 | 41,454 |

A **candidate budget crossover** appears: the preference prior helps under tight action budgets but steers the DFS into longer detours as the budget expands. This is exploratory, not a pre-registered discovery; the result is sensitive to task family and seed. A plausible next intervention is state-dependent, risk-aware skill admission with cheap online counterfactuals, not blanket preference weights.

## Validation and boundaries

Five local unit tests exercised memory/liveness, disjoint roots, no implicit success, support quorum and strict action budgets. All four arms use the same legal local action observations and episodic memory. Per-episode wall time excludes generator/reference cost; the model is small and CPU timing may shift across hosts. Use `verify_stable_091.py` to reproduce stable decisions instead of literal timings. Do not label the observed performance as COG-ATTN solving arbitrary extreme puzzles, chess, Sudoku or 15-puzzles. A full trained COG-ATTN agent and a broader external puzzle benchmark are still required.
