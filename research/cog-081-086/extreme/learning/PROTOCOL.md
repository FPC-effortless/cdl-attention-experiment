# COG-ATTN-091 · Experience-conditioned maze skills (scope-locked after initial pilot)

## Question
Can a lightweight, local-perception skill controller trained only on its own verified puzzle experiences outperform a strong untrained, goal-directed explorer on unseen harder mazes, including changed worlds? Compare outcome, action budget and CPU latency separately. Do not treat this as a full neural COG-ATTN model.

## Interfaces and training
- Input at each step: current cell, goal coordinate, four immediate passable neighbor actions. The controller does not query full walls, shortest path, global BFS, or the gold completion trace. The trusted environment does expose the `Maze` object to the verifier, which checks final traces with `apply()`.
- Source experiences: development seeds 8701/8707, difficulty levels 1–5 only, ten world roots. The exploratory controller finds successful routes via local exploration with backtracking and stores loop-erased, verified executed trajectories. No teacher-provided shortest paths.
- Admission: a feature-action association must occur on at least two independent source-world roots, not merely many frames from one root. The full local feature is sign of target row/column displacement plus current 4-neighbor legal-action mask.
- The pinned unmodified COG-066 `acquire()` is also run on the verified traces, to expose the number of learned action fragments. The fragments are **not** executed by this policy; their presence is not evidence of model skill application.
- Calibration: development levels 6–7 select the fixed alpha from {0, 0.25, 0.75, 1.5, 3} on successes, actions and alpha in that order. No reserved records are read at fit/calibration time.
- Closed control: each task budget is max(80, 8*n+64) physical legal moves, including backtracking. Failures exhaust the budget, never count as solved. Full trajectory verified by environment. Timing is policy execution, not training or environment generation.

## Comparators
1. Goal-directed DFS with full episodic visited-cell memory and zero learned features (heuristic control).
2. Same DFS and memory, plus admitted feature-action preferences (learned).
3. Same-size preferences with each action identity cyclically shifted (shuffled association control).
4. Same DFS/memory and random unvisited-neighbor ordering, seeded per world (random control).

## Evaluation and replication
- Original sealed run: reserved seeds 9701/9707, ten levels each (20 world setups), plus six independently scored obstacle-change replans = 26 decision tasks. Shifted setups from same root are correlated; never count as independent new environments.
- Fresh follow-up: additional new seeds 9803/9809/9811/9817 on same generation family, 40 maze roots and 12 shifted versions = 52 decisions. The policy and alpha are frozen from development before generation/scoring. This fresh-seed panel is post-initial replication, not a separately preregistered external investigation.
- Exploratory post hoc budget sensitivity: among fresh levels 8–10, compare budgets x1/x2/x4. This does not retune the policy, and is not confirmatory.
- No claims about solved chess/Sudoku/sliding/15-puzzle/Hanoi tasks; no end-to-end neural COG-ATTN controller; no general AI/attention replacement inference. Both candidate and control receive identical per-step sensor and episodic memory. The verifier and generator are privileged, but the policy is not.

## Gate
A candidate is economically promotable only if it demonstrates stronger held-out task completion at a fixed action budget **and** lower full end-to-end inference cost at equivalent competence after amortizing training and verification. Otherwise retain the heuristic control and regard observations as exploratory. Keep disaggregated tiers and failed cases; do not aggregate correlated shifted variants as independent samples.
