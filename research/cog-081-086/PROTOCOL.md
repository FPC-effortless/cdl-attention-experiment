# COG-ATTN-081–090 · Puzzle Intelligence Challenge — frozen pilot protocol

**Research status:** puzzle fixtures, executable reference solvers, deterministic primary and replication cohorts, and a standalone answer scorer are implemented. **COG-ATTN is NOT connected or evaluated.** This is a local puzzle *benchmark* pilot, not evidence of cognitive competence. Its task solvers are handwritten ground-truth controls.

## Scientific question

Can COG-ATTN acquire and retain action-producing reasoning skills from legal interactions and sparse feedback, then transfer them to unfamiliar problems **without preinstalled puzzle solvers or a fixed language readout**? Distinguish faster replay from novel competence; distinguish legality, goal success, skill verification, and authorization. Charge all stages of inference and recovery.

## 081–090 experimental program

| Experiment | Intervention / scope | Falsifier and required comparison |
| --- | --- | --- |
| 081 | Generator, interpreter, and independent validators | No illegal reference paths; puzzles not silently malformed; input/answer split and duplicate audit |
| 082 | Chess K+Q/K+R mate-in-one; expand to full legal chess | Legal move and exhaustive opponent-reply check. Later compare tactical generalization on disjoint Lichess themes and ratings |
| 083 | Sudoku 4×4 and 9×9 | Every row/column/block valid; uniqueness; track contradictions, guesses, and work |
| 084 | Fully visible 7×7/11×11 maze; change a wall after planning | Compare shortest paths, memory-reuse, unguarded stale path, verified cache invalidation and replan |
| 085 | Hanoi and 8-puzzle | Measure discovery of subgoals and transfer from 3 to 5 discs, shorter to longer scrambles; do not count repeated same setup as new generalization |
| 086 | Experience-driven skill acquisition | Learner gets state, legal actions, feedback, rewards and optional demonstrations on acquisition cohort only. No gold solution/search oracle at evaluation time |
| 087 | Causal and perceptual uncertainty | Add partial-observation mazes, reversible probes, keys/doors, hidden environment rule changes, no observation vs informative probes |
| 088 | Skill composition and counterfactual repairs | Try learned procedures on novel goals and sizes, verify candidate skills on isolated state, reject invalid ones and measure rollback |
| 089 | Complete inference-cost gates | CPU and GPU where available; elapsed from observation to terminal state includes encoding, routing, retrieval, search, probes, validation, memory writes, recovery and execution. Compare exact-search and Transformer/RL baselines at matched competence |
| 090 | Sealed replication and independent continuation | Fresh puzzles, disjoint seeds and parameterizations; publish checksums, failures, per-instance traces and bounded conclusions |

## Pilot configurations actually implemented

- Six deterministic seeds: **8101, 8107, 8113** primary; **9101, 9107, 9113** replication. Six instances per domain per seed, 11 conditions, 396 total trials.
- Restricted chess: white king + queen **or** rook against black king; white to move; positions with exactly one mate in one. No pawns, castling, promotion, threefold, full-game strategy or multi-piece chess. This is a narrow chess subgame, not complete chess.
- Sudoku: 4×4 and 9×9 with a single counted solution, generated from legal solved boards with clue removal and uniqueness checks.
- Maze: 7×7, 11×11 and 11×11 revisable maps, fully observed; solve via BFS; shifted paths intentionally invalid but a replacement path exists. Partially observed mazes are future work.
- Hanoi: canonical three- and five-disc tasks, explicitly repeated across seeds. They are **not** independent held-out task instances. Transfer to unfamiliar counts is future work.
- Sliding puzzle: 3×3 at scramble lengths 8 and 14, scored with A* Manhattan search. State and exact path costs measured.
- All non-Hanoi states are deduplicated across the primary and replication cohorts. A direct cache-hit on a previously unseen task is disallowed by per-episode cache isolation in this pilot; its 0% unseen figure is **by construction**, not evidence of any learning system's limitations.

## Baselines and gates

1. **Reference search**: preprogrammed BFS, A*, Sudoku MRV DFS or exhaustive mate-in-one enumeration, used only by the **evaluator** as an upper-bound competence control. No COG-ATTN access permitted.
2. **Simple heuristic**: domain-specific, shallow handcrafted rule (first candidate, Sudoku singles, goal-directed unvisited route). This is a weak diagnostic, not a uniform random baseline.
3. **Exact episodic replay**: cache exact verified action sequence for one repeated state. Charge hashing and replay validation. No novel transfer implied.
4. **Unguarded stale path**: when maze geometry changes, wrongly reuse the cached action sequence keyed only by task identity. Compare to state-bound guarded retrieval and explicit replanning.
5. **Actual COG-ATTN**: NOT RUN. Requires integration of its policy/observation and memory interfaces with the puzzle adapters, acquisition and retained memory lifecycle.
6. **Later Transformer / RL controls**: NOT RUN. Give all agents identical observation/action interfaces, training budgets, step caps, and timing harnesses.

**Precommit criteria for later model run** (hypotheses, not observations): (i) zero illegal executed actions; (ii) measurable positive held-out success over a no-memory and training-budget-matched control; (iii) improvement on novel puzzles after skill acquisition, not merely on exact repeats; (iv) 100% rejection of invalid stale paths in this controlled shift cohort; (v) detailed replan costs and matched-competence end-to-end inference; (vi) holdout integrity with no solver access or test-solution leakage. Use paired per-seed bootstrap intervals; never treat correlated puzzles within one seed as independent replicates.

## Action submission and scoring

The file `puzzle_cases.jsonl` provides non-oracle puzzle observations, IDs, domains and input digests. Submit a separate JSONL file with `{"id":"...", "actions": [...]}`. `evaluate_submissions.py` scores it by replaying actions in a fresh environment. Action format:

- Chess: `[["Q", from_square_index, to_square_index]]` or `[["R", from, to]]`, or `[["K", from, to]]`, with a1=0 and h8=63.
- Sudoku: `[[zero_based_flat_index, digit], ...]`.
- Maze and sliding: `['U','D','L','R', ...]`.
- Hanoi: `[[from_peg, to_peg], ...]`.

Missing predictions count as unsuccessful tasks, not as correct abstentions. The evaluator reports coverage and success separately. Add a closed-loop, turn-by-turn judge for partial-observation challenges in 087. Agents must not import `puzzles.py` or other reference-solver code in real qualification runs; the pilot ships the reference solvers openly for independent validator inspection only.

## Suggested authentic external suites (not downloaded or evaluated)

- Lichess official rated puzzle export: https://database.lichess.org/#puzzles — filter tactical motifs and rating strata, hold puzzle families out, then independently verify full legal chess rules.
- Farama MiniGrid: https://minigrid.farama.org/ — door/key tasks, local observations and environment shifts.
- Sokoban: https://github.com/EnvCommons/sokoban — push-planning, deadlocks and irreversible actions.

## Safety and provenance

No GitHub writes, live issues, external services, paid compute or external model inference were exercised by the pilot. The branch `research/cog-071-080-graded-commitment` contains COG issue-workflow components, but no general chess/Sudoku/maze adapter. Running its issue text classifier or workflow planner against puzzle states would not constitute puzzle competence. Separate primary and replication data and publish failure traces; preserve negative results and do not market verifier success as learned intelligence.
