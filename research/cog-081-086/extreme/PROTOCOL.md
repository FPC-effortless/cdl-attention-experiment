# COG-ATTN-087–090 protocol — calibrated puzzle difficulty escalation

## Objective and null hypothesis

Test whether COG-ATTN can acquire progressively difficult reasoning skills under bounded compute and transfer to unseen instances. Null: gains are attributable only to memorized examples, oracle search, action templates, or larger compute. Reference environment certification is separate from policy qualification.

## Difficulty axes (do not conflate)

1. **State complexity:** size, number of unconstrained variables, branching factor, and partial observation.
2. **Solution horizon:** verified optimal move length or proven lower bound.
3. **Search / deduction effort:** expansions, MRV decisions, backtracks, invalid actions, and measured elapsed time.
4. **Robustness:** changed world, adversarial replies, no-goal and inconsistent-information cases.
5. **Economic cost:** full episode wall time; memory reads/writes, inference, search, verification, tool use, corrections and failed attempts.

A level number is a *curriculum setting*, not a calibrated equivalence between different domains or an IQ score. Store both requested level and independently measured metrics. Do not equate 15-puzzle scramble length to optimal path length, Sudoku blank count to human difficulty, or chess endgame depth to full chess playing strength.

## Strict train/test protocol

- Development seeds `8701, 8707`, reserved seeds `9701, 9707`, with disjoint non-Hanoi input IDs and no reserved metric access during training/selection. Full enumeration ensures an irreducible two-position maximum-depth 8-puzzle cohort; one state per split.
- Train progressively, but evaluate periodically on locked, unseen instances at each level. At least one held-out new topology/encoding family is required before asserting transfer. Do not adapt policy or difficulty thresholds after looking at reserved performance.
- Control for action opportunities, feedback, solver availability and total training compute. A reference oracle can certify problems; it cannot be invoked by the evaluated learner at test time.
- Count canonical repeated Hanoi setups as a procedural scaling control only; four repeated sessions are not four independent generalization samples.
- Represent dynamic-maze shift with old/new environment digests and explicit invalidation. Evaluate changed route and unsafe stale-route baseline separately. If all valid reroutes are unavailable, exclude and disclose the task, not an impossible success label.
- For chess >1, require a state-contingent strategy that withstands every legal Black reply, not just a single principal variation. Current implementation only supports KQK/KRK 1–4 White moves.

## Difficulty auditing and failure policy

- Sudoku: uniqueness is verified under a bounded solution enumerator. Reject deletions where uniqueness cannot be established within budget. Record achieved blanks, guesses, search nodes, and budget exhaustion; if threshold is not reached, downgrade measured demand rather than claiming the target was met.
- 8-puzzle: all 181,440 reachable states are enumerated backwards from the canonical goal, certifying each selected exact distance up to the graph diameter 31. Exclude duplicate 31-move fixtures.
- 15-puzzle: construct only via reversible legal walks and verify a return path. Do not claim an optimal-distance certificate, or include it in an optimality-weighted aggregate.
- Hanoi: minimal actions are exactly `2^n-1` from the three-peg recurrence. Fully replay only <=12 discs; above that, check a prefix and explicitly label complete execution as untested.
- Maze: generate connected graphs, solve by independent BFS, replay the solution, and test dynamic replans against the changed state. Track route length and BFS expansions; monotonic increase in size does not guarantee monotonic instance difficulty, so publish measured values.
- Chess: prove exact minimum *White move* bound 1–4 by checking smaller depth failure and exhaustive opponent responses. Black heavy-piece capture is a draw, not a win for White; stalemate is not mate. This is a bounded endgame subgame.

## Admission gates for a future COG-ATTN model run

- Zero illegal executed actions in verified environments (invalid proposals may be blocked by the verifier, and must still be reported).
- At each level, report task successes / eligible attempts, coverage, abstentions, timeouts, revisions, and independent seeds. At least six independent held-out seeds are needed for inferential claims; the current two-seed reserved pilot is insufficient.
- Learning from interaction: positive held-out advantage over a no-memory control and matched compute-only control. Demonstrate skill reuse on a different starting state, problem size or latent rule, not exact-state replay.
- Frontier: explicit per-task wall-time and action budgets. Report resource exhaustion separately from logical impossibility and evaluator limits. For long Hanoi introduce streaming actions and checkpointed verifier before any full success claim.
- Inferential statistics: per-seed paired intervals, not per-puzzle independence; show failed as well as passed difficulty tiers.
- Preserve a matched full self-attention comparator and comparable competence before reporting any speed claim.

## Not yet implemented

Full chess rules or external FEN-rated positions; interactive partially-observed maze perception; full 4×4 sliding optimal-distance proof; agent training; million-step streaming action execution; end-to-end COG vs Transformer inference comparison. These are the next experiment requirements, not achievements of this benchmark extension.
