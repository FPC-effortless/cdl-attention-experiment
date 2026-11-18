# SKILL-006 protocol review

Two separate Codex subagents in the same session reviewed the protocol at e3391d7. Neither accessed six-record results for this study. These are internal technical reviews.

## Scientific review

The scientific reviewer approved a bounded, exhaustive action-cost comparison subject to the clarifications below. The reviewer recomputed the four controller totals and every hindsight minimum from the 256-row development archive.

- Model size: one multioutput CART with depth at most three and at most fifteen nodes. Four features count misplaced records, duplicated target values (record count minus unique target count), reciprocal pairs and directed three-cycles. Count each closed cycle once, allowing outside references into the cycle.
- Training duration: one deterministic fit on all 3,408 assignments of sizes three through five, with all three cost labels (10,224 labels). Freeze the fitted tree before evaluating the new controller on six records. Use primitive, SWAP, ROTATE3 tie priority.
- Dataset: enumerate all 46,656 six-record assignments. This size is held out from SKILL-006 fitting. Study 005 previously used the same assignments with different controllers. Report results for permutations and repeated-output assignments.
- Training regime: infer each procedure once from its provided trace. Every controller uses the same primitive fallback and first eligible lexicographic binding, with at most one skill call. Compare the three fixed arms, longest-cycle-first (eligible ROTATE3, then eligible SWAP, then primitive), and per-goal hindsight. Report arm selection and binding checks.
- Interpretation: primary cost excludes routing and binding search. A strict win supports only the finite-roster weighted execution claim. A latency or broader capability claim needs additional evidence.

## Pipeline review

The pipeline reviewer conditionally approved research execution with the frozen fitted tree, complete pinned inputs, the actual TrainLab preflight and post-run reconciliation, and measured execution counters. Claim eligibility is bounded by the declared conditions. The six-record outcome must be published even if the proposed selector misses its gate.
