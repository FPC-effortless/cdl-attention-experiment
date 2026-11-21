# COG-ATTN-087–090: graded puzzle challenge

This follow-up to draft PR #75 specifies a ten-level curriculum from **Novice** to **Frontier**, spanning chess KQK/KRK endgames, uniquely verified Sudoku, mazes, 8-puzzle, 15-puzzle and Tower of Hanoi.

The full generator, scorer, tests, 210 local test cases and reference traces have been produced and passed local invariant checks. **This GitHub commit publishes only the frozen tier specification and protocol summary**; the local implementation and full evidence are **not yet present in this GitHub branch**. No hosted reproduction is claimed.

## Frozen difficulty and evaluation notes

* 8-puzzle optimal distance is certified by exhaustively enumerating all 181,440 states; at the maximum distance 31, only two instances exist, one per split.
* 15-puzzle scrambles have verified reversal witnesses; scramble length is **not** a certified optimum.
* Hanoi requires `2^n - 1` moves. At 22 discs this is 4,194,303 actions; full action-stream execution is still needed.
* The bounded chess verifier proves forced mate in 1–4 **White moves**, considering all Black replies. This is not general chess strength.
* Sudoku uniqueness is tested with an explicit solver work budget; achieved blanks and search effort must be reported separately from requested blanks.
* Level 8–10 maze shifts invalidate old paths but allow valid replans. A two-phase replan control does not prove partial-observation competence.

## Pending gates

1. Transfer the exact tested generator/scorer/tests and reference evidence; verify the published SHA-256 case/measurement digests.
2. Run an independent GitHub-hosted replay.
3. Integrate real COG-ATTN policies with action observation, memory, skill induction, corrective feedback, and a streaming evaluator for long tasks.
4. Run learning/no-memory/COG-066/search and attention-based matched-compute controls. Do not report reference-solver correctness as agent performance.

**Status:** specification published, local reference benchmark verified, model trial and hosted replay not yet performed.
