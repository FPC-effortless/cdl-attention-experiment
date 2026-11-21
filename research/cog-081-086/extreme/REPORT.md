# COG-ATTN-087: ten-tier extreme puzzle benchmark — evidence report

**Result type:** verified problem generation and reference-control logic; **not evaluated COG-ATTN agent performance**.

Generated **210** cases across 10 tiers, with 40 restricted endgame chess, 40 uniquely verified Sudoku, 40 maze, 38 8-puzzle, 12 15-puzzle, and 40 canonical Hanoi attempts.

### Extreme / Frontier observations

- **Chess:** restricted K+Q/K+R vs K forced mate in up to four White moves, with universal quantification over every Black king reply. Position conditions have individually verified exact minimal bounds; no claim of general chess strategy.
- **Sudoku:** level 10 requests up to 58 blank cells. The four cases actually achieved **58, 56, 57 and 53 blanks** respectively. Generator guarantees uniqueness where proven and records MRV expansions; blanks are not themselves an adequate human difficulty metric.
- **Maze:** level 10 graphs are 151×151. Verified optimal routes span **394–478 moves**; BFS expansions span **11,840–12,067**. All 12 level 8–10 maze instances admitted an obstacle-shift witness, invalidated the old shortest route and allowed a valid replan.
- **8-puzzle:** the complete 181,440-state reachable graph proves exact distance **31** is the maximum. Exactly two states have this distance under the canonical goal. One was allocated to development and one to reserved; two other requested duplicates were excluded.
- **15-puzzle:** level 8/9/10 instances are solvable because inverse witness walks of 24/48/96 steps replay to the goal, but **none** was certified to require that many moves optimally.
- **Hanoi:** minimum moves at Frontier: `2^22-1 = 4,194,303`. This is a symbolic proof of minimum length only, **not evidence of an executed million-step policy**. Full reference paths were replayed through 12 discs; above 12 only prefixes were checked.

**Split audit:** all 170 non-Hanoi challenge inputs are unique across the two panels. The 40 canonical Hanoi entries intentionally repeat, and must be excluded from independent cross-instance generalization counts.

**Validation:** 10 local `unittest` cases passed, including exact 8-puzzle state-space enumeration, Sudoku uniqueness, adverse action rejection, no implicit success on missing predictions, chess mate horizon verification, reversible 15-puzzle witnesses and disjoint split check. Re-run and compare frozen case/evidence checksums before accepting hosted results.

**Research finding:** There is now a mechanically checked difficulty curriculum with progressively larger/harder tasks. It is not yet evidence that COG-ATTN acquired chess or puzzle-solving skills. Next experiment: a closed-loop COG-ATTN perception→memory→plan→act interface, with model training without test-time reference solvers, plus memory/skills ablations and matched-compute controls.
