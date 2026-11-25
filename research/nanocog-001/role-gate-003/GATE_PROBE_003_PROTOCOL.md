# ROLE-GATE-003: finite simplex reachability (descriptive)

ROLE-GATE-002 failed its predeclared paired support gate; no matched-dose
intervention was run. This distinct engineering diagnostic asks
**reachability**, not direction, necessity, or effect attribution:
can changing only four final Layer-2 offset gate scores produce a
correct answer in the frozen model at any of 11 fixed points?

Reuse exactly the previously exposed 32-family synthetic panel and frozen
ROLE-002 quarter-prefix / clip1 update-1280 gated32 checkpoints, weak1371
and strong1171. This is not a fresh holdout, language claim, model repair,
training allocation, or promotion. No protected MAIN/TEST or alternate
data. CPU FP32, one thread, ≤45 s per seed, one attempt each, no retry.

## Fixed intervention

The grid contains 4 corners (one lag), 6 midpoints (two lags) and 1
uniform point, in deterministic numeric-index order with lag index 0
first (oldest/lag3), through index 3 (current/lag0). For intended
proportions z use p=0.0001 + (1-4*0.0001)*z per coordinate. Convert to
natural log gate scores in FP32; only raise floor-coordinate log scores
by the smallest verified number of ULPs needed for actual FP32 softmax
weights ≥0.0001, maintain absolute per-coordinate tolerance 2e-7 from
the intended point and normalized sum within 2e-7. Report realized
weights/rounding. No answer, candidate truth, or predicted logit enters
the grid. Fixed numeric branch replaces only last token scores at
the second layer; every other token, residual, value path and frozen
model tensor unchanged.

For each 8-row batch run native capture, exact same-row overwrite
control, the 11 choices, and exact native repeat after hooks removed,
448 forwards per seed. Assert all other token logits exact, same-row
and repeat full parity, and checkpoint tensor immutability. Save only
native and 11 grid-branch final logits, all 287 float32 classes; no
truth staged, no optimizer. Seal both successful native receipts and
artifacts before a separately reviewed judge. No inference or work on
directional comparisons is authorized by the staging packet.

## Prospective readout contract

- Report native plus per-grid-choice strict ranks and correct-color vs
  paired foil, failing ties. Track rescues and damages relative to
  native per fixed condition, especially weak baseline's wrong rows,
  target first/second, and strict both-query pairs under a single choice.
- **Row reachability bound:** among weak native error rows, report how
  many **any** fixed candidate rescues; separately give the upper
  benchmark formed by allowing native or any grid choice per row.
  These are answer-conditional oracle bounds for this finite grid, not
  a learned or supplied routing policy.
- **History reachability bound:** both queries for the same first-six-
  token history must be correct under **the same fixed choice**;
  count histories where at least one grid choice succeeds, and
  separately count native-or-any-shared-choice as a finite upper
  benchmark. Never form both-correct hits via different row-wise grid
  choices.
- **Single global fixed policy:** publish every candidate's standalone
  strict/conditional counts and the retrospective best across the
  same panel, with paired rescues/damages and ties disclosed. A best
  existing choice on these rows does not generalize or justify any
  newly fitted policy. Native can beat every fixed or history-shared grid.
- Interpreting a low upper bound is limited to these tested options;
  it is not inability over the full continuum/other components. A large
  oracle vs global gap is consistent with context-dependent usefulness
  of choices, not proof that a deployed controller could choose them.

Pins: exact ROLE-GATE-001 exposed actor and separate judge are
44f380f0e934450e3c809162578381e670683602a4e52a4487c89fd2197a8116 and
d5160ddcee95efcc28e6a9e042485cd9469ddae2827802e81efbdeea3cd1b848.
Checkpoint SHA-256: strong1171
b2cbd5aa6cb1acd345f54bd562ece11e6d85d696b86cd87adc4cb13f8d3c810c,
weak1371 517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890.
