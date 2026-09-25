# PNDS-GATE-001 Stage 3b — Result Record: F1 Triggered (Negative Result)

> Status: **FALSIFIED under the pre-registered criteria.** No PNDS claim is
> promoted from Stage 3b. The environment is sound; the learning protocol is
> not sufficient at this key-space size. This record is written after the
> pre-registration (`3a03225`), Amendment 1 (`528ad89`), and the property tests
> (`2a53338`), and is the first genuine negative result of the PNDS program.

## 0. Summary

Stage 3b asked whether a learned router must read **key-indexed persistent
state** to route correctly. The answer the experiment returns is:

- **The environment is sound.** Every control behaves exactly as predicted.
- **The state is provably informative and provably readable.** An oracle that
  reads the true key block scores **1.0000**; a router trained by REINFORCE on
  a *single fixed* key scores **1.0000**.
- **But REINFORCE on sparse binary reward cannot acquire 70 independent key
  blocks at feasible scale.** The learned arm sits at chance (0.15) and does
  not beat the stateless arm.
- Under criterion **F1** of `3a03225` §7 — "`learned` does not beat
  `learned_no_state` by a margin that is positive on a pre-specified majority
  of seeds" — **the stage is falsified.** Measured: **4 / 10** seeds positive,
  mean margin **−0.011**, binomial one-sided p = **0.581**.

This is a **learning-protocol boundary**, not evidence that persistent state is
uninformative. The distinction matters and is proven in §3.

## 1. Headline numbers

Primary configuration: `dim = 8`, `k = 4` (70 keys), `T = 6`, `n_candidates = 8`,
chance = 0.125, 250 training streams × 8 episodes, 300 held-out test streams.

| Arm | success | vs chance | role |
|---|---|---|---|
| `learned` (REINFORCE, reads state) | **0.1500** | +0.025 | **the arm under test** |
| `learned_no_state` (same budget, state withheld) | **0.1333** | +0.008 | F1 comparator |
| `true_key` (oracle reads true key block) | **1.0000** | +0.875 | representability ceiling |
| `wrong_key` (`do(K ← K')`) | **0.1667** | +0.042 | causal control |
| `corrupt_key` (`do(t ← t')`) | **0.1400** | +0.015 | causal control |
| `t_static` (reads `t`, ignores `K`) | **0.1133** | −0.012 | leak control |
| `static` (agreement with query) | **0.1300** | +0.005 | shortcut control |
| `anti_static` | **0.1067** | −0.018 | shortcut control |
| `random` | **0.1600** | +0.035 | floor |
| `oracle` (gold index) | **1.0000** | — | ceiling |

Intervention effect `true_key − wrong_key` = **0.8333**, `true_key − corrupt_key`
= **0.8500**. Both far above the F2 threshold of 0.30, so **F2 passes**.
`static` and `anti_static` are within 0.02 of chance, well inside the F3 bound of
0.10, so **F3 passes**. **F1 fails.**

Note that several arms sit slightly *above* chance and `random` sits at 0.1600.
With `success` defined as relational satisfaction and only one episode per test
stream, the sampling spread at n = 300 is roughly ±0.02 (sd ≈ 0.035); the
positive-looking offsets are noise, and `random` at 0.16 confirms the band.

## 2. The F1 test, properly: ten seeds

F1 is stated as a majority-of-seeds criterion, so the single-seed margin of
+0.0167 was not sufficient evidence either way. Re-run over 10 independent
training seeds (250 streams, 100 held-out test streams each):

| seed | `learned` | `learned_no_state` | margin |
|---|---|---|---|
| 0 | 0.1800 | 0.1700 | **+0.0100** |
| 1 | 0.1700 | 0.1100 | **+0.0600** |
| 2 | 0.1000 | 0.1900 | −0.0900 |
| 3 | 0.1300 | 0.0800 | **+0.0500** |
| 4 | 0.1200 | 0.1400 | −0.0200 |
| 5 | 0.1600 | 0.1500 | **+0.0100** |
| 6 | 0.1500 | 0.2100 | −0.0600 |
| 7 | 0.1600 | 0.1700 | −0.0100 |
| 8 | 0.1700 | 0.2000 | −0.0300 |
| 9 | 0.1800 | 0.2100 | −0.0300 |

Positive margins: **4 / 10**. Mean margin **−0.011**. Exact one-sided binomial
test against H0: p = 0.5 gives p = P(X ≥ 4 | n=10) = 595/1024 = **0.581**.

This is indistinguishable from a coin flip. **F1 is triggered**, and the
persistent state contributes nothing *measurably* to the learned router under
this protocol.

## 3. Why this is a protocol boundary, not evidence against persistent state

Three facts, all measured on the corrected implementation, jointly pin the
failure to the learner rather than to the substrate:

1. **The relation is exactly representable in the basis.** The `true_key` oracle
   scores **1.0000** with margin exactly 2 per `528ad89` §3's algebra
   (`2a − T` = +2 for gold, ≤ 0 for every distractor).
2. **REINFORCE learns it when the key space has one element.** Training on a
   single fixed key for 3000 streams reaches **1.0000**.
3. **REINFORCE does not learn it across all 70 keys at any tested scale or
   learning rate.**

Scaling the number of streams (1k → 16k) does not help:

| streams | 1,000 | 2,000 | 4,000 | 16,000 |
|---|---|---|---|---|
| `learned` | 0.1400 | 0.2050 | 0.1700 | 0.1800 |

And a learning-rate sweep at 3,000 streams rules out a tuning artifact:

| lr | 0.05 | 0.20 | 1.00 |
|---|---|---|---|
| `learned` | 0.1750 | 0.1600 | 0.1500 |

No setting learns. The interpretation is credit assignment: reward is a single
sparse binary bit per episode, and the correct gradient is spread over
`1 + 70·8 = 561` weights with only one 8-dim block active per episode. Each key
block sees roughly 1/70 of the updates, so per-key effective sample size is
~4 episodes per block at 250 streams × 8 — well below what a 2-margin
discrimination needs under stochastic REINFORCE. The signal exists (fact 1) and
is acquirable when isolated (fact 2); it is not acquirable in aggregate at
feasible scale.

**This is a real and reportable boundary condition.** It is not a license to
conclude that persistent state is unnecessary in PNDS — the oracle and
fixed-key results directly refute that. It says this specific pairing (sparse
REINFORCE + 70-way latent key indexing) is the wrong instrument.

## 4. Three implementation errors I made, and a retraction

The corrected basis was not the first basis. An earlier version was defective,
and measurements taken on it are **not valid findings**. They are recorded here
so they are not cited as evidence by mistake.

**Error 1 — the basis collapsed to a constant.** The first "key-indexed"
feature set used `+1` on key positions and `0` elsewhere. Summed over all 70
key blocks each candidate's feature total is `C(7,3) = 35` times its total
`t`-agreement, which by the Amendment 1 exact-`T` constraint is the constant
`35·T`. The basis therefore **provably could not separate gold from
distractors**, and returned 281 non-zero features where the design required 5.

On that broken basis three "diagnostics" were measured:

- "data starvation" — accuracy not scaling with the number of streams
- "7× negative-gradient interference" — 42.86 vs 300 pushes per key block
- "a supervised ceiling at chance"

**All three are retracted.** They were artifacts of the constant basis, not
properties of the environment or of REINFORCE. The correct scaling and
learning-rate diagnostics are the tables in §3, measured on the fixed basis.

**Error 2 — a false degeneracy guard.** I added a `ValueError` claiming the
basis was degenerate at `k = dim/2`. The algebra in `528ad89` §3 disproves this
immediately: with the ±1 encoding the true-key block scores `2a − T` = +2 for
gold and ≤ 0 for distractors at **any** `k`. The guard was removed.

**Error 3 — load-bearing negative features.** The instinct to make non-key
positions contribute `0` rather than `−1` was wrong, and it is not cosmetic.
Without the negative terms the basis is the constant above and the whole
experiment silently measures nothing. The `−agree` on non-key positions is what
makes the basis key-relative rather than key-decorated.

The corrected basis, for the record:

```
phi_{K', i}(c, t) = +[c_i == t_i]    if i in K'
                  = -[c_i == t_i]    if i not in K'
```

## 5. Correction to the record

What does need recording is the basis itself, since it was the site of all three
implementation errors in §4 and is the reason the representability argument
behind `528ad89` §4's `true_key` = 1.0000 holds. It is the key-relative form:

```
phi_{K', i}(c, t) = +[c_i == t_i]    if i in K'
                  = -[c_i == t_i]    if i not in K'
```

For the true key `K` this block scores `a − (T − a) = 2a − T`, i.e. **+2** for
gold (`a = k`) and **≤ 0** for every distractor (`a ≤ k − 1`). The margin is
exactly 2 at every `k`, which is the property that made the `k = dim/2` guard of
Error 2 self-evidently wrong.

Recording this as a design note for the next stage: **a basis whose unused
positions contribute zero rather than a negative term will silently collapse to
a constant under an exact-`T` constraint.** That mistake produced a full family
of false negative diagnostics (§4) and would not have been caught by any
criterion in the pre-registration, because the pre-registration has no basis
specification to violate. If Stage 3b is re-run on a reduced key space, the basis
should be specified in the design document, not only in the implementation.

## 6. What is promoted, and what is not

**Promoted (E2-grade evidence, unchanged):**
- Stage 2c — context-gated relational relevance (learned 0.8853 vs static
  0.0000, 5 seeds, 25/25 deltas positive).
- Stage 3a — the learned weights are necessary (`joint` lesion 0.2027 / 0.1220 /
  0.0680, t > 26), including the negative-lesion finding that `qc_only` restores
  1.0000 and `gated_only` lands at exactly the `anti_static` confound 0.5674.

**Not promoted:** any Stage 3b claim about a learned router reading persistent
state. The stage is closed as a negative result.

**Retained as environment-level findings (not learner claims):**
- The Stage 3b environment passes all its own controls. The exact-`T`
  construction closes the `t`-only shortcut *structurally* — `t_static` sits at
  chance across six configurations in `528ad89` §4 and again here.
- The `do(K ← K')` intervention is causal and large: 0.8333.

## 7. Decision required

Under the asymmetric promotion rule a negative result stops architectural
expansion and diagnoses rather than adds complexity. The diagnosis here is
complete and points at the instrument, not the architecture. Two paths, and this
is the first genuine negative result of the program so it is worth choosing
deliberately:

**A. Report the boundary as the Stage 3b result.** Honest, complete, and it is
what was measured. The claim becomes: persistent state in PNDS is provably
informative and causally load-bearing, but outcome-only training cannot acquire
a 70-way key-indexed read, which constrains which training signals a PNDS
substrate can plausibly be trained with. This is a real constraint on the
program, discovered by the pre-registered test doing its job.

**B. Reduce the key space and re-run.** `k = 2` gives 28 keys instead of 70,
raising per-key effective sample size ~2.5× with no change to the environment's
invariants or controls. This tests whether the failure is discrete (a
sample-size threshold) or continuous (a credit-assignment ceiling). It is a
*diagnostic on the negative result*, which the promotion rule permits, not new
architecture.

Path B is the one that separates "undertrained" from "untrainable by this
method," which is the distinction §3 currently cannot make. It does risk
tuning the experiment until it passes, so if it is taken, the F1 majority-of-seeds
criterion and the 10-seed protocol in §2 must be retained verbatim and the
result reported whether or not it passes.

## 8. Provenance

| artifact | commit | role |
|---|---|---|
| `STAGE_3B_DESIGN.md` | `3a03225` | pre-registration, criteria F1–F5 |
| `STAGE_3B_AMENDMENT_1.md` | `528ad89` | corrected environment; no basis specified, none corrected |
| `test_stage3b_invariants.py` | `2a53338` | property tests, pass on corrected / fail on defective |
| `stage3b_persistent.py` | this commit | corrected key-relative basis implementation |
| this document | this commit | result record, F1 triggered |

All measurements in this document were taken on the implementation committed
here, on Termux under the control-plane constraint, with numpy only. No results
were discarded; the 5-seed run in §2 was extended to 10 seeds because the
5-seed result (3/5, p = 0.5) was not decisive, and both halves are reported.
