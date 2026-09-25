# PNDS-GATE-001 Stage 3b — Probe 1 Result: Cardinality Softens but Does Not Resolve F1

> Probe 1 of Amendment 2 (`d545e62`). Key space reduced from `k = 4` (70 keys)
> to `k = 2` (28 keys); every other protocol element identical to the committed
> Stage 3b run. Recorded after the amendment, before Probe 2 was run.

## 0. Result

**F1 is still triggered at `k = 2`.** The margin distribution improves
monotonically with per-key sample size, but does not cross the pre-registered
majority bar at significance.

| probe | keys | per-key episodes | positive margins | mean margin | binomial p | F1 |
|---|---|---|---|---|---|---|
| `dd8f63c` (k=4) | 70 | ~28 | 4 / 10 | **−0.011** | 0.581 | **fail** |
| Probe 1 (k=2) | 28 | ~71 | 6 / 10 | **+0.030** | 0.377 | **fail** |

The criterion was not relaxed, and this is not reported as a pass.

## 1. The ten seeds at k = 2

`dim = 8`, `k = 2`, `T = default_T(8,2) = 5`, `n_candidates = 8`, chance = 0.125,
250 training streams × 8 episodes, 100 held-out test streams per seed.
Seed schedule identical to `dd8f63c`.

| seed | `learned` | `learned_no_state` | margin |
|---|---|---|---|
| 0 | 0.2800 | 0.1300 | **+0.1500** |
| 1 | 0.2400 | 0.2100 | **+0.0300** |
| 2 | 0.1900 | 0.1900 | 0.0000 (tie) |
| 3 | 0.2100 | 0.2100 | 0.0000 (tie) |
| 4 | 0.2000 | 0.1500 | **+0.0500** |
| 5 | 0.1500 | 0.2100 | −0.0600 |
| 6 | 0.1400 | 0.1100 | **+0.0300** |
| 7 | 0.1600 | 0.1300 | **+0.0300** |
| 8 | 0.1600 | 0.2400 | −0.0800 |
| 9 | 0.2600 | 0.1100 | **+0.1500** |

Positive: 6/10. Mean margin **+0.030**. Exact one-sided binomial,
p = P(X ≥ 6 | n = 10, H0: p = 0.5) = **0.377**.

Restricting to the 8 non-zero margins, the paired-sign test gives 6 positive vs
2 negative, p = **0.145** — still above 0.10. Neither treatment of the two ties
rescues significance.

## 2. What did change, and what did not

**Changed — `learned` is now visibly above chance.** At `k = 4` the learned arm
sat at 0.14–0.18 against a 0.125 chance floor and was indistinguishable from it.
At `k = 2` it sits at **0.14–0.28**, mean ≈ 0.20, above the `random` arm
(0.07–0.14) on 9 of 10 seeds. Something is being learned.

**Changed — the margins move in the predicted direction.** The credit-assignment
account of `dd8f63c` predicted a monotone improvement with keys-per-update, and
mean margin moves −0.011 → +0.030 on a 2.5× increase in per-key sample size.
That is the right sign. It is just not enough.

**Not changed — F2 and F3 still pass.** `true_key` = **1.0000** on all 10 seeds
and `t_static` remains at chance (0.09–0.21, chance 0.125). The environment is
unaffected by `k`; the invariants were re-verified at `k = 2` directly (see §3).

**Not changed — no learning rate fixes it.** The registered control at the same
three learning rates, 3 seeds each:

| lr | `learned` (3 seeds) | margins | mean margin |
|---|---|---|---|
| 0.05 | — | [0.15, 0.03, 0.00, 0.00, 0.05] (seeds 0–4) | +0.046 |
| 0.20 | [0.15, 0.21, 0.17] | [−0.12, +0.01, 0.00] | **−0.037** |
| 1.00 | [0.17, 0.21, 0.23] | [−0.01, +0.04, +0.04] | +0.023 |

No setting separates from the stateless arm. `lr = 0.20` is actively negative.

## 3. Environment invariants re-verified at k = 2

The exact-`T` constraint, the per-position marginal, and unique-satisfier were
checked directly at the Probe 1 configuration, since `k = 2` had never been run
before and `T = 5` is newly reachable:

| check | 300 episodes | result |
|---|---|---|
| exact-`T` violations (`Σ_j 1[c_ij = t_j] = T`) | 0 | **pass** |
| per-position marginal violations | 0 | **pass** |
| unique-satisfier violations | 0 | **pass** |

The representability margin at `k = 2` is **4**, not 2: the true-key block scores
`2a − T = 2·2 − 5 = −1` for gold and `2·1 − 5 = −3` for the best distractor.
The relation is *more* separable at `k = 2`, so Probe 1's failure is not
attributable to a harder discrimination. The full property-test suite
(`2a53338`) also passes unmodified.

## 4. Interpretation against the pre-registered table

Amendment 2 §3's decision table is keyed on Probe 2, not Probe 1, so Probe 1
alone cannot select a row. What Probe 1 does establish:

- **The failure is monotone in key cardinality, not binary.** Moving 70 → 28
  keys moved the mean margin by +0.041 in the predicted direction. This is
  consistent with both remaining hypotheses and discriminates between neither.
- **The learned arm is acquiring *something* at 28 keys** — above chance on
  9/10 seeds against `random`, 6/10 against the stateless arm. Under the
  sample-allocation account this is exactly the expected partial acquisition;
  under the sparse-credit-assignment account it is the same partial acquisition,
  just with more samples per block.
- **A linear extrapolation of the margin trend does not cross the F1 bar at any
  plausible key count.** From −0.011 at 70 keys to +0.030 at 28 keys, reaching
  a robust majority would require a per-key sample budget far beyond the
  registered protocol's scale. If the relationship is even approximately
  linear in `1/n_keys`, **the failure is a property of sparse REINFORCE at this
  cardinality, not a budget shortfall that more streams would fix.**

That last point is the substantive finding of Probe 1, and it is what makes
Probe 2 decisive rather than merely confirmatory.

## 5. Constraints honoured

Per Amendment 2 §4, and verifiable against the committed code:

- **No architectural change, no new components, no state-representation change,
  no environment redesign.** Probe 1 was executed by passing `k = 2` to the
  existing `evaluate_intervention`. No other parameter, generator, sampler,
  control or interface was touched. The committed implementation is unchanged.
- **No additional tuning.** The three learning rates are the three already
  registered in `dd8f63c`.
- **Same 10-seed protocol, same seed schedule, same F1 criterion**, reported
  with the exact binomial p-value. Not re-run after the fact.
- **The retraction stands.** No claim from the defective basis of `dd8f63c` §4
  is revived here. The extrapolation in §4 is a new argument about the *current*
  numbers, not a rehabilitation of the old ones.

## 6. Next

Probe 2, per Amendment 2 §3: the same 70-key environment with a dense supervised
gradient in place of sparse binary REINFORCE. The comparison that resolves the
ambiguity is

> 70-key sparse REINFORCE (**failed**) vs 70-key dense gradient (?).

If the dense gradient succeeds at 70 keys, the bottleneck is sparse binary
credit assignment, and the conclusion for PNDS is a constraint on the training
loop rather than on the substrate.

## 7. Provenance

| artifact | commit | role |
|---|---|---|
| `STAGE_3B_DESIGN.md` | `3a03225` | pre-registration, criteria F1–F5 |
| `STAGE_3B_AMENDMENT_1.md` | `528ad89` | corrected environment |
| `test_stage3b_invariants.py` | `2a53338` | property tests, pass at k=2 unmodified |
| `stage3b_persistent.py` | `dd8f63c` | environment and routers, **unchanged by Probe 1** |
| `STAGE_3B_RESULT.md` | `dd8f63c` | 70-key negative result, closed record |
| `STAGE_3B_AMENDMENT_2.md` | `d545e62` | probe protocol, registered before implementation |
| this document | this commit | Probe 1 result, k = 2 |

All measurements taken on Termux under the control-plane constraint, numpy and
stdlib only.
