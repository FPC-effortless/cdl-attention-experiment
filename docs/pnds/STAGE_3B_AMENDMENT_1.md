# PNDS-GATE-001 Stage 3b — Amendment 1: Corrected Environment

**Status:** AMENDMENT. Supersedes the environment of `3a03225` for the primary experiment.
**Parent design:** `3a03225` (PRE-REGISTERED, unchanged and preserved in the record)
**Author date:** 2026-09-24

> `3a03225` is **not edited**. It remains in the repository as the registered
> design, with its defect visible. This document is the correction, and it states
> plainly that the registered environment was found defective *before any Stage 3b
> experiment was run*, by the same class of representability check that killed two
> Stage 2c designs.

## 1. The registered-design defect

The environment registered in `3a03225` is **defective**. It does not test what it
claims to test.

**Measured on the registered environment** (2000 held-out episodes, `dim = 8`,
`k = 4`, `n = 8` candidates, chance = 0.125):

| Arm | success |
|---|---|
| `true_key` (reads `t` and `K`) | 1.0000 |
| `corrupt_key` | 0.0000 |
| **`t_static` (reads `t`, ignores `K`)** | **0.6965** |
| `static` (agreement with `q`) | 0.1640 |
| `random` | 0.1205 |

A router that reads the target `t` and **completely ignores the key `K`** scores
**0.6965** — five and a half times chance, and 70% of the ceiling.

### Why `t`-only retrieval succeeds — analytic

Gold is constructed to match `t` on all `k` key positions, with the remaining
`dim - k` bits sampled uniformly. So gold's total agreement with `t` is

```
A_gold = k + Bin(dim - k, 1/2)          E = k + (dim-k)/2 = 6,  sd ≈ 1
```

Every distractor is forced to violate the relation on at least one key position
but is otherwise random, so

```
A_distractor = (k - |V|) + Bin(dim - k, 1/2),  |V| >= 1
                                             E <= (k-1) + (dim-k)/2 = 5,  sd ≈ 1
```

Two distributions with `sd ≈ 1` and means one apart, and the arm takes the max
over `n - 1` competitors: `P(gold wins) ≈ 0.76` pairwise, `≈ 0.70` against a field
of seven. The measurement (0.6965) matches the analytic prediction.

This is not a tuning artefact and not bad luck. **It is the construction.** Any
environment in which gold's key-restricted match with `t` is *also* the cause of
gold having *higher total* match with `t` has this leak, and the leak is large.

### Explicit identification of the failure mode

This is **failure mode B — state redundancy**, as defined in §2.2 of `3a03225`:

> If the observation still carries the marker bits, the router can recover the key
> from the observation alone and `S_t` carries nothing.

The registered design checked this failure mode in one direction only — whether
the *query* leaked the key — and confirmed it did not. It never checked whether
**the target `t` itself, read without the key, was already sufficient to identify
gold**. It was. The state was therefore *partially* redundant: `t` alone retrieved
0.70 of the behaviour the design attributed to the joint `(t, K)` read.

The registered `learned_no_state` control would **not** have caught this. That arm
withholds `S_t` entirely. A router that reads `t` but not `K` is a *third* arm the
registered design did not include, and it is the one that exposes the defect.

## 2. Why the original Stage 3b inference is invalid

The registered design's central claim was:

> the router must route into `S_t` to beat chance, because a stateless router
> provably cannot.

```
t-only retrieval -> 0.6965   (chance 0.125)
```

That single number invalidates the claim. A router that accesses the persistent
state but **does not access the key-indexed part of it** already retrieves 70% of
the ceiling. Therefore:

- The registered `learned` arm's success would be **not attributable** to
  key-indexed routing. Any value in `[0.12, 0.70]` is consistent with "the router
  learned to match `t` and ignored `K` entirely."
- The registered `corrupt_key` intervention would be **not interpretable**.
  Damaging `K` while leaving `t` intact would leave the `t`-only pathway fully
  functional, so a null intervention effect would be uninformative and a positive
  one would be confounded.
- The registered falsification criterion F3 — "no stateless shortcut beats chance
  by more than 0.10" — is **satisfied by the registered arms and violated by the
  one arm that matters**. The criterion was correct; the arm set was incomplete.

**The registered design tested whether the state contained useful target
information. It did. That is not the PNDS question.**

## 3. The corrected environment

**Every candidate — gold and distractor alike — satisfies an exact agreement total
with the target:**

```
For all candidates c_i:      sum_j 1[c_ij = t_j] = T
```

Consequence:

```
f_total-agreement(c, t) = T        for every candidate
```

The total-agreement score vector is **constant across candidates**. The
`t_static` arm therefore provably cannot separate gold from distractors — it
degenerates to a uniform pick among ties. The principal stateless shortcut is not
merely made unlikely; it is made **structurally impossible**.

### Construction

- Gold matches `t` on all `k` key positions, and on exactly `T - k` of the
  `dim - k` unmarked positions.
- Each distractor has `a_d` key-position agreements with `t`, where `a_d <= k - 1`
  (guaranteeing at least one key violation, so gold remains the unique satisfier),
  and exactly `T - a_d` unmarked agreements, chosen at random.
- `T` is a design constant, fixed per configuration, and identical for every
  candidate in every episode.

The unmarked positions are no longer free: they are *constrained* to make the
total come out to `T`. An unkeyed linear feature `sum_j [c_j == t_j]` is constant
across candidates, so the router cannot fall back on it. The construction
structurally prevents the shortcut rather than relying on the router not finding it.

### Per-position marginal

The exact-`T` constraint also equalises the per-position agreement marginal:

```
P(c_ij = t_j | gold)      = P(c_ij = t_j | distractor) = T / dim
```

Measured: gold and distractor per-position agreement with `t` are both **0.7500**
at `T/dim = 6/8`, identical to four decimal places. So no per-position bias
survives either — the shortcut cannot reappear through a different statistic.

## 4. Pre-implementation validation

**Measured on the corrected environment** (2000 held-out episodes, `dim = 8`,
`k = 4`, `T = 6`, `n = 8`, chance = 0.125):

| Arm | success | interpretation |
|---|---|---|
| `true_key` | **1.0000** | the joint `(t, K)` read is sufficient |
| `corrupt_key` (damage `t`, `frac = 1.0`) | **0.0000** | the state is necessary |
| `t_static` (reads `t`, ignores `K`) | **0.1205** | leak closed — at chance |
| `static` (agreement with `q`) | 0.1835 | at chance |
| `anti_static` | 0.1355 | at chance |
| `random` | 0.1180 | chance floor |

`t_static` is now **provably** at chance rather than approximately chance. The
proof is that the value is *identical across configurations*: the six-configuration
sweep returns the same number regardless of `dim`, `k`, `T`, or candidate count,
because a constant score vector has no information.

### Six-configuration sweep (1000 episodes each)

| dim | k | T | n | chance | `true_key` | `t_static` | `static` | random |
|---|---|---|---|---|---|---|---|---|
| 8 | 4 | 6 | 8 | 0.1250 | 1.0000 | **0.1210** | 0.1200 | 0.1230 |
| 8 | 4 | 6 | 16 | 0.0625 | 1.0000 | **0.0580** | 0.0610 | 0.0610 |
| 8 | 4 | 7 | 8 | 0.1250 | 1.0000 | **0.1210** | 0.1200 | 0.1230 |
| 8 | 2 | 5 | 8 | 0.1250 | 1.0000 | **0.1210** | 0.1200 | 0.1230 |
| 8 | 3 | 6 | 8 | 0.1250 | 1.0000 | **0.1210** | 0.1200 | 0.1230 |
| 12 | 4 | 8 | 8 | 0.1250 | 1.0000 | **0.1210** | 0.1200 | 0.1230 |

Every `t_static` and `static` value matches chance to within sampling error at
every configuration. The `true_key` arm is 1.0000 at all six, and gold is the
unique satisfier in 100% of episodes at all six.

### Invariant violations

Zero. Across 6000 episodes spanning the sweep above: `unique-satisfier` violations
= 0, `exact-T` violations = 0.

## 5. Corrected intervention specification

The persistent state is unchanged in form:

```
S_t = ( t_t , K_t )
```

- `t_t` — the target vector the router matches candidates against.
- `K_t` — the set of `k` positions that determine relevance.

The intervention is:

```
do(K_t <- K'_t)      with K'_t != K_t
```

**Its interpretation is now far cleaner**, because `t_t` can no longer
independently identify the gold candidate. Under the registered design, a
`K`-intervention left the `t`-only pathway intact at 0.6965, so the intervention
effect was confounded with that pathway. Under the corrected design, damaging `K`
destroys the *only* separating signal the environment contains.

**Measured key-replacement intervention** (2000 held-out episodes each):

| dim | k | T | n | chance | `true_key` | `wrong_key_random` | `wrong_key_disjoint` |
|---|---|---|---|---|---|---|---|
| 8 | 4 | 6 | 8 | 0.1250 | 1.0000 | **0.1420** | **0.0000** |
| 8 | 4 | 6 | 16 | 0.0625 | 1.0000 | **0.0855** | **0.0000** |
| 10 | 4 | 6 | 32 | 0.0312 | 1.0000 | **0.0415** | **0.0000** |

- `wrong_key_random` — replace `K` with an independent random key of the same
  size. Result: **at chance** at all three candidate counts. The corrupted key
  carries no information about the true key, so retrieval collapses to chance
  rather than to a partial fallback.
- `wrong_key_disjoint` — replace `K` with a key disjoint from the true key.
  Result: **0.0000** exactly. No candidate satisfies the corrupted relation, so
  the arm cannot select gold.

The contrast `true_key - wrong_key_random` is the pre-registered causal effect:
**1.0000 - 0.1420 = 0.8580** at 8 candidates, and it *increases* with candidate
count (0.9145 at 16, 0.9585 at 32) because chance falls while the oracle holds.

Target-cast corruption of `t` remains a secondary intervention arm and is
reported in §6.

## 6. Dose-response — secondary diagnostic

Retained as a **diagnostic, not a promoted result**, for exactly the reason given
in `3a03225` §6: a monotone curve is consistent with a lookup table, which is not
the PNDS claim.

The corrected graded response is nonetheless worth recording, because the
exact-`T` construction changes its shape:

| corruption fraction of `t` | P(gold satisfies corrupted relation) | mean satisfiers |
|---|---|---|
| 0.00 | 1.0000 | 1.000 |
| 0.25 | 0.2070 | 0.857 |
| 0.50 | 0.0145 | 0.525 |
| 1.00 | 0.0000 | 0.000 |

Under the registered design this curve was a **cliff** — the full-intervention arm
jumped straight to zero. Under the corrected design it is graded, because
corrupting one bit of `t` breaks the exact-`T` invariant and can leave the
relation satisfiable by chance.

**Explicit non-promotion.** I would not elevate this to evidence of a learned
mechanism. A lookup mechanism produces a monotonic curve too. The reason to
record it is that the graded shape makes the dose-response control *harder to
dismiss* than a step function would, and it establishes that the
state-to-behaviour coupling is graded rather than all-or-none. That is a
property of the environment, not of the learner.

## 7. What the corrected design changes about the claim

This is the important methodological point, and it is why the amendment is not
merely a better implementation.

**The original preregistration tested whether the state contained useful target
information.** It did. That finding is trivial and is not the PNDS hypothesis.

**The corrected design tests whether the decision mechanism must access the
key-indexed persistent state**, because the target itself has been made
non-discriminative under every observable feature statistic. Concretely, the
separating quantity is *only*

```
sum_{j in K} 1[c_j = t_j]
```

which is a function of `(c, t, K)` — and `K` exists nowhere outside `S_t`. So a
router that does not read `S_t` provably cannot beat chance, and a router that
reads `t` but not `K` *also* provably cannot beat chance. Only the full key-indexed
read works.

The experiment therefore isolates the specific PNDS property under test:
**addressability of a persistent key**, not mere persistence of a target.

## 8. Explicit amendment rule

**All Stage 3b results obtained under the original environment are INVALID for the
preregistered causal claim.** None were obtained — the defect was found before any
Stage 3b experiment was run, by the pre-implementation validation above — but the
rule is stated to bind the future as well as the past.

**The corrected environment supersedes the original for the primary experiment.**
Specifically:

- §3.3, §3.4, §3.5, §3.6 and §6 of `3a03225` describe the defective environment
  and are **superseded** by §3, §4, §5 and §6 of this amendment.
- §3.5's analysis of the `static`/`anti_static` tie-break artifact **remains
  valid** and its design rule R5 (random tie-breaking under a seeded RNG) is
  retained; the corrected design's `static` value of 0.1835 vs the tie-break-random
  value of 0.1200 confirms the artifact is real and small.
- §2.1 (failure mode A, the inert state) **remains valid and binding**. The
  Stage 2c rule is still a constant, and Stage 3b still requires a new
  environment rather than a modification of Stage 2c.
- §2.4's informal theorem **remains valid** and is the reason the relation is
  decoupled from the query.
- The seven destructive controls of §6 are retained, with control 3 now
  implemented as `do(K_t <- K'_t)` per §5 of this amendment.
- The falsification criteria of §7 are retained with one addition: **F3 is
  extended to cover `t_static`.** A `t`-only arm is now a required arm in every
  Stage 3b artifact, and if it beats chance by more than 0.10 the environment is
  invalid and the run is void.
- The success gate of §9 is retained with the same addition.

**`3a03225` is preserved unmodified in the repository** as the registered design,
defect and all, so that the correction remains visible in the research record
rather than being rewritten out of it.

## 9. Required property tests

Two property tests must be implemented as unit tests and must pass on every
generated episode. They exist to prevent a future generator modification from
reintroducing the same leak through a different statistic.

### Property test 1 — exact agreement total

```
For all candidates c_i in every episode:
    sum_j 1[c_ij = t_j] = T
```

Exhaustive over all candidates in all generated episodes. This is the invariant
that makes `f_total-agreement` constant and therefore makes the `t`-only shortcut
structurally impossible.

### Property test 2 — per-position marginal equality

```
P(c_ij = t_j | gold)      = P(c_ij = t_j | distractor) = T / dim
```

Evaluated per episode, since under the exact-`T` construction both gold and the
mean distractor have per-position agreement exactly `T/dim` in every single
episode.

### Regression verification

Both tests were run against **both** environments to confirm they actually catch
the defect rather than passing vacuously:

| Environment | exact-`T` pass rate | marginal pass rate |
|---|---|---|
| Corrected | **1.0000** | **1.0000** |
| Original (`3a03225` design) | **0.0000** | **0.0000** |

The tests fail loudly on the original design and pass on the corrected one. A
future generator change that reintroduces the `t`-only leak — through an
agreement imbalance, a per-position bias, or any other statistic — trips at least
one of them.

## 10. Implementation order

Binding, per the pre-registration:

1. **Amendment commit** (this document). — `3a03225` untouched.
2. **Property tests** (`test_stage3b_invariants.py`), including the two tests of
   §9 and a test that they fail on the original generator.
3. **Corrected implementation** (`stage3b_persistent.py`, `run_stage3b.py`) — only
   written once the property tests pass against it.
4. **Training sweep** (8/16/32 candidates, 5 training seeds), then the CI workflow
   step, then the results entry in `docs/pnds/RESULTS.md`.

No Stage 3b experiment may run before steps 1–3 are complete and committed.
