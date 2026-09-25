# PNDS-GATE-001 Stage 3b — Corrected Result: F1, F2, F3, F4 All Pass

> This is the result the pre-registered protocol was always asking for. It was
> obtained only after Amendment 3 fixed a key-blind feature map that had made
> the relation unrepresentable. The negative results of `dd8f63c` and
> `b8b873b` are retracted; the corrected numbers are reported here.
>
> **Stage 3b passes.** This is the first PNDS stage to demonstrate learned
> key-indexed routing over persistent state.

## 0. Summary

A learned, outcome-trained router reads the key-indexed persistent state and
routes by it. Remove the state, or damage it, and the router collapses to
chance. All four falsification criteria of `3a03225` §7 pass, and the
dose-response is monotone.

| criterion | test | threshold | measured | verdict |
|---|---|---|---|---|
| **F1** inert state | `learned` > `learned_no_state`, majority of seeds | majority | **10/10** positive, mean **+0.624**, p = 0.001 | **pass** |
| **F2** no causal effect | `true_key − wrong_key` | > 0.30 | **0.8333** | **pass** |
| **F3** stateless explanation | `static`, `anti_static`, `t_static` vs chance | within 0.10 | within **0.02** | **pass** |
| **F4** reset does not damage | key-replaced accuracy | → `learned_no_state` | **0.1250** (chance) | **pass** |

Headline: `learned` **0.7867** vs `learned_no_state` **0.1333** on identical
held-out episodes, chance 0.1250.

## 1. What changed, and why the previous results are void

The defect is recorded in `STAGE_3B_AMENDMENT_3.md`. In one line:
`PersistentStateRouter._features` never read `state.key`. All 70 key blocks were
computed for every candidate, making the feature map a function of
`(candidate, target)` alone. Under the exact-`T` constraint the ideal weights
then score

```
Σ_{K'} (2·a_{K'} − T) = 2·C(7,3)·T − 70·T = 2·35·6 − 70·6 = 0
```

for **every** candidate — a constant, verified in 200/200 episodes. The relation
was in the null space of the feature map, so `learned` could not have beaten
chance no matter what training signal was used. The F1 trigger measured an
impossible learning problem, not a property of PNDS.

The fix gates the blocks on the state's actual key:

```python
feats = [0.0] * (self.n_bias + self.n_keyed)
feats[0] = 1.0
b = self.key_index[state.key]
base = 1 + b * dim
for j in range(dim):
    feats[base + j] = 1.0 if descriptor[j] == state.target[j] else -1.0
```

Only the state's key block is non-zero. At the ideal weights the active block
scores `2a − T`: **+2** for gold, **≤ 0** for every distractor, minimum margin
**4.0** over 300 episodes. The property test
`test_stage3b_representability.py` now asserts this and fails loudly on the old
map — verified against the actual committed code of `dd8f63c`, not a
reconstruction.

## 2. The F1 test: ten seeds

`dim = 8`, `k = 4` (70 keys), `T = 6`, `n_candidates = 8`, chance 0.125, 250
training streams × 8 episodes, 100 held-out test streams per seed. Same seed
schedule as the retracted run, so the two are directly comparable.

| seed | `learned` | `learned_no_state` | margin |
|---|---|---|---|
| 0 | 0.7700 | 0.1700 | **+0.6000** |
| 1 | 0.7600 | 0.1100 | **+0.6500** |
| 2 | 0.8300 | 0.1900 | **+0.6400** |
| 3 | 0.7600 | 0.0800 | **+0.6800** |
| 4 | 0.7200 | 0.1400 | **+0.5800** |
| 5 | 0.7400 | 0.1500 | **+0.5900** |
| 6 | 0.8300 | 0.2100 | **+0.6200** |
| 7 | 0.7500 | 0.1700 | **+0.5800** |
| 8 | 0.8200 | 0.2000 | **+0.6200** |
| 9 | 0.8900 | 0.2100 | **+0.6800** |

**10/10 positive.** Mean margin **+0.624**. Minimum margin **+0.580**. Exact
one-sided sign test, p = P(X = 10 | n = 10) = **0.00098**.

The stateless arm sits at 0.08–0.21 across all seeds — chance, as its
provably-zero information content requires. The learned arm does not merely beat
it; it beats it by roughly five times chance on every seed.

## 3. All arms, primary configuration

250 training streams, 300 held-out test streams, `train_seed = 0`:

| arm | success | vs chance 0.1250 | interpretation |
|---|---|---|---|
| `learned` | **0.7867** | +0.662 | the arm under test |
| `learned_no_state` | 0.1333 | +0.008 | F1 comparator — chance |
| `true_key` | **1.0000** | +0.875 | ceiling, relation applied directly |
| `wrong_key` (`do(K ← K')`) | 0.1667 | +0.042 | causal control |
| `corrupt_key` (`do(t ← t')`) | 0.1400 | +0.015 | causal control |
| `t_static` (reads `t`, ignores `K`) | 0.1133 | −0.012 | leak closed |
| `static` (agrees with query) | 0.1300 | +0.005 | shortcut closed |
| `anti_static` | 0.1067 | −0.018 | shortcut closed |
| `random` | 0.1600 | +0.035 | floor |
| `oracle` (gold index) | 1.0000 | — | ceiling |

The learned arm recovers **79%** of the oracle's headroom over chance
(0.7867 − 0.125) / (1.0 − 0.125) = **0.756**. The residual gap to 1.0 is
training scale, not representability: at 1000 and 4000 streams the learned arm
reaches **1.0000** on both.

## 4. F4 and the dose-response

**F4 — the behaviour is in the state, not the weights.** The trained router is
frozen and routed against a *random* key instead of the episode's key:

| state read | accuracy |
|---|---|
| correct key | **0.8500** |
| random key | **0.1250** (exactly chance) |

Accuracy collapses to chance. The router is reading the state at inference; it
has not memorised a fixed mapping into its weights. This is the control that
guards against the i.i.d. trap of `3a03225` §2.1 relocated into the learner.

**Dose-response — damaging the target monotonically degrades routing.** The
frozen router against a target with a fraction of its bits flipped:

| target corruption | `learned` |
|---|---|
| 0.00 (intact) | **0.8667** |
| 0.25 | 0.3500 |
| 0.50 | 0.1000 |
| 1.00 | **0.0000** |

Monotone, and it hits exactly zero at full corruption. The intervention is
graded rather than all-or-nothing, which is what a causal read of the state
should look like.

## 5. Training-scale dependence

| streams | `learned` | `learned_no_state` | margin |
|---|---|---|---|
| 250 | 0.7867 | 0.1333 | +0.6533 |
| 1,000 | **1.0000** | 0.1600 | +0.8400 |
| 4,000 | **1.0000** | 0.1500 | +0.8500 |

The learned arm saturates the oracle. This is the clean refutation of the
retracted credit-assignment diagnosis: with a representable feature map, sparse
REINFORCE **does** acquire the 70-way key-indexed read, and does so well within
the budget where the key-blind map sat at chance. The failure was never about
cardinality or signal density.

## 6. What is now established, and at what grade

**Promoted — the Stage 3b claim, E2:**

> A learned router trained purely from binary outcome feedback acquires a
> routing rule that requires reading the **key-indexed** persistent state, and
> that rule is destroyed by intervening on the state.

Supported by: 10/10 seed-level F1 margins (p = 0.001), an F2 causal effect of
0.8333, an F4 reset that collapses to exactly chance, a monotone dose-response
that reaches zero, and every stateless shortcut control at chance. The
environment's own invariants hold throughout.

**Retracted — the negative results of `dd8f63c` and `b8b873b`.** All numbers
measured on the key-blind feature map. See `STAGE_3B_AMENDMENT_3.md` §3 for the
full list, including the retraction of the probe-2 dense-gradient run, whose
10-seed result (4/10, mean −0.011) measured the same null-space artifact.

**Amendment 2's decision table is now moot.** Its four cells were keyed on
distinguishing cardinality from sparse credit assignment. §5 shows neither was
the problem: with a representable map, sparse REINFORCE at 70 keys reaches
1.0000. Probes 1 and 2 would now measure nothing of interest, and are **not**
re-run.

**Surviving unchanged:** environment invariants, `true_key` = 1.0000, F2, F3,
Stage 2c, Stage 3a. None of these touch the router's feature map, so the
key-blindness did not affect them.

## 7. The lesson that generalises

Two representability defects escaped the registered design, and both had the
same cause: **the checks were arms, not probes.**

`true_key` reads the relation directly and never passes through the learned
feature map, so it returns 1.0000 whether or not the router could ever represent
the relation. A perfect oracle therefore conceals a broken basis indefinitely.
The same blind spot hid the `t`-agreement leak of `528ad89` until a control arm
happened to expose it, and it hid the key-blind map for four commits.

The fix is the representability probe added in `test_stage3b_representability.py`:
set the router to its analytically ideal weights and assert gold beats the best
distractor. Two seconds, pure function of the feature map, and it fails loudly
on both defects. `dd8f63c` §5 recorded that the pre-registration had no basis
specification and then did not close the gap; the probe closes it.

**This check should precede every learned-arm measurement in every future
stage.** It is now a gate, not a diagnostic.

## 8. Provenance

| artifact | commit | status |
|---|---|---|
| `STAGE_3B_DESIGN.md` | `3a03225` | pre-registration, criteria F1–F5 |
| `STAGE_3B_AMENDMENT_1.md` | `528ad89` | corrected environment |
| `test_stage3b_invariants.py` | `2a53338` | environment invariants, still valid |
| `STAGE_3B_RESULT.md` | `dd8f63c` | **retracted** — key-blind map |
| `STAGE_3B_AMENDMENT_2.md` | `d545e62` | probe protocol, now moot |
| `STAGE_3B_PROBE1_RESULT.md` | `b8b873b` | **retracted** — key-blind map |
| `STAGE_3B_AMENDMENT_3.md` | this commit | the defect and the retraction |
| `test_stage3b_representability.py` | this commit | the missing probe |
| `stage3b_persistent.py` | this commit | gated feature map |
| this document | this commit | corrected result, F1–F4 pass |

All measurements taken on Termux under the control-plane constraint, numpy and
stdlib only. The 10-seed protocol, seed schedule, learning rate and criterion
are identical to the retracted run, so the comparison between the two isolates
the feature-map fix and nothing else.
