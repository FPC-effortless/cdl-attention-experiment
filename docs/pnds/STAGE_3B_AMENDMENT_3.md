# PNDS-GATE-001 Stage 3b — Amendment 3: The Router's Feature Map Is Key-Blind

> This amendment **retracts** the Stage 3b negative result of `dd8f63c` and the
> Probe 1 result of `b8b873b`. Both measured a router that provably could not
> represent the relation it was being asked to learn. The F1 "failure" was an
> artifact of the learner's feature map, not a finding about persistent state
> or about REINFORCE.
>
> Found by Probe 2, which is the one thing Amendment 2 asked for: a decisive
> test of whether the bottleneck was the training signal.

## 0. The defect

`PersistentStateRouter._features` in `dd8f63c` documents itself as:

> One dim-length block per possible key; only the state's key block is active.

It does not do this. The implementation is:

```python
def _features(self, query, descriptor, state):
    t = state.target
    agree = [1.0 if descriptor[j] == t[j] else -1.0 for j in range(dim)]
    feats = [1.0]
    for kk in self.all_keys:
        feats.extend(agree[j] if j in kk else -agree[j] for j in range(dim))
    return feats
```

**`state.key` is never read.** Every one of the 70 blocks is computed and
non-zero for every candidate. The feature vector is a function of `(candidate,
target)` only — which is exactly the observable the Amendment 1 exact-`T`
constraint was designed to make non-discriminative.

The router is **key-blind**. `state.key` reaches the router only through the
docstring.

## 1. Why this makes the relation unrepresentable — provably

Score the ideal key-relative weights `w = [+1 on key positions, −1 off]` placed
in every block. For candidate `c`, let `a_{K'}(c)` be its agreement count on
key `K'`. Then

```
score(c) = Σ_{K'} ( 2·a_{K'}(c) − T )
```

Each position `j` appears in `C(dim−1, k−1) = C(7,3) = 35` of the 70 keys, so

```
Σ_{K'} a_{K'}(c) = 35 · (total agreements of c) = 35 · T     [exact-T constraint]
```

therefore

```
score(c) = 2·35·T − 70·T = 0        for every candidate, in every episode.
```

**Measured:** across 200 episodes, the score vector is constant in **200/200**.
The ideal solution lies in the null space of the feature map, and the router's
accuracy at the ideal weights is **0.1300** — chance.

## 2. What the corrected feature map does

Gate the blocks on the state's actual key, so that only the state's key block
is non-zero and the other 69 are exactly zero:

```python
def _features(self, query, descriptor, state):
    feats = [0.0] * (1 + len(self.all_keys) * dim)
    feats[0] = 1.0
    b = self.key_index[state.key]
    for j in range(dim):
        feats[1 + b*dim + j] = 1.0 if descriptor[j] == state.target[j] else -1.0
    return feats
```

At the same ideal weights, this scores the true-key block as `2a − T`:

- gold: `a = k` → `+2` (at k=4) or `+4` (at k=2)
- best distractor: `a ≤ k−1` → `≤ 0`

**Measured on the corrected map:** 300/300 episodes correct, **minimum margin
4.0**. The relation is representable again, as it was before the key-blindness
silently removed that property.

This is the fix that `dd8f63c` §3 and `STAGE_3B_RESULT.md` §5 both claimed was
already in place. The algebra in those documents is correct; the code did not
implement it.

## 3. Retractions

**Retracted — `dd8f63c`, the Stage 3b primary result.** F1 was triggered by a
router that could not represent the relation. The statement "`learned` does not
beat `learned_no_state`" is true but vacuous: `learned` was not capable of
beating chance, for a structural reason that has nothing to do with persistent
state. **The F1 trigger is not evidence about PNDS persistent state.**

**Retracted — the learning-protocol interpretation.** `STAGE_3B_RESULT.md` §3
argued the failure was credit assignment across 70 keys, supported by three
measurements: the stream scaling table (1k→16k, 0.14–0.21), the learning-rate
sweep (0.05/0.20/1.00, 0.15–0.18), and the fixed-key REINFORCE result of 1.0000.
All three were taken on the key-blind map. The fixed-key result is now
uninterpretable — with a key-blind map, a fixed key removes the very ambiguity
that key-blindness creates, so 1.0000 there is consistent with the defect rather
than inconsistent with it. **The credit-assignment diagnosis is withdrawn.**

**Retracted — `b8b873b`, Probe 1.** Same defect, same reason. The k=2 margins
(6/10 positive, mean +0.030) measure a key-blind router and cannot support the
cardinality interpretation placed on them.

**Retracted — the probe-2 measurements taken on the dense router before the
defect was found**, specifically the 10-seed dense run (4/10 positive, mean
−0.011) and the learning-rate extension (lr 0.005/0.5, mean −0.013/+0.013).
`DenseGradientRouter` inherits the key-blind `_features`, so its failure was
the same artifact. **Amendment 2's decision table cannot be read from those
numbers** — every cell was measured against a null-space feature map.

**The three retracted diagnostics of `dd8f63c` §4 remain retracted.** They were
artifacts of the earlier `+1/0` basis. This is a fourth, separate defect, not a
revival of those three.

## 4. What survives

Not everything is retracted. The following remain valid, because they do not
touch the router's feature map at all:

| result | status | why it survives |
|---|---|---|
| Environment invariants (exact-`T`, marginal, unique-satisfier) | **valid** | properties of the generator, verified by `2a53338` at k=4 and k=2 |
| `true_key` oracle = 1.0000 | **valid** | applies the relation directly; never uses the linear basis |
| F2: intervention effect 0.8333 | **valid** | measured on the oracle arm, not the learned arm |
| F3: `static` / `anti_static` / `t_static` at chance | **valid** | these arms read no state or read `t` only; basis-independent |
| Stage 2c (0.8853 vs 0.0000) | **valid** | separate stage, separate router |
| Stage 3a weight-lesion results | **valid** | separate stage, separate router |

The environment itself is untouched. The exact-`T` construction, the structural
closure of the `t`-only shortcut, and the causal intervention are all sound.
**What broke was the learner, and only the learner.**

## 5. Why the defect was invisible to every pre-registered criterion

This is the most consequential part of the record, because it is the second time
a representability defect escaped the registered design.

`3a03225` §7 and `528ad89` both contain representability checks, and both are
formulated as **arms**: `true_key`, `static`, `anti_static`, `t_static`. An arm
that reads the relation directly will always score 1.0000, because it does not
pass through the learned feature map. So a basis defect can hide indefinitely
behind a perfect oracle.

`dd8f63c` §5 recorded this gap and then did not close it:

> the pre-registration has no basis specification, which is why the defect was
> invisible to every criterion in it.

That observation was correct about the earlier defect and is now the exact
description of this one.

**The missing check is a representability probe, not an arm:** instantiate the
router, set its weights to the analytically ideal values, and assert that gold
outscores every distractor on held-out episodes. That test takes seconds, it is
a pure function of the feature map, and it would have failed loudly at `dd8f63c`,
`d545e62`, and `b8b873b`. It is now added as a unit test in `2a53338`'s successor
and must pass before any Stage 3b number is recorded again.

This is also why the fixed-key 1.0000 result was actively misleading rather than
merely uninformative. With a key-blind map, a fixed key makes the ambiguity
irrelevant and the task learnable — so the one measurement that appeared to
localise the failure to cardinality was in fact confirming the defect.

## 6. Corrected program state

> F1: **retracted** — measured on a key-blind router.
> F2/F3: **valid** — environment and intervention controls pass.
> Substrate representability: **valid** — 300/300, min margin 4.0 on the gated map.
> Learner acquisition under any signal: **unresolved** — no measurement taken on
> the corrected map is yet available.

Nothing about the substrate has changed. The question of whether a learner can
acquire the key-indexed read is **open again**, and it is open because it was
never validly closed.

## 7. Authorised next steps

1. Fix `PersistentStateRouter._features` to gate on `state.key` (§2).
2. Add the representability probe of §5 as a unit test. It must fail on the
   committed key-blind map and pass on the gated map.
3. Re-run the F1 10-seed protocol of `dd8f63c` on the corrected map, with the
   same seed schedule and the same majority-of-seeds criterion.
4. Re-run Probe 1 (k=2) and Probe 2 (dense gradient) on the corrected map, and
   only then read Amendment 2's decision table.

No architectural change, no new components, no state-representation change, no
environment redesign, no additional tuning. The binding constraints of
Amendment 2 §4 carry forward unchanged.

## 8. Provenance

| artifact | commit | status |
|---|---|---|
| `STAGE_3B_DESIGN.md` | `3a03225` | pre-registration; environment spec valid, criteria valid |
| `STAGE_3B_AMENDMENT_1.md` | `528ad89` | corrected environment; **valid** |
| `test_stage3b_invariants.py` | `2a53338` | environment property tests; **valid**, extended by this amendment |
| `STAGE_3B_RESULT.md` | `dd8f63c` | **retracted** — F1 artifact of key-blind map |
| `STAGE_3B_AMENDMENT_2.md` | `d545e62` | probe protocol; valid, but its decision table is not yet readable |
| `STAGE_3B_PROBE1_RESULT.md` | `b8b873b` | **retracted** — same defect |
| this document | this commit | Amendment 3 |

Written before the corrected implementation was committed, per protocol.
