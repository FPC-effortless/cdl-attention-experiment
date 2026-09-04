# CASM-X20 results — 2026-09-04

## Status

**VALID EXPERIMENT. POSITIVE CONTROL: PASS. LEARNED INSTANTIATION: SEEN-COMPETENT 3/6. STRONG STATE-INSTANTIATION EXTENSION: INELIGIBLE. STRUCTURE-BLIND CONTROL: SEEN-COMPETENT 0/6.**

X20 tests whether answer-only learning plus an explicit storage cost can infer which supplied observed candidates deserve executable working-state records.

Preregistration before implementation/execution:

`7cdef5dcaf90b9af11c5ae8afeb4741350521b21`

Exact evaluated head:

`36d33e0ee172f27a10d19798f5580fe9b393993c`

Workflow:

`33909957309`

The exact-head integrity gate passed and all six train/evaluate/provenance jobs completed successfully.

Frozen result documentation begins at commit `51e510d63f98d33a2a7f72828ff97ecc74152704`; the preregistration/result cross-link was then finalized at `c92105538219c3f861b34d841892c3097aed87b6`.

## Artifact provenance

| Seed | Artifact ID | SHA-256 digest |
|---|---:|---|
| `20261171` | `9952725769` | `sha256:c85b2efc4406cd274a36938b07f6d3cd5a7b68e222dba47cea8b6797aae377d1` |
| `20261172` | `9952584112` | `sha256:8dce4d1722dcf8d2f477db5fa69c6c954d42d6ddf0ebc2d624e34c7b3ac2a765` |
| `20261173` | `9952736760` | `sha256:c12e6ecc8c9a36ab7a29325b7b775982419393c1fac87f4b08c8bb8259353948` |
| `20261174` | `9952753445` | `sha256:c80911f447e53c17b793a9c7d89081eea2f192cde18044b3e16a8cdd173aedaf` |
| `20261175` | `9952713028` | `sha256:23847fe26aca566da596f66de705425d385cd4c03593631e6de25e854513ebc7` |
| `20261176` | `9952709350` | `sha256:ab1d733f649873d8b5c9033d0fec8bc30c03b4799439b5a9c61fd069b2e76ff6` |

All downloaded artifact digests were independently rechecked against the workflow metadata before classification.

## Positive controls

`canonical_live_mask` is exactly 100% on every seed, every `n=2..6`, every hard/soft answer-final cell, step-state cell and live-register cell through depth 96.

`all_records` is also exactly 100% throughout. The executor and benchmark are therefore valid, and the task is solvable both with the correct minimal state set and with all eight records present.

## Frozen learned-instantiation classification

The preregistration requires **all six seeds** to satisfy the full trained-cardinality gate before unseen `n=5,6` extension is eligible.

| Seed | Seen status | Min hard seen answer | Min soft seen answer | Min hard recall | Min live gate | Max distractor gate | Unseen strong diagnostic |
|---|---|---:|---:|---:|---:|---:|---|
| `20261171` | FAIL | 7.8125% | 99.6094% | 0.3333 | 0.44484 | 3.01e-5 | FAIL |
| `20261172` | FAIL | 8.9844% | 100% | 0.5000 | 0.62380 | 6.36e-5 | FAIL |
| `20261173` | FAIL | 9.7656% | 100% | 0.3724 | 0.48201 | 9.63e-6 | FAIL |
| `20261174` | PASS | 100% | 100% | 1.0000 | 0.99999936 | 9.54e-7 | PASS |
| `20261175` | PASS | 100% | 100% | 1.0000 | 0.99999915 | 7.97e-7 | PASS |
| `20261176` | PASS | 100% | 100% | 1.0000 | 0.99999976 | 1.24e-6 | PASS |

Thus `learned_instantiation` is fully seen-competent on only **3/6** seeds. Under the frozen every-seed rule, strong and partial state-instantiation extension are **not eligible for a formal claim**.

The three competent seeds are nevertheless a sharp diagnostic result: seeds `20261174..20261176` are exactly 100% on hard and soft capability at every seen and unseen `n=2..6` suite, with hard precision/recall/F1 = 1, zero record-count error, live gates effectively one, and distractor gates approximately 1e-6. They therefore demonstrate that the graph-conditioned constructor is capable of learning the intended discrete state-instantiation rule and extending it from trained live cardinalities 2/3/4 to unseen 5/6.

## Failure mechanism on seeds 20261171..20261173

The failing seeds do **not** behave like indiscriminate or structure-blind selectors.

Their graph-conditioned constructors drive distractor gates essentially to zero while leaving required live gates fractional. Seen soft execution remains nearly exact:

- seed `20261171`: minimum seen soft answer 99.6094%, minimum seen soft step-state exactness 99.9349%, minimum seen soft live-register accuracy 99.9756%;
- seeds `20261172` and `20261173`: seen soft answer, step state and live-register metrics are 100% at their minima.

But hard `g >= 0.5` instantiation under-selects required records, producing hard precision 1.0 with poor recall and catastrophic hard execution.

The training histories show a late relaxation bifurcation rather than failure to identify distractors. All three failing seeds first reach near-binary live gates and very low answer loss, then some live gates drift downward while distractors remain near zero:

- `20261171`: mean live gate about 0.960 at step 4000 -> 0.460 at step 12000;
- `20261172`: about 0.99945 at step 4000 -> 0.642 at step 12000;
- `20261173`: about 0.999998 at step 4000 -> 0.503 at step 12000.

By contrast, the three passing seeds remain saturated near one on live records through the end of training.

This is consistent with a **continuous-relaxation loophole** in the frozen objective. Soft gates can preserve enough executable signal for the learned executor while reducing `0.05 * mean(gate)`. The objective therefore does not uniquely force a discrete existence decision. Which basin training reaches is seed-sensitive.

The evidence does **not** support reducing the hard threshold after the fact: `g>=0.5` was frozen, and several live means are already below 0.5. The scientific repair should change the training relaxation, not reinterpret the completed experiment.

## Structure-blind control

`structure_blind_gate` is seen-competent on **0/6** seeds under the frozen hard criteria. It generally retains broad fractional mass rather than isolating the causal live set; representative distractor means remain roughly 0.35–0.45 while hard selection collapses to approximately one record.

Descriptively, graph conditioning clearly produces a much stronger live/dead separation than the structure-blind ablation. However, the preregistered structural-constructor causal comparison is **ineligible** because both learned regimes were required to meet full seen competence on all six seeds.

## Scientific conclusion

X20 establishes four bounded findings:

1. **The benchmark and gated executor are valid.** Canonical-live and all-record controls are exact on all seeds/cardinalities/depths.
2. **A graph-conditioned constructor can learn exact discrete working-state instantiation and extrapolate from live cardinalities 2/3/4 to 5/6.** This occurs on 3/6 fresh seeds and is therefore capability evidence, not a robust every-seed PASS.
3. **The dominant failure is discrete commitment, not distractor identification.** Failing graph-conditioned seeds suppress distractors to near zero and retain near-perfect soft execution, but leave some necessary live records fractionally gated and fail the frozen hard existence contract.
4. **The frozen soft-gate objective is not a robust training surrogate for discrete record existence.** Late seed-dependent drift toward fractional live occupancy prevents a formal state-instantiation extension claim.

X20 does not establish robust learned state creation, entity discovery, semantic role induction, cross-episode persistence, deletion, program induction or verifier-guided repair.

## Decision

Do **not** advance directly to reuse/merge or controller/program-induction experiments.

The immediate successor should be a narrow preregistered X20 repair that preserves the graph constructor, executor, data, supervision, memory cost, train/unseen cardinalities and hard evaluation, while changing only the differentiable existence relaxation so the training forward pass represents a genuinely discrete record decision.

The clean treatment is a deterministic straight-through binary gate:

`g_soft = sigmoid(logit)`

`g_hard = 1[g_soft >= 0.5]`

`g_ST = g_hard + g_soft - stop_gradient(g_soft)`

Use `g_ST` for training execution and storage cost, compare it against an exact soft-X20 replication from bit-identical initialization, and retain a structure-blind straight-through control. This directly tests whether eliminating the fractional-state loophole converts the demonstrated 3/6 capability into robust learned state instantiation.
