# CASM-X6R results — 2026-08-31

## Status

**CASM-X6R satisfies the preregistered strong learned executable binding criterion on every seed.**

This result supersedes only the scientific interpretation of CASM-X6 v0. The v0 run remains explicitly invalidated in `CASM_X6_V0_INVALID_2026-08-31.md` because its approximate Sinkhorn transform violated categorical probability mass and produced negative nominal negative-log-likelihood values. No v0 metric is used as evidence here.

Exact X6R evaluated head:

`f512d4d17eb88bc92a7d8ee0e2cbeec3dfb59d74`

Workflow run:

`33384393765`

All integrity tests passed. All three 6,000-step train/evaluate jobs passed the continuous non-negative-loss guard, post-training probability validator and artifact upload.

Train/eval seed pairs:

- `20260911` / `20260991`
- `20260912` / `20260992`
- `20260913` / `20260993`

## Validity correction

X6R preserves the original X6 scientific question and thresholds while replacing the invalid finite alternating Sinkhorn transform.

The learned model still has only 16 trainable 4×4 binding scores. X6R enumerates the 24 one-to-one permutations of four slots, scores each permutation from those 16 values, applies a softmax over the 24 scores, and forms the binding as their convex weighted sum.

Because every permutation matrix is doubly stochastic, this binding is probability preserving by construction. The largest observed final row-sum or column-sum error across all three learned seeds was approximately `1.19e-7`.

The integrity suite additionally attacked the transform with adversarially sharp/conflicting scores and checked probability mass before and after optimizer updates. The runner aborted on any non-finite or negative categorical loss. No such abort occurred.

## Unchanged supervision contract

- train depth: 8;
- 6,000 optimization steps;
- only final external register `0` is supervised;
- external registers `1–3` are never target labels;
- no teacher forcing;
- no intermediate state targets;
- no semantic operator labels;
- no binding labels;
- no identity target or identity prior;
- no binding/permutation/entropy regularizer;
- every register-specific state access passes through the binding;
- discrete learned evaluation uses the model's own best one-to-one projection;
- all regimes have 246,176 parameters and share data, initialization family and optimization budget.

## Preregistered thresholds

The strong learned executable binding result required **every seed** to satisfy all of the following, after the canonical positive control also passed:

- learned answer-final accuracy ≥99% on every suite;
- learned full step-state exactness ≥95% at depths 24, 48 and 96;
- learned hidden-register accuracy ≥99% at depths 24, 48 and 96;
- mean row maximum ≥0.90;
- mean column maximum ≥0.90;
- best-permutation score ≥3.60 / 4.00.

No averaging across a failed seed was permitted.

## Binding identification by seed

All learned bindings began near the uninformative `1/4` matrix and converged to sharp, valid, non-identity one-to-one alignments.

| Seed | Row max | Column max | Best permutation score | Learned permutation | Min learned training loss |
|---|---:|---:|---:|---|---:|
| 20260911 | 0.993155 | 0.993155 | 3.972619 / 4 | `[1, 3, 0, 2]` | 0.001384 |
| 20260912 | 0.991661 | 0.991661 | 3.966642 / 4 | `[2, 3, 1, 0]` | 0.001997 |
| 20260913 | 0.994393 | 0.994393 | 3.977573 / 4 | `[2, 1, 3, 0]` | 0.000796 |

The different permutations are expected gauge symmetry. X6R does not score identity correspondence as success: internal slot names are arbitrary. What matters is that each seed independently selects a sharp one-to-one binding and executes correctly after decoding through that binding.

## Learned-binding execution

### Seed 20260911

The learned binding achieved exactly 100% answer-final accuracy, full step-state exactness and hidden-register accuracy on every suite from IID depth 8 through stress depth 96.

### Seed 20260912

| Suite | Answer final | Step-state exact | Hidden-register accuracy |
|---|---:|---:|---:|
| IID depth 8 | 100.0000% | 99.9674% | 99.9891% |
| composition depth 12 | 99.7396% | 99.4575% | 99.8264% |
| composition depth 24 | 99.4792% | 99.4792% | 99.7758% |
| stress depth 48 | 99.2188% | 99.5931% | 99.7197% |
| stress depth 96 | 99.2188% | 98.9393% | 99.2902% |

### Seed 20260913

The learned binding achieved exactly 100% answer-final accuracy, full step-state exactness and hidden-register accuracy on every suite from IID depth 8 through stress depth 96.

## Worst-case preregistered result

The scientifically decisive summary is the worst seed/suite value, not the mean:

| Criterion | Required | Worst observed | Pass |
|---|---:|---:|---|
| Answer-final accuracy, every suite | ≥99% | **99.21875%** | yes |
| Step-state exactness, depths 24/48/96 | ≥95% | **98.93935%** | yes |
| Hidden-register accuracy, depths 24/48/96 | ≥99% | **99.29019%** | yes |
| Mean row maximum | ≥0.90 | **0.991661** | yes |
| Mean column maximum | ≥0.90 | **0.991661** | yes |
| Best-permutation score | ≥3.60 / 4 | **3.966642 / 4** | yes |

The canonical-binding positive control is exactly 100% on all three accuracy criteria across every seed and suite, so the control prerequisite also passes.

## Depth-96 aggregate

For descriptive context only, mean learned-binding depth-96 metrics across the three seeds are:

- answer-final accuracy: **99.7396%**;
- full step-state exactness: **99.6464%**;
- hidden-register accuracy: **99.7634%**.

The worst-seed table above, rather than these means, determines the claim.

## Diffuse-binding negative control

The fixed uniform `1/4` binding does not solve the task.

At depth 96, its three-seed means are approximately:

- answer-final accuracy: **26.22%**;
- full step-state exactness: **4.39%**;
- hidden-register accuracy: **25.52%**.

Thus merely exposing a four-slot state container without resolving external identities to distinct slots is insufficient in this setup.

## Qualified result

The preregistered strong result is supported:

> **Within a supplied four-variable explicit-state ontology and one-to-one binding family, fixed-answer-only supervision can learn a sharp external-register ↔ internal-slot alignment from near-uniform initialization and use it to recover an essentially exact hidden executable trajectory far beyond training depth.**

This extends X5 in one important direction. X5 hard-coded the correspondence between external register identity and internal state slot. X6R learns that correspondence from the same single persistent answer channel while registers 1–3 remain permanently absent from the loss.

The fact that independent seeds converge to different non-identity permutations while retaining correct decoded computation is positive evidence that the model is learning a functional binding rather than being pulled toward a privileged identity labeling.

## What this does not establish

X6R is **not autonomous state-schema discovery**.

The experiment still supplies:

- exactly four external variables;
- exactly four internal slots;
- categorical value domain `0..15`;
- the existence of a one-to-one binding family;
- command, argument and destination identities;
- the explicit recurrent state interface;
- the local shared transition architecture;
- the rule that register-specific access must be mediated through the binding.

Also, transition parameters are shared across slots. Therefore hidden variables can benefit from a transition law learned through the observed answer channel once the binding structure makes those variables participate in causal computation. X6R should not be described as four independent hidden state variables being directly learned without any structural coupling.

## Next falsifiable question

The dominant remaining structural prior is the ontology itself.

A useful successor should make the number and role of internal variables less explicit while preserving identifiability. A clean design is to provide **more candidate internal slots than true external variables** (for example, 8 candidate slots for a 4-variable world), require the model to learn a sparse injective external-variable→slot assignment from fixed-answer-only supervision, and evaluate decoded execution plus whether surplus slots remain unused or acquire reproducible auxiliary roles.

Success must be defined functionally rather than by matching arbitrary slot numbers, because internal-slot permutation symmetry remains unavoidable.