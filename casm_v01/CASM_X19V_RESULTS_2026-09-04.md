# CASM-X19V results — 2026-09-04

## Status

**INVALID REPLAY. NO BETA64 SCIENTIFIC CLAIM IS AUTHORIZED.**

CASM-X19V was preregistered as a one-shot post-training address-margin validation of X19D. Its frozen replay prerequisite required the beta16 replay to preserve the X19D discrete outcome classes before any beta64 interpretation was allowed.

Preregistration:

`51e3d5fb5b262962048224120d2c71593ee68729`

Exact executable head:

`acbd51fe47fa914a36cab89ce330777fba092f87`

Workflow:

`33892961048`

The workflow, integrity gate, and all six replay/evaluate jobs completed successfully on that exact head.

## Artifact provenance

| Seed | Artifact ID | SHA-256 digest |
|---|---:|---|
| `20261161` | `9945002228` | `sha256:a6e13f983f33756b602107110ed532f23fb733e94a1b3a28dc82c16dc359caf7` |
| `20261162` | `9945105954` | `sha256:e3066587fb8230e0bafbae7f72c42f86b4916e695939c6df61de4bb5d3eb08ab` |
| `20261163` | `9945093863` | `sha256:58bbe3966f3c228a0010193e2e75eb1919c53b1a129cd32b2e2c0136fcc51091` |
| `20261164` | `9945104813` | `sha256:a0862d24deaab5f4b993e494afa3ec1ad72c70de0ccd09015010c917f8104054` |
| `20261165` | `9945115454` | `sha256:867b1110083ca2465dfaa4cff8a8b7694d52041e0f580d9acb1c7e8a282a00de` |
| `20261166` | `9945094385` | `sha256:8665f378837c1fddf98a54e82e736f4777ceace226b0c31c745259cfbf9e37f4` |

Every artifact binds to executable head `acbd51fe47fa914a36cab89ce330777fba092f87`.

## Replay prerequisite check

Frozen X19D classification required beta16 replay to reproduce:

- canonical positive control exact;
- both learned recurrences seen-competent 6/6;
- unseen hard execution/self-address exact 6/6;
- `orthogonal_recursive` strong unseen 5/6;
- `unconstrained_recursive` strong unseen 1/6;
- `frozen_random_orthogonal` strong unseen 4/6.

The replay reproduces the canonical, learned seen-competence, hard unseen, orthogonal 5/6, and frozen-random 4/6 outcome classes. It **does not reproduce the unconstrained strong-unseen class**.

Independent artifact classification gives `unconstrained_recursive` beta16 strong unseen **0/6**, not X19D's frozen **1/6**. In particular, seed `20261161`, which was the sole X19D unconstrained strong PASS, now misses the frozen soft thresholds at unseen n=6:

- stress-depth-48 soft answer-final: `98.4375%` (<99%);
- stress-depth-96 soft answer-final: `97.265625%` (<99%);
- stress-depth-96 soft hidden-register accuracy: approximately `98.4497%` (<99%).

The archived X19D seed `20261161` artifact had stress-depth-96 unconstrained soft answer-final `100%`, step-state exactness about `99.8047%`, and hidden-register accuracy about `99.9308%`.

Therefore the replay changed a preregistered discrete pass/fail outcome class. Under the frozen X19V rule, the experiment is **INVALID REPLAY** and no beta64 scientific claim may be promoted.

## Descriptive beta64 behavior — non-claim evidence only

For transparency, the post-training beta64 counterfactual is numerically strong. Under an independent application of the frozen X19D strong criteria, every seed at unseen n=5,6 passes for:

- `canonical_keyed`: 6/6;
- `frozen_random_orthogonal`: 6/6;
- `unconstrained_recursive`: 6/6;
- `orthogonal_recursive`: 6/6.

For the learned orthogonal recurrence, unseen mean soft self-address probabilities are effectively one at beta64 on every seed, including seed `20261166`; its worst observed competing unseen cosine remains approximately `0.84818`, but beta64 converts this to maximum competing soft address probability only about `6.03e-5`.

These observations are **descriptive diagnostics only** because the replay prerequisite failed. They cannot be used to claim that beta64 validated the X19D substrate.

## Scientific conclusion

X19V exposes a reproducibility/optimization-sensitivity boundary in the X19D replay path. The frozen X19D conclusions remain unchanged:

1. both learned recurrence families are hard-exact on unseen n=5,6 on all six X19D seeds;
2. the orthogonal/noncontractive recurrence materially improves long-horizon role geometry and beta16 soft execution relative to the unconstrained recurrence;
3. frozen random orthogonal codes show that this benchmark primarily tests extensible addressable identities, not learned semantic role discovery.

X19V does **not** authorize a stronger beta64 claim.

## Model-development decision

Do not spend another experiment tuning role-code temperature or fixed-slot allocation. The next model-development experiment should isolate **state instantiation/reuse/deletion decisions** using a substrate whose address identities are supplied or otherwise fixed, so constructor/address calibration cannot confound whether the model learns which computational state should exist.

The successor must therefore test structure creation as a decision problem, not role-code generation.