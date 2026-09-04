# CASM-X19D results — 2026-09-04

## Status

**VALID EXPERIMENT. POSITIVE CONTROL: PASS. LEARNED PAIR: SEEN-COMPETENT 6/6. STRONG UNSEEN: unconstrained 1/6, orthogonal 5/6. FORMAL STRONG CONSTRUCTOR EXTENSION: NOT ESTABLISHED.**

X19D isolates constructor dynamics from learned variable→fixed-slot allocation by using a deterministic role-keyed transient memory substrate.

Preregistration before implementation:

`18b22779bf5915253d81da0816f4c9278b0199a4`

Pre-implementation frozen-random falsifier addendum:

`6a9ed68f0a0ab924a81b2f82eec9d92952036a21`

Exact evaluated head:

`58b4d1ca346536b4ca97b65ca8e2a6fe4d2be12f`

Workflow:

`33890002808`

Integrity gate: **PASS**.

All six train/evaluate/provenance jobs completed successfully.

## Artifact provenance

| Seed | Artifact ID | SHA-256 digest |
|---|---:|---|
| `20261161` | `9943969858` | `sha256:e3a48310391dbaf4b69161aaaf6952b934432d5095addbb3f4a5cf83e4ae693c` |
| `20261162` | `9943970647` | `sha256:82a626d5bd01d2f98e40f785e3e166452ff614c8a29ec980412fe8c5ac529213` |
| `20261163` | `9943972667` | `sha256:e15d91e3a52d5fd0336df66f64ef48f988c956fcdc2214cea2d24f7be044f698` |
| `20261164` | `9943968931` | `sha256:bdd7fbcb68033de3631d5718badbed34f44f60e2909a36ac1e87000d03044fe7` |
| `20261165` | `9943950446` | `sha256:c7c240ecdf796744a688bbd8e05ea4e47aac607ed61fbec935f7c70ed6f5b16c` |
| `20261166` | `9943960459` | `sha256:f0092bcd9c43feec433d621bcaaa6c5af9e5a9589e2f51af58e24e93fe6f7f78` |

All artifacts bind to evaluated head `58b4d1ca346536b4ca97b65ca8e2a6fe4d2be12f`.

## Positive control

`canonical_keyed` is exactly competent on every seed and every `n=2..6` hard/soft suite through depth 96. The role-keyed substrate and executor therefore satisfy the frozen positive-control prerequisite.

## Learned seen competence

Both learned recurrences satisfy the full seen-constructor prerequisite on **all 6/6 seeds** at trained `n=2,3,4`:

- hard and soft answer-final >=99% on every suite;
- deep hard/soft step-state exactness >=95%;
- deep hard/soft hidden-register accuracy >=99%;
- all active roles hard-address their own records uniquely;
- mean soft self-address probability >=0.90.

The learned orthogonal-vs-unconstrained comparison is therefore eligible.

## Unseen structural extension

### Hard execution

A critical diagnostic result is that **both learned recurrences are hard-exact on unseen `n=5,6` on every seed**. Every generated active role hard-addresses its own memory record uniquely, with zero ties. Therefore role identities `r4,r5` remain distinct enough for exact hard content addressing on all six seeds.

This is a major difference from X19's learned fixed-slot bridge, where unseen roles were forced back into previously used physical-slot identities.

### Soft execution

The frozen strong criterion also requires soft execution and soft self-address margin. Here the treatments diverge sharply.

- `unconstrained_recursive`: strong unseen PASS on **1/6** seeds (`20261161`). On the other seeds, hard execution remains exact but soft address leakage degrades execution, with the worst unseen minimum soft answer-final falling to approximately `25.39%`.
- `orthogonal_recursive`: strong unseen PASS on **5/6** seeds (`20261161..20261165`). The sole failure is seed `20261166`: hard execution is exact and all roles uniquely self-address, but fixed-beta soft retrieval is insufficiently sharp. Worst unseen soft answer-final is `85.94%`, deep soft step-state exactness about `88.20%`, and deep soft hidden-register accuracy about `92.65%`.

For seed `20261166`, the orthogonal role orbit has unseen mean soft self-address probability about `0.968` at n=5 and `0.946` at n=6, with maximum competing-role cosine similarity about `0.848`. The failure is therefore soft-memory cross-talk between distinct role keys, not a hard role collision.

Under the frozen every-seed rules, `orthogonal_recursive` does **not** receive a strong or partial constructor-extension PASS.

## Frozen-random falsifier

`frozen_random_orthogonal` demonstrates that learned role semantics are not automatically required by this controlled benchmark once one transient record is supplied per external variable.

- hard execution and hard self-addressing are exact on unseen `n=5,6` on all six seeds;
- full frozen strong soft criteria pass on 4/6 seeds;
- seeds `20261163` and `20261165` fail only the soft capability thresholds despite unique hard addressing.

Thus arbitrary noncontractive recurrent identity codes are often sufficient for execution in the role-keyed substrate. X19D cannot support the claim that useful role semantics themselves were learned. The benchmark currently tests extensible computational identity/addressability more strongly than semantic role discovery.

## Constructor geometry diagnostics

The long-horizon post-training diagnostics through `r31` strongly separate the learned treatments.

### Unconstrained recurrence

Across seeds, the unconstrained normalized `(I + 0.1 A)` recurrence progressively approaches a low-change attractor:

- first role with cosine similarity >0.99 to an earlier role occurs around recurrence indices 10–18;
- maximum off-diagonal cosine similarity by the 32-role prefix reaches approximately `0.993` to `0.9996`;
- perturbation gains depart materially from 1 and can amplify several-fold.

### Orthogonal recurrence

The learned Cayley recurrence remains noncontractive:

- no seed produces a role with cosine similarity >0.99 to an earlier role through `r31`;
- perturbation gains remain approximately 1.0 through 31 recurrence steps;
- role norms remain approximately one;
- maximum off-diagonal cosine similarity through the 32-role prefix remains below approximately `0.917` on every seed.

This supports the mechanistic conclusion that the orthogonal bias prevents the fixed-point/contractive pathology observed in X19 and in the unconstrained X19D control.

## Scientific conclusion

X19D establishes three bounded findings:

1. **Removing the learned fixed-slot bridge exposes that recurrent role identities can extend hard-executable addressing to unseen roles 4 and 5.** Both learned recurrences achieve exact hard unseen execution on all six seeds.
2. **Approximately norm/angle-preserving recurrence materially improves soft executable identity extension.** The orthogonal treatment reaches the frozen strong unseen criterion on 5/6 seeds versus 1/6 for the parameter-matched unconstrained recurrence.
3. **The remaining formal failure is primarily a soft retrieval-margin issue in the diagnostic role-keyed memory, not hard role collapse at n=5/6.** This does not authorize a strong PASS because the preregistration required both hard and soft criteria on every seed.

X19D does not establish semantic role discovery, learned cardinality, learned state creation/reuse, program induction, persistence, or verifier-guided repair.

## Decision

Do **not** return to slot allocators and do **not** yet claim X20 authorization under the frozen X19D rule.

Run one final preregistered substrate-validation experiment that keeps the trained constructor family fixed and tests whether the sole remaining failure is fixed-temperature soft content-addressing margin. This validation must include the frozen-random orthogonal falsifier. If a sharper but still fixed role-keyed addressing contract makes both random and learned orthogonal codes robust, conclude that this controlled task needs extensible identity codes rather than learned role semantics, and move the research frontier to learned state-instantiation/reuse decisions. If orthogonal recurrence still fails robustly, continue constructor work rather than advancing to X20.