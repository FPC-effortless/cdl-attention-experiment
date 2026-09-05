# CASM-X20T frozen scientific result

## Verdict

**VALID experiment. Positive control PASS 6/6. `hard_only_st` seen competence 0/6. `soft_x20_replication` seen competence 3/6. `soft_credit_hard_storage` seen competence 2/6. `dual_credit_hard_storage` seen competence 4/6. Structure-blind credit seen competence 0/6. Counterfactual-credit repair criterion: FAIL. Strong state-instantiation extension: FAIL / INELIGIBLE at the treatment level. Reuse/merge remains blocked.**

CASM-X20T tested whether same-target counterfactual soft execution can provide usable answer credit to the same raw record-existence gates while keeping hard evaluation and binary storage pressure fixed.

The result is materially better than X20R/X20S but does not satisfy the preregistered 6/6 gate. The equal-weight `dual_credit_hard_storage` treatment is fully seen-competent on four of six fresh seeds (`20261201`, `20261203`, `20261204`, `20261206`), and every one of those four also satisfies the full strong unseen `n=5,6` thresholds. The two remaining seeds fail for different reasons: `20261202` under-instantiates needed live state and is an optimization failure; `20261205` solves the task but over-instantiates / selects distractors and therefore fails the frozen state-selection thresholds.

The surrogate-only `soft_credit_hard_storage` treatment passes only seeds `20261201` and `20261204`, both also strong on unseen `n=5,6`. Thus adding the hard task path improves robustness relative to surrogate-only credit in this seed panel, but the preregistered criterion for a bounded credit-assignment repair claim is not met.

The structure-blind credit treatment is seen-competent on 0/6 seeds, so the four successful dual-credit graph-conditioned seeds are not reproduced by the structure-blind control. This is useful mechanistic evidence, but it does not override the required every-seed success rule.

## Frozen provenance

- frozen X20S result/base: `9d4ebc5805f11e7e6e208de006878508111e201e`
- X20T preregistration commit: `9365e9e1ce3242df6305abe9fd816e66298caa64`
- authorized scientific executable head: `9850822a3801fe4fbf65323ddbd38eb518257ecf`
- scientific workflow run: `33941165128`
- contract/integrity job: PASS
- all six train/evaluate jobs: PASS, including provenance validation and artifact upload
- Python: `3.11.16`
- PyTorch: `2.14.0+cpu`
- harness: `casm-x20t-counterfactual-credit-v0-2026-09-05`
- train seeds: `20261201..20261206`
- eval seeds: `20261281..20261286`
- every result JSON records exact executable head `9850822a3801fe4fbf65323ddbd38eb518257ecf`

## Artifact digests

Downloaded ZIP bytes were independently SHA-256 checked and exactly match GitHub's recorded artifact digests.

| train seed | artifact | SHA-256 |
|---:|---|---|
| 20261201 | `casm-x20t-seed-20261201` | `3da92de53c53962ecfb72fe92e25e72bb89f6257aad8f5daecb0716f840898ae` |
| 20261202 | `casm-x20t-seed-20261202` | `50118846f5f659c9d7773e88299698fa26faddbf4757c9023fb6ef3aec76f088` |
| 20261203 | `casm-x20t-seed-20261203` | `8d3ce018e812750222ea045eb424c76474cf5aec20fb7ab46299d1c9708fed28` |
| 20261204 | `casm-x20t-seed-20261204` | `43981e6097a2963e6b68b478b92deed84287eacdfdb3e342b8413fd881e7676b` |
| 20261205 | `casm-x20t-seed-20261205` | `7bd64e1b32293ae526053f560fa4de7b70025ec8ae410687c1d24a3773e4e2fc` |
| 20261206 | `casm-x20t-seed-20261206` | `6cbc39119bf72efac4e6a88f8109da4de237a313e2854cd09717240f85606190` |

## Provenance contract verification

Every result satisfies the frozen run contract:

- 12,000 training steps;
- batch size 128;
- train depth 12;
- `eval_n=256`;
- 8 candidates;
- trained live cardinalities `[2,3,4]`, exactly 4,000 steps each;
- unseen live cardinalities `[5,6]`;
- fixed hard threshold `g_soft >= 0.5`;
- exact objective identifiers from the preregistration;
- no live-mask/cardinality/intermediate-state/hidden-state supervision;
- no learned per-candidate gate table;
- same final-answer target for hard and soft paths;
- counterfactual soft path adds no labels.

## Preregistered classification

### Positive-control validity

`canonical_live_mask`: **PASS 6/6** across `n=2..6` and depths `12/24/48/96` under the frozen hard thresholds. X20T is valid and the learned-treatment results are interpretable.

### Regime summary

| treatment | full seen competence | qualifying strong unseen |
|---|---:|---:|
| `hard_only_st` | 0/6 | ineligible |
| `soft_x20_replication` | 3/6 | 3/6 seeds individually strong, treatment not 6/6 |
| `soft_credit_hard_storage` | 2/6 | 2/6 seeds individually strong, treatment not 6/6 |
| `dual_credit_hard_storage` | 4/6 | 4/6 seeds individually strong, treatment not 6/6 |
| `soft_credit_hard_storage_structure_blind` | 0/6 | ineligible |

No learned treatment reaches full seen competence on all six seeds, so no treatment is formally eligible for the preregistered 6/6 strong-extension claim or reuse/merge successor gate. The formal partial unseen criterion also does not rescue any treatment because the required treatment-level seen gate is not satisfied.

### Hard-only replication

`hard_only_st`: **0/6 seen competent**. All six seeds reproduce the hard-forward collapse with near-zero learned gates, zero hard existence recall/F1, and seen hard-answer accuracy near chance. This confirms the X20R/X20S failure under the pinned X20T environment.

### Soft X20 replication

`soft_x20_replication`: **3/6 seen competent** (`20261201`, `20261202`, `20261204`), and those three are also strong on unseen `n=5,6`.

The remaining seeds reproduce the prior continuous-relaxation instability: `20261203` is weak/under-instantiated; `20261205` has perfect soft capability but fails hard selection/capability; `20261206` has perfect soft answer capability but severe hard under-selection. Continuous relaxation remains seed-sensitive.

### Soft-credit + hard-storage

`soft_credit_hard_storage`: **2/6 seen competent** (`20261201`, `20261204`), both also strong on unseen `n=5,6`.

The other four seeds generally achieve very high raw-soft answer capability but remain below the hard threshold on needed live records. Representative worst seen hard-existence recall values are approximately `0.50`, `0.33`, `0.33`, and `0.38` for seeds `20261202`, `20261203`, `20261205`, and `20261206` respectively.

### Dual hard+soft credit + hard storage

`dual_credit_hard_storage`: **4/6 seen competent** (`20261201`, `20261203`, `20261204`, `20261206`). Every one of those four also passes all strong unseen `n=5,6` thresholds.

The two failures are mechanistically distinct:

- `20261202`: under-instantiation / optimization failure. Worst seen hard and soft answer accuracy is about `8.2%`, hard recall `0.25`, hard F1 `0.40`, and count error `3.0`.
- `20261205`: capability succeeds but state selection fails. Hard and soft answer capability is 100%, but worst seen hard precision is about `0.618`, hard F1 about `0.750`, mean record-count error about `1.49`, mean live gate falls to about `0.629`, and mean distractor gate reaches about `0.182`.

Therefore dual credit substantially improves robustness relative to the hard-only treatment, but it does not establish robust discrete state construction under the preregistered every-seed criterion.

### Structure-blind credit ablation

`soft_credit_hard_storage_structure_blind`: **0/6 seen competent**. Every seed fails the hard-answer optimization boundary and state-selection criteria. The successful graph-conditioned dual-credit seeds are not reproduced by this ablation.

### Counterfactual-credit repair claim

**FAIL.** The preregistered repair claim required at least one graph-conditioned credit treatment to pass full seen competence 6/6 and strong unseen extension 6/6, while the structure-blind treatment did not. Neither graph-conditioned credit treatment reaches 6/6 seen competence.

The bounded non-qualifying observation is that adding equal-weight hard+soft answer credit raises strict seed robustness from 0/6 in the hard-only replication to 4/6, with all four successful seeds also extrapolating strongly to unseen cardinalities. This is evidence that answer-credit design matters, not a completed repair claim.

## Scientific boundary

The sequence now narrows the problem further:

1. X20 showed the supplied graph constructor can learn exact discrete state instantiation on some soft-training seeds, but the continuous relaxation is unstable.
2. X20R showed naive hard-forward straight-through training collapses discrete existence 6/6.
3. X20S showed storage removal/delay/ramping does not repair that collapse.
4. X20T shows same-target soft counterfactual credit can materially improve robust hard state construction, especially when combined with the hard task path, but the result remains seed-sensitive and fails the 6/6 gate through both under-instantiation and over-instantiation modes.

The remaining frontier is still the **discrete state-construction estimator/objective/credit-assignment boundary**, now specifically robust per-record credit and selection calibration. Reuse/merge, lifecycle, persistent cross-episode state, controller/program induction, and verifier-guided repair remain unauthorized.

## Successor boundary

Reuse/merge is not authorized. A future experiment may test a newly preregistered final-answer-only mechanism that assigns more local counterfactual credit to individual record-existence decisions and resolves both under-selection and distractor over-selection without live-mask/cardinality supervision or evaluation repair. X20T itself is frozen and must not be adapted post hoc.
