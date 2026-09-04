# CASM-X20S frozen scientific result

## Verdict

**VALID experiment. Positive control PASS 6/6. Every learned X20S regime: seen competence 0/6. Strong state-instantiation extension: FAIL / INELIGIBLE. Partial extension: FAIL. Storage-onset repair hypothesis: FAIL. Do not advance to reuse/merge.**

CASM-X20S tested whether X20R's all-absent hard-forward failure was caused primarily by applying storage pressure before answer learning had established an executable path. It compared immediate storage pressure, zero storage pressure, delayed abrupt pressure, delayed ramped pressure, and a structure-blind delayed-ramp ablation under the same frozen binary-forward straight-through estimator.

The delayed-onset hypothesis is falsified under the preregistered setup. All graph-conditioned treatments, including `no_storage_st`, fail seen competence on all six fresh seeds. The answer-only treatments have already lost useful hard-forward execution during the 1,000-step zero-storage warmup, before any delayed storage penalty is introduced. Therefore delaying or ramping storage pressure cannot rescue the state constructor in this setup.

The strongest bounded interpretation is that the remaining failure lies upstream of storage timing: the frozen hard-forward straight-through answer-learning dynamics themselves do not robustly establish useful discrete record existence. This result does not identify the exact alternative estimator/objective required.

## Frozen provenance

- frozen X20R result/base: `a4b8ee98dd300dc51e4398c84020a2e90c2cccc6`
- X20S preregistration commit: `cc9168011e8a2f578bc7dce9879dc42a6161e201`
- authorized scientific executable head: `375cfda56c7099e1e2c1a9f46800a04b19bf0e06`
- scientific workflow run: `33924893046`
- contract/integrity job: PASS
- all six train/evaluate jobs: PASS, including provenance validation and artifact upload
- Python: `3.11.16`
- PyTorch: `2.14.0+cpu`
- harness: `casm-x20s-storage-onset-v0-2026-09-04`
- train seeds: `20261191..20261196`
- eval seeds: `20261271..20261276`
- all result JSON files report the exact authorized scientific executable head and the same pinned runtime versions

## Artifact digests

Downloaded ZIP bytes were independently SHA-256 checked and match GitHub's recorded artifact digests exactly.

| train seed | artifact | SHA-256 |
|---:|---|---|
| 20261191 | `casm-x20s-seed-20261191` | `5ae1d013c254abbffaaa37ea0259f8f71f9983c55f2fadcdc83154e30cf0c970` |
| 20261192 | `casm-x20s-seed-20261192` | `261d9c1b77d93d401897f9ca8e79a752ff237d991d8811c5d72830fb962887ad` |
| 20261193 | `casm-x20s-seed-20261193` | `50fb7665dc1def457f66925ae173c41e457d0768eafe291fc5eff765e4be415f` |
| 20261194 | `casm-x20s-seed-20261194` | `9def3e313ffb90a8de9a254fbaf0c69a3d8e0b218e0adf5854e76e24340219b9` |
| 20261195 | `casm-x20s-seed-20261195` | `92a168f292b0dc39d51318559b7d163bf8e03033a2e44bd6595c9a778649279b` |
| 20261196 | `casm-x20s-seed-20261196` | `60a813b6c4c87ae2c53e68c4681c21deb92e377376241073a24a3477730fe454` |

## Provenance contract verification

Every result records and satisfies the frozen run contract:

- 12,000 training steps;
- batch size 128;
- train depth 12;
- `eval_n=256`;
- 8 candidates;
- trained live cardinalities `[2,3,4]` with exactly 4,000 training steps each;
- unseen live cardinalities `[5,6]`;
- exact X20R binary-forward straight-through estimator;
- hard evaluation threshold `g_soft >= 0.5`;
- no live-mask/cardinality/hidden-state supervision;
- no per-candidate learned gate table;
- deterministic schedule values matching the preregistration;
- identical final storage coefficient `0.05` for the scheduled selective-state treatments.

## Preregistered classification

### Positive-control validity

`canonical_live_mask`: **PASS, 6/6 seeds**, across live cardinalities `n=2..6` and depths `12/24/48/96` under the frozen thresholds.

X20S is therefore valid and the learned-treatment failures are interpretable.

### Seen competence

| treatment | seeds passing full seen competence |
|---|---:|
| `immediate_st` | 0/6 |
| `no_storage_st` | 0/6 |
| `delayed_abrupt_st` | 0/6 |
| `delayed_ramp_st` | 0/6 |
| `delayed_ramp_structure_blind` | 0/6 |

Every seed in every learned treatment has at least one seen IID/base hard-answer cell below the preregistered 80% optimization-failure boundary. In fact, across seen cells the learned regimes' hard answer-final accuracy remains approximately `2.7%..9.0%`, far below competence.

### Immediate straight-through replication

`immediate_st` reproduces the X20R all-absent failure:

- hard existence F1 = `0` throughout seen and unseen cells;
- mean hard record count = `0` throughout;
- raw learned gates collapse near zero;
- seen hard answer-final remains approximately chance-level.

This confirms that the pinned environment and fresh-seed panel reproduce the X20R treatment failure.

### No-storage diagnostic

`no_storage_st`: **FAIL, seen competence 0/6**.

Removing the storage penalty entirely does not establish correct hard-forward execution. Seen hard answer-final remains approximately `2.7%..9.0%`. Gate behavior varies by seed: some solutions under-instantiate, while seed `20261193` over-instantiates roughly six or more records, but none approaches the frozen state-selection and capability thresholds.

Therefore X20S does **not** support the hypothesis that storage pressure is necessary for the initial hard-forward collapse. Answer-only hard-forward training itself is insufficient under this estimator and optimizer.

### Delayed abrupt storage

`delayed_abrupt_st`: **FAIL, seen competence 0/6**.

The treatment receives exactly zero storage penalty through step 1000, yet by step 500 all six seeds already have answer loss at the no-state ceiling. Full storage pressure beginning at step 1001 therefore occurs after the answer-only phase has already failed to establish a useful discrete execution path. Final learned gates collapse to zero hard records across evaluated cells.

### Delayed ramp storage

`delayed_ramp_st`: **FAIL, seen competence 0/6**.

The same failure is present during the zero-storage warmup. The 1001..2000 ramp cannot repair a path that was not established before pressure begins. Final hard existence F1 and hard record count are zero throughout the evaluated cells.

### Storage-onset causal claim

**FAIL.** Neither delayed treatment satisfies the required 6/6 seen competence, much less 6/6 strong unseen extension. Because the zero-storage diagnostic also fails, the evidence specifically rejects storage-onset timing as the sufficient repair mechanism under this frozen setup.

### Strong and partial unseen extension

No learned treatment is extension-eligible because seen competence is 0/6 for all of them. Strong unseen extension is therefore **FAIL / INELIGIBLE**.

For completeness, no learned treatment satisfies the preregistered partial unseen criterion either. No averaging or isolated cell performance can rescue the failed seen gate.

### Structure-blind ablation

`delayed_ramp_structure_blind`: **seen competence 0/6**.

Because both graph-conditioned and structure-blind scheduled treatments fail seen competence, X20S cannot make a new causal claim about dependency/version connectivity under scheduled hard-forward training.

## Scientific boundary

The experimental sequence now establishes a narrower boundary:

1. X20: the supplied graph constructor can solve state instantiation on some soft-training seeds, but hard discretization is not robust.
2. X20R: naively moving record existence into a binary-forward straight-through training path collapses learned state existence on 6/6 seeds under immediate storage pressure.
3. X20S: removing, delaying, or ramping storage pressure does not repair that binary-forward optimization failure; even zero-storage answer-only hard-forward training is seen-incompetent on 6/6 seeds.

The next authorized research must therefore remain on the **discrete state-construction estimator/objective/credit-assignment boundary**. A successor may test a newly preregistered estimator or optimization mechanism that provides usable answer credit to discrete existence decisions without hidden state-selection labels, but X20S itself is frozen and must not be adapted post hoc.

Do **not** advance to reuse/merge, lifecycle deletion, persistent cross-episode state, controller/program induction, or verifier-guided repair from this result.

## Successor gate

The preregistered condition for reuse/merge was a valid graph-conditioned learned treatment passing full seen competence on 6/6 seeds and strong unseen `n=5,6` extension on 6/6 seeds. X20S has no such treatment.

**Reuse/merge is not authorized.**
