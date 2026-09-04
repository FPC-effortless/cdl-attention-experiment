# CASM-X20R frozen scientific result

## Verdict

**VALID experiment. Positive control PASS. Straight-through seen competence: 0/6. Strong state-instantiation extension: FAIL / INELIGIBLE. Partial extension: FAIL. Do not advance to reuse/merge.**

CASM-X20R tested whether replacing X20's fractional soft training-forward existence with deterministic binary-forward straight-through existence would repair the X20 hard/soft mismatch.

It did not. On every one of the six preregistered fresh seeds, the graph-conditioned straight-through constructor collapsed all candidate existence gates below the fixed `g_soft >= 0.5` hard threshold. Hard record count was exactly zero throughout all evaluated seen and unseen cardinalities, hard existence precision/recall/F1 were zero, and seen IID hard answer accuracy remained far below the preregistered 80% optimization-failure boundary.

Under the frozen preregistration this is therefore an **optimization failure at the discrete state-construction objective**, not evidence about unseen state-instantiation extension. Reuse/merge, lifecycle, persistent reuse, controller/program induction, and verifier-guided repair are not authorized successors from this result.

## Frozen provenance

- frozen X20 base: `3225172c78ca44ad57a26d64b13ae24f122b96bb`
- X20R preregistration commit: `5ee4b29e79e0acba91a0324adecdc800b95ca05a`
- authorized executable head: `300edab0ef4abcf130db4ce4957c00035da246de`
- scientific workflow run: `33918201861`
- contract/integrity job: PASS
- all six train/evaluate jobs: PASS, including provenance validation and artifact upload
- train seeds: `20261181..20261186`
- eval seeds: `20261261..20261266`

The earlier head `e528fe91c35f6c6c4393c24299c697a1fe4de21c` / workflow `33917685533` remains non-evidence because training never ran there.

## Artifact digests

| train seed | artifact | GitHub artifact SHA-256 |
|---:|---|---|
| 20261181 | `casm-x20r-seed-20261181` | `caa362b69311ff54a05a522aa9c9bb271874bf18b80e6c0c709e681255943be2` |
| 20261182 | `casm-x20r-seed-20261182` | `7928282638c5aceccfd7d616437830f81bc83becf8239e0cce5b889c602715ec` |
| 20261183 | `casm-x20r-seed-20261183` | `a80dbcd28575ab5a0bfda1d33281784de8021e536c554de71816b8b759187109` |
| 20261184 | `casm-x20r-seed-20261184` | `9b3e4d4da431afa88d7734ac885c35067d377161431ff7369160f911b8d26416` |
| 20261185 | `casm-x20r-seed-20261185` | `d3b75eb982e5bbe9a291e5a51c7c1f85de3d55cbb39ce6de8babb566a2b94b58` |
| 20261186 | `casm-x20r-seed-20261186` | `e1f8365222a2de1be11b8e6ed8aac91cd677832f7a0958d856984c767e745291` |

The downloaded ZIP bytes were independently SHA-256 checked and matched these six GitHub artifact digests exactly.

## Preregistered classification

### Positive-control validity

`canonical_live_mask`: **PASS, 6/6 seeds** across live cardinalities `n=2..6` and depths `12/24/48/96` under the frozen thresholds.

Therefore X20R is scientifically valid and the treatment failure is interpretable.

### Straight-through seen competence

`straight_through_instantiation`: **FAIL, 0/6 seeds**.

For all six seeds and all trained cardinalities `n=2,3,4`:

- hard existence F1 = `0.0`;
- mean hard record count = `0.0`;
- mean live gate converged to approximately `1e-10..1e-8`, depending on seed/cell;
- hard answer and state metrics are far below the frozen competence thresholds;
- every seed has seen IID/base hard answer accuracy below 80%, so every seed meets the preregistered definition of an optimization failure.

Representative seen-IID hard answer accuracies across `n=2,3,4` are only about `4.7%..8.2%` for seed 20261181 and similarly low for the remaining seeds. No seed is extension-eligible.

### Strong and partial unseen extension

Because seen competence is 0/6, unseen results are formally **ineligible** for a state-instantiation extension claim.

For completeness, the treatment also fails both the strong and partial unseen thresholds: at `n=5,6`, hard existence F1 remains `0.0` and hard record count remains `0.0` for every seed.

### Matched soft-X20 replication

`soft_x20_replication` reproduces the original X20 instability rather than showing a universal dataset/executor failure:

- seed `20261185` passes full seen competence and strong unseen extension;
- the other 5/6 soft seeds fail the every-seed robustness criterion, generally through the same fractional hard/soft mismatch or a weak soft solution.

Thus X20's prior seed bifurcation remains reproducible under the matched comparator.

### Continuous-relaxation repair claim

**FAIL.** The causal-repair criterion requires the straight-through treatment itself to pass 6/6 seen and 6/6 unseen. It passes 0/6 seen.

The evidence instead shows that naively putting the hard existence decision directly into the training forward path creates a stronger failure mode: the storage gradient and straight-through surrogate permit/drive an absorbing all-absent solution from which answer learning does not recover under this frozen objective and optimizer.

This last sentence is a bounded mechanistic interpretation of the observed collapse, not a claim that all discrete estimators will behave this way.

### Structure-blind ablation

`straight_through_structure_blind`: **seen competence 0/6** and likewise collapses to zero hard records.

Because neither graph-conditioned nor structure-blind straight-through regimes are seen-competent, X20R cannot make a new causal claim about graph connectivity under discrete training.

## Scientific boundary

The result narrows the remaining problem:

1. X20 showed that the supplied program graph contains enough information for some seeds to identify and execute the live state, but the continuous relaxation is not robustly discretized.
2. X20R shows that replacing that relaxation with a naive binary-forward straight-through estimator does **not** repair robustness; it instead collapses all learned state existence.
3. The next authorized work must remain on the **state-construction objective/optimization boundary** and must be preregistered before implementation.
4. Do not advance to identity reuse/merge, lifecycle deletion, persistent cross-episode memory, controller/program induction, or verifier-guided repair until robust discrete instantiation is established.

A scientifically justified successor should isolate why the hard-forward objective has an absorbing empty-state optimum without changing the task, labels, hidden-state access, evaluation threshold, or claim boundary. Candidate mechanisms may be studied only through a new preregistered experiment, not by adapting X20R post hoc.
