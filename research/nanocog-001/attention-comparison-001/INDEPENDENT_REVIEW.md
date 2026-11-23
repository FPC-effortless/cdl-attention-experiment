# Independent review — COG-ATTENTION-COMPARE-001

Reviewer: `/root/candidate_readiness`, a separate agent session that did not author the proposal. Review date: 22 November 2026, America/Los_Angeles. The reviewer shares the execution environment with the author; this is not independent-machine reproduction or a distinct GitHub identity.

## Decision

**ACCEPT AS A PROSPECTIVE SOURCE PROPOSAL AFTER THE CLARIFICATIONS BELOW. NOT EXPERIMENT ADMISSION.**

The proposal honestly distinguishes source reconstruction, software readiness, experiment admission and scientific results. It reports zero new model runs, retains unresolved identities and resource allocations as null/open, and does not claim that any modern baseline was defeated. Its three comparison tracks and complete-cost policy substantially address the user's fairness objective. The current deliverable is a protocol and admission inventory, not an executable model comparison.

Three narrow clarifications are requested before treating the cross-file proposal as a stable handoff. None requires a model run or opening evaluation data. They are not evidence of a failed scientific experiment.

## Findings requiring clarification

### R1 — Explicitly map every planned arm to its admission record

`CAMPAIGN.json` names `NANOCOG_PACKET`, `TRAINED_LOCAL_ONLY`, `KDA_GQA_3_TO_1`, `GDN_GQA_3_TO_1`, and `GDN2_GQA_3_TO_1`. `BASELINE_ADMISSION.json` instead uses `USER_CANDIDATE`, `KDA_HYBRID`, `GDN_HYBRID`, and `GDN2_HYBRID`, and has no independently trained local-only admission entry. Human readers can infer some intended correspondences; a later integrator should not need to.

Add explicit campaign-ID mapping for each inventory record and a trained-local-only record, or rename consistently. Preserve the distinction between an independently trained local-only model and disabling packets in a trained candidate. A generic hybrid entry must specify that this campaign's proposed global component is GQA, not silently inherit MLA from Kimi Linear.

Severity: moderate handoff/manifest ambiguity. This does not currently dispatch anything because the campaign explicitly forbids dispatch.

### R2 — Make the superiority bound direction unambiguous

`STATISTICAL_PLAN.md` says the interval must "cross the predeclared superiority threshold in the favorable direction." An interval that straddles the threshold would not establish superiority.

State explicitly: for a higher-is-better candidate-minus-baseline effect, the adjusted lower bound must exceed the positive practical-superiority margin; for lower-is-better quality, the adjusted upper bound must be below the negative margin. Ratio claims require the adjusted lower bound above the frozen ratio threshold. Keep the existing noninferiority rules, multiplicity family and pending statistical implementation gates.

Severity: wording ambiguity at a decision boundary. The surrounding rules are otherwise conservative and correct.

### R3 — Carry forward the inherited PULSE control without substituting it for the LM baseline

The inherited `QUALIFICATION.md` calls for unchanged pinned PULSE-006 direct-event and matched-broker lineages where executable, and distinguishes historical replay from retraining on a new panel. The new proposal appropriately treats that history separately, but does not explicitly identify its remaining role.

Add a brief statement that source-compatible PULSE controls remain required for applicable relational/mechanistic continuity checks when recoverable under the inherited contract. They are not general-language modern-baseline arms, and unavailable source/weights/recipes remain unavailable rather than being replaced silently. This preserves the inherited gate without bloating the primary LM roster or granting protected-data access.

Severity: inherited-scope clarity. The existing opening statement already says the proposal does not replace or lower prior gates.

## Verified strengths and scope

- Source reference `8374f72ba9324fa4a11d0d3ecb40a6db0e311c73` is separated from the unselected experimental source, configuration and checkpoint.
- The current packet candidate is not conflated with the older approximately 31M COG-LM lineage.
- Common-backbone, training-budget and quality-constrained serving comparisons answer different questions and retain distinct conclusions.
- Capacity matching does not use inactive-parameter padding; necessary backbone changes must be disclosed in separate comparisons.
- Equal tuning resources and development access are required. Failed trials and divergence remain accounted for.
- Training horizon, gradient horizon and state reset/detach behavior are separated. A persistent inference state is not treated as proof of long-range training.
- The KDA+GQA arm is explicitly named as a variant, not the released Kimi Linear model.
- CSA2 is conditional on adequate selector training and cached decoding, with the implementation limitation scoped to the cited NeMo implementation.
- No literature GPU speed number is inserted as a local comparator result; CPU/reference and optimized accelerator measurements remain separate.
- Native and oracle results are separated; raw-text input, collateral preservation, multi-fact density and unseen wording are part of the intended evaluation.
- Evaluation failures remain in success denominators; completion-time censoring is distinguished from consumed time and successful completion.
- Noninferiority is not inferred from a nonsignificant difference. Proposed margins and five-seed planning are explicitly unaccepted and unpowered.
- The inference-first preference is applied after quality requirements. It is not used to add incompatible units or excuse omitted training cost.
- No restricted Library references, raw labels, checkpoint bytes or credentials occur in the new proposal/index append inspected here.

## Statistical source checks

Read-only inspection of the supplied Veritas source snapshot confirms:

- At most 1,000 unit labels and 32 comparison labels.
- `units × seeds × comparisons <= 20,000`.
- `units × seeds × comparisons × draws <= 5,000,000`.
- At most 10,000 draws, with a two-sided adjusted tail of `alpha / (2 × comparisons)`.
- Inference eligibility can be reached with only two units, two seeds and one expected tail draw; this is not a power guarantee.
- For 1,000 units, 5 seeds and 4 comparisons, the 20,000-cell design permits only 250 draws under that operation cap. At alpha 0.05, each tail has 1.5625 expected draws.
- The existing point estimator is an equal-cell mean of candidate-minus-baseline values. It does not implement token-weighted NLL or arbitrary cost-ratio estimands.
- The function may report `TESTED` while retaining `scientific_qualification = NOT_QUALIFIED`.

The proposal correctly preserves these as statistical admission gaps rather than weakening resource checks or downsampling final observations. No native source module was imported or executed for this review. The reviewer did not validate bootstrap coverage empirically; that remains required before confirmation.

## Roster and measurement assessment

The proposed packet/local-only/GQA/MLA/KDA+GQA core is a defensible first controlled comparison, with GDN/GDN2 challengers and conditional CSA2 scope disclosed. It would not alone prove superiority over all efficient architectures, released frontier models, other scales, or other hardware. The protocol explicitly limits those claims.

The timing design includes complete sessions, tokenizer/input preparation, prefill, decode, routing/index updates, state maintenance and recovery. Fixed-output microbenchmarks are separated from actual completed tasks. This supports the intended comparison once implemented. Exact batch/concurrency cells, precision policy, timing block count, cache-reuse policy, output policy and statistical latency estimand must still be frozen before measurement; the planning grids are not measured capabilities.

The absent trained candidate, unintegrated modern baselines, unknown runner allocation and unvalidated weighted/ratio analysis are real execution prerequisites. The proposal does not portray user permission as the missing scientific evidence.

## Index preservation and concrete checks performed

A standard-library byte comparison verified that the original `source_snapshot/nanocog_results_index.md` (5,993 bytes) is an exact prefix of the proposed `RESULTS_INDEX.md` (7,014 bytes). Existing history is unchanged; the append is explicitly prospective and disclaims completeness of the historical index.

Both new JSON files parsed successfully. This is syntax validation, not schema/admission certification. The reviewer read all seven proposed files, the source statistical implementation, and the inherited qualification/model readiness records. No model, checkpoint, protected dataset, provider or native experiment was executed.

## Reviewed file hashes before requested clarifications

| File relative to `research/nanocog-001` | SHA-256 |
| --- | --- |
| `RESULTS_INDEX.md` | `6c9935a9feb0875e4c6d33c9a3361ef1916d77ed1bff1e0228b48639b4570407` |
| `attention-comparison-001/BASELINE_ADMISSION.json` | `746d9a97a153b87356c67adc458c2b030b713c7ead9bfe3971d19f1aa3d811b6` |
| `attention-comparison-001/CAMPAIGN.json` | `7106bdb722963fad12c226274e91a61bdc797931188f00c8f61daa420df09abb` |
| `attention-comparison-001/PROTOCOL.md` | `0ed0c0f13b88305803cd4b71f4e25da43216c982a33edaa330b96563b7fabccf` |
| `attention-comparison-001/READINESS.md` | `bb3193d5e71c54aa55a0cdba2c2b082c6bb75edbc64ef4ed366e05a9a73af6a2` |
| `attention-comparison-001/SOURCES.md` | `76517470bd6f113ae7a5866293a4358ede6a05c851f97830e44ee841e920064c` |
| `attention-comparison-001/STATISTICAL_PLAN.md` | `a024abde192607521d6ef11dad56b828410f082b380d3fb0ccb593784abb7371` |

Any amended proposal needs a new exact-file review record. This review does not establish merge-authoritative GitHub approval, experiment admission, independent-machine model reproduction, or scientific qualification.

## Final amendment review

**Final disposition: ACCEPT_PROSPECTIVE_SOURCE_ONLY. R1, R2 and R3 are resolved in the exact files listed below. Model execution remains NOT ADMITTED and scientific qualification remains NOT QUALIFIED.**

The reviewer reread the amended passages and JSON records. The configuration map covers all eight planned core/challenger/conditional IDs exactly, every mapped inventory ID resolves, and each proposed recurrent hybrid explicitly uses a 3:1 recurrent:GQA ratio. A distinct trained-local-only record now preserves its independent training and state-accounting obligations. Both inventory and campaign retain the inherited diagnostic controls. The protocol preserves exact-source PULSE continuity and text-facing deterministic controls without admitting their execution or making them general-language competitors. The statistical superiority sentence now requires the appropriate adjusted bound strictly beyond the practical margin.

Metadata checks confirmed that dispatch and execution flags remain false, completed model runs remain zero, results remain empty and all admission gates remain OPEN. The original results-index bytes remain an exact unchanged prefix. These are document/metadata checks only; no native model or campaign code was executed.

| Final reviewed file relative to `research/nanocog-001` | SHA-256 |
| --- | --- |
| `RESULTS_INDEX.md` | `6c9935a9feb0875e4c6d33c9a3361ef1916d77ed1bff1e0228b48639b4570407` |
| `attention-comparison-001/BASELINE_ADMISSION.json` | `399961898ab8650ff9ca81e0d8b4b8f26e36f24c67b89f337f274d34cc38d1c8` |
| `attention-comparison-001/CAMPAIGN.json` | `d69cac7c1d697e8dcba7f2bcafa8de7dfa4502038d50a1ef6f476cc7540c14e5` |
| `attention-comparison-001/PROTOCOL.md` | `06e9534442697147bf321328f30e681cbe6c4bd44dd6fb9f6e77904263bc51c2` |
| `attention-comparison-001/READINESS.md` | `bb3193d5e71c54aa55a0cdba2c2b082c6bb75edbc64ef4ed366e05a9a73af6a2` |
| `attention-comparison-001/SOURCES.md` | `76517470bd6f113ae7a5866293a4358ede6a05c851f97830e44ee841e920064c` |
| `attention-comparison-001/STATISTICAL_PLAN.md` | `be3c4f31ade0c697ae343f25169c1c6a72161460617399a5c353a920222360b2` |

This disposition is suitable for a reviewable documentation/JSON draft. It does not certify the future numerical implementation, external baseline installation, statistical coverage, resource availability, merge authorization, or research outcome. Any subsequent change to the listed bytes requires scope-appropriate review before using this exact-file disposition.
