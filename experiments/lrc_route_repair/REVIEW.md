# LRC route repair: independent review

## Review scope and status

This document records prospective review conditions for the sentence-based LRC route repair on branch `research/lrc-route-repair-002-20261108`. It is not a capability result. The new model, frozen protocol and completed evidence will be reviewed separately as they become available. The reviewer owns this document and `audit.py`; implementation and training belong to the other experiment authors.

The local `AGENTS.md` and `fpc-universal` research workflow were read. The recovered original sentence adapter and `lrc_diagnosis_001/mechanism/MECHANISM_DIAGNOSIS.md` were inspected. No original source or result was changed.

## Defects the repair must address

The archived writer assigns all four location records within an era the same hard key. The archived first page/slot decision is a deterministic function of the query alone; its second decision is a function of the query and first retrieved content. Neither observes the order or contents of the bucket whose ordinal it selects.

For two histories containing the same records in different orders, unchanged query and record encodings imply the same chosen ordinal. If the target record changes ordinal inside a collision bucket, that ordinal cannot identify the target in both histories. More page capacity does not supply the missing observation.

A repair therefore needs useful person distinctions in the learned key or a content-aware decision among a strictly bounded set of candidates, with an explicit overflow policy. The intended candidate combines both. Merely lowering a dense auxiliary loss, increasing address occupancy, or adding a teacher-forced first hop does not establish either repair.

## Prospective regression and mechanism checks

The thresholds and complete cohort are the responsibility of the frozen protocol. The tests below specify what must be measured and what a claim of correctness means. Structural checks and learned-checkpoint results must be reported separately.

| Check | Construction | Required observation |
|---|---|---|
| Person distinctions | Hold era, relation and payload fixed; change the subject across the four people. Inspect hard writer keys and candidate scores. | Report pairwise hard-key collision rate and distinct keys per era/relation. The trained model must not silently retain the original four-person collapse and call it repaired. If collisions remain, report whether the bounded content selector resolves them. |
| Location payload invariance | Use the original value-change quartets, whose only edit is the relevant location token. | Track the gold record's hard writer key across all four alternatives. Answer predictions must follow the changed payload when the correct route is selected. |
| Partnership payload invariance | Use bridge-switch quartets, whose only edit is the first supporting fact's partner value. | Track the first supporting record's writer key across the four alternatives. The second query/selected location must respond to the changed bridge. |
| Same facts, different order | Reverse records within each era and map occurrence IDs exactly. Include unique-score synthetic collision cases in the index/selector tests. | With no truncation or ties, the selected record content and answer are unchanged while occurrence IDs follow the permutation. Query-only content-blind ordinals fail this test. Report any learned route failures rather than averaging them away. |
| Candidate-content sensitivity | Fix the query and returned candidate keys, then exchange the candidate contents and occurrence mapping. | A unique-best content score follows the matching content. The selector must receive actual candidate features, not just query, bucket length, page or rank. |
| Two distinct hops | Inspect selected IDs and the two labeled supporting records. | The task's two correct records are unique and distinct. Report hop 1, hop 2, joint-support and distinct-selected-ID rates independently of answer accuracy. |
| Bridge dependence | Four variants keep the question, all locations, and all irrelevant facts fixed; four bridge values imply four different answers. | Report answer accuracy, perfect-quartet accuracy and both-support accuracy. A single unchanged answer per quartet scores 25% and zero perfect quartets; that cannot establish two-hop execution. |
| Candidate cap | Construct empty, singleton, exact-cap and over-cap buckets, and histories with increasing N. | Actual inspected candidates per hop never exceed the declared cap. No empty-bucket fallback performs a global search. Include bucket population, retained/returned counts, omitted counts and overflow indicators. |
| No hidden scan | Inspect normal query code and instrument index reads. Test multiple N values using already built memory. | One-time encoding/index construction may touch N records. Each ordinary query accesses only its bounded candidate sets; any dense training-only scorer is absent from that path. |
| No support leakage | Review the normal forward signature/callers and replay identical records/query with perturbed external labels/metadata. | Ordinary predictions and routing depend only on learned weights, records and question. Semantic parsing is permitted in the independent dataset auditor, not in model inference or key construction. |

Permutation invariance has a precise boundary: retaining only the most recent C records from an overflowing bucket is order-dependent by design. Such overflow cannot be hidden by applying the invariance claim only after observing failures. The protocol must declare its policy, expose every overflow, and separate bounded-runtime correctness from task adequacy. Increasing C is a changed runtime budget and must be reported against the archived cap of two candidates per read.

## Required experimental boundary

1. Engineering decisions use explicitly named DEV data and DEV model seeds only. Reusing the exposed original evaluation sets is diagnostic, not fresh validation.
2. Before the complete cohort is fitted, freeze the model, losses, hyperparameters, training data, seeds, baseline definitions, candidate budget, evaluation conditions, gates and timing method. Hash all scientific source files and the protocol.
3. Fit every preregistered arm and seed to its declared endpoint. Do not select a checkpoint on TEST. Preserve negative or failed fits and record any excluded run with the reason.
4. Close all fits and hash every checkpoint before generating the fresh TEST cohorts. Generate TEST exactly once from the frozen seeds. Do not launch further fitting after TEST exposure under the same confirmatory claim.
5. Score ordinary label-free inference first. Diagnostic interventions, semantic parsing, key tables, forced reads and shuffled labels are clearly labeled and cannot be counted as ordinary competence.
6. Recompute the complete result table from saved predictions and independently replay a bounded checkpoint subset before reporting a decision.

The cohort should include the original sentence LRC training, repaired LRC with correct supports, repaired LRC with jointly shuffled support pairs, and the conventional supervised memory network. Three fresh fit seeds and fresh IID, held-out-pair and bridge-switch worlds provide the intended confirmatory scope. Ablations of each repair component are useful mechanism evidence if frozen and budgeted in advance; an unrun ablation remains an unresolved attribution limit.

The historical IID and held-out capability gate was at least 95% answer accuracy and 90% perfect value-change quartets in every seed. The new protocol must also name its two-support and bridge requirements before fitting. The held-out generator fixes the target partner to the next cyclic person, so held-out answer accuracy alone does not prove that the first record was used causally.

## Fairness and cost accounting

- Report allocated and active parameter counts for every arm. Approximately 19k is the current comparison target, not a general restriction on future research.
- Shared data, labels and update count are necessary but do not equalize compute. Record training examples, forward calls, dense training scorer use, optimizer steps, elapsed fit time and thread settings.
- Time ordinary inference including record encoding and index construction for the original classification workload. If prepared-memory query latency is also reported, label it separately and measure the analogous baseline work. Excluding index construction from one arm alone is not a matched end-to-end result.
- Record candidate inspections per hop, total reads, overflows and null reads. A model scanning all 16 candidates before returning one is a dense memory network variant for this accounting, even if its final output is a hard index.
- Quality gates precede cost ranking. The user's inference-to-training preference is represented by `J = 12 ln(I / I_memnet) + ln(T / T_memnet)` when comparable timing is available. It is not a requirement for a twelvefold inference speedup.
- CPU timings under concurrent training are resource receipts, not clean cross-arm speed comparisons. Use isolated warmed measurements and preserve individual repetitions and the aggregation rule.

## Limits that remain even after a successful repair

This finite, fixed-template sentence task supplies record boundaries and a small known vocabulary. A successful repair would establish the measured routing behavior on fresh worlds from these generators. It would not establish raw-byte Tiny Shakespeare improvement, general language understanding, large-memory scaling, learned entity recognition in unrestricted text, or novelty relative to prior memory architectures.

The no-scan claim concerns the prepared-memory query phase. Building the memory remains linear in the number of records, and learned keys may require index rebuilding after model updates. Collision handling with a finite cap also has an explicit recall limit when the relevant item is omitted.

## Review record

- Prospective checklist issued to the implementation author and index author before the repair implementation was present in this directory.
- Candidate design received: width-32 sentence encoder; normalized learned writer and first/second reader MLPs; 1,600 steps of continuous support metric learning; a 16-centroid spherical codebook fitted to unique TRAIN record strings with five inertia-selected restarts; 400 reader-distillation steps with frozen encoder/writer; two code probes and at most two candidates per probe; a value-only answer decoder trained from detached labeled final-support representations. Exact production source and frozen protocol remain pending.
- Design assessment: defensible as a new LRC-derived sentence-routing variant. This changes the training method, discretization, coherent first-hop training input and answer decoder as well as the two original defects. Without a frozen factorial ablation the causal attribution is to the combined recipe. No claim that it is a minimal patch or an architecturally novel result is justified.
- Runtime accounting: two probes times two candidates means at most four scored records per hop, twice the archived maximum of two. Comparing the query with 16 fixed centroids is codebook work independent of history length; it must not be confused with scanning all history records. K-means time belongs in training cost.
- Shuffled-control condition raised: the joint support-pair shuffle must apply consistently wherever support indices affect training. If correct final-support values are intentionally retained for the answer decoder while routing labels are shuffled, that alternative must be stated explicitly and its detached gradient boundary verified; it is not an entirely shuffled-support arm.
- Index author reports 15 structural tests, including forbidden-iteration wrappers for million-entry overflow buckets and 100,000 distinct-code dictionaries. Independent code inspection confirms that overflow returns no candidates after checking bucket length, and lookup uses one dictionary `get`. These checks do not by themselves establish that the neural resolver scores only the returned candidates.
- Auditor parser qualification executed on 8 IID TRAIN-schema examples, 8 already designated DEV held-out examples, and 8 bridge qualification examples. Exact labels/templates and tensor digest parity passed; deliberately corrupted answer labels were rejected for all three kinds. No fresh TEST tensors were generated or read.
- Complete cohort and fresh TEST evidence: pending.
- Independent postfit audit: pending.
