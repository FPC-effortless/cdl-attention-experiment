# LRC route repair: independent review

## Review scope and status

This document records the prospective conditions, pre-freeze production review, and completed independent audit for the sentence-based LRC route repair on branch `research/lrc-route-repair-002-20261108`. The reviewer owns this document and `audit.py`; implementation and training belong to the other experiment authors. The completed evidence is reported at the end, separately from the prospective checks.

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
- Candidate design received: width-32 sentence encoder; normalized learned writer and first/second reader MLPs; 1,600 steps of continuous support metric learning; a 16-centroid spherical codebook fitted to unique TRAIN record strings with five inertia-selected restarts; 400 reader-distillation steps with frozen encoder/writer; two code probes and at most two candidates per probe; a value-only answer decoder trained from detached labeled final-support representations. Production integration was still pending at this initial design review; its subsequent review is below.
- Design assessment: defensible as a new LRC-derived sentence-routing variant. This changes the training method, discretization, coherent first-hop training input and answer decoder as well as the two original defects. Without a frozen factorial ablation the causal attribution is to the combined recipe. No claim that it is a minimal patch or an architecturally novel result is justified.
- Runtime accounting: two probes times two candidates means at most four scored records per hop, twice the archived maximum of two. Comparing the query with 16 fixed centroids is codebook work independent of history length; it must not be confused with scanning all history records. K-means time belongs in training cost.
- Shuffled-control condition raised: the joint support-pair shuffle must apply consistently wherever support indices affect training. If correct final-support values are intentionally retained for the answer decoder while routing labels are shuffled, that alternative must be stated explicitly and its detached gradient boundary verified; it is not an entirely shuffled-support arm.
- Index author reports 15 structural tests, including forbidden-iteration wrappers for million-entry overflow buckets and 100,000 distinct-code dictionaries. Independent code inspection confirms that overflow returns no candidates after checking bucket length, and lookup uses one dictionary `get`. These checks do not by themselves establish that the neural resolver scores only the returned candidates.
- Auditor parser qualification executed on 8 IID TRAIN-schema examples, 8 already designated DEV held-out examples, and 8 bridge qualification examples. Exact labels/templates and tensor digest parity passed; deliberately corrupted answer labels were rejected for all three kinds. No fresh TEST tensors were generated or read.
- Complete cohort and fresh TEST evidence: completed after the production review below.
- Independent postfit audit: passed; results and limits are recorded below.

## Production review before confirmation

The production `model.py`, `train.py`, `run.py`, `index.py` and prospective `PROTOCOL.md` were inspected. Ordinary inference accepts records and question only. It computes fixed-codebook scores, retrieves bounded posting lists, deduplicates addresses/candidates and scores only the gathered candidate keys. Exact score ties fail closed. The first retrieved value determines the second query; a missing first record now aborts the chain. The answer head uses only the actual final selected value.

The review found a concrete Stage-B clipping bug: frozen encoder/writer parameters retained their last Stage-A gradients, and clipping all router parameters included those stale gradients. Production now clears the frozen gradients and clips only the active parameters. The original DEV key-probe source/results remain preserved as executed; the corrected full model was separately qualified on DEV seed 84112. The implementation author reports 100% answer and both-support scores on all three DEV sets, 100% on the paired-collision stress, 34.375% for the content-blind ordinal collision ablation, and 100% with one probe. These are development observations, not the fresh confirmation cohort.

The final protocol declares 12 fixed fits across four arms and seeds 901–903; 1,600 key-learning plus 400 reader-distillation steps; 12,900 total repair parameters with 512 codebook buffer values; two optimizer calls per minibatch; joint support shuffling in every teacher/target use; original-LRC and MemNet references; five fresh TEST sets generated after fit closure; and explicit answer, quartet, joint-support, overflow, candidate-bound, permutation and shuffled-control screens. Source hashing covers the imported original Python package as well as the new scientific source. Receipts preserve both key optimizer phases, RNG state, exact minibatch digests and closed update-log hashes.

The reviewer independently reran the 15 index tests successfully. The index/model author and root report all 29 structural tests passing, including instrumented candidate matrix operations in a 10,008-record history, content swaps, exact ties, permutations and null-chain aborts. These structural results establish the access contract under their fixtures, not learned task quality or large-memory recall.

No blocking review finding remained before the frozen confirmation launch. The independent postfit auditor was implemented without opening the fresh TEST artifacts. Its planned scope was source/output/fit hashes, all 24,000 minibatch rows, optimizer schedules, tensor labels, quartet edits, input overlap, all 90 result cells, per-read bucket accounting, key diagnostics, declared gates and 16 native replay examples per cell. Audit success and scientific gate success remain separate fields.

## Completed independent audit

**Integrity passed. All six frozen repair quality/mechanism screens passed. The repaired model still loses the measured CPU cost comparison to MemNet.**

The audit ran only after all 12 fits, fresh scoring, and isolated timing had finished. It did not fit a model, generate another evaluation cohort, select a checkpoint, change the runtime probe count, or retime any model. The scientific freeze was local commit `6c3976e`. Evidence is in `/workspace/scratch/3f2444385d54/lrc_route_repair_002/confirmation/independent_audit.json`.

The following were independently checked:

- Source hashes include the new scientific source and all imported original-package Python files. Checkpoint hashes, closure/evaluation ordering, embedded and external receipts, optimizer/RNG states, exact shared minibatch streams, and all 24,000 finite update-log rows agree.
- All five 1,024-example TEST sets have exact valid templates, unique distinct gold support records, correct answers, and the declared single-token quartet edits. There is no exact input overlap with TRAIN or between the five TEST sets. Quartet members remain dependent examples.
- All 90 score cells and 92,160 saved predictions reconcile with the saved tensors. The six frozen screens independently recompute to PASS.
- Every saved repair read was checked by independently rebuilding its discrete bucket membership. Returned/inspected counts, probes, bucket sizes, overflow counts, null chains, ties and selected-ID eligibility agree with the claimed bounded access contract.
- Direct native replay used 16 examples per cell, spread over four full quartets across each set: 1,440 examples in 90 cells. Every discrete field matched exactly, including answers, selected support IDs, write/read codes and counters. Maximum per-example loss difference was `1.1920928955078125e-7`.
- The cost auditor reconciled all 270 recorded block means, nine median timing summaries, 8,640 measured complete calls, training-time ratios and weighted cost values. This verifies arithmetic and recorded scope; it is not an independent hardware replication.

### Fresh quality and mechanism findings

Repair and MemNet both achieved 100% answer accuracy, 100% perfect answer quartets, and 100% both-support accuracy in every one of the 15 seed-by-primary-split cells. The repaired model therefore fixes the measured sentence-task behavior without outperforming the conventional reference on quality.

| Arm | Mean answer accuracy over six withheld cells | Mean both-support accuracy over six withheld cells |
|---|---:|---:|
| Repair | 100.0000% | 100.0000% |
| MemNet | 100.0000% | 100.0000% |
| Fresh original LRC control | 40.3809% | 8.1380% |
| Repair with shuffled support pairs | 34.3424% | 1.9206% |

Every repaired story used exactly 16 hard codes. For both partnership and location facts, every era had four distinct person keys; the measured person-pair collision rate was zero. The relevant writer key remained unchanged across every location-payload quartet and every partnership-bridge quartet. These findings hold in all 15 primary cells. Normal inference scored exactly two candidate records per hop, with no overflow or ties.

All six reversal checks preserved answer predictions, mapped support occurrences and record-aligned writer keys. Under the deliberately paired-code collisions, content-aware resolution retained 100% answers, both supports and perfect quartets for every seed, with at most four candidate scores per hop. Replacing only that collision decision by the declared content-blind lowest-occurrence selector reduced bridge answer accuracy to 34.2773%, 40.0391% and 30.4688% (mean 34.9284%). This is a controlled selector stress test, not a retraining ablation of every architectural ingredient.

The fixed one-probe diagnostic also scored 100% on the bridge set in all three seeds. It remains a diagnostic: selecting that faster runtime after TEST and presenting it as the frozen primary configuration would be test-driven selection.

The newly fitted original controls had mean location-code counts between 1.0 and 1.5 per era across seed/split cells. The exact single-location-code collapse in every era was established for the archived checkpoints; it must not be incorrectly attributed to all newly fitted controls.

### Cost findings and decision boundary

| Seed | Repair complete call | MemNet complete call | Repair / MemNet | Repair full fit | MemNet full fit | Weighted cost J |
|---|---:|---:|---:|---:|---:|---:|
| 901 | 0.565311 ms | 0.214150 ms | 2.6398× | 9.4538 s | 8.8656 s | +11.7126 |
| 902 | 0.608827 ms | 0.207866 ms | 2.9289× | 8.6518 s | 8.8992 s | +12.8675 |
| 903 | 0.566014 ms | 0.209577 ms | 2.7008× | 9.1228 s | 9.6187 s | +11.8694 |

The reported inference values are medians of 30 complete-call block means for each model/seed. Repair is faster and more accurate than the fresh original LRC control, but matches MemNet quality while taking approximately 2.64–2.93 times its complete-call CPU latency. Training costs are similar. Positive J in all three seeds means the repaired model loses the declared inference-weighted cost comparison against MemNet.

The supported conclusion is a successful combined architectural/training repair of this finite sentence-routing task. Remaining limits include the fixed known vocabulary/templates, 16-record workload, 16-code capacity, doubled maximum candidate budget, dense and teacher-supported training, differing parameter counts, three fit seeds, and lack of an ingredient-by-ingredient training ablation. There is no demonstrated Tiny Shakespeare gain, new-entity generalization, long-memory recall, broad language-model capability, or cost superiority over the conventional baseline.

### Evidence hashes

| Artifact | SHA-256 |
|---|---|
| Prefit manifest | `47f79043bd4736ab09371795d30b72d8da8507a8459b0249009f66bebb34ff4c` |
| Fit closure | `fd44625cf7514615c6204aad4de2d1d33db52f2e3796c25220fbdd700b41b7dc` |
| Scoring completion | `b551c3c1c479e9785e87b8649ec7e65d5e970d2756df7ec3b06c832c645318e6` |
| Result table | `8ea961c5ce0a980e2bfa02052a30e2f2fc5a2bf18a5569e3946e43a3baa4410f` |
| CPU benchmark | `8999b50e39197c89518bb608799c3b17f3991685639e4857312dcecffba68d42` |
| Independent audit | `aeb895e77bf0e872b47c73c276a1c1649b42346dd1ceac2c66a8a06eff5a503f` |
