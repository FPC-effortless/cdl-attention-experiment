# COG-FUSION-001: prospective independent review plan

## Status and purpose

This review plan was written before inspection of any COG-FUSION-001 training or final evaluation result. It specifies implementation and experimental-integrity checks, not results. The study is a local diagnostic-scale evaluation of a new fused implementation. It cannot by itself establish frontier capability, general language-model quality, attention replacement, or an efficiency advantage.

The root runner will lock the precise architecture/configuration, dataset hashes and sizes, train/dev/test partitions, optimizer, update budget, any CPU cap, stopping rule, evaluation sizes, and source hashes prospectively after a TRAIN-only timing calibration. None of those unresolved quantities is silently authorized or selected by this review plan.

The intended paired initialization seeds are **17, 29, 41**. Both candidate and separately trained memory-off control use the same nominal parameterization, same initial tensors within a seed, matched semantic training examples, and the same fixed optimizer-update budget. All local backbone weights are trainable. All six final outcomes must be reported. Any incomplete fit remains incomplete; it must not become a selected winner.

## Main research question

Can this new continuously carried COG-based model learn to convert event text into protected factual updates, route a question through actual selected records, and use the resulting evidence in character generation, while preserving correct answers under irrelevant edits and changing them correctly after relevant corrections or retractions?

Keep these distinct:

1. Typed storage and recovery correctness with supplied operations.
2. Learned semantic correctness of the writer and query interface.
3. Causal use of selected factual evidence by the trained character generator.
4. End-to-end whole-history and complete-counterfactual-pair correctness.
5. Measured implementation cost on this host and workload.

A success in one category must not replace a failure or missing measurement in another.

## Experiment-lock audit

Before accepting any final evaluation result, check that the frozen manifest identifies:

- New semantic family identities and seeds for TRAIN, DEV, familiar held-out evaluation, and syntax-shift evaluation.
- The exact wording/template partition, ontology, character tokenizer and unsupported-character policy.
- Final architecture ID, exact total/trainable parameter counts, initialization/checkpoint origin, and whether all intended backbone parameters receive gradients.
- Definitions of candidate, separately trained memory-off control, acute evidence removal, strict routing, and hybrid routing.
- Optimizer/update budget, batch structure, carry/reset policy, TBPTT boundaries, loss terms/weights, checkpoint selection rule, and CPU timing/termination rules.
- Deterministic dataset and source hashes, plus all calibration work and its exclusion from scored fits.
- Definitions and denominators for every primary metric and intervention panel.

The old MEMORY-INTEGRATION-003 fresh panel is already exposed development evidence. It must not be relabeled a fresh final test for this study.

## Dataset and leakage review

Inspect generator, exported data and training batch construction directly.

- Semantic roots must be disjoint across splits. Counterfactual worlds of one root stay in the same split. Augmentations and role/name permutations of one root remain attached to that root.
- Check exact duplicate histories/pairs across partitions. Report intentionally shared primitive facts/templates separately from accidentally duplicated complete examples.
- Build tokenizer from an explicit training vocabulary or declare a fixed inherited tokenizer. Do not use held-out labels or template identities to adapt tokenization, thresholds or model construction.
- Held-out syntax templates must actually be absent from TRAIN and DEV. Report unseen vocabulary separately from new syntax; their combination cannot isolate syntax generalization.
- Training may use supervised operations, routes, support identities and answer targets in its loss. Inference must not receive them. Trace evaluator-only fields through every call boundary.
- No filename, family index, RNG seed, padded position, branch tag, request ID, gold world ID or output length may leak the correct label into model inputs.
- Counterfactual labels must differ where the pair is intended to test answer changes. Reject duplicate worlds or pairs with unchanged targets from that denominator.
- Check relevant and irrelevant edit labels against an independent typed reference, including both corrected endpoints and changed first-hop links.
- If training uses gold ledgers or support teacher forcing, disclose it and preserve fully predicted writes/routes/ledgers at final evaluation. Report exposure-bias failures rather than repairing them with reference state.
- Do not filter final evaluation to already-correct writers or routes for the headline result. Such conditioned diagnostics must preserve denominators and remain secondary.

## Model-path and state-regime review

Inspect the actual production inference methods and hooks, not only the training loss.

- Native recurrent state carries across events and questions according to the same declared regime during training and deployment. Only the event-local accumulator resets at declared boundaries.
- Truncated backpropagation detaches are explicit. A detach is not a state reset. Training horizons must include tasks requiring factual retention beyond locally useful recurrence.
- Every new input character and generated character is processed once in the claimed single-pass path. Count any additional semantic encoder pass separately.
- The factual archive and the COG internal recurrent board are distinct state objects. Interventions on one do not prove use of the other.
- Writer inference uses only the input stream and allowed carried state. Schema-valid operation prediction is not proof of correct interpretation.
- Strict routing predicts a full logical key before exact lookup. A typed executor may follow fixed links, but this should not be described as learned second-hop routing unless the learned mechanism actually selects that hop.
- Hybrid candidate discovery has an explicit bound. Count records fetched and scored before filtering. Full scans followed by top-k are not bounded discovery. Record indexing/codebook work separately.
- Hop two must consume the actual first selected value. Gold bridge values, silently repaired addresses, and chosen-only-on-success diagnostics cannot enter the candidate inference path.
- Retrieved evidence reaches the original character-token logits through trainable parameters. Actual generated text is scored. A nine-way answer label converted to a string is a separate rendering control.
- Fixed-length decoding, stop characters, formatting constraints and teacher forcing are explicit. Do not give the candidate the gold answer length during inference.
- Entire native backbone trainability and actual evidence-path gradients are checked on a TRAIN fixture. Finite gradients alone establish a possible path, not useful semantics.

## Independent infrastructure checks

These checks use supplied typed operations/fixtures and must be labeled engineering evidence.

| Contract | Adversarial check | Required outcome |
|---|---|---|
| Protected updates | Change one identity repeatedly at full capacity | Unrelated records retain exact contents; overwrite does not allocate another current identity |
| Explicit capacity | Insert a new identity when the declared capacity is exhausted | Declared overflow/refusal/paging outcome; no silent unrelated eviction |
| Identity stability | Change values and physical placement while preserving logical key | Reads continue addressing the same identity |
| Missing/deleted dependencies | Resolve unknown, insert key, delete key, recreate key | Old packet invalidates at every relevant change |
| Fork identity | Substitute different contents with the same numeric revision | Old packet is rejected or rebuilt |
| Source and schema identity | Erase a source or change interpretation/model identity | Old dependent answer cannot commit |
| Pending correction | Interrupt a relevant edit, restart, ask before retry | Dependent answer withheld until successful resolution; unrelated answer permitted |
| Coherent transaction | Inject failure around fact, continuation, RNG and reply publication | Recovery exposes one coherent committed state |
| Reply publication | Invalidate a dependency during inference | No newly committed stale reply; no unvalidated answer published first |
| Request replay | Retry the same ID/content after an uncertain response | Return the historical receipt without repeating state mutation |
| Request collision | Reuse an ID for different text | Explicit conflict |
| Current re-query | Same question under a new request ID after correction | Current factual dependencies are used |
| Concurrent mutation | Modify a target after writer snapshot or a support during answer generation | Reject stale commit rather than silently rebasing |
| Sampling restart | Compare uninterrupted with restored complete state | Same tokens/RNG/state/counters under identical processing partition |
| Incremental input | Resume using only new suffix | No historical prompt encoding and no growing token KV cache |
| Hybrid limits | Empty bucket, tie, collision, overflow, changed first-hop link | Declared failure or correct bounded selection; no hidden fallback |

If an API intentionally has a weaker contract, document and test that contract; do not mark the stronger row passed.

## Learned evaluation and causal checks

The headline end-to-end score should report complete pairs over complete histories. A strict success requires all required predicted operations, every post-event factual ledger, and all scored final generated answers to be correct in both worlds. Record final-answer success separately so that error cancellation remains visible.

Preserve per-seed, per-arm counts for:

- Exact operation correctness, destination/value/commit components, invalid proposals, and damage to previously correct unrelated records.
- Whole-history ledger correctness and first failure location.
- Actual first/second selected support correctness, where the architecture predicts those supports.
- Generated exact-answer correctness and complete counterfactual-pair correctness.
- Relevant-correction response, unrelated-edit invariance, retraction/unknown behavior and stale-neural-state conflicts.
- Familiar held-out and syntax-shift families, separated by task and history length.
- Snapshot/restart continuation and no-replay accounting through actual trained inference.

Use these interventions on the trained model without fitting or selecting new checkpoints:

1. Change only the accepted relevant fact while holding incoming recurrent state and question fixed. Evaluate both directions.
2. Change an irrelevant fact under the same fixed inputs.
3. Replace the first-hop bridge with a different valid bridge while keeping terminal facts present.
4. Remove or zero the evidence connection while keeping weights fixed.
5. Remove/retract a required record or source and check the prescribed unknown behavior.

An acute intervention measures dependence of fixed weights. A separately trained memory-off control measures what optimization can achieve without the path. Report them separately. A large logit change that produces the wrong answer is not a useful factual-control success.

## Fairness and cost review

The candidate and trained memory-off control must receive matched semantic examples, initialization and training budget. Explain precisely which paths are zeroed and whether auxiliary writer/route losses still train both arms. Equal nominal parameters do not imply equal effective capacity, equal FLOPs or equal elapsed time.

Strict/hybrid comparisons must share the backbone, store, output interface, pending/recovery policy and question set. If hybrid weights have separate training, account for that training and selection. A new hybrid cannot inherit historical LRC retrieval scores.

Time event interpretation, writes, candidate discovery, scoring, evidence validation/rebuild, generated-token inference, sampling, durable I/O and recovery separately and in total. Record one-thread setting, hardware/runtime, cold/warm treatment, repetitions/order, concurrent workloads, failure outcomes and serialization bytes.

Report fixed neural-state bytes separately from active-record bytes, archive/history growth, parameters and peak process memory. Evidence reuse is a hypothesis: validating several records may cost as much as exact resolution. A reduction in archive searches alone does not establish lower end-to-end latency.

No architecture-efficiency conclusion follows from this diagnostic. Any inference-weighted comparison using `12 ln(I/I_baseline) + ln(T/T_baseline)` requires an explicit matched capability gate and comparable baselines first.

## Review verdict and evidence artifacts

The final independent review should identify inspected source/data/checkpoint hashes; commands and checks actually executed; every reproduced defect and its repaired status; remaining unknowns; and whether the results follow from raw records.

Allowed verdicts include engineering checks passed with learned qualification failed, learned in-distribution success with transfer failure, or incomplete due to a declared run cap. Do not turn missing evidence into PASS, conceal failed seeds, remove inconvenient families, or rename previously inspected test outcomes as prospective confirmation.
