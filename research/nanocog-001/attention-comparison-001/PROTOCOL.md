# COG-ATTENTION-COMPARE-001 — fair comparison of the complete attention replacement

**State: PROPOSED; NOT ADMITTED FOR MODEL EXECUTION.** Prepared 22 November 2026 (America/Los_Angeles). This is a prospective comparison specification, not a result, completed preregistration, resource allocation, or permission to open a protected evaluation set. It extends [nanoCOG qualification](../QUALIFICATION.md); it does not replace or lower existing gates.

## 1. Decision and falsifier

Question: can the complete nanoCOG system preserve required language, relational-update and long-context quality while delivering lower measured inference cost than competent learned attention alternatives?

A useful result must distinguish: (a) better quality at comparable resources, (b) lower cost at acceptably preserved quality, and (c) an unresolved tradeoff. A smaller cache, a passing oracle, fewer logical reads, or faster incorrect answers is insufficient. The central falsifier is failure on a required quality stratum, failure to establish noninferiority, or failure to demonstrate a useful complete-session cost improvement with uncertainty accounted for. An invalid or undertrained baseline makes the comparison inconclusive.

The primary candidate is the nanoCOG packet-based decoder, with exact checkpoint/configuration still to be selected prospectively. Its local-only ablation is trained independently. An engineered typed parser/store is an explicitly scaffolded arm or ceiling, not native raw-text competence. The historical approximately 31M COG-LM lineage and PULSE checkpoints remain distinct studies.

## 2. Source identity and starting state

Read-only reconstruction used CDL draft PR88 at `8374f72ba9324fa4a11d0d3ecb40a6db0e311c73`. That branch is being updated by another session. The immutable source is a reviewed starting reference; it is not a frozen experiment candidate. The upstream nanochat reference is `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.

The candidate README says tokenizer loading, optimizer grouping, initialization, packing and serving are not yet drop-in equivalents to upstream. The packet reference uses dense surrogate routing during training and a Python temporal loop; functional decoding copies its state. A qualified owned-state inference path is available in source. These implementation facts must be resolved or measured explicitly before efficiency claims.

The latest inspected local study has nine 37,414-parameter fits, four TRAIN-adequate fits, and no arm passing every held local-binding gate. It supplies useful diagnostic evidence, not a modern-LLM architecture ranking. Existing exposed cohorts remain development evidence for subsequent changes.

No trained general-language nanoCOG checkpoint, matched modern baseline suite, exact data/tokenizer freeze, or accelerator allocation was verified in this session. Current Python has no installed Torch and no visible NVIDIA runtime. No model fits, native model imports, checkpoint loads, protected evaluation, or comparative timing occurred for this proposal.

## 3. Three comparisons, with separate conclusions

| Track | Keep fixed or prospectively bounded | Allow and report | Supported claim |
|---|---|---|---|
| A: common-backbone mechanism | Tokenizer bytes, licensed training examples, per-seed sample order, nonpadding token and target exposure, loss visibility, document boundaries, backbone interfaces and comparable capacity | Required mixer-specific gates/positions/adapters; actual FLOPs and time | Effect of the proposed mixer in this training regime |
| B: training-budget frontier | Data access policy, hardware class, equal tuning opportunity and one declared training budget | Architecture-specific optimizer settings, width/depth and tokens within budget | Quality attainable for the declared resource budget |
| C: quality-constrained serving | Quality gates, workload, input/output lengths, hardware, concurrency, cache opportunities and precision policy | Each model's correct optimized kernel and serving path | Latency, throughput or memory advantage at acceptable quality |

Equal parameters, tokens, FLOPs and wall time cannot generally be enforced simultaneously. Track B must state whether the binding budget is measured training work or hardware time; these are different experiments. Do not quietly change the budget definition.

For Track A, preserve the common shell wherever compatible and report both total unique parameters and nonembedding parameters. The existing target is at most 1% distance where feasible. If maintaining the same shell exceeds this, retain the fixed-shell contrast and add a separately specified meaningful capacity-matched contrast. No unused-parameter padding and no silent FFN resizing. Count all routing heads, latent projections, state controllers and learned readouts. State bytes and per-token active computation are separate quantities from trainable parameter count.

Track C compares complete deployed systems after their quality gates. Reference CPU and optimized accelerator panels are separate. A Python recurrence versus a fused GPU kernel can characterize those implementations, but cannot isolate an inherent architectural speed advantage.

## 4. Baselines and fair admission

The core proposed roster is:
1. NanoCOG packet candidate.
2. A separately trained local-only candidate backbone.
3. A causal GQA Transformer using the qualified device-appropriate backend.
4. An MLA Transformer with differentiable training and verified compressed-cache inference.
5. A KDA hybrid with a declared 3:1 KDA:GQA ratio in the common shell.

The fifth arm is explicitly **KDA+GQA**, not the released Kimi Linear model. A 3:1 KDA:MLA comparison is a separately named variant. Gated DeltaNet+GQA and Gated DeltaNet-2+GQA are planned challengers subject to the same tuning, parity and resource admission. Freeze the final primary family before confirmatory exposure. Unrun baselines stay UNRUN and cannot be described as beaten.

Preserve the inherited unchanged PULSE-006 direct-event and matched-broker references wherever exact source, weights, recipes and comparable authorized relational inputs are recoverable. They are relational diagnostic controls, not general-language competitors. Frozen replay and retraining remain separate. An unavailable required control must be recorded as UNAVAILABLE with a narrowed claim; do not silently substitute it. Text-facing deterministic parser/store and lexical/recency controls receive the same actor-visible information and incur their complete costs.

FlashAttention-2/3/4 and SDPA backend choices are implementations of an attention arm, not separately trained architecture families. Record the backend that actually ran. GQA admission includes head grouping, dropout disabled in evaluation, position-aware cached masking and full/chunk/step parity.

MLA must preserve latent and positional caches. An expanded-cache-only implementation is not sufficient for its memory-efficiency comparison. Hybrid recurrent models must count convolution buffers and global-attention KV state as well as recurrent matrices.

CSA2 is conditional. The reviewed NeMo implementation supports differentiable full-sequence attention but freezes released indexer weights and does not support cached decoding. A from-scratch comparator needs a justified selector-training procedure, Full/Reindex/Reuse semantics, hierarchical indexing, local windows, compression and a verified decoding path. Do not use a random untrained selector as a straw baseline. A reduced mechanism without these properties is CSA2-inspired and cannot support a claim about beating DeepSeek-V4.1. Released large checkpoints may appear only in a separate practical-system reference panel with their pretraining advantage disclosed.

See BASELINE_ADMISSION.json for sources and mechanism-specific requirements.

## 5. Training adequacy and scale

The first substantive study should use the largest feasible, well-trained common scale, chosen after integration and measured resource profiling. **125M total parameters is an illustrative planning point, not an imposed limit or proof of adequacy.** The actual size must accommodate the identical tokenizer and untied embedding/output head. A second meaningful scale is required before a broad attention-replacement claim. No final token budget is asserted here.

Use training and development learning curves and TRAIN/CAL competence to establish that each comparator can learn the tested tasks. A fixed token-to-parameter ratio is not a guarantee. Predeclare training milestones, minimum competence, divergence criteria and an extension rule using development data; lock the resulting confirmatory allocation before final evaluation. If a baseline is still learning materially or cannot solve calibration, repair or extend under the symmetric budget policy before attributing its failure to architecture.

For long-context claims, training must expose relevant dependency lengths, corrections and multi-fact reasoning. Record forward context length, gradient horizon, state detach/reset schedule, auxiliary supervision and effective delayed-target counts. Carrying a state across chunks does not by itself show gradients trained long-range writes. Evaluate within trained support and length extrapolation separately.

Allocate equal total tuning opportunity, using an explicitly measured resource budget and development access policy. Equal trial counts are not sufficient if trial costs differ. Each architecture may use a competent optimizer and search space. Count failed trials, architecture tuning, auxiliary losses and distillation. Freeze selected configurations before confirmatory prediction; retain the complete trial roster.

Planning starts with five independent training seeds per selected configuration, then uses a separate variance/power pilot to determine final seeds and independent families. Five seeds is not a guarantee of adequate power. Pair data-order seeds, not fictitious identical initial weights across different shapes.

## 6. Evaluation panels

### A. Natural language and code

Use exact licensed TRAIN/DEV/evaluation manifests with document/repository grouping, near-duplicate and contamination checks. The shared tokenizer is fitted only on permitted training data. Freeze loss masking, EOS, packing, truncation and reset semantics.

Primary language metric: token-weighted held-out NLL; derive perplexity from the same aggregate. Report domain, source, context-length and position breakdowns. Keep padding out of target counts. Preserve document clusters for uncertainty. Continuous-state and reset-state variants must have identical information opportunities.

### B. Changing relational information

Use raw token histories for every native arm. Include assignment, overwrite, transfer, delete, absent queries, nontrivial untouched records, multi-hop chains, delayed corrections and conflicting statements with explicit temporal semantics.

The primary relational outcome requires both counterfactual histories correct plus preservation of an untouched third relationship. Also report full-state correctness, stale-answer rate, collateral changes, both query roles, abstention and complete denominators. Hold out entity/name groups, paraphrase families and compositions. Do not reuse previously exposed COG cohorts as fresh confirmation.

Oracle retrieval, oracle binding, correct-value injection and exact readout remain diagnostic arms. They identify bottlenecks; they do not enter native capability rankings. Any supplied IDs, parsers, grammar annotations or additional supervision must be equally available to comparators or separately labeled.

### C. Long-context and complete-task use

Include multiple facts distributed across positions, interacting revisions, dense distractors, exact identity binding, repository dependencies and long-document QA. A single needle test is insufficient. If an acting-system claim is made, use the same tool environment, policy, tool budget, verifier, output limits and retry policy for all arms, and score actual task completion.

Proposed profile lengths are 1K, 4K, 16K, 64K and 128K tokens, only where model capacity, training and the verified runner support them. These are workload design points, not advertised candidate capabilities. 1M-token evaluation is deferred until admitted. Vary information density as well as filler length; fixed memory must be tested under rising numbers of independently recoverable facts and adversarial collisions.

## 7. Complete costs and state

For each workload record TTFT, prefill time, inter-token latency, completed-request p50/p95, output throughput, throughput at a fixed latency target, success within budget, peak allocated/reserved device memory, host RSS and retained state per live session.

Time routing, index construction/updates, compression, reads, writes, rejection work, collisions/eviction, output projection, tokenization, state reset/fork, cache resume and recovery. No free prebuilt index or omitted adaptation pass. Count external stores, previous embeddings, positional metadata, convolution buffers, quantization scales, copies and allocated unused capacity. Report logical state separately from actual allocations.

Use identical hardware/precision policies, documented CPU threads or GPU clocks/power, randomized paired order, warmup and synchronized device measurements. Report cold compile/load/setup separately from warmed request service. Timing loops are measurement repeats, not independent trained seeds. Never compare a CPU time with a literature GPU number as a speedup.

Fixed-output-token microbenchmarks isolate execution; actual completion benchmarks also count extra output tokens, retries and failures. For continuous learning, charge updates and verify retention under equal permitted feedback.

For a stated deployment volume Q, report:
C_request = C_training/Q + C_prefill + C_generation + C_index_update + C_recovery.
All terms must use the same units. Evaluate several deployment volumes. Apply the user's inference-first priority after quality qualification; the 12:1 preference does not justify adding raw seconds to NLL or ignoring training/tuning cost.

## 8. Statistical decision and failure treatment

Use the same held families for each seed and model; keep all dependent counterfactuals, variants and source documents together. Training seeds and evaluation families are crossed. Define the primary endpoint family, comparator roster, direction, weighting and multiplicity correction prospectively.

Proposed adoption margins, pending operational justification and independent freeze:
- Higher-is-better complete-task success: noninferiority margin 1 percentage point.
- Lower-is-better token NLL: margin 0.02 nats/token (about 2.02% perplexity).
- Complete-session latency: useful speedup threshold 1.20x.

These are starting proposals, not accepted tolerances and not permission to lower any stricter existing task gate. Each critical stratum must meet its absolute competence/collateral floor. Require simultaneous quality noninferiority and a confidence bound exceeding the useful efficiency threshold. Nonsignificant quality differences do not show equivalence. Better-quality claims require a separate superiority test.

Use a separate pilot to simulate power/coverage for seed variance, family variance, paired discordance and runtime blocks. A suggested target is 90% power for the smallest useful effect, but no power is claimed until estimated. Preserve the full primary multiplicity family across endpoints, baselines and declared operating points; any reduced roster must be chosen before evaluation.

Reuse Veritas's paired crossed-comparison design where its estimator and bounds fit. Its current equal-cell estimator does not automatically implement token-weighted NLL or every cost-ratio estimand. Its operation limits also constrain large corrected bootstrap families. Adequate weighted/ratio analysis and interval precision must be validated before confirmatory admission; do not downsample observations or weaken correction to satisfy a software cap. See STATISTICAL_PLAN.md.

Failures stay in the roster. Evaluation crashes, timeouts and invalid actions fail success-within-budget; completion latency is censored or infeasible, not zero. Undefined NLL remains missing and can block inference. Training divergence consumes resources and cannot be replaced by a favorable seed. Document external-fault same-seed reruns under a symmetric predeclared rule.

## 9. Execution stages and deliverables

1. **Source/integration admission:** bind actual candidate, baseline source revisions, tokenizer, adapters, ownership, training paths and complete state. Pass causality, full/chunk/step parity, gradient connectivity, resets/forks and cross-document isolation.
2. **TRAIN/CAL adequacy and profiling:** use a separately reviewed resource envelope and data-only freeze. Establish competence and actual cost; do not draw the main architecture conclusion.
3. **Confirmatory freeze:** finish all source/data/config hashes, tuning roster, training envelope, seed/family counts, endpoint margins, power analysis, scoring and private/public boundaries. Independent reviewer must bind exact versions.
4. **Matched training:** retain all seeds, checkpoints, budgets, learning curves and attempts. No prediction/scoring worker may alter a fit.
5. **Blind prediction and separate scoring:** close model hashes, predict once under the declared plan, score with evaluator-only truth and independently reproduce primary aggregates.
6. **Optimized serving and scale confirmation:** qualify parity of optimized paths, measure complete costs at admitted quality, and reproduce the central finding at a second meaningful scale before a general claim.

The immediate next engineering unit is the common nanochat-compatible tokenizer/data/optimizer and explicit state adapter, with the unchanged attention baseline retained. CPU correctness is useful here; it is not a substitute for the eventual accelerator study.

## 10. Status and index discipline

CAMPAIGN.json is a planning manifest, **not a native TrainLab dispatch manifest**. It deliberately retains nulls for unresolved identities and allocations. No job should be launched from it. READINESS.md records actual findings and blockers. This proposal adds a prospective entry to the existing results index; it does not claim reconciliation of every historical result.

The user has requested the comparison. Ordinary source inspection, design, bounded software checks and review may proceed. Protected data boundaries, actual paid-resource allocation and scientific promotion remain governed by their specific contracts. A report or merged protocol does not establish model success or grant resource capacity.

Every later job must retain exact source/recipe/data/tokenizer/environment identity, checkpoint and raw-output hashes, actual process/job receipt, measured resources, failed attempts, independently recomputable scoring, reviewer identity and claim scope. Public artifacts contain safe aggregates and references; private labels/checkpoints remain within their authorized boundary.

