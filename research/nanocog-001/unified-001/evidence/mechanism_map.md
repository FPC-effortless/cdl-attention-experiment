# COG-UNIFIED-001: evidence-to-mechanism map

Prepared 22 November 2026 (America/Los_Angeles). This document records how the inspected positive mechanisms can inform one proposed COG/CSM architecture. It does not report a trained fused model, prove transfer of historical results, or certify the new reference implementation. The source manifest pins 17 complete report/source reads. Historical experiments were not rerun.

## Decision

Use a shared causal representation, factorized identity/role/value interfaces, hard bounded publication and routing, an exact versioned record executor, separately supervised routing and writing, a readout trained against the same record representation, and demand-aware reuse. Treat historical checkpoints as separate evidence, not plug-compatible parts.

The strict primary candidate preserves the existing nanoCOG boundary: no token-to-token self-attention, QK softmax or dense KV attention, and no borrowed SSM. A learned dot-product reader is an attention-style mechanism even when it reads only a few records. The BIND studies therefore justify transferring objectives, factorization and diagnostic controls; their actual soft readers belong in a separately named hybrid comparator. The final primary graph and all learnable interfaces still require new matched training and qualification.

The architecture may implement multiple modes, but its reported native accuracy must include all parsing, identity induction, routing, writing, reading and decoding errors. Gold keys, operations, records, answers and reference plans remain oracle controls. Veritas is the experiment/evidence environment, not a hidden source of the correct semantic answer at inference.

## 1. Continuity and exact contracts

### Current implementation boundary

At the inspected nanoCOG reference, the token decoder has a causal local mixer and bounded publish/subscribe packets. Its topic and publish projections are learned, forward routes are hard, and training uses a dense straight-through surrogate. Reference execution has a Python time loop and functional state copies. A separately owned-state inference path does not establish fused accelerator performance. The model follows a pinned nanochat shell, but its tokenizer, initialization, optimizer grouping, batching and serving are not yet drop-in compatible.

The stable typed diagnostic lane is explicitly engineered. It uses the PULSE taught ontology: four box records, four cargo records, terminal owners and one possible link to a box. It does not discover open-world identity or ontology. Its exact code is a structural control, not evidence that the learned PULSE model already implements it. [N1, N2]

### Temporal and update conventions to preserve

The current state contract reads from state **before applying the current packet**. The current token may choose the address; it does not read a value it writes at that same step. Padding neither applies a packet nor advances time. A live invalid packet changes no record but advances the visible clock. Legal material changes increment the addressed record revision; an identical SET or clearing an already empty record does not increment its revision. Atomicity applies to all fields of **one packet**, not a distributed transaction.

Resolution supports one stable link, not arbitrary transitive graphs. An unresolved link produces an explicit invalid result rather than an arbitrary owner. The new unified proposal may extend this to bounded multi-link resolution, transactions or a growing archive, but those must be versioned extensions with new proofs and implementation checks. Existing results do not automatically cover them. [N2]

### Gradient boundary

Hard argmax writes and hard read-address selection stop answer-loss gradients to routing, operation and write heads. A downstream value embedding can learn while the decisions that placed the value remain untrained by that loss. Auxiliary raw logits expose a route for explicit supervision. A straight-through or other estimator is a separate learning design with its own mismatch and stability questions; the exact-state proof says nothing about estimator quality. [N2, N3]

### PULSE lineage

The PULSE-001 ledger records a source proposal for learned publication, subscriptions, bounded requests, brokerage and state updates, with 21 reported Lean structural theorems and 12 host/algebra checks. That milestone explicitly had no native neural execution, fit or accepted scientific capability result. The original private source/proofs were not recovered in this mapping subtask, so these counts are reported continuity, not a new proof audit. Publication starvation and cumulative rescoring/copy costs were already acknowledged. [P1]

## 2. Mechanisms selected for fusion

| Mechanism and evidence | Recorded positive | Strongest limitation or reversal | Placement in the unified proposal |
|---|---|---|---|
| Causal local processing plus bounded literal lookup, COG-LM-002 | Joint COG+lookup: 183.5 warm CPU decode tokens/s versus 157.3 for Mamba-2+lookup; live state 72.5 versus 779.1 KiB. | WikiText PPL 50.8989 versus 42.10; one training seed and 2,097,152 target presentations. Bank adds cost without meaningful correction benefit. This is a separate ~31M lineage, not current nanoCOG. | Use causal local computation as the ordinary token path; retain literal lookup only as an explicit optional channel. Do not merge historical weights or forecast the same speedup. |
| Active-frame restriction, COG-ATTN-049 | All arms correct on 48 bindings, three seeds and five lengths. At 32,768 tokens weighted full/scoped baseline cost divided by local cost: 4.24/1.71. | Weighted total includes fit+setup+12×online. Online versus scoped is 1.28 on loopback and 1.05 with 2 ms request charge. Typed small-model CPU task with visible frames. | Make relevance/active scope explicit and account for its discovery cost. Compare against a scope-matched attention control. |
| Support-selection supervision, BIND-001A | Native strict main-held completion 2,038/4,608→3,819/4,608; all 18 matched cells improve. Binding mode 315/1,536→1,117/1,536. | No fit qualifies; synthetic one-hop, known vocabulary, explicit boundaries. Existing reader uses dot-product inference. | Add matched training-only support/routing loss on a causal candidate set, alongside answer loss. No gold support at normal inference. Transfer to hard routing remains a new hypothesis. |
| Factorized identity/role scoring, BIND-001B | Familiar strict source-and-value completion 308→348/384, positive in all three seeds. | Shifted syntax 77→74/384; no fit qualifies; coordinate blocks were not shown to discover semantic roles. | Separate identity, relation and value interfaces, but train their alignment and include pooled/unfactorized controls. Do not assume factorization alone induces robust binding. |
| Readout alignment, BIND-001D | Correct-source taught-shift recovery 455→1,214/1,536; native joint 264→691/1,536. Matched familiar-only refit reaches 458 and 266. Correct-source gain occurs in six checkpoints. | Untaught templates correct-source 46→58/1,536; familiar repair loses 34 families versus familiar-only refit. Fresh groups reuse 144 exposed individual keys. | Train the output reader on the exact representation produced by the writer, including matched paraphrase/rewrite supervision and preservation examples. Use exact reads to test selection separately from decoding. |
| Stable two-hop retrieval, recovered LRC interface | 768/768 exact answers and 768/768 paths on fresh bridge fixtures after code/checkpoint identity recovery. | Defined typed interface. The original CSM reader in the same integration routes only 253/768 correctly. No full-stack or language guarantee. | Bounded multi-step record traversal with separate route/path/value outcomes, and explicit unresolved outcomes. Keep literal identity, learned alias binding and physical storage position separate. |
| Factorized action representations, COG-ATTN-053 | EPIC narration accuracy 8,813/9,668=91.16% versus 6,937=71.75%; novel combinations 467/742 versus zero. | Supplied action labels and strong lexical dependence; catalog-word masking collapses performance. Joint precision/recall gate fails. | Share compositional entity/role/action factors where the task supports them; preserve raw-text and strong lexical controls. This is a representation prior, not a natural-language theorem. |
| Exact persistent/versioned state, COG-ATTN-057 and current typed contract | Exact store retains all 5,760 panel transitions and all 800 supplied mappings per seed through the 100th labelled task; typed executor has explicit locality and rejection semantics. | Tasks provide context IDs and demonstrations; silent changes cause 433 wrong old plans until observation/reprobe. No global neural no-forgetting claim. | Protect exact committed records and their versions; distinguish history from current projection. Changes require observation and accepted new events. Raw bytes surviving is weaker than reliable future answer access. |
| Sufficient-state indexing, COG-ATTN-059 | 23,040/23,040 online answers agree; largest-size ratio 3.18 versus attention rescan. | Strongest cache-aware exact control reduces gain to 1.12 and uses the same payload bytes; 2× gate fails. | Maintain task-valid sufficient statistics when an exact contract proves them sufficient. Charge index build/update and compare with a cache-aware control. |
| Demand-aware reuse, COG-ATTN-055 | Five fresh seeds: 15,360/15,360 paths, median 3.29× hot-demand and 1.14× uniform speed, including construction. | Initial churn-prone policy slows uniform demand to 0.40×. Fixed 32-state graph and free observed epochs. | Compile/cache only after a measured or declared break-even criterion; include dependency/version invalidation and bounded admission/eviction cost. |
| Learned decision ordering, COG-ATTN-093b/c | 093c solves 32/32 with 38,950→18,699 expansions and 213.840→110.241 ms batch CPU; shuffled control fails to recover benefit. | A*, legality and admissible heuristic supplied; 238.894 ms acquisition/setup bill; chess counterpart is much slower. | Optional procedure/action ordering head in a domain adapter; do not place A*, Sudoku rules or a game solver in the universal language-model core. |

Sources: LM2; T1; B1–B3; G1; S1; N2 in the reference list.

### Literal lookup is its own mechanism

The 31M LM's short-delay success was specifically tied to an observed-successor route. In COG-LM-003, suppressing that contribution changed the original 256-token literal result from 14/16 to 0/16. It is not evidence that the recurrent state or revision bank learned semantic retention over that span. The unified design must expose this channel and include a separately trained no-lookup baseline and a cost-matched inference intervention. [LM3]

### A larger memory signal is not a content guarantee

Continuous computation restored a gradient path to an earlier fact, and utilization pressure raised the median natural-text read/hidden ratio from approximately 0.011% to 7.12%. All ten LM-003 fits nevertheless scored 0/16 reworded answers at every tested delay. These findings justify testing gradient connectivity and semantic content separately. They do not justify a requirement to maximize bank amplitude, which could harm language loss without teaching binding. [LM3]

## 3. Shared interfaces instead of checkpoint concatenation

The historical systems have different vocabularies, feature spaces, ontologies, input privileges, training objectives and output heads. Numerical shape compatibility is insufficient for semantic compatibility.

A unified implementation should declare one interface for each boundary:

1. **Causal observation:** shared tokenizer and prefix-only features; event boundary is learned or explicitly supplied only in a labelled scaffolded arm.
2. **Binding proposal:** predicted entity/relation identity, operation, payload and supporting observation span, with raw logits for training. Supplied gold keys never enter ordinary prediction.
3. **Publication and lookup:** bounded packet routing with exact ownership/session semantics. A packet topic is not assumed to equal a semantic identity.
4. **State transition:** exact typed operations, material-change/version rules, rejection reasons and cost. Live links and snapshots have different semantics.
5. **Read result:** payload plus resolved identity/version/dependencies, or an explicit failure/unknown. Avoid losing identity when converting back to a vector.
6. **Output:** learned language head and, where supported, a constrained exact factual renderer. The unrestricted language head is not made truthful merely by attaching a correct memory.
7. **Selective adaptation:** proposal, bounded update and measured retention checks. Parameter locality does not imply output locality.
8. **Reuse:** dependency certificate, semantic/model version, construction/update/eviction cost, invalidation and recomputation behavior.

Train these interfaces together, retaining matched controls for each added signal. The inference graph must be audited for actual attention-like operands and hidden privileged fields. A separate hybrid may combine a bounded learned content reader with the exact state interface, but its results must not be attributed to the strict attention-free arm.

## 4. Mechanisms that must remain optional or experimental

- **Current-event bypass:** PULSE-006 improved addressing from 61.1% to 69.6% and joint address/commit/value from 38.8% to 43.4%, but strict pairs were only 8/2,400 versus 6/2,400, with a paired interval crossing zero. Preserve it as an explicit factorial design factor, not a demonstrated complete fix.
- **Stable keys alone:** the frozen-reader follow-up recovered 0/21 EVENT and 0/5 CONTEXT convention-correct pairs when annotated positions were supplied. Stable identity resolves an address ambiguity; it does not certify content encoding or final decoding.
- **Persistent neural weight editing:** finite-suite retention and bounded parameter movement cannot establish universal no-forgetting. Retain separate protected-domain assumptions and downstream behavior tests.
- **Large state and larger models:** the failed learned-bank results do not justify scale-up before demonstrating content-sensitive use. Capacity may still matter; its adequacy has to be established by a matched experiment.
- **Domain solvers:** task-specific supplied planning machinery remains an adapter/control. The learned reusable principle is selective ordering or reuse, not proof that the model learned the solver.

The public results index records the first two limits and the LM-004 stop decision. [I1]

## 5. What composes, and what does not

Structural invariants can compose if every operator preserves the same stated invariant. Examples include read-before-write causality, rejection preserving record contents, a protected raw record remaining unchanged, and a budget cap on issued requests. These are candidate proof obligations.

Behavioral gains do not compose by arithmetic. The 16.7% CPU decode gain, 38.7 percentage-point binding improvement, 1.94× search speedup and typed 4.24× weighted-cost ratio belong to different systems and workloads. Their product is not a prediction of unified speed or quality. A selector repair can change the reader's distribution; a new writer can invalidate a decoder alignment; a stronger cache can make indexing savings disappear; a hard exact transition can remove useful gradients.

For every transferred mechanism, require:

- its isolated matched control;
- removal and replacement interventions on the fused candidate;
- retraining of the relevant ablation;
- complete native task outcomes, including unchanged-third and counterfactual direction checks;
- inference and training cost under shared visibility;
- preservation of all failed seeds and conditions.

For semantic preservation, live LINK dependencies matter: changing a different raw slot can intentionally change a linked answer. Cache invalidation and parallel conflict checks must use the transitive read dependency set, not only the first addressed slot. For learned adapters, preserving a record's bytes does not preserve the prediction if routing, embeddings or the decoder change.

## 6. PR90 comparison obligations to carry forward

The current comparison proposal distinguishes:

| Track | Fixed or bounded | Permitted conclusion |
|---|---|---|
| Common-backbone mechanism | Same tokenizer, permitted examples, order/exposure, loss visibility and compatible shell | Effect of the replaced mechanism in that training regime |
| Training-budget frontier | Data-access policy, hardware class, equal tuning opportunity and explicit budget | Quality obtainable for that resource budget |
| Quality-constrained serving | Required quality, workload, hardware, lengths, concurrency and cache opportunity | Latency/throughput/state benefit at acceptable quality |

Core proposed learned references are GQA, MLA and explicitly named 3:1 KDA+GQA, plus the separately trained local-only candidate. GDN/GDN2 hybrids are challengers; CSA2 admission is conditional. Backend variants of FlashAttention are not independently trained architecture families. Exact inherited PULSE and deterministic/lexical references remain diagnostic controls where eligible. [C1]

The proposal requires:

- Shared tokenizer/untied embedding-output accounting and separate total/nonembedding/active/state counts.
- No dead-parameter padding or concealed extra supervision.
- TRAIN/CAL adequacy before attributing a comparator failure to architecture.
- Explicit gradient horizon, state reset/detach schedule and delayed-target exposure.
- Separate fresh confirmatory family/wording/entity/composition splits, with no reuse of exposed COG panels as fresh evidence.
- Full cost including routing/index construction and updates, copies, resets, forks, eviction, rejection, recovery, output, tokenization, training failures and tuning.
- Critical-stratum competence plus quality noninferiority and a useful efficiency bound.
- Crossed seed/family statistics and adequate power, not treating repeated families as independent subjects.
- Separate CPU-reference and optimized accelerator conclusions.
- A second meaningful scale before a broad replacement claim.

The proposed 1-percentage-point task margin, 0.02-nat NLL margin and 1.20× useful latency ratio are **unaccepted planning proposals**, not mathematical guarantees or permission to lower stricter task gates. Neither 125M parameters nor five seeds establishes adequacy by itself. CAMPAIGN.json is a planning manifest, not an executable admitted run. [C1, C2]

## 7. Index discipline and coverage

The exact active public source index path is:

`research/nanocog-001/RESULTS_INDEX.md`

The inspected PR90 copy appends a prospective comparison entry without rewriting historical outcomes. It is explicitly a navigable subset, not a complete historical audit. A unified architecture entry should be labelled **PROPOSED ARCHITECTURE / FORMAL ABSTRACT CORE / NO TRAINED FUSION RESULT**, with separate statuses for source specification, abstract proofs, reference implementation checks, neural integration, trained behavior and comparative cost.

PULSE's broader lane index is reported as `provenance/records/CURRENT_LANE_INDEX.json`, SHA-256 `abe0ec5159b286d7b4137abb138c5621669adf00b62a3d1ded4efb71c51b6742`, inside its preserved private evidence. This subtask did not recover that private file or reconcile all 18 historical lane packages. A new index must not claim that it did. [P1]

Earlier CSM/BDH results in conversation history are useful background but were not freshly recovered here as standalone original evidence. Protected-state and update semantics are carried forward through the directly read typed-state contract and reported exact-store results. This mapping does not promote earlier NLL, cache speedups or bounded-memory comparisons into fresh facts, and does not assert that every historical archive is available.

## References

- **N1 — nanoCOG implementation boundary:** [README](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1a91168b216a49543cc839153fa299a262320730/research/nanocog-001/README.md).
- **N2 — exact typed-state contract:** [STATE_CONTRACT.md](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1a91168b216a49543cc839153fa299a262320730/research/nanocog-001/nanocog/state/STATE_CONTRACT.md).
- **N3 — inherited qualification requirements:** [QUALIFICATION.md](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1a91168b216a49543cc839153fa299a262320730/research/nanocog-001/QUALIFICATION.md).
- **T1 — typed cost and recovered LRC:** [COG-ATTN-047–050](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1a91168b216a49543cc839153fa299a262320730/research/cog-attn-handoff/continuation/cog-047-050/README.md).
- **LM2 — language and CPU cost:** [COG-LM-002](https://github.com/FPC-effortless/cdl-attention-experiment/blob/45aad6d8d3540215794e8ce5edd43a5b4ab8ee3d/research/cog_lm_002/REPORT.md).
- **LM3 — gradient/utilization and literal attribution:** [COG-LM-003](https://github.com/FPC-effortless/cdl-attention-experiment/blob/186d577e7783bd6b8daa13b8d0028c0b174bf849/research/cog_lm_003/REPORT.md).
- **B1 — support supervision:** [COG-BIND-001A](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1a91168b216a49543cc839153fa299a262320730/research/nanocog-001/cog-bind-001a/RESULT.md).
- **B2 — factorized routing:** [COG-BIND-001B](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1a91168b216a49543cc839153fa299a262320730/research/nanocog-001/cog-bind-001b/RESULT.md).
- **B3 — reader repair:** [COG-BIND-001D](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1a91168b216a49543cc839153fa299a262320730/research/nanocog-001/cog-bind-001d/RESULT.md).
- **G1 — factorization, persistence, reuse and sufficient state:** [COG-ATTN-051–060](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1a91168b216a49543cc839153fa299a262320730/research/cog-attn-handoff/continuation/cog-051-060/README.md).
- **S1 — learned search ordering:** [COG-ATTN-092–095](https://github.com/FPC-effortless/cdl-attention-experiment/blob/a4dfa94b28d8e8afa05a6a50570d584567c7a5c1/research/cog-081-086/extreme/followup_092_095/REPORT.md).
- **I1 — public results index and preserved negative outcomes:** [RESULTS_INDEX.md](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1a91168b216a49543cc839153fa299a262320730/research/nanocog-001/RESULTS_INDEX.md).
- **C1 — modern comparison protocol:** [PROTOCOL.md](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1a91168b216a49543cc839153fa299a262320730/research/nanocog-001/attention-comparison-001/PROTOCOL.md).
- **C2 — actual comparison readiness:** [READINESS.md](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1a91168b216a49543cc839153fa299a262320730/research/nanocog-001/attention-comparison-001/READINESS.md).
- **P1 — reported PULSE structural milestone:** [Veritas #482 comment 7071453328](https://github.com/FPC-effortless/veritas/issues/482#issuecomment-7071453328).

The adjacent source_manifest.json records exact commit and Git blob identities for the retrieved file sources, including the governing AGENTS and universal workflow instructions.

