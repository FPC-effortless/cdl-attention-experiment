# COG-ATTN-002–010 — bounded attention-replacement experiment cycle

Date: 2026-11-20. Scientific status: **NOT_QUALIFIED for general-purpose LLM attention replacement.** This package contains local research code, model checkpoints, per-seed aggregate outcomes, timing observations, the replay audit, and a reconstruction recipe for the authentic-language experiment. No repository was modified or publication performed.

## Executive findings
- Same-meaning exact retrieval can exchange full-history softmax for bounded indexed memory without altering the measured outputs: 3,840 per-example reader-substitution checks on the learned byte parser, plus aggregate parity across 128 seed/regime bridge runs.
- Interface-level integration of source-derived TAC ID001, CDL CSM skill-machine procedures, LRC bounded index, and Veritas consequence-trained verifier is feasible, but system accuracy is low under stale or corrupted writes. It does not load all original trained models.
- At 9,104 trainable parameters the attention-free parser plus engineered executor reaches 99.87% on a familiar grammar but 4.04% on a held-out grammar. The 9,038-parameter full causal byte Transformer remains weak on direct task answers. The comparator improves substantially when coupled to the same executor. Neither qualifies as an LLM.
- Verification alone is insufficient: independent confirmation can raise precision while preserving stale values. Independent evidence must be allowed to generate a corrected proposal, reducing shared-corruption errors.
- Individually observed primitive transitions can support planning without enumerating a library of complete programs. Explicit state-space planning is not neural latent reasoning.
- A last-write timestamp reduction preserves all measured final memory states; sparse association scales only if feature discrimination and posting load scale with entity count.
- Authentic Banking77 text can route to 77 simulated actions with 82.63% accuracy using reward-driven sparse associations. This is intent classification, not unrestricted grounded dialogue or a live financial workflow.

## Scope and evidence matrix

| Study | Executed measurement | Qualification |
|---|---|---|
| 002 | Source-derived routine bridge and equal-reader intervention, 64 procedural seeds × 2 sensor regimes | PARTIAL: not original trained CSM/LRC/BDH/Neuraxon checkpoints |
| 003 | Three learned sparse byte encoders, 768 sessions per cohort | Familiar grammar passes, held-out grammar fails; trained with auxiliary labels |
| 004 | Three evidence regimes, same-family/independent/adaptive/revision interventions, 64 seeds × 2 regimes | Positive on simulated independent-channel evidence; no physical costs |
| 005 | Single-transition learning, 8,192 goal tests | Positive bounded compositional skill transfer |
| 006 | Learned model + bounded BFS and greedy controls | Positive explicit planning; neural latent reasoning not implemented |
| 007 | 256 last-write size/seed parity checks, vectorized CPU timing | Positive for independent precomputed writes, not whole-network GPU speed |
| 008 | Three nearly parameter-matched full byte Transformers, smaller byte arm, shared-parser message attention and GRU | Underqualified baseline; no general LLM result |
| 009 | Official Banking77 split, reward-simulated action routing | Partial authentic language, no authentic outcomes/production workflow |
| 010 | Exact reader preparation/read timing, 21 sparse index fixtures, full-session CPU timing, held-out long histories | Conditional: feature capacity must grow; no optimized accelerator or equal-quality LLM comparison |

## 002: pinned source-derived integration

Source references (the local module contains relevant routines transcribed with formatting compacted, not byte-identical full upstream packages):
- TAC-Prime ID001 router/state: [routing.py](https://github.com/FPC-effortless/TAC-Prime/blob/e51a921f82575d57ecf048e82a3898ca968a4224/tacm/tacm/id001/routing.py), blob `ff4cb3947b6fc6bd5ea0a438ac246028d12a3ac8`; `state.py` blob `95ba39a058e9ce7c9314eeac20a8d02c05940f55` at the same revision. Router identity embeddings replaced by normalized additive synaptic prototypes learned from known identity observations. It still scans all identities and uses softmax over entity identities, not historical token attention.
- CDL CSM skill machine: [pilot.py](https://github.com/FPC-effortless/cdl-attention-experiment/blob/2a3dd21fcbd55e87eca3c4e5bb05846136b86d4f/research/csm-bdh002-handoff/skill001/pilot.py), blob `b0d4dd35495dcaa06837d5d3181c29cd77e09061`. COPY/CLEAR procedure induction, admission, binding, execution. Swap induced from four state transitions; fixed composition vocabulary supplied. No original neural state encoder loaded.
- LRC: [index.py](https://github.com/FPC-effortless/cdl-attention-experiment/blob/research/lrc-route-repair-002-20261108/experiments/lrc_route_repair/index.py), blob `9be6701d7d331e6849b2539db202de971b3e5c01` at branch lookup. Index returns all occurrences if bounded bucket fits, otherwise overflow miss. Address mapping in this bridge is explicit `entity % capacity`; content resolver checks exact identity labels rather than using original learned LRC keys.
- Veritas [verification.py](https://github.com/FPC-effortless/veritas/blob/86066226e39cdc9a436f58af300360af2d681af7/examples/csm_nx2/verification.py), blob `afbef9d3459a57f37524c59398e26ddd78c2ff45`; retrieved through PR 581. `Confidence` uses 1200 prediction-versus-future-observation training opportunities. Learned admission equals fixed two-of-three agreement in the numeric verifier test.
- `pilot001.py` is the previous pilot source; its synthetic observation/communication generation is reused. In bridge inference an oracle simulator returns specified identity bits, not the answer. Query hop count and structured cue arrive separately.

64 seeds (20000–20031, 30000–30031), 64 requests each, per sensor regime, 7 bridge arms; 57,344 task evaluations. 64 live records with four overwrite rounds; noise probability 0.12, additional shared corruption probability 0.12. CSM execution composes verified swap operations, then outputs an acquired discrete communication symbol. Data include explicit labels and event boundaries. Answer success can include accidental equivalent values; strict success also demands correct route and all traversed records. Native64/native32/full-history reader counts match for every seed; overloaded cap yields 100% abstention; unchecked can win because the admission gate retains old values. Results: `native_results.json`; timings not collected for whole native bridge.

| Panel | Independent answer | Shared-error answer |
|---|---:|---:|
| Primary | 67.09% | 53.52% |
| Replication | 62.30% | 47.17% |
| Pooled | 64.70% | 50.34% |

## 003/008: neural training and controls

Synthetic fixed-width ASCII byte messages (`a>b:5`, `a/b=5`, `a b 5`); a fourth grammar `5<b-a` is held out, as is reversed query syntax. Twelve entities, eight outcomes, 16 training-history writes (12 initial and four overwrites), chain lengths 1/2/4, 1,536 train sessions per seed for shared parser and matched byte model. Evaluation is 256 new sessions per seed/cohort, no gold test labels passed to models. All models have known environmental message boundaries. The modular encoder retains eight of 32 activations and learns record/query heads from auxiliary labels. Exact executor is hard-coded last-write consolidation plus pointer following. Output is an action/value ID; no generic language generation. Shared-parser neural controls train on the same encoder representation and auxiliary supervision available to the parser. The smaller byte model has a separate training stream sampled from the same generator. The matched 18d causal byte model uses slot embeddings plus sinusoidal positions and record, query and answer losses. It has 9,038 parameters; the modular learned parser has 9,104, excluding hard-coded state semantics. 240 parser steps with batch 48 vs 1,200 matched attention steps with batch 24; training compute is not matched. Full byte attention attends to all preceding padded byte positions, not only message endpoints.

| Cohort | Modular | Byte Transformer | Transformer + same exact executor | Oracle exact executor |
|---|---:|---:|---:|---:|
| IID | 99.87% | 17.32% | 87.37% | 100% |
| One hop | 100% | 18.88% | 95.31% | 100% |
| Four hops | 99.87% | 17.32% | 78.12% | 100% |
| Additional 64 overwrites | 99.74% | 13.41% | 14.06% | 100% |
| Unseen grammar | 4.04% | 13.54% | 4.82% | 100% |

Chance for eight balanced outcomes is approximately 12.5%. Direct neural baselines are weak and not converged as useful task solvers; no quality-matched LLM speed conclusion. The larger message Transformer (61,848 including parser) scores 16.15% IID; GRU (70,808 including parser) 16.67%. It is scientifically stronger to compare the trained byte Transformer plus the same executor (87.37%) than to claim direct model superiority from underqualified baselines. Three seeds per arm, 15 cohorts, two byte models plus message Transformer and GRU; checkpoint audit reproduced 210 aggregate checks. The same shared parser and exact state were used in 3,840 one-by-one reader swaps with the engineered full-history oracle-key softmax: every answer matched, including errors/abstentions. Distinguish this strong **functional exact retrieval equivalence** from general Transformer learning.

## 004: verification and recovery

`diagnostic_results.json`: fresh random truth at each epoch; shared-error condition corrupts 25% of observations in a common channel; one independent probe yields high write precision when required, but fewer admissions. `recovery_results.json`: 32 persistent records, a revision shock at epoch four, 25% of keys in shared sensor fault group, 4 maximum per-event observations, independent readings reliable with probability 0.90 and proposal 0.68; same synthetic observations available to all arms (different consumed channels). Simulator explicitly supplies trusted revisions to version-aware arm. Policies see the outcomes, not gold truth. Pooled 16,384 online observations per regime. Independent-only uses three independent readings and majority; adaptive consumes 3 or 4 depending on agreement. No real independent source has been demonstrated.

| Shared-corruption recovery policy | Mean reads | Online exact accuracy | First post-shift epoch accuracy |
|---|---:|---:|---:|
| Same family | 4.00 | 65.88% | 51.27% |
| One independent confirmation | 4.00 | 63.37% | 46.83% |
| Adaptive independent proposal | 3.59 | 95.23% | 85.31% |
| Independent-only proposal | 3.00 | 99.05% | 97.75% |

Revision invalidation improves epistemic honesty by refusing explicitly old-version records, not necessarily accuracy when abstentions count incorrect. Observation independence and cost are assumptions; shared-world sensors may not be independent. Strong recovery justifies a new integration experiment on the entire architecture; not validated integrated repair here.

## 005/006: primitive consequence learning and explicit planning

Random action meanings in 6×6 toroidal state space, 4 actions. 115/144 distinct state-action transitions observed with supplied state resets and identity. No complete goal/execution trace supplied in transition learning; planner constructs new sequences via finite graph search. 8,192 goal trials over 64 seeds; test start/goal pairs can coincide with one-step observed transitions, so not all are compositional. Random policy consumes up to 12 actions, greedy up to 12; BFS has explicit model checks; comparison has different reasoning budgets and not a learned latent representation. `diagnostic_results.json` contains counts and actions. `long_n` counts only nonabstaining nontrivial paths in that script, so it must not be used as the denominator for all long-goal performance.

| Policy | Success | Model checks/request (mean) |
|---|---:|---:|
| Random | 24.06% | 0 |
| Greedy | 78.63% | 15.28 |
| BFS cap 32 | 25.88% | 28.58 |
| BFS cap 128 | 91.59% | 68.48 |
| BFS cap 1000 | 99.58% | 69.06 |
| Oracle transitions | 100% | 69.36 |

The finite planner has no external token attention. It is explicit search, not latent cognition or proof of arbitrary skill acquisition.

## 007: associative last-write schedule

For each record, reduce all write timestamps by max and gather the corresponding value. 64 seeds × 4 sizes = 256 parity checks, plus exact equivalence of two chunk reductions. A serial Python loop and vectorized NumPy `maximum.at` perform the same final-state operation. Input generation is excluded; prepared keys, values are included in reduction. One BLAS/OMP thread; no GPU. Timing medians across runs: 256 events 35.72/6.91 µs; 4,096 542.30/12.26 µs; 16,384 2278.51/33.39 µs; 65,536 8887.49/126.18 µs. 5.17× to 70.44× speedup relative to Python, not native serial C or full attention. Dependence of writes on previous memory state is not covered.

## 009: authentic-language intent selection via binary consequence feedback

Data: [BANKING77 official train/test split](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data), PolyAI (Casanueva et al.), [CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE). 10,003 training / 3,080 test utterances, 77 intents; train blob `98e2543cf482d0dca7bfb175ebe35d98efad95be`, test blob `799687a8367359432985b8b13d85a2baf73f92dd`, license blob `2f244ac814036ecd9ba9f69782e89ce6b1dca9eb`. Source was fetched in the connected GitHub runtime. Data not bundled. `banking77.mjs` reproduces the JS computation after exact blob verification; source reproduction was not locally rerun because network unavailable. No human-user interaction occurred. Simulator labels identify rewarding opaque action IDs; learner tries actions in random cyclic order until success. Learner counts sparse features for the successful action; word/bi-gram and optional character 3/4-grams; TF-IDF-weighted centroid cosine ranking. Random action permutation for each of three seeds, all use same test utterances, so not independent corpora. Reward count 391,556–393,571; large enumeration cost, equivalent to obtaining intent labels by costly trial. Test does not query reward. Margin threshold 0.8 is a descriptive diagnostic, not independently calibrated.

| Model | Accuracy | Top 3 | Coverage at margin | Accuracy among answered |
|---|---:|---:|---:|---:|
| Word/bigram | 82.50% (2541/3080) | 95.03% | 66.17% | 94.75% |
| Word/char | 82.63% (2545/3080) | 94.77% | 63.02% | 95.57% |

No language generation or banking action was performed. This is real utterance intent selection with simulated semantic reward, not authentic environmental skill execution. Word-only almost matches char feature arm; lexical baseline remains important. Random 1/77 ~1.30%. Future controls: equal feedback-budget policy, learned compositional representations, independently calibrated abstention, human/task environment interactions.

## 010: inference and sparse candidate scaling

One-thread CPU, batch 1. Timing includes complete trained parsing/context forward and single answer selection from padded byte messages, does not include vocabulary tokenization, real environmental I/O, persistent-cache decode, GPU kernels or feedback. Median of three per-seed medians, five repeats each in `matched_timing.json`.

| History+query messages | Padded bytes | Modular µs | Byte Transformer µs | Ratio |
|---:|---:|---:|---:|---:|
| 17 | 85 | 226.51 | 581.53 | 2.57× |
| 77 | 385 | 324.58 | 2643.61 | 8.14× |
| 269 | 1345 | 741.38 | 18725.71 | 25.26× |
| 781 | 3905 | 2010.29 | 137960.48 | 68.63× |

Held-out extension (`long_results.json`), 32 additional sessions per seed and length, three seeds: 269 messages — modular 96/96 (100%), direct byte Transformer 17/96 (17.71%); 781 messages — modular 96/96 (100%), direct byte Transformer 14/96 (14.58%). Longer histories retain the same 12 live entities, only increasing overwrites. This extension was run after the initial audit and was not separately replayed. Do not compare these as matched-quality LLM latency. Fixed live-entity count of 12 means the hard-coded executor does not retain raw history. `diagnostic_results.json` has exact prepared reader timings across 512–131072 historical writes (8 overwrites per live entity), including separately measured memory preparation. At 131072 writes, hash preparation 11.62 ms and lookup 0.166 µs; dense NumPy softmax preparation 32.18 ms and lookup 1199.34 µs. Oracle keys and last-write semantics, not a trained LLM kernel.

`sparse_results.json`: 3 seeds, 7 n/d configurations, 256 query cues per configuration (21 fixtures). Sparse prototypes 8 features, 4 labeled training examples/entity, 50% query corruption, sparse inverted posting counts; top-four heap for candidates with a score. Cap per feature 8,32,128, or uncapped. Fixed 128 features becomes ambiguous as entity universe grows. With D=2N, cap 32 and 8 queried features, max 256 posting touches, measured about 68.5 at N=16384. Feature vocabulary grows linearly; this is not fixed-memory and relies on synthetic learned identities being discriminating. Sparse posting allocation and preprocessing are not included in query timing. Candidate coverage at N=16384: 1.17% fixed D128 cap32, 38.28% fixed D128 uncapped, 100% D32768 cap32. At N4096: 7.81%, 70.57%, 100% respectively. At N16384, D32768, cap32 median query 18.20 µs versus fixed D128 uncapped 4454.38 µs; representation capacities and accuracy differ, so no pure kernel speed claim.

## Reproduce and audit

Tested Python 3.13, NumPy and PyTorch CPU; exact environment in `environment.txt`. Run from package directory with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

```
python sequence.py
python raw_attention.py
python raw_attention_matched.py
python diagnostics.py
python native_bridge.py
python recovery.py
python sparse_index.py
python audit.py
python long_eval.py
# separately obtain exact official train/test CSV files from source above
node banking77.mjs /path/to/train.csv /path/to/test.csv > banking_reproduced.json
sha256sum -c SHA256SUMS  # on pristine archive only, before regenerating results/checkpoints
```

`python audit.py` replays 64 diagnostic jobs, 128 bridge jobs, 128 recovery jobs and 21 sparse configurations; checks aggregate predictions against all nine checkpoint packages across five cohorts; asserts 3,840 per-example indexed vs full-history softmax answer equivalences. Excludes timing, does not independently retrain; not an independent auditor. Node code contains the same core experiment as connector run with extra input hashing; authentic data are not in archive and no local replay claimed. Long-history extension is an exploratory post hoc cohort, not preregistered. All timing machine-specific; no error bars or multiple-testing adjustment. Experiment choices made here without pre-registration, no claims of hypothesis-testing significance.

## Next qualification gate

1. Fix grammar generalization (withheld compositional forms, contextual aliases, many unseen entities) using semantic consequences and equal training/feedback budgets. The oracle executor shows that explanation/grounding is the core issue.
2. Use full pinned upstream checkpoints and install compatible dependencies. Reintegrate independent proposal recovery and a version-aware memory gate; include correlated-error environments and real observation cost.
3. Compare modular agent to strong converged full-attention and attention+executor baselines on identical raw streams and resource budgets, with match-quality targets and multiple disjoint corpora. Include GRU, MemNet, state-space and indexed/sparse-attention baselines. No externally supplied gold records at eval; remove auxiliary labels in a separate arm.
4. Learn informative sparse routing with measured posting depth, collision/overflow p99, growth and write/repair cost. Validate long-context multi-hop and open-ended grounded communication rather than intent-only testing.
5. Measure quality-matched inference over actual prefill/decode/streaming including caches, hardware-optimized kernels, memory allocation, observation cost, and training amortization. Inference-weighted objective requires quality gates before ranking speed.