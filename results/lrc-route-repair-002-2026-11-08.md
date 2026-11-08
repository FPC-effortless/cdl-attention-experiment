# LRC-ROUTE-REPAIR-002: identity-preserving keys and content-aware collision resolution

Completed 8 November 2026. Frozen fitting/evaluation commit: `6c3976e` on local review branch `research/lrc-route-repair-002-20261108`.

## 1. Result and decision

The two diagnosed defects are repaired in a new sentence-level LRC routing implementation. Across three fresh initialization seeds, it achieved **100% answer accuracy, 100% retrieval of both intended supports, and 100% perfect answer quartets on every fresh test split**. This includes two IID cohorts, two withheld-pair cohorts, and a partner-switch cohort. The conventional supervised memory network also scored 100% throughout.

There are 5,120 fresh examples, organized as 1,280 counterfactual quartets, evaluated separately for each seed. The repair therefore made 15,360 correct primary predictions. These are repeated model evaluations of the same five cohorts, not 15,360 independent test worlds. The independent audit passed all 90 recorded score cells, including controls and diagnostics, containing 92,160 saved predictions.

The capability and mechanism screens pass. **The cost comparison still favors the memory network.** Mean complete-call latency is 0.580 ms for repaired LRC, versus 0.211 ms for MemNet and 0.922 ms for the freshly retrained original LRC. The repair is about 2.76 times as slow as MemNet at this 16-record CPU workload. Complete training takes about 9 seconds for both repair and MemNet.

The evidence supports retaining this repair as a working research implementation and correctness reference. MemNet remains the stronger efficiency reference. This is a repair of the supplied-sentence task; no new Tiny Shakespeare or general language-model result is claimed.

Primary evidence is in `confirmation/results.json`, `predictions.pt`, `benchmark.json`, and `independent_audit.json` in the accompanying evidence bundle. The complete per-seed quality and cost tables appear below.

## 2. What changed in the mechanism

### Keys retain person identity

The previous diagnosis established that the archived agreement checkpoints assigned every person's location record within an era to the same discrete address. Their page/rank policy could not recover the correct person because it could not observe the candidate contents and ordering needed to distinguish the colliding records.

The repair learns a continuous fine key for each record and compatible reader keys. A shared learned codebook converts both writers and readers into the same discrete address space. The writer uses the original SentenceEncoder design, trained afresh, followed by a normalized 32→32→32 GELU network. The first reader has the same network dimensions. The second reader takes the query's sentence representation concatenated with the actual first selected record value, then applies a 64→64→32 network.

No model receives person IDs, era IDs, relation IDs, parsed fields, canonical semantic addresses, support indices, or answers during ordinary inference. Key formation learns from the two support-occurrence labels already available during training. Sentence boundaries and the finite vocabulary remain supplied, as they were for every model in the earlier comparison.

On every fresh example in every repaired seed, the auditor found **16 occupied addresses per story**. Within each era, the four partner facts have four distinct codes and the four location facts have four distinct codes. There are zero hard-key collisions between people within an era/relation group. Changing a location payload or the first record's partner value leaves that edited record's own address stable in **100% of the tested quartets**. Identity distinction and payload independence are measured properties of the fitted model.

The fresh original controls are not identical to the archived checkpoints: their mean location codes per era range from 1.0 to 1.5 across cells. The exact one-code-for-four-people collapse remains established for the archived models; it is not asserted for every new control.

### Collision resolution observes candidate contents

The old page/rank prediction is replaced with an actual bounded candidate comparison. At each hop, the reader probes two addresses. Each bucket can return at most two records. The resolver computes query–fine-key similarity only for those returned candidates and selects the unique maximum. An exact tie produces a null read. An overfull bucket returns no candidate prefix and records overflow. A missing first bridge aborts the dependent chain.

Occurrence order is absent from the content-score decision. If a record moves, its key and value move together and selection follows the record. The implementation separately counts returned references and continuous-key scores actually computed. This prevents a nominally sparse index from hiding a later full-record scan.

The fixed maximum is **four scored candidate records per hop**. Ordinary test examples used two per hop; forced collisions used at most four. The allowance is twice the archived adapter's two candidates per hop. Reading the codebook also requires 16 centroid comparisons per hop. Initial record encoding and index construction process the full memory once. All these costs are included in complete-call timing.

### The answer uses the retrieved chain

The final 32→32→4 answer head receives only the actual final selected sentence value. It has no direct query input. A null chain supplies zero. The second reader uses the actual first selected value at inference, with no gold bridge substitution during capability evaluation.

This is an architectural and learning revision, not a two-line correction to the old fitted weights. It replaces stochastic routing/ordinal credit assignment, adopts aligned continuous keys and quantization, and changes decoder training. The improvement belongs to that combined recipe. The collision intervention separately demonstrates the role of content-sensitive selection; this experiment does not identify the individual contribution of every training choice.

## 3. Fixed learning schedule and prospective controls

All four confirmatory arms use the same original 8,192 training examples and identical minibatch streams for each initialization seed. Every fit completes 2,000 minibatches of 32, without early stopping or checkpoint selection. Arms are repair, its shuffled-support control, original agreement LRC, and supervised MemNet. Initialization seeds 901, 902, and 903 were fixed before fitting.

Repair Stage A uses 1,600 minibatches. It optimizes cosine support cross entropy at temperature 0.08 plus positive reader/writer alignment, with AdamW learning rate 0.003. Gold first values appear only in this training objective. Dense comparisons with all records exist during training.

At the boundary, spherical K-means fits 16 centers to the learned writer keys of all 64 unique TRAIN token strings. Five restarts are selected exclusively by TRAIN inertia. No semantic IDs, metadata, or TEST strings enter clustering. Encoder and writer then freeze, their old gradients are cleared, and the Stage-A optimizer state is retained. Stage B uses 400 minibatches to train reader code targets plus a contrastive term, at learning rate 0.002. Clipping considers only active reader parameters.

The decoder trains on the detached gold-final sentence value throughout all 2,000 minibatches. Its optimizer and gradient clipping are separate from the key network, so answer loss cannot alter key formation. The repair has two optimizer step calls per minibatch despite using one encoded batch. Original LRC uses its archived agreement trainer with two full forwards; MemNet uses its archived supervised objective with one forward. Equal minibatches do not imply equal computation, objective, or supervision use.

The shuffled arm permutes both support indices jointly within each minibatch. That same shuffled pair supplies the teacher first value, every key/code target, and the decoder's final teacher value. Answer labels stay correct. Its mean withheld accuracy is 34.3424%, so the correctly supervised repair gains 65.6576 percentage points. Its bridge-switch accuracy remains near 25%, with zero perfect bridge quartets in every seed. The shuffled control is not described as exactly chance on every split.

Repair has **12,900 allocated trainable parameters**, with 9,540 still trainable in Stage B, plus 512 nontrainable codebook values. Original LRC has 18,879 parameters; MemNet has 19,129. The validated smaller width was retained rather than adding unused weights. This is a declared parameter difference, not a parameter-matched comparison.

Development was separate. Three support-only probes used initialization seeds 84101–84103 on DEV seeds 84001/84002/84003, with unchanged successful outcomes. Full integration used 84111. Review then corrected stale frozen gradients in Stage-B clipping and made a null first bridge abort the chain. Seed 84112 qualified those corrections on the same DEV cohorts. All development outcomes and their chronology are retained. No fresh TEST was opened to select the repair.

The fitting/evaluation source and imported original Python dependencies were hashed before confirmation. All 12 fits closed before TEST generation. Fresh seeds 85002/85003 and 85005/85006 provide two IID/withheld pairs; 85004 provides bridge switches. Each set has 256 quartets, or 1,024 examples. Audit found no exact TRAIN/TEST or cross-TEST input overlap. Withheld pairs combine known names; they do not introduce unknown entity names or new sentence templates.

## 4. Complete primary quality results

Values in each cell are ordered by initialization seed **901 / 902 / 903**. A perfect quartet means all four counterfactual answers were correct.

| Model | Fresh split | Answer accuracy: 901 / 902 / 903 | Both supports: 901 / 902 / 903 | Perfect quartets: 901 / 902 / 903 |
| --- | --- | --- | --- | --- |
| Repaired LRC | fresh_iid_a | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| Repaired LRC | fresh_withheld_a | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| Repaired LRC | fresh_iid_b | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| Repaired LRC | fresh_withheld_b | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| Repaired LRC | fresh_bridge | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| Repair, shuffled supports | fresh_iid_a | 31.25% / 30.96% / 39.94% | 2.25% / 1.17% / 3.61% | 5.86% / 3.52% / 3.91% |
| Repair, shuffled supports | fresh_withheld_a | 33.30% / 29.10% / 38.57% | 3.32% / 0.59% / 0.29% | 5.86% / 2.73% / 3.91% |
| Repair, shuffled supports | fresh_iid_b | 30.66% / 30.66% / 37.40% | 0.68% / 1.37% / 2.25% | 4.30% / 4.30% / 2.73% |
| Repair, shuffled supports | fresh_withheld_b | 30.47% / 29.88% / 44.73% | 5.08% / 1.95% / 0.29% | 5.47% / 3.12% / 7.42% |
| Repair, shuffled supports | fresh_bridge | 25.00% / 24.71% / 25.88% | 2.44% / 1.86% / 2.83% | 0.00% / 0.00% / 0.00% |
| Original LRC, retrained | fresh_iid_a | 41.11% / 46.09% / 45.80% | 14.45% / 15.23% / 4.30% | 21.48% / 28.12% / 27.73% |
| Original LRC, retrained | fresh_withheld_a | 44.92% / 34.38% / 44.04% | 13.67% / 5.47% / 7.03% | 26.56% / 12.50% / 25.39% |
| Original LRC, retrained | fresh_iid_b | 46.39% / 45.21% / 43.16% | 21.09% / 16.80% / 5.08% | 28.52% / 26.95% / 24.22% |
| Original LRC, retrained | fresh_withheld_b | 45.21% / 32.32% / 41.41% | 12.50% / 3.91% / 6.25% | 26.95% / 9.77% / 21.88% |
| Original LRC, retrained | fresh_bridge | 25.20% / 24.61% / 24.80% | 16.21% / 14.16% / 6.35% | 0.00% / 0.00% / 0.00% |
| Supervised memory network | fresh_iid_a | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| Supervised memory network | fresh_withheld_a | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| Supervised memory network | fresh_iid_b | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| Supervised memory network | fresh_withheld_b | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| Supervised memory network | fresh_bridge | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |

MemNet's hard-read diagnostic also achieved 100% answers and both-support retrieval on every split in every seed. It still scores every memory record before selecting a hard read; that diagnostic is not indexed inference.

The fixed screens require at least 95% answers, 90% perfect quartets, and 95% both-support retrieval in every seed/split. Additional conditions require bounded candidate access, no overflow on this task, permutation invariance, and at least 10 percentage points mean withheld advantage over shuffled labels. All six reported gates pass. These are engineering screens; three seeds and dependent quartets do not establish statistical equivalence or a 100% population success rate.

## 5. Tests directed at the defects

**Partner switch.** Within a quartet, only the queried partner fact's person value changes. The four people have distinct locations, so the answer must follow the new bridge. Repair answers all four variants correctly in every one of the 256 quartets in each seed. The freshly trained original LRC has zero perfect bridge quartets in every seed. This tests the possibility of a query shortcut that ignores the first retrieved record.

**Location-value switch.** IID/withheld quartets change only the relevant location value. The repair changes its answer while preserving the edited record's address. Independent key diagnostics also establish address stability when the first record's partner payload changes.

**Forced collisions.** Divide every learned address by two on both read and write sides, merging 16 addresses into eight paired buckets while preserving continuous fine keys. Content-based resolution retains 100% answers, both supports, and perfect quartets in all seeds. A deliberately content-blind smallest-occurrence selector gives 34.2773%, 40.0391%, and 30.4688% answer accuracy, averaging 34.9284%. This is a controlled selector intervention, not a retrained reconstruction of every old page policy.

**Record reversal.** Reverse records inside each era. All six normal/paired-collision seed conditions preserve answers, map selected occurrences back to the same records, and move writer codes with their records. This directly tests the old ordering defect.

**One probe.** A one-code diagnostic also gives 100% on the fresh bridge set in all seeds. The frozen default remains two probes. No cheaper runtime was selected after TEST or substituted for the main timing.

**Structural tests.** All 29 index/model tests pass. They verify content-dependent winners, actual winner changes when candidate keys swap, overflow without hidden traversal, first-null abort, second-null handling, ties, deduplication, and permutation mapping. Instrumentation on a 10,008-record fixture counts exactly four candidate-key comparisons per hop and excludes noncandidate key accesses. This tests the access contract, not learned retrieval on 10,008 natural records using the fixed 16-code model.

## 6. Measured CPU cost

Fits ran sequentially with one Torch/BLAS thread. Complete-call timing includes sentence encoding and index construction. Each model/seed uses the same 32 TRAIN examples, 32 warmups, and 30 interleaved blocks of 32 calls. Reported latency is the median of those block means. Prepared-memory timing is not substituted for complete-call timing.

| Model | Seed | Complete call, ms | Complete fit, seconds | Inference / MemNet | J |
| --- | ---: | ---: | ---: | ---: | ---: |
| Repaired LRC | 901 | 0.565311 | 9.4538 | 2.6398 | 11.7126 |
| Original LRC, retrained | 901 | 0.904675 | 39.2431 | 4.2245 | 18.7784 |
| Supervised memory network | 901 | 0.214150 | 8.8656 | 1.0000 | 0.0000 |
| Repaired LRC | 902 | 0.608827 | 8.6518 | 2.9289 | 12.8675 |
| Original LRC, retrained | 902 | 0.954051 | 37.7833 | 4.5897 | 19.7318 |
| Supervised memory network | 902 | 0.207866 | 8.8992 | 1.0000 | 0.0000 |
| Repaired LRC | 903 | 0.566014 | 9.1228 | 2.7008 | 11.8694 |
| Original LRC, retrained | 903 | 0.906833 | 39.1070 | 4.3270 | 18.9810 |
| Supervised memory network | 903 | 0.209577 | 9.6187 | 1.0000 | 0.0000 |

Means of the seed summaries:

| Model | Parameters | Complete call | Complete fit | Mean J |
| --- | ---: | ---: | ---: | ---: |
| Repaired LRC | 12,900 | 0.580 ms | 9.076 s | +12.150 |
| Original LRC | 18,879 | 0.922 ms | 38.711 s | +19.164 |
| MemNet | 19,129 | 0.211 ms | 9.128 s | 0 |

The repair reduces call latency by about 37% and fit time by about 77% relative to original LRC in this run. Compared with MemNet, inference is 2.64–2.93 times slower, with similar training times. The retained preference is:

`J = 12 ln(I / I_MemNet) + ln(T / T_MemNet)`

Positive J favors MemNet, and all three repair values are positive. This is an inference-weighted cost preference, not a requirement to be twelve times faster. Original LRC fails quality before cost selection; repair passes quality but loses the measured cost comparison.

These timings cover this Python/PyTorch CPU implementation at 16 records. They do not measure autoregressive generation, physical memory bandwidth, amortized persistent memory, or scaling to larger memories. The measured implementation includes its Python index and diagnostic counters. No unmeasured optimized runtime is claimed.

## 7. Independent integrity audit

The independent audit passed all 12 checkpoint, optimizer, RNG, and per-update-log receipts; 24,000 update-log rows; five dataset validators; all 90 score cells and 92,160 saved predictions; all six gates; and reconstructed bucket accounting for every recorded repair read.

Native replay covered 1,440 evenly distributed examples across 90 cells. Every discrete prediction, selected support, and code field matched exactly. The maximum per-example loss discrepancy was 1.1920929 × 10⁻⁷. The audit also reconciled 270 timing block means, nine timing summaries, and 8,640 measured complete calls.

The prefit manifest binds 23 source files, including original imported dependencies. The auditor's semantic parser validates data and key behavior; it never supplies a model input or substitutes for normal inference. Integrity passing does not change the negative cost result against MemNet.

Audit SHA-256: `aeb895e77bf0e872b47c73c276a1c1649b42346dd1ceac2c66a8a06eff5a503f`.

## 8. Remaining limits and research decision

This establishes a working learned-key/content-resolution mechanism on the supplied-sentence, four-person task. It does not establish a general language model, new-entity generalization, segmentation discovery, or a Tiny Shakespeare improvement. The earlier byte-level results remain separate until this learner is adapted and tested under language-model supervision.

The codebook has 16 addresses. With two occurrences permitted per bucket, at most 32 occurrences can remain in nonoverflowing buckets; duplicates can cause overflow earlier. The index can store more history, but overflow deliberately prevents a hidden unbounded query scan. Supporting growing or repeatedly updated memories requires additional capacity/routing work and evaluation. This experiment does not solve scalable memory.

The next research step is to preserve this working correctness reference while addressing measured query/index overhead and capacity growth. A cheaper default, learning without support labels, or a language-model adaptation requires a new frozen evaluation. The present evidence supports continued work from the repaired mechanism, while the equally accurate and faster MemNet remains the baseline to beat.

## 9. Reproduction and artifacts

Source is in `experiments/lrc_route_repair/` on the review branch. Main files are `model.py`, `index.py`, `train.py`, `run.py`, `benchmark.py`, `audit.py`, `PROTOCOL.md`, `tests_index.py`, and `tests_model.py`. A separate patch supplies the code changes. The evidence archive preserves results and checkpoints rather than duplicating repository-backed source.

The environment is Python 3.12, PyTorch 2.5.1+cpu, CPU FP32, one thread. Exact Python/platform metadata is in the prefit manifest. The recovered `lrc_vb001_comparison` package must be importable. Its archived source SHA-256 is `20196fc1c54a7fcfc177d977883728fe5e07edccb6491a8b8e8921c5e4efe3f5`. Imported original Python files are individually bound by the manifest.

From the review worktree, put its root, the recovered source parent, and the installed Torch directory on `PYTHONPATH`, and set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`:

```bash
python -m unittest experiments.lrc_route_repair.tests_index experiments.lrc_route_repair.tests_model -v
python -m experiments.lrc_route_repair.run prepare --output /absolute/new-output
python -m experiments.lrc_route_repair.run fit --output /absolute/new-output
python -m experiments.lrc_route_repair.run evaluate --output /absolute/new-output
python -m experiments.lrc_route_repair.benchmark --output /absolute/new-output
python -m experiments.lrc_route_repair.audit --output /absolute/new-output --source /absolute/recovered/lrc_vb001_comparison
```

The runner refuses checkpoint overwrites and fitting after closure, verifies frozen sources, and constructs TEST only after all fits close. The evidence bundle contains the 12 confirmation checkpoints, two full-model DEV checkpoints, fit/evaluation data, per-example outputs, optimizer/RNG states, receipts, logs, timing samples, independent audit, and this report. The three key-only DEV probes did not save checkpoints; their captured outcomes and provenance are committed with the source. No remote publication or merge is claimed.
