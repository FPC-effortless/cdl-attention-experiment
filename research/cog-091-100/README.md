# COG-ATTN-091–100 | Attention-function substitutes without full self-attention

**Status:** locally executed bounded synthetic subsystem pilots, 21 November 2026; **not integrated COG-ATTN**, not a claim of Transformer replacement, published on GitHub draft PR #78; independent-investigator replication remains outstanding. This package deliberately preserves negative results, costs, and held-out mistakes.

## Executive result (leakage-audited version)

The study tests five functional gaps: lossless retroactive querying (091), trainable feature/routing credit (092), narrow subword generalization (093), simultaneous specialist inference (094), and provisional competing hypotheses (095), plus bounded program induction from demonstrations (096). A small Transformer is a supervised diagnostic baseline in 099. 097a and 097b are now completed synthetic subsystem diagnostics. Experiments 098 and full 099–100 qualification remain open.

**IMPORTANT:** An earlier internal language pilot had 72 examples/seed and exact single-action sentence overlap across pair splits. That version is **superseded** by this leakage-audited 60-example/seed version. `make_data` asserts disjoint complete utterance strings across train/dev/test. No claims below use the discarded 72-case numbers. All six seeds were rerun after this correction.

### 091: the raw-episode invariant

A memory with three mentions per item must recover the date associated with a specific historic note after later statements. The typed-slot control stores only the latest value per item; the raw scan and schema-agnostic inverted lexical index retain every utterance. There are 128 queries per seed at each scale and three seeds (9107, 9203, 9301).

| History records | Eager slot accuracy, average | Pretokenized scan accuracy | Indexed raw accuracy | Pretokenized scan median µs/query | Indexed raw median µs/query | Median index build, ms (not counting tokenization) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,536 | 33.6% | 100% | 100% | 39.61 | 2.57 | 3.15 |
| 6,144 | 37.2% | 100% | 100% | 195.64 | 4.05 | 11.14 |
| 24,576 | 37.2% | 100% | 100% | 1,145.69 | 9.19 | 63.52 |

The lexical index is **not** a fixed-size memory substitute; its storage scales with episode count, and it depends on exact token matches. It can miss synonyms, spelling variants, implicit reference and high-posting tokens. We added the stronger pretokenized linear scan after noticing that the original raw-scan baseline re-tokenized every message. The original naive scan remains in `results.json` for audit but must not be used as the main comparator. Tokenization and index construction are separately recorded; the end-to-end break-even for indexing depends on the query/write workload and memory costs. The histories are synthetic and do not model natural language referents.

### 092–094/099: learned compositional action grounding, no self-attention in proposed arms

Six given operations (`schedule`, `cancel`, `move`, `assign`, `approve`, `reject`) plus `defer`. The training, dev and held-out splits are disjoint by **ordered operation pair** and exact utterance string. Each example uses one of five constructed wording templates. Half of held test examples use an unseen but same-stem inflection. Three development seeds (9107,9203,9301) and **three separately retained replication seeds** (9409,9503,9601), with 60 held cases/seed. All models train for up to 38 epochs with held-out *development* checkpoint selection and the same optimizer settings; param counts differ. The baseline Transformer is small and trained from scratch, not pretrained or tuned to parity.

| Model | Parameters | Primary /180 | Fresh replication /180 | Combined /360 |
| --- | ---: | ---: | ---: | ---: |
| First-verb lexical rule | 0 | 54 | 54 | 108 (30.0%) |
| Trainable orderless character bag | 10,119 | 80 | 61 | 141 (39.2%) |
| GRU with frozen input projection | 20,343 | 74 | 74 | 148 (41.1%) |
| End-to-end trainable GRU | 20,343 | 165 | 160 | 325 (90.3%) |
| **Parallel temporal specialist convolutions** | **19,023** | **177** | **174** | **351 (97.5%)** |
| 2-layer, four-head tiny self-attention | 25,895 | 173 | 151 | 324 (90.0%) |

The gains over bag-of-characters and frozen-feature GRU support **task-driven input feature learning and compositional word-order processing** on this grammar. The local convolution specialists compute k=1,3,5 temporal features independently and merge in one feedforward pass. The small Transformer has more parameters; it is **not** a state-of-the-art attention comparison. Unlike the real COG controller, all these models use direct supervised action labels and a fixed action vocabulary.

Median local CPU cost for **60 online separate predictions** including character encoding and model inference (seed variation, not matched quality): bag about 20–22 ms; parallel specialists about 28 ms; recurrent about 39–40 ms; tiny Transformer about 43–50 ms. Preparation/training cost is recorded separately; full authorization, database, retrieval, and retry economics are **not** measured here. No claim of end-to-end COG inference speedup or GPU speedup is warranted. Model quality is unequal, so a matched-competence efficiency victory is not established.

The development-calibrated action gate is **not a safety guarantee**: primary parallel specialists made 3 wrong proposals among 144 admitted; fresh replication made 6 wrong among 144. These are *semantic proposal errors*, even when a calibrated development threshold seemed adequate. Signed authorization and independent semantic verification must remain external. These are also known-grammar generated outputs; there is no permission system in these scripts.

### 095: continuous competing hypotheses

A separate six-dimensional independent-sigmoid multi-label head is trained to keep both action alternatives when the input says “perhaps A or B.” It is **not** a self-attention matrix and never makes a persistent write. An ordinary single-choice classifier can learn `defer` while its unused action logits contain no information about the competing hypotheses; on 72 ambiguous held cases the parallel single-choice head placed *both* correct actions in its top two only **10/72** times. The separate two-hot supervised head retained both on **63/72**, while its decision head deferred on **72/72** ambiguous cases. Total dual-head decision accuracy was 359/360 across six seeds. Eight or more candidate hypotheses are not yet tested; no compositional coreference or real clarifying conversation was executed. The two-hot training targets explicitly disclose the intended candidate pair; this experiment does not learn ambiguity from unlabelled experience.

### 096: in-context operator-name binding via a supplied skill DSL

Five primitive string transformations and depth-≤2 compositions form a small, *engineered* hypothesis language. An unseen arbitrary operator label receives five demonstration input/output pairs. The learner keeps only programs consistent with all demonstrations and **abstains** if multiple or zero programs remain. With three seeds ×90 episodes = 270, 27 episodes had deliberate contradictory evidence. It correctly handled 177/270 total cases, including 150 correctly accepted predictions and 27 justified contradiction abstentions. The remaining 93 were abstentions due to non-unique hypotheses; **zero accepted wrong outputs** were observed. Verbatim demonstration replay solved 0/270 fresh inputs by construction. These results show narrow symbolic hypothesis induction, not unrestricted zero-schema concepts or learning language from life experience.

## Design direction: ability-equivalent but not mechanism-equivalent

```
Input / perception
  ├── Lossless append-only raw episodes + provenance + version/epoch
  │      └── generic lexical and eventually learned sparse candidate retrieval
  ├── trainable local temporal / subword specialist encoders (no dense QK matrix)
  ├── independent, continuous multi-label provisional hypotheses
  └── demonstrated-skill inducer with bounded, auditable program search
         ↓
  bounded candidates + uncertainty + counterfactual/causal evidence
         ↓
  existing COG signed authorization, source/epoch checks, semantic gate,
  transactional commit, rollback, and audit log — NEVER bypassed
```

This does **not** yet replicate full self-attention. It lacks unconstrained semantic querying without lexical anchors, true downstream credit through discontinuous candidate selection and the executor, broad long-distance cross-token composition, unsupervised acquisition of entirely new conceptual schemas, or pretrained language competence. A 100%-accurate synthetic rule system cannot establish general reasoning.

## 097–100: prospective qualification, not yet executed

- **097 selective retrieval budget:** 16/64/256/1024/32768 episodes; bounded candidate budget k=2,4,8,16, and a charged fallback, against exact raw scan and a trained Transformer at matched retrieval/answer competence. Include high-document-frequency tokens, colliding lexical mentions, absent indexes, paraphrases, and deliberately misleading candidates.
- **098 long-horizon revisions:** timestamped raw archive, signed source correction, stale-epoch invalidation, retroactive referent changes, conflicting witnesses; evaluate reversal pairs, unauthorized and semantically wrong commits independently, plus counterfactual rollback.
- **099 qualification:** train real COG perception/routing jointly with feedback on prior issue, MultiWOZ, unfamiliar operations and actual interactive puzzles, comparing no-memory, exact memory, trained Transformer, non-attention specialist, and resource-matched frozen-input controls. Charge token/byte encoding, writes, index maintenance, retrieval, speculation, human/probe simulation, verification, repair, output and wall CPU/GPU separately. Same training opportunity, quality bands, and coverage targets before latency comparison.
- **100 independent replay:** disjoint human/authentic datasets, sealed source bundles, primary/replication seeds, manifest/checkpoints, no training/test text leakage, and a separate GitHub-hosted replay. Preserve the original negative results and do not merge ahead of upstream PR #74/#75 checks.

### Explicit gates before promoting an integrated COG agent

1. **Semantic correctness:** ≥95% correct interpretation on genuinely novel language; report both precision at admitted coverage and recall of correct actions. A request may be permitted yet misunderstood.
2. **Memory:** ≥99% eligible correct-episode candidate coverage; stable across long history, collisions, retroactive changes and high-frequency anchor words, at a frozen probe budget.
3. **Authorization:** zero unauthorized commits on attacked local test suite, but track signed yet incorrect actions independently; no bypass of transaction/epoch/revision checks.
4. **Ambiguity:** ≥95% two-support coverage on fully novel ambiguity families; abstain if candidate provenance is missing or conflicting.
5. **Economics:** demonstrate statistically credible reduced complete-system CPU latency against a competence-matched attention model; include training and index amortization across realistic workloads. Report speed only in competent strata.
6. **Skill acquisition:** show novel-action and transfer gains without access to the evaluator's solver or fixed task-specific oracle; allow `unknown` if the provided primitives cannot explain demonstrations.

These are **proposed future gates**, not results or historical preregistration.

## Reproduce locally

Python 3.11 and PyTorch 2.10.0+cpu were used; `pytest` 9.0.2. CPU single-thread (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`). `experiment.py` uses only the standard library; the other models require PyTorch. Outputs have per-seed predictions/errors, confidence gating, timing, raw synthetic decision traces, and explicit limits.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python experiment.py --out results.json
python semantic.py --out semantic-results.json
python hypotheses.py --out hypotheses-results.json
python replicate.py
sha256sum -c SHA256SUMS
```

`SHA256SUMS` reflects the published local package and must be regenerated if rerun outputs are overwritten (timers are intentionally variable). Compare result **counts** and per-case decisions instead of binary-equal timing JSON. Audit scripts and all source are supplied; no external credentials or data fetch required. GitHub upstream context: [`PR #74`](https://github.com/FPC-effortless/cdl-attention-experiment/pull/74) and [`PR #75`](https://github.com/FPC-effortless/cdl-attention-experiment/pull/75). The latter remains a draft based on #74; this local package was **not** merged or uploaded to that repository.


## COG-ATTN-097a executed episodic grounding bridge (2026-11-21)

Added `experiment_097.py`, `test_097.py`, `test_097_boundary.py`, the byte-identical COG signed boundary source in `vendor_097/boundary.py`, per-case `097-results.json` and the qualified [097 report](REPORT_097.md). The six-seed read-only synthetic bridge achieved 287/288 determinate indexed reads and 72/72 appropriate ambiguity deferrals with the multi-hypothesis model, while a fixed-schema typed lookup remained a stronger cost control. **No real COG controller or authority flow was driven by the learned proposals; these files are now preserved on GitHub draft PR #78.**


## COG-ATTN-097b: source-grounded language retrieval with no structured entity ID

New: `experiment_097b.py`, `test_097b.py`, `REPORT_097b.md`, `PROTOCOL_097b.md`, and three raw result files (`097b-results.json`, `097b-scale-2048.json`, `097b-scale-32768.json`).

Across six seeds (678 answerable held-out synthetic requests), the learned dual encoder reached 431/678 versus 588/678 for char-TFIDF. A reader equipped with 14 *additional witnessed examples containing the held wording* reached 671/678, while support-shuffled control obtained 353/678. This is verified phrase reuse, not unseen-language semantic interpolation. It costs about 2.8 ms of local scorer time/query versus about 44 µs for cheap token overlap (but the accuracies differ). The raw archive preserved corrections; the 108 contexts with missing or multiple explicit names were deferred by engineered entity resolution. No signed write, authentic data, pretrained Transformer, general coreference, zero-schema model or economically admitted speed improvement is claimed.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiment_097b.py --out 097b-results.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q test_097b.py test_097.py test_097_boundary.py
```

See `REPORT_097b.md` for failure analysis and full limitations. Previous 091/097a results remain unchanged.
