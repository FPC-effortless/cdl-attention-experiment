# Learned fine-key development probe

## Status and scope

The first proposed implementation and its two unchanged initialization replications learned usable fine addresses on three **development** sets. This is a key/retrieval probe, with **no answer decoder**, and does not establish final answer accuracy, a capability gate, a speed advantage, Tiny Shakespeare performance, or language-model generality. The final model and prospective evaluation are a separate integration task.

No unsuccessful fits were discarded. The first run was seed 84101, followed by exactly two unchanged replications, 84102 and 84103. Hyperparameters and architecture were not adjusted between these runs. This is not an ablation: it does not establish which individual ingredient was necessary.

## Repair mechanism

The probe uses the original `SentenceEncoder` **design**, trained afresh. It does not assume that the original failed checkpoint's frozen continuous states preserve person identity. Word embeddings and learned within-sentence positional weights receive support-based training; no token-index parser, era/person/relation metadata, semantic key labels, or gold answer enters the key forward path.

For learned sentence encoding `E`, the normalized continuous keys are:

```text
write_key(record)       = normalize(writer_MLP(E(record)))
first_read_key(query)   = normalize(first_MLP(E(query)))
second_read_key(q, left)= normalize(second_MLP(concat(E(q), E(left))))
```

The writer and first-reader MLPs are `32 → 32 → 32`, with GELU between the linear layers. The second-reader MLP is `64 → 64 → 32`. The first retrieved **sentence value** enters the second query; the fine writer key is not substituted for that value. The key-only model has 11,712 trainable parameters before freezing.

Stage A trains for 1,600 updates using the existing support indices. The second query uses the gold first support **during training only**. The loss is support-index cross entropy over cosine similarities to the 16 training records, with temperature 0.08, plus `1 - mean(positive cosine)`. The explicit positive alignment makes reader and writer points close in one joint geometry, which helps make a shared quantizer meaningful. This explanation is the design rationale, not an ingredient-level causal attribution from the experiment.

A shared 16-center spherical K-means codebook is then fitted to the 64 unique record token strings found in the **training set**. Deduplication uses raw token-tensor equality. No semantic grouping is supplied. There are five restarts and up to 100 iterations per restart; selection uses only training inertia. Both readers and writers choose their hard address by `argmax(normalized_key @ codebook.T)`. Sixteen codes match the available joint capacity of the prior 2-relation × 8-address model, although the parameterization is different.

Stage B freezes the sentence encoder and writer, retains the codebook as a buffer, and fits only the two readers for 400 updates. The objective is code-label cross entropy at temperature 0.08 plus 0.25 times continuous support-index cross entropy. The code labels come from each gold supporting record's learned write code, not semantic IDs. All three runs already retrieved both supports correctly before this stage; therefore these measurements do not show that distillation is necessary.

No conventional memory-network teacher was needed. The dense training comparisons serve as a training objective, not a dense inference fallback.

## Hard retrieval contract

`infer_bounded(model, records, query)` accepts no gold supports or semantic metadata. It writes each record once to its learned hard address, then performs two sequential reads. Each read probes at most two codes and examines at most two records per code. Buckets larger than two records are explicit overflow misses; they are never scanned to find the gold record. Returned candidates are selected by the read key's cosine score against the candidate's cached fine key. The second read is computed from the actual first selected value, including a zero value after a null first read.

The cap is four scored records per hop, eight per two-hop query. The observed count here was two per hop, because every story had 16 occupied codes and one record per code. Codebook scoring examines 16 fixed centers, independent of the number of memory records. Initial encoding, writing, and index construction are linear in the supplied memory size and are not claimed to be constant-cost operations. This Python probe is not a latency benchmark or an optimized persistent index.

The diagnostic fields named `oracle_first_dense_support_hit` and `oracle_first_code_agreement` explicitly use the gold first record to inspect representation quality. They are separate from `hard_index_support_hit` and `hard_index_both_hit`, which come exclusively from the bounded, gold-free inference function. Their separation matters when reproducing this probe.

## Fixed run settings

| Setting | Value |
|---|---|
| Model initialization seeds | 84101, 84102, 84103 |
| Training data | Original generator, seed 71001, 2,048 quartets / 8,192 examples |
| DEV IID data | Seed 84001, 64 quartets / 256 examples |
| DEV withheld-pair data | Seed 84002, 64 quartets / 256 examples |
| DEV partner-switch data | Seed 84003, 64 quartets / 256 examples |
| Batch size | 32 |
| Stage A optimizer | AdamW, learning rate 0.003, weight decay 0 |
| Stage B optimizer | AdamW, learning rate 0.002, weight decay 0 |
| Gradient norm cap | 1 |
| Cosine temperature | 0.08 |
| Positive-alignment weight | 1 |
| Codebook | 16 centers, shared between writer and readers |
| Codebook RNG | Initialization seed + 20,000 |
| Batch RNG | Initialization seed + 10,000 |
| Inference probes / capacity | 2 / 2 |
| Runtime | PyTorch 2.5.1+cpu, one Torch thread |

All three development sets were created before the fitting loop and are expressly development data. None of the prior scored held-out datasets or a new final test set was used for model selection. Quartet members are related counterfactual examples, not independent statistical units. The partner-switch data generator is the existing `lrc_diagnosis_001/repair_pilot/bridge_data.py`, invoked with a new development seed.

## Observed development results

The following are actual hard-index retrieval results after Stage B. They were identical immediately after K-means, before reader-code distillation.

| Initialization seed | DEV IID both supports | DEV withheld both supports | DEV partner-switch both supports | Mean occupied codes per story | Overflow probes | Null-read fraction |
|---|---:|---:|---:|---:|---:|---:|
| 84101 | 100% | 100% | 100% | 16.0 | 0 | 0 |
| 84102 | 100% | 100% | 100% | 16.0 | 0 | 0 |
| 84103 | 100% | 100% | 100% | 16.0 | 0 | 0 |

For every run and set, both oracle-first dense support-hit rates and both oracle-first read/write-code agreement rates were also 100%. Each 256-example set used 1,024 bucket probes and scored 1,024 candidate records in total, with a maximum of two candidates scored per hop.

| Seed | Training K-means inertia | Fitting and evaluation wall time reported by probe |
|---|---:|---:|
| 84101 | 0.000525900162756443 | 8.59509062899997 s |
| 84102 | 0.0003455309197306633 | 8.521747766000772 s |
| 84103 | 0.0007031988352537155 | 8.425494140999945 s |

These wall times are run receipts, not a controlled speed comparison. The two replication processes ran concurrently on separate single-thread CPU fits.

## Integration guidance and limitations

The production model can reuse `LearnedKeys.write`, `query_first`, `query_second`, and `spherical_kmeans` directly or reproduce them exactly. The dedicated candidate index should own causal visibility, overflow policy, and deterministic tie handling. This probe's local Python index has no public streaming API and resolves exact score ties by candidate order; the recorded experiment did not test adversarial ties or record permutation.

For a separate answer decoder, train on detached query/left/right sentence values if preserving this key-training trajectory is required. Initialize the key module before the decoder to preserve its initial RNG lineage. The probe establishes no answer-value decoding result; a place value might be present in the retained sentence encoding, but that must be measured. Any answer training, final capability comparison, and cost accounting must be reported separately.

The evidence supports proceeding to a prospective integrated-model test with this fixed recipe. It does not show scalability to more identities, unobserved vocabulary, large or skewed memories, different record formats, or arbitrary multi-hop tasks. The 16-center codebook and overflow policy must remain explicit when those assumptions change. No general claim that hard-addressed memory beats a conventional memory network follows from this probe.

## Reproduction and evidence hashes

Run from the worktree root, replacing the seed with each of 84101, 84102, and 84103:

```bash
PYTHONPATH=/workspace/scratch/3f2444385d54/deps:/workspace/scratch/3f2444385d54/lrc_twohop_source/extracted OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 experiments/lrc_route_repair/key_learning_probe.py --seed 84101 --steps 1600 --distill-steps 400
```

The logs retain all four emitted Stage A loss samples, the pre-distillation results, and the final result receipt. Seed 84101 was captured from the tool-returned stdout and reserialized as JSONL after the run; numerical data and results are preserved, but byte formatting differs from its original stdout. Seeds 84102 and 84103 were saved directly from the returned stdout. No model checkpoints were retained for this deliberately small development probe; the script and seed reproduce the fits.

| Artifact | SHA-256 |
|---|---|
| `key_learning_probe.py` | `3452fff20fc84a5648e1f13aa8c91916fec30a79ffef00fb123d8361bc1fc37a` |
| `dev_probe_outputs/key_probe_84101.jsonl` | `45d71f79348b42776fdbb1ddbc250f60d314597f235847b934fd941d9714efc5` |
| `dev_probe_outputs/key_probe_84102.jsonl` | `4748d1ca2a29b851193e41322f4a6956d94b28d2d225cd8cd0bb3cc928d4f326` |
| `dev_probe_outputs/key_probe_84103.jsonl` | `cc806ba478b0dd3f8c600cb8dfc3ba137cd2a48b12f375a7ee721562508bb689` |
| Training tensor digest | `8cd4f0dc7ed824954e0b342c17d50e1d014654a0b84902852353e03498ce69c2` |
| DEV IID tensor digest | `5409911f4fe65041857ec38193c1c488cfc02723b3b363d7d384872a8af3504b` |
| DEV withheld tensor digest | `7388f45243918b8db9db32121b406b2ea44211fcb5996428973201b04f7fff94` |
| DEV partner-switch tensor digest | `dfd43283e3b50abbf201a09951dfc44940210c1dc13b7384ccff50e196e9941b` |
