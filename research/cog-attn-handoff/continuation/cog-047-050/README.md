# COG-ATTN 047–050 qualification attempt

The runs combine a language fusion experiment, a snapshot-lineage guard, recovery of original checkpoints, public dialogue tracking, and a controlled continual workload with independently fitted causal and factorized controllers. Research Lab OS registered 24 TrainLab jobs. Same-host auditors checked the receipts and replayed 12,960 timed-cell outcomes, 1,440 cached decoder outputs and 288 raw-text bridge turns. A paired timing repeat used the v3 weights from those jobs and preserved the saved outcomes. Veritas ecosystem reference reverification passed. The domain replays use their own exact state oracles.

| Gate | Evidence and current status |
| --- | --- |
| 047, authentic multi-turn | Annotation replay passed. Prediction and safe update coverage on public human-authored MultiWOZ remain below gate. Live user interaction and native task execution remain open. |
| 048, original trained integration | Original LRC and CSM/BDH weights ran in the versioned-store bridge. LRC passed its held fixture. The CSM reader failed held routing competence. The original trained TAC checkpoint remains unavailable in the accessible checkout and releases. |
| 049, independently fitted causal comparison | The corrected typed, CPU comparison qualified at all five context lengths, with both full-prefix and public-scope causal controls. Earlier single-head attempts remain in the evidence. Useful accuracy on authentic language and accelerator comparisons remain open. |
| 050, continual end-to-end | The typed loopback and raw-text bridges ran through repeated observable shifts, missing evidence and lineage checks. The authentic, distributed, live-response gate remains open. |

The qualification boundaries follow the [COG-ATTN handoff](https://github.com/FPC-effortless/cdl-attention-experiment/blob/research/cog-attn-reproducibility-20261120/research/cog-attn-handoff/041-046/README.md) and the frozen [protocol](protocol.md).

## Grounding and admission

The runtime combines a complete-utterance structural parser, a frozen MiniLM role head, separate-role NLI scores, and a logistic correctness score fit on disjoint templates from threshold calibration. The verb lexicon comes from prior transition induction. The structural contract describes what counts as a complete single instruction. A trailing contradiction deliberately violates it. Test examples on the typed branch instantiate documented forms. This branch checks contract conformance.

| Held type | Determinate examples | Fused candidates correct before gating | Pre-clarification commits | Correct commits |
| --- | ---: | ---: | ---: | ---: |
| Typed | 384 | 325 | 384 | 384 |
| Implicit | 384 | 259 | 19 | 19 |
| Correction | 384 | 317 | 5 | 4 |
| Ambiguous | 0 | — | 0 of 384 | — |

The calibrated branch admitted 408/1,152 determinate commands at 407/408 exact precision. Mixed coverage was 35.4% versus the 70% gate. Implicit coverage was 4.95%. Raw fusion raised implicit exactness from 144 to 259 out of 384 and correction exactness from 207 to 317 compared with the NLI path on the same text. The effect respects a fresh held phrasing split and leaves the admission gap visible. Earlier low-coverage values come from different fixtures.

Snapshot verification authenticates the receipt, binds it to the current command and epoch, and walks derived-snapshot ancestry to raw acquisitions. Two matching receipts count only if their acquisition sets are disjoint. The guard rejected all four shared-origin tests and the matched incomplete and cyclic cases. Using service names accepted all four shared pairs. Disclosed independent acquisitions passed. Falsely declared separate acquisitions passed too. The test issuer's HMAC key is a local fixture credential. Authentication confirms which issuer made a claim while the acquisition truth still depends on that issuer and the public authority boundary.

In the online raw-text bridge the same frozen scores were computed at runtime, with and without epoch-scoped reuse. Each arm completed 144 turns across three seeds. All 90 intended commits per arm matched exact actions and replayed state, while ambiguous and unavailable evidence caused deferral. The memoized arm reused 24 bindings and took 1.15–1.28 times less elapsed time across paired streams. The separate verifier used actual TCP loopback and simulator responses. This is a within-model timing result on recurrent inventory commands. Initial external model training time is unavailable.

## Authentic dialogue and original modules

The MultiWOZ 2.2 slice comes from revision fe0c8e65cfcd8462bd33c86e35f21addc84ca82b. The first test shard replayed all annotated states on 1,669 human-authored user turns in its hotel dialogues. The predictive subset turns estimate one typed hotel-area update, using the raw prefix and prior predicted state carried forward through storage. Across three seeds the local model reached 3,352/5,007 exact states and the one-block attention control reached 2,143/5,007. Update exactness was 158/366 and 46/366 respectively. Gold-annotation replay and learned grounding remain distinct.

The expanded recovery run fit 26,815 turns from the 17 official training shards, calibrated on dev shard 002, and evaluated untouched test shard 002. The current-turn model recovered 79/97 true changes before gating, but only 79/212 attempted updates were exactly right. The longer-context sparse model reached 64/97 true changes and 64/221 exact attempts. Neither model found a 99%-precision release threshold with any coverage during calibration, so both guarded test runs deferred all updates. The corpus records crowdsourced conversations and state labels. These experiments never perform an external booking or interact with a live customer.

The LRC confirmation archive was verified before restoration. The copied original model, index, generator and encoder hashes match the prefit receipt. All three checkpoint digests match the checkpoint receipts. The learned LRC produced 768/768 exact answers and 768/768 exact two-hop paths on fresh bridge examples, and the adapter wrote its observed outputs to the versioned store. The original CSM reader and commit checkpoints are also byte-matched to the public handoff. They routed 253/768 held requests correctly. The original commit predicate was evaluated on a supplied typed workflow and released 253 writes. This recovers two genuine trained modules on their respective interfaces. One authentic grounder does not jointly compose all three historical modules. The pinned versions and recovered byte identities are in sources.json.

## Context and complete measured cost

Three independently initialized v3 fits per arm had equal examples and updates. Each causal model uses four attention heads, with biases initialized to distinguish public active-role positions. The full variant makes all prefix keys and values available, and the scoped causal variant sees the same public active frame as the local controller. The synthetic stream includes three visible authority changes, a blocked stale write at each change, a segment with missing votes, and coherent wrong replies from mirrored sources. All qualified arms predicted all 48 bindings on all seeds at all five lengths. They committed 36 intended actions, deferred the same 12 unavailable actions, and matched intended state throughout. The 48 logical actions repeat across context lengths. This is evidence about scaling with archives under a visible frame contract. Numeric KV parity was exact on 1,440 checked turns. The full prefix KV buffer at 32,768 tokens uses 8 MiB for the 32-wide FP32 model. This measures cache tensor bytes.

The table gives the median paired baseline/local ratio across the three seeds using the median of three interleaved timing repetitions. All uncharged requests went over a real localhost TCP connection. The charged rows add 2 ms in the responder for every verification or clarification round trip. C_total includes observed fit time, cold/setup and 12 times measured workload elapsed. Online ratios include the workload only. Inference, verification, clarification, repair and commit breakdowns are retained separately.

| Context tokens | Full causal C_total ratio, loopback | Scoped causal C_total ratio, loopback | Full causal C_total ratio, +2 ms | Scoped causal C_total ratio, +2 ms |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 3.17 | 1.72 | 1.68 | 1.24 |
| 512 | 3.14 | 1.69 | 1.71 | 1.25 |
| 2,048 | 3.22 | 1.71 | 1.70 | 1.23 |
| 8,192 | 3.21 | 1.72 | 1.79 | 1.25 |
| 32,768 | 4.24 | 1.71 | 2.13 | 1.23 |

At 32,768 tokens, the online ratio against the scoped control was 1.28 on loopback and 1.05 with the additional request charge. The weighted totals therefore include a substantial fit-cost component. The two initial attempts showed single-head competence instability. The recorded v3 results use a revised inductive prior and fresh seeds. This supports a competence-matched claim for a typed CPU workload with visible shifts. Wide-area latency, independent external verification, cross-host atomicity, and accelerators remain untested. CUDA was unavailable.

## Reproduction

Use Python 3.12, NumPy 2.3.5, PyTorch 2.15 CPU, transformers 4.51.3, ONNX Runtime 1.24.4, and the Veritas ecosystem packages at eec439492407b0aa26922c23b5ae40d95289f4eb. Re-run in a fresh directory and preserve supplied data, checkpoint and recipe trees.

    export TOKENIZERS_PARALLELISM=false
    veritas-ecosystem run "$PWD/ecosystem"
    (cd recipe && sha256sum -c ../upstream.sha256)
    (cd recipe/minilm && sha256sum -c ../../encoder.sha256)
    (cd recipe/nli && sha256sum -c ../../nli.sha256)
    python recipe/fusion.py fusion.json
    python run_bridges.py modules
    python run_bridges.py online
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_bridges.py dialogue-repair
    python launch.py
    python launch_v2.py
    python launch_v3.py
    python kv_check.py > kv-audit.log
    python audit.py > audit.log
    python timing_replay.py > timing-replay.log
    veritas-ecosystem verify "$PWD/ecosystem"

The archive includes the source code, pretrained binary weights and original checkpoint receipts, derived corpus snapshots and source shard hashes, raw results, and a local lab database. External scientific replication and full 047–050 qualification remain open.
