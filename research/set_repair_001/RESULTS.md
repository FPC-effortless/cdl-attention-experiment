# SET-REPAIR-001 — the latent consistency repair fails its screen

14 November 2026. Twelve matched continuation fits completed. The tested SET-row MSE objective greatly reduces old-value-dependent representation differences, but performs worse on average than ordinary continuation for both models. Do not promote this repair. The result applies to the fixed weight 1 objective and 256-update budget; it does not rule out every operation-specific objective or curriculum.

## Matched behavioral result

Each model/stage is evaluated on the same 80 held-out paired groups, 160 worlds and 640 queries per seed. Values and causal templates are familiar; full histories are new and disjoint from training. Aggregates below cover three seeds and are descriptive, not independent-sample statistical estimates.

| Model | Stage | Answers /1920 | Accuracy | Complete maps /480 | Complete-map accuracy |
|---|---|---:|---:|---:|---:|
| CSM | Starting checkpoints |1583|82.45%|202|42.08%|
| CSM | Ordinary continuation |1801|93.80%|365|76.04%|
| CSM | SET invariance objective |1781|92.76%|354|73.75%|
| Pointer slot | Starting checkpoints |1431|74.53%|140|29.17%|
| Pointer slot | Ordinary continuation |1601|83.39%|241|50.21%|
| Pointer slot | SET invariance objective |1567|81.61%|207|43.12%|

The strong improvement over starting checkpoints is mostly attributable to additional training on the new paired stream distribution. It cannot be credited to the added invariance objective. This new validation bank differs from previous studies; compare stages within this table, not its starting percentages against earlier panels.

Perseed normal query counts (each /640), seeds 101/202/303:

- CSM control 600/596/605; repair 607/568/606. Paired deltas +7/−28/+1. The repair rescues 64 answers and harms 84 overall.
- Pointer control 597/492/512; repair 582/475/510. Paired deltas −15/−17/−2. The repair rescues 183 and harms 217 answers overall.

Both architectures fail every prespecified behavioral repair criterion except query preservation: mean query and map improvement, no seed query regression, no template×seed drop exceeding 2percentagepoints, and no same-value probe seed regression all fail. The worst template deltas include CSM 202 DELETEroot −7.24pp and pointer 202 DELETEroot −13.82pp. Normal queries preserve all user records exactly in all 12 continued checkpoints. Three starting checkpoints do not: preservation is 568/640 for CSM 202, 592/640 for pointer 101 and 560/640 for pointer 202; other starting checkpoints preserve 640/640. Both continuation objectives repair those observed query-mutation failures on this bank.

## Where regressions appear

Each model/stage has 1440 concrete-value and 480 UNKNOWN queries across seeds.

| Model | Stage | Concrete correct /1440 | UNKNOWN correct /480 |
|---|---|---:|---:|
| CSM | Control |1324|477|
| CSM | Repair |1301|480|
| Pointer | Control |1124|477|
| Pointer | Repair |1107|460|

CSM 202 loses 28 answers relative to its control: root−3, alias−7, spare−8 and unrelated−10. Thus damage is not confined to the record constrained by the new objective. It is a behavioral regression of the jointly trained network, not evidence that QUERY physically overwrote those records. Pointer regressions also span roles and include loss of UNKNOWN correctness.

The secondary, already-known target-only diagnostic is not a fresh holdout:

| Model | Stage | Single SET /96 | Same-value rewrite /96 | Changed-value rewrite /672 |
|---|---|---:|---:|---:|
| CSM | Control |96|90|637|
| CSM | Repair |93|90|625|
| Pointer | Control |82|79|541|
| Pointer | Repair |83|73|500|

CSM repair seed 101 reaches 224/224 changed-value answers and 32/32 same-value answers, but seeds 202/303 do not reproduce that complete success. Selecting that seed would contradict the fixed comparison. Ordinary continuation resolves all single-SET cases across the three CSM seeds on this known diagnostic without the new penalty.

## The auxiliary loss succeeds without improving the task

Normal-validation root vectors are captured immediately after the shared SET, before later deletion/detachment operations. Paired histories differ only in the root's initial old value; COPY snapshots are correctly allowed to remain different. No paired answer equality is imposed.

| Model | Metric, mean across seeds | Control | Repair |
|---|---|---:|---:|
| CSM | Paired root-row MSE |1.00347|0.00941|
| CSM | Root-vector RMS |0.87343|0.89147|
| CSM | Between-new centroid MSE |0.37221|0.38185|
| Pointer | Paired root-row MSE |1.00279|0.00859|
| Pointer | Root-vector RMS |0.89784|0.91934|
| Pointer | Between-new centroid MSE |0.42232|0.30165|

The approximately 99% reduction in paired MSE is not explained by uniform vector-norm shrinkage: RMS increases slightly in every repair seed. Representation variance changes, and pointer new-value centroid separation decreases. These are descriptive geometric statistics across differing backgrounds, not a causal proof of why answers regress. CSM centroid separation does not fall on average despite its behavioral regression. Matching internal vectors is therefore insufficient as the success metric.

Erasing stored values before QUERY reduces normal continuation accuracy sharply: CSM control 173/1920, repair 193/1920; pointer control 171/1920, repair 173/1920. This supports dependence on retained values in this task. It does not establish correctly learned storage, routing or reasoning in every case.

## Design and execution

The two continuations of each starting checkpoint use identical input schedules, queried keys, process labels, 8192 answer exposures and fresh AdamW states. Both compute the auxiliary term; only its coefficient changes from 0 to 1. The treatment is explicitly an added semantic training constraint. Both architectures receive that same treatment; no architecture-specific supervision advantage is introduced.

800 generated paired groups contain 1600 unique full histories. Forge allocates 640 train, 80 validation and 80 unused groups. No model scores validation before all 12 fits close; the unused third partition is never model-scored. Independent forward/backward semantics agree. Four initial SETs distinguish these histories structurally from the earlier training streams. The absence_snapshot panel explicitly copies from a deleted root; it makes no recreation claim.

All 12 continuations complete 256 updates without failure, early stopping or retries. Recorded training-loop durations sum to 190.02 seconds; evaluation of 18 checkpoints takes 9.49 seconds. These are execution receipts, not a controlled cost comparison. Existing Forge→TrainLab→Veritas→ResearchLab interfaces were used without infrastructure modifications or paid compute. Five native gradient/semantic tests passed before fitting. Parent model/encoder source is loaded from verified source bytes, avoiding parent bytecode-cache execution.

A total of 23,040 validation predictions and 5,184 secondary probe predictions are preserved, with 18 evaluated model identities and 11,520 normal-validation root vectors. Six starting checkpoints are reference evaluations, not six new fits. Full fit/evaluation/source/weights/schedule receipts and raw evidence accompany the source.

## Disposition

Retain ordinary continuation as the stronger tested reference and keep the invariance objective unpromoted. The earlier intervention showed that the direct old-row pathway contributes to errors; it did not guarantee that a raw representation-matching loss would repair the learned system without collateral changes. This experiment makes that distinction concrete.

The next targeted question is how to improve correct value discrimination and replacement while retaining other operations. A task-facing SET objective or staged value-learning comparison would need its own fixed, matched training budget and fresh evaluation; neither was tested here. Do not respond to this negative result by silently selecting a seed, changing the penalty or extending the repair runs.

## Exact evidence

- Dispatch: af7c92d4ca58c480a12dd2348a05a22410adfb5f4ba58ec264362f64e7400414.
- Freeze: f1d29873a3fdb9cfb7fa4cb9af28ba996e246a2b7c7a05a840db61b78a4d2790.
- Forge manifest: fda13c70d0af3bde6d2cd2ca59454fd768eb31e84924edf03b9fefead4468cd4.
- Training closure: 13cf8f0dcd09eb03f819304d56b09b774f5b29417c37b12feee9172a29ea19dc.
- Validation predictions: 0e8847c58afb5d59bc35c84b1b2493f59da278c0301cf41a0cf2fcff06fbf17c.
- Probe predictions: 14109f308d800b3517e99ff99bcaef6deddfc21b3957ef1f10fe19ab3faed524.
- Root vectors: b94654b9cbf60227b1d150bf6e737abb9d0a9f49557f600ca36753f0d9297315.

Independent post-execution review verifies all 12 training receipts, 256 optimizer steps per fit, matched initial/schedule identities and paired first-batch loss components. It independently rescores all 23,040 validation predictions and 5,184 probes, and fully replays all 18 checkpoints. All saved logits and 11,520 root vectors agree exactly (maximum difference 0), with unchanged weights and no reviewer optimizer calls or historical-data accesses. Both failed repair screens are independently reproduced. Audit replay is not an additional scientific replicate.
