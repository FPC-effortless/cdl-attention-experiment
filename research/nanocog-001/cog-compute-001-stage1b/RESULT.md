# COG-COMPUTE-001 Stage 1b: training-only support supervision

**Result: 0/6 qualified.** On fresh six-node maps, annotating the supporting source position during training improved exact answers in one seed and did not change the other two. Neither arm completed an answer-changing family on either query axis. This is a negative result for the tested objective and architecture, not for latent binding in general.

## Locked comparison

Two fresh arms at each of seeds 173, 271 and 907 used the same 96,192-parameter COG backbone: one tied block applied twice, with local lag operators and causal latent summaries. Both received identical initialized backbone weights, 3,240 original training rows, paired batch order, 1,408 AdamW updates, 33,792 answer draws and 101,376 language targets per fit. The intervention adds a 36-way support-position classifier on the normalized query-end hidden vector (3,492 additional training parameters, fixed loss weight 1, separate head gradient clipping). Its label is available only in the auxiliary training capsule. The head is removed at checkpoint save; both inference arms consume only the same raw prompt bytes and emit unmasked full-vocabulary tokens. No persistent memory, inference-time oracle, explicit selector or additional loop. This is matched exposure, not identical training FLOPs. The three answer-only refits reproduce the original Stage 1 final-state and batch-draw hashes.

The primary panel enumerates both relations and every subject in 12 new six-node topologies across two shuffled layouts (288 prompts). Within each matched pair, facts and their order stay identical and exactly one query token changes. Per seed: 720 answer-changing subject pairs, 116 answer-changing relation pairs and 28 answer-invariant relation pairs. All 12 maps contribute to every axis. The prospectively frozen gate required at least 260/288 exact free answers and 11/12 complete maps on each axis, requiring both correct answers for every enrolled pair across layouts. Evaluation differs from the older mixed near/shuffled training distribution. Forge and an independent raw-data reconstruction verified split/provenance and pair semantics.

## Results

| Seed | Answer-only exact /288 | Support-aux exact /288 | Rescued / harmed rows | Support head on final 3,072 TRAIN draws |
|---|---:|---:|---:|---:|
| 173 | 48 | 48 | 20 / 20 | 2,841 |
| 271 | 48 | 48 | 15 / 15 | 2,698 |
| 907 | 48 | 66 | 30 / 12 | 2,983 |

Every arm and seed completed **0/12** answer-changing maps on both axes and **0/116** answer-changing relation pairs. The auxiliary arm completed 22/720 subject pairs in seed 907; every other arm/seed completed 0/720. The invariant-relation complete-map counts were 0/1/2 of 12 without the auxiliary head versus 0/0/1 with it, in seed order. No arm got every query for a map right. Of note, ignoring the query and selecting any one in-map destination per layout scores exactly 48/288 by construction; this is a reference accuracy, not a demonstrated description of what the models predicted.

On the supplemental K4 panel, exact answers were 46/42/44 vs 42/42/59 out of 192 each. Six of 12 K4 roots reuse prior CAL-only topology classes with new labeled maps. The primary K6 set uses topologies unseen in the preceding CoT and Stage 1 studies. All 2,880 generated responses across both panels had correct output format. Supplying the public answer marker yielded bit-exact matching answer logits and did not rescue a single result.

The auxiliary-position counts measure repeated training draws, not transfer, causal use of a position, or the presence of a selector at inference. The seed-907 row improvement is not a robust solution to binding. Formatting is not what limited this run. The study does not measure whether same-map counterfactual training, an architecture with role-conditioned composition, or greater capacity would help.

## Decision and reproduction

Keep multi-hop qualification blocked. First isolate access from composition with fresh, fully visible raw-token examples that cross subject, relation, position and value. An identical-width, tiny learned attention comparator can test task learnability; the COG candidate should learn within-layer query-conditioned interaction rather than receiving a supplied source or adding persistent records. Do not use this exposed panel as another sealed test. A later latent loop requires independent evidence that carrying a previous pass's message matters at the same number of applications.

The sole native invocation finished the six fits and six raw evaluations in 270.317 serial role seconds (one CPU core; not serving latency), with independent reconstruction of the sealed score. Public aggregate only; replay includes raw synthetic predictions, exact inputs, source and new model artifacts under owner-only custody.
- Data ledger: `a44752a6abdf4744fc3e06418991b39259b6443c5857262d4d5a759a840177cb`
- Execution lock: `2498488424108aa02a6916e03b6e5863ca46ce12bb3428979a4047ebbddbe796`
- WIRE: `6e6c30d03fa55900c5b5e66b49049b76951f6cabb60d73b8e4ce260d39831a7d`
- Score: `a62724cfa48d2079440e0a9f6791053859c03318fb31b7933992c2a9af5acddf`
- Independent recount receipt: `dad5a0223df65d2a41d90e1342e3cc3fde13a465824393e34c8d7eaa0db61b1e`
