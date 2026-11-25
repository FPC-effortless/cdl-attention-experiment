# PULSE-006: correcting record selection did not repair frozen custody answers

Completed November 22, 2026. This is a **frozen, reference-assisted diagnostic** on the already exposed synthetic COG-PULSE-005 development cohort, not a model training, new qualification test, or nanoCOG result. Keep it separate from the original PULSE-006 fresh-confirmation study. Five matched seeds per mode saw the same 480 counterfactual families. Each pair counted only if both worlds were correct.

## Results

| Frozen training mode | Native reads | First slot supplied | Second slot supplied | Both slots supplied |
| --- | ---: | ---: | ---: | ---: |
| Direct current-event path (EVENT) | 1/2,400 | 1/2,400 | 4/2,400 | 2/2,400 |
| Broker-context control (CONTEXT) | 3/2,400 | 1/2,400 | 6/2,400 | 3/2,400 |
| EVENT, archived convention-correct chains in both worlds | 0/21 | 0/21 | 0/21 | 0/21 |
| CONTEXT, archived convention-correct chains in both worlds | 0/5 | 0/5 | 0/5 | 0/5 |

Second-slot assistance gave the best aggregate paired score, gaining four EVENT pairs while losing one, and gaining five CONTEXT pairs while losing two. These small rescues did not change the floor. In the 26 seed–family cases that already had the entire relevant chain correct **under the archived first-12-coordinate decoding convention**, no assisted arm solved both worlds. With both positions supplied, 5/42 individual EVENT worlds and 0/10 CONTEXT worlds in that restricted subset were correct. Complete chain agreement occurred in only 21/2,400 EVENT and 5/2,400 CONTEXT seed–family cases. These are repeated evaluations of 480 families, not independent new families.

All four variants kept the complete original 8×80 board and encoded question unchanged. Reference slots are provided only to assisted readers, not the native arm. With first-slot assistance, the native second hop is recomputed from the changed first read. The frozen decoder receives native vectors, not artificial one-hot values or supplied answer labels. The convention-correct subset does not establish that the other 68 coordinates form a code the frozen synthesis can interpret. Hard slot selection also differs from soft mixtures seen during training.

## Verification

- Native read reproduction was bit-exact over 9,600 original forward evaluations (300 batches) from all ten checkpoints. The evaluator independently matched native logits and argmax, source vector hashes, no-op empty slots, board decoding, reference routing weights and archived native result counts.
- Ten candidate restores, 20 total checkpoint decodes including parent weights, and 48,000 reader-world evaluations counting the native call and four explicit read variants. One CPU thread; no fits, no protected qualification data, no external compute. Prediction and separate weight-free verification took 21.8 and 5.6 seconds of subprocess wall time, respectively. These are diagnostics costs, not serving latency.
- Forge checked the exact projected cohort and separate truth binding. Independent exact checks admitted each closure; TrainLab and Research Lab recorded both completed phases. The native receipts are engineering diagnostics, not scientific qualification or an independent cross-machine rerun.

**Decision for nanoCOG:** Do not commit to training a stable-key-only repair of these checkpoints. Exact record selection was not enough even on the small subset where the stored chain looked right. Prioritize whether the updater preserves the right identity and value across edits, and whether a reader can reliably decode those exact stored values. Use a fixed-store/readout control to separate the two before a full language fit. This does not prove that learned stable keys are useless, or that no other decoder can use the native board.
