# COG-BINDING-ROUTE-GPU-003: a correct window did not reliably rescue sparse subject binding

All nine fits finished on a Tesla T4 and the original TrainLab receipt passed integrity and declared execution checks. The prespecified dense anchor passed, but neither sparse arm passed the prespecified all-seed rescue test. This is a synthetic binding diagnostic, not a new trained repair or a natural-language claim.

## Primary result: 256-token subject binding

Each seed was evaluated on the same 512 independent paired families. A complete pair requires both counterfactual answers to be exact. The focus group is native pair failures where neither history selected a window containing the target value. For that group, correct and wrong triplet insertions change only the payload at the named rank; the other payload, the native routing weights and the learned reader are fixed.

| Arm | Seed | Native full pairs / 512 | Value window in both histories | Full target triplet in both histories | Focus denominator | Rank 1 correct / wrong | Rank 2 correct / wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense COG | 173 | 503 | 512 | 512 | 0 | not applied | not applied |
| Dense COG | 271 | 505 | 512 | 512 | 0 | not applied | not applied |
| Dense COG | 907 | 498 | 512 | 512 | 0 | not applied | not applied |
| Top-2 pooled | 173 | 509 | 512 | 512 | 0 | 0 / 0 | 0 / 0 |
| Top-2 pooled | 271 | 24 | 103 | 7 | 148 | 10 / 0 | 2 / 0 |
| Top-2 pooled | 907 | 132 | 254 | 147 | 80 | 5 / 0 | 2 / 0 |
| Top-2 separated | 173 | 485 | 511 | 511 | 0 | 0 / 0 | 0 / 0 |
| Top-2 separated | 271 | 9 | 107 | 0 | 118 | 0 / 0 | 0 / 0 |
| Top-2 separated | 907 | 11 | 104 | 12 | 103 | 1 / 0 | 0 / 0 |

Dense uses all positions; its top-two columns are diagnostic, not a restriction of its native computation. Empty denominators are not zero-percent effects. The required rank-one effect was at least +20 percentage points versus the matched wrong triplet with at least 64 focus families **in every seed**, independently for each sparse arm. No pooling across seeds to pass this gate.

For the two weak seeds, rank-one correct payloads rescued 10/148 and 5/80 pooled focus pairs (versus zero for each wrong payload); separated rescued 0/118 and 1/103 (also versus zero). In the strong seed there were no focus failures, so it is not a third eligible replication. As a separate dependency check on that strong seed, swapping in the wrong rank-one triplet reduced both sparse arms to 0/512 complete pairs, despite unchanged weights. Correct evidence matters when the recipient already knows how to use it.

The fixed-weight intervention tests the trained interface as it is. Failure to rescue does not show that routing is irrelevant or establish a reader-only failure: the gold payload can be weakly encoded, assigned the wrong slot weight, or out-of-distribution in the forced message combination. Native evidence recall is necessary for some predictions but does not by itself demonstrate that the reader uses that evidence.

## Method and verification

Three original 6,407,424-parameter arms × three seeds; 1,536 actual updates at 32 pairs/update; identical initial weights and draws across arms within seed. Six independent held cohorts of 512 families each at 64, 128 and 256 tokens across two task types. Native forward parity was checked exactly in the pinned scorer. Detached recount independently reconstructed route/prediction counts and checked original source pins, TrainLab receipt, disjoint data, fit draw digests and checkpoint integrity. No persistent memory, new supervision, real-language, or latency claim.

- Recounted 1206 counts over 3,072 held families.
- Original native receipt SHA-256: d07aaebe1805c69bb7a3d0308c577aa4d7ef1b00a107e81bf53291e498b0a6b8.
- Detached audit SHA-256: c7a67dbb09dd6687d16357085b557feafac47c9feb4b32b209c6409f5262ede8.
- Uploaded source SHA-256: 835203869a7120811c2bd6e89ccfb834543f9fccb1f584d33180d372112cd451.
