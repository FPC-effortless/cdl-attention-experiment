# COG-UNIFIED-016 — developmental results, 2026-11-26

Source lineage: UNIFIED-015 local reference, fan-in initialization. CPU PyTorch 2.10.0. New seeds 4121, 4133, 4153. No architecture promotion or official campaign credit.

## Depth pilot
| Layers / width | Parameters | Mean validation BPB | Mean test BPB | State bytes |
|---|---:|---:|---:|---:|
| 1 / 224 | 1,000,117 | 3.7006581953 | 3.7593956958 | 1,792 |
| 2 / 160 | 1,000,256 | 3.7713354039 | 3.7984857887 | 2,560 |

| Seed | L1 validation | L2 validation | L1 test | L2 test |
|---:|---:|---:|---:|---:|
|4121|3.7177585509|3.7566654722|3.7300513626|3.7782181400|
|4133|3.6684009307|3.7679039674|3.7329681848|3.8276678048|
|4153|3.7158151042|3.7894367720|3.8151675398|3.7895714214|

Six 400-update fits; 614,400 byte exposures. Previously exposed corpus. No sustained 10M or formal muP study.

## Semantic custody
Familiar surface aliases, synthetic English-like statements, competing entity, held reordering, final-answer supervision except explicit auxiliary arm. 300 updates per fit, 144,000 total structured history exposures.

| Model | Familiar 16 events | Familiar 1024 | Held 1024 |
|---|---:|---:|---:|
| Bag, 8 slots |0/288|0/288|0/288|
| Soft, 8 slots |3/288|0/288|0/288|
| Hard, 8 slots |0/288|0/288|0/288|
| Hard, 4 slots |0/288|0/288|0/288|
| Aux-supervised hard, 8 slots |288/288|288/288|62/288|

The grammar-aware keyed-store control passed 1728/1728 selected histories; no neural oracle fit. Auxiliary supervision gives known training-grammar subject, update decision and read-destination labels; this does not establish autonomous language understanding. Primary gate failed.

## Replays and diagnostics
21 trained checkpoints; 21/21 exact recorded evaluation replays; 6,900 optimizer updates. Hard mask checks: 13,406; errors: 0; selected writes: 1,954. 72 no-fit eviction-policy evaluations. On 64-record capacity, recent query LRU=100%, popular query LFU=84.10%. This is a classical policy comparator, not a trained COG mechanism. Separate state protection from correct interpretation. Both hard unassisted arms have identical scored outcomes despite selected writes; cause not isolated.

First-layer effective rank on 224D 1-layer model: 52.474 initially, 8.025 at 100 updates, 11.069 at 400. In 160D 2-layer model: 44.606 initially, 7.299 at 100, 9.675 at 400. No new rank-expansion training.

## Artifact
Original local ZIP: cog_unified_016_evidence.zip
Bytes: 23010196
SHA-256: 22c58f74da19dad051931f266a3e0ac404e0c85cbf79859c81dcaac5565bd401
CRC: passed. Includes 21 .pt files, receipts, source, protocol and verification. Original corpus is not approved for public redistribution. Remote binary status: UPLOAD_BLOCKED; the hash is not evidence of remote bytes.

No full integration, sustained 10M, formal muP, GPU or matched-quality lifecycle-cost study was completed. Keep these open rather than representing the developmental cycle as full qualification.
