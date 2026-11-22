# COG-COMM-ROLE-001: completed local binding ablation

**Decision: no arm qualified.** The local mixer learned the bounded task
in two of three seeds with additional exposures. Neither a wider mixer,
static gating nor the parameter-matched attention control met the
all-seed screen. Keep full COG-COMM training, packet retention and
broader syntax/transfer claims out of scope.

Fresh public synthetic panel only: 320 TRAIN and 120 disjoint joint
name/color combination families, familiar names/words and one seven-token
grammar. Each family has eight exact factorial worlds. All models have
all evidence inside their causal field. Judges require strict top-1 across
all 287 vocabulary IDs, not supplied alternatives. Both thresholds
(912/960 individual worlds, 108/120 all-eight families) must hold for
each seed. Neither historical EASY nor protected TEST was opened.
 
| Arm | Seed | Control 128 worlds | Control 128 all 8 | Control 1,280 worlds | Control 1,280 all 8 | FIT 1,280 all 8 | Pass |
|---|---:|---:|---:|---:|---:|---:|:---:|
| gated32 | 1171 | 12/960 | 0/120 | 946/960 | 112/120 | 306/320 | yes |
| gated32 | 1271 | 18/960 | 0/120 | 952/960 | 116/120 | 308/320 | yes |
| gated32 | 1371 | 0/960 | 0/120 | 547/960 | 0/120 | 0/320 | no |
| static32 | 1171 | 11/960 | 0/120 | 880/960 | 73/120 | 246/320 | no |
| static32 | 1271 | 18/960 | 0/120 | 934/960 | 104/120 | 277/320 | no |
| static32 | 1371 | 0/960 | 0/120 | 899/960 | 95/120 | 263/320 | no |
| attention32 | 1171 | 0/960 | 0/120 | 877/960 | 69/120 | 186/320 | no |
| attention32 | 1271 | 0/960 | 0/120 | 936/960 | 107/120 | 305/320 | no |
| attention32 | 1371 | 0/960 | 0/120 | 935/960 | 108/120 | 308/320 | yes |
| gated48 | 1171 | 25/960 | 0/120 | 953/960 | 115/120 | 313/320 | yes |
| gated48 | 1271 | 13/960 | 0/120 | 786/960 | 36/120 | 114/320 | no |
| gated48 | 1371 | 0/960 | 0/120 | 951/960 | 115/120 | 311/320 | yes |

## Training diagnostics (mean over last 32 updates)
| Arm | Seed | Prefix CE (nats) | Answer CE (nats) | Sibling foil loss | Preclip norm | Clipping rate |
|---|---:|---:|---:|---:|---:|---:|
| attention32 | 1171 | 1.6599 | 0.4892 | 0.2185 | 11.7278 | 1.0000 |
| attention32 | 1271 | 1.6323 | 0.2047 | 0.1353 | 9.9932 | 1.0000 |
| attention32 | 1371 | 1.6963 | 0.1451 | 0.0751 | 5.6097 | 1.0000 |
| gated48 | 1171 | 1.6438 | 0.0181 | 0.0018 | 2.3652 | 1.0000 |
| gated48 | 1271 | 1.6155 | 0.4734 | 0.3849 | 9.0983 | 1.0000 |
| gated48 | 1371 | 1.6749 | 0.0232 | 0.0045 | 4.4527 | 1.0000 |
| static32 | 1171 | 1.6575 | 0.3966 | 0.1504 | 7.4747 | 1.0000 |
| static32 | 1271 | 1.6430 | 0.2968 | 0.1081 | 7.3210 | 1.0000 |
| static32 | 1371 | 1.7028 | 0.4222 | 0.1486 | 10.7253 | 1.0000 |
| gated32 | 1171 | 1.6485 | 0.0639 | 0.0077 | 2.3612 | 1.0000 |
| gated32 | 1271 | 1.6209 | 0.1246 | 0.0625 | 4.1631 | 1.0000 |
| gated32 | 1371 | 1.6720 | 0.8068 | 0.6862 | 4.3674 | 1.0000 |

## Interpretation and decision

- Duration mattered: no arm solved a complete CONTROL family at 128; every
  arm improved by 1,280. Changing duration also changes experience and
  dataset coverage (128 unique TRAIN families in updates versus all 320
  repeated four times). These data do not establish convergence.
- The width-32 gated reference and width-48 gated arm each clear the
  screen in two seeds, but the failing seed changes. Width alone is
  not a reliable fix under this schedule. The static arm avoids a
  catastrophic complete-family collapse but still qualifies in zero
  seeds. Matched attention qualifies in one and is one family short in
  another; this does not establish general superiority of either mixer.
- Prefix CE at 1,280 spans roughly 1.62–1.70 nats across arms and seeds,
  while answer losses vary far more. Every trajectory was clipped on
  all of its last 32 updates. Inspect objective weighting and gradient
  allocation before a further architecture scale-up. This observation
  alone neither identifies clipping as the cause nor validates a repair.
- The attention control matches 37,414 parameters and per-layer lag
  range, not FLOPs, streaming capabilities, or end-to-end inference
  cost. These tiny-CPU native process timings are execution observations,
  not serving throughput.

## Custody and reproducibility

All 12 admitted TRAIN-only fits completed (15,360 updates total).
Native summaries and separate independent audits reconcile file hashes,
per-update schedule and actual token exposures, and all 24 endpoints.
The 48 fresh, label-free native replay runs produced 84,480 rows of
287 finite logits, verified before weight-free scoring. Both verification
lanes inspected diagnostics, not any protected split. Raw logits remain
private and are regenerable from exact endpoint models and actor inputs.
The original source is preserved in draft PR #88. Scope is bounded
mechanism feasibility; native eligibility `DECLARED_CONDITIONS_ONLY`,
qualification `NOT_QUALIFIED`.

- Veritas/TrainLab commit: eec439492407b0aa26922c23b5ae40d95289f4eb
- Exact TRAIN SHA256: 02607c19e760e164989865a54422eb7cd78fed7d117a7fa2ff6c4df9d700f9f8
- Actor ledger SHA256, baseline: 9f53d947e7c81950e52368c2a6f71aee2362551c9d3676cfc011cf707413f970
- Actor ledger SHA256, comparators: 84570c59ca60ec960b2e9d4df50a75a0cb5525ef3080a87ee8d7582514540d3d
- Prediction seal SHA256, baseline: a3649904c94cfac51627a1b49a97bf5492a1c732c55fc21e699cf7d2948a22b9
- Prediction seal SHA256, comparators: 309a6757a70d0ac624b8f9767f4471c8e9ac38ba405d8b1648f7a0f61758c8a5

Independent output recount: 594 matched scalar/count checks for the 12
baseline predictions and 1,782 for the 36 comparator predictions.
Aggregate SHA256: baseline `36dd9aaa7d50ad0657fe6ee4b967d5466dee7adb2ba13e13df3bb98e7c234942`;
comparators `3d5e8e660807b8187f1bf55bebccb244be115df87a42fe576fb161753b373621`.
For a prospective repair, vary one training choice at a time starting
with objective weighting or clipping. Use a fresh frozen panel for any
promotion claim; this panel has now been inspected.
