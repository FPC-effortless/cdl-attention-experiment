# COG-BINDING-LOCALIZE-002: reduced diagnostic hits a training floor

This reduced CPU run does not identify why the larger sparse models failed. All nine fits scored zero complete pairs in each of the six held cohorts, including dense COG. Providing the correct triplet at either sparse rank also scored zero. Because the dense reference failed, the null rescue is not evidence that routing is irrelevant or that the GPU reader cannot use the evidence.

## 256-token subject cohort

Each row is 512 paired families, evaluated identically across arms and seeds. “Neither history” means neither top-two set overlaps a window containing the target value. For the dense arm it is only a top-two diagnostic; dense readout itself uses all positions.

| Arm | Seed | Value window absent from both native top-two sets | Native complete pairs | Correct triplet, either rank |
|---|---:|---:|---:|---|
| cog_dense | 173 | 454 | 0 | not applied |
| cog_dense | 271 | 459 | 0 | not applied |
| cog_dense | 907 | 459 | 0 | not applied |
| cog_top2_pooled | 173 | 459 | 0 | 0 at both ranks |
| cog_top2_pooled | 271 | 488 | 0 | 0 at both ranks |
| cog_top2_pooled | 907 | 486 | 0 | 0 at both ranks |
| cog_top2_separate | 173 | 476 | 0 | 0 at both ranks |
| cog_top2_separate | 271 | 490 | 0 | 0 at both ranks |
| cog_top2_separate | 907 | 493 | 0 | 0 at both ranks |

## Verification and follow-up

Identical 248,704-parameter backbone and within-seed training draws, 512 updates of eight pairs each, 3,072 distinct held families. Intervention holds the other payload and both native weights unchanged; wrong-triplet controls use the other changed fact. Exact native logit parity was checked on each scored batch. A detached implementation recounted 954 numeric counts from predictions and routes. No inference-speed or large-model claim.

Separate no-training engineering fixture reproduced the original GPU export failure and passed with the documented runner output path, TRAINLAB_OUTPUT_DIR/metrics.json. The original failed receipt remains unchanged. The scientific follow-up must include a competent full-size dense reference and retain weights and route traces.

- Protocol SHA-256: ab7c26852dcfa14d569c12a7d4013109e1f9fb46c2309090223c1eabcc01225b
- Detached audit SHA-256: b270b1426987d0f29b546ddea747e91717ab3c0efdf8a789258c335f97a29ec7
