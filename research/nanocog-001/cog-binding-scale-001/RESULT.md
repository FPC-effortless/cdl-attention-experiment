# COG-BINDING-SCALE-001: GPU execution recovered, qualification not met

## Outcome

All 24 planned fits exported prediction tapes on a Tesla T4; the exact Kaggle v1 job is nevertheless ERROR. After the last fit, the worker attempted to write metrics using TRAINLAB_METRICS_FILE, which its runner did not provide. The original TrainLab receipt remains failed (PLAN_ADMITTED, NOT_QUALIFIED), and the provider did not emit SCORE.json. Recorded hashes in the failed receipt verify; the receipt does not bind the prediction files as approved run artifacts. No receipt was rewritten and no new training was run.

The frozen detached scorer and an independent recount agree on 288 cells. These are **recovered diagnostic scores, not a qualified execution**. On the toughest subject cohort, keeping the two routed messages separate did not produce a reliable gain over pooling, nor did increasing model size rescue subject binding. At 6.4M parameters, dense COG and query-only SDPA did materially better than the top-2 variants across seeds. The 24.6M SDPA control also collapsed on that cohort; it cannot be treated as a competent large reference or evidence that COG is a better language model.

## Terminal synthetic paired results at 256 tokens

Complete pairs out of 2,048 for seeds 173 / 271 / 907. The subject and relation cohorts are distinct.

| Parameters | Arm | Subject | Relation |
|---:|---|---:|---:|
| 6,407,424 | Dense pooled COG | 2003 / 2023 / 1981 | 2039 / 2038 / 2038 |
| 6,407,424 | Top-2 pooled COG | 2040 / 127 / 539 | 2043 / 2048 / 2047 |
| 6,407,424 | Top-2 separated COG | 1978 / 43 / 44 | 2048 / 2048 / 2047 |
| 6,407,424 | Query-only SDPA | 2029 / 2036 / 2031 | 2037 / 2039 / 2039 |
| 24,611,328 | Dense pooled COG | 1879 / 1067 / 1438 | 2039 / 2005 / 2010 |
| 24,611,328 | Top-2 pooled COG | 67 / 126 / 88 | 2048 / 2048 / 2048 |
| 24,611,328 | Top-2 separated COG | 57 / 89 / 99 | 1511 / 1677 / 1920 |
| 24,611,328 | Query-only SDPA | 19 / 0 / 5 | 111 / 0 / 2 |

Frozen numerical gate: separated top-2 needs at least 90% across all cohorts and seeds and at least +2 points (percentage points) over pooled top-2 at 256-token subject in every seed. The recovered counts miss it at both sizes. The comparison holds the routing rule and training draw schedule fixed, not the routes learned separately by each trained arm. Parameter matching is not FLOP or serving-quality matching.

## Scope and verification

1,536 updates and 98,304 worlds per fit, four arms × three seeds at each of two widths. Identical initial hashes across arms within seed and width, and the reported training digest matches the independently built disjoint split audit. Six held cohorts (subject/relation at 64/128/256 tokens), each with 2,048 families. The predictions are next-value markers on constructed language, not ordinary text generation or real-world reasoning. No persistent memory.

- Kaggle source payload SHA-256: b08744a465061f6475615cc5d7c4b9bb141cf8ff11081fd6c7d3af424c5fc48d
- Failed native receipt SHA-256: 5e026d881ac584722af6a06265cec5b4ef9bf3115785ec998071113e7cfbcb5e
- Detached score SHA-256: ebb15002d09d181fdead4778141edd4ff1cc53a8cb81b95e7e827ed594fe3338
- Train/held split digest: fc8406bce78600e472b986760a3f34a32f1ee37233509468b6b7ba77fd2f6c82
