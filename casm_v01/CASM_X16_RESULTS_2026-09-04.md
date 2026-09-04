# CASM-X16 results — 2026-09-04

## Provenance

- preregistration: `7968a5685d5f7a5d3b0f534d6dbad09f73e5c7f9`
- evaluated implementation head: `2a74b92a149a69f17ffe5df9be8e67522b9cb0f8`
- workflow: `33836516813`
- integrity gate: PASS
- all three train/evaluate/provenance jobs: PASS

Artifacts:

- seed `20261111`: artifact `9923723703`, sha256 `a687652faefc2e0f4a5a040e39c263a08bb09e70ef0e8d4fbce7950e83e3eba8`
- seed `20261112`: artifact `9923711082`, sha256 `173003df7ccf0296877cc0d87838cfd2266fae3daeecabff0d90ed9678e28b62`
- seed `20261113`: artifact `9923719374`, sha256 `d0eafe7eef11bd77fa80c87ef8743529654f7ef43ed6aa45e85a65bf35797372`

## Frozen classification

**X16 is valid. Persistent dual-price state materially improves seen-cardinality optimization robustness, but it does not produce cardinality extrapolation under the frozen criteria. The preregistered causal extrapolation comparison is INELIGIBLE because `dual_neutral` fails seen competence on seeds 20261112 and 20261113.**

### Positive control

`canonical_functional` is exactly 100% for every seed, every `n=2..6`, every hard/soft evaluation suite through depth 96. The positive-control prerequisite passes.

### Seen-cardinality competence

`dual_priced` passes the complete seen prerequisite on **all three seeds** at `n=2,3,4`:

- hard and soft answer-final: 100% on every suite;
- hard and soft deep step-state exactness: 100%;
- hard and soft deep hidden-register accuracy: 100%;
- exactly `n` unique hard slots and zero collisions;
- final mean soft row max >=0.99994 on every seen cell.

Per-seed final priced hard assignments:

- seed 20261111: n2 `[0,2]`, n3 `[0,2,1]`, n4 `[0,2,1,3]`
- seed 20261112: n2 `[7,1]`, n3 `[7,4,1]`, n4 `[7,4,1,0]`
- seed 20261113: n2 `[2,7]`, n3 `[2,5,7]`, n4 `[2,5,4,7]`

This descriptively improves the seen-competence count from X15's 2/3 capacity-conserving seeds to **3/3** new X16 priced seeds. This is not a paired causal comparison across experiments.

`dual_neutral` does not satisfy the seen prerequisite on every seed:

- seed 20261111 is exactly competent at n2/n3/n4;
- seed 20261112 collapses at seen n3/n4, with IID answer-final 24.61% / 16.02%;
- seed 20261113 collapses at seen n3/n4, with IID answer-final 25.39% / 15.23%.

Thus the frozen dual-price causal extrapolation comparison is ineligible even though the priced treatment is robustly seen-competent.

### Unseen n=5,6

`dual_priced` fails both strong and partial unseen criteria on every seed. Hard collisions remain and execution collapses far below thresholds.

Seed 20261111:

- n5 assignment `[0,2,1,3,1]`, 4/5 unique, 1 collision, row-max 0.9092;
  - worst hard answer 25.39%, deep step 2.99%, deep hidden 24.72%;
  - worst soft answer 20.31%, deep step 2.15%, deep hidden 24.10%.
- n6 assignment `[0,2,1,3,1,2]`, 4/6 unique, 2 collisions, row-max 0.9971;
  - worst hard answer 17.97%, deep step 0.13%, deep hidden 17.81%.

Seed 20261112:

- n5 assignment `[7,4,7,1,1]`, 3/5 unique, 2 collisions, row-max 0.8728;
  - worst hard answer 16.02%, deep step 0.55%, deep hidden 20.28%.
- n6 assignment `[7,4,4,1,7,1]`, 3/6 unique, 3 collisions, row-max 0.8723;
  - worst hard answer 10.55%, deep step 0.10%, deep hidden 15.42%.

Seed 20261113:

- n5 assignment `[2,5,5,7,7]`, 3/5 unique, 2 collisions, row-max 0.9070;
  - worst hard answer 19.92%, deep step 0.60%, deep hidden 18.65%.
- n6 assignment `[2,5,5,7,7,6]`, 4/6 unique, 2 collisions, row-max 0.8217;
  - worst hard answer 14.45%, deep step 0.18%, deep hidden 17.29%.

The generated price state is active rather than degenerate: maximum observed training dual price was approximately 13.63, 11.49, and 8.00 for seeds 11, 12, and 13 respectively. Nevertheless the final unseen allocations remain colliding.

## Supported claim

Within this controlled supplied-ontology benchmark, an eight-round unit-capacity projected primal-dual state substantially stabilizes answer-supervised allocation on the trained cardinality range: the priced treatment is exactly competent on n=2,3,4 for all three new seeds while the matched zero-price control is not. However, persistent global prices are **not sufficient** to extend the learned allocation rule to unseen n=5,6.

The result narrows the remaining bottleneck: it is no longer simply lack of resource competition, weak collision gradients, missing occupancy state, or absence of persistent global capacity state. The learned variable→slot preference function still fails to generate an extensible allocation for newly appearing variable descriptors.

## Next falsifier

Before changing the learned representation, the next clean experiment should distinguish **finite dual-convergence horizon** from **preference-function extrapolation failure**. Compare the frozen 8-round allocator with a much longer fixed dual horizon from identical learned initialization/training conditions. If the longer horizon remains colliding, the evidence shifts strongly toward the preference representation rather than insufficient constraint-iteration depth. No claim should move to active-cardinality or variable-identity inference unless an every-seed unseen PASS is obtained.
