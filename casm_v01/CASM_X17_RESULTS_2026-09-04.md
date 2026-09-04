# CASM-X17 — Dual-horizon convergence falsifier — Results

## Classification

**VALID NEGATIVE RESULT.** Both trained learned regimes are fully competent on the trained cardinalities on all three preregistered seeds, so the frozen paired horizon comparison is eligible. Increasing the projected dual-price horizon from 8 to 64 rounds does **not** produce strong or partial cardinality generalization to unseen `n=5,6`. Re-evaluating the trained 8-round weights with 64 inference rounds and zero additional optimization also fails.

The supported conclusion is therefore narrow but decisive: under the frozen X17 allocator, data, executor, descriptors, objective, optimizer, and seeds, insufficient 8-round dual convergence is not the explanation for the persistent `n=5,6` allocation failure. This does not prove that no larger or different optimization procedure could help; it falsifies the preregistered 8-vs-64 horizon hypothesis.

## Frozen provenance

- preregistration: `36b2d08a4eadd5398bb56e3383fff30b4d5da848`
- evaluated implementation head: `be87495f9ec14f832fbc27b7a4d39b76fd4d324f`
- workflow: `33843674832`
- integrity gate: PASS
- all three train/evaluate/provenance jobs: PASS

Artifacts:

- seed `20261121`: artifact `9926143364`, digest `sha256:8c12f145c59fe8517a4ae65b050273c9f58de93c1e3a5149d17a50831c221d02`
- seed `20261122`: artifact `9926103357`, digest `sha256:c73e10d0454251491298436bfa692b3d24c5dc658fb423e7c904f96b4a183642`
- seed `20261123`: artifact `9926099891`, digest `sha256:cda4f3503c118b97b40006cd2405cb29622e784b5bbe1010f0330ffa8317254c`

## Positive-control validity

`canonical_functional` is exactly 100% on every seed, every `n=2..6`, and every evaluated depth through 96 for hard answer-final, deep step-state exactness, and hidden-register accuracy. X17 is therefore valid.

## Seen-cardinality competence

Every learned/evaluation horizon is fully competent on all three seeds for trained `n=2,3,4`:

- `dual_short_8`: hard/soft answer, deep step-state, and hidden-register metrics are 100%; exactly `n` unique hard slots; zero collisions; mean row max ~1.0.
- `dual_long_64`: same every-seed seen competence.
- `dual_short_eval64`: copied short-trained weights evaluated at 64 rounds, also same every-seed seen competence.

The horizon comparison is therefore not confounded by seen optimization failure.

## Unseen topology and worst capability

Metrics below are the minimum across the frozen suites for answer-final and across depths 24/48/96 for step-state exactness and hidden-register accuracy.

### Seed 20261121

| regime | n | hard assignment | unique/collisions | row max | hard answer / step / hidden | soft answer / step / hidden |
|---|---:|---|---|---:|---|---|
| `dual_short_8` | 5 | `[0,2,1,4,1]` | 4/1 | 0.901504 | 0.250000 / 0.027913 / 0.254567 | 0.203125 / 0.019043 / 0.242442 |
| `dual_short_8` | 6 | `[0,2,1,4,1,2]` | 4/2 | 0.999413 | 0.167969 / 0.003174 / 0.186076 | 0.167969 / 0.003255 / 0.186442 |
| `dual_long_64` | 5 | `[6,0,3,5,7]` | 5/0 | 0.745078 | 1.000000 / 1.000000 / 1.000000 | 0.242188 / 0.029460 / 0.256683 |
| `dual_long_64` | 6 | `[6,0,5,5,7,2]` | 5/1 | 0.722216 | 0.179688 / 0.013021 / 0.245085 | 0.199219 / 0.012777 / 0.239884 |
| `dual_short_eval64` | 5 | `[0,2,1,4,1]` | 4/1 | 0.900000 | 0.250000 / 0.027913 / 0.254567 | 0.195312 / 0.021444 / 0.247294 |
| `dual_short_eval64` | 6 | `[0,2,1,4,1,4]` | 4/2 | 0.791134 | 0.167969 / 0.002930 / 0.177970 | 0.144531 / 0.002116 / 0.170231 |

Seed 21 contains the strongest diagnostic edge case: the independently trained 64-round model reaches a collision-free, hard-exact `n=5` solution, but the soft binding is diffuse (`row max 0.745078`) and soft execution is far below threshold. At `n=6`, hard topology also collides. Under the frozen criteria this is not strong or partial generalization.

### Seed 20261122

| regime | n | hard assignment | unique/collisions | row max | hard answer / step / hidden | soft answer / step / hidden |
|---|---:|---|---|---:|---|---|
| `dual_short_8` | 5 | `[0,1,3,2,3]` | 4/1 | 0.899931 | 0.269531 / 0.028605 / 0.247803 | 0.230469 / 0.023112 / 0.246267 |
| `dual_short_8` | 6 | `[0,1,0,2,3,2]` | 4/2 | 0.966258 | 0.113281 / 0.002441 / 0.192961 | 0.109375 / 0.002686 / 0.191732 |
| `dual_long_64` | 5 | `[2,1,0,3,2]` | 4/1 | 0.705180 | 0.179688 / 0.027669 / 0.289703 | 0.179688 / 0.014160 / 0.239878 |
| `dual_long_64` | 6 | `[2,3,1,0,0,2]` | 4/2 | 0.833361 | 0.148438 / 0.002441 / 0.201571 | 0.156250 / 0.003255 / 0.193481 |
| `dual_short_eval64` | 5 | `[0,1,3,2,3]` | 4/1 | 0.866214 | 0.269531 / 0.028605 / 0.247803 | 0.242188 / 0.029826 / 0.250326 |
| `dual_short_eval64` | 6 | `[0,1,3,2,3,2]` | 4/2 | 0.818541 | 0.195312 / 0.001790 / 0.181323 | 0.148438 / 0.002930 / 0.198771 |

### Seed 20261123

| regime | n | hard assignment | unique/collisions | row max | hard answer / step / hidden | soft answer / step / hidden |
|---|---:|---|---|---:|---|---|
| `dual_short_8` | 5 | `[7,1,5,3,7]` | 4/1 | 0.911813 | 0.160156 / 0.022624 / 0.287008 | 0.171875 / 0.022949 / 0.255798 |
| `dual_short_8` | 6 | `[7,1,7,3,5,1]` | 4/2 | 0.841789 | 0.105469 / 0.000651 / 0.185848 | 0.125000 / 0.000326 / 0.176294 |
| `dual_long_64` | 5 | `[4,2,1,0,4]` | 4/1 | 0.863273 | 0.156250 / 0.023275 / 0.285350 | 0.179688 / 0.027466 / 0.272105 |
| `dual_long_64` | 6 | `[4,2,1,0,6,2]` | 5/1 | 0.752623 | 0.171875 / 0.017782 / 0.250578 | 0.144531 / 0.001139 / 0.164591 |
| `dual_short_eval64` | 5 | `[7,1,5,3,5]` | 4/1 | 0.859068 | 0.203125 / 0.028727 / 0.262299 | 0.195312 / 0.023926 / 0.245667 |
| `dual_short_eval64` | 6 | `[7,1,5,3,5,6]` | 5/1 | 0.746090 | 0.222656 / 0.016846 / 0.256795 | 0.156250 / 0.003743 / 0.180013 |

## Frozen classification

- `dual_short_8`: seen competent on every seed; strong unseen FAIL; partial unseen FAIL.
- `dual_long_64`: seen competent on every seed; strong unseen FAIL; partial unseen FAIL.
- `dual_short_eval64`: seen competent on every seed; strong unseen FAIL; partial unseen FAIL.

Because both trained horizons satisfy the seen prerequisite on every seed, the paired horizon conclusion is eligible: **64 projected dual rounds are insufficient to rescue the cardinality extrapolation failure under the frozen X17 system.**

Because the inference-only long view also fails, simply allocating more iterations to the already-trained 8-round preference function is likewise insufficient.

## Supported claim and next boundary

X13–X17 progressively removed weak collision gradients, absent occupancy information, myopic capacity response, absent persistent prices, and short price-iteration horizon as sufficient explanations. The remaining evidence now points to the learned variable→slot **preference representation**: it learns a high-quality allocation over the trained variable-descriptor region but does not reliably extend that preference geometry to newly appearing variable descriptors.

The next experiment should change the representation, not the allocator. A clean first falsifier is to replace the current cardinality-relative external descriptor with a fixed global external-coordinate descriptor while keeping the validated 8-round dual allocator, executor, objective, supervision, train/unseen split, and hard unrepaired evaluation unchanged. Success would support descriptor-frame shift as the missing factor; failure would motivate a genuinely procedural/recursive role generator.
