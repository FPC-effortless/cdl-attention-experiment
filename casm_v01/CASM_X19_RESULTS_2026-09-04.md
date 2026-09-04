# CASM-X19 results — 2026-09-04

## Provenance

- base/frozen X18R result: `a1ebf25a560b2156fd9bc0013ea280763fe054eb`
- preregistration: `426df8dec82e900d75d0dd701c5021e4146ab47d`
- authorized evaluated head: `611c0cecf4c2772b2ea258de65aed002c28e8a76`
- workflow: `33883985700`
- integrity gate: PASS
- all six train/evaluate/provenance jobs: PASS

Artifacts:

- seed `20261151`: artifact `9941436791`, sha256 `ae23db00bf8b9130e0f0b9dc25010316ed212dea08ee9d8b3a83f73a4ad05716`
- seed `20261152`: artifact `9941442002`, sha256 `251ee668cc36a2a856cfd62bdbb363d752b2cab886c2ee5607f4780cc28539c9`
- seed `20261153`: artifact `9941567987`, sha256 `c18cea9894942840b121e91abf6d86b6f71f58254c7c214e27ddf47ec638322d`
- seed `20261154`: artifact `9941556424`, sha256 `3f51d56ebc96bec49446dbd41672612ee6f12e38350a20854f8cf9b39f661eb8`
- seed `20261155`: artifact `9941542190`, sha256 `86f3ae1542cc612b92b07db35bc925c3f5bd4b2d14adf41a56b6ca48f78da97c`
- seed `20261156`: artifact `9941479706`, sha256 `51f9a44e5b615e3f233901a472e85df8c168ce0c326ffcceed773d87b4e84fdf`

Two earlier heads are pre-training non-evidence. `a786dd0c781eec4d486efd1e7b146bbe9b2e3fb2` and `4fe22e53764c058a47ad8507a49cafeac180279b` were rejected by integrity assertions before any seed training. The superseding head changes only those test assertions; the preregistration, model, runner, optimizer, data, seeds, role recurrence and storage bridge are unchanged.

## Frozen classification

**VALID EXPERIMENT. POSITIVE CONTROL: PASS. SIX-SEED PAIRED RECURSIVE-vs-STATIC COMPARISON: INELIGIBLE. STRUCTURAL EXTENSION: NOT ESTABLISHED.**

The positive-control prerequisite passes on every seed and every `n=2..6` cell.

The paired causal comparison is ineligible because both learned regimes were preregistered to require full seen-role competence on all six seeds, but each has one seen-role failure:

- `static_global_roles`: seen-role competence passes 5/6 seeds and fails seed `20261152` at `n=4`; hard and soft stress-depth-96 answer-final accuracy are `98.4375%`, below the frozen `99%` threshold. Its hard storage topology is nevertheless collision-free and sharp: `[3,0,2,1]`, four unique slots, row-max `0.999994`.
- `recursive_roles`: seen-role competence passes 5/6 seeds and fails seed `20261155`; at `n=4` the learned hard topology is `[4,7,5,5]`, only three unique slots with one collision. Minimum hard answer-final is `26.9531%` and minimum soft answer-final is `28.9062%`, so this is a preregistered seen optimization failure, not a near-threshold miss.

Because the six-seed seen prerequisite is not met, no causal claim that recurrence is better or worse than the static control is authorized.

## Seed-level seen competence

| seed | static global roles | recursive roles |
| --- | --- | --- |
| 20261151 | PASS | PASS |
| 20261152 | FAIL | PASS |
| 20261153 | PASS | PASS |
| 20261154 | PASS | PASS |
| 20261155 | PASS | FAIL |
| 20261156 | PASS | PASS |

## Unseen-role diagnostics

Although neither learned regime is structural-extension-eligible under the frozen rules, the unseen behavior is diagnostic and highly consistent.

For `recursive_roles`, every seed fails far below the preregistered partial-extension boundary at unseen `n=5,6`. On the five seeds that are fully seen-competent, `r0..r3` map to four distinct executable storage identities, but `r4` immediately reuses an existing hard storage identity and `r5` reuses another existing identity or the same attractor. Representative hard assignments:

| seed | n=4 | n=5 | n=6 |
| --- | --- | --- | --- |
| 20261151 | `[0,3,2,1]` | `[0,3,2,1,1]` | `[0,3,2,1,1,1]` |
| 20261152 | `[0,3,2,1]` | `[0,3,2,1,1]` | `[0,3,2,1,1,1]` |
| 20261153 | `[5,7,6,3]` | `[5,7,6,3,3]` | `[5,7,6,3,3,3]` |
| 20261154 | `[5,7,6,4]` | `[5,7,6,4,4]` | `[5,7,6,4,4,4]` |
| 20261155 | `[4,7,5,5]` | `[4,7,5,5,5]` | `[4,7,5,5,5,5]` |
| 20261156 | `[7,3,4,6]` | `[7,3,4,6,4]` | `[7,3,4,6,4,6]` |

Across recursive seeds, unseen minimum answer-final accuracy is typically only about `15–23%`, deep step-state exactness falls to single-digit percentages, and the hard topology has fewer unique storage identities than active roles. Soft execution is similarly poor, so the failure is not caused only by hard argmax discretization.

The post-training role diagnostics show that this is not merely an allocator-style collision. The generic residual recursive cell itself tends to contract after the trained recurrence horizon. Consecutive role cosine similarities after approximately `r3` rise toward an attractor:

- seed `20261151`: `r3→r4 0.9797`, then `0.9925`, `0.9923`, `0.9952`;
- seed `20261152`: `0.9907`, `0.9853`, `0.9826`, `0.9722`;
- seed `20261153`: `0.9490`, `0.9661`, `0.9842`, `0.9906`;
- seed `20261154`: `0.9376`, `0.9988`, `0.9995`, `0.9999`;
- seed `20261155`: `0.9836`, `0.9989`, `0.9997`, `1.0000`;
- seed `20261156`: `0.8825`, `0.8969`, `0.9077`, `0.9079`.

Minimum pairwise cosine distance through `r7` is correspondingly tiny for the recursive treatment (`3.6e-05` to `9.4e-03` across seeds), while the static global-coordinate control remains much more separated (`0.0539` to `0.1390`). This supports a constructor-dynamics diagnosis: repeated application of the unconstrained residual role cell generally enters a fixed-point/low-change regime beyond the role horizon trained by task loss.

The static control reveals a second independent boundary. It also usually maps unseen roles back into the four storage identities learned on `n<=4`, despite receiving a direct global index coordinate. Seed `20261153` is the informative exception: at `n=5` its hard assignment is collision-free `[6,0,3,1,4]` and hard execution is exact, but mean row-max is only `0.890991` and soft answer-final falls to `29.2969%`; at `n=6` it collides. Thus the shared role-to-storage bridge is itself not a robust extrapolator to unused physical storage identities.

## Supported conclusion

X19 does **not** establish recursive structural extension. The paired recursive-vs-static causal comparison is formally ineligible because each learned regime misses the six-seed seen prerequisite once.

The diagnostic evidence nevertheless narrows the architectural problem substantially:

1. the generic recursive role cell is not horizon-stable; beyond the trained role positions it usually contracts toward an attractor, so new recursive applications do not reliably create new computational identities;
2. the fixed physical storage bridge is an additional extrapolation bottleneck even for a nonrecursive global-coordinate role representation;
3. these failures are not evidence to resume dual-price, occupancy, collision-allocator, matching or other allocator tuning.

The next model-development experiment should therefore change the **constructor representation/dynamics** while separately preventing the legacy fixed-slot bridge from dominating the scientific conclusion.

## Successor boundary

Do **not** advance directly to the preregistered X20 dynamic-state-instantiation milestone, because X19 recursive roles did not pass strongly.

The immediate successor should be a narrow constructor-dynamics experiment. A strong candidate is a parameter-shared **noncontractive / approximately norm-preserving recursive role transition** compared against the X19 unconstrained residual cell, with explicit long-horizon role-separation diagnostics and perturbation tests. Storage must remain a controlled bridge or be neutralized in a way that does not hand the model a correct variable-to-storage lookup.

The question should be whether structural recurrence can remain generative under repeated application, not which allocator can force another slot to be used.
