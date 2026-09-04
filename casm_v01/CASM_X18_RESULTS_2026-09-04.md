# CASM-X18 results — 2026-09-04

## Provenance

- preregistration: `a3f849d41a56660ece9d4d8a39eb63b27176b583`
- evaluated implementation head: `f1a5478c5b41fb4aea4ff73fe7fdbcf96b1c3ddd`
- workflow: `33866452552`
- integrity gate: PASS
- all three train/evaluate/provenance jobs: PASS

Artifacts:

- seed `20261131`: artifact `9934456928`, sha256 `36c09bdeec6660a77de8267b95d3e6601eb04d9e19e955d3b5b792ec99a5d080`
- seed `20261132`: artifact `9934553826`, sha256 `d6615c1e95e7089cf6066b8259b3f35617421fa99a6cd0084f5bf55fff6f4d05`
- seed `20261133`: artifact `9934548813`, sha256 `429f4a58b162c1bd820ec08335cf74ea95e2e89a0d676d1c967d0942d513fa16`

## Frozen classification

**VALID EXPERIMENT; PAIRED DESCRIPTOR-FRAME EXTRAPOLATION COMPARISON INELIGIBLE.**

The deterministic positive control satisfies the full frozen prerequisite on every seed and every `n=2..6`. The X18-specific integrity gate passes and all evidence is bound to the preregistered exact implementation head.

However, neither learned treatment satisfies seen-cardinality competence on every seed:

- `relative_descriptor` is seen-competent on seeds `20261132` and `20261133`, but fails seen `n=4` on seed `20261131`.
- `global_descriptor` is seen-competent on seed `20261131`, but fails seen `n=4` on seed `20261132` and collapses already on seen `n=2,3,4` on seed `20261133`.

The preregistered causal descriptor-frame comparison is therefore **INELIGIBLE**. No claim that the global coordinate frame improves or harms cardinality extrapolation is authorized from this run.

## Positive control

`canonical_functional` is exactly competent throughout the evaluation grid for all three seeds. The positive-control prerequisite passes.

## Seen-cardinality results

### Seed 20261131

`relative_descriptor`:

- n2 assignment `[6,5]`, zero collisions, all hard/soft capability metrics exactly 100%.
- n3 assignment `[5,7,6]`, zero collisions; worst deep step-state exactness 99.8291%, worst deep hidden-register accuracy 99.9308%, answer-final 100%.
- n4 assignment `[7,7,6,7]`, only 2/4 unique slots with 2 collisions despite row-max 1.0; worst hard/soft answer-final 12.8906%, deep step-state exactness 2.9785%, hidden-register accuracy 23.7440%.

This is an optimization failure under the frozen rule because seen IID depth-8 answer is far below 80%.

`global_descriptor` is exactly competent at n2/n3/n4 on this seed, with assignments `[7,4]`, `[7,4,5]`, `[7,4,5,1]`, zero collisions, row-max 1.0, and all hard/soft capability metrics 100%.

### Seed 20261132

`relative_descriptor` is exactly competent at n2/n3/n4 with assignments `[7,0]`, `[7,4,0]`, `[7,4,6,0]`, zero collisions, and all hard/soft capability metrics 100%.

`global_descriptor`:

- n2 assignment `[1,0]`, zero collisions, exactly competent.
- n3 assignment `[1,0,2]`, zero collisions, but worst answer-final is 98.0469%, below the 99% seen threshold.
- n4 assignment `[1,0,2,0]`, only 3/4 unique slots with 1 collision and row-max 0.8754; worst hard answer-final 27.3438%, soft answer-final 26.9531%, hard deep step-state 5.2083%, soft deep step-state 4.2643%, hard deep hidden 26.0037%.

Thus the global treatment fails seen competence and the n4 cell is an optimization failure.

### Seed 20261133

`relative_descriptor` is exactly competent at n2/n3/n4 with assignments `[6,5]`, `[6,5,4]`, `[6,5,7,4]`, zero collisions, and all hard/soft capability metrics 100%.

`global_descriptor` collapses across the full seen range:

- n2 `[6,6]`, 1/2 unique, worst answer-final 19.1406%.
- n3 `[6,6,6]`, 1/3 unique, worst answer-final 17.1875%.
- n4 `[6,6,6,6]`, 1/4 unique, worst answer-final 13.6719%, deep step-state 1.2207%, deep hidden 14.0408%.

All of these are frozen optimization failures.

## Unseen-cardinality diagnostics

Because the every-seed seen prerequisite fails, unseen results are **descriptive diagnostics only** and cannot be used for the paired causal descriptor-frame classification.

The two relative-descriptor seeds that are fully seen-competent still reproduce the earlier aliasing boundary:

- seed 20261132: n5 `[7,4,7,0,0]` (3/5 unique), n6 `[7,4,7,0,0,0]` (3/6 unique); unseen capability is far below partial thresholds.
- seed 20261133: n5 `[6,5,7,4,6]` (4/5 unique), n6 `[6,5,7,7,4,4]` (4/6 unique); unseen capability is far below partial thresholds.

The one global-descriptor seed that is fully seen-competent also fails immediately when new variables appear:

- seed 20261131 n5: `[7,4,5,1,7]`, 4/5 unique, 1 collision, row-max 1.0; worst hard/soft answer-final 14.4531%, deep step-state 3.1331%, deep hidden 28.7506%.
- seed 20261131 n6: `[7,4,5,1,7,4]`, 4/6 unique, 2 collisions, row-max 0.999974; worst hard/soft answer-final 12.8906%, deep step-state 0.1628%, deep hidden 20.4818%.

This descriptively shows the cardinality-invariant coordinate by itself does not prevent the learned scorer from reusing roles already occupied by trained variables. It is not an every-seed extrapolation claim because the treatment fails seen competence on the other two seeds.

## Optimization diagnosis

The failures are not explained by a harness mismatch:

- the relative X18 path is tested to be numerically identical to the X16 priced path when weights match;
- relative/global treatments start bit-identically and are parameter matched;
- both use the same batches, objective, executor, eight-round dual allocator, optimizer and learning-rate schedule;
- the positive control is exact on every seed;
- the X18 integrity suite passes.

The failure trajectories instead show seed-sensitive optimization instability in the learned preference/dual system.

For seed 20261131, the relative treatment remains at an n4 one-collision allocation `[5,7,6,6]` through step 4500, then undergoes a large gradient event (pre-clip norm approximately 99.2 around step 5400) and settles into `[7,7,6,7]`; the n4 max dual price reaches 16 and the branch does not recover.

For seed 20261132, the global treatment remains colliding at n4 throughout training and finishes `[1,0,2,0]`.

For seed 20261133, the global treatment collapses by roughly step 3000 to `[6,6,6,6]`; the n4 dual price reaches 24 and the collision barrier saturates at approximately 6.9078, after which the branch remains trapped.

This indicates that before interpreting descriptor-frame extrapolation or moving to a more expressive role generator, the training path needs a robustness repair.

## Claim correction relative to X16

X16's statement that the priced treatment was seen-competent on all 3/3 of its new seeds remains true for that frozen seed panel. X18 provides a fresh three-seed replication of the exact relative/X16 learned path and obtains only 2/3 seen-competent seeds. Therefore the broader claim should be narrowed:

> Eight-round projected dual prices can make the learned allocation system exactly competent on the trained n=2,3,4 range, but that competence is not yet robust across random seeds.

Do not describe X16's 3/3 result as established seed-robust optimization without this qualification.

## Next experiment

Do **not** move directly to the recursive-role hypothesis from X18. The preregistration requires optimization diagnosis when either descriptor treatment fails seen competence.

The clean next falsifier should isolate the backward path through the iterative dual solver while preserving its forward allocation exactly:

- full-gradient unrolled dual-price control;
- stop-gradient dual-price treatment in which the eight forward price updates are numerically identical but price-state history is detached from backpropagation before the final binding gradient is applied.

This directly tests whether the seed instability is caused by differentiating through the eight projected dual iterations rather than by the forward resource-allocation mechanism itself. Only after a reliably seen-competent base is recovered should the experiment move to a recursive/procedural role generator for n=5,6 extrapolation.
