# LRC-001 engineering pilot protocol

Written before the first fit. This is an engineering pilot, not a confirmatory
test or an extension of any frozen LRR/CASM experiment. Keep source/result
provenance; never classify a short smoke run as evidence for the hypothesis.

Hypothesis for a later confirmatory protocol: coupling lookup, ordered operator
conditioning and reference continuation improves previously unseen relational
composition versus independent routing under the same retrieval budget.

## Fixed pilot

- CPU, one thread; seeds 101, 102, 103; all failures retained.
- Coupled, independent, no-reference; identical seeded initial tensors and data.
- Width 24, 2 layers, 4 relation codes, 8 addresses, 2 reads, cap 2, local kernel 5.
- Raw bytes, random 4-entity function worlds, no parsed structure in model inputs.
- 20 AdamW updates at 0.002, batch size 2, gradient norm cap 1.
- One uniformly selected eligible decision field and one valid coordinate per
  update; one uniformly selected different category. Shared Gumbel noise replay,
  two full forwards per update. Pairwise credit weight 1, advantage clipped to 1.
- Training example seeds: `seed*1_000_000 + step*batch_size + row`.
- DEV example seeds 700000..700007 in each diagnostic condition. Paired conditions
  deliberately share latent worlds. No DEV tuning after observing the pilot.
- No confirmatory TEST data is opened. Future confirmation must preregister fresh
  model/world seeds, sample sizes, effect thresholds and all selection rules.

## Engineering acceptance

1. Unit/integration checks pass: strict causality, document isolation, padding,
   cap adherence, collision preservation, null behavior, ordered operands,
   actual reference influence, counterfactual write effects, common-noise replay,
   optimizer/RNG resume and equal allocated parameters.
2. Every arm completes finite optimization, writes checkpoint/optimizer/RNG,
   trace, source digests and all DEV metrics, including failures and controls.
3. Costs include every counterfactual forward, write, lookup, inspected candidate,
   reference gather and operator execution. Wall-clock training and evaluation
   are separate; no peak-memory or optimized throughput claim is made.

Lower DEV NLL is a diagnostic, not an acceptance requirement or promotion gate.
The pilot cannot qualify language competence, establish a causal treatment
effect, or justify expanding the architecture. Do not choose a winning seed.

## Future confirmatory design (not yet registered or executed)

Freeze source and training settings after development, then open fresh seeds and
worlds only once. Evaluate all arms at matched examples, parameter budget and
read cap; report the difference in actual training costs as well. Define minimum
competence before testing superiority. Use paired world/seed differences and
uncertainty intervals for answer loss and exact multi-byte generation. Add
separately trained local, qualified attention and SSM controls before any model
quality/efficiency claim. Report mean, dispersion and every collapsed seed.
