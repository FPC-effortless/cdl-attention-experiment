# CASM-X9R2 results — 2026-08-31

## Frozen provenance

- preregistration: `067f6481ab80ec1a8a5f4b1db7bef58d6a1e28f9`
- evaluated implementation head: `91b5b3920e4f5921caf89b911452d419bcdb561b`
- workflow run: `33395984723`
- integrity gate: PASS
- all three train/evaluate jobs: PASS

Artifacts:

- seed `20261041`: artifact `9759609986`, sha256 `995bbea67963fdd24434717982b5611aff2af8047ffd57b1ba4f3bac77f94474`
- seed `20261042`: artifact `9759623742`, sha256 `08e26c051829f74d05ad3b841371a6bf1502ead9db993edef62f0d01e77035d1`
- seed `20261043`: artifact `9759612199`, sha256 `f2fc7954c3845d788091433b798b3cb92655849df694977dad59eec4425c9ef7`

## Frozen classification

**CASM-X9R2 is a strict Validity PASS.**

For the decisive `cosine_decay_stable` treatment, every new seed and every cardinality `n=2..6` achieved:

- hard answer-final accuracy: 100% on every suite;
- hard step-state exactness: 100% at every evaluated depth, including 24/48/96;
- hard hidden-register accuracy: 100% at every evaluated depth, including 24/48/96.

Thus all frozen validity thresholds are satisfied without averaging or exception.

The paired `fixed_lr_replication` also achieved 100% on every evaluated cell for all three new seeds.

## Per-seed minima across all cardinalities and suites

| seed | regime | min answer-final | min deep step-state exact | min deep hidden-register |
| --- | --- | ---: | ---: | ---: |
| 20261041 | fixed LR | 100.00% | 100.00% | 100.00% |
| 20261041 | cosine | 100.00% | 100.00% | 100.00% |
| 20261042 | fixed LR | 100.00% | 100.00% | 100.00% |
| 20261042 | cosine | 100.00% | 100.00% | 100.00% |
| 20261043 | fixed LR | 100.00% | 100.00% | 100.00% |
| 20261043 | cosine | 100.00% | 100.00% | 100.00% |

## Optimization-stability diagnostic

The preregistered descriptive stability hypothesis required cosine to have both lower/equal post-step-4000 maximum pre-clipping gradient norm and lower/equal final training loss on **every** paired seed.

| seed | fixed max grad | cosine max grad | fixed final loss | cosine final loss | stability condition |
| --- | ---: | ---: | ---: | ---: | --- |
| 20261041 | 260.83 | 115.09 | 4.29e-5 | 2.21e-5 | PASS |
| 20261042 | 58.00 | 230.56 | 2.31e-6 | 7.49e-6 | FAIL |
| 20261043 | 429.61 | 22.97 | 1.80e-5 | 8.31e-6 | PASS |

Therefore the **cosine-specific stability hypothesis is not supported on every seed**.

The correct interpretation is:

1. the local-equivariant executor/training setup is capable of strict, seed-replicated cardinality transfer on the new preregistered seeds;
2. cosine decay is not required for that result, because the fixed-LR replication is also exact everywhere;
3. the earlier X9R cross-seed degradation is better treated as seed-sensitive optimization variance than as a necessary consequence of absolute slot identity or a proven learning-rate pathology;
4. the frozen X9R2 successor condition is nevertheless met because the decisive cosine treatment passes every validity cell.

## Successor authorization

The X9R2 preregistration states that only a strict Validity PASS authorizes learned-binding work. That condition is now satisfied.

The next experiment should keep this validated local-equivariant executor and compare:

- X9-style independent descriptor-to-row binding generation; versus
- a coordinated/permutation-equivariant set binding generator.

The purpose is to isolate whether cross-variable coordination is required to allocate unseen variables to distinct internal slots when cardinality increases beyond training.

## Claim boundary

X9R2 validates only the executor/training substrate for the next binding experiment. It does not establish learned binding, cardinality inference, ontology discovery or open-ended reusable computation.
