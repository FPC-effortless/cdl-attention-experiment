# CASM-X18R results — 2026-09-04

## Provenance

- preregistration: `215651017f29c71be5bd3beafde536c312dc1943`
- evaluated implementation head: `a0f8984649210f3729338a35aded4ac937c543e4`
- workflow: `33868415717`
- integrity gate: PASS, 53/53 tests
- all six train/evaluate/provenance jobs: PASS

Artifacts:

- seed `20261141`: artifact `9935178712`, sha256 `aeb8c5a8dc50469a6073fe7d195b959488cd69435073562816b658b423c82c97`
- seed `20261142`: artifact `9935289190`, sha256 `f1e351148273374350efdf751d55f0110f2c4811cd5017bce21c6711b1d25275`
- seed `20261143`: artifact `9935301434`, sha256 `d79cca6133c788cdf5ed6a29f4282ae4ea029ba4c2199a325fcb3e3964c0d253`
- seed `20261144`: artifact `9935298208`, sha256 `37899df0692d38328647b7bdcfcd0b8a5f8ff19eb48a98e211710dbbad69cf8e`
- seed `20261145`: artifact `9935291137`, sha256 `f06308a0841d465c875b4a1db9f36daead32708392d6db2b53341ca068d479a3`
- seed `20261146`: artifact `9935207548`, sha256 `ce8b049ee4f02f01593deab1d160caa5f80d567bf63b2cb9d3f1b851d67560ed`

The earlier head `f549317f6217c1a71df585fd94837118cf552f8d` is not scientific evidence. Its contract job failed before training because one float32 row-normalization assertion used `atol=1e-7`, tighter than the established `1e-5` binding contract. The superseding head changes only that integrity-test tolerance; no model, optimizer, data, seeds, or scientific condition changed.

## Frozen classification

**VALID EXPERIMENT. DETACHED-PRICE ROBUST REPAIR: FAIL. STRONG STABILIZATION: FAIL.**

The preregistered robust-repair threshold required `dual_detached_prices` to satisfy the complete seen-cardinality competence criterion on all 6/6 fresh seeds. It passes only 3/6 (`20261141`, `20261145`, `20261146`).

The matched `dual_fullgrad` control passes 4/6 (`20261142`, `20261143`, `20261145`, `20261146`). Therefore detaching the dual-price history does not improve robustness on this seed panel and cannot be adopted as a stabilization repair.

Both learned regimes remain seed-sensitive even though the integrity gate establishes that their forward prices and bindings are bit-identical for matched logits before optimization and that the only treatment difference is the backward path through iterative price history.

## Seed-level outcomes

| seed | full-gradient | detached-price |
| --- | --- | --- |
| 20261141 | FAIL | PASS |
| 20261142 | PASS | FAIL |
| 20261143 | PASS | FAIL |
| 20261144 | FAIL | FAIL |
| 20261145 | PASS | PASS |
| 20261146 | PASS | PASS |

Representative failure topology:

- seed `20261141`, full-gradient: n2 `[6,6]`, n3 `[6,6,5]`, n4 `[5,5,5,7]`; minimum hard/soft answer across seen cells 12.5%.
- seed `20261142`, detached-price: n4 `[4,2,0,0]`, one collision, row-max `0.8975`, minimum answer 25.0%.
- seed `20261143`, detached-price: n4 `[1,4,0,4]`, one collision, row-max `0.8701`, minimum answer 24.22%.
- seed `20261144`, both treatments fail: full-gradient n4 `[7,5,7,5]`; detached-price n4 `[4,1,1,1]`.

Passing seeds are exact under the frozen criteria: hard/soft answer-final 100% across every suite, deep step-state exactness 100%, deep hidden-register accuracy 100%, zero collisions, and final row-max at or effectively at 1.0.

## Supported conclusion

X18R rules out a simple backward-path explanation for the remaining optimization instability. Removing gradient history through the projected dual-price state changes optimization materially on individual seeds, but does not produce a reliable repair and is not superior to the full-gradient control across the six-seed panel.

Combined with X9-X18, this is the stopping point for the allocator-development line. The repeated pattern is now clear: collision penalties, occupancy state, capacity feedback, persistent prices, longer dual horizons, coordinate changes, and a backward-only price-history intervention can each alter optimization, but none yields a robust extensible constructor of computational state.

## Architectural pivot

The next research object is no longer `variable -> slot preference`. The next model-development phase should separate computational role from storage location and test recursive role construction directly.

Target decomposition:

`observations -> construct computational roles -> instantiate working state -> shared execution -> outcome`.

The first successor experiment should test a shared recursive role transition, e.g. `r_{i+1}=F_theta(r_i, context_i)`, against a parameter-matched static/nonrecursive role control. Training should expose only the first four generated roles and evaluation should require applying the same role transition beyond the training horizon.

Storage/allocation must be treated as an implementation bridge or separately controlled factor so failure of a legacy slot allocator cannot be mistaken for failure of recursive role generation. Resource state should be retained later as a cost/creation controller, not treated as the primary source of role identity.
