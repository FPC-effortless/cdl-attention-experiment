# CASM-X11 Results — 2026-09-01

## Provenance

- preregistration: `9b73389238e818f598fba5daeb89a059dffc4d44`
- evaluated implementation head: `ff00a0b56acd7d6919d9d43f39454ad40efa52bd`
- workflow: `33420802509`
- integrity gate: PASS
- train/eval seeds: `20261061/20261141`, `20261062/20261142`, `20261063/20261143`
- artifacts:
  - seed 61: `9769294201`, sha256 `346de241b8006ad8ed43fd3f1862496581fcda8defecdb57eabbec2ced8dcd27`
  - seed 62: `9769516581`, sha256 `acc5556d007d4b707fb609d7af81d7260d5c648febb68267fb9724b5e5125e28`
  - seed 63: `9769400421`, sha256 `706d483421f57132381110623f284ea9171f4adcc3b877da0c7a2882d1aad525`

## Frozen classification

**CASM-X11 does not pass.** The pairwise-overlap resource-competition prior is insufficient under the preregistered every-seed criteria.

The canonical functional positive control is exactly competent on every seed and every `n=2..6`, so X11 is valid for learned-binding interpretation.

### No-competition relational controls

Both no-competition relational regimes fail seen-cardinality competence on all three new seeds. This replicates X10's relational-collapse problem.

### Independent + overlap competition

- seed `20261061`: catastrophic seen-cardinality optimization failure; all rows collapse to one slot in representative unseen cases.
- seeds `20261062` and `20261063`: **exactly competent on seen `n=2,3,4`** in hard and soft execution, with sharp collision-free seen bindings.
- both competent seeds fail immediately on unseen allocation:
  - seed 62 `n=5`: `[3,5,5,4,2]` = 4 unique slots; `n=6`: `[3,5,5,5,2,2]` = 3 unique slots.
  - seed 63 `n=5`: `[7,6,2,3,7]` = 4 unique slots; `n=6`: `[7,6,2,3,7,6]` = 4 unique slots.

Thus competition sometimes repairs seen optimization but does not produce an extensible allocation rule. Because seed 61 fails seen competence, the preregistered anti-collapse support criterion is not met.

### Coordinated + overlap competition

- seed `20261061`: exact seen competence and exact unseen `n=5` execution through depth 96 with collision-free assignment `[7,5,4,2,3]`; at `n=6` it collapses to `[7,5,7,2,3,2]` and execution fails.
- seed `20261062`: seen-cardinality optimization failure; representative unseen bindings collapse to a single slot with row-max about `0.25`.
- seed `20261063`: seen-cardinality optimization failure; representative unseen bindings remain low-confidence/colliding with row-max about `0.25`.

The seed-61 `4 -> 5` rescue therefore does **not** replicate. Coordinated competition is not seen-competent on every seed, so no coordination-effect claim is eligible.

## Preregistered decision table

- positive-control prerequisite: **PASS**
- `relational_independent_no_competition` seen competence: **FAIL**
- `relational_coordinated_no_competition` seen competence: **FAIL**
- `relational_independent_competitive` seen competence on every seed: **FAIL**
- `relational_coordinated_competitive` seen competence on every seed: **FAIL**
- strong unseen-cardinality generalization: **FAIL for every learned regime**
- partial unseen generalization: **FAIL for every learned regime**
- overlap competition supported as robust anti-collapse mechanism: **NO**
- coordination effect after competition: **INELIGIBLE**

## Mechanistic interpretation

The result narrows the bottleneck from executor validity to allocation optimization.

A pairwise overlap penalty can help some seeds form sharp distinct bindings, but it is not a complete resource-allocation objective. It does not directly reward each variable for committing to a compact resource footprint, and symmetric/diffuse configurations can remain poor optimization states. Conversely, a sharp shared-slot collapse can still occur when answer-loss pressure dominates the finite overlap cost.

A useful successor should therefore retain the validated executor, answer-only supervision, row/slot permutation symmetry, and unrepaired hard evaluation while replacing overlap-only competition with a soft scarcity objective that simultaneously:

1. charges each variable for spreading mass across many slots; and
2. charges each slot for occupancy above unit capacity.

This remains an explicit structural/resource prior, not autonomous ontology discovery.
