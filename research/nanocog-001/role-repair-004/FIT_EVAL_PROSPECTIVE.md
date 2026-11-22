# ROLE-REPAIR-004 prospective bounded fit and replay contract

Known weak frozen ROLE-002 backbone seed 1371, final endpoint 1280,
quarter-prefix/clip1, SHA256
`517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890`.
Native baseline is the exact zero-adapter replay; no native refit.
Do not claim transfer to unseen backbone seeds. Both actors see only
literal query and causal local features as defined for their respective
arms; freeze all native tensors and train the added 552 only.

Compare tap-content and literal-only controls for each of three
new adapter seeds 1471, 1571, 1671. Pair exact adapter initial tensors,
data/schedule, update budget, hardware, final-gate mask, and objective.
Use exactly the original 320 TRAIN families and derive canonical sibling
pairs by assignment flip. 1,280 updates, one family (8 worlds) per update,
save at 128 and 1,280 from same trajectories, no tuning or checkpoint
selection. AdamW 0.001 peak, 8-step linear warmup, 0.01 weight decay,
maximum L2 norm 1.0. Loss = answer cross-entropy over all 287 IDs plus
same paired softplus on answer versus foil. The parameter-free earlier
prefix logits remain unchanged and provide no adapter gradient. Count
actual updates and exact sampling geometry. No retrospective oracle
selected gate labels, no task labels in actor, no use of current
evaluation panel during tuning. If TrainLab admission or an invariance
check fails, stop once and preserve failure; no silent retry.
Budget: one CPU thread, 180s max/run, one research fit per arm/seed.

## Locked panel and decision

Fresh panel: generated before any fit with RNG20261127 from 120 exact
joint combinations after excluding all 712 previously used families.
960 prompts across 120 families, 12 of them with unseen-in-TRAIN name pairs at the family
level and 108 with familiar name pairs. Both strata use familiar
individual vocabulary and syntax. Separate actor and truth projections.
Hashes:
- Actor `6915a754aa598d5608f0041850b3897ecfeef0bd8731835815e24333ba6c98d1`
- Judge `0a3ae7b6d30ed4554b908c44411cb3e9e74c49e1942ef93c5fe8fad5b5005c23`

Primary at 1,280: ≥912/960 strict worlds and ≥108/120 all-eight families
in each adapter seed; ties fail. Native comparison, 128 endpoint, query
flip and clause order, two-color diagnostic, and 108/12 name-pair strata
are reported separately. For an interpretation that extra tap content
helps, require the content arm to pass both primary floors in every
seed and beat its paired literal-only arm on strict both-query histories
in each; no material first-clause regression vs original native
(first-clause strict accuracy no more than 5 worlds below native of 480)
in any seed. If both pass, report both; a preference requires the explicit
paired contrast, not model size or a pooled aggregate. If neither passes,
state tested scope, retain outputs and do not relax thresholds.
No holdout-general language, memory, or serving cost claim either way.

No fit or prediction is admitted solely by this document. Exact closure
and independent review are required before each stage and all outputs
must be sealed before an independent, weight-free score. No paid compute,
protected TEST, old MAIN, or main/local research promotion.
