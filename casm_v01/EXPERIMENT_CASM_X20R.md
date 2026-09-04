# CASM-X20R — straight-through discrete state instantiation repair

## Status and purpose

CASM-X20R is preregistered before implementation or execution.

Frozen X20 result base:

`3225172c78ca44ad57a26d64b13ae24f122b96bb`

X20 is valid but not a robust state-instantiation PASS. The graph-conditioned constructor is exactly correct on hard/soft execution and state selection for 3/6 seeds across both trained live cardinalities 2/3/4 and unseen 5/6. On the other 3/6 seeds, distractor gates still collapse to approximately zero and soft execution remains near-perfect, but required live gates drift to fractional values and fail the frozen hard `g>=0.5` existence contract.

X20R tests the narrow mechanistic hypothesis implied by that evidence:

> the dominant X20 failure is a continuous-relaxation loophole, where fractional records can retain executable signal while lowering the storage penalty, rather than inability of the graph constructor to identify causally live candidates.

X20R changes only the learned gate used in the **training forward pass**. It does not change the constructor architecture, graph, executor, dataset, supervision, storage coefficient, optimizer, train/unseen cardinalities, hard threshold, or evaluation contract.

## Question

> If learned record existence is forced to be discrete in the training forward pass while retaining a straight-through gradient, does the X20 graph-conditioned constructor become robustly competent across fresh seeds and extend state instantiation from live cardinalities 2/3/4 to unseen 5/6?

## Frozen base substrate

Inherit exactly from X20:

- 8 supplied candidate entities;
- version-correct temporal dependency graph;
- shared reverse-topological graph constructor;
- deterministic nonlearned candidate identity codes;
- local-equivariant gated executor;
- final-answer-only task supervision;
- no live-mask/cardinality/intermediate/hidden/role/address labels;
- `lambda_storage = 0.05`;
- train live cardinalities `{2,3,4}` only;
- unseen live cardinalities `{5,6}`;
- train depth 12;
- evaluation depths 12/24/48/96;
- batch size 128;
- AdamW, 12,000 steps, cosine LR `2e-3 -> 2e-4`, weight decay `1e-4`, gradient clip `1.0`;
- hard evaluation threshold `g_soft >= 0.5`;
- no top-k, cardinality repair, matching, rerouting, or hidden oracle.

The X20 generator and its destructive-write/version-liveness regressions are frozen unchanged.

## Decisive intervention: deterministic straight-through existence

Let the constructor emit the same raw sigmoid probability as X20:

`g_soft = sigmoid(logit)`.

Define the hard forward existence decision:

`g_hard = 1[g_soft >= 0.5]`.

The straight-through training gate is:

`g_ST = g_hard + g_soft - stop_gradient(g_soft)`.

Therefore:

- forward value of `g_ST` is exactly `0` or `1`;
- backward derivative follows `g_soft`;
- training execution cannot exploit fractional record occupancy;
- training storage cost counts the hard forward record set while retaining a gradient through `g_soft`.

For straight-through learned regimes:

`L = L_answer(execute(g_ST)) + 0.05 * mean(g_ST)`.

No entropy, polarization, cardinality, gate-label, or live-mask auxiliary loss is added.

## Frozen regimes

1. `canonical_live_mask` — unchanged X20 positive control.
2. `all_records` — unchanged X20 all-eight-record control.
3. `soft_x20_replication` — exact X20 graph-conditioned learned constructor using raw soft gates for training execution/storage. This is the matched continuous-relaxation comparator.
4. `straight_through_instantiation` — parameter-identical graph-conditioned constructor using `g_ST` for training execution/storage.
5. `straight_through_structure_blind` — parameter-identical structure-blind constructor using the same straight-through rule.

For each seed, all three learned regimes must start from bit-identical compatible parameters and share the exact same training batches. The only graph-treatment difference between `soft_x20_replication` and `straight_through_instantiation` is soft vs straight-through training gate. The structure-blind regime removes dependency/version connectivity exactly as in X20.

## Fresh seeds

Use six fresh paired seeds, fixed before implementation:

- train `20261181`, eval `20261261`;
- train `20261182`, eval `20261262`;
- train `20261183`, eval `20261263`;
- train `20261184`, eval `20261264`;
- train `20261185`, eval `20261265`;
- train `20261186`, eval `20261266`.

No seed replacement, selective rerun, or adaptive hyperparameter change is permitted.

## Integrity requirements

Before scientific training, tests must establish:

1. the branch base is exactly frozen X20 result `3225172c78ca44ad57a26d64b13ae24f122b96bb`;
2. the X20 generator/data contract is unchanged;
3. `soft_x20_replication` is numerically identical to X20 learned-instantiation training/evaluation when weights and batches match;
4. `straight_through_instantiation` and `soft_x20_replication` have identical parameter counts and bit-identical initial state for each seed;
5. raw `g_soft` values are identical between matched soft/ST graph models before optimization;
6. straight-through forward gate values are exactly binary `{0,1}`;
7. straight-through forward record existence is exactly the frozen `g_soft>=0.5` decision with no top-k/count repair;
8. the straight-through storage term's forward value is exactly the hard record fraction;
9. nonzero gradients from answer loss reach the graph constructor through `g_ST` on falsifier batches;
10. nonzero storage gradients reach raw gate logits through `g_ST` when task loss is held fixed;
11. learned regimes receive no live mask, active cardinality, final target value as constructor input, hidden trajectory, or causality label;
12. no learned per-candidate gate/ID table is introduced;
13. structure-blind ST receives the same candidate/program observations except dependency/version connectivity propagation;
14. hard evaluation remains raw `g_soft>=0.5` and is not changed by the training intervention;
15. all gates/losses/states abort on non-finite values;
16. X20 positive-control executor regressions remain exact through n=2..6 and depth 96.

## Frozen classification

### Positive-control validity

`canonical_live_mask` must satisfy every seed and `n=2..6`:

- hard answer-final >=99% every suite;
- hard deep step-state exactness >=95%;
- hard deep live-register accuracy >=99%.

Failure invalidates X20R.

### Straight-through seen competence

`straight_through_instantiation` is extension-eligible only if **every 6/6 seed** at trained live `n=2,3,4` satisfies:

- hard and raw-soft answer-final >=99% every suite;
- hard and raw-soft deep step-state exactness >=95%;
- hard and raw-soft live-register accuracy >=99%;
- hard existence precision >=0.95;
- hard existence recall >=0.95;
- mean absolute hard record-count error <=0.25;
- mean raw soft gate on live candidates >=0.90;
- mean raw soft gate on distractors <=0.10.

Any hard IID/base answer-final <80% on a seen cardinality is an optimization failure.

### Strong state-instantiation extension

A strong X20R PASS requires every straight-through seed at unseen live `n=5,6` to satisfy the same capability and state-selection thresholds. No averaging rescues a failed seed/cell.

### Partial extension

If strong fails but every unseen straight-through capability cell has hard/raw-soft answer >=90%, deep step/live-register >=80%, and hard existence F1 >=0.80, classify partial and report the exact boundary.

### Continuous-relaxation repair effect

The straight-through intervention receives a **causal repair PASS** if:

1. `straight_through_instantiation` passes full seen competence on 6/6 and strong unseen extension on 6/6; and
2. the paired `soft_x20_replication` fails the same every-seed robustness criterion or exhibits at least one seed with the X20 fractional-hard/soft mismatch (raw-soft capability >=99% on seen cells while hard existence/capability fails).

Because starts, parameters, batches, objective coefficient and architecture are matched, this would support the bounded claim that eliminating fractional forward occupancy repairs the X20 training-relaxation failure under this controlled setup.

If both soft and straight-through regimes pass 6/6 strongly, report robust state-instantiation capability but **do not** attribute the improvement causally to straight-through gating; the original X20 split would remain seed-panel variance.

If straight-through fails seen competence, do not interpret unseen results as state-instantiation extension.

### Structure ablation

If graph-conditioned straight-through is seen-competent 6/6 while `straight_through_structure_blind` is not, report that dependency/version connectivity is required for robust discrete selection under this controlled constructor. Do not claim a separate unseen extrapolation effect unless both regimes are seen-competent.

## Claim boundary

Even a strong X20R result establishes only learned **instantiation/ignore of supplied candidate records from a supplied program graph** under answer-only supervision plus explicit memory cost.

It does not establish:

- discovery of entities from raw language/perception;
- semantic role induction;
- cross-episode persistence;
- reuse/merge of aliased state;
- deletion/lifecycle decisions;
- program/controller induction;
- verifier-guided repair.

## Successor boundary

Only a 6/6 strong straight-through state-instantiation PASS authorizes the next model-development experiment: **reuse/merge of repeated or aliased observations into persistent working-state identity**, followed by lifecycle deletion. If X20R fails, remain on the discrete state-construction objective/optimization boundary and do not advance to controller/program induction.
