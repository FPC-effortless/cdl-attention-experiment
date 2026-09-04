# CASM-X20 — learned working-state instantiation

## Status and purpose

CASM-X20 is preregistered before implementation or execution.

The allocator/role-code line is closed for model-development purposes. X19D established robust hard executable identity/addressability beyond the trained role horizon, but the frozen-random falsifier showed that this benchmark does not require learned role semantics once every supplied variable is automatically granted a transient record. X19V is an invalid replay and cannot strengthen that claim.

X20 therefore moves the scientific object from **how a state record is addressed** to **whether a state record should exist at all**.

The immediate target is a controlled PLM constructor primitive:

`observations + supplied program -> decide which candidate entities deserve executable working-state records -> execute -> outcome`.

X20 does not yet learn program selection, cross-episode persistence, deletion over time, or semantic role discovery. It isolates record instantiation/ignore decisions under fixed supplied candidate identities.

## Question

> Can answer-only learning plus an explicit storage cost identify and instantiate the minimal causally live working-state set from a larger pool of observed candidate entities, generalizing from programs with 2–4 live entities to programs with 5–6 live entities, without receiving the live set or active cardinality as a label/input?

## Benchmark: live-state selection world

Every example contains exactly **8 observed candidate entities** with deterministic nonlearned candidate identity codes and categorical initial values in the existing 0..15 domain.

Candidate identity is supplied only so X20 can isolate **record existence** from role-code learning. No learned per-candidate ID table is permitted.

### Live set

- candidate `0` is always the output entity;
- choose `n-1` additional live candidates uniformly from candidates `1..7`;
- training live cardinality `n in {2,3,4}` only;
- unseen evaluation live cardinality `n in {5,6}`;
- `n` itself and the live-set mask are never supplied to the learned constructor.

### Program contract

Construct a depth-12 program over all eight candidate IDs with two interleaved components:

1. **live computation**: every live candidate is guaranteed to lie on a causal dependency path to the final value of candidate `0`;
2. **dead computation**: distractor candidates are also read/written by syntactically similar operations, but those computations are guaranteed not to influence the final output candidate `0`.

Every example must mention at least one distractor in program commands, and the generator should balance live/dead operand counts enough that simple `mentioned/not-mentioned` heuristics cannot solve the selection task.

Ground-truth liveness is produced by the generator only for evaluation/integrity checks; it is never available to the training loss.

## Fixed address substrate

X20 deliberately removes constructor-address geometry as a variable.

Use eight deterministic orthonormal/fixed candidate keys solely for read/write addressing of whichever records are instantiated. There is:

- no learned role recurrence;
- no learned role-to-slot scorer;
- no cosine-temperature experiment;
- no dual prices;
- no fixed-slot collision allocator;
- no matching or collision repair.

The scientific object is the **existence gate** for each candidate record.

## Learned constructor

For each candidate `i`, produce one existence probability

`g_i = sigmoid(h_theta(candidate_code_i, program_representation, output_code))`.

The constructor must be shared over candidates. No parameter may have first dimension 8 or otherwise implement a free per-candidate gate table.

### Program representation

Represent the supplied program as a temporal dependency graph derived only from program syntax:

- candidate/version nodes;
- operation nodes carrying the supplied command-family identity;
- directed edges from the current source versions to each operation and from each operation to the new destination version;
- the final version of candidate 0 is marked as the output root.

Use a parameter-shared message-passing constructor over this graph. The message passing may see command family and graph connectivity but must not receive generator liveness labels, final target values, active cardinality, or hidden execution trajectories.

The learned gate for candidate `i` is decoded from its initial/version structural representation plus the propagated output-root context.

Freeze exact hidden width, message-passing depth, activation, and parameter count in implementation tests.

## Soft and hard instantiation

Training uses soft gates for differentiability.

A record with gate `g_i` interpolates between its ordinary categorical value state and the existing `EMPTY` state. Read/write/update influence for that candidate is multiplied by the same gate so `g_i -> 0` removes the record from effective executable state.

Hard evaluation uses `g_i >= 0.5` as record existence, with no top-k selection, no cardinality repair, and no ground-truth-count adjustment.

A missing hard record is `EMPTY`; accesses to an absent dead record may execute as EMPTY but cannot be silently redirected to another record.

## Frozen regimes

1. `canonical_live_mask` — positive control using the generator's true live mask; validates that the gated executor can solve every n=2..6 world when the correct state set is supplied.
2. `all_records` — fixed all-eight-record control. Demonstrates task solvability without selection and quantifies storage inefficiency. It receives no structural reward for using fewer records.
3. `learned_instantiation` — decisive treatment; shared graph-conditioned existence gates trained only from final answer loss plus storage cost.
4. `structure_blind_gate` — parameter-matched negative/ablation control whose gate sees candidate code and aggregate command-family histogram but not dependency-graph connectivity/output-root propagation.

The two learned regimes must have matched parameter counts as closely as architecturally possible; any unavoidable mismatch must be documented before execution and must not arise from per-candidate parameters.

## Objective

Task supervision remains final answer register/entity 0 only.

For `learned_instantiation` and `structure_blind_gate`:

`L = L_answer + lambda_storage * mean_i(g_i)`

with frozen `lambda_storage = 0.05`.

No live-mask cross entropy, gate target, cardinality target, intermediate-state loss, hidden-register target, semantic operator label, role label, address label, or reconstruction loss is permitted.

`all_records` trains only on `L_answer` and has gates fixed to one.

The storage term is interpreted as an explicit finite-memory prior, not as evidence that the model discovered an energy law.

## Data and optimization

- candidate entities per example: exactly 8;
- live cardinality train: `{2,3,4}` only, deterministic repeated schedule;
- live cardinality unseen eval: `{5,6}`;
- program depth: 12;
- batch size: 128;
- optimizer: AdamW;
- exactly 12,000 steps;
- cosine learning rate `2e-3 -> 2e-4`, no warmup;
- weight decay `1e-4`;
- global gradient clip `1.0`;
- eval_n `256`;
- evaluation execution depths: base depth 12 plus composition/stress programs at depths 24,48,96 generated under the same live/dead causal contract.

Use six fresh paired seeds, fixed before implementation:

- train `20261171`, eval `20261251`;
- train `20261172`, eval `20261252`;
- train `20261173`, eval `20261253`;
- train `20261174`, eval `20261254`;
- train `20261175`, eval `20261255`;
- train `20261176`, eval `20261256`.

No seed replacement or selective rerun is permitted.

## Integrity requirements

Before training, tests must establish:

1. X19V result base is exactly `79f478f02a9c4a0153f85bcd2cb355ae9e776714`;
2. every example has exactly 8 candidates and true live n in the requested range;
3. candidate 0 is live and is the final output target;
4. every true live candidate has a causal path to final candidate 0;
5. every distractor lacks a causal path to final candidate 0 despite at least one distractor program mention;
6. simple mention count alone is not definitionally identical to liveness;
7. training batches contain only live n=2,3,4;
8. n=5,6 examples are never generated/inspected in any learned train-time forward or diagnostic;
9. learned constructor receives no live mask, active n, final target value, hidden state target, or generator causality label;
10. no learned per-candidate ID/gate table exists;
11. fixed address keys are nonlearned and distinct;
12. hard instantiation uses raw `g>=0.5` with no count repair/top-k;
13. setting all gates to the canonical live mask recovers positive-control execution;
14. setting all gates to one recovers ordinary all-record execution;
15. setting a required live record to zero can change/fail execution on falsifier cases;
16. removing a genuinely dead record does not change the canonical final answer on generator-certified examples;
17. changing hidden/intermediate targets while preserving final answer target leaves all learned losses unchanged;
18. changing final answer target can change answer loss;
19. gradients from total loss reach graph constructor and executor;
20. storage penalty gradient pushes gates downward when task loss is held fixed;
21. structure-blind and graph-conditioned controls consume identical candidate observations/program commands except for graph connectivity propagation;
22. all losses/gates/states abort on non-finite values.

## Frozen classification

### Positive-control validity

`canonical_live_mask` must satisfy every seed and n=2..6:

- hard answer-final >=99% every suite;
- hard deep step-state exactness >=95%;
- hard deep hidden/live-register accuracy >=99%.

Failure invalidates X20.

### Seen instantiation competence

`learned_instantiation` is extension-eligible only if every seed at trained live n=2,3,4 satisfies:

- hard and soft answer-final >=99% every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft live-register accuracy >=99%;
- hard existence precision >=0.95;
- hard existence recall >=0.95;
- mean absolute hard record-count error <=0.25;
- mean gate on live candidates >=0.90;
- mean gate on distractors <=0.10.

IID/base answer-final <80% on any seen cardinality is optimization failure.

### Strong state-instantiation extension

Strong PASS requires every seed at unseen live n=5,6:

- the same capability thresholds as seen;
- hard existence precision and recall >=0.95;
- mean absolute hard record-count error <=0.25;
- mean live gate >=0.90 and distractor gate <=0.10.

No averaging rescues a failed seed/cell.

### Partial extension

If strong fails but every unseen capability cell has answer >=90%, deep step/live-register >=80%, and existence F1 >=0.80, classify partial and report the exact boundary.

### Structural-constructor effect

A graph-conditioned-vs-structure-blind causal comparison is eligible only if both learned regimes meet the full seen capability threshold on all six seeds. The gate-structure comparison may then report whether dependency/output-root propagation materially improves state selection. If the structure-blind model cannot learn the seen task, do not claim causality from the extrapolation comparison.

## Claim boundary

Even a strong X20 PASS would establish only that a supplied graph-conditioned constructor can learn **which observed candidate states to instantiate/ignore** under answer-only supervision plus explicit memory cost, and extend that decision to larger live-set cardinalities.

It would not establish:

- discovery of candidate entities from raw language/perception;
- learned semantic role identities;
- persistent memory across episodes;
- autonomous deletion after use;
- controller/program induction;
- verifier-guided repair.

## Successor boundary

If X20 passes strongly, the next experiment should introduce **reuse/merge across repeated or aliased observations** and then lifecycle deletion, rather than returning to role-code generation. Only after state creation/reuse is robust should CASM proceed to learned controller/program induction.