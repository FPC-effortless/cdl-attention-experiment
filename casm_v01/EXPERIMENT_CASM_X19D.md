# CASM-X19D — Noncontractive recursive role dynamics

## Status and purpose

CASM-X19D is preregistered before implementation or execution.

CASM-X19 is valid but cannot support a paired recursive-vs-static extrapolation claim because each learned regime misses six-seed seen competence once. Its diagnostics nevertheless expose two separable boundaries:

1. the generic recursive role transition usually contracts toward a low-change/fixed-point attractor beyond the trained role horizon;
2. the learned fixed-slot storage bridge is itself an unseen-role extrapolation bottleneck.

X19D isolates the first boundary. It does **not** resume allocator development and it does **not** advance the PLM program to full X20 dynamic state instantiation.

The experiment removes learned variable-to-fixed-slot assignment from the scientific path and uses a deterministic role-keyed memory substrate solely to test whether recursive role dynamics remain discriminative and executable beyond the trained recurrence horizon.

## Question

> Under an identical role-keyed executable memory, does a parameter-shared approximately norm/angle-preserving recursive transition maintain executable role separation beyond the trained role horizon more reliably than a parameter-matched unconstrained normalized linear recurrence?

This is a constructor-dynamics experiment. It is not variable discovery, cardinality discovery, learned memory creation/reuse, program induction, persistence, or verifier-guided repair.

## Frozen regimes

Three regimes receive identical task batches and evaluation suites:

1. `canonical_keyed` — positive control with deterministic orthonormal role keys; validates the role-keyed memory and executor for every `n=2..6`.
2. `unconstrained_recursive` — learned seed plus repeated application of one normalized unconstrained linear transition.
3. `orthogonal_recursive` — learned seed plus repeated application of one Cayley-parameterized orthogonal transition.

The two learned regimes must have exactly equal total/trainable parameter counts, receive bit-identical initial learned seed vectors and bit-identical raw transition parameter tensors, and differ only in the map derived from that raw transition tensor.

## Frozen role dimension and recurrence

Use role dimension `32`.

Both learned regimes contain:

- learned seed `r_seed in R^32`;
- one raw learned matrix `A in R^(32x32)`;
- the same validated local-equivariant transition executor;
- no learned external-ID, role-index, recurrence-step, active-cardinality, slot-ID, or memory-address embedding/table.

Let `r_0 = normalize(r_seed)` and fixed `alpha = 0.1`.

### Unconstrained recurrence

`M(A) = I + alpha A`

`r_{i+1} = normalize(M(A) r_i)`.

The same `A` is reused for every role step. No index/cardinality input is permitted.

### Orthogonal/noncontractive recurrence

Construct the skew-symmetric matrix

`S(A) = A - A^T`.

Use the Cayley transform

`Q(A) = (I - alpha S)^(-1) (I + alpha S)`.

Then

`r_{i+1} = Q(A) r_i`.

`Q(A)` must satisfy `Q^T Q ~= I` to the preregistered numerical tolerance. The same `Q` is reused for every role step. No index/cardinality input is permitted.

The treatment therefore introduces a noncontractive geometric inductive bias without adding a role lookup table, external coordinate, resource allocator, collision repair, or extra learned parameters.

## Frozen role-keyed memory substrate

X19D removes the learned role-to-physical-slot bridge.

For a supplied active cardinality `n`, deterministically instantiate exactly one transient memory record per supplied external variable:

`Memory = {(r_i, p_i)} for i=0..n-1`,

where `p_i` is the categorical value distribution stored in that record.

This one-record-per-supplied-variable instantiation is a **diagnostic substrate**, not a learned creation decision and not an X20 claim. The active cardinality remains supplied by the benchmark exactly as before.

### Addressing

Commands still supply external argument/destination indices `a,b,dst` exactly as in existing CASM. To address a record, the corresponding generated role is used as the query key.

For query role `q=r_i` and record keys `r_j`, use fixed cosine-attention logits

`ell_j = beta * cosine(q, r_j)`

with frozen `beta = 16.0`.

Soft addressing is row softmax over the active records.

Hard addressing is independent argmax of the same logits with no tie repair, sorting, matching, masking, or external-index bypass.

The query index may select which already-generated role vector is used as the query, but may not directly select/read/write a memory record. All value access must pass through role-key similarity.

### Read / write / decode

- soft read: weighted mixture of categorical record values;
- hard read: value of the argmax-addressed record;
- the validated shared local transition kernel consumes command identity plus the addressed `a`, `b`, and `dst` values;
- soft write uses the destination-address distribution as the write weights over records;
- hard write updates only the argmax-addressed record;
- external decoding likewise queries each generated role against memory keys rather than indexing records directly.

No physical eight-slot bank, slot descriptor, role-to-slot scorer, dual price, occupancy state, capacity controller, Sinkhorn, Hungarian matching, hard collision repair, or structural collision penalty exists in X19D.

## Frozen executor and supervision

Retain the validated local-equivariant transition kernel and the same variable-contextual task family.

Retain answer-only task supervision:

- final external register `0` is the only task target;
- no teacher forcing;
- no intermediate-state targets;
- no hidden-register targets;
- no semantic operator labels;
- no role labels or role-distance targets;
- no addressing labels;
- command family and `a,b,dst` remain supplied;
- active cardinality `n` remains supplied.

There is **no explicit role-separation, orthogonality, contrastive, entropy, collision, or retrieval-target loss**. Any useful role geometry must be selected through answer-only execution loss plus the architectural recurrence bias.

## Frozen data / optimization

Use the X19 data and optimization budget:

- train cardinalities `n in {2,3,4}` only;
- deterministic `2,3,4` repeated schedule;
- train depth `8`;
- batch size `128`;
- exactly `10,000` optimizer steps;
- AdamW;
- weight decay `1e-4`;
- cosine LR `2e-3 -> 2e-4`, no warmup;
- global gradient clipping `1.0`;
- evaluation `n=2..6`, depths `8,12,24,48,96`, `eval_n=256`;
- hard and soft role-addressed execution evaluation.

Training at `n<=4` may generate only `r_0..r_3`. Roles `r_4` and beyond must not be generated, inspected, regularized, logged, or evaluated before all 10,000 optimizer steps finish.

## Frozen seed panel

Use six fresh paired seeds:

- train `20261161`, eval `20261241`;
- train `20261162`, eval `20261242`;
- train `20261163`, eval `20261243`;
- train `20261164`, eval `20261244`;
- train `20261165`, eval `20261245`;
- train `20261166`, eval `20261246`.

No seed may be replaced, omitted, selectively rerun, or promoted based on outcome.

## Integrity requirements

Before training, tests must establish:

1. `canonical_keyed` executes the role-keyed store without direct external-index record access;
2. learned regimes have exactly equal total/trainable parameter counts;
3. learned seed and raw `A` start bit-identically between treatments;
4. role dimension is exactly 32;
5. no learned per-index/cardinality/step/address/slot table or embedding exists;
6. the same raw `A` parameter is reused at every recurrence step;
7. learned recurrence receives no `i` or `n` input;
8. unconstrained recurrence is exactly `normalize((I+0.1A)r)`;
9. orthogonal recurrence is exactly the Cayley transform of `A-A^T` with `alpha=0.1`;
10. `Q^T Q` maximum absolute error <= `1e-5` at initialization and after a representative optimizer step;
11. both treatments receive identical batches, optimizer hyperparameters, LR schedule and clipping;
12. role-keyed soft addressing uses only cosine similarity of generated query/key roles and fixed `beta=16`;
13. hard addressing is raw argmax with no repair;
14. direct record indexing by command `a,b,dst` is forbidden in read/write/decode paths;
15. permuting memory-record presentation order while permuting stored values identically leaves decoded external behavior invariant;
16. duplicating two role keys can create an unrepaired hard addressing ambiguity and is not automatically corrected;
17. changing hidden/intermediate targets or semantic labels while preserving final register-0 target leaves loss unchanged;
18. changing final register-0 target can change loss;
19. gradients from answer loss reach learned seed, raw recurrence matrix and executor in both learned regimes;
20. training schedule is exactly `2,3,4` repeated;
21. runner cannot generate or inspect `r_4+` or any `n=5,6` learned execution before training completes;
22. all losses, roles, Cayley solves, attention weights, memory values and executor outputs abort on non-finite values.

## Post-training constructor diagnostics

Only after optimization, unroll each learned recurrence through `r_31`.

Report:

- role norms `r_0..r_31`;
- pairwise cosine-similarity matrix;
- minimum pairwise cosine distance for prefixes 4, 6, 8, 16, 32;
- maximum off-diagonal cosine similarity for the same prefixes;
- consecutive cosine similarity;
- first recurrence index whose cosine similarity to any earlier role exceeds `0.99`, if any;
- soft self-address probability and nearest competing address probability for each role through `r_31` using the role-keyed memory;
- hard self-address uniqueness/ties.

### Perturbation stability diagnostic

After training only, use eight deterministic perturbations of `r_0` with pre-normalization Euclidean magnitude `1e-3`, renormalize, and unroll through `r_31`.

For each perturbation report

`gain_i = ||r_i' - r_i||_2 / ||r_0' - r_0||_2`

for every `i=0..31`, plus maximum and final gain.

These diagnostics cannot rescue failed task thresholds.

## Frozen classification

### Positive-control prerequisite

`canonical_keyed` must satisfy every seed and every `n=2..6`:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95% at depths 24/48/96;
- hard and soft deep hidden-register accuracy >=99% at depths 24/48/96.

Failure invalidates X19D.

### Seen constructor competence

A learned recurrence is extension-eligible only if every seed at `n=2,3,4` has:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- every active role hard-addresses its own memory record uniquely;
- mean active-role soft self-address probability >=0.90.

IID depth-8 answer-final <80% on any seen cardinality is optimization failure.

### Strong constructor extension

A learned recurrence passes strongly only if every seed at unseen `n=5,6` has:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- every active role hard-addresses its own memory record uniquely;
- mean active-role soft self-address probability >=0.90.

No averaging rescues a failed seed/cell.

### Partial constructor extension

If strong fails but every unseen hard and soft cell has answer-final >=90% and deep step-state/hidden-register >=80%, classify partial and report the exact boundary.

### Noncontractive-constructor effect

A causal comparison is eligible only if both learned recurrences satisfy seen constructor competence on all six seeds.

If eligible:

- orthogonal strong PASS + unconstrained fail: support noncontractive role dynamics as the decisive treatment under this role-keyed substrate;
- both strong PASS: orthogonal/noncontractive bias is not necessary under this benchmark;
- orthogonal partial + unconstrained below partial: report partial improvement only;
- both fail unseen: role recurrence remains insufficient even after removing learned fixed-slot allocation;
- either learned regime fails six-seed seen competence: paired causal comparison is ineligible; diagnose constructor optimization without returning to allocator tuning.

## Claim boundary

Even a strong orthogonal PASS would show only that a supplied parameter-shared noncontractive recurrence can maintain executable role separation beyond its trained role horizon in a controlled role-keyed memory with supplied variables, cardinality and commands.

It would **not** show:

- discovery of variables or cardinality from raw observations;
- learned decision to instantiate/reuse/delete memory records;
- full X20 dynamic working-state construction;
- program induction;
- cross-episode persistence;
- verifier-guided repair.

## Successor boundary

Only if a recursive constructor passes strongly and robustly on all six seeds should the program advance to X20, where memory/state instantiation itself becomes a learned architectural object rather than this deterministic diagnostic substrate.

If the orthogonal recurrence fails unseen while remaining seen-competent, investigate richer compositional constructor dynamics or role-generation state, not fixed-slot allocation.
