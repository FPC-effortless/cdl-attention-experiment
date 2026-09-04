# CASM-X19D preregistration addendum — random-code falsifier

## Timing and authority

This addendum is committed **before any X19D implementation or execution**. It is part of the frozen X19D preregistration and supersedes only the `Frozen regimes`, constructor initialization, and interpretation/classification details below. All other requirements in `EXPERIMENT_CASM_X19D.md` remain unchanged.

The purpose is to distinguish **learned constructor dynamics** from a simpler possibility exposed by the role-keyed diagnostic substrate: because the current CASM variables are exchangeable, arbitrary sufficiently separated recurrent identity codes may already be enough for executable addressing.

## Amended frozen regimes

Use four regimes on identical task batches:

1. `canonical_keyed` — deterministic orthonormal direct role keys; positive control for the role-keyed memory/executor.
2. `frozen_random_orthogonal` — a nonlearned recurrent-code falsifier. Sample a seed role and raw matrix once from the run seed, construct the same Cayley-orthogonal recurrence used by the learned orthogonal treatment, freeze both constructor tensors permanently, and train only the executor.
3. `unconstrained_recursive` — learned seed plus parameter-shared normalized unconstrained linear recurrence.
4. `orthogonal_recursive` — learned seed plus parameter-shared Cayley-orthogonal recurrence.

The two **learned** regimes remain the parameter-matched causal pair. `frozen_random_orthogonal` is a diagnostic falsifier and is not required to match their trainable constructor parameter count.

## Frozen initialization

For each run, before cloning regimes:

- sample one constructor seed vector from `Normal(0,1)` and normalize it to unit length;
- sample one raw `32 x 32` transition matrix from `Normal(0,0.5)`;
- use bit-identical copies of these tensors for `frozen_random_orthogonal`, `unconstrained_recursive`, and `orthogonal_recursive`;
- `frozen_random_orthogonal` stores them as nontrainable buffers/parameters with `requires_grad=False`;
- the two learned treatments store bit-identical trainable copies with `requires_grad=True`.

`alpha=0.1` and `beta=16.0` remain unchanged.

No role vector beyond `r_3` may be generated before training finishes in any recurrent regime, including the frozen-random falsifier. Thus no run may inspect whether the sampled recurrence happens to extrapolate before the scientific budget is spent.

## Additional integrity requirements

Before training, tests must also establish:

1. frozen-random seed/matrix tensors are bit-identical to the learned pair at initialization;
2. frozen-random constructor tensors have `requires_grad=False` and do not change after a representative optimizer step;
3. the learned pair's seed/raw-matrix tensors are trainable and bit-identical at step zero;
4. all three recurrent regimes derive the orthogonal map through exactly the same Cayley implementation when operating in orthogonal mode;
5. no role diagnostics beyond `r_3` are run on the frozen-random regime before optimization completes.

## Amended post-training reporting

Run the same `r_0..r_31` separation/addressing diagnostics for `frozen_random_orthogonal` as for both learned recurrent regimes. Perturbation gain is also reported for the frozen random recurrence.

## Random-code falsifier classification

`frozen_random_orthogonal` is evaluated against the same seen and unseen task/addressing thresholds as the learned recurrences, but it is interpreted separately:

- **frozen random strong PASS**: arbitrary recurrent identity codes are sufficient for this controlled role-keyed benchmark. Do not claim that X19D demonstrates learned role construction, even if the learned orthogonal treatment also passes. The supported result is that separating role identity from storage removes the fixed-slot bottleneck and that this benchmark requires extensible distinguishable identities rather than learned role semantics.
- **frozen random FAIL + learned orthogonal strong PASS**: stronger evidence that task-selected noncontractive constructor dynamics matter beyond generic random identity codes.
- **frozen random PASS + learned pair unstable/fail**: optimization of constructor geometry is unnecessary or harmful under this benchmark; retain the architectural lesson but narrow the learning claim.
- **all recurrent regimes fail**: role-keyed storage alone is insufficient under the frozen addressing/executor contract.

The original learned `orthogonal_recursive` vs `unconstrained_recursive` causal comparison remains eligible only if both learned regimes satisfy six-seed seen constructor competence.

## Successor discipline

A frozen-random strong PASS by itself does **not** authorize a claim that the model learned to create computational roles. It may, however, justify treating role identity generation as a generic address-generation primitive and moving the next research question toward **when/what state should be instantiated**, provided the result is explicitly framed as supplied-cardinality dynamic record instantiation rather than learned structural creation.
