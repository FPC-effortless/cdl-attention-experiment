# CASM-X3 results — 2026-08-31

## Executive result

CASM-X3 produced a strong positive result under the preregistered weak-supervision contract.

Workflow run: `33368882991`.

Exact evaluated head: `48ef61aceba686d0201082853143ce2763f3a1fa`.

Train/eval seeds:

- `20260881` / `20260961`
- `20260882` / `20260962`
- `20260883` / `20260963`

The contract/integrity job passed before training, then all three independent training/evaluation jobs passed, including metric validation and artifact upload.

The strongest justified finding is:

> **In this controlled contextual Markov program world, a shared local transition over an explicit categorical state can learn the correct intermediate computational trajectory from final-state supervision alone, without teacher-forced intermediate states or semantic-operator labels, and execute that learned transition exactly through depth 96.**

This is a statement about weakly supervised trajectory learning with a known state schema and known sparse transition interface. It is not evidence of open-ended state-variable or schema discovery.

## Preregistered setup

All regimes use the same `SoftExplicitTransitionModel`, cloned from identical initial parameters.

- four registers with values `0..15`;
- opaque contextual commands;
- command semantics depend on current state;
- fixed training depth 8;
- no semantic-operator labels;
- no true previous-state teacher forcing;
- every regime rolls its own predicted differentiable categorical state forward;
- evaluation rolls its own hard predicted state forward;
- identical 246,160-parameter architecture for every regime;
- identical training batches within each seed;
- 4,000 optimizer steps, batch size 128.

Supervision regimes:

- `process`: target state after all eight training transitions;
- `quarter`: target state only after transitions 4 and 8;
- `final`: target state only after transition 8.

The preregistration was committed before the training workflow.

## Integrity checks

The contract suite requires all of the following:

1. supervision indices exactly match the preregistered process/quarter/final contract;
2. all regimes start from identical parameters;
3. rollout is invariant to arbitrary corruption of all `target_states`;
4. final-only loss is invariant to arbitrary corruption of every intermediate target state;
5. quarter-supervision loss is invariant to corruption of every unsupervised target state;
6. final-only gradients are finite;
7. hard autoregressive rollout is valid through depth 96.

All checks passed before training.

Code inspection also confirms that `SoftExplicitTransitionModel` does not consume the batch's semantic-operator labels in either rollout or loss. The learned transition receives only predicted state, opaque command and register arguments, and the same sparse state-value features used by the qualified X2 transition interface.

## Three-seed result

Every regime achieved exact performance on every suite in every seed.

### Mean final-state exact accuracy

| Evaluation suite | Process | Quarter | Final only |
|---|---:|---:|---:|
| IID depth 8 | **100.00%** | **100.00%** | **100.00%** |
| held-out composition depth 12 | **100.00%** | **100.00%** | **100.00%** |
| held-out composition depth 24 | **100.00%** | **100.00%** | **100.00%** |
| stress depth 48 | **100.00%** | **100.00%** | **100.00%** |
| stress depth 96 | **100.00%** | **100.00%** | **100.00%** |

### Mean step-state exact accuracy

| Evaluation suite | Process | Quarter | Final only |
|---|---:|---:|---:|
| IID depth 8 | **100.00%** | **100.00%** | **100.00%** |
| held-out composition depth 12 | **100.00%** | **100.00%** | **100.00%** |
| held-out composition depth 24 | **100.00%** | **100.00%** | **100.00%** |
| stress depth 48 | **100.00%** | **100.00%** | **100.00%** |
| stress depth 96 | **100.00%** | **100.00%** | **100.00%** |

Per-register accuracy is also 100% throughout.

There is no observed seed instability: all three final-only seeds are exactly 100% on final-state exactness and step-state exactness at every evaluation depth.

## Preregistered verdict

The positive-control competence gate required process supervision to reach at least 95% mean IID depth-8 final-state exactness. Observed: **100%**.

The strong final-only result required:

1. mean IID depth-8 final exactness >=90% — observed **100%**;
2. mean IID depth-8 step-state exactness >=90% — observed **100%**;
3. depth-24 final exactness within 10 percentage points of process — observed gap **0 pp**;
4. no final-only seed below 80% IID depth-8 final exactness — observed minimum **100%**.

All preregistered strong-result criteria are therefore satisfied cleanly.

## Optimization cost

Weak supervision did not reduce eventual accuracy, but it did make optimization slower.

Using the logged 100-step checkpoints, mean first checkpoint below each training-loss threshold was approximately:

| Loss threshold | Process | Quarter | Final only |
|---|---:|---:|---:|
| `<0.1` | 533 steps | 633 steps | 867 steps |
| `<0.01` | 867 steps | 1,033 steps | 1,667 steps |
| `<0.001` | 1,167 steps | 1,167 steps | 2,700 steps |

Thus final-state-only supervision requires materially more optimization to identify the same local transition, even though it converges to the same exact execution rule by the 4,000-step budget.

## Why this is not endpoint shortcut evidence

A final-only learner could in principle fit final outcomes without reconstructing the intended intermediate computation. X3 therefore preregistered step-state exactness as a decisive diagnostic.

The final-only regime achieves **100% exact intermediate state trajectories** at IID depth 8 and at held-out composition/stress depths 12, 24, 48, and 96. The learned model is therefore not merely reaching the correct endpoint on the tested distribution; its hard rollout reproduces the ground-truth state after every transition.

The architecture also contains no timestep embedding or separate depth-specific transition modules. The same shared transition kernel is recursively applied at every step. Exact extrapolation from fixed training depth 8 to depth 96 is therefore consistent with learning the local transition law rather than memorizing an eight-step endpoint map.

## Qualified interpretation

X2B established that, once previous-state and sparse feature access are matched, an explicit stateless transition accumulates less long-horizon error than a GRU carrying a redundant latent recurrent state.

X3 adds a different result:

> **The explicit transition does not require the correct intermediate state to be fed back as teacher-forced training input. In this controlled world, end-state supervision alone is sufficient to learn a transition whose recursively generated intermediate states are exact.**

This strengthens the case for the computational pattern:

> explicit sufficient state + shared local transition + recurrent execution

and shows that dense process labels are not intrinsically required for that pattern on this benchmark.

## Important claim boundary

X3 does **not** demonstrate autonomous discovery of the state representation.

The experiment supplies:

- the four-register state schema;
- the discrete value domain;
- which register is the destination of each transition;
- the argument register identities;
- direct differentiable access to the relevant predicted register values;
- the architectural rule that computation must update this explicit state through one shared local transition.

The learner discovers the transition function under weak supervision, not the ontology or topology of the state itself.

The final target is also the complete final machine state, which is richer supervision than a scalar reward, binary verifier result, or natural-language answer.

Accordingly, the next research question should not repeat X3 with fewer copies of the same state label. It should weaken **what the learner is told about the state**.

## Next falsifiable step

The next experiment should hold the now-qualified transition mechanism fixed while removing part of the supplied state structure. A useful progression is:

1. final answer / verifier supervision rather than full final-state supervision;
2. partial observation of the final state;
3. latent slots that must align to registers without slot labels;
4. unknown write target or learned sparse access;
5. state-schema growth/splitting/merging under changing tasks.

The critical falsifier is whether the learned internal/explicit state remains causally adequate and compositionally executable when the correct state variables are no longer directly specified by the training target.
