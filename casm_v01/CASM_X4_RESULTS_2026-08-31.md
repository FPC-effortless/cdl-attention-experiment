# CASM-X4 results — 2026-08-31

## Executive result

CASM-X4 identifies a clear supervision boundary.

Workflow run: `33369793788`.

Exact evaluated head: `3e691f4906835789c2e6b4e10afa3cf455b5ea69`.

Train/eval seeds:

- `20260891` / `20260971`
- `20260892` / `20260972`
- `20260893` / `20260973`

The integrity suite passed before training and all three training/evaluation jobs completed successfully.

The main finding is:

> **A single randomly queried final register value per training program is sufficient, in this controlled system, to identify the shared explicit-state transition and recover the complete unobserved state trajectory exactly through depth 96. A single parity bit is not sufficient under the same architecture, data, and optimization budget.**

## Supervision contract

All three 246,160-parameter models start from identical parameters, see identical program batches, receive no intermediate state targets, receive no semantic-operator labels, and roll their own predicted differentiable state forward.

The only difference is terminal supervision:

- `full_final`: all four final register values;
- `one_register`: one uniformly sampled final register value per training example;
- `one_parity_bit`: even/odd parity of one uniformly sampled final register value per training example.

The query register is used by the loss only. It is not supplied to the transition kernel.

Across 4,000 steps × 128 examples = 512,000 training examples per seed, query counts were effectively uniform. Each register was queried about 128,000 times per seed.

This distinction matters: `one_register` is a randomly projected terminal observation across the dataset, not a permanently fixed observable register.

## Integrity checks

Before training, the suite verified that:

1. all regimes start from identical parameters;
2. one-register loss is invariant to every intermediate target and all three unqueried final registers;
3. parity loss is invariant to every intermediate target;
4. parity loss is invariant when the queried final value changes but parity is preserved;
5. parity loss changes when queried final parity flips;
6. weak losses ignore semantic-operator labels;
7. query sampling is approximately uniform;
8. finite nonzero gradients reach the transition network through depth 8.

All checks passed.

## Three-seed mean full-state accuracy

### Exact final-state accuracy

| Evaluation suite | Full final state | One register | One parity bit |
|---|---:|---:|---:|
| IID depth 8 | **100.00%** | **100.00%** | 1.74% |
| held-out composition depth 12 | **100.00%** | **100.00%** | 2.69% |
| held-out composition depth 24 | **100.00%** | **100.00%** | 2.00% |
| stress depth 48 | **100.00%** | **100.00%** | 3.65% |
| stress depth 96 | **100.00%** | **100.00%** | 2.52% |

### Exact step-state accuracy

| Evaluation suite | Full final state | One register | One parity bit |
|---|---:|---:|---:|
| IID depth 8 | **100.00%** | **100.00%** | 8.18% |
| held-out composition depth 12 | **100.00%** | **100.00%** | 6.20% |
| held-out composition depth 24 | **100.00%** | **100.00%** | 3.81% |
| stress depth 48 | **100.00%** | **100.00%** | 3.46% |
| stress depth 96 | **100.00%** | **100.00%** | 2.95% |

### Per-register accuracy

| Evaluation suite | Full final state | One register | One parity bit |
|---|---:|---:|---:|
| IID depth 8 | **100.00%** | **100.00%** | 49.13% |
| held-out composition depth 12 | **100.00%** | **100.00%** | 40.84% |
| held-out composition depth 24 | **100.00%** | **100.00%** | 29.25% |
| stress depth 48 | **100.00%** | **100.00%** | 22.66% |
| stress depth 96 | **100.00%** | **100.00%** | 19.30% |

The one-register result is exactly 100% on all three metrics at every depth in every seed; there is no observed seed instability.

## Preregistered verdict

`full_final` positive control required >=95% IID depth-8 final-state exactness. Observed: **100%**.

The strong `one_register` criterion required:

- >=90% IID depth-8 full final-state exactness — observed **100%**;
- >=90% IID depth-8 step-state exactness — observed **100%**;
- depth-24 final-state accuracy within 10 pp of full-final — observed gap **0 pp**;
- no seed below 80% IID final-state exactness — observed minimum **100%**.

Therefore the strong one-register criterion is satisfied completely.

The strong `one_parity_bit` criterion fails by a wide margin. Its mean IID depth-8 full final-state exactness is only 1.74%, with mean step-state exactness 8.18%.

## Optimization behavior

One-register supervision converges more slowly than full-final supervision but reaches the same exact mechanism.

Mean first logged checkpoint below training loss thresholds:

| Threshold | Full final state | One register | One parity bit |
|---|---:|---:|---:|
| `<0.1` | 1,067 steps | 1,467 steps | 633 steps |
| `<0.01` | 1,433 steps | 2,800 steps | not reached |
| `<0.001` | 2,300 steps | 3,233 steps | not reached |

Final parity-bit training losses after 4,000 steps were approximately `0.094`, `0.060`, and `0.093` across the three seeds. Thus the parity learner acquires substantial signal about the supervised bit but does not converge to an exact terminal-bit predictor, and—more importantly—does not identify the exact value-level state dynamics.

## Interpretation

### What one-register supervision establishes

The X3 result did not require the full final machine state after all.

Because the transition kernel is shared across programs, training on a random terminal projection gives distributed observational coverage: different examples supervise different registers. Across the full training distribution that is enough to identify a transition rule whose **unobserved** registers and intermediate states are also exact.

The strongest qualified claim is therefore:

> **Given the supplied explicit state schema and sparse transition interface, partial terminal observations distributed across diverse trajectories can identify the complete executable state-transition law without intermediate supervision.**

This is closer to system identification from partial observations than to ordinary process imitation.

### What parity supervision does not establish

The parity condition fails to recover exact machine state. This is consistent with an observability/identifiability bottleneck: a parity target collapses sixteen possible register values into two equivalence classes, while the underlying operator set includes copy, modular add/subtract, max, min, xor, increment, and decrement. Exact value dynamics can matter for later transitions even when a particular terminal parity does not distinguish them.

However, X4 does **not** prove an information-theoretic impossibility for one-bit supervision. The parity optimization itself has not converged to zero loss, so the conservative conclusion is:

> one parity bit per program is insufficient **under this architecture, data distribution, and preregistered 4,000-step optimization budget** to identify the exact value-level transition.

A dedicated observability analysis would be required to separate fundamental non-identifiability from harder optimization.

## Research consequence

The evidence chain now supports:

1. explicit sufficient state + shared local transition is a strong long-horizon execution bias (X2B);
2. true intermediate state teacher forcing is not required (X3);
3. complete final-state supervision is not required: random one-register terminal observations suffice (X4);
4. reducing terminal supervision to one parity bit crosses a boundary where exact state recovery fails under the current setup (X4).

The next useful experiment should test whether the full trajectory can be recovered when the terminal observation is **fixed and task-like** rather than randomly covering all registers across examples—for example, always supervise one designated answer register while the other registers remain permanently latent to the loss. That removes the distributed-coverage advantage of random queried registers and is a cleaner bridge toward answer-only reasoning supervision.
