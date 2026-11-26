# Optional numeric read-compute-write diagnostic

This protocol is prospective to the numeric fits. It does not alter the locked
v1 language experiment, and the fitted numerical module is not described as a
jointly trained language model.

## Question and interfaces

Can a 32-coordinate GRU controller retain the first actually supplied 8D record,
combine it with the second, and propose their arithmetic mean through a learned
linear write head? Source identities and the destination are supplied typed
interfaces. Neither learned language parsing nor learned record selection is
being qualified by this diagnostic.

RecordComputer has GRUCell(8,32) and Linear(32,8): 4,296 parameters and 128 bytes
of float32 controller state. Each new two-read task explicitly starts with a
zero controller. The first value is consumed; the second is consumed; the
learned head proposes the result. The executor never substitutes arithmetic for
that proposal.

## Matched experiment

- Paired seeds: 1701, 1702, 1703.
- Inputs: independent coordinates of two vectors, each Uniform[-0.75,0.75].
- Target: their arithmetic mean; targets are available to training/evaluation
  only, never to the candidate inference path.
- Connected arm: both reads enter one controller in order.
- Separately trained control: identical parameterization and initialization,
  but clear the controller after the first read and before the second.
- Exactly 1,000 updates per arm, batch size 64, AdamW learning rate 0.003,
  weight decay 0, default beta/epsilon; squared error averaged over coordinates.
- Both arms receive precisely the same fresh input batches in each seed and
  the same number of updates. Train with one CPU thread. No early stopping,
  checkpoint selection, or hyperparameter search.
- Training input generator seed: 900000 + model seed. Held-out generator seed:
  99170401, common across arms/seeds and never consumed by fitting.
- Final held-out test: 4,096 independent pairs. Evaluation includes the connected
  endpoint, independently trained reset control, and acute reset of the connected
  endpoint. A first-input permutation and gradients to the actual first vector
  test whether the retained first value influences output.
- Resource cap: 300 process CPU seconds per paired seed including both fits.
  A cap hit records an incomplete seed; it cannot count as a passed endpoint.
- Preserve all endpoints, loss traces, source/data hashes, exact update counts,
  wall/process CPU timings, and evaluation arrays in machine-readable results.

Without access to the first independent uniform vector, the Bayes-optimal
coordinate predictor is half the second vector. Its expected MSE is
Var(first)/4 = (1.5^2/12)/4 = 0.046875. This is an expectation for the declared
distribution, not a guaranteed finite-sample score or a natural-language bound.

## Prospectively declared component gates

Each of all three seeds must finish 1,000 updates per arm. Each connected model
must achieve held-out coordinate MSE <0.002, improve by at least 10x over its
separately trained reset control, and worsen to MSE >0.03 under acute reset.
The first input must have a finite nonzero gradient to the numerical proposal.
Report every seed and failure. These gates qualify only this narrow numeric
computation task; even a pass does not qualify the optional feedback projections
for language, event interpretation, or retrieval.

## Durable executor and extension checks

The same SQLite Store receives actual numeric input records. A compute operation
reads two packets from one common store revision, applies the learned model to
those actual payloads, and atomically publishes the derived vector, controller
continuation and reply using a compare-and-swap on that revision. Every supported
store mutation advances revision. This makes the common-revision check plus
commit CAS sufficient to exclude source mutations between either read and
publication; no store schema or frozen v1 source is changed.

The destination cannot alias either support. The derived value is a typed
envelope binding both complete parent packets, model identity and implementation
source hash. Reuse calls read_derived, which validates both parents and nested
derived dependencies, then confirms one unchanged revision across validation.
A raw Store.resolve of such an envelope is not a validity-qualified numeric
answer. Relevant changes, source erasure, missing support, pending edits, or
interpretation changes fail closed. Unrelated changes may preserve the result.

Test untouched-record preservation, source invalidation, exact continuation
restoration, idempotent requests, capacity failure, and rollback before/after
commit. The optional full-model wrapper supplies direct controller-to-query,
semantic-write and character-logit paths; structural gradient tests verify
their existence. Only RecordComputer is fitted by this experiment. The base
checkpoint and those extra projections remain separately identified.

The staged typed interface begin_read/finish_read also permits a process restart
between the two actual reads. It saves the first controller and complete first
packet. Resumption validates that packet, resolves both current dependencies at
one revision, consumes only the second value into the saved controller, and uses
the same atomic publication rule. A source correction while paused blocks the
derived write. reset_controller explicitly cancels an unfinished operation;
starting another computation cannot silently discard that unfinished read.
