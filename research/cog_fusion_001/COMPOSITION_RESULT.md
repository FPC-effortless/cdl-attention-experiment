# Optional numeric component: executed results

All three prospectively specified numeric fits passed their narrow component
gates. This is a new, separately supervised supplied-address vector computation
diagnostic. It does not qualify the fused language model, unfamiliar wording,
learned source selection, or the untrained optional feedback projections.

## Learned computation

Each pair of arms shared initialization, 64,000 training pairs and 1,000 optimizer
updates. Every pair consisted of two independent eight-dimensional vectors with
coordinates sampled from Uniform[-0.75,0.75]. The target was their mean. Both arms
had 4,296 parameters. The connected arm preserved its 32-coordinate controller
between reads; the separately trained control cleared it. Controller state is
128 float32 bytes. Three model seeds were run on one CPU thread.

The final test contains 4,096 fresh input pairs, shared across seeds and arms;
these are not 12,288 independent test examples.

| Seed | Connected MSE | Separately trained reset MSE | Acute reset of connected MSE | Permuted first input MSE |
|---|---:|---:|---:|---:|
| 1701 | 0.000047803 | 0.046965625 | 0.047765724 | 0.092938982 |
| 1702 | 0.000052634 | 0.047045451 | 0.048284911 | 0.093256332 |
| 1703 | 0.000053657 | 0.047058344 | 0.048267506 | 0.092908219 |

The conditional predictor that knows only the second vector has theoretical
expected coordinate MSE 0.046875 on this distribution. Its measured error on
this finite test sample was 0.046714384. The separately trained reset controls
converged close to that reference. The connected controller's low error and its
acute-reset/permutation degradation establish useful first-read retention on
this task. Gradients from the sum of output coordinates to first-input
coordinates were finite, with mean absolute derivatives 0.49666, 0.49918 and
0.49568, respectively. This is consistent with the desired half-weight on the
first vector; it is not a proof of exact averaging for all values.

All six endpoints completed. The paired process CPU totals, including both
arms plus evaluation/checkpoint work, were 4.6763, 2.8263 and 2.8950 seconds.
These are training-run measurements, not complete language-serving costs or an
attention-baseline comparison. No checkpoint was selected using the test set.

## Actual durable read-compute-write

The separately specified durable check took the first 16 held-out pairs from
each seed without outcome selection. Actual numeric source records were read
from Store, passed through the fitted recurrent controller, and the actual
learned output was committed as a typed derived record with controller state
and reply. No arithmetic replacement was used by this candidate path.

All 48 cases passed. The largest difference between single-record execution and
the saved batched neural output was 1.1920928955078125e-7, below the predeclared
1e-6 tolerance. There were 24 restarts between the first and second reads, 48
restarts after publication, and 48 controller resets followed by validated
derived-record reads. Controller snapshots matched exactly at restoration.
Untouched records remained unchanged.

Both supporting sources were tested separately for erasure in every seed: all
six erased-support cases blocked derived reuse. These failures are the required
dependency policy, and are not counted as retention successes. An unrelated
result remained available. Each seed's injected before-commit failure rolled
back the proposed derived output and controller snapshot together.

## Architecture connection and boundaries

RecordRuntime also accepts the optional FullFusionModel. Its actual shared
record controller can feed proposed numeric writes, semantic-write logits,
subsequent retrieval keys and the original character head. Structural tests
establish those paths and verify that zero controller preserves v1 outputs.
The numerical experiment fits only RecordComputer. The base language checkpoint
and the additional controller-to-language/query projections were not jointly
trained by these runs.

The typed compute interface supplies source keys and destination. Ordinary
language processing does not automatically invoke a numeric operation. Every
new complete computation starts with a reset record controller; begin_read and
finish_read expose an explicit staged computation with a saved first read.

Derived records bind both complete parent packets, the model identity, and the
composition source hash. read_derived validates them on reuse. A raw lookup of
the envelope alone is not a current-validity guarantee. Source updates, erasure,
pending intent and incompatible interpretation can invalidate reuse. Storage,
provenance validation and transaction work remain part of total operating cost.

## Evidence and reproduction

- Prospective design: COMPOSITION_PROTOCOL.md.
- Fit lock: evidence/composition/PROSPECTIVE_LOCK.json. All locked hashes still
  matched after the run.
- Fit results and all endpoints: evidence/composition/RESULTS.json and seed
  directories containing connected.pt, reset_control.pt, heldout_outputs.pt,
  RESULT.json and DURABLE_RESULT.json.
- Durable selection: evidence/composition/DURABLE_PROTOCOL.json.
- Durable results: evidence/composition/DURABLE_RESULTS.json.
- Commands: `python train_composition.py --lock-only`, review the prospective
  lock, then `python train_composition.py`; with the recorded durable protocol
  present, `python verify_composition.py`.
- Focused engineering checks: `PYTHONPATH=. python -m pytest -q
  tests/test_composition.py tests/test_full_model.py` (21 tests passed).

Do not overwrite a prior or partial run directory. Checkpoint loaders bind the
actual composition implementation hash; changing that implementation requires
an explicitly new version/run rather than relabeling the existing endpoints.
