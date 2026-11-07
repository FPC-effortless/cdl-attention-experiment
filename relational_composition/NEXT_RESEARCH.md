# Next research gate after LRC-002 and V4 integration

Status: design proposal, not a preregistered or executed experiment. Freeze an
executable protocol, dataset hashes, budgets and seeds before any follow-on fit.
The historical LRC-002 evaluation remains closed.

## Question

Can outcome-based credit learn useful record selection on raw text, including
older records, before we claim a benefit from coupled relational composition?
LRC-002 establishes endpoint decoding capacity, not route discovery. V4 establishes
potential access to old records, not learned page selection.

## Separate the learning obstacles

1. Copy control: supplied endpoint references, answer-only loss, identifiers made
   from the training alphabet. Establish decoding of previously unseen identifier
   combinations. Keep new-character output as a distinct stress population.
2. Reader diagnostic: oracle writer addresses with learned reads; then oracle
   addresses with learned page/slot choices. These receive privileged structure
   and cannot qualify the raw-text architecture. Measure per-hop candidate
   admission, selection conditional on admission, and exact free-running answers.
3. Full discovery: one record per byte, no oracle endpoints/addresses or semantic
   labels in model inputs. Compare full-text next-token loss and an explicitly
   answer-supervised arm. Count every alternative forward and index operation.
4. Only after competence, compare coupled and independent decisions with matched
   parameter and read budgets. Also compare V3 and V4 on old-record questions;
   quantify whether V4's accuracy benefit repays its latency and memory costs.

Use same-alphabet disjoint identifiers, relation-order splits, cross-mention
paraphrases, repeated updates, historical-version questions, and collision loads
above the newest-page cap. Avoid aliasing answer positions with task labels.
Keep answer lengths balanced and include a trained local model and independently
randomized-answer control. Record complete trajectories for failures, including
misses, nulls, chosen pages, ordered operands and carried references.

## Conditions before execution

Specify new initialization and generator seeds; keep all arms on paired worlds.
Freeze train/validation/test partitions and a fixed optimization budget. Choose
thresholds on development tasks only; evaluate held-out tasks once after all fits
close. Report every seed and collapse. Reserve the >=90% every-seed criterion for
the copy diagnostic unless a new competence threshold is justified prospectively.
Do not use the previous 0% unfamiliar-character result as an identity test.

The binary-page credit surrogate differentiates a conditional local comparison;
it is not an established unbiased estimator of the full routing objective.
Holding other coordinates of an overridden decision tensor fixed must be stated
explicitly. Future experiments must distinguish frozen-coordinate interventions
from fully recomputed downstream policy trajectories.

## Promotion rule

Accuracy, causal parity, historical accessibility and memory ceilings are hard
gates. Among qualified models use the recorded preference
`12 * log(inference / reference_inference) + log(training / reference_training)`.
Measure complete costs, including indexing, gathers, cache memory and routing
credit. This preference is not a prediction of GPU performance. No current model
has passed the full learning gate, so the selector should return no winner.
