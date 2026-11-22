# INFO-001 matched native readout check (no training, no intervention)

The stagewise probe can recover the selected color from the pre-output
state, but it was not compared to native decoding on these same rows.
Run two frozen, label-free native observers on the already pinned 1,536
TRACE prompts; the primary comparison is the original 512 probe-held
rows. No new panel, train updates, output substitutions or retuning.
The actor now emits all 287 final-token logits and does not receive
labels, candidate colors, or probe membership. This is a separate stage
from read-only representation capture.

After sealing, compare native 287-ID strict rank, native restricted
eight-color and two-color ranks/log losses, and the already fixed
eight-color linear probe on the final captured representation plus raw
query with matching eight- and two-color scores. Never compare eight
or two candidates against 287 as a reader victory. Report paired
rescues/damages, not just a pooled difference. A probe may use the
normalized literal query alongside the final hidden tensor; mark that result as
an augmented diagnostic readout, not a pure replacement of the LM
head. It remains model-free fitting on the registered probe TRAIN only.
Retain the existing probe recipe’s predicted probabilities via a
deterministic model-free replay with no changes to seeds, data or
hyperparameters. This is not a new probe selection or an update to the
checkpoint. Verify replay aggregates against the existing result.

This separates a candidate readout/access bottleneck from a
representation/composition bottleneck only provisionally. Neither a
failed probe nor a frozen-head comparison proves information absent.
No protected TEST/MAIN, paid compute or model updates. INFO-002 remains
a separately reviewed, prospective causal design with fresh families.

Exact replay also hashes the refreshed FP32 pre-output tensor and
requires equality to the same-order tensor extracted from the prior
sealed traces before joining native score and probe results. No new
activation values are emitted by the readout actor.
