# COG value-path repair: matched training did not fix the binding failure

**Result: zero of three paired seeds met the frozen local quality gates.**
The new contextual tap-feature adapter added zero complete second-clause
triplets over its matched trained control in every seed. The earlier
INFO-002 transplant established causal sensitivity to contextual tap values;
it did not establish that this frozen-backbone adapter could learn to make
that value path selective. This follow-up tested that distinction.

## Scope and controls

The two arms use one common frozen 37,414-parameter gated32 weak ROLE-002
checkpoint, with a matched 800-parameter, zero-start residual immediately
before the final Layer-2 pointwise mix. Both retain the original contextual
gate and value path. Only the **added** path's tap features differ:
contextual tap content, or a literal-query/lag feature control. Both optimize
the same full-vocabulary answer loss and target/distractor margins on an
identical 1,280-update order per seed. Native is the frozen original, without
adapter. These are three adapter seeds, not three independently trained
backbones.

The fresh held evaluation uses 32 name-pair groups for edit triplets and 120
ordinary all-eight families. Their unordered name-pair pools are disjoint
from the adapter's TRAIN pool and from each other; 1,432 earlier joint
keys were excluded. This does not make the groups unseen by the frozen
backbone: 28/32 edit pairs and 41/45 ordinary name pairs occurred in its
original training. The grammar and 16 name and eight color vocabularies are
controlled, not evidence of general language transfer.

## Final held endpoints

A complete triplet requires the correct original answer, the new answer
after the target edit, and the original answer after the distractor edit.
Every answer must be a unique maximum of the full 287-token vocabulary.

| Arm / adapter seed | Second-clause triplets / 128 | Wrong distractor insertions / 128 | Ordinary worlds / 960 | All-eight families / 120 |
|---|---:|---:|---:|---:|
| Frozen native | 98 | 30 | 858 | 70 |
| Literal/lag 2071 | 98 | 30 | 852 | 70 |
| Context taps 2071 | 98 | 30 | 855 | 70 |
| Literal/lag 2171 | 98 | 30 | 851 | 70 |
| Context taps 2171 | 98 | 30 | 853 | 70 |
| Literal/lag 2271 | 98 | 30 | 851 | 70 |
| Context taps 2271 | 98 | 30 | 854 | 70 |

The content arm needed at least seven additional complete second-clause
triplets over the matched control in **each** seed, alongside at least
912/960 ordinary worlds and 108/120 complete families. It met none of
these three requirements in any seed. Strict preserved-gold answers were
98/128 in every arm, including native. For the separate same-history
four-answer check, all arms and both fixed endpoints obtained 97/128.
At 128 updates all six trained branches matched native on the headline
counts. At 1,280 updates the contextual arm's ordinary accuracy fell
by 3–5 worlds versus native, while its first-clause accuracy was unchanged
or one world better. Equal aggregate counts do not imply identical outputs.

## Decision and replay

Do not scale this adapter, increase memory, or tune its held-set thresholds.
The measured contrast argues against repairing the selected frozen model
with this small final value residual; it does not rule out learning
query-specific bindings jointly in the original representations. The next
decision should compare end-to-end learned local composition against an
appropriate matched control, without relaxing the original binding gate.
Only passed quality can clear fixed-slot retention/routing and later
full-cost matched-quality inference comparisons. No serving latency claim,
MAIN/protected TEST access, or architecture promotion.

All six native TRAIN and all 13 exact blind V3 actor receipts verified.
The first evaluator versions failed on checkpoint loading before prompt
decoding or a forward pass; the original failed receipts were retained.
The V3 loader kept weights-only deserialization, preserved every checkpoint,
and passed two complete TRAIN-only actor runs before the blind run.
The governing weights-free scorer was sealed separately. Reported scorer
SHA-256 `b0e0f06eece2cda87c4d92e293f4e3b67cf7ff00e4add519b030023a96c8dc8c`;
result SHA-256 `34c6f504e7d5b0eacb4fa6ca4986bf71b2ac58a7c72dfbc57b1105d63eb3f92f`.
The public sources and aggregate report exclude private checkpoints,
prompts, truth and raw logits. The private archive retains those materials
and both failed evaluator attempts for reproducibility.
