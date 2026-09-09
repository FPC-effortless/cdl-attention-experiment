# CASM-X20V — narrow reuse/merge experiment

Preregistered before implementation from frozen X20U result `38e776c6e410d55bf62208faa55923533abb90c0`.

## Question

Can the qualified graph-conditioned discrete constructor reuse repeated or aliased supplied observations as one working-state identity without reducing capability or creating duplicate active records?

## Frozen treatments

- `no_reuse_control`: exact frozen X20U qualifying treatment.
- `reuse_merge_graph`: additive graph-conditioned reuse/merge proposal using only supplied candidate/program tensors and compatible identity/version/dependency signatures; hard decision threshold remains `g_soft >= 0.5`.
- `reuse_merge_structure_blind`: same proposal with dependency/version compatibility removed.

No live-mask, cardinality, merge-label, hidden-state, intermediate-target, count, top-k, lookup-table, or post-hoc threshold supervision is permitted. Frozen X20U model, executor, objective, optimizer, data, runtime and evaluation remain unchanged except for the additive reuse/merge mechanism.

## Gate

All six fresh seeds must pass every inherited X20U capability threshold on seen and strong unseen `n=5,6` cells. Reuse/merge additionally requires duplicate active identity rate <=0.05, merge precision >=0.95, merge recall >=0.90, and mean hard record-count error <=0.25. Any seed or cell failure fails the treatment; no averaging or selective rerun.

## Seeds and runtime

Train/eval pairs fixed before implementation: `20261221/20261301`, `20261222/20261302`, `20261223/20261303`, `20261224/20261304`, `20261225/20261305`, `20261226/20261306`.

Inherit Python `3.11.16`, PyTorch `2.14.0+cpu`, 12,000 steps, batch size 128, train depth 12, eval n=256, AdamW, cosine LR `2e-3 -> 2e-4`, weight decay `1e-4`, clip norm `1.0`.

## Integrity

The preregistration commit must precede implementation. Frozen X20U files remain byte-identical. The no-reuse control must numerically replicate X20U. The structure-blind ablation may differ only by removing dependency/version compatibility. Every result records exact Git SHA, runtime versions, seed pair, objective/reuse contract, evaluation contract, and non-finite checks.

## Successor rule

Only a full 6/6 seen plus 6/6 strong unseen pass for `reuse_merge_graph`, with no matching full pass for the structure-blind treatment, can unlock later lifecycle or persistence work. Otherwise freeze X20V at the reuse/merge boundary.
