# OVERWRITE-PATHS-051 — direct old-row inputs contribute to overwrite failures

14 November 2026. Completed frozen-checkpoint intervention diagnostic, following MEMORY-OVERWRITE-PATHS-050. Zero fits. One original native run produced 13,824 predictions from six fixed checkpoints across eight conditions.

The direct old-row inputs to the proposed new value are a material contributor to overwrite errors in these checkpoints. Zeroing those inputs at the final SET improves changed-value accuracy in every seed for both models. Forcing a full-strength final write alone does not repair the problem. This is a causal intervention on a fixed diagnostic grid, not a learned model improvement or an architecture qualification.

## Changed-value overwrites

These are the 224 off-diagonal old/new pairs per seed: four keys × eight new values × seven different old values. All three seeds are reported; aggregates count correlated probes and are not independent sample estimates. Normal inference uses raw tokens. Categorical-only supplied controls produce identical logits and state traces on all 6,912 matched condition pairs, so the same table applies to that diagnostic regime.

| Final-SET intervention | CSM seeds101 /202 /303, each /224 | CSM total /672 | Pointer seeds101 /202 /303, each /224 | Pointer total /672 |
|---|---|---:|---|---:|
| None |179 /196 /169|544 (80.95%)|191 /158 /145|494 (73.51%)|
| Full gain only |177 /195 /169|541 (80.51%)|194 /154 /141|489 (72.77%)|
| Zero direct old-row reads only |203 /216 /224|643 (95.68%)|201 /168 /189|558 (83.04%)|
| Zero reads + full gain |203 /217 /224|644 (95.83%)|203 /168 /189|560 (83.33%)|

At learned gain, zeroing direct reads rescues 110 and harms 11 CSM answers, net +99; it rescues 74 and harms 10 pointer answers, net +64. Thus the intervention is helpful in aggregate, not uniformly beneficial. Full gain alone rescues 2/harms 5 CSM and rescues 3/harms 8 pointer answers. The factorial comparison separates the direct transform-input pathway from residual interpolation.

## Rewriting the same fact can fail

The diagonal has 32 same-value rewrites per seed. Its semantic answer is unchanged.

| Condition | CSM seeds101 /202 /303 | Pointer seeds101 /202 /303 |
|---|---|---|
| Empty-row SET baseline |29 /31 /32|29 /24 /27|
| Same-value rewrite, no intervention |28 /29 /22|30 /24 /18|
| Same-value rewrite, zero direct reads |29 /31 /32|29 /24 /27|

CSM 303 therefore falls from 32/32 on a single SET to 22/32 when that same value is written again; zeroing the direct old-row inputs restores 32/32 on this panel. Baseline and rewrite differ in event timing/history as specified in the design; the intervention's effect within the same rewrite history is the causal comparison. Full gain alone leaves these same-value counts unchanged.

## Distinguishing old-value responses from existing confusions

On changed-value overwrites, 15 CSM and 6 pointer old-value outputs are newly induced relative to the matching empty-row baseline. These are observed old-value answers, not proof of a specific retrieval mechanism. Zeroing the direct reads reduces newly induced old-value outputs to 0 in both models, at either gain setting.

Some wrong answers still equal the old value after the intervention: they were already the baseline prediction for that key/new value. The fully crossed grid separates these coincidences from overwrite-induced changes, unlike the earlier old=new+1 panel. Diagonal old==new answers are correct and never counted as stale errors.

## What the state traces establish

Across all 13,824 predictions, the target row is exactly unchanged between the final SET and the start of QUERY. Every QUERY preserves all four user value and pointer rows, and every final SET preserves the other three user value and pointer rows. This rules out intervening NOP changes and query mutation of user records as explanations for these recorded failures; the query transformation/readout may still misinterpret an otherwise useful representation.

All masked categorical decisions are correct in this grid; unsupervised CSM operators are not included in that claim. Supplied-categorical control therefore changes neither the recorded states nor logits here. Checkpoint state dictionaries are unchanged before/after.

The preregistered pointer invariant passes in all seeds: under supplied categorical controls, zero direct reads and full final-SET gain, every overwrite prediction matches its similarly intervened empty-row baseline. Maximum absolute logit differences are 4.77e-7,4.77e-7 and9.54e-7, below 1e-5. This is numerical agreement within a stated tolerance, not bitwise identity.

For CSM, zero reads plus full gain also restores all baseline argmax predictions, but old-conditioned logits still vary in seeds101/202 (maximum ranges 0.02437 and0.39575). State traces likewise retain differences. CSM's controller/operator pathways were deliberately left intact. The experiment therefore does not establish that all old-state influence has been removed. The current probe does not separately intervene on those remaining paths.

`TRACE_ANALYSIS.json` is a post-execution descriptive analysis of the captured vectors, explicitly separate from the predeclared factorial summary. It fits no decoder and runs no model.

## Scientific implication and next repair

This finding supports repairing how SET forms a replacement representation before scaling the architecture comparison. Turning the overwrite gain up is insufficient when the proposal itself already depends on old contents. It does not justify suppressing memory access globally: COPY requires value transport, while LINK requires pointer/address handling. In this contract LINK does not commit the value-transform output, so this assay does not establish a need for its direct transform-read slice.

The next learned repair should teach SET-specific independence from prior contents, under equivalent information and a fixed total budget for both architectures. Compare an unchanged objective against an operation-appropriate invariance objective or learned read-admission rule; preserve COPY/LINK and deletion behavior and evaluate fresh histories. Keep the empty-row value discrimination failure as a separate target: even the most effective intervention here reaches only the existing 95.83% CSM and 83.33% pointer baseline, not 100%. Any repair must earn its result through learning; privileged intervention scores cannot substitute for that test.

## Limits and execution

Four known keys/eight known values, target-only constructed histories, one correlated diagnostic subject. No natural-language, long-history, noisy-observation, scaling, general lifecycle, inference-cost or architecture-superiority conclusion. No new fit, protected-test evaluation, paid compute or infrastructure change. Forge's TRAIN label is bookkeeping only. Existing pinned Forge→TrainLab→Veritas→ResearchLab interfaces preserve inputs, execution and evidence lineage.

The original CPU command succeeded in 14.324753 seconds, 432 batches, max 32 examples per batch, one Torch thread, within its 180-second cap. Duration is an execution receipt, not a fair cost comparison. Six random-initialized native software tests passed before trained inference, with additional independent checks of no-op full states, gain boundaries and cleanup. Pre-execution import correction is disclosed in DEVIATIONS.md; no trained outputs of this experiment were observed before its freeze.

- Dispatch: 775c753e9d59379402dbae22cbdd6391694adde7c6b34ede1719efdf3de7fcb6.
- Forge manifest: 8296e77ab59f2032178bbf15df79986d5b7956b61f93e6400c1435ced3d20212.
- Original run: 20261114T063903-5b12d1f06518.
- Receipt SHA256: 3326e91db5012ec4ed94ac808a8f1c0f484018515bb6f7e8fb231879c59c4087.
- Predictions SHA256: bca93d2db8f9dc7277e2feae213f30ef775ae51cb9932d499c7824c1c1fbc6b1.
- State traces SHA256: 56fda6a742ddbf9c5bc7b0434b3e7f51f4c9cf3550cdc930431568245fea8fe0.

Independent post-execution review rescored all 13,824 predictions and replayed all 432 batches under a separate 180-second audit cap. Full logits and all three captured state vectors match exactly (maximum difference 0); weights, NPZ alignment, actual batch counters, categorical equivalence and pointer invariants verified. The audit replay took 10.46 seconds and is not another independent scientific replicate. No historical data access attempts or optimizer steps occurred in the guarded audit.
