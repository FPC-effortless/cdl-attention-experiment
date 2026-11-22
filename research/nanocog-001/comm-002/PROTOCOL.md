# ROLE-002: reduce prefix-loss weight without changing architecture

ROLE-001 showed local binding is learnable but unstable across seeds. The
unsuccessful trajectories often learned ordinary prefix prediction while
still failing the role task. This motivates, but does not prove, an
objective interference hypothesis. Compare fixed prefix cross-entropy
weight 0.25 with the established 1.0-weight control. Change nothing else:
gated32, exact initialization, three matched seeds (1171/1271/1371), TRAIN
bytes, batches, AdamW, clipping, paired foil loss and 128/1280 endpoints.
No retuning after scoring, and no unreported seeds. Check whether prefix
prediction or role selectivity degrades; prefix-loss change precludes a
general language-quality claim without natural text evaluation.

## Evaluation and stops

Primary: 1,280-step strict full-vocabulary accuracy, with both 912/960
worlds and 108/120 all-eight families in **each** seed on a fresh panel.
Secondary: exact 128-step output, paired differences relative to the
old checkpoint for each same-seed scenario, two-color error analysis,
worst seed and loss/gradient diagnostics. A model that passes in every
seed must also outperform control on its worst seed to motivate the next
gate; if the baseline passes too, report that the different panel changes
the stability diagnosis rather than claiming objective necessity. No
claim of natural-language generalization, efficiency, or long-state use.

Fresh panel: sample 120 without replacement, using random.Random(20261123)
shuffle of the canonical sorted remaining joint name-pair/color-pair keys,
after excluding all exact 320 FIT and 120 old CONTROL families. Maintain
the same 2^3 factorial and seven causal tokens. Exclude historical and
protected splits. Prepare and hash the label-free and separate judged
projections before model scoring. Neither arm sees this panel during fit.
A genuinely new gate must not reuse this panel if tuning is needed.

Train once per seed (1,280 updates, one thread, 180s maximum each,
no paid compute) through exact reviewed native adequacy closure.
Use frozen original control weights, not a fresh selected fit.
Seal all accepted prediction artifacts before exact independent scoring.
Stop this branch on invalid source, integrity, or metric failure and
report it. If the intervention fails, retain the negative result and
choose a different single-variable repair (e.g. clip threshold) with
fresh evidence; do not rerun or scale blindly. If it qualifies, first
confirm the independent audit, then tackle selective packet retention
under the existing fixed budget rather than training the main run.
