# COG-BIND-TIME-PROC-005: locator supervision and causal role swaps

**Outcome:** frozen: temporal FAIL, native FAIL, ordinary: temporal FAIL, native FAIL, process: temporal FAIL, native FAIL.
**Predeclared specific process-supervision win:** FAIL.

## Design

SCOPE-004 changed scores without changing hard event selections. Both PROC-005 arms use the exact same scope architecture, W start, zero-output residual, mixed-layout data and training schedule. The process comparator adds 0.35 times negative log soft locator mass at TRAIN action/trigger/cue spans to the event-choice loss. No spans or typed IDs enter the native actor. Frozen Y/Z readout is shared; there is no phase supervision in either fit.

The covered combinations were fitted in both arms. Off-diagonal combinations are withheld from both fits but historically evaluated. Fresh TRAIN uses the same six formerly development actor names as SCOPE, with all prior exact prompts and canonical frames excluded. Current split names are disjoint.

## Exact, strict outcomes by seed

| Pane | Seed | Arm | Distinct W event rows | Distinct strict before/after pairs | Distinct strict chronology pairs | Complete W temporal octets | Native exact event+phrase+pointer |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| held-familiar | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,024/2,048 (98.83%) |
| held-familiar | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,024/2,048 (98.83%) |
| held-familiar | 20371 | process | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,024/2,048 (98.83%) |
| held-familiar | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20471 | process | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20571 | frozen | 380/512 (74.22%) | 190/256 (74.22%) | 176/256 (68.75%) | 44/64 (68.75%) | 1,520/2,048 (74.22%) |
| held-familiar | 20571 | ordinary | 356/512 (69.53%) | 176/256 (68.75%) | 174/256 (67.97%) | 43/64 (67.19%) | 1,424/2,048 (69.53%) |
| held-familiar | 20571 | process | 324/512 (63.28%) | 162/256 (63.28%) | 152/256 (59.38%) | 38/64 (59.38%) | 1,296/2,048 (63.28%) |
| held-covered | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,024/2,048 (98.83%) |
| held-covered | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,024/2,048 (98.83%) |
| held-covered | 20371 | process | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,024/2,048 (98.83%) |
| held-covered | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20471 | process | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20571 | frozen | 492/512 (96.09%) | 246/256 (96.09%) | 236/256 (92.19%) | 59/64 (92.19%) | 1,968/2,048 (96.09%) |
| held-covered | 20571 | ordinary | 448/512 (87.50%) | 222/256 (86.72%) | 194/256 (75.78%) | 48/64 (75.00%) | 1,792/2,048 (87.50%) |
| held-covered | 20571 | process | 448/512 (87.50%) | 224/256 (87.50%) | 192/256 (75.00%) | 48/64 (75.00%) | 1,792/2,048 (87.50%) |
| held-offdiagonal | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,024/2,048 (98.83%) |
| held-offdiagonal | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,024/2,048 (98.83%) |
| held-offdiagonal | 20371 | process | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,024/2,048 (98.83%) |
| held-offdiagonal | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20471 | process | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20571 | frozen | 464/512 (90.62%) | 232/256 (90.62%) | 220/256 (85.94%) | 55/64 (85.94%) | 1,856/2,048 (90.62%) |
| held-offdiagonal | 20571 | ordinary | 468/512 (91.41%) | 232/256 (90.62%) | 222/256 (86.72%) | 55/64 (85.94%) | 1,872/2,048 (91.41%) |
| held-offdiagonal | 20571 | process | 424/512 (82.81%) | 212/256 (82.81%) | 168/256 (65.62%) | 42/64 (65.62%) | 1,696/2,048 (82.81%) |

There are 64 fresh roots per held pane, eight distinct route contexts per root and four matched downstream cases per route. No independence claim for aligned panes, seeds, or repeated route rows. A native pointer must name the correct event and phrase as well as the correct token. Ties fail and phase interventions remain diagnostics.

## Mechanism contrast and familiar robustness

| Seed | Process supervision vs ordinary scope head on covered strict pairs, pp [root 95% CI] | Off-diagonal strict pairs, pp [root 95% CI] | Weakest familiar layout: frozen / ordinary / process strict pairs |
|---:|---:|---:|---|
| 20371 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) |
| 20471 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) |
| 20571 | +0.78 [-4.69, +7.03] | -7.81 [-14.84, -0.78] | 3_3: 28/64 (43.75%) / 2_2: 20/64 (31.25%) / 2_2: 16/64 (25.00%) |

The registered 80% familiar per-layout flag is separate from aggregate qualification. No mean or gate supersedes a weak layout.

## Separating phase labels from ranking

| Pane | Seed | Arm | Both actual event phases exactly labeled | Query phase exactly labeled | Event rank with supplied gold query phase (diagnostic only) |
|---|---:|---|---:|---:|---:|
| held-familiar | 20371 | frozen | 500/512 (97.66%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20371 | ordinary | 512/512 (100.00%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20371 | process | 512/512 (100.00%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20471 | frozen | 492/512 (96.09%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20471 | ordinary | 436/512 (85.16%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20471 | process | 512/512 (100.00%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20571 | frozen | 304/512 (59.38%) | 512/512 (100.00%) | 380/512 (74.22%) |
| held-familiar | 20571 | ordinary | 288/512 (56.25%) | 508/512 (99.22%) | 360/512 (70.31%) |
| held-familiar | 20571 | process | 272/512 (53.12%) | 512/512 (100.00%) | 324/512 (63.28%) |
| held-covered | 20371 | frozen | 500/512 (97.66%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20371 | ordinary | 508/512 (99.22%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20371 | process | 512/512 (100.00%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20471 | frozen | 488/512 (95.31%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20471 | ordinary | 432/512 (84.38%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20471 | process | 504/512 (98.44%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20571 | frozen | 276/512 (53.91%) | 512/512 (100.00%) | 492/512 (96.09%) |
| held-covered | 20571 | ordinary | 228/512 (44.53%) | 508/512 (99.22%) | 452/512 (88.28%) |
| held-covered | 20571 | process | 252/512 (49.22%) | 512/512 (100.00%) | 448/512 (87.50%) |
| held-offdiagonal | 20371 | frozen | 500/512 (97.66%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20371 | ordinary | 508/512 (99.22%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20371 | process | 512/512 (100.00%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20471 | frozen | 488/512 (95.31%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20471 | ordinary | 424/512 (82.81%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20471 | process | 508/512 (99.22%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20571 | frozen | 288/512 (56.25%) | 512/512 (100.00%) | 464/512 (90.62%) |
| held-offdiagonal | 20571 | ordinary | 256/512 (50.00%) | 508/512 (99.22%) | 472/512 (92.19%) |
| held-offdiagonal | 20571 | process | 248/512 (48.44%) | 512/512 (100.00%) | 424/512 (82.81%) |

Correct absolute phase labels are scored without flipping latent channels to fit results. They are not equivalent to useful event ranking or a correct native answer.

## Training adequacy

| Seed | Arm | Trainable route parameters | Updates | Event-choice presentations | Added locator labels | Exact TRAIN rows | Strict TRAIN phase pairs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20371 | ordinary | 16,158 | 384 | 49,152 | 0 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20371 | process | 16,158 | 384 | 49,152 | 294,912 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | ordinary | 16,158 | 384 | 49,152 | 0 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | process | 16,158 | 384 | 49,152 | 294,912 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | ordinary | 16,158 | 384 | 49,152 | 0 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | process | 16,158 | 384 | 49,152 | 294,912 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |

## Secondary within-checkpoint diagnostics

| Pane | Seed | Fit | Fixed named locator joint alignment (distinct rows) | Role swap hard choice changes (distinct rows) | Both choices reversed among aligned initially correct pairs |
|---|---:|---|---:|---:|---:|
| held-familiar | 20371 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-familiar | 20371 | process | 512/512 (100.00%) | 0/512 (0.00%) | 0/256 (0.00%) |
| held-familiar | 20471 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-familiar | 20471 | process | 368/512 (71.88%) | 0/512 (0.00%) | 0/184 (0.00%) |
| held-familiar | 20571 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-familiar | 20571 | process | 420/512 (82.03%) | 0/512 (0.00%) | 0/130 (0.00%) |
| held-covered | 20371 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-covered | 20371 | process | 512/512 (100.00%) | 0/512 (0.00%) | 0/256 (0.00%) |
| held-covered | 20471 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-covered | 20471 | process | 392/512 (76.56%) | 0/512 (0.00%) | 0/196 (0.00%) |
| held-covered | 20571 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-covered | 20571 | process | 380/512 (74.22%) | 0/512 (0.00%) | 0/180 (0.00%) |
| held-offdiagonal | 20371 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-offdiagonal | 20371 | process | 512/512 (100.00%) | 0/512 (0.00%) | 0/256 (0.00%) |
| held-offdiagonal | 20471 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-offdiagonal | 20471 | process | 368/512 (71.88%) | 0/512 (0.00%) | 0/184 (0.00%) |
| held-offdiagonal | 20571 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-offdiagonal | 20571 | process | 376/512 (73.44%) | 0/512 (0.00%) | 0/164 (0.00%) |

The swap edits learned action and trigger summaries and positions, recomputing the weighted mixture and signed distance, with unchanged symmetric product, cue, base W, query and readout. This is an internal perturbation, not a text edit, and cannot establish what a model would do on an arbitrary edited sentence. The other five alignments and position checks are archived.


The full report and private replay retain the sealed methodology, measurement, independent recount and paired root contrasts. This controlled split-view task does not demonstrate new natural-language syntax, learned segmentation, general memory binding or matched-quality serving efficiency.

## Decision

Locator supervision taught the auxiliary head to point to the named tokens
but did not establish that those roles govern selection. The ordinary arm
never aligned all three named positions for both events on the weak seed's
mixed views. Process supervision reached 380/512 and 376/512 distinct
cases, below the 90% sensor floor. Conditioning on examples that had both
the original choices correct and all spans matched, its action/trigger swap
reversed none of 180 covered and none of 164 off-diagonal before/after
pairs. In fact no hard event choice changed in any of the 27,648 expanded
pane/seed rows for the process arm, although nearly all scores changed.
These are repeated observations, not independent experimental units.

Frozen W / ordinary / process strict route pairs on the weak seed were
246 / 222 / 224 of 256 on the covered pane and 232 / 232 / 212 on the
off-diagonal pane. The measured differences against ordinary were +0.78 pp
(root interval -4.69 to +7.03) and -7.81 pp (-14.84 to -0.78). No arm passed
all registered route or native criteria. Although the auxiliary
locator model learned some exact labels, neither it nor the simpler matched
model earned promotion in this run.

A post-score diagnostic subtracts the recorded residual from the stored
event phase logits, reconstructs the exact-arithmetic comparison sign, and
excludes near ties under the declared rounding guard. It found only 16
expanded choice differences in the process arm, all for weak-seed
off-diagonal cases, and none in the ordinary arm. This is a numerical
reconstruction from recorded outputs, not a separate actor ablation.
It cannot exclude an effect of the auxiliary labels on shared W
representations during training. It shows why an attention map is not
evidence that final decisions use the assigned roles.

I would close this auxiliary branch as a proposed repair. A stronger test
requires the final route to consume the bound action, trigger and cue
through an explicit composition path instead of giving an already capable
W scorer an unconstrained residual option. Compare that against frozen W
at matched competence and demand native counterfactual transfer before
porting it into memory writes or language modeling.
