# COG-BIND-TIME-COMP-006: no-bypass event composition

**Outcome:** frozen: temporal FAIL, native FAIL, ordinary: temporal PASS, native PASS, process: temporal FAIL, native FAIL.
**Predeclared specific process-supervision win:** FAIL.

## Design

PROC-005 learned visible role localization but swapping the assigned roles did not change any choices in its residual architecture. COMP-006 removes the separate W event-phase/pooling output entirely from both matched fitted arms. The learned scoped composition now supplies all event-phase logits. The original W query scorer, encoders, start checkpoints and frozen downstream Y/Z reader are controlled. The no-update original W ranker is an external anchor, not an inference fallback; the fitted heads start with tied event scores, not old W ranks. Both optimized arms receive identical raw inputs and base event-choice targets. Process additionally receives TRAIN-only action/trigger/cue locator loss with weight 0.35. No labeled spans enter the actual predictor.

New prompt and canonical root IDs are disjoint from all ancestors. FIT reuses the same already migrated actor-name bank as PROC and no historical root. Covered and off-diagonal mixed pairs refer to train presence in this protocol; their broader grammar patterns were historically exposed. The shared local token encoder can already encode sentence-wide information into a located token feature. Removing the old rank bypass does not force each latent slot to contain only its named role.

## Exact, strict outcomes by seed

| Pane | Seed | Arm | Distinct W event rows | Distinct strict before/after pairs | Distinct strict chronology pairs | Complete W temporal octets | Native exact event+phrase+pointer |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| held-familiar | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,020/2,048 (98.63%) |
| held-familiar | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,020/2,048 (98.63%) |
| held-familiar | 20371 | process | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,020/2,048 (98.63%) |
| held-familiar | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20471 | ordinary | 508/512 (99.22%) | 254/256 (99.22%) | 252/256 (98.44%) | 63/64 (98.44%) | 2,032/2,048 (99.22%) |
| held-familiar | 20471 | process | 500/512 (97.66%) | 250/256 (97.66%) | 248/256 (96.88%) | 62/64 (96.88%) | 2,000/2,048 (97.66%) |
| held-familiar | 20571 | frozen | 436/512 (85.16%) | 218/256 (85.16%) | 212/256 (82.81%) | 53/64 (82.81%) | 1,744/2,048 (85.16%) |
| held-familiar | 20571 | ordinary | 504/512 (98.44%) | 252/256 (98.44%) | 252/256 (98.44%) | 63/64 (98.44%) | 2,016/2,048 (98.44%) |
| held-familiar | 20571 | process | 484/512 (94.53%) | 242/256 (94.53%) | 240/256 (93.75%) | 60/64 (93.75%) | 1,936/2,048 (94.53%) |
| held-covered | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,020/2,048 (98.63%) |
| held-covered | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,020/2,048 (98.63%) |
| held-covered | 20371 | process | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,020/2,048 (98.63%) |
| held-covered | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20471 | ordinary | 504/512 (98.44%) | 252/256 (98.44%) | 248/256 (96.88%) | 62/64 (96.88%) | 2,016/2,048 (98.44%) |
| held-covered | 20471 | process | 484/512 (94.53%) | 242/256 (94.53%) | 228/256 (89.06%) | 57/64 (89.06%) | 1,936/2,048 (94.53%) |
| held-covered | 20571 | frozen | 472/512 (92.19%) | 236/256 (92.19%) | 224/256 (87.50%) | 56/64 (87.50%) | 1,888/2,048 (92.19%) |
| held-covered | 20571 | ordinary | 500/512 (97.66%) | 250/256 (97.66%) | 244/256 (95.31%) | 61/64 (95.31%) | 2,000/2,048 (97.66%) |
| held-covered | 20571 | process | 492/512 (96.09%) | 246/256 (96.09%) | 240/256 (93.75%) | 60/64 (93.75%) | 1,968/2,048 (96.09%) |
| held-offdiagonal | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,020/2,048 (98.63%) |
| held-offdiagonal | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,020/2,048 (98.63%) |
| held-offdiagonal | 20371 | process | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,020/2,048 (98.63%) |
| held-offdiagonal | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20471 | ordinary | 504/512 (98.44%) | 252/256 (98.44%) | 248/256 (96.88%) | 62/64 (96.88%) | 2,016/2,048 (98.44%) |
| held-offdiagonal | 20471 | process | 492/512 (96.09%) | 246/256 (96.09%) | 236/256 (92.19%) | 59/64 (92.19%) | 1,968/2,048 (96.09%) |
| held-offdiagonal | 20571 | frozen | 472/512 (92.19%) | 236/256 (92.19%) | 224/256 (87.50%) | 56/64 (87.50%) | 1,888/2,048 (92.19%) |
| held-offdiagonal | 20571 | ordinary | 488/512 (95.31%) | 244/256 (95.31%) | 232/256 (90.62%) | 58/64 (90.62%) | 1,952/2,048 (95.31%) |
| held-offdiagonal | 20571 | process | 476/512 (92.97%) | 238/256 (92.97%) | 232/256 (90.62%) | 58/64 (90.62%) | 1,904/2,048 (92.97%) |

There are 64 fresh roots per held pane, eight distinct route contexts per root and four matched downstream cases per route. No independence claim for aligned panes, seeds, or repeated route rows. A native pointer must name the correct event and phrase as well as the correct token. Ties fail and phase interventions remain diagnostics.

## Mechanism contrast and familiar robustness

| Seed | Process supervision vs ordinary no-bypass on covered strict pairs, pp [root 95% CI] | Off-diagonal strict pairs, pp [root 95% CI] | Weakest familiar layout: frozen / ordinary / process strict pairs |
|---:|---:|---:|---|
| 20371 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) |
| 20471 | -3.91 [-8.59, +0.78] | -2.34 [-6.25, +1.56] | 0_0: 64/64 (100.00%) / 1_1: 62/64 (96.88%) / 0_0: 58/64 (90.62%) |
| 20571 | -1.56 [-5.47, +2.34] | -2.34 [-7.03, +2.34] | 3_3: 38/64 (59.38%) / 2_2: 60/64 (93.75%) / 2_2: 56/64 (87.50%) |

The registered 80% familiar per-layout flag is separate from aggregate qualification. No mean or gate supersedes a weak layout.

## Separating phase labels from ranking

| Pane | Seed | Arm | Both actual event phases exactly labeled | Query phase exactly labeled | Event rank with supplied gold query phase (diagnostic only) |
|---|---:|---|---:|---:|---:|
| held-familiar | 20371 | frozen | 488/512 (95.31%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20371 | ordinary | 484/512 (94.53%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20371 | process | 512/512 (100.00%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20471 | frozen | 480/512 (93.75%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20471 | ordinary | 448/512 (87.50%) | 512/512 (100.00%) | 508/512 (99.22%) |
| held-familiar | 20471 | process | 480/512 (93.75%) | 512/512 (100.00%) | 500/512 (97.66%) |
| held-familiar | 20571 | frozen | 324/512 (63.28%) | 512/512 (100.00%) | 436/512 (85.16%) |
| held-familiar | 20571 | ordinary | 412/512 (80.47%) | 512/512 (100.00%) | 504/512 (98.44%) |
| held-familiar | 20571 | process | 328/512 (64.06%) | 512/512 (100.00%) | 484/512 (94.53%) |
| held-covered | 20371 | frozen | 492/512 (96.09%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20371 | ordinary | 472/512 (92.19%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20371 | process | 512/512 (100.00%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20471 | frozen | 476/512 (92.97%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20471 | ordinary | 448/512 (87.50%) | 512/512 (100.00%) | 504/512 (98.44%) |
| held-covered | 20471 | process | 464/512 (90.62%) | 512/512 (100.00%) | 484/512 (94.53%) |
| held-covered | 20571 | frozen | 296/512 (57.81%) | 512/512 (100.00%) | 472/512 (92.19%) |
| held-covered | 20571 | ordinary | 420/512 (82.03%) | 512/512 (100.00%) | 500/512 (97.66%) |
| held-covered | 20571 | process | 336/512 (65.62%) | 512/512 (100.00%) | 492/512 (96.09%) |
| held-offdiagonal | 20371 | frozen | 492/512 (96.09%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20371 | ordinary | 472/512 (92.19%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20371 | process | 512/512 (100.00%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20471 | frozen | 472/512 (92.19%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20471 | ordinary | 444/512 (86.72%) | 512/512 (100.00%) | 504/512 (98.44%) |
| held-offdiagonal | 20471 | process | 456/512 (89.06%) | 512/512 (100.00%) | 492/512 (96.09%) |
| held-offdiagonal | 20571 | frozen | 292/512 (57.03%) | 512/512 (100.00%) | 472/512 (92.19%) |
| held-offdiagonal | 20571 | ordinary | 432/512 (84.38%) | 512/512 (100.00%) | 488/512 (95.31%) |
| held-offdiagonal | 20571 | process | 352/512 (68.75%) | 512/512 (100.00%) | 476/512 (92.97%) |

Correct absolute phase labels are scored without flipping latent channels to fit results. They are not equivalent to useful event ranking or a correct native answer.

## Training adequacy

| Seed | Arm | Trainable route parameters | Updates | Event-choice presentations | Added locator labels | Exact TRAIN rows | Strict TRAIN phase pairs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20371 | ordinary | 14,932 | 384 | 49,152 | 0 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20371 | process | 14,932 | 384 | 49,152 | 294,912 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | ordinary | 14,932 | 384 | 49,152 | 0 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | process | 14,932 | 384 | 49,152 | 294,912 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | ordinary | 14,932 | 384 | 49,152 | 0 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | process | 14,932 | 384 | 49,152 | 294,912 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |

## Secondary within-checkpoint diagnostics

| Pane | Seed | Fit | Fixed named locator joint alignment (distinct rows) | Role swap hard choice changes (distinct rows) | Both choices reversed among aligned initially correct pairs |
|---|---:|---|---:|---:|---:|
| held-familiar | 20371 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-familiar | 20371 | process | 476/512 (92.97%) | 0/512 (0.00%) | 0/238 (0.00%) |
| held-familiar | 20471 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-familiar | 20471 | process | 300/512 (58.59%) | 8/512 (1.56%) | 0/150 (0.00%) |
| held-familiar | 20571 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-familiar | 20571 | process | 128/512 (25.00%) | 0/512 (0.00%) | 0/64 (0.00%) |
| held-covered | 20371 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-covered | 20371 | process | 484/512 (94.53%) | 0/512 (0.00%) | 0/242 (0.00%) |
| held-covered | 20471 | ordinary | 0/512 (0.00%) | 4/512 (0.78%) | n/a (no support) |
| held-covered | 20471 | process | 360/512 (70.31%) | 0/512 (0.00%) | 0/180 (0.00%) |
| held-covered | 20571 | ordinary | 0/512 (0.00%) | 4/512 (0.78%) | n/a (no support) |
| held-covered | 20571 | process | 84/512 (16.41%) | 4/512 (0.78%) | 0/42 (0.00%) |
| held-offdiagonal | 20371 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-offdiagonal | 20371 | process | 468/512 (91.41%) | 0/512 (0.00%) | 0/234 (0.00%) |
| held-offdiagonal | 20471 | ordinary | 0/512 (0.00%) | 0/512 (0.00%) | n/a (no support) |
| held-offdiagonal | 20471 | process | 288/512 (56.25%) | 0/512 (0.00%) | 0/144 (0.00%) |
| held-offdiagonal | 20571 | ordinary | 0/512 (0.00%) | 52/512 (10.16%) | n/a (no support) |
| held-offdiagonal | 20571 | process | 100/512 (19.53%) | 4/512 (0.78%) | 0/50 (0.00%) |

The swap edits learned action and trigger summaries and positions, recomputing the weighted mixture and signed distance, with unchanged symmetric product, cue, base W, query and readout. This is an internal perturbation, not a text edit, and cannot establish what a model would do on an arbitrary edited sentence. The other five alignments and position checks are archived.


The full report and private replay retain the sealed methodology, measurement, independent recount and paired root contrasts. This controlled split-view task does not demonstrate new natural-language syntax, learned segmentation, general memory binding or matched-quality serving efficiency.

## Decision and source-audit limit

The ordinary no-bypass composer passes this bounded temporal and exact
downstream benchmark in all three seeds, without using the legacy W event
head. On weak 20571, frozen W vs ordinary gives 236→250/256 strict
before/after pairs on covered layouts and 236→244/256 on off-diagonal.
Locator supervision gives 246 and 238, and fails the off-diagonal absolute
route threshold. It also reverses none of the 42 and 50 originally correct
name-aligned weak pairs. The ordinary arm's success does not identify the
learned slots as genuine action/trigger/cue operands.

A source audit found the distinction: the inherited W encoder broadcasts
the mean of each event into every token and runs three contextual
convolutions. Therefore identifying a named token position does not
constrain what information crosses the slot interface. An inert audit
changed only the last word in an eight-token historical TRAIN event;
the untouched first token's contextual vector changed. A valid cue-only
weighting of the implemented composition head can flip its output without
using either named operand and is invariant to swapping the operands.
The fixed binary query rank can only preserve or reverse one event order.
Finally, the historical grammar admits a deterministic rule using the
temporal marker and whether the following word is 'the'. This run is a
valid bounded temporal routing result but not a faithful test of
compositional cue binding, query-dependent relation choice, nor broad
language reasoning. Keep the original run immutable as evidence; redesign
the payload interface and benchmark before making stronger claims.
