# COG-BIND-TIME-SCOPE-004: learned relative scope in event selection

**Outcome:** frozen: temporal PASS, native PASS, ordinary: temporal FAIL, native FAIL, scope: temporal FAIL, native FAIL.
**Predeclared specific scope design win:** FAIL.

## Design

PHASE-003 improved absolute phase classification without improving event rank. SCOPE-004 compares frozen W with two new parameter-identical residual heads trained at the same budget from the same W seed. Anonymous soft token summaries use only the raw event region. The matched pooling arm uses a weighted aggregate square and its weighted mean position; the scope arm uses a product of two distinct learned summaries and their signed expected position distance. This is one bundled architecture contrast, not a one-factor effect of distance alone. Both residual outputs start at zero and modify the event phase logits. No auxiliary phase, role or marker supervision; W still uses learned query phase and normalized comparison. Frozen Y/Z readout is shared.

Covered pairs 0/2 and 1/3 are in both FIT projections; off-diagonal 0/3 and 1/2 are withheld from both current fits but historically evaluated. Root families are fresh by canonical and raw disjointness. The finite old lexical bank was exhausted for an entire new 128-family training cut, so six formerly development actor names are reserved for this training split, without reusing old roots or moving held names. Current splits have distinct actor names. This is not a general new-language test.

## Exact, strict outcomes by seed

| Pane | Seed | Arm | Distinct W event rows | Distinct strict before/after pairs | Distinct strict chronology pairs | Complete W temporal octets | Native exact event+phrase+pointer |
|---|---:|---|---:|---:|---:|---:|---:|
| held-familiar | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,012/2,048 (98.24%) |
| held-familiar | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,012/2,048 (98.24%) |
| held-familiar | 20371 | scope | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,012/2,048 (98.24%) |
| held-familiar | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20471 | scope | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20571 | frozen | 444/512 (86.72%) | 222/256 (86.72%) | 212/256 (82.81%) | 53/64 (82.81%) | 1,776/2,048 (86.72%) |
| held-familiar | 20571 | ordinary | 464/512 (90.62%) | 232/256 (90.62%) | 232/256 (90.62%) | 58/64 (90.62%) | 1,856/2,048 (90.62%) |
| held-familiar | 20571 | scope | 464/512 (90.62%) | 232/256 (90.62%) | 232/256 (90.62%) | 58/64 (90.62%) | 1,856/2,048 (90.62%) |
| held-covered | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,012/2,048 (98.24%) |
| held-covered | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,012/2,048 (98.24%) |
| held-covered | 20371 | scope | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,012/2,048 (98.24%) |
| held-covered | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20471 | scope | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20571 | frozen | 488/512 (95.31%) | 244/256 (95.31%) | 232/256 (90.62%) | 58/64 (90.62%) | 1,952/2,048 (95.31%) |
| held-covered | 20571 | ordinary | 476/512 (92.97%) | 238/256 (92.97%) | 228/256 (89.06%) | 57/64 (89.06%) | 1,904/2,048 (92.97%) |
| held-covered | 20571 | scope | 476/512 (92.97%) | 238/256 (92.97%) | 228/256 (89.06%) | 57/64 (89.06%) | 1,904/2,048 (92.97%) |
| held-offdiagonal | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,012/2,048 (98.24%) |
| held-offdiagonal | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,012/2,048 (98.24%) |
| held-offdiagonal | 20371 | scope | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,012/2,048 (98.24%) |
| held-offdiagonal | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20471 | scope | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20571 | frozen | 488/512 (95.31%) | 244/256 (95.31%) | 240/256 (93.75%) | 60/64 (93.75%) | 1,952/2,048 (95.31%) |
| held-offdiagonal | 20571 | ordinary | 484/512 (94.53%) | 242/256 (94.53%) | 236/256 (92.19%) | 59/64 (92.19%) | 1,936/2,048 (94.53%) |
| held-offdiagonal | 20571 | scope | 484/512 (94.53%) | 242/256 (94.53%) | 236/256 (92.19%) | 59/64 (92.19%) | 1,936/2,048 (94.53%) |

There are 64 fresh roots per held pane, eight distinct route contexts per root and four matched downstream cases per route. No independence claim for aligned panes, seeds, or repeated route rows. A native pointer must name the correct event and phrase as well as the correct token. Ties fail and phase interventions remain diagnostics.

## Mechanism contrast and familiar robustness

| Seed | Scope composition vs pooling on covered strict pairs, pp [root 95% CI] | Off-diagonal strict pairs, pp [root 95% CI] | Weakest familiar layout: frozen / pooling / scope strict pairs |
|---:|---:|---:|---|
| 20371 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) |
| 20471 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) |
| 20571 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 3_3: 46/64 (71.88%) / 2_2: 52/64 (81.25%) / 2_2: 52/64 (81.25%) |

The registered 80% familiar per-layout flag is separate from aggregate qualification. No mean or gate supersedes a weak layout.

## Separating phase labels from ranking

| Pane | Seed | Arm | Both actual event phases exactly labeled | Query phase exactly labeled | Event rank with supplied gold query phase (diagnostic only) |
|---|---:|---|---:|---:|---:|
| held-familiar | 20371 | frozen | 472/512 (92.19%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20371 | ordinary | 464/512 (90.62%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20371 | scope | 508/512 (99.22%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20471 | frozen | 496/512 (96.88%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20471 | ordinary | 508/512 (99.22%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20471 | scope | 508/512 (99.22%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20571 | frozen | 348/512 (67.97%) | 512/512 (100.00%) | 444/512 (86.72%) |
| held-familiar | 20571 | ordinary | 368/512 (71.88%) | 512/512 (100.00%) | 464/512 (90.62%) |
| held-familiar | 20571 | scope | 368/512 (71.88%) | 512/512 (100.00%) | 464/512 (90.62%) |
| held-covered | 20371 | frozen | 488/512 (95.31%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20371 | ordinary | 480/512 (93.75%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20371 | scope | 508/512 (99.22%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20471 | frozen | 492/512 (96.09%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20471 | ordinary | 500/512 (97.66%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20471 | scope | 500/512 (97.66%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20571 | frozen | 296/512 (57.81%) | 512/512 (100.00%) | 488/512 (95.31%) |
| held-covered | 20571 | ordinary | 344/512 (67.19%) | 512/512 (100.00%) | 476/512 (92.97%) |
| held-covered | 20571 | scope | 344/512 (67.19%) | 512/512 (100.00%) | 476/512 (92.97%) |
| held-offdiagonal | 20371 | frozen | 488/512 (95.31%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20371 | ordinary | 468/512 (91.41%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20371 | scope | 508/512 (99.22%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20471 | frozen | 488/512 (95.31%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20471 | ordinary | 508/512 (99.22%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20471 | scope | 508/512 (99.22%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20571 | frozen | 336/512 (65.62%) | 512/512 (100.00%) | 488/512 (95.31%) |
| held-offdiagonal | 20571 | ordinary | 352/512 (68.75%) | 512/512 (100.00%) | 484/512 (94.53%) |
| held-offdiagonal | 20571 | scope | 352/512 (68.75%) | 512/512 (100.00%) | 484/512 (94.53%) |

Correct absolute phase labels are scored without flipping latent channels to fit results. They are not equivalent to useful event ranking or a correct native answer.

## Training adequacy

| Seed | Arm | Trainable route parameters | Updates | Event-choice presentations | Exact TRAIN rows | Strict TRAIN phase pairs |
|---:|---|---:|---:|---:|---:|---:|
| 20371 | ordinary | 16,158 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20371 | scope | 16,158 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | ordinary | 16,158 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | scope | 16,158 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | ordinary | 16,158 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | scope | 16,158 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |

## Secondary within-checkpoint diagnostics

| Pane | Seed | Fitted arm | Zero scalar-position strict pairs | Negated scalar-position strict pairs | Largest joint named locator alignment (among six post-score mappings) |
|---|---:|---|---:|---:|---:|
| held-familiar | 20371 | ordinary | 256/256 (100.00%) | 256/256 (100.00%) | 0/512 (0.00%) (mapping 012) |
| held-familiar | 20371 | scope | 256/256 (100.00%) | 256/256 (100.00%) | 12/512 (2.34%) (mapping 201) |
| held-familiar | 20471 | ordinary | 256/256 (100.00%) | 256/256 (100.00%) | 0/512 (0.00%) (mapping 012) |
| held-familiar | 20471 | scope | 256/256 (100.00%) | 256/256 (100.00%) | 0/512 (0.00%) (mapping 012) |
| held-familiar | 20571 | ordinary | 232/256 (90.62%) | 232/256 (90.62%) | 0/512 (0.00%) (mapping 012) |
| held-familiar | 20571 | scope | 232/256 (90.62%) | 232/256 (90.62%) | 0/512 (0.00%) (mapping 012) |
| held-covered | 20371 | ordinary | 256/256 (100.00%) | 256/256 (100.00%) | 0/512 (0.00%) (mapping 012) |
| held-covered | 20371 | scope | 256/256 (100.00%) | 256/256 (100.00%) | 0/512 (0.00%) (mapping 012) |
| held-covered | 20471 | ordinary | 256/256 (100.00%) | 256/256 (100.00%) | 0/512 (0.00%) (mapping 012) |
| held-covered | 20471 | scope | 256/256 (100.00%) | 256/256 (100.00%) | 0/512 (0.00%) (mapping 012) |
| held-covered | 20571 | ordinary | 238/256 (92.97%) | 238/256 (92.97%) | 0/512 (0.00%) (mapping 012) |
| held-covered | 20571 | scope | 238/256 (92.97%) | 238/256 (92.97%) | 0/512 (0.00%) (mapping 012) |
| held-offdiagonal | 20371 | ordinary | 256/256 (100.00%) | 256/256 (100.00%) | 0/512 (0.00%) (mapping 012) |
| held-offdiagonal | 20371 | scope | 256/256 (100.00%) | 256/256 (100.00%) | 0/512 (0.00%) (mapping 012) |
| held-offdiagonal | 20471 | ordinary | 256/256 (100.00%) | 256/256 (100.00%) | 0/512 (0.00%) (mapping 012) |
| held-offdiagonal | 20471 | scope | 256/256 (100.00%) | 256/256 (100.00%) | 0/512 (0.00%) (mapping 012) |
| held-offdiagonal | 20571 | ordinary | 242/256 (94.53%) | 242/256 (94.53%) | 0/512 (0.00%) (mapping 012) |
| held-offdiagonal | 20571 | scope | 242/256 (94.53%) | 242/256 (94.53%) | 0/512 (0.00%) (mapping 012) |

The largest observed locator fraction is descriptive and may capitalize on held-out outcomes; the full six mappings are archived. No selected mapping contributes to any gate. Centered mean and signed distance have different ranges and learned distributions; negation is not a dose-matched sham or a decomposition of the two-head contrast.


The full report and private replay retain the sealed methodology, measurement, independent recount and paired root contrasts. This controlled split-view task does not demonstrate new natural-language syntax, learned segmentation, general memory binding or matched-quality serving efficiency.

## Decision

Only original frozen W met the aggregate route and downstream exact-answer
gates on this panel. For the weakest seed its strict before/after event
pairs were 244/256 on both mixed panes. Both fitted heads reached 238/256
on the covered mix and 242/256 on the held-off-diagonal mix, and missed the
per-row threshold on both. Both improved familiar exact event rows from
444/512 to 464/512. The other two seeds tied at perfect event selection.

The scope and pooling heads made the same strict event selection on every
one of 27,648 expanded pane/seed cases. This is a repeated-row count, not
27,648 independent observations. Their scores did change numerically on
27,640 of those rows; the largest absolute scope residual on the recorded
phase logits was about 0.01016. Setting the scope scalar position feature
to zero or negating it changed no hard event choice. Anonymous locator
weights provided no evidence for joint named localization on either mixed
pane, under any of the six mappings. A head with nonzero effects on logits
need not alter behavior. These are diagnostic differences in this fitted
model, not a theorem about every learned scope operator.

Do not promote this residual or keep changing its width on the same task.
Across the recent studies, explicit chronology supervision and this
relative-scope architecture both failed to improve strict selection over
their matched controls. An actionable alternative would supply a training
signal at the action/trigger binding itself, then require unseen raw-linguistic
scope transfer and causal role interventions before crediting the mechanism.
The finite grammar scorer passes all held labels but does not count as a
learned improvement. This study does not settle general binding.
