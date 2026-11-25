# COG-BIND-TIME-COVER-002: fixed-loss temporal layout coverage

**Outcome:** frozen: temporal FAIL, native FAIL, ordinary: temporal FAIL, native FAIL, coverage: temporal FAIL, native FAIL.
**Predeclared specific coverage win:** FAIL.

## Design

In TIME-001, both matched continuation objectives regressed compared to the original W router. COVER-002 holds the ordinary event-choice NLL, starting checkpoint, optimizer and root identities fixed. It replaces 50% of homogeneous TRAIN layouts with mixed direct/inverse pairs in one arm. Total raw words, per-shape exposure, phase targets, training updates and architecture stay matched. The treatment changes within-passage form co-occurrence, not the discrete labels or router architecture. Exact temporal predictions flow through the same frozen Y phrase selector and Z phrase-local noun reader across all arms.

Both mixed panes use fresh roots. Covered forms 0/2 and 1/3 were present in coverage TRAIN, not ordinary TRAIN. Off-diagonal pairs 0/3 and 1/2 were withheld from both current continuations, not globally unknown: historical studies previously evaluated those pairs. Primitive forms are known. Familiar and reader-only controls retain the original temporal layout. No oracle IDs or typed timing are supplied to the model.

## Exact, strict outcomes by seed

| Pane | Seed | Arm | Distinct W event rows | Distinct strict before/after pairs | Distinct strict chronology pairs | Complete W temporal octets | Native exact event+phrase+pointer |
|---|---:|---|---:|---:|---:|---:|---:|
| held-familiar | 20371 | frozen | 508/512 (99.22%) | 254/256 (99.22%) | 252/256 (98.44%) | 63/64 (98.44%) | 2,012/2,048 (98.24%) |
| held-familiar | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,028/2,048 (99.02%) |
| held-familiar | 20371 | coverage | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,028/2,048 (99.02%) |
| held-familiar | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20471 | coverage | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20571 | frozen | 440/512 (85.94%) | 220/256 (85.94%) | 204/256 (79.69%) | 51/64 (79.69%) | 1,760/2,048 (85.94%) |
| held-familiar | 20571 | ordinary | 428/512 (83.59%) | 214/256 (83.59%) | 204/256 (79.69%) | 51/64 (79.69%) | 1,712/2,048 (83.59%) |
| held-familiar | 20571 | coverage | 460/512 (89.84%) | 228/256 (89.06%) | 222/256 (86.72%) | 55/64 (85.94%) | 1,840/2,048 (89.84%) |
| held-covered | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,028/2,048 (99.02%) |
| held-covered | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,028/2,048 (99.02%) |
| held-covered | 20371 | coverage | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,028/2,048 (99.02%) |
| held-covered | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20471 | coverage | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20571 | frozen | 484/512 (94.53%) | 242/256 (94.53%) | 240/256 (93.75%) | 60/64 (93.75%) | 1,936/2,048 (94.53%) |
| held-covered | 20571 | ordinary | 404/512 (78.91%) | 202/256 (78.91%) | 176/256 (68.75%) | 44/64 (68.75%) | 1,616/2,048 (78.91%) |
| held-covered | 20571 | coverage | 454/512 (88.67%) | 224/256 (87.50%) | 204/256 (79.69%) | 50/64 (78.12%) | 1,816/2,048 (88.67%) |
| held-offdiagonal | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,028/2,048 (99.02%) |
| held-offdiagonal | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,028/2,048 (99.02%) |
| held-offdiagonal | 20371 | coverage | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,028/2,048 (99.02%) |
| held-offdiagonal | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20471 | coverage | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20571 | frozen | 464/512 (90.62%) | 232/256 (90.62%) | 224/256 (87.50%) | 56/64 (87.50%) | 1,856/2,048 (90.62%) |
| held-offdiagonal | 20571 | ordinary | 412/512 (80.47%) | 206/256 (80.47%) | 188/256 (73.44%) | 47/64 (73.44%) | 1,648/2,048 (80.47%) |
| held-offdiagonal | 20571 | coverage | 460/512 (89.84%) | 228/256 (89.06%) | 216/256 (84.38%) | 54/64 (84.38%) | 1,840/2,048 (89.84%) |

There are 64 fresh roots per held pane, eight distinct route contexts per root and four matched downstream cases per route. No independence claim for aligned panes, seeds, or repeated route rows. A native pointer must name the correct event and phrase as well as the correct token. Ties fail and phase interventions remain diagnostics.

## Mechanism contrast and familiar robustness

| Seed | Coverage vs ordinary on covered strict pairs, pp [root 95% CI] | Off-diagonal strict pairs, pp [root 95% CI] | Weakest familiar layout: frozen / ordinary / coverage strict pairs |
|---:|---:|---:|---|
| 20371 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0_0: 62/64 (96.88%) / 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) |
| 20471 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) |
| 20571 | +8.59 [+3.12, +14.84] | +8.59 [+3.12, +14.06] | 3_3: 42/64 (65.62%) / 2_2: 48/64 (75.00%) / 0_0: 54/64 (84.38%) |

The registered 80% familiar per-layout flag is separate from aggregate qualification. No mean or gate supersedes a weak layout.

## Training adequacy

| Seed | Arm | Trainable route parameters | Updates | Target presentations | Exact TRAIN rows | Strict TRAIN phase pairs |
|---:|---|---:|---:|---:|---:|---:|
| 20371 | ordinary | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20371 | coverage | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | ordinary | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | coverage | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | ordinary | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | coverage | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |

The full report and private replay retain the sealed methodology, measurement, independent recount and paired root contrasts. This controlled split-view task does not demonstrate new natural-language syntax, learned segmentation, general memory binding or matched-quality serving efficiency.

## Interpretation and decision

The mixed-layout continuation improved the weak seed's strict before/after
accuracy over equal-budget ordinary continuation in both mixed panes: 202 to
224 and 206 to 228 of 256, both +8.59 percentage points. Fixed-root 95%
intervals were [+3.13, +14.84] and [+3.13, +14.06]. The other two seeds had
no room to improve. Coverage also improved weak-seed familiar routing from
440/512 frozen and 428/512 ordinary to 460/512, with all four familiar
layout cells now over the separate 80% pair floor. This differs from the
familiar regressions of TIME-001 but is not evidence of general language
robustness.

Coverage did not beat the frozen model's mixed-pane pair accuracy (242/256
and 232/256), and no arm qualified across all cohorts and seeds on the
stricter two-pane task. Keep the frozen selector as the mixed-pane reference.
Do not promote a coverage-only repair or use perfect TRAIN competence to
waive a held-out gate. The confidence intervals above are contrasts against
the ordinary matched update and not superiority over frozen W.

The learned requested phase was almost always correct. In the off-diagonal
weak-seed case, coverage raised exact event ranking from 412 to 460 of 512
relative to ordinary training while both event phase labels classified
correctly actually fell from 288 to 240. Normalized event ranking with gold
query phase was 460/512 for coverage and 464/512 for frozen. More successful
comparison is not the same as learning a faithful canonical phase label.
This warrants inspecting event-side composition and preserving the original
router; it does not justify more of the same continuation. Test any
chronology-specific repair prospectively with fresh roots, per-layout
preservation and the fixed downstream exact-source gate.
