# COG-BIND-TIME-PHASE-003: event chronology supervision

**Outcome:** frozen: temporal FAIL, native FAIL, ordinary: temporal FAIL, native FAIL, phaseaux: temporal FAIL, native FAIL.
**Predeclared specific chronology-supervision win:** FAIL.

## Design

COVER-002 found that mixed-layout training helped relative to homogeneous continuation but remained below frozen W on mixed panes. PHASE-003 holds the underlying mixed TRAIN input byte-identical and adds only 0.35× event-phase cross entropy over the two actual events, with before=0 and after=1. Both fitted arms start at the same W checkpoint per seed and use identical route target presentations and optimizer budgets. Extra TRAIN-only chronology labels are counted separately. No query-phase label, explicit checker, gold test address or oracle is used by the model. Y role selection and the Z noun reader stay frozen and shared.

The matched lay-out families are fresh and independent by root. Covered pairs 0/2 and 1/3 are present in both FIT projections; off-diagonal 0/3 and 1/2 are withheld from both fits but have been evaluated historically. The experiment does not test unfamiliar language, general syntax, or a single unified input without declared boundaries.

## Exact, strict outcomes by seed

| Pane | Seed | Arm | Distinct W event rows | Distinct strict before/after pairs | Distinct strict chronology pairs | Complete W temporal octets | Native exact event+phrase+pointer |
|---|---:|---|---:|---:|---:|---:|---:|
| held-familiar | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,036/2,048 (99.41%) |
| held-familiar | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,036/2,048 (99.41%) |
| held-familiar | 20371 | phaseaux | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,036/2,048 (99.41%) |
| held-familiar | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20471 | phaseaux | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-familiar | 20571 | frozen | 372/512 (72.66%) | 186/256 (72.66%) | 184/256 (71.88%) | 46/64 (71.88%) | 1,488/2,048 (72.66%) |
| held-familiar | 20571 | ordinary | 380/512 (74.22%) | 190/256 (74.22%) | 180/256 (70.31%) | 45/64 (70.31%) | 1,520/2,048 (74.22%) |
| held-familiar | 20571 | phaseaux | 356/512 (69.53%) | 176/256 (68.75%) | 168/256 (65.62%) | 42/64 (65.62%) | 1,424/2,048 (69.53%) |
| held-covered | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,036/2,048 (99.41%) |
| held-covered | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,036/2,048 (99.41%) |
| held-covered | 20371 | phaseaux | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,036/2,048 (99.41%) |
| held-covered | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20471 | phaseaux | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-covered | 20571 | frozen | 452/512 (88.28%) | 226/256 (88.28%) | 200/256 (78.12%) | 50/64 (78.12%) | 1,808/2,048 (88.28%) |
| held-covered | 20571 | ordinary | 416/512 (81.25%) | 208/256 (81.25%) | 180/256 (70.31%) | 45/64 (70.31%) | 1,664/2,048 (81.25%) |
| held-covered | 20571 | phaseaux | 412/512 (80.47%) | 204/256 (79.69%) | 178/256 (69.53%) | 44/64 (68.75%) | 1,648/2,048 (80.47%) |
| held-offdiagonal | 20371 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,036/2,048 (99.41%) |
| held-offdiagonal | 20371 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,036/2,048 (99.41%) |
| held-offdiagonal | 20371 | phaseaux | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,036/2,048 (99.41%) |
| held-offdiagonal | 20471 | frozen | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20471 | ordinary | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20471 | phaseaux | 512/512 (100.00%) | 256/256 (100.00%) | 256/256 (100.00%) | 64/64 (100.00%) | 2,048/2,048 (100.00%) |
| held-offdiagonal | 20571 | frozen | 452/512 (88.28%) | 226/256 (88.28%) | 216/256 (84.38%) | 54/64 (84.38%) | 1,808/2,048 (88.28%) |
| held-offdiagonal | 20571 | ordinary | 412/512 (80.47%) | 206/256 (80.47%) | 184/256 (71.88%) | 46/64 (71.88%) | 1,648/2,048 (80.47%) |
| held-offdiagonal | 20571 | phaseaux | 412/512 (80.47%) | 204/256 (79.69%) | 186/256 (72.66%) | 46/64 (71.88%) | 1,648/2,048 (80.47%) |

There are 64 fresh roots per held pane, eight distinct route contexts per root and four matched downstream cases per route. No independence claim for aligned panes, seeds, or repeated route rows. A native pointer must name the correct event and phrase as well as the correct token. Ties fail and phase interventions remain diagnostics.

## Mechanism contrast and familiar robustness

| Seed | Phase supervision vs ordinary on covered strict pairs, pp [root 95% CI] | Off-diagonal strict pairs, pp [root 95% CI] | Weakest familiar layout: frozen / ordinary / phase supervision strict pairs |
|---:|---:|---:|---|
| 20371 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) |
| 20471 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) / 0_0: 64/64 (100.00%) |
| 20571 | -1.56 [-6.25, +3.12] | -0.78 [-6.25, +4.69] | 3_3: 22/64 (34.38%) / 2_2: 38/64 (59.38%) / 3_3: 22/64 (34.38%) |

The registered 80% familiar per-layout flag is separate from aggregate qualification. No mean or gate supersedes a weak layout.

## Separating phase labels from ranking

| Pane | Seed | Arm | Both actual event phases exactly labeled | Query phase exactly labeled | Event rank with supplied gold query phase (diagnostic only) |
|---|---:|---|---:|---:|---:|
| held-familiar | 20371 | frozen | 480/512 (93.75%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20371 | ordinary | 496/512 (96.88%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20371 | phaseaux | 492/512 (96.09%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20471 | frozen | 484/512 (94.53%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20471 | ordinary | 480/512 (93.75%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20471 | phaseaux | 512/512 (100.00%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-familiar | 20571 | frozen | 316/512 (61.72%) | 512/512 (100.00%) | 372/512 (72.66%) |
| held-familiar | 20571 | ordinary | 212/512 (41.41%) | 512/512 (100.00%) | 380/512 (74.22%) |
| held-familiar | 20571 | phaseaux | 272/512 (53.12%) | 508/512 (99.22%) | 356/512 (69.53%) |
| held-covered | 20371 | frozen | 476/512 (92.97%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20371 | ordinary | 472/512 (92.19%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20371 | phaseaux | 472/512 (92.19%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20471 | frozen | 488/512 (95.31%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20471 | ordinary | 480/512 (93.75%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20471 | phaseaux | 508/512 (99.22%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-covered | 20571 | frozen | 304/512 (59.38%) | 512/512 (100.00%) | 452/512 (88.28%) |
| held-covered | 20571 | ordinary | 208/512 (40.62%) | 512/512 (100.00%) | 416/512 (81.25%) |
| held-covered | 20571 | phaseaux | 268/512 (52.34%) | 508/512 (99.22%) | 408/512 (79.69%) |
| held-offdiagonal | 20371 | frozen | 476/512 (92.97%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20371 | ordinary | 472/512 (92.19%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20371 | phaseaux | 472/512 (92.19%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20471 | frozen | 484/512 (94.53%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20471 | ordinary | 492/512 (96.09%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20471 | phaseaux | 512/512 (100.00%) | 512/512 (100.00%) | 512/512 (100.00%) |
| held-offdiagonal | 20571 | frozen | 304/512 (59.38%) | 512/512 (100.00%) | 452/512 (88.28%) |
| held-offdiagonal | 20571 | ordinary | 228/512 (44.53%) | 512/512 (100.00%) | 412/512 (80.47%) |
| held-offdiagonal | 20571 | phaseaux | 264/512 (51.56%) | 508/512 (99.22%) | 408/512 (79.69%) |

Correct absolute phase labels are scored without flipping latent channels to fit results. They are not equivalent to useful event ranking or a correct native answer.

## Training adequacy

| Seed | Arm | Trainable route parameters | Updates | Event-choice presentations | Added event-phase labels | Exact TRAIN rows | Strict TRAIN phase pairs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20371 | ordinary | 15,304 | 384 | 49,152 | 0 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20371 | phaseaux | 15,304 | 384 | 49,152 | 98,304 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | ordinary | 15,304 | 384 | 49,152 | 0 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | phaseaux | 15,304 | 384 | 49,152 | 98,304 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | ordinary | 15,304 | 384 | 49,152 | 0 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | phaseaux | 15,304 | 384 | 49,152 | 98,304 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |

The full report and private replay retain the sealed methodology, measurement, independent recount and paired root contrasts. This controlled split-view task does not demonstrate new natural-language syntax, learned segmentation, general memory binding or matched-quality serving efficiency.

## Decision

The extra labels improved the weak seed's exact before/after classification
of both events from 208 to 268 of 512 distinct route cases in the covered
mix and 228 to 264 in the off-diagonal mix. Both values missed the 90% named
chronology criterion. They did not improve either final ranking endpoint:
strict before/after pairs moved 208 to 204 of 256 in the covered mix and 206
to 204 in the off-diagonal mix. Neither paired root interval rules out zero.
Frozen W was stronger on both mixes at 226/256, but also below the new gate.
The other two seeds tied at perfect event ranking. The three computed routes
used the same Y/Z readout; no fit modified noun spelling or event/role layout.

This comparison disproves neither a role-sensitive composition mechanism
nor a better training recipe. It rejects this added event-label objective at
the registered budget. It also reinforces that naming an event's phase and
using the relative ranking to choose the right event are different outcomes.
In the weak off-diagonal seed, the auxiliary fit parsed the query phase on
508/512 cases; its real route selected the right event on 412/512, and
supplying true query phase still gave only 408/512. Do not attribute this
gap to noun copying or to understanding the word "before."

Do not continue cycling loss weights on this fixed design. A genuinely new
candidate should test a learned event representation that explicitly binds
the trigger to the action and correctly composes cue direction with scope.
Use a small finite typed reference only as a diagnostic ceiling, fresh
held-out scope/wording families, the same frozen readout and matched controls.
A typed reference is not a learned-model claim. No additional candidate is
tested in this study.

The independent recount also checked the binary probability comparator: if
both actual event-phase labels and the requested phase are correct, the
event with matching phase must rank first in exact arithmetic. There were
no observed failures or ties in 68,416 full-wire rows satisfying that
condition (repeated rows, seeds and arms, not independent observations).
In the weak covered seed, the auxiliary model gained fully correct
phase-and-query classifications (208 to 268 distinct rows) but lost
correct ranks without correctly naming both phases (208 to 144), resulting
in 416 to 412 correct ranks overall. On the off-diagonal pane those parts
moved 228 to 264 and 184 to 148, leaving 412 correct ranks unchanged.
This is an observed stage partition, not proof of a training mediation.
