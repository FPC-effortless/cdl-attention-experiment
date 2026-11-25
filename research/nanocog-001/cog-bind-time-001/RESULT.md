# COG-BIND-TIME-001: learned temporal event selection

**Scientific decision:** frozen: temporal PASS, native PASS, ordinary: temporal FAIL, native FAIL, paired: temporal FAIL, native FAIL. Preregistered specific win for the margin objective: **FAIL**.

## Why this comparison

On the exposed 001Z both-novel panel for seed 20571, 704 of 8,192 choices selected the wrong event. In 608 of these, the model classified the question's requested phase correctly but ranked the events wrong. Supplying the true query phase to the scorer still left 640 rank failures. The two component errors canceled in 32 cases, so their counts must not be added as disjoint causes.

TIME-001 compares frozen W, ordinary NLL continuation, and continuation with NLL plus a smooth worst-margin penalty over the equalized temporal family. Both fitted models see the same labels, raw text, starting checkpoint and optimizer schedule. Y role routing and Z phrase-local copy are frozen and reused byte-for-byte across arms. No oracle enters native prediction.

## Fresh held-novel strict outcomes

| Seed | Arm | Exact source event | Strict before/after pairs | Strict chronology pairs | Temporal octets | Native exact source+phrase+pointer |
|---:|---|---:|---:|---:|---:|---:|
| 20371 | frozen | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) | 1,024/1,024 (100.00%) | 256/256 (100.00%) | 1,988/2,048 (97.07%) |
| 20371 | ordinary | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) | 1,024/1,024 (100.00%) | 256/256 (100.00%) | 1,988/2,048 (97.07%) |
| 20371 | paired | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) | 1,024/1,024 (100.00%) | 256/256 (100.00%) | 1,988/2,048 (97.07%) |
| 20471 | frozen | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) | 1,024/1,024 (100.00%) | 256/256 (100.00%) | 2,048/2,048 (100.00%) |
| 20471 | ordinary | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) | 1,024/1,024 (100.00%) | 256/256 (100.00%) | 2,048/2,048 (100.00%) |
| 20471 | paired | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) | 1,024/1,024 (100.00%) | 256/256 (100.00%) | 2,048/2,048 (100.00%) |
| 20571 | frozen | 1,968/2,048 (96.09%) | 976/1,024 (95.31%) | 952/1,024 (92.97%) | 236/256 (92.19%) | 1,968/2,048 (96.09%) |
| 20571 | ordinary | 1,792/2,048 (87.50%) | 896/1,024 (87.50%) | 848/1,024 (82.81%) | 212/256 (82.81%) | 1,792/2,048 (87.50%) |
| 20571 | paired | 1,840/2,048 (89.84%) | 912/1,024 (89.06%) | 872/1,024 (85.16%) | 216/256 (84.38%) | 1,840/2,048 (89.84%) |

Each of the 64 independently generated held roots has eight unique temporal raw rows. The full scorer expands each through fixed downstream roles and noun swaps to 32 rows, so divide temporal-only counts by four to recover unique raw W cases. These are repeated, not independent evidence. Strict pairs and octets require every group member correct. Event rank ties fail.

## Objective contrast and parser diagnostics

| Seed | Paired minus ordinary phase-pair change on held-novel (pp, root 95% interval) | Frozen event accuracy on familiar | Ordinary event accuracy on familiar | Paired event accuracy on familiar |
|---:|---:|---:|---:|---:|
| 20371 | +0.00 [+0.00, +0.00] | 2,048/2,048 (100.00%) | 2,016/2,048 (98.44%) | 2,016/2,048 (98.44%) |
| 20471 | +0.00 [+0.00, +0.00] | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) |
| 20571 | +1.56 [-3.12, +5.47] | 1,760/2,048 (85.94%) | 1,568/2,048 (76.56%) | 1,680/2,048 (82.03%) |

| Seed | Arm | Correct query phase | Both event phases correct | Correct event if true query phase used | Correct event if true event chronology used |
|---:|---|---:|---:|---:|---:|
| 20371 | frozen | 2,048/2,048 (100.00%) | 1,872/2,048 (91.41%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) |
| 20371 | ordinary | 2,048/2,048 (100.00%) | 1,920/2,048 (93.75%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) |
| 20371 | paired | 2,048/2,048 (100.00%) | 1,936/2,048 (94.53%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) |
| 20471 | frozen | 2,048/2,048 (100.00%) | 1,936/2,048 (94.53%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) |
| 20471 | ordinary | 2,048/2,048 (100.00%) | 1,872/2,048 (91.41%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) |
| 20471 | paired | 2,048/2,048 (100.00%) | 1,856/2,048 (90.62%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) |
| 20571 | frozen | 2,032/2,048 (99.22%) | 944/2,048 (46.09%) | 1,984/2,048 (96.88%) | 2,032/2,048 (99.22%) |
| 20571 | ordinary | 2,048/2,048 (100.00%) | 832/2,048 (40.62%) | 1,792/2,048 (87.50%) | 2,048/2,048 (100.00%) |
| 20571 | paired | 2,032/2,048 (99.22%) | 960/2,048 (46.88%) | 1,856/2,048 (90.62%) | 2,032/2,048 (99.22%) |

Oracle phase diagnostics never replace native outcomes. Correct source-event provenance is required; a coincidental correct noun from the wrong source does not count.

## Training and limits

| Seed | Arm | Parameters | Optimizer updates | Target presentations | TRAIN exact rows | TRAIN phase pairs |
|---:|---|---:|---:|---:|---:|---:|
| 20371 | ordinary | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20371 | paired | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | ordinary | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20471 | paired | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | ordinary | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |
| 20571 | paired | 15,304 | 384 | 49,152 | 1,024/1,024 (100.00%) | 512/512 (100.00%) |

Time study holds the previously tested pointer fixed; it does not retest the original payload perturbation gate. All primitive grammar fragments were familiar to training. TIME tests reserved compositions of those fragments at a fixed split-view interface, not natural language or learned boundary discovery. Source, scoped data and blind exports were sealed; native run records and independent recount are preserved in the reproducibility package. No quality-adjusted inference claim.

## What to keep

Frozen W was the only method to meet the registered temporal and native gates
on this fresh, controlled panel, in all three seeds. In the predeclared weakest
seed, true distinct event rows declined 492/512 -> 448/512 under ordinary
continuation and 460/512 with the margin penalty. Strict before/after pairs
fell 244/256 -> 224/256 and 228/256. The margin arm regained four pairs over
ordinary continuation, an inconclusive +1.56 percentage points (fixed-seed
root interval -3.13 to +5.47); neither repair qualifies. All six fits were
perfect on the registered TRAIN endpoints. The other two seeds tied at perfect
novel event selection. This supports retaining W for this bounded test, not
claiming it fixed the failures on 001Z's other held patterns or the general case.

## Secondary coverage and layout check

The fresh matched-score aggregate still hides a weak layout. In seed 20571,
frozen W got only 64/128 distinct familiar route rows correct when both
clauses used form 3. Ordinary and paired continuations got 72/128 and 68/128
in that cell, but lost accuracy elsewhere. All four familiar form-pair cells
had exactly 256 unique TRAIN rows per layout, and the held deliberately mixed
layout pairs had zero exact TRAIN co-occurrences. This does not establish
why the rank fails or a gain from balancing: the aggregate fit changed the
allocation of successes, and even nominally balanced exposure is not proof of
behavioral coverage. The secondary audit leaves the frozen decisions unchanged.

Do not spend the next allocation on another scalar objective adjustment. Use a
fixed ordinary-loss anchor with matched exposures, vary representation-relevant
layout/collision coverage, and require per-layout counterfactual performance on
fresh roots and held compositions. Vary physical context, unseen prefixes and
delays separately from any diagnostic internal position offsets. The related
CSM language lane needs its own raw text/byte coverage audit and language-only
control before attributing cross-domain loss to forgetting. No CSM fit or
layout-shift run was executed in TIME-001.
