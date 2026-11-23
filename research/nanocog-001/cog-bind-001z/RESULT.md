# COG-BIND-001Z: phrase isolation before noun selection

**Isolated copy gate:** full-context did not pass; phrase-local passed. **Native end-to-end gate:** full-context did not pass; phrase-local did not pass. The preregistered locality-specific gain did not pass.

## What the experiment isolates

001Y showed that finding the correct phrase did not reliably recover the right noun. 001Z freezes the W temporal router and Y phrase router on the same pristine text in both arms. Two plausible noun candidates appear in each phrase of a separately expanded copy view. Both readers use the same parameters in number and form, eligibility mask and phrase-relative positions. The full-context reader encodes both phrases together; the phrase-local reader keeps them apart before feature mixing. Both score every eligible raw token for all four candidate phrases. No gold event, role or noun is supplied to the predictor. Training uses the correct event, phrase and item to supervise the item pointer. Neither reader learns the visible phrase boundary in this experiment.

The split routing and copy views deliberately hold the routing pathway independent of the expanded copy text. This tests the copy interface, not whether the end-to-end system can route from one unified text stream. The held conditions use fresh synthetic roots and novel combinations of familiar forms, not unrestricted natural language.

## Both-novel exact source-pointer outcomes

| Condition | Seed | Reader | Rows | Selected-payload pairs | Sibling-payload pairs | Crossed quartets |
|---|---:|---|---:|---:|---:|---:|
| Correct event + phrase (diagnostic) | 20371 | full-context | 7,680/8,192 (93.75%) | 1,776/2,048 (86.72%) | 1,904/2,048 (92.97%) | 888/1,024 (86.72%) |
| Correct event, learned phrase | 20371 | full-context | 7,488/8,192 (91.41%) | 1,712/2,048 (83.59%) | 1,840/2,048 (89.84%) | 856/1,024 (83.59%) |
| Native learned event + phrase | 20371 | full-context | 7,488/8,192 (91.41%) | 1,712/2,048 (83.59%) | 1,840/2,048 (89.84%) | 856/1,024 (83.59%) |
| Correct event + phrase (diagnostic) | 20371 | phrase-local | 8,064/8,192 (98.44%) | 1,984/2,048 (96.88%) | 2,016/2,048 (98.44%) | 992/1,024 (96.88%) |
| Correct event, learned phrase | 20371 | phrase-local | 7,872/8,192 (96.09%) | 1,920/2,048 (93.75%) | 1,952/2,048 (95.31%) | 960/1,024 (93.75%) |
| Native learned event + phrase | 20371 | phrase-local | 7,872/8,192 (96.09%) | 1,920/2,048 (93.75%) | 1,952/2,048 (95.31%) | 960/1,024 (93.75%) |
| Correct event + phrase (diagnostic) | 20471 | full-context | 8,192/8,192 (100.00%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) |
| Correct event, learned phrase | 20471 | full-context | 8,192/8,192 (100.00%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) |
| Native learned event + phrase | 20471 | full-context | 8,192/8,192 (100.00%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) |
| Correct event + phrase (diagnostic) | 20471 | phrase-local | 8,192/8,192 (100.00%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) |
| Correct event, learned phrase | 20471 | phrase-local | 8,192/8,192 (100.00%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) |
| Native learned event + phrase | 20471 | phrase-local | 8,192/8,192 (100.00%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) |
| Correct event + phrase (diagnostic) | 20571 | full-context | 7,764/8,192 (94.78%) | 1,932/2,048 (94.34%) | 1,932/2,048 (94.34%) | 964/1,024 (94.14%) |
| Correct event, learned phrase | 20571 | full-context | 7,764/8,192 (94.78%) | 1,932/2,048 (94.34%) | 1,932/2,048 (94.34%) | 964/1,024 (94.14%) |
| Native learned event + phrase | 20571 | full-context | 7,116/8,192 (86.87%) | 1,780/2,048 (86.91%) | 1,780/2,048 (86.91%) | 888/1,024 (86.72%) |
| Correct event + phrase (diagnostic) | 20571 | phrase-local | 8,192/8,192 (100.00%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) |
| Correct event, learned phrase | 20571 | phrase-local | 8,192/8,192 (100.00%) | 2,048/2,048 (100.00%) | 2,048/2,048 (100.00%) | 1,024/1,024 (100.00%) |
| Native learned event + phrase | 20571 | phrase-local | 7,488/8,192 (91.41%) | 1,888/2,048 (92.19%) | 1,888/2,048 (92.19%) | 944/1,024 (92.19%) |

The selected swap changes the correct emitted noun; the sibling swap must leave it unchanged. Crossed quartets require all four raw perturbation combinations to be correct. A tied pointer counts as incorrect. Each held view contains 64 independent roots shared across seeds and arms; rows/pairs/quartets are grouped within a root and must not be counted as independent replications.

## Locality effect and routing ceiling

| Seed | Isolated crossed-quartet difference, pp (root 95% interval) | Frozen event+phrase native row ceiling | Full native noun exact | Local native noun exact | Familiar isolated row gap, pp |
|---:|---:|---:|---:|---:|---:|
| 20371 | +10.16 [+4.30, +16.02] | 8,000/8,192 (97.66%) | 7,488/8,192 (91.41%) | 7,872/8,192 (96.09%) | +4.83 |
| 20471 | +0.00 [+0.00, +0.00] | 8,192/8,192 (100.00%) | 8,192/8,192 (100.00%) | 8,192/8,192 (100.00%) | -0.20 |
| 20571 | +5.86 [+2.34, +9.77] | 7,488/8,192 (91.41%) | 7,116/8,192 (86.87%) | 7,488/8,192 (91.41%) | +0.29 |

Native noun accuracy is shown separately from identifying its exact source. Neither diagnostic that supplies the correct event or phrase is credited as native output. Local-copy isolation was checked by changing the other phrase while retaining the target phrase and its positions; the frozen routing outputs were checked for bit parity across both arms.

## Matched training and validation

| Seed | Arm | Parameters | Optimizer updates | Item-target presentations | TRAIN exact pointer rows | TRAIN complete toggle sets |
|---:|---|---:|---:|---:|---:|---:|
| 20371 | full-context | 12,927 | 384 | 49,152 | 16,384/16,384 (100.00%) | 4,096/4,096 (100.00%) |
| 20371 | phrase-local | 12,927 | 384 | 49,152 | 16,384/16,384 (100.00%) | 4,096/4,096 (100.00%) |
| 20471 | full-context | 12,927 | 384 | 49,152 | 16,384/16,384 (100.00%) | 4,096/4,096 (100.00%) |
| 20471 | phrase-local | 12,927 | 384 | 49,152 | 16,384/16,384 (100.00%) | 4,096/4,096 (100.00%) |
| 20571 | full-context | 12,927 | 384 | 49,152 | 16,384/16,384 (100.00%) | 4,096/4,096 (100.00%) |
| 20571 | phrase-local | 12,927 | 384 | 49,152 | 16,384/16,384 (100.00%) | 4,096/4,096 (100.00%) |

Identical paired initializations, optimizer schedules and raw example exposure. The architectures need not have identical physical compute because their batching geometry differs. There is no matched-quality serving benchmark. Native FIT and blind PREDICT receipts, Forge projections, root-paired analysis and independent recount are preserved with the private replay.

## Decision

The proposed phrase-isolation repair did not earn promotion. Inspect TRAIN competence and the per-orientation, selected/sibling and route-ceiling results before deciding whether the next intervention should target optimization, scope representation, or routing. No broad binding or attention-replacement claim.

The isolated phrase-local reader passed in all three seeds: 8,064, 8,192
and 8,192 exact pointers out of 8,192, versus 7,680, 8,192 and 7,764 for
whole-continuation encoding. This is 24,448/24,576 versus 23,636/24,576
across repeated seed–row observations, not 24,576 independent trials. Its
crossed-quartet gains were +10.16, +0.00 and +5.86 percentage points.
The zero gain is a perfect-control ceiling; it still fails the
preregistered every-seed +5 pp rule. Do not alter that rule after scoring.

Two phrase-local runs passed natively. In the remaining run, the learned
phrase selector and noun pointer were perfect with the correct event, but
the frozen temporal selector chose the right event in only 7,488/8,192
cases (91.41%). All 704 native wrong answers in that seed are accounted
for by wrong or tied temporal selection on this panel. This moves the
bottleneck back to event selection for that seed, not to larger memory or
another copy intervention. Keep the qualified isolated reader as a
control and retest learned temporal selection against a matched control
on fresh cases before removing the supplied boundaries and split views.

The phrase-local head's logits were bit-identical in all 276,480
unchanged-phrase comparisons. The full-context head's logits changed in
110,592 of those comparisons. These are checks of a particular tensor
path, not independent semantic outcomes or evidence that all full-context
logit changes caused wrong answers.
