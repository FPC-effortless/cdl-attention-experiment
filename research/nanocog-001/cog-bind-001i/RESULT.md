# COG-BIND-001I: learned token matching improves retrieval; binding still falls short
## Decision
A small learned token-matching term improved strict binding on all 12
matched seed-by-wording comparisons. On the hardest combined rewrite,
strict full-family success rose from 159/768 to 349/768 (20.7%→45.4%).
Neither model passed any registered CAL or HELD cell. This is a scoped
selector improvement, not general binding or attention replacement.

## Matched mechanism and procedure
Both arms share the exact pooled model and initial shared tensors under
three paired seeds, same E-v2 words and labels, same support and soft answer
cross-entropies plus 0.75 exact-slot value CE in TRAIN only. 1,024 updates,
81,920 actor rows per fit, identical words in each matched pair. The roles
arm adds 128 weights (29,480 vs 29,352, +0.44%) to learn two token-weighted
summaries for each record and for the question. It adds half a balanced
pair-cosine score to the existing pooled rank. The learned channels receive
no entity/relation labels or role masks and need not correspond to those
two roles. The same value head decodes six per-record outputs, among eight
colors, with the native choice taken from the unique top-ranked slot.
No tuning from CAL/HELD or broadened compute.

256 fresh cross-block rectangles matched across FF (familiar), SF
(fact rewrite), FS (query rewrite) and SS (both). All exact prompts and
whole rectangles are new; 72 individual held keys were used in ancestral
studies, though not in this study's TRAIN split. Native strict means all
16 target rows get their unique correct slot and value. No new language
domains or authentic text are tested. Counts out of 768 are 256 semantic
groups measured three times, not independent cases.

| View | Native strict joint pooled / roles | Strict selection pooled / roles | Correct-slot strict value pooled / roles |
| --- | ---: | ---: | ---: |
| FF | 287 / 507 | 287 / 507 | 768 / 768 |
| SF | 188 / 387 | 192 / 400 | 752 / 718 |
| FS | 231 / 471 | 231 / 471 | 768 / 768 |
| SS | 159 / 349 | 163 / 364 | 752 / 718 |

On SS the seed pairs were 76→108, 40→76, and 43→165 out of 256.
Rescues versus regressions: FF 311/91, SF 287/88, FS 327/87, SS 263/73.
The gains are retrieval-side, not improved value reading: on held fact
rewrites with four words after the value, exact-slot target-row decoding
fell from 6,112/6,144 to 6,030/6,144. With terminal values on those same
views both arms decoded all 6,144 target rows from the correct slot.

## Query-swap stress and shortcut
Within one fixed history the query changes exactly one entity or one
relation token; both queries must select their different correct records.
SS entity-only pair selection rose 3,170→4,460/6,144 (51.6%→72.6%);
relation-only rose 3,112→4,546/6,144 (50.7%→74.0%). Each numerator is a
model–pair observation over paired repeats. Both keys and unchanged anchor
coverage are tracked separately.
A fixed TRAIN-IDF word-overlap diagnostic, with no role labels, is stronger
on the fact-rewritten views: on SS 1,744/2,048 entity and 1,748/2,048
relation swaps; on SF perfect 2,048/2,048 each. It predicts evidence only,
not the answer. This shortcut makes the test useful for isolating learned
retrieval but insufficient for claiming semantic generalization.

## Isolating the new term after training
For fixed learned backbone weights the added score per slot is bounded by
±0.5, so an old top-rank margin greater than 1 cannot be overturned in exact
arithmetic. We reran the role checkpoints with only this term removed,
with no fit and same inputs. Per-slot value logits were bit-identical.
On SS, strict completion fell only from 349 to 341 (8 families rescued by
the active term, 0 harmed). On FF 507→494, SF 387→374, FS 471→447.
Most gains survive removing it at inference. This distinguishes its
small immediate effect from the full change observed after training with
the added branch, but it does not uniquely identify a training mechanism.
No claim that each role head became a meaningful semantic specialist.

## Verification and next
Six matched fits, 30 sealed prediction packets (130,560 prompt rows),
independently rescored. Keep the exact-match comparator and this trained
selector as controls, and build the next test around aliases, role reversals
and realistic lexical decoys where word overlap can no longer find the answer.
Measure selection, value and native joint together. Do not promote or claim
a speed advantage. No language-model, latency, memory-system or downstream
qualification took place.
