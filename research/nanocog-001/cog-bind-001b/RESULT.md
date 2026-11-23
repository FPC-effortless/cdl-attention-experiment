# COG-BIND-001B: factorized routing helped familiar wording, not shifted wording
## Decision
No fit qualified. The learned two-channel scoring penalty improved strict
familiar-syntax performance in this constructed holdout (308→348 of 384
model–family evaluations, 80.2%→90.6%) but not shifted syntax (77→74 of 384,
20.1%→19.3%). Do not promote the architecture, infer that channels discovered
semantic roles, scale context, or advance downstream plasticity/system gates.

## Matched, frozen experiment
Same clause-local encoder and decoder, 28,872 nominal trainable scalars per
arm, identical initialization within seeds 6371/6471/6571 and identical
sample schedules. Both models receive training-only support labels plus answer
loss; use no name/role annotations, exact lookup or target record at inference.
The intervention subtracts 0.35 times the disagreement between two coordinate
block scores from the pooled dot product. Both receive six explicitly
segmented clauses and one question and classify only eight known colors.
1,024 optimizer steps and 81,920 exposed rows per fit. Data are generated,
containing disjoint entity–relation keys across TRAIN/CAL/HELD, with known
atomic vocabulary. Four held cells contain independent groups of 64 each;
the shifted groups are not literal re-renders of familiar held groups.
The strict unit requires unique correct record and value for all 16 target
rows per group across worlds and presentations. Anchors are separate.
All six passed the TRAIN probe; only one per arm passed both CAL cells.
No seed in either arm passed the four registered held cells.

| Syntax | Pooled dot | Two-channel penalty | Correct-record supplied: pooled / two-channel |
| --- | ---: | ---: | ---: |
| Familiar | 308/384 | 348/384 | 382/384 / 382/384 |
| Shifted | 77/384 | 74/384 | 129/384 / 162/384 |

| Seed | Familiar pooled / two-channel | Shifted pooled / two-channel |
| --- | ---: | ---: |
| 6371 | 100/128 / 122/128 | 40/128 / 20/128 |
| 6471 | 107/128 / 112/128 | 15/128 / 22/128 |
| 6571 | 101/128 / 114/128 | 22/128 / 32/128 |

The same groups recur across seeds, so these pooled counts describe model–
family evaluations, not independent sample sizes. Familiar paired family
rescues/losses: 61/21. Shifted paired family rescues/losses: 48/51.
The small aggregate shift conceals a reversal across seeds. The design
does not isolate syntax from other between-cohort composition changes.

## Localization, not a learned semantic-state pass
On shifted wording, the correct-record intervention lifted strict success
from 151/768 to 291/768 across both arms; 477 still incomplete. Native shifted
readout chose the correct record but gave a wrong value on 1,582/12,288 target
rows; conversely, it emitted the right value after selecting the wrong record
on 116/12,288. Masking the filler at the scoring layer changed no held cell's
joint counts or strict totals. The complete-record symbolic reference scored
all 1,536 held model–family evaluations; this checks the fixture and is not
learned. We did not inject a full canonical record bank into a frozen shared
learned interface, so this is not the full semantic-binding oracle ladder.
The forced-record readout is diagnostic, not an admitted inference result.

An independent wire recount re-evaluated all 18 TRAIN-probe, CAL and held
summaries from raw logits and truth; all primary tested totals matched.
The full replay records fit and runtime receipts, not evidence of competitive
end-to-end inference, real-text generalization or cross-machine replication.

## Next
Freeze a same-history syntax rewrite panel and separately measure record
selection and value decoding from the correct record. Compare representation
and decoder repair under matched supervision, with an exact reference and
an intervention that swaps only the value while retaining the record key.
That will determine whether the route or the encoding/readout of the selected
record needs the next architecture change. Keep longer chains, persistent edits
and system efficiency deferred.
