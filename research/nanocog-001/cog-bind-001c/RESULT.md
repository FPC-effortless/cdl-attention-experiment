# COG-BIND-001C: rewording the evidence is the larger failure
## Decision
On the same fixed synthetic facts, original wording was much easier for the six
frozen BIND-001B models. Rewriting only the fact statements reduced strict joint
source-and-value success from 1,341/1,536 to 291/1,536 model–family evaluations
(87.3%→18.9%). Rewriting only the question reduced it to 909/1,536 (59.2%).
Both shifts hurt; the larger loss for fact rewrites also occurs at known source.

A family passes only if all 16 changed-target views select the unique correct
record and emit the unique correct value from the eight-color vocabulary.
The correct-source column instead forces every gold record and checks its
value across the same 16 views. Anchor metrics are stored separately.
Familiar and shifted are paired views of the same 256 previously exposed
groups; models/seeds are repeated measurements, not independent samples.

| Fact wording | Question wording | Strict joint source + value | Strict value with correct source |
| --- | --- | ---: | ---: |
| Familiar | Familiar | 1,341/1,536 | 1,528/1,536 |
| Shifted | Familiar | 291/1,536 | 526/1,536 |
| Familiar | Shifted | 909/1,536 | 1,528/1,536 |
| Shifted | Shifted | 265/1,536 | 526/1,536 |

## Localization
- Rewriting the facts alone preserved every record identity, value, position,
  query, filler, and original template index. Even given the correct record,
  strict value accuracy fell from 99.5% to 34.2%. Row-level gold-record value
  accuracy fell from 24,560/24,576 to 20,466/24,576, so most rows still decode;
  correlated or repeated mistakes can break a complete group.
- Rewriting the question alone changed retrieval. Strict support fell from
  1,347 to 911 of 1,536. It changed no per-record logits, as expected from the
  clause-local encoder and reader with no direct question-to-value path.
- The two-channel scorer does not rescue the main shifted-fact failure.
  Strict joint counts pooled vs. two-channel out of 768 per arm were:
  familiar/familiar 641 vs. 700; shifted/familiar 164 vs. 127;
  familiar/shifted 432 vs. 477; shifted/shifted 133 vs. 132.
  Do not add the two isolated drops: a group can fail after either intervention.
- A part of the problem survives perfect record selection. This isolates the
  **record-encoding/decoding path**, not the encoder versus the decoder inside
  it. Neither the effect of word order alone nor missing semantic information
  in the hidden state is identified by these interventions.

## Verification and scope
Six exact frozen 001B checkpoints, four actor-only views of every original
group, 122,880 predictions; no fit, model selection, promotion, or external test.
Independent parser audit verified all 20,480 rewritten actor rows. Original
001B logits and ranks matched exactly on the 30,720 corresponding prediction
rows (92,160 field checks); all six per-slot logits arrays were bit-exact across
61,440 query-only row pairs. Scores are recomputed from saved raw rankings and
logits. This is a synthetic diagnostic with explicit sentence segmentation,
not an authentic text, free-form language or end-to-end cost result.

## Next
Train a small, matched probe on separate generated TRAIN groups using frozen
record representations to see whether the color remains recoverable after
fact rewording. Compare against the frozen decoder and an equal-budget
readout repair on a newly frozen evaluation panel with independently generated
wording. If the probe recovers the value, improve the reader; if it does not,
treat that as evidence about the probe class rather than proof of absent
information, and test the record encoder directly. Do not advance two-hop,
plasticity or system benchmarks on this result.
