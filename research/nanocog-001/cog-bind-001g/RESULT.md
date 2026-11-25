# COG-BIND-001G: directional alignment did not fix value transfer
## Decision
The alignment loss made same-fact representations point in more similar
directions on both unseen phrasings. That did not make the decoder more
reliable. With the correct record supplied, the alignment arm completed
fewer families than stronger direct value supervision on both new forms:
576→532/768 and 666→590/768. Native performance improved on one new form
and regressed on the other. No fit qualified; do not promote cosine
alignment as a fix for binding.

## What was matched
Six scratch fits, three paired seeds (9371, 9471, 9571), one exact pooled COG
architecture with 29,352 parameters per fit and identical initialization
within pair. CPU, fixed 1,024 updates, 81,920 rows/fit (40,960 matched wording
pairs); identical targets, records, exposure and per-seed nonpadding token
counts (5,908,792; 5,907,676; 5,908,652). No outcome-based fit choice.
Fixed objectives:
- Control: native support CE + native soft-answer CE + 0.75 correct-slot CE.
- Alignment: same native objectives + 0.50 correct-slot CE + 0.25 symmetric
  cosine distance of the 48-D value vector for the same queried fact.
Both arms saw both views of each selected training example. Correct slots are
used outside the forward graph only to build supervised TRAIN losses; no
evaluation model input includes a correct slot, key or label. The shared
base and equal nominal bonus weight do not match gradients or loss scale.

## Fresh held synthetic results
256 fresh held semantic rectangles rendered in three ways, with fixed facts,
questions, answers, record order and filler. SF and TF replace only fact
wording with two previously unused template families. CAL, HELD and TRAIN
keys are disjoint for this study, but every individual HELD key had appeared
in an ancestral study; whole rectangles and exact prompts are fresh.
All 77 CAL/HELD word types appear in each scheduled wording stream for each
seed. Strict family success entails all 16 target views with unique correct
record and value; anchors scored separately. Each cell counts 768 repeated
model–families (256 groups measured in three seeds), not 768 independent
cases. Left/right are direct-supervision control / cosine alignment.

| Wording | Native joint | Correct record selection | Value given correct record |
| --- | ---: | ---: | ---: |
| FF familiar | 503 / 586 | 503 / 586 | 768 / 768 |
| SF unseen facts | 169 / 235 | 227 / 317 | 576 / 532 |
| TF unseen facts | 214 / 195 | 259 / 272 | 666 / 590 |

On SF the native gain was 66 families with selection up 90 and value given
the correct record down 44, not a repair of the value decoder. Native
improved in two seeds on SF (52→44, 51→55, 66→136) and one seed on TF
(77→61, 86→64, 51→70). Correct-record value dropped on each new family
in all three seeds. All six fits failed full CAL and HELD criteria.
Native paired strict rescues/losses were 184/101 on familiar, 160/94 on SF,
and 123/142 on TF. Source/value bins are intersections of strict behaviors
using the same hard slot logits, not causal estimates.

## Geometry and what it can guarantee
Across all six seed × rewrite comparisons, alignment reduced mean
same-value cosine distance, but also reduced mean different-value cosine
distance. It increased mean vector norms in every comparison and did not
consistently reduce raw Euclidean same-fact displacement (it fell only in
two of six comparisons). Measured on the sealed value vectors of same-key
same-value pairs across wordings and same-key changed-value worlds within
each wording, with duplicate presentation positions not counted twice.
Cosine agreement alone does not bound raw movement or decoder output.

For the actual two-layer exact-GELU decoder, a sufficient condition for the
same unique color on a known slot is raw displacement below m/(2L), where
m is the original positive logit margin and L can be upper-bounded by
(1 + 1/sqrt(2*pi*e)) times the two spectral matrix norms multiplied.
This follows from the derivative of GELU and bounding the two competing
logit changes. This is a conditional certificate, not a training guarantee,
and says nothing about selecting the correct slot or unseen semantic input.

## Next decision
Do not scale this cosine objective. If continuing this mechanism, supervise
the value output on paraphrases while explicitly testing separation for
different values in the same record; compare against the simpler direct CE
control. The held changes should target semantic transfer and counteract
norm shortcuts. Run real-text and matched inference evaluations only after
quality clears the agreed gates. No binding fix, language, speed or downstream
qualification is claimed here.

Verification: six fits and 24 sealed prediction files (99,840 prompt rows),
with the 18 held-only float32 vector sidecars retained. Strict scores were
reconstructed from full logits with exact tie rejection, not trusted flags.
