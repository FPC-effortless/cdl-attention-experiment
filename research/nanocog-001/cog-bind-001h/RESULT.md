# COG-BIND-001H: correct-slot decoding is perfect here; selection is not fixed
## Decision
On this fresh synthetic panel, both training objectives read the right
value in every strict case when supplied the correct record. That is a
bounded win for the value-readout path, not evidence the new loss caused
it. End-to-end success remained far below the agreed gate because neither
model reliably found that record. Do not claim binding fixed or scale
this objective on these results.

## Matched experiment
Three paired scratch initializations and the exact pooled 29,352-parameter
COG model; identical E-v2 text, ordering, labels and optimizer. 1,024
updates ×80 rows, identical paired 81,920 rows per fit, identical nonpadding
tokens within seed (5,909,872; 5,909,192; 5,908,384). Both arms use native
support CE and soft-answer CE and direct exact-slot CE of weight 0.50.
Control adds 0.25 more exact-slot CE. Treatment substitutes 0.125 JS
between matching same-fact wordings and 0.125 margin-1 symmetric
changed-value hinge, applied to known-slot TRAIN logits only.
Both receive identical paired exposures: 40,960 same-fact row pairs and
16,384 changed-value pairs per wording. The hinge excludes anchors and
is averaged over both directions and wordings. Labels never enter the
model inputs; nominal loss weights are not equal effective gradients.

No CAL/HELD tuning. Fresh 256 held rectangles, paired edits of facts only,
two previously unused facts template families with familiar queries.
All held values occupy the last word in their rendered facts. New
rectangles and prompts, but all 72 individual keys appeared in earlier
roles. This is controlled synthetic syntax; the comparison is within H,
not a causal numerical improvement over BIND-001G with different templates.

## Held results
Strict family: every one of 16 changed-target rows needs the unique known
record and unique correct one of eight values. Correct-record scores use the
same candidate logits with the gold index supplied only in evaluation.
Counts below are control / output-level treatment; 768 model–families per
wording (256 groups repeated across three seeds, not independent samples).

| Wording | Strict native joint | Strict record selection | Strict value given correct record |
| --- | ---: | ---: | ---: |
| Familiar | 499 / 526 | 499 / 526 | 768 / 768 |
| Unseen fact family S | 314 / 318 | 314 / 318 | 768 / 768 |
| Unseen fact family T | 288 / 316 | 288 / 316 | 768 / 768 |

All strict misses are selection-only under this output contract.
There were no passed full CAL or HELD cells in any fit.
Treatment's native strict gain is not seed-stable on novel facts:
S (145→130, 86→108, 83→80) and T (142→128, 48→80, 98→108).
Familiar (199→200, 137→157, 163→169). Paired rescues/losses:
familiar 147/120, S 138/134, T 150/122. On S the native target row
count decreased (10,328→10,270 of 12,288) despite the small family gain.

The opposing-world pair scores confirm exact value reversal when supplied
correct records: all 6,144/6,144 paired observations of each held style
in both arms. With native selection, both opposing worlds were correct
in 5,442/5,512 (familiar), 4,788/4,768 (S), and 4,696/4,762 (T), each
out of 6,144. These observations include two presentations per query.
Anchors are reported separately in the replay.

## Why this is not a new guarantee
The logged margin term was active on 99.4–100% of direction-specific
comparisons in the first 32 TRAIN updates, and on exactly zero in the
last 32 in both arms. The direct-supervision control also reached
zero TRAIN hinge with no explicit hinge penalty. The bundle compares
the whole objective, not individual JS or hinge effects.

For a known slot with original positive posterior margin m, JS in nats
below m²/8 is sufficient to preserve its unique top answer, by the total
variation bound. This condition is not established for arbitrary wordings
and says nothing about finding the record. The margin hinge compares
two candidate values, not every other class.

## Next decision
Stop tuning the value-side loss on these templates. Move the next study
to role-aware record selection and a key-swap test in which only the query
or intended record changes. Keep a direct-supervision control and test
both same-slot value preservation and full native answers on untouched
more demanding wording and realistic text. The current held forms are
now exposed. No inference-timing, LLM, general binding, plasticity or
system qualification.

Verification: all six fits and 24 sealed prediction packets (99,840 rows).
Independent score uses actual rank and per-slot logits with exact tie
rejection, not opaque evidence flags.
